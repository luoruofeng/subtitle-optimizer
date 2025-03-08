import asyncio
import json
import os
import re
import traceback
from types import SimpleNamespace
from typing import List
import edge_tts
import numpy as np
from pysrt import SubRipFile, SubRipItem
import pysrt
from subtitle_optimizer.utils import detect_encoding,merge_short_subs,time_stretch_audio,call_llm_api, detect_language, format_merged_text
from subtitle_optimizer.exceptions import LanguageMismatchError
import whisper
from pydub import AudioSegment
import torch
from typing import Union, List, Tuple
import time
from functools import wraps
from moviepy import VideoFileClip,AudioFileClip

def retry_on_permission_error(max_retries=3, delay=0.5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except PermissionError as e:
                    retries += 1
                    if retries >= max_retries:
                        raise
                    print(f"Retry {retries}/{max_retries} due to {e}")
                    time.sleep(delay)
        return wrapper
    return decorator

# 在删除和重命名操作处添加装饰器
@retry_on_permission_error()
def safe_remove(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)

@retry_on_permission_error()
def safe_rename(src, dst):
    os.rename(src, dst)

class SubtitleOptimizer:
    def __init__(self, api_key=None):
        """
        初始化 SubtitleOptimizer 类的实例。

        参数:
            api_key (str, 可选): 用于调用 LLM API 的 API 密钥。默认为 None。
        """
        self.api_key = api_key
        # 初始化本地Whisper大模型
        # 注释掉的代码表示使用 "large-v3" 版本的 Whisper 模型，该模型通常具有更高的精度，但可能需要更多的计算资源和时间。
        # self.whisper_model = whisper.load_model("large-v3")
        # 使用 "medium.en" 版本的 Whisper 模型，该模型专为英语设计，在精度和性能之间取得了较好的平衡。
        self.whisper_model = whisper.load_model("medium.en")

    ###################视频变速######################
    def adjust_video_speed(self, video_path: str, speed_factor: float = 0.7) -> None:
        """
        调整视频播放速率并覆盖原文件
        
        参数:
            video_path (str): MP4视频文件路径
            speed_factor (float): 速率倍数（默认0.7倍降速）
        
        示例:
            obj = SubtitleOptimizer()
            obj.adjust_video_speed("test.mp4", 0.5)  # 降速至0.5倍
        """
        if speed_factor==1:
            print("视频速率已为正常速率，无需调整")
            return
        try:
            video_path = os.path.abspath(video_path)

            voice_wav = time_stretch_audio(video_path, speed_factor)
            audio = AudioFileClip(voice_wav)  # 新增代码

            if os.path.exists(video_path) is False:
                raise FileNotFoundError(f"未找到指定的视频文件：{video_path}")

            # 加载视频
            video = VideoFileClip(video_path)
            # 生成临时文件路径
            temp_path = video_path.replace(".mp4", "_temp.mp4")
            print(f"调整视频速率至 {speed_factor} 倍中... 保存到临时文件：{temp_path}")
            
            # 视频处理
            adjusted_video = video.with_speed_scaled(speed_factor).with_audio(audio)
            temp_path = video_path.replace(".mp4", "_temp.mp4")
            
            # 写入临时文件（新增remove_temp参数）
            adjusted_video.write_videofile(
                temp_path,
                codec='libx264',
                audio_codec='aac',
                threads=4,
                logger=None,
                remove_temp=True  # 强制清理临时文件
            )
            
            # 显式释放资源（关键修复点）
            for obj in [video, audio, adjusted_video]:
                if obj: obj.close()
            
            # 覆盖原文件（新增重试机制）
            retry_count = 0
            while retry_count < 3:
                try:
                    os.replace(temp_path, video_path)
                    break
                except PermissionError:
                    time.sleep(0.5)  # 等待500ms重试
                    retry_count +=1
            
            print(f"覆盖完成: {video_path}")

        except Exception as e:
            # 最终资源清理（新增）
            for obj in [video, audio, adjusted_video]:
                if obj and hasattr(obj, 'close'): obj.close()
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(voice_wav):
                os.remove(voice_wav)
            


    ####################将srt转换为语音######################

    def edge_tts_voice(self,args):
        async def _text_to_speech():
            # 获取参数内容
            content = args.content
            save_path = args.save_path
            language = args.language
            voice = args.voice
            rate = getattr(args, 'rate', '+0%')  # 添加语速参数，默认为正常语速

            # 检查内容是否为空
            if not content:
                raise ValueError("Content cannot be empty")

            # 设置默认语言为英文
            if language is None:
                language = "en"

            # 设置默认语音
            if voice is None:
                if language == "en":
                    voice = "en-US-MichelleNeural"  # 默认英文
                elif language == "en-child":
                    voice = "en-US-AnaNeural"  # 默认儿童英文
                elif language == "zh-cn":
                    voice = "zh-CN-XiaoxiaoNeural"  # 默认中文
                elif language == "zh-tw":
                    voice = "zh-TW-HsiaoChenNeural"  # 默认台湾中文
                else:
                    raise ValueError(f"Unsupported language: {language}")
                
            communicate = edge_tts.Communicate(
                content, 
                voice, 
                rate=rate
            )
            await communicate.save(save_path)
            print(f"Audio saved successfully at {save_path}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_text_to_speech())
        finally:
            loop.close()

    def generate_voice_from_srt(self, srt_path: str):
        """处理SRT文件生成双语语音并调整时长"""
        # 创建语音文件夹
        srt_path = os.path.abspath(srt_path)
        voice_dir = os.path.join(os.path.dirname(srt_path) , f"{os.path.basename(srt_path).split(".")[0]}-voice")
        if not os.path.exists(voice_dir):
            os.mkdir(voice_dir)
        else:
            raise FileExistsError(f"目标文件夹已存在：{voice_dir}")
        
        # 处理SRT文件
        subs = pysrt.open(srt_path, encoding='utf-8')
        cn_audio_files = []
        
        for idx, sub in enumerate(subs, start=1):
            # 分割字幕内容
            lines = sub.text.split('\n')
            
            # 合并中英文字幕
            merged = ['', '']
            current_lang = 0  # 0-中文 1-英文
            for line in lines:
                has_chinese = bool(re.search(r'[\u4e00-\u9fff]', line))
                target = 0 if has_chinese else 1
                merged[target] += ' ' + line.strip()
            
            print(f"\n处理字幕：{idx} | 中文：{merged[0]} | 英文：{merged[1]}")

            # 生成语音文件
            for i, text in enumerate(merged):
                if not text.strip():
                    continue
                
                prefix = 'cn' if i == 0 else 'en'
                filename = os.path.join(voice_dir , f"{prefix}-{idx:04d}.mp3") 
                if prefix == 'cn':
                    args = SimpleNamespace(
                        content=text,
                        save_path=str(filename),
                        language='zh-cn',
                        rate='+33%',
                        voice=None
                    )
                else:
                    args = SimpleNamespace(
                        content=text,
                        save_path=str(filename),
                        language='en',
                        rate='+0%',
                        voice=None
                    )
                
                # 调用语音生成（需处理异步）
                self.edge_tts_voice(args)


                if prefix == 'cn':
                    cn_audio_files.append((filename, sub))
        
        # 调整中文语音时长（修复变速失效问题）
        for filepath, sub in cn_audio_files:
            audio = AudioSegment.from_file(filepath)
            original_duration = len(audio) / 1000  # 转换为秒
            
            # 计算目标时长（毫秒转秒）
            target_duration = (sub.end.ordinal - sub.start.ordinal) / 1000.0
            
            # 有效速度范围（基于语音清晰度研究）
            min_speed = 0.7  # 最低语速（低于0.7x会导致严重失真）
            max_speed = 2.0  # 最高语速（高于2.0x会丢失语音特征）
            
            if abs(original_duration - target_duration) > 0.1:  # 调整容差阈值至0.1秒
                print(f"调整语音时长：{filepath} | 当前={original_duration:.2f}s | 目标={target_duration:.2f}s")
                
                # 修复1：使用正确的速度因子计算（当需要减速时取倒数）
                speed_factor = original_duration / target_duration
                
                # 修复2：速度限制逻辑重构
                if speed_factor > 1:  # 加速模式
                    speed_factor = min(max_speed, speed_factor)
                    processing_method = "加速"
                else:  # 减速模式
                    speed_factor = max(min_speed, 1/speed_factor)
                    processing_method = "减速"
                
                print(f"速度系数：{speed_factor:.2f}x ({processing_method})")
                
                # 修复3：使用speedup方法并保持音高
                adjusted_audio = audio.speedup(
                    playback_speed=speed_factor,
                    chunk_size=150,  # 减小块大小提升精度
                    crossfade=25      # 添加交叉淡化
                )
                
                # 新增：动态滤波处理（根据速度调整截止频率）
                if speed_factor > 1.5:
                    adjusted_audio = adjusted_audio.low_pass_filter(6000)  # 抑制高频噪声
                elif speed_factor < 0.7:
                    adjusted_audio = adjusted_audio.high_pass_filter(200)  # 增强低频
                    
                # 新增：振幅归一化防止削波
                adjusted_audio = adjusted_audio.normalize()
                
                # 修复4：使用临时文件避免覆盖问题
                temp_path = f"{filepath}.tmp"
                adjusted_audio.export(temp_path, format="mp3", bitrate="192k")  # 固定比特率
                
                # 原子替换文件（Windows需处理文件占用问题）
                if os.path.exists(filepath):
                    safe_remove(filepath)
                safe_rename(temp_path, filepath)
            else:
                print(f"跳过调整（差值{abs(original_duration - target_duration):.2f}s < 0.1s）：{filepath}")


    ####################通过txt生成SRT文件######################
    @staticmethod
    def _split_sentences(content):
        pattern = r'([，。！？；：、…\u2026.!?;:])[\s]*'
        result = re.split(pattern, content)
        
        sentences = []
        # 初步合并句子与标点
        for i in range(0, len(result), 2):
            combined = (result[i] + (result[i+1] if i+1 < len(result) else '')).strip()
            if combined:
                sentences.append(combined)
        
        # 处理单独标点合并到前一句
        merged = []
        for sentence in sentences:
            # 判断当前句子是否仅由标点构成
            if re.fullmatch(r'^[，。！？；：、….!?;:]+$', sentence):
                # 若前一句存在且非标点，则合并
                if merged and not re.fullmatch(r'^[，。！？；：、….!?;:]+$', merged[-1]):
                    merged[-1] += sentence
                else:
                    merged.append(sentence)
            else:
                merged.append(sentence)
        return merged


    def _generate_srt_from_segments(self, txt_path: str, mp4_path: str, segments_path: str = None):
        """处理单个文件对，支持可选的分段时间戳文件"""
        srt_path = os.path.splitext(mp4_path)[0] + ".srt"
        if os.path.exists(srt_path):
            print(f"SRT文件已存在：{srt_path} 跳过处理\n")
            return
        # 时间戳来源判断
        if os.path.exists(segments_path):
            # 从文件加载预生成的时间戳
            try:
                with open(segments_path, 'r', encoding='utf-8') as f:
                    segments = json.load(f)  # 加载JSON数据
            except json.JSONDecodeError:
                print("错误：文件内容不是有效的JSON格式")
                traceback.print_exc()
            except PermissionError:
                print("错误：无权限读取文件")
                traceback.print_exc()
            except Exception as e:
                print(f"未知错误：{e}")
                traceback.print_exc()
        else:
            print(f"⏳ 正在提取 {mp4_path} 中的时间戳... segments_path 不存在:{segments_path}")
            # 通过Whisper生成时间戳
            result = self.whisper_model.transcribe(
                mp4_path,
                word_timestamps=True,
                language="en",
                task='transcribe',
                fp16=torch.cuda.is_available()
            )
            segments = result["segments"]
        print(f"✅ 提取到 {len(segments)} 个时间戳段")
        
        # 读取文本内容（逻辑不变）
        content = ""
        if txt_path is not None:
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read()
        elif segments is not None:
            for segment in segments:
                content+= segment.get("text", "")
        sentences = SubtitleOptimizer._split_sentences(content) # 按照标点符号划分句子
        print(f"✅ 提取到 {len(sentences)} 个句子")
        print(f"句子：{sentences}\n")
        
        # 生成SRT内容
        all_words = [word for segment in segments for word in segment.get("words", [])]
        srt_content = []
        current_word_idx = 0

        for sentence in sentences:
            # 新增标点规范化处理（解决中英文符号差异）
            normalized_sentence = re.sub(r'[，,]', ',', sentence)  # 统一中文逗号为英文
            sentence_words = normalized_sentence.split()
            
            matched = False
            max_retry = 13  # 最大合并尝试次数
            
            # 扩展匹配范围，考虑单词合并可能性
            for i in range(current_word_idx, min(len(all_words), current_word_idx + 200)):
                # 尝试合并1到13个单词的组合
                for merge_count in range(1, max_retry + 1):
                    if i + merge_count > len(all_words):
                        break
                    
                    # 生成合并后的单词及时间戳
                    merged_word = ''.join([all_words[idx]['word'].strip() for idx in range(i, i + merge_count)])
                    merged_start = all_words[i]['start']
                    merged_end = all_words[i + merge_count - 1]['end']
                    
                    # 规范化比较（忽略大小写和空格）
                    if merged_word.lower() == sentence_words[0].lower():
                        # 验证后续单词是否连续匹配
                        full_match = True
                        word_ptr = i + merge_count
                        
                        for word_idx in range(1, len(sentence_words)):
                            # 继续尝试合并后续单词
                            found = False
                            for cnt in range(1, max_retry + 1):
                                if word_ptr + cnt > len(all_words):
                                    break
                                
                                next_merged = ''.join([all_words[idx]['word'].strip() 
                                                    for idx in range(word_ptr, word_ptr + cnt)])
                                
                                if next_merged.lower() == sentence_words[word_idx].lower():
                                    word_ptr += cnt
                                    merged_end = all_words[word_ptr - 1]['end']
                                    found = True
                                    break
                            
                            if not found:
                                full_match = False
                                break
                        
                        if full_match:
                            srt_content.append(
                                f"{len(srt_content)+1}\n"
                                f"{self._format_timestamp(merged_start)} --> {self._format_timestamp(merged_end)}\n"
                                f"{sentence}\n"
                            )
                            current_word_idx = word_ptr
                            matched = True
                            break
                    
                    if matched:
                        break
                if matched:
                    break

            if not matched:
                # 增强错误信息可读性
                expected = '|'.join(sentence_words)
                actual = '|'.join([w['word'] for w in all_words[current_word_idx:current_word_idx+20]])
                raise ValueError(
                    f"无法匹配句子：'{sentence}'\n"
                    f"预期单词序列：{expected}\n"
                    f"实际后续单词：{actual}"
                )


        # 保存SRT文件
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_content))
            print(f"✅ SRT文件已保存至：{srt_path}")
        
        merge_short_subs(srt_path)  # 合并短句
        print(f"合并后的srt文件：")
        for line in open(srt_path, 'r', encoding='utf-8'):
            print(line)

    def generate_srt_from_directory(self, folder_path: str):
        """处理文件夹中的文件对"""
        mp4_files = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]
        for mp4_file in mp4_files:
            try:
                base_name = os.path.splitext(mp4_file)[0]
                # 生成三种可能的相关文件路径
                txt_path = os.path.join(folder_path, f"{base_name}.txt")
                segments_path = os.path.join(folder_path, f"{base_name}_segments.txt")
                mp4_path = os.path.join(folder_path, mp4_file)
                
                # 仅当txt文件存在时才处理
                if not os.path.exists(txt_path):
                    txt_path = None
                    
                self._generate_srt_from_segments(txt_path, mp4_path, segments_path)
                print(f"✅ 处理_generate_srt_from_txt_process_directory完成：{mp4_file}")
            except Exception as e:
                print(f"❌ 处理_generate_srt_from_txt_process_directory失败：{str(e)}")
                traceback.print_exc()
            print("\n")
            

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """将秒数转换为SRT时间格式"""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02}:{int(minutes):02}:{seconds:06.3f}".replace(".", ",")


    ####################提取视频中的文本######################
    def extract_segments_from_mp4(self, mp4_path: str) -> str:
        try:
            # 判断MP4文件是否存在
            if not os.path.exists(mp4_path):
                raise FileNotFoundError(f"未找到指定的MP4文件：{mp4_path}")

            # 转录视频内容（自动提取音频）TODO
            result = self.whisper_model.transcribe(
                mp4_path, 
                language="en", 
                word_timestamps=True,
                task='transcribe',
                fp16=torch.cuda.is_available()
            )
            extracted_text = result.get("text", "")
            segments = result.get("segments", [])  # 新增：获取分段信息
            print(f"\nsegment:{segments}\n")
            # 构建主文本输出路径
            base_path = os.path.splitext(mp4_path)[0]
            txt_path = f"{base_path}.txt"
            segments_path = f"{base_path}_segments.txt"  # 新增：分段文件名

            # 删除已存在的同名文件
            for path in [txt_path, segments_path]:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"已删除现有文件：{path}")

            # 写入主文本文件
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
                print(f"✅ 主文本已保存至：{txt_path}")
                print(f"text:{extracted_text}\n")

            # 新增：写入分段文本文件
            if segments:
                print(f"检测到分段信息，共 {len(segments)} 个片段")
                with open(segments_path, 'w', encoding='utf-8') as seg_file:
                    # 使用json.dump()将列表序列化为JSON格式写入文件
                    json.dump(segments, seg_file, ensure_ascii=False, indent=4)
                print(f"✅ 分段数据已JSON序列化保存至：{segments_path}")

            return extracted_text
            
        except Exception as e:
            print(f"❌ 转录失败：{str(e)}")
            raise RuntimeError(f"视频转录失败: {str(e)}") from e

    ####################将字幕中的短句合并为长句######################

    def merge_srt(self, input_srt_file: str, output_srt_file: str = None) -> None:
        """处理SRT文件合并并保存"""
        # 读取输入文件
        subs = pysrt.open(input_srt_file,"utf-8")
        
        # 调用原有合并逻辑
        merged_subs = self._merge_srt(subs)
        
        # 确定输出路径
        save_path = output_srt_file if output_srt_file else input_srt_file
        

        # 写入处理结果
        merged_subs.save(save_path, encoding='utf-8')
        
        print(f"处理完成，保存路径：{save_path}")


    def _merge_srt(self, subs: SubRipFile,max_merge_lines=999) -> SubRipFile:
        merged_subs = []
        i = 0
        
        while i < len(subs):
            current_group = [subs[i]]
            merged_text = subs[i].text
            next_candidate = i + 1
            
            print(f"\n开始处理字幕组，起始索引: {i}")
            print(f"初始合并组: {[sub.text for sub in current_group]}")
            print(f"初始合并文本: '{merged_text}'")

            while True:
                if next_candidate >= len(subs) or len(current_group) >= max_merge_lines:
                    print(f"终止合并 - 剩余字幕: {len(subs)-next_candidate}, 当前合并行数: {len(current_group)}")
                    break
                
                next_sub = subs[next_candidate]
                next_next_sub = subs[next_candidate + 1] if next_candidate + 1 < len(subs) else None
                
                try:
                    args = [merged_text, next_sub.text]
                    if next_next_sub:
                        args.append(next_next_sub.text)
                    
                    # 打印合并判断参数
                    print(f"\n尝试合并 {len(current_group)} 行到 {len(current_group)+1} 行")
                    print(f"当前合并文本: '{args[0]}'")
                    print(f"下一行文本:  '{args[1]}'")
                    print(f"第三行存在:   {args[2] if len(args)>2 else '无'}")

                    if self._should_merge(*args):
                        current_group.append(next_sub)
                        merged_text = self._format_text(current_group)
                        next_candidate += 1
                        print(f"✅ 合并成功！当前合并行数: {len(current_group)}")
                        print(f"更新后合并文本: '{merged_text}'")
                    else:
                        print(f"❌ 合并终止 {len(current_group)} 行")
                        break
                except LanguageMismatchError as e:
                    print(f"🚫 语言不匹配异常: {str(e)}")
                    break

            new_sub = self._create_merged_sub(current_group, merged_text)
            merged_subs.append(new_sub)
            print(f"\n生成合并条目: {new_sub.start} -> {new_sub.end}")
            print(f"最终合并文本: '{new_sub.text}'\n")
            
            i = next_candidate
        
        print(f"\n合并完成，总条目数: {len(merged_subs)}")
        return SubRipFile(items=merged_subs)
        
    def _format_text(self, group):
        return format_merged_text([sub.text for sub in group])
    
    def _create_merged_sub(self, group, text):
        return SubRipItem(
            start=group[0].start,
            end=group[-1].end,
            text=text
        )
    
    def _should_merge(self, text1: str, text2: str, text3: str=None) -> bool:
        print(f"\n执行合并判断 _should_merge:")
        print(f"text1 ({len(text1)} chars): '{text1}'")
        print(f"text2 ({len(text2)} chars): '{text2}'")
        print(f"text3 ({len(text3) if text3 else 'None'}): '{text3 if text3 else '无'}'")

        if detect_language(text1) != detect_language(text2):
            print("语言不匹配触发异常")
            raise LanguageMismatchError("Language mismatch")

        # 打印语言检测结果
        lang1 = detect_language(text1)
        lang2 = detect_language(text2)
        lang3 = detect_language(text3) if text3 else None
        print(f"语言检测结果: text1={lang1}, text2={lang2}, text3={lang3}")

        if lang1 != lang2 or lang2 != lang3:
            print(f"语言不匹配: {lang1} != {lang2} != {lang3} text1={text1} text2={text2} text3={text3}")
            return False

        prompt = ""
        if text3 is not None and lang1 == lang2 and lang2 == lang3:
            prompt = f"在一个没有标点符号的字幕文件中，第三行字幕是“{text3}”的情况下，第一行字幕和第二行是否可能是属于同一句话?可能属于的回答我“Y”，可能不属于回答我“N”，不要回答任何信息。\n第一行: {text1}\n第二行: {text2}"
        else:
            prompt = f"在一个没有标点符号的字幕文件中，在没有第三行字幕的情况下，第一行字幕和第二行是否可能是属于同一句话?可能属于的回答我“Y”，可能不属于回答我“N”，不要回答任何信息。\n第一行: {text1}\n第二行: {text2}"
        
        print(f"发送给LLM的提示:\n{prompt}")
        response = call_llm_api(prompt, api_key=self.api_key)
        print(f"LLM响应原始结果: '{response}'")
        
        result = 'Y' in response
        print(f"合并判断最终结果: {'Y' if result else 'N'}")
        return result


    ####################翻译######################
    def add_translation(self, input_srt_file: str, output_srt_file: str = None, 
                   target_lang: str = "zh") -> None:
        """添加双语字幕并保存SRT文件
        :param target_lang: 目标语言代码（默认中文）
        """
        input_srt_file = os.path.abspath(input_srt_file)
        # 判读后缀是否正确
        if not input_srt_file.endswith('.srt'):
            raise ValueError("输入文件必须为SRT格式")
        subs = pysrt.open(input_srt_file, encoding=detect_encoding(input_srt_file))
        processed_subs = self._translate_subs(subs, target_lang)
        save_path = output_srt_file if output_srt_file else input_srt_file
        processed_subs.save(save_path, encoding='utf-8')
        print(f"双语字幕已生成，保存路径：{save_path}")

    def _translate_subs(self, subs: SubRipFile, target_lang: str) -> SubRipFile:
        for sub in subs:
            src_text = sub.text.strip()
            
            # 语言检测与过滤（示例仅处理英文翻译）
            if detect_language(src_text) not in ['en']:
                continue  # 跳过非英文内容
            
            # 构造翻译指令（支持格式控制）
            prompt = f"""将以下字幕精确翻译为{target_lang}，保持专业术语，禁止添加其他内容，专业术语可以直接保持英文原文，如果是疑问句可以适当添加语气助词。直接回复翻译后的结果，不要回答其他无关话语。
            {src_text}"""
            print(f"提问：{prompt}")
            # 调用翻译API（支持失败重试机制）
            try:
                translated = call_llm_api(
                    prompt,
                    api_key=self.api_key,
                    model="qwen-max",
                    temperature=0.2  # 降低随机性
                )
                
                print(f"回答：{translated.strip()}\n")
                # 清洗API返回结果
                sub.text = f"{src_text}\n{translated.strip()}"
            except Exception as e:
                print(f"翻译失败：{str(e)}，使用whisper翻译")
                # 使用whisper翻译
                translated = self.whisper_model.translate(src_text, target_lang=target_lang)
                print(f"whisper回答：{translated.strip()}\n")
                sub.text = f"{src_text}\n{translated.strip()}"
        return subs

    ####################修改字幕中的单词的错误拼写######################
    def correct_spelling(self, input_srt_file: str, output_srt_file: str = None) -> None:
        """处理SRT文件合并并保存"""
        # 读取输入文件
        subs = pysrt.open(input_srt_file, encoding='utf-8')
        
        # 调用原有合并逻辑
        merged_subs = self._correct_spelling(subs)
        
        # 确定输出路径
        save_path = output_srt_file if output_srt_file else input_srt_file
        
        # 写入处理结果
        merged_subs.save(save_path, encoding='utf-8')
        
        print(f"处理完成，保存路径：{save_path}")
    
    
    def _correct_spelling(self, subs: SubRipFile) -> SubRipFile:
        for sub in subs:
            
            # 生成拼写修正指令
            prompt = f"""请修正以下字幕内容的单词拼写错误，回答修正后的字幕内容，不要返回其他任何无关的内容：
            {sub.text}"""
            print(f"问题：{prompt}")
            # 调用LLM API（支持多平台调用）
            corrected = call_llm_api(
                prompt, 
                api_key=self.api_key,
                model="qwen-max"  # 推荐使用最新模型
            )
            sub.text = corrected.strip()  # 取最后一行防止附加说明
            print(f"回答：{corrected.strip()}\n")
        return subs

    def _extract_audio_segment(self, audio, start_sec: float, end_sec: float):
        """音频切割工具（需安装pydub）"""
        return audio[start_sec*1000:end_sec*1000]

    def correct_and_save(self, srt_path: str, audio_path: str):
        """完整处理流程"""
        subs = pysrt.open(srt_path, encoding='utf-8')
        corrected = self.correct_spelling(subs, audio_path)
        corrected.save(srt_path, encoding='utf-8')
        print(f"已完成拼写修正并覆盖保存：{srt_path}")


    ####################将字幕中的长句才分为短句(有bug不可用)######################


    def split_long_lines(self, input_srt_file: str, output_srt_file: str = None, max_line_count=99) -> None:
        """
        优化字幕文件：拆分超长行并重新计算时间轴
        :param input_srt_file: 输入SRT文件路径
        :param output_srt_file: 输出SRT文件路径（空则覆盖原文件）
        """
        subs = pysrt.open(input_srt_file, encoding='utf-8')
        mp4file = input_srt_file.replace('.srt', '.mp4')
        if not os.path.exists(mp4file):
            raise FileNotFoundError("未找到对应的视频文件"+mp4file)
        
        # 加载视频并提取音频
        audio = AudioSegment.from_file(mp4file, format="mp4")

        new_subs = SubRipFile()
        for idx, sub in enumerate(subs):
            if len(sub.text) <= max_line_count:
                new_subs.append(sub)
                continue
            
            # 调用大模型进行语义拆分‌:ml-citation{ref="2" data="citationList"}
            prompt = f"将以下一行长字幕内容按英文语义拆分为多行字幕显示，保持原句内容的不变，不要返回语句外的其他任何多余的内容，将拆分出来的多行字幕的每一行用换行隔开：\n{sub.text}"
            print(f"提问：{prompt}")
            response = call_llm_api(prompt, self.api_key, model="qwen-max")
            split_lines = [line.strip() for line in response.split('\n') if line.strip()]
            print(f"回答：{"\n".join(split_lines)}\n")
            # 分割时间段并生成新字幕项‌:ml-citation{ref="1" data="citationList"}
            temp_mp3 = os.path.join(os.path.dirname(input_srt_file), "temp_clip.mp3")
            try:
                time_segments = self._split_time_segment(sub, split_lines, audio, temp_mp3)
            finally:
                if os.path.exists(temp_mp3):
                    os.remove(temp_mp3)
            print(f"\n  split_lines_len:{len(split_lines)}, time_segments_len:{len(time_segments)}\n")
            for i, (text, (start, end)) in enumerate(zip(split_lines, time_segments)):
                new_item = SubRipItem(
                    index=len(new_subs)+1,
                    start=pysrt.SubRipTime(milliseconds=start),
                    end=pysrt.SubRipTime(milliseconds=end),
                    text=text
                )
                print(f"new_item:{new_item}")
                new_subs.append(new_item)
        
        save_path = output_srt_file or input_srt_file
        new_subs.save(save_path, encoding='utf-8')
    
    def _split_time_segment(self, sub: SubRipItem, split: List[str], audio,temp_mp3) -> List[tuple]:
        """
        使用Whisper精确分割时间段‌:ml-citation{ref="1" data="citationList"}
        """
        split_count = len(split)
        start_ms = sub.start.ordinal
        end_ms = sub.end.ordinal
        segment_duration = (end_ms - start_ms) / split_count
        if split_count > 2:
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)
            clip = audio[start_ms:end_ms]  # start_ms秒到end_ms秒（单位：毫秒）
            clip.export(temp_mp3, format="mp3")

            # 判断当前设备是否为 GPU
            use_gpu = torch.cuda.is_available()
            # 调用Whisper模型
            result = self.whisper_model.transcribe(
                temp_mp3, 
                language="en",
                word_timestamps=True,
                task='transcribe',  # 明确任务类型
                fp16=use_gpu  # GPU 启用 FP16，CPU 禁用
            )
            return self._align_whisper_timestamps(result, split)
        
        # 简单平均分配时间
        return [(start_ms + i*segment_duration, 
                start_ms + (i+1)*segment_duration) 
                for i in range(split_count)]


    def _align_whisper_timestamps(self, whisper_result, split_list: List[str]) -> List[tuple]:
        """
        基于动态时间规整的精细化时间戳对齐
        """
        # 提取识别文本与时间戳序列
        recognized_words = [word['word'].lower() for segment in whisper_result['segments'] 
                        for word in segment['words']]
        word_timestamps = [{'start': word['start'], 'end': word['end']} 
                        for segment in whisper_result['segments'] 
                        for word in segment['words']]

        # 构建目标文本序列（处理标点符号）
        target_words = []
        for line in split_list:
            target_words.extend(re.findall(r"\w+", line.lower()))

        # 使用动态时间规整(DTW)对齐
        alignment_path = self._dtw_alignment(recognized_words, target_words)

        # 按分割点分配时间戳
        split_indices = self._find_split_indices(alignment_path, len(split_list))
        
        # 合并对应时间范围
        time_ranges = []
        for i in range(len(split_indices)-1):
            start_idx = split_indices[i]
            end_idx = split_indices[i+1]-1
            start_time = word_timestamps[start_idx]['start'] * 1000  # 转毫秒
            if end_idx < len(word_timestamps):
                end_time = word_timestamps[end_idx]['end'] * 1000
            else:
                end_time = word_timestamps[-1]['end'] * 1000  # 取最后一个有效时间戳
                print(f"Split index {end_idx} exceeds timestamp array length")
            time_ranges.append((start_time, end_time))
        
        return time_ranges

    def _dtw_alignment(self, src_seq, tgt_seq):
        """
        动态时间规整对齐算法实现
        """
        # 创建距离矩阵（此处使用Levenshtein距离）
        distance_matrix = np.zeros((len(src_seq)+1, len(tgt_seq)+1))
        for i in range(len(src_seq)+1):
            for j in range(len(tgt_seq)+1):
                if i == 0 or j == 0:
                    distance_matrix[i][j] = max(i, j)
                else:
                    cost = 0 if src_seq[i-1] == tgt_seq[j-1] else 1
                    distance_matrix[i][j] = cost + min(
                        distance_matrix[i-1][j],    # 插入
                        distance_matrix[i][j-1],    # 删除 
                        distance_matrix[i-1][j-1]   # 替换
                    )

        # 回溯寻找最优路径
        i, j = len(src_seq), len(tgt_seq)
        path = [(i, j)]
        while i > 0 and j > 0:
            min_dir = np.argmin([
                distance_matrix[i-1][j], 
                distance_matrix[i][j-1],
                distance_matrix[i-1][j-1]
            ])
            if min_dir == 0:
                i -= 1
            elif min_dir == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
            path.append((i, j))
        
        return list(reversed(path))
    
    def _find_split_indices(self, alignment_path: List[tuple], num_splits: int) -> List[int]:
        """动态分割点检测算法"""
        # 初始化分割点容器（包含起始点0）
        split_points = [0]
        current_split = 0
        target_step = len(alignment_path) // num_splits
        
        # 阶段1：基于DTW路径的初步分割
        for path_idx, (src_idx, tgt_idx) in enumerate(alignment_path):
            if tgt_idx > current_split:
                # 记录当前分割点对应的源序列位置
                split_points.append(src_idx)
                current_split += 1
                
                # 提前达到分割数时退出
                if current_split >= num_splits:
                    break
        
        # 阶段2：均匀填充剩余分割点（处理不足情况）
        while len(split_points) < num_splits + 1:
            last_point = split_points[-1]
            remaining = len(alignment_path) - last_point
            step = max(1, remaining // (num_splits - len(split_points) + 1))
            split_points.append(last_point + step)
        
        # 阶段3：后处理优化（确保单调递增）
        processed = [split_points[0]]
        for p in split_points[1:]:
            if p <= processed[-1]:
                processed.append(processed[-1] + 1)
            else:
                processed.append(p)
        
        # 截断至实际路径长度
        processed[-1] = len(alignment_path) - 1
        return sorted(list(set(processed)))[:num_splits+1]