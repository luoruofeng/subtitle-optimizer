import os
import re
from typing import List
import numpy as np
from pysrt import SubRipFile, SubRipItem
import pysrt
from subtitle_optimizer.utils import call_llm_api, detect_language, format_merged_text
from subtitle_optimizer.exceptions import LanguageMismatchError
import whisper
from pydub import AudioSegment
import torch


class SubtitleOptimizer:
    def __init__(self, api_key=None):
        self.api_key = api_key
        # 初始化本地Whisper大模型
        # self.whisper_model = whisper.load_model("large-v3")
        self.whisper_model = whisper.load_model("medium.en")

        


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
        subs = pysrt.open(input_srt_file, encoding='utf-8')
        processed_subs = self._translate_subs(subs, target_lang)
        save_path = output_srt_file if output_srt_file else input_srt_file
        processed_subs.save(save_path, encoding='utf-8')
        print(f"双语字幕已生成，保存路径：{save_path}")

    def _translate_subs(self, subs: SubRipFile, target_lang: str) -> SubRipFile:
        """核心翻译逻辑（支持LLM API和Whisper双引擎）"""
        for sub in subs:
            src_text = sub.text.strip()
            
            # 语言检测与过滤（示例仅处理英文翻译）
            if detect_language(src_text) not in ['en']:
                continue  # 跳过非英文内容[6](@ref)
            
            # 构造翻译指令（支持格式控制）
            prompt = f"""将以下字幕精确翻译为{target_lang}，保持专业术语，禁止添加其他内容，专业术语可以直接保持英文原文。直接回复翻译后的结果，不要回答其他无关话语。
            {src_text}"""
            print(f"提问：{prompt}")
            # 调用翻译API（支持失败重试机制）
            try:
                translated = call_llm_api(
                    prompt,
                    api_key=self.api_key,
                    model="qwen-max",
                    temperature=0.2  # 降低随机性[8](@ref)
                )
                
                print(f"回答：{translated.strip()}\n")
                # 清洗API返回结果
                sub.text = f"{src_text}\n{translated.strip()}"
            except Exception as e:
                print(f"翻译失败：{str(e)}，保留原文")
        
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
            
            # 生成拼写修正指令[6](@ref)
            prompt = f"""请修正以下字幕内容的单词拼写错误，回答修正后的字幕内容，不要返回其他任何无关的内容：
            {sub.text}"""
            print(f"问题：{prompt}")
            # 调用LLM API（支持多平台调用）[6,8](@ref)
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
        subs = pysrt.open(srt_path)
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
        subs = pysrt.open(input_srt_file)
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