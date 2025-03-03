from typing import List
import numpy as np
from pysrt import SubRipFile, SubRipItem
import pysrt
from subtitle_optimizer.utils import call_llm_api, detect_language, format_merged_text
from subtitle_optimizer.exceptions import LanguageMismatchError
import whisper

class SubtitleOptimizer:
    def __init__(self, api_key=None):
        self.api_key = api_key
        # 初始化本地Whisper大模型（推荐使用large-v3版本）[4](@ref)
        self.whisper_model = whisper.load_model("large-v3") 


    ####################修改字幕中的单词的错误拼写######################
    def correct_spelling(self, subs: SubRipFile, audio_path: str) -> SubRipFile:
        """
        通过Whisper语音识别和LLM API修正字幕拼写错误
        :param subs: 字幕文件对象
        :param audio_path: 对应的音频文件路径
        :return: 修正后的字幕文件对象
        """
        # 加载音频并转换为numpy数组[4](@ref)
        audio = whisper.load_audio(audio_path)
        
        for sub in subs:
            # 提取字幕时间段（毫秒转换为秒）
            start_sec = sub.start.ordinal / 1000
            end_sec = sub.end.ordinal / 1000
            
            # 切割音频片段[4](@ref)
            segment = self._extract_audio_segment(audio, start_sec, end_sec)
            
            # 语音识别（使用Whisper的decode模式提升精度）[4](@ref)
            result = self.whisper_model.transcribe(
                segment,
                language=detect_language(sub.text),  # 复用现有语言检测
                word_timestamps=False
            )
            whisper_text = result["text"].strip()
            
            # 生成拼写修正指令[6](@ref)
            prompt = f"""请修正以下字幕文本的拼写错误，保持时间轴同步：
            原文本：{sub.text}
            参考转写：{whisper_text}
            要求：
            1. 修正明显的拼写/语法错误
            2. 保留专业术语
            3. 输出格式：仅返回修正文本"""
            
            # 调用LLM API（支持多平台调用）[6,8](@ref)
            corrected = call_llm_api(
                prompt, 
                api_key=self.api_key,
                model="qwen-max"  # 推荐使用最新模型
            )
            sub.text = corrected.split("\n")[-1]  # 取最后一行防止附加说明
        
        return subs

    def _extract_audio_segment(self, audio, start_sec: float, end_sec: float):
        """音频切割工具（需安装pydub）"""
        from pydub import AudioSegment
        return audio[start_sec*1000:end_sec*1000]

    def correct_and_save(self, srt_path: str, audio_path: str):
        """完整处理流程"""
        subs = pysrt.open(srt_path)
        corrected = self.correct_spelling(subs, audio_path)
        corrected.save(srt_path, encoding='utf-8')
        print(f"已完成拼写修正并覆盖保存：{srt_path}")
    


    ####################将字幕中的短句合并为长句######################

    def merge_srt(self, input_srt_file: str, output_srt_file: str = None) -> None:
        """处理SRT文件合并并保存"""
        # 读取输入文件
        with open(input_srt_file, 'r', encoding='utf-8') as f:
            subs = srt.parse(f.read())
        
        # 调用原有合并逻辑
        merged_subs = self._merge_srt(subs)
        
        # 确定输出路径
        save_path = output_srt_file if output_srt_file else input_srt_file
        
        # 写入处理结果
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(merged_subs))
        
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
                if next_candidate >= len(subs) or len(current_group) >= self.max_merge_lines:
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


    ####################将字幕中的长句才分为短句######################


    def split_long_lines(self, input_srt_file: str, output_srt_file: str = None, max_line_count=33) -> None:
        """
        优化字幕文件：拆分超长行并重新计算时间轴
        :param input_srt_file: 输入SRT文件路径
        :param output_srt_file: 输出SRT文件路径（空则覆盖原文件）
        """
        subs = pysrt.open(input_srt_file)
        audio = whisper.load_audio(input_srt_file.replace('.srt', '.mp4'))  # 假设视频文件与SRT同名‌:ml-citation{ref="1" data="citationList"}
        
        new_subs = SubRipFile()
        for idx, sub in enumerate(subs):
            if len(sub.text) <= max_line_count:
                new_subs.append(sub)
                continue
                
            # 调用大模型进行语义拆分‌:ml-citation{ref="2" data="citationList"}
            prompt = f"将以下一行长字幕内容按英文语义拆分为多行字幕显示，保持原句内容的不变，不要返回语句外的其他任何多余的内容，将拆分出来的多行字幕的每一行用换行隔开：\n{sub.text}"
            response = call_llm_api(prompt, self.api_key, model="qwen-max")
            split_lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            # 分割时间段并生成新字幕项‌:ml-citation{ref="1" data="citationList"}
            time_segments = self._split_time_segment(sub, len(split_lines), audio)
            for i, (text, (start, end)) in enumerate(zip(split_lines, time_segments)):
                new_item = SubRipItem(
                    index=len(new_subs)+1,
                    start=pysrt.SubRipTime(milliseconds=start),
                    end=pysrt.SubRipTime(milliseconds=end),
                    text=text
                )
                new_subs.append(new_item)
        
        save_path = output_srt_file or input_srt_file
        new_subs.save(save_path, encoding='utf-8')
    
    def _split_time_segment(self, sub: SubRipItem, split_count: int, audio: np.ndarray) -> List[tuple]:
        """
        使用Whisper精确分割时间段‌:ml-citation{ref="1" data="citationList"}
        """
        start_ms = sub.start.ordinal
        end_ms = sub.end.ordinal
        segment_duration = (end_ms - start_ms) / split_count
        
        # 当分割次数较多时采用Whisper精确识别‌:ml-citation{ref="1" data="citationList"}
        if split_count > 2:
            result = self.whisper_model.transcribe(
                audio[start_ms//1000:end_ms//1000],
                word_timestamps=True
            )
            return self._align_whisper_timestamps(result, split_count)
        
        # 简单平均分配时间
        return [(start_ms + i*segment_duration, 
                start_ms + (i+1)*segment_duration) 
                for i in range(split_count)]

    def _align_whisper_timestamps(self, whisper_result, expected_splits: int) -> List[tuple]:
        """
        对齐Whisper识别的时间戳与拆分后的文本‌:ml-citation{ref="1" data="citationList"}
        """
        # 实现逻辑：根据语音停顿和词边界调整时间分割点
        # （此处需要根据实际语音特征进行动态调整）
        return [(word['start']*1000, word['end']*1000) 
                for word in whisper_result['words'][:expected_splits]]