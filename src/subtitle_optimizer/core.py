from pysrt import SubRipFile, SubRipItem
import pysrt
from subtitle_optimizer.utils import call_llm_api, detect_language, format_merged_text
from subtitle_optimizer.exceptions import LanguageMismatchError
import whisper

class SubtitleOptimizer:
    def __init__(self, api_key=None, max_merge_lines=999):
        self.api_key = api_key
        self.max_merge_lines = max_merge_lines
        # 初始化本地Whisper大模型（推荐使用large-v3版本）[4](@ref)
        self.whisper_model = whisper.load_model("large") 

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
                model="gpt-4-turbo"  # 推荐使用最新模型
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
    
    def optimize(self, subs: SubRipFile) -> SubRipFile:
        merged_subs = []
        i = 0
        
        while i < len(subs):
            current_group = [subs[i]]
            merged_text = subs[i].text
            next_candidate = i + 1
            
            print(f"\n[DEBUG] 开始处理字幕组，起始索引: {i}")
            print(f"[DEBUG] 初始合并组: {[sub.text for sub in current_group]}")
            print(f"[DEBUG] 初始合并文本: '{merged_text}'")

            while True:
                if next_candidate >= len(subs) or len(current_group) >= self.max_merge_lines:
                    print(f"[DEBUG] 终止合并 - 剩余字幕: {len(subs)-next_candidate}, 当前合并行数: {len(current_group)}")
                    break
                
                next_sub = subs[next_candidate]
                next_next_sub = subs[next_candidate + 1] if next_candidate + 1 < len(subs) else None
                
                try:
                    args = [merged_text, next_sub.text]
                    if next_next_sub:
                        args.append(next_next_sub.text)
                    
                    # 打印合并判断参数
                    print(f"\n[DEBUG] 尝试合并 {len(current_group)} 行到 {len(current_group)+1} 行")
                    print(f"[DEBUG] 当前合并文本: '{args[0]}'")
                    print(f"[DEBUG] 下一行文本:  '{args[1]}'")
                    print(f"[DEBUG] 第三行存在:   {args[2] if len(args)>2 else '无'}")

                    if self._should_merge(*args):
                        current_group.append(next_sub)
                        merged_text = self._format_text(current_group)
                        next_candidate += 1
                        print(f"[DEBUG] ✅ 合并成功！当前合并行数: {len(current_group)}")
                        print(f"[DEBUG] 更新后合并文本: '{merged_text}'")
                    else:
                        print(f"[DEBUG] ❌ 合并终止 {len(current_group)} 行")
                        break
                except LanguageMismatchError as e:
                    print(f"[DEBUG] 🚫 语言不匹配异常: {str(e)}")
                    break

            new_sub = self._create_merged_sub(current_group, merged_text)
            merged_subs.append(new_sub)
            print(f"\n[DEBUG] 生成合并条目: {new_sub.start} -> {new_sub.end}")
            print(f"[DEBUG] 最终合并文本: '{new_sub.text}'\n")
            
            i = next_candidate
        
        print(f"\n[DEBUG] 合并完成，总条目数: {len(merged_subs)}")
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
        print(f"\n[DEBUG] 执行合并判断 _should_merge:")
        print(f"text1 ({len(text1)} chars): '{text1}'")
        print(f"text2 ({len(text2)} chars): '{text2}'")
        print(f"text3 ({len(text3) if text3 else 'None'}): '{text3 if text3 else '无'}'")

        if detect_language(text1) != detect_language(text2):
            print("[DEBUG] 语言不匹配触发异常")
            raise LanguageMismatchError("Language mismatch")

        # 打印语言检测结果
        lang1 = detect_language(text1)
        lang2 = detect_language(text2)
        lang3 = detect_language(text3) if text3 else None
        print(f"[DEBUG] 语言检测结果: text1={lang1}, text2={lang2}, text3={lang3}")

        if lang1 != lang2 or lang2 != lang3:
            print(f"[DEBUG] 语言不匹配: {lang1} != {lang2} != {lang3} text1={text1} text2={text2} text3={text3}")
            return False

        prompt = ""
        if text3 is not None and lang1 == lang2 and lang2 == lang3:
            prompt = f"在一个没有标点符号的字幕文件中，第三行字幕是“{text3}”的情况下，第一行字幕和第二行是否可能是属于同一句话?可能属于的回答我“Y”，可能不属于回答我“N”，不要回答任何信息。\n第一行: {text1}\n第二行: {text2}"
        else:
            prompt = f"在一个没有标点符号的字幕文件中，在没有第三行字幕的情况下，第一行字幕和第二行是否可能是属于同一句话?可能属于的回答我“Y”，可能不属于回答我“N”，不要回答任何信息。\n第一行: {text1}\n第二行: {text2}"
        
        print(f"[DEBUG] 发送给LLM的提示:\n{prompt}")
        response = call_llm_api(prompt, api_key=self.api_key)
        print(f"[DEBUG] LLM响应原始结果: '{response}'")
        
        result = 'Y' in response
        print(f"[DEBUG] 合并判断最终结果: {'Y' if result else 'N'}")
        return result