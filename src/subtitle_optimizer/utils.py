import re
import os
from typing import List, Dict, Optional
import numpy as np
from openai import OpenAI
from openai.types.chat import ChatCompletion

import os
import subprocess
import tempfile
import os
import subprocess
import tempfile
import librosa
import soundfile as sf

def time_stretch_audio(mp4_path: str, rate: float, gain_factor: float = 1.5):
    """
    将MP4文件的音频流变速处理并保存为同名WAV文件
    参数说明：
    mp4_path : 输入MP4文件路径（如"/data/video.mp4"）
    rate     : 变速比率（0.5-2.0），例如0.7表示减速到80%速度
    """
    # 验证输入参数有效性
    if not os.path.isfile(mp4_path):
        raise FileNotFoundError(f"输入文件不存在: {mp4_path}")
    if not (0.5 <= rate <= 2.0):
        raise ValueError("变速比率应在0.5到2.0之间")

    # 生成输出路径（与原文件同目录同名，后缀改为.wav）
    base_name = os.path.splitext(os.path.basename(mp4_path))[0]
    output_path = os.path.join(
        os.path.dirname(mp4_path),
        f"{base_name}_stretched.wav"
    )

    # 创建临时文件保存提取的原始音频
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        temp_audio_path = tmp_file.name

    try:
        # 使用FFmpeg提取音频流（网页7提到的MP4结构分离特性）
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", mp4_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "44100",  # 统一采样率
            "-ac", "2",      # 统一双声道
            temp_audio_path
        ]
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)

        # 加载音频并处理
        y, sr = librosa.load(temp_audio_path, sr=None, mono=False)
        y_stretched = librosa.effects.time_stretch(y, rate=rate)

        # =========== 新增音量控制代码 ===========
        # 应用线性增益（推荐1.0-3.0）
        y_stretched *= gain_factor
        # 防止削波（限制在[-1,1]范围）
        y_stretched = np.clip(y_stretched, -1.0, 1.0)
        # ======================================

        # 保存处理后的音频
        sf.write(output_path, y_stretched.T, sr, subtype='PCM_16')
        
        print(f"处理完成，保存至: {output_path}")
        return output_path

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"音频提取失败: {e.stderr.decode()}") from e
    except Exception as e:
        raise RuntimeError(f"处理异常: {str(e)}") from e
    finally:
        # 清理临时文件
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)



def call_llm_api(
    prompt: str,
    api_key: Optional[str] = None,
    # model: str = "deepseek-r1-distill-llama-70b",  
    model: str = "qwen-max",
    system_prompt: str = "你是一个专业的AI助手，用简洁准确的答案回答问题",
    temperature: float = 0.3,
    max_tokens: int = 2000
) -> str:
    """
    调用阿里云百炼平台的DeepSeek大模型API
    
    参数：
    :param prompt: 用户输入的提示文本
    :param api_key: 阿里云API密钥（默认从环境变量DASHSCOPE_API_KEY读取）
    :param model: 模型名称，默认为deepseek-r1
    :param system_prompt: 系统角色设定
    :param temperature: 生成多样性控制（0.1-1.0）
    :param max_tokens: 最大生成token数
    
    返回：
    :return: 模型生成的文本内容
    
    示例：
    >>> response = call_llm_api("如何用Python实现快速排序？")
    """
    # 自动获取API Key的优先级[4,6](@ref)
    final_api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not final_api_key:
        raise ValueError("未提供API Key且环境变量DASHSCOPE_API_KEY未设置")

    client = OpenAI(
        api_key=final_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 固定接入点[8,11](@ref)
    )

    try:
        completion: ChatCompletion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=300  # 增加超时设置
        )
        
        if completion.choices and completion.choices[0].message.content:
            return completion.choices[0].message.content.strip()
        else:
            return "模型未返回有效响应"
            
    except Exception as e:
        error_msg = f"API调用失败：{str(e)}"
        if hasattr(e, 'response'):
            error_msg += f"\n状态码：{e.response.status_code}\n错误详情：{e.response.text}"
        raise RuntimeError(error_msg) from e

# utils.py
def detect_language(text: str, threshold=0.15) -> str:
    """改进后的语言检测方法"""
    # 中日韩统一表意文字区块检测
    cjk_regex = re.compile(
        r'[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF\u3000-\u303F\uFF00-\uFFEF]'
    )
    cjk_count = len(cjk_regex.findall(text))
    return 'zh' if cjk_count / max(len(text),1) > threshold else 'en'

def format_merged_text(texts: list) -> str:
    base_lang = detect_language(texts[0])
    if base_lang == 'en':
        return ' '.join([t.strip() for t in texts])
    else:
        return ''.join([t.strip() for t in texts])