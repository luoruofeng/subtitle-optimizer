import re
import os
from typing import List, Dict, Optional
from openai import OpenAI
from openai.types.chat import ChatCompletion

from subtitle_optimizer.exceptions import LanguageMismatchError

def call_llm_api(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = "deepseek-r1-distill-llama-70b",  
    system_prompt: str = "你是一个专业的AI助手，用简洁准确的答案回答问题",
    temperature: float = 0.3,
    max_tokens: int = 2000
) -> str:
    """
    调用阿里云百炼平台的DeepSeek大模型API
    
    参数：
    :param prompt: 用户输入的提示文本
    :param api_key: 阿里云API密钥（默认从环境变量DASHSCOPE_API_KEY读取）[4](@ref)
    :param model: 模型名称，默认为deepseek-r1[8](@ref)
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