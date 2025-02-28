import pytest
from src.subtitle_optimizer.utils import detect_language, format_merged_text

def test_detect_language():
    # 纯中文检测
    assert detect_language("这是一个测试") == "zh"
    # 纯英文检测
    assert detect_language("This is a test") == "en"
    # 混合语言检测
    assert detect_language("Hello 你好") == "zh"
    # 符号处理
    assert detect_language("!!!") == "en"
    # 空字符串处理
    assert detect_language("") == "en"

def test_format_merged_text_english():
    texts = ["Hello,", "world!", "  How are you?"]
    result = format_merged_text(texts)
    assert result == "Hello, world! How are you?"

def test_format_merged_text_chinese():
    texts = ["你好，", "世界！", "  今天好吗？"]
    result = format_merged_text(texts)
    assert result == "你好，世界！今天好吗？"
