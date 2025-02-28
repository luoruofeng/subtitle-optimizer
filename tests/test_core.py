import pytest
from src.subtitle_optimizer.core import SubtitleOptimizer
import pysrt
from unittest.mock import Mock

@pytest.fixture
def sample_subs2():
    return [
        pysrt.SubRipItem(1, start="00:00:01,000", end="00:00:03,000", text="Did you say"),
        pysrt.SubRipItem(2, start="00:00:03,000", end="00:00:05,000", text="there's a bridge over there"),
        pysrt.SubRipItem(3, start="00:00:05,000", end="00:00:07,000", text="There's only"),
        pysrt.SubRipItem(4, start="00:00:07,000", end="00:00:09,000", text="an arch over there"),
        pysrt.SubRipItem(5, start="00:00:09,000", end="00:00:13,000", text="not a bridge")
    ]

def test_merge_english_subtitles2(sample_subs2):
    optimizer = SubtitleOptimizer(max_merge_lines=999)  # 允许连续合并
    
    # 模拟连续合并判断流程（需传递第三行参数）
    optimizer._should_merge = Mock(side_effect=[
        True,  # 合并第1-2行（检测第三行存在）
        False,  # 合并第2-3行（检测第四行存在）
        True,  # 合并第3-4行（检测第五行存在）
        True,  # 合并第4-5行（无后续行）
        False  # 终止合并
    ])
    
    merged = optimizer.optimize(sample_subs2)
    
    # 验证合并结果
    assert len(merged) == 2, "应合并为1条完整字幕"
    assert merged[0].text == "Did you say there's a bridge over there", "文本拼接错误"
    assert merged[0].start == "00:00:01,000", "起始时间未取首行时间"
    assert merged[0].end == "00:00:05,000", "结束时间未取末行时间"

    assert merged[1].text == "There's only an arch over there not a bridge", "文本拼接错误"
    assert merged[1].start == "00:00:05,000", "起始时间未取首行时间"
    assert merged[1].end == "00:00:13,000", "结束时间未取末行时间"
    


@pytest.fixture
def sample_subs():
    return [
        pysrt.SubRipItem(1, start="00:00:01,000", end="00:00:03,000", text="Our intense discussion"),
        pysrt.SubRipItem(2, start="00:00:03,000", end="00:00:05,000", text="that day effectively"),
        pysrt.SubRipItem(3, start="00:00:05,000", end="00:00:07,000", text="brought about significant changes to the company's"),
        pysrt.SubRipItem(4, start="00:00:07,000", end="00:00:09,000", text="financial department management"),
        pysrt.SubRipItem(5, start="00:00:09,000", end="00:00:13,000", text="plan")
    ]

def test_merge_english_subtitles(sample_subs):
    optimizer = SubtitleOptimizer(max_merge_lines=999)  # 允许连续合并
    
    # 模拟连续合并判断流程（需传递第三行参数）
    optimizer._should_merge = Mock(side_effect=[
        True,  # 合并第1-2行（检测第三行存在）
        True,  # 合并第2-3行（检测第四行存在）
        True,  # 合并第3-4行（检测第五行存在）
        True,  # 合并第4-5行（无后续行）
        False  # 终止合并
    ])
    
    merged = optimizer.optimize(sample_subs)
    
    # 验证合并结果
    assert len(merged) == 1, "应合并为1条完整字幕"
    assert merged[0].text == "Our intense discussion that day effectively brought about significant changes to the company's financial department management plan", "文本拼接错误"
    assert merged[0].start == "00:00:01,000", "起始时间未取首行时间"
    assert merged[0].end == "00:00:13,000", "结束时间未取末行时间"
    
    # 验证合并逻辑参数传递
    # 验证调用参数（示例验证第一次调用）
    optimizer._should_merge.assert_any_call(
        "Our intense discussion",  # 当前已合并行（初始为单行）
        "that day effectively",       # 下一行文本
        "brought about significant changes to the company's"  # 第三行文本（判断是否可连续合并）
    )

def test_max_merge_limit(sample_subs):
    optimizer = SubtitleOptimizer(max_merge_lines=2)
    optimizer._should_merge = Mock(side_effect=[True, True])
    
    merged = optimizer.optimize(sample_subs)
    assert len(merged) == 3
    assert merged[0].text == "Our intense discussion that day effectively"


def test_empty_subtitles():
    optimizer = SubtitleOptimizer()
    subs = pysrt.SubRipFile()
    merged = optimizer.optimize(subs)
    assert len(merged) == 0

def test_single_subtitle(sample_subs):
    optimizer = SubtitleOptimizer()
    merged = optimizer.optimize(sample_subs[:1])
    assert len(merged) == 1
    assert merged[0].text == "Our intense discussion"