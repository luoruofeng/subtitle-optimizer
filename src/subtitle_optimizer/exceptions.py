class MergeError(Exception):
    """字幕合并异常基类"""
    pass

class LanguageMismatchError(MergeError):
    """语言不一致异常"""
    def __init__(self, lang1: str, lang2: str):
        super().__init__(f"语言不一致无法合并：{lang1} vs {lang2}")

class TimeOverlapError(MergeError):
    """时间轴重叠异常"""
    def __init__(self, line1: int, line2: int):
        super().__init__(f"字幕行 {line1} 和 {line2} 存在时间轴重叠")