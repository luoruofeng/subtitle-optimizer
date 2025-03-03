import argparse
import sys
from subtitle_optimizer.core import SubtitleOptimizer
import os


OPTIMIZER:SubtitleOptimizer = None

def main():
    # 创建主解析器
    parser = argparse.ArgumentParser(prog="so", description="字幕优化工具")
    subparsers = parser.add_subparsers(dest="command", required=True, help="支持的子命令")

    # 添加子命令：correct_spelling
    correct_parser = subparsers.add_parser("correct_spelling", help="拼写检查与修正")
    correct_parser.add_argument("-i", "--input", required=True, help="输入字幕文件路径")
    correct_parser.add_argument("-o", "--output", help="输出文件路径（默认覆盖原文件）")
    correct_parser.add_argument("--lang", default="en", help="语言类型（默认：英文）")

    # 添加子命令：merge_srt
    merge_parser = subparsers.add_parser("merge_srt", help="合并多个字幕文件")
    merge_parser.add_argument("-i", "--input", required=True, help="输入字幕文件路径（支持通配符）")
    merge_parser.add_argument("-o", "--output", required=True, help="合并后的输出文件路径")

    # 添加子命令：split_long_lines
    split_parser = subparsers.add_parser("split_long_lines", help="拆分过长字幕行")
    split_parser.add_argument("-i", "--input", required=True, help="输入字幕文件路径")
    split_parser.add_argument("-o", "--output", help="输出文件路径（默认覆盖原文件）")
    split_parser.add_argument("--max-length", type=int, default=33, help="每行最大字符数（默认：33）")

    # 解析参数并执行对应逻辑
    args = parser.parse_args()

    # 获取 DASHSCOPE_API_KEY 环境变量
    api_key = os.getenv('DASHSCOPE_API_KEY')

    # 检查是否成功获取到 API 密钥
    if api_key:
        print(f"成功获取 DASHSCOPE_API_KEY: {api_key}")
    else:
        print("未找到 DASHSCOPE_API_KEY 环境变量，请确保已正确设置。")
        sys.exit(1)  # 以非零状态码退出程序，表示发生了错误

    global OPTIMIZER
    OPTIMIZER = SubtitleOptimizer(
        api_key=api_key
    )
    
    if args.command == "correct_spelling":
        handle_correct_spelling(args)
    elif args.command == "merge_srt":
        handle_merge_srt(args)
    elif args.command == "split_long_lines":
        handle_split_lines(args)

def handle_correct_spelling(args):
    print(f"执行拼写修正：输入={args.input}, 输出={args.output}, 语言={args.lang}")

def handle_merge_srt(args):
    print(f"合并字幕：输入={args.input}, 输出={args.output}")
    OPTIMIZER.merge_srt(args.input,args.output)

def handle_split_lines(args):
    print(f"拆分行：输入={args.input}, 输出={args.output}, 最大长度={args.max_length}")
    OPTIMIZER.split_long_lines(args.input,args.output)

if __name__ == "__main__":
    main()