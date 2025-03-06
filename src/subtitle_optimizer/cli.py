import argparse
import sys
import traceback
from subtitle_optimizer.core import SubtitleOptimizer
import os


OPTIMIZER:SubtitleOptimizer = None

def main():
    # 创建主解析器
    parser = argparse.ArgumentParser(prog="so", description="字幕优化工具")
    subparsers = parser.add_subparsers(dest="command", required=True, help="支持的子命令")
    
    # 添加子命令：process_srt_with_voice
    process_srt_with_voice_parser = subparsers.add_parser("process-srt-with-voice", help="通过srt生成mp3")
    process_srt_with_voice_parser.add_argument("-i", "--input", required=True, help="输入字幕文件路径")

    # 添加子命令：correct_spelling
    correct_parser = subparsers.add_parser("correct-spelling", help="拼写检查与修正")
    correct_parser.add_argument("-i", "--input", required=True, help="输入字幕文件路径")
    correct_parser.add_argument("-o", "--output", help="输出文件路径（默认覆盖原文件）")
    correct_parser.add_argument("--lang", default="en", help="语言类型（默认：英文）")

    
    # 添加子命令：adjust_video_speed
    adjust_video_speed_parser = subparsers.add_parser("adjust-video-speed", help="视频变速")
    adjust_video_speed_parser.add_argument("-i", "--input", required=True, help="输入mp4文件路径")
    adjust_video_speed_parser.add_argument("-s", "--speed", required=False, default=0.9, type=float, help="速率（默认 0.9）")

    
    # 添加子命令：merge_srt
    merge_parser = subparsers.add_parser("merge-srt", help="合并多个字幕文件")
    merge_parser.add_argument("-i", "--input", required=True, help="输入字幕文件路径（支持通配符）")
    merge_parser.add_argument("-o", "--output", required=True, help="合并后的输出文件路径")

    # 添加子命令：split_long_lines
    split_parser = subparsers.add_parser("split-long-lines", help="拆分过长字幕行")
    split_parser.add_argument("-i", "--input", required=True, help="输入字幕文件路径")
    split_parser.add_argument("-o", "--output", help="输出文件路径（默认覆盖原文件）")
    split_parser.add_argument("--max-length", type=int, default=99, help="每行最大字符数（默认：99）")

    # 添加子命令：add_translation
    add_translation_parser = subparsers.add_parser("add-translation", help="翻译字幕文件")
    add_translation_parser.add_argument("-i", "--input", required=True, help="输入字幕文件路径（支持通配符）")
    add_translation_parser.add_argument("-o", "--output", required=False, help="合并后的输出文件路径")

    # 添加子命令：extract_text_to_txt
    extract_text_to_txt_parser = subparsers.add_parser("extract-text-to-txt", help="读取mp4中的文本到txt文件")
    extract_text_to_txt_parser.add_argument("-i", "--input", required=True, help="mp4文件")

    # 在现有的subparsers中添加以下代码
    generate_srt_parser = subparsers.add_parser("generate-srt-from-folder", 
                                            help="通过包含了同名的TXT和MP4的文件夹生成SRT字幕文件")
    generate_srt_parser.add_argument("-i", "--input", nargs='+', required=True,
                              help="输入路径：1) 文件夹路径 或 2) 文件路径元组 (txt, mp4 [, segments])")

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
    
    if args.command == "correct-spelling":
        handle_correct_spelling(args)
    elif args.command == "merge-srt":
        handle_merge_srt(args)
    elif args.command == "split-long-lines":
        handle_split_lines(args)
    elif args.command == "add-translation":
        add_translation(args)
    elif args.command == "extract-text-to-txt":
        extract_text_to_txt(args)
    elif args.command == "generate-srt-from-folder":
        handle_generate_srt_from_folder(args)
    elif args.command == "process-srt-with-voice":
        handle_process_srt_with_voice(args)
    elif args.command == "adjust-video-speed":
        handle_adjust_video_speed(args)

def handle_adjust_video_speed(args):
    print(f"🎥 视频变速处理：调整视频速率为{args.speed}\n输入={args.input}")
    OPTIMIZER.adjust_video_speed(args.input,args.speed)

def handle_process_srt_with_voice(args):
    print(f"🔊 开始生成语音文件：将字幕文件转换为MP3音频\n输入={args.input}")
    OPTIMIZER.process_srt_with_voice(args.input)

def handle_generate_srt_from_folder(args):
    print(f"🔄 正在生成SRT字幕文件：将基于TXT文本内容创建时间轴字幕\n输入参数={args.input}")
    try:
        # 处理文件夹路径
        if len(args.input) == 1 and os.path.isdir(args.input[0]):
            OPTIMIZER.generate_srt_from_folder(args.input[0])
        # 处理文件元组（支持2-3个文件路径）
        elif 2 <= len(args.input) <= 3:
            OPTIMIZER.generate_srt_from_folder(tuple(args.input))
        else:
            raise ValueError("无效的输入参数数量")
    except Exception as e:
        print(f"❌ SRT生成失败：{str(e)}\n错误追踪：")
        traceback.print_exc()

def handle_correct_spelling(args):
    print(f"🔍 启动AI拼写校正：使用{args.lang.upper()}语言模型检查字幕文件\n输入={args.input} 输出={args.output or '覆盖原文件'}")
    OPTIMIZER.correct_spelling(args.input,args.output)

def handle_merge_srt(args):
    print(f"🧩 开始合并字幕文件：自动对齐时间轴并消除重叠片段\n输入模式={args.input} 输出路径={args.output}")
    OPTIMIZER.merge_srt(args.input,args.output)

def handle_split_lines(args):
    print(f"✂️ 执行字幕行拆分：按每行最多{args.max_length}字符优化可读性\n输入={args.input} 输出={args.output or '覆盖原文件'}")
    OPTIMIZER.split_long_lines(args.input,args.output)

def add_translation(args):
    print(f"🌐 启动多语言翻译：通过DASHSCOPE API进行跨语言转换\n输入={args.input} 输出={args.output}")
    OPTIMIZER.add_translation(args.input,args.output)

def extract_text_to_txt(args):
    print(f"🎵 正在提取音轨文本：使用语音识别技术转换MP4音频内容[1,5](@ref)\n输入视频={args.input}")
    OPTIMIZER.extract_text_to_txt(args.input)


if __name__ == "__main__":
    main()