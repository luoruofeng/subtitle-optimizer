import argparse
import os
import traceback

import pysrt
from subtitle_optimizer.core import SubtitleOptimizer

def main():
    parser = argparse.ArgumentParser(
        description="Optimize SRT subtitles via command line"
    )
    parser.add_argument("-i", "--input", help="Input SRT file path")
    parser.add_argument("-o", "--output", default="output.srt", 
                       help="Output file path (default: output.srt)")
    parser.add_argument("--api-key", help="LLM API key (optional)")
    parser.add_argument("--max-merge", type=int, default=999,
                       help="Max lines to merge (default: 999)")
    
    args = parser.parse_args()
    
    optimizer = SubtitleOptimizer(
        api_key=args.api_key,
        max_merge_lines=args.max_merge
    )
    
    try:
        if os.path.exists(args.input) is False:
            print(f"Error: Input file not exists: {args.iput}")
            exit(1)
        subrip_file = pysrt.open(args.input)
        optimized_subs = optimizer.optimize(subrip_file)
        optimized_subs.save(args.output, encoding='utf-8')
        print(f"Optimized subtitle saved to {args.output}")
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()