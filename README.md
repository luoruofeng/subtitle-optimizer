# subtitle-optimizer

[![PyPI Version](https://img.shields.io/pypi/v/subtitle-optimizer.svg)](https://pypi.org/project/subtitle-optimizer/)

An intelligent toolkit for optimizing SRT/ASS subtitle files with AI-powered sentence merging and proofreading.

## Features
- ​**Smart Merging**: Combine fragmented subtitles into natural sentences using LLM
- ​**Multilingual Support**: Auto-detect English/Chinese and apply formatting rules
- ​**Error Handling**: Fallback to local Ollama models if cloud API fails

## Installation
```bash
pip install subtitle-optimizer
```


# Quick Start
```
from subtitle_optimizer import SubtitleOptimizer

optimizer = SubtitleOptimizer(api_key="your-api-key")
optimized_subs = optimizer.optimize("input.srt")
optimized_subs.save("output.srt")
```


# Advanced Options
```
# Customize merge behavior
optimizer = SubtitleOptimizer(
    max_merge_lines=5, 
    llm_endpoint="http://localhost:11434"
)
```