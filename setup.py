from setuptools import setup, find_packages
import os

def get_requirements():
    """从 requirements.txt 读取依赖列表"""
    with open('requirements.txt') as f:
        return [
            line.strip() 
            for line in f.readlines()
            if line.strip() and not line.startswith(('#', '-'))
        ]
    

setup(
    name="subtitle-optimizer",
    author="luoruofeng",
    author_email="717750878@qq.com",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=get_requirements(), 
    entry_points={
        'console_scripts': [
            'so=subtitle_optimizer.cli:main'
        ]
    }
)