# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'isaaclab_logistics_vla' python package."""

import os
import re

from setuptools import setup


def parse_toml_simple(toml_path):
    """简单的TOML解析器，只解析我们需要的字段。"""
    with open(toml_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 提取版本号
    version_match = re.search(r'version\s*=\s*"([^"]+)"', content)
    version = version_match.group(1) if version_match else "0.1.0"
    
    # 提取仓库URL
    repo_match = re.search(r'repository\s*=\s*"([^"]+)"', content)
    repository = repo_match.group(1) if repo_match else "https://github.com/your-repo/IsaacLab-Logistics-VLA"
    
    # 提取描述
    desc_match = re.search(r'description\s*=\s*"([^"]+)"', content)
    description = desc_match.group(1) if desc_match else "Extension containing logistics VLA benchmark tasks for robot learning."
    
    # 提取关键词列表
    keywords_match = re.search(r'keywords\s*=\s*\[(.*?)\]', content, re.DOTALL)
    if keywords_match:
        keywords_str = keywords_match.group(1)
        # 提取所有引号内的字符串
        keywords = [k.strip().strip('"').strip("'") for k in re.findall(r'"([^"]+)"', keywords_str)]
    else:
        keywords = ["robotics", "vla", "logistics"]
    
    return {
        "package": {
            "version": version,
            "repository": repository,
            "description": description,
            "keywords": keywords,
        }
    }


# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_PATH = os.path.join(EXTENSION_PATH, "config", "extension.toml")

# 使用简单解析器读取TOML文件
EXTENSION_TOML_DATA = parse_toml_simple(EXTENSION_TOML_PATH)

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    "numpy<2",
    "torch>=2.7",
    "torchvision>=0.14.1",
    "protobuf>=4.25.8,!=5.26.0",
    "toml>=0.10.2",  # For reading extension.toml in runtime (not needed for setup)
    # basic logger
    "tensorboard",
    # VLA specific dependencies
    "transformers>=4.30.0",  # For language models (BERT/CLIP)
    "sentencepiece>=0.1.99",  # For text processing
    "pillow>=9.0.0",  # For image processing
]

PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu128"]

# Installation operation
setup(
    name="isaaclab_logistics_vla",
    author="Isaac Lab Logistics VLA Developers",
    maintainer="Isaac Lab Logistics VLA Developers",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    dependency_links=PYTORCH_INDEX_URL,
    packages=["isaaclab_logistics_vla"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 4.5.0",
        "Isaac Sim :: 5.0.0",
    ],
    zip_safe=False,
)
