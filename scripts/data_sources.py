#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""config/data_sources.yaml 路径解析：${VAR} / \\$VAR 与旧写法 <SPI_ROOT>/<AoT_ROOT>/<WFP_ROOT>/<obj>
统一按环境变量展开（<obj> ← $WFP_OBJ）；仍有未解析占位符时报错退出，而不是带着字面路径继续跑。"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import yaml

WFP_ROOT = Path(__file__).resolve().parent.parent
DS_PATH = WFP_ROOT / "config" / "data_sources.yaml"
_LEGACY = {"<SPI_ROOT>": "${SPI_ROOT}", "<AoT_ROOT>": "${AOT_ROOT}", "<AOT_ROOT>": "${AOT_ROOT}",
           "<WFP_ROOT>": "${WFP_ROOT}", "<obj>": "${WFP_OBJ}"}


def load() -> dict:
    return yaml.safe_load(DS_PATH.read_text(encoding="utf-8")) or {}


def resolve(value: str | None, what: str, required: bool = True) -> Path | None:
    """展开环境变量；WFP_ROOT 缺省为本仓库根。未解析的占位符 → required 时报错退出。"""
    if not value:
        if required:
            sys.exit(f"错误：{what} 未配置（config/data_sources.yaml 或命令行参数）。")
        return None
    s = str(value)
    for a, b in _LEGACY.items():
        s = s.replace(a, b)
    env = dict(os.environ)
    env.setdefault("WFP_ROOT", str(WFP_ROOT))
    s = re.sub(r"\$\{(\w+)\}|\$(\w+)", lambda m: env.get(m.group(1) or m.group(2), m.group(0)), s)
    left = list(dict.fromkeys(re.findall(r"\$\{\w+\}|\$\w+", s)))
    if left:
        if required:
            sys.exit(f"错误：{what} 路径含未设置的环境变量 {left}：{s}\n"
                     f"请 export {' '.join(x.strip('${}') + '=...' for x in left)}，或用命令行参数直接指定路径。")
        return None
    return Path(os.path.expanduser(s))
