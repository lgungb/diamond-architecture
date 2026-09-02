# -*- coding: utf-8 -*-
# ============================================================
# 文件：环境体检.py
# 作用：把"当前 Python 环境到底怎么回事"一次性查清楚，输出诊断报告。
#       当出现"torch 找不到 / 库装错环境 / 显卡看不到"这类问题时，
#       单独运行本文件，把输出整段发给维护者，就能快速定位问题。
#
# 运行方式（任选其一）：
#   - 终端：python 工具/环境体检.py
#   - notebook 单元格：!python 工具/环境体检.py
#
# 需要条件：只依赖 Python 标准库，无需任何第三方库即可运行。
# ============================================================

import sys          # 获取当前解释器信息
import os           # 获取当前工作目录
import subprocess   # 执行外部命令（pip / nvidia-smi 等）


def 运行命令(命令: list) -> str:
    """执行一条命令并返回输出文本；失败时返回错误说明。

    参数：命令 是命令加参数组成的列表，例如 ["pip", "--version"]。
    返回：命令的输出字符串。
    """
    try:
        结果 = subprocess.run(命令, capture_output=True, text=True, timeout=60)
        return (结果.stdout or 结果.stderr).strip() or "(无输出)"
    except Exception as 异常:
        return f"(执行失败：{异常})"


def 检查能否导入(库名: str) -> str:
    """尝试导入一个库，返回"是否成功 + 版本信息"。

    重点说明：pip 显示装了某库，不代表【当前这个 Python】能 import 它。
    这里用 __import__ 在当前解释器里实测，结果才靠谱。
    参数：库名 例如 "torch"。
    返回：描述字符串。
    """
    try:
        模块 = __import__(库名)
        return f"[有] {库名} = {getattr(模块, '__version__', '未知版本')}"
    except Exception as 异常:
        return f"[无] {库名} → {type(异常).__name__}: {异常}"


def 主函数():
    """打印一份完整的环境体检报告。"""
    print("=" * 60)
    print("【环境体检报告】")
    print("=" * 60)
    print("【1】Python 解释器路径：")
    print("    " + sys.executable)
    print("【2】Python 版本：")
    print("    " + sys.version.replace("\n", " "))
    print("【3】当前工作目录：")
    print("    " + os.getcwd())
    print()
    print("【4】关键库能否导入（注意：pip 显示装了 ≠ 当前解释器能 import）：")
    for 库名 in ["torch", "transformers", "modelscope",
                 "accelerate", "peft", "datasets", "qwen_vl_utils"]:
        print("    " + 检查能否导入(库名))
    print()
    print("【4.5】torch 的 GPU 状态（CPU 版 的 torch.version.cuda 是 None）：")
    try:
        import torch
        print("    torch 版本:", torch.__version__)
        print("    torch.cuda.is_available():", torch.cuda.is_available())
        print("    torch.version.cuda:", torch.version.cuda)
    except Exception as 异常:
        print("    torch 不可用:", 异常)
    print()
    print("【5】当前解释器对应的 pip 版本：")
    print("    " + 运行命令([sys.executable, "-m", "pip", "--version"]))
    print("【6】pip 记录里是否装过 torch（未装会显示提示）：")
    print("    " + 运行命令([sys.executable, "-m", "pip", "show", "torch"]))
    print()
    print("【7】显卡信息（nvidia-smi）：")
    print("    " + 运行命令(["nvidia-smi"]))
    print("=" * 60)
    print("请把以上整段输出发给维护者。")
    print("=" * 60)


# ---------- 程序入口 ----------
if __name__ == "__main__":
    主函数()
