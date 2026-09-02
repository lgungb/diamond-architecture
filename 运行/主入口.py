# -*- coding: utf-8 -*-
# ============================================================
# 文件：主入口.py
# 作用：整个项目的【唯一启动入口】。用户只需要运行这一条命令：
#       python 运行/主入口.py
#       它会先做环境自检，然后自动跑完整实验，最后生成报告。
#
# 需要的条件：
#   - 在项目根目录下运行（或任意位置运行，代码会自动定位项目根）。
#   - 需要装好 requirements.txt 里的依赖。
# ============================================================

# ---------- 导入标准库 ----------
import sys                          # 系统相关
import os                           # 环境变量
from pathlib import Path            # 路径工具

# ---------- 第一步：把项目根目录加入模块搜索路径 ----------
# 这样无论在哪个目录运行，都能正确找到 配置/模型/数据 等子包。
项目根 = Path(__file__).resolve().parent.parent
if str(项目根) not in sys.path:
    sys.path.insert(0, str(项目根))

# ---------- 第二步：设置 PyTorch 显存分配策略 ----------
# expandable_segments 能显著减少显存碎片（尤其反复加载/释放大模型时），
# 缓解 "CUDA out of memory"。必须在导入 torch 之前设置才生效。
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def 主函数():
    """程序主流程：自检 → 读配置 → 跑实验。"""
    # 1) 环境自检（缺库/缺显卡会有明确提示）
    print("第一步：环境自检 ...")
    from 工具.环境自检 import 运行自检
    运行自检()
    # 2) 硬性检查：PyTorch 必须"能导入 + 版本够新 + 有 GPU"。
    #    三个条件缺一不可，否则直接停下并给出指引：
    #    ① torch 已安装（能 import）
    #    ② torch 版本 >= 2.5（transformers 5.x 硬性要求，否则会"禁用 PyTorch"）
    #    ③ 有可用 GPU（2B 全参数微调在 CPU 上不可行）
    try:
        import torch  # noqa: F401  只验证能导入，不真正使用
    except ImportError:
        print(
            "\n❌ 当前 Python 环境里【没有安装 PyTorch（torch）】，实验无法运行。\n"
            "原因排查（两种最常见情况）：\n"
            "  ① 你在笔记本里用 pip 装库时，装到了另一个 Python 环境（终端 python 和 notebook 内核是两套）。\n"
            "  ② torch 确实没装过。\n"
            "解决办法（在 notebook 里逐个执行，装完必须【重启内核/运行时】再重跑本程序）：\n"
            "    !pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
            "装好 torch 后，重新运行本程序即可继续。"
        )
        return
    # 版本检查：transformers 5.x 要求 torch>=2.5，否则会禁用 PyTorch
    try:
        主版本号, 次版本号 = (int(段) for 段 in torch.__version__.split("+")[0].split(".")[:2])
    except Exception:
        主版本号, 次版本号 = 0, 0   # 解析失败时按最旧处理，宁可报错也不要悄悄放行
    if (主版本号, 次版本号) < (2, 5):
        print(
            f"\n❌ 当前 torch 版本太旧（{torch.__version__}），transformers 5.x 要求 >=2.5。\n"
            "解决办法：先确认你在【GPU 实例】上（看 nvidia-smi 是否存在），然后执行：\n"
            "    !pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124\n"
            "装完重启内核再重跑。"
        )
        return
    # GPU 检查：没有 GPU 时 2B 全参微调在 CPU 上不可行（会跑几小时到几天）
    if not torch.cuda.is_available():
        print(
            "\n❌ 当前环境【没有可用 GPU】（torch 是 CPU 版，或当前是 CPU 实例，连 nvidia-smi 都没有）。\n"
            "本实验要全参数微调 2B 模型，CPU 上不现实。\n"
            "解决办法：到魔搭【新建 Notebook】时选择【GPU 免费实例】（例如 A10 24G），\n"
            "在该实例里重新 git clone 本仓库、安装依赖后运行。\n"
            "（如果你有 GPU 但没被识别，重启实例后再看 nvidia-smi 是否出现）"
        )
        return
    # 3) 读配置
    from 配置.配置 import 配置
    # 4) 跑完整实验
    from 实验.实验总控 import 运行完整实验
    运行完整实验(配置)
    # 5) 完成提示
    print("\n🎉 实验全部完成！请查看 结果/报告.md 获取对比结论。")


# ---------- 程序入口 ----------
# 只有直接运行本文件时才执行，被 import 时不执行。
if __name__ == "__main__":
    主函数()
