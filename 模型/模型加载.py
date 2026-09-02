# -*- coding: utf-8 -*-
# ============================================================
# 文件：模型加载.py
# 作用：负责两件事：
#       1. 从魔搭(modelscope)下载模型到本地缓存（只下载一次）。
#       2. 把模型和分词器加载进显存，返回给上层使用。
#       因为 Qwen3.5 是"多模态"模型（文字/图像/音频都能输入），
#       所以本文件做了"自适应加载"：先试普通文本加载方式，
#       不行就自动切换成多模态加载方式，用户不需要操心。
#
# 需要的条件：
#   - 需要联网（第一次下载模型约 4.5GB）。
#   - 需要安装 modelscope 和 transformers（版本要求见 requirements.txt）。
#   - 推荐有 NVIDIA 显卡（配置里【设备】填 cuda）。
# ============================================================

# ---------- 导入标准库 ----------
import torch                          # 深度学习框架
from pathlib import Path              # 路径工具


# ---------- 一、下载模型 ----------
def 查找已下载模型路径(配置):
    """在本地缓存目录里查找目标模型是否已经下载完成。

    判断标准（modelscope 的缓存规则）：
      下载好的模型放在
        <缓存目录>/models/<模型ID中"/"换成"--">/snapshots/<某个版本号>/ 目录下。
      只要某个"快照目录"里存在模型配置文件（config.json 或 configuration.json），
      就认为这个模型已经下载完整，直接复用该路径，不用联网重新下载。
    参数：
      配置 ：配置项对象（含 模型ID 和 模型缓存目录）。
    返回：
      模型本地路径（字符串）；如果本地没有缓存，返回 None。
    """
    # modelscope 会把模型ID里的 "/" 替换成 "--" 作为缓存目录名（例如 Qwen/Qwen3.5-2B → Qwen--Qwen3.5-2B）
    标准目录 = 配置.模型缓存目录 / "models" / 配置.模型ID.replace("/", "--")
    # 个别缓存可能用原始 ID 命名，也兼容检查一下
    候选目录列表 = [标准目录]
    原始ID目录 = 配置.模型缓存目录 / "models" / 配置.模型ID
    if 原始ID目录 != 标准目录:
        候选目录列表.append(原始ID目录)
    # 逐个候选目录检查
    for 目录 in 候选目录列表:
        快照目录 = 目录 / "snapshots"
        if not 快照目录.is_dir():
            continue  # 没有 snapshots 目录 → 这个模型肯定没下过
        # 遍历所有快照，只要有一个带配置文件的，就认为模型已下载完整
        for 快照 in sorted(快照目录.iterdir()):
            if 快照.is_dir() and (
                (快照 / "config.json").is_file() or (快照 / "configuration.json").is_file()
            ):
                return str(快照)   # 找到了，直接返回这个快照路径
    # 全部没找到 → 没有缓存
    return None


def 下载模型(配置) -> str:
    """拿到模型在本地硬盘上的路径；如果本地没有缓存，才从魔搭联网下载。

    流程（这就是本次加的逻辑）：
      1. 先调用 查找已下载模型路径 检查本地缓存。
      2. 已下载 → 直接返回本地路径，不联网、不重复下载。
      3. 没下载（或配置了 是否强制重新下载=True）→ 调用 snapshot_download 联网下载。
      4. 下载完返回本地路径。
    参数：
      配置 ：配置项对象（里面含 模型ID、模型缓存目录、是否强制重新下载）。
    返回：
      模型本地路径（字符串）。
    """
    # 第一步：检查本地缓存，有就直接用（除非用户强制重新下载）
    已缓存路径 = 查找已下载模型路径(配置)
    if 已缓存路径 is not None and not 配置.是否强制重新下载:
        print(f"检测到模型本地缓存，跳过下载，直接使用：{已缓存路径}")
        return 已缓存路径
    # 第二步：本地没有缓存（或强制重下）才联网下载
    from modelscope import snapshot_download
    if 配置.是否强制重新下载:
        print(f"已开启【强制重新下载】，将从魔搭重新获取模型：{配置.模型ID}")
    else:
        print(f"本地没有模型缓存，正在从魔搭下载：{配置.模型ID}（约 4.5GB，请耐心等待）")
    # snapshot_download 是魔搭官方下载函数；cache_dir 指定存到哪
    本地路径 = snapshot_download(
        model_id=配置.模型ID,
        cache_dir=str(配置.模型缓存目录),
    )
    print(f"模型已就绪，本地路径：{本地路径}")
    return 本地路径


# ---------- 二、加载模型主干（自适应方式） ----------
def 加载模型主干(配置, 本地路径: str):
    """根据配置的【加载方式】加载模型主干部分（只含权重，不含分词器）。

    说明：Qwen3.5 官方推荐用 AutoModelForImageTextToText（多模态类）加载；
    但部分用户可能拿到的是纯文本版或希望用文本方式加载，因此做多种方式：
      - 配置.加载方式 == "纯文本" ：只试 AutoModelForCausalLM（纯文本解码器）
      - 配置.加载方式 == "多模态"：只试 AutoModelForImageTextToText（多模态）
      - 配置.加载方式 == "自动"  ：先试文本，失败再试多模态（最省心，默认）
    如果所有方式都失败（最常见原因：transformers 版本太老、不认识 qwen3_5
    架构），会抛出一个带【安装指引】的明确报错，而不是让用户看到一堆堆栈。
    参数：
      配置     ：配置项对象。
      本地路径 ：模型在本地硬盘上的目录路径。
    返回：
      一个 transformers 模型对象（放在 CPU 上，还没搬到显卡）。
    """
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    def 尝试加载(加载器, 方式名: str):
        """用指定加载器尝试加载一次；失败打印原因并返回 (None, 原因)（不抛异常）。"""
        try:
            print(f"  正在用【{方式名}】方式加载模型 ...")
            return 加载器.from_pretrained(本地路径, trust_remote_code=True), None
        except Exception as 异常:
            print(f"    【{方式名}】方式加载失败：{异常}")
            return None, str(异常)

    # 根据配置分流：决定要尝试哪几种方式、按什么顺序
    if 配置.加载方式 == "纯文本":
        尝试列表 = [(AutoModelForCausalLM, "纯文本")]
    elif 配置.加载方式 == "多模态":
        尝试列表 = [(AutoModelForImageTextToText, "多模态")]
    else:  # "自动"
        尝试列表 = [
            (AutoModelForCausalLM, "文本"),
            (AutoModelForImageTextToText, "多模态"),
        ]

    # 依次尝试，拿到第一个成功的模型就返回；同时收集失败原因，用于后面给准确报错
    失败原因汇总 = []
    for 加载器, 方式名 in 尝试列表:
        模型, 原因 = 尝试加载(加载器, 方式名)
        if 模型 is not None:
            return 模型
        if 原因:
            失败原因汇总.append(f"【{方式名}】{原因}")

    # 全部失败：根据失败原因给【针对性】报错，两种最常见原因必须区分开：
    #   ① 失败原因里出现 "PyTorch" / "Disabling PyTorch"
    #      → 是 torch 缺失或版本太旧（transformers 5.x 要求 torch>=2.5，否则直接禁用 PyTorch）。
    #        注意：这不是 transformers 版本问题！解决方法是装 GPU 版的新 torch。
    #   ② 其余情况 → 通常是 transformers 版本太老，不认识 qwen3_5 架构，需要升级 transformers。
    全部失败原因 = "；".join(失败原因汇总)
    if "PyTorch" in 全部失败原因 or "Disabling PyTorch" in 全部失败原因:
        raise RuntimeError(
            "模型加载失败：当前 Python 环境里【PyTorch(torch) 不可用】。\n"
            "transformers 5.x 检测到 torch 缺失或版本太旧（要求 torch>=2.5），自动禁用了 PyTorch，"
            "所以模型加载失败——这【不是】 transformers 版本问题。\n"
            "解决办法（在 GPU 实例上）：\n"
            "  第 1 步：pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124\n"
            "  第 2 步：重启内核/运行时后再重新运行本程序。\n"
            "检查是否装对：python -c \"import torch; print(torch.__version__, torch.version.cuda)\"\n"
            "应输出类似 2.6.0+cu124 的版本号，且第二段不是 None（None 说明是 CPU 版）。"
        )
    # 其它原因：通常是 transformers 版本太老，不认识 qwen3_5 架构
    raise RuntimeError(
        "模型加载失败（所有方式都不行）。最常见原因是 transformers 版本太老，"
        "不认识 qwen3_5 架构（Qwen3.5 要求 transformers>=5.3.0）。\n"
        "请依次尝试：\n"
        "  第 1 步：pip install \"transformers>=5.3.0\"\n"
        "  第 2 步（还不行时）：pip install git+https://github.com/huggingface/transformers.git\n"
        "装完后再重新运行本程序。"
    )


# ---------- 三、设置权重精度 ----------
def 设置模型精度(模型, 数据类型: str):
    """把模型权重统一转成配置指定的精度。

    为什么要转换：模型原始文件可能是 fp16 或 bf16，统一转成一种精度后，
    后续计算差值、保存补丁时的数值口径才一致。
    参数：
      模型     ：加载好的模型。
      数据类型 ："bfloat16" 或 "float16"。
    """
    精度映射 = {
        "bfloat16": torch.bfloat16,   # bf16：数值范围大，训练稳，现代显卡都支持
        "float16": torch.float16,     # fp16：老显卡兼容，但数值范围小，容易溢出
    }
    if 数据类型 not in 精度映射:
        raise ValueError(f"不支持的数据类型：{数据类型}，只能是 bfloat16 或 float16")
    # 把模型所有参数搬到目标精度
    return 模型.to(精度映射[数据类型])


# ---------- 四、加载分词器 ----------
def 加载分词器(本地路径: str):
    """加载分词器，并确保它有"补位符"（padding 需要用到）。

    说明：Qwen3.5 是多模态模型，个别环境直接加载 AutoTokenizer 可能失败，
    这时会自动改用 AutoProcessor，并从处理器里取出它自带的分词器。
    参数：本地路径 模型目录。
    返回：分词器对象。
    """
    from transformers import AutoTokenizer
    try:
        # 常规方式：直接加载分词器
        分词器 = AutoTokenizer.from_pretrained(本地路径, trust_remote_code=True)
    except Exception as 异常:
        # 备用方式：加载多模态处理器，再取它内部的 tokenizer
        print(f"AutoTokenizer 加载失败（{异常}），正在改用 AutoProcessor ...")
        from transformers import AutoProcessor
        处理器 = AutoProcessor.from_pretrained(本地路径, trust_remote_code=True)
        分词器 = 处理器.tokenizer
    # 如果模型没定义补位符（Qwen 系列常见），用结束符顶上（业界标准做法）
    if 分词器.pad_token is None:
        分词器.pad_token = 分词器.eos_token
    return 分词器


# ---------- 五、总入口 ----------
def 加载模型和分词器(配置):
    """一次搞定：下载模型 → 加载主干 → 设精度 → 加载分词器 → 搬到显卡。

    参数：
      配置 ：配置项对象。
    返回：
      (模型, 分词器)
      模型默认处于【评估模式】(model.eval())，需要训练时上层会再切回训练模式。
    """
    # 第一步：拿到模型本地路径
    本地路径 = 下载模型(配置)
    # 第二步：加载模型主干（可能自动切换文本/多模态方式）
    模型 = 加载模型主干(配置, 本地路径)
    # 第三步：统一精度
    模型 = 设置模型精度(模型, 配置.数据类型)
    # 第四步：加载分词器
    分词器 = 加载分词器(本地路径)
    # 第五步：搬到指定设备（显卡或 CPU）
    设备 = torch.device(配置.设备)
    模型 = 模型.to(设备)
    # 默认切到评估模式（推理/算 Fisher 用）
    模型.eval()
    print(f"模型加载完成，参数约 {sum(p.numel() for p in 模型.parameters()) / 1e9:.2f} B，设备：{配置.设备}")
    return 模型, 分词器
