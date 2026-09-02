# -*- coding: utf-8 -*-
# ============================================================
# 文件：模型工具.py
# 作用：一组和模型打交道的"小工具函数"，供其它模块复用：
#       - 收集命名参数、统计参数数量
#       - 把模型状态保存到硬盘 / 从硬盘读回
#       - 重置模型为基座状态 / 统计显存占用
#       - 把数据集包成 DataLoader（批量取数据）
#
# 需要的条件：
#   - 依赖 torch。
# ============================================================

import torch                          # 深度学习框架
import gc                             # Python 垃圾回收（释放内存用）
from pathlib import Path              # 路径工具（用来自动创建保存目录）


# ---------- 一、参数收集与统计 ----------
def 收集命名参数(模型) -> dict:
    """把模型的全部可训练参数收集成一个字典：{参数名: 参数张量}。

    说明：后面算差值、应用补丁都要按名字定位参数，这个字典是"索引表"。
    """
    return dict(模型.named_parameters())


def 统计参数总数(模型) -> int:
    """统计模型一共有多少个参数（所有参数的元素个数之和）。"""
    return sum(参数.numel() for 参数 in 模型.parameters())


# ---------- 二、模型状态存取 ----------
def 保存状态到磁盘(模型, 保存路径) -> str:
    """把模型当前的全部参数（权重）保存成文件，之后可以完整还原。

    保存内容：{参数名: 张量}，并把张量搬到 CPU、转成 fp16 再存，
    这样文件更小、不占显存。注意：这是"完整权重"，文件可能好几个 GB。
    参数：
      模型     ：要保存的模型。
      保存路径 ：存到哪个文件（例如 结果/全量微调_状态.pt）。
    返回：
      保存路径（字符串）。
    """
    保存路径 = str(保存路径)
    # 【重要】自动创建目标文件所在的目录（例如 结果/ 目录）。
    # 因为 结果/ 目录被 .gitignore 忽略，仓库刚 clone 下来时它还不存在，
    # 如果不先建目录，torch.save 会报"目录不存在"（FileNotFoundError）。
    Path(保存路径).parent.mkdir(parents=True, exist_ok=True)
    # 收集所有参数，统一搬到 CPU 并转 fp16 深拷贝
    状态字典 = {名字: 参数.detach().cpu().half().clone() for 名字, 参数 in 模型.named_parameters()}
    torch.save(状态字典, 保存路径)
    print(f"已保存模型状态到：{保存路径}")
    return 保存路径


def 从磁盘读状态(读取路径) -> dict:
    """从文件读回之前保存的模型状态字典（{参数名: 张量}）。"""
    读取路径 = str(读取路径)
    return torch.load(读取路径, map_location="cpu", weights_only=True)


def 把状态载入模型(模型, 状态字典: dict):
    """把一组状态字典整体载入模型（用于把模型重置成基座状态）。

    参数：
      模型     ：目标模型（会被原地覆盖权重）。
      状态字典 ：从磁盘读回的 {参数名: 张量}。
    """
    # strict=False：允许状态字典和模型不完全一致（有的模型自带 buffer 参数）
    模型.load_state_dict(状态字典, strict=False)


# ---------- 三、显存与释放 ----------
def 释放模型(模型):
    """彻底释放模型占用的显存/内存，为加载下一个模型腾地方。

    做法：先删除模型引用，再强制垃圾回收，最后清空显卡缓存。
    """
    del 模型
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def 当前显存占用MB() -> float:
    """返回当前进程占用显卡显存多少 MB（方便观察内存够不够）。"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


# ---------- 四、数据批量加载 ----------
def 构建数据加载器(配置, 数据集, 是否打乱: bool):
    """把数据集包成 PyTorch 的 DataLoader（支持按批取数据、可打乱）。

    参数：
      配置     ：配置项对象（含 批次大小）。
      数据集   ：分类数据集对象（见 数据/合成数据.py）。
      是否打乱 ：训练集传 True（打乱顺序），测试/校准集传 False。
    返回：
      一个 DataLoader 对象，for 循环取出来的就是一批样本。
    """
    from torch.utils.data import DataLoader
    return DataLoader(
        数据集,
        batch_size=配置.批次大小,   # 每批多少条
        shuffle=是否打乱,           # 是否打乱
    )


# ---------- 五、类别 token 编号 ----------
def 拿类别token编号(分词器, 类别到编号: dict, 设备) -> torch.Tensor:
    """把类别词（甲/乙/丙）转成它们在词表里的 token 编号，并搬到目标设备。

    返回：形状为 (类别数,) 的整数张量，例如 [15741, 180, 211]。
    """
    # 按"类别名 → 编号"的顺序，把类别名逐个切分成 token id
    编号列表 = [分词器(类别词, add_special_tokens=False)["input_ids"][0] for 类别词 in 类别到编号]
    return torch.tensor(编号列表, dtype=torch.long, device=设备)
