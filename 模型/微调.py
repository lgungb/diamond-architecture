# -*- coding: utf-8 -*-
# ============================================================
# 文件：微调.py
# 作用：实现两种微调方式的训练逻辑：
#       1. 【全参数微调】—— 模型的所有参数都参与更新，对应白皮书"阶段一"，得到专家模型 B。
#       2. 【LoRA 微调】  —— 只在少量旁路参数上更新（业界最常用的轻量微调），作为对照组。
#       两者的训练核心是一个"分类训练器"：喂一批句子，让模型学会输出正确类别。
#
# 需要的条件：
#   - 需要 torch。
#   - 跑 LoRA 需要安装 peft 库（见 requirements.txt）。
# ============================================================

# ---------- 导入标准库 ----------
import torch                          # 深度学习框架


# ---------- 一、分类训练器 ----------
class 分类训练器:
    """负责"喂数据、算损失、更新参数"的通用训练器。

    它不知道模型内部长什么样，只要模型能接受 input_ids/attention_mask
    并返回 logits 就行。所以同一个训练器既能训全参数模型，也能训 LoRA 模型。

    损失函数说明：只取句子【最后一个位置】在【三个类别词】上的得分做交叉熵。
    因为我们的提示模板以"类别是："结尾，模型要预测的下一个词就是类别本身。
    """

    def __init__(self, 配置, 模型, 类别token编号: torch.Tensor):
        """初始化训练器。

        参数：
          配置           ：配置项对象。
          模型           ：要训练的模型（全参模型 或 包了 LoRA 的模型）。
          类别token编号  ：三个类别词在词表中的编号，形状 (3,)。
        """
        self.配置 = 配置
        self.模型 = 模型
        # 注意：模型对象不一定有 .device 属性，取"第一个参数的设备"最保险
        self.类别token编号 = 类别token编号.to(next(模型.parameters()).device)
        # 用 AdamW 优化器（业界主流，稳定）。
        # 只对"需要梯度"的参数生效（LoRA 模式下基座权重被冻结，就不用白占优化器）
        self.优化器 = torch.optim.AdamW(
            (参数 for 参数 in 模型.parameters() if 参数.requires_grad),
            lr=配置.学习率,          # 学习率
            weight_decay=0.01,       # 权重衰减（轻微防止过拟合）
        )
        # 交叉熵损失（分类任务的标准损失）
        self.损失函数 = torch.nn.CrossEntropyLoss()

    def 训练(self, 训练数据加载器) -> torch.nn.Module:
        """跑完整训练流程，返回训练好的模型。

        参数：训练数据加载器 是 DataLoader（每次取出一批样本）。
        返回：训练好的模型（原地训练，直接返回同一个对象）。
        """
        # 切到训练模式（开启 dropout、允许梯度更新）
        self.模型.train()
        # 开启梯度检查点：用一点计算量换大量显存，2B 全参微调强烈建议
        if self.配置.是否开启梯度检查点:
            try:
                self.模型.gradient_checkpointing_enable()
            except Exception:
                print("提示：当前模型不支持梯度检查点，已跳过（不影响运行）。")
        # 固定随机种子，保证可复现
        torch.manual_seed(self.配置.随机种子)

        训练步数 = 0                    # 总共看了几批数据
        已完成全部轮数 = False          # 是否已经达到配置的训练轮数
        # 外层循环：训练轮数
        for 轮次 in range(self.配置.训练轮数):
            if 已完成全部轮数:
                break
            本轮损失累加 = 0.0          # 用来打印平均损失
            批次数 = 0                 # 本轮看了几批
            # 内层循环：遍历每一批样本
            for 批次 in 训练数据加载器:
                # 1) 前向 + 反向，算出一份（已被累积步数缩小的）损失
                单批损失 = self._计算单批损失(批次)
                # 2) 反向传播（损失已经除以累积步数，所以多个小步加起来等于一大步）
                单批损失.backward()
                本轮损失累加 += 单批损失.item()
                批次数 += 1
                训练步数 += 1
                # 3) 攒够"梯度累积步数"才真正更新一次参数
                if 训练步数 % self.配置.梯度累积步数 == 0:
                    self.优化器.step()          # 更新参数
                    self.优化器.zero_grad()     # 清空梯度
                    # 打印进度（让用户看到在训练）
                    if self.配置.是否显示详细日志:
                        print(
                            f"  轮次 {轮次 + 1}/{self.配置.训练轮数}，"
                            f"步数 {训练步数}，平均损失 {本轮损失累加 / 批次数:.4f}"
                        )
                # 4) 如果设了最大训练步数，到点就提前停（冒烟测试用）
                if self.配置.最大训练步数 > 0 and 训练步数 >= self.配置.最大训练步数:
                    已完成全部轮数 = True
                    break
        # 训练结束，切回评估模式
        self.模型.eval()
        return self.模型

    def _计算单批损失(self, 批次: dict) -> torch.Tensor:
        """对一批样本做一次前向，算出损失（这一步是核心计算）。

        参数：批次 是字典，含 输入编号/注意力掩码/标签编号。
        返回：损失张量（标量）。
        """
        设备 = next(self.模型.parameters()).device
        # 把批次的三个部分搬到模型所在设备
        输入编号 = 批次["输入编号"].to(设备)
        注意力掩码 = 批次["注意力掩码"].to(设备)
        标签编号 = 批次["标签编号"].to(设备)
        # 混合精度前向：bf16/fp16 既省显存又快（仅在显卡上启用，CPU 上直接算）
        训练精度 = torch.bfloat16 if self.配置.训练精度 == "bfloat16" else torch.float16

        def 算前向和损失():
            """内部小函数：执行一次前向并算出交叉熵损失。"""
            输出 = self.模型(input_ids=输入编号, attention_mask=注意力掩码)
            # 兼容不同模型的返回格式（有的返回对象带 .logits，有的返回元组）
            logits = 输出.logits if hasattr(输出, "logits") else 输出[0]
            # 只取最后一个位置的、在三个类别词上的得分，形状 (批次, 3)
            类别得分 = logits[:, -1, self.类别token编号]
            # 交叉熵：希望正确类别的得分最高
            return self.损失函数(类别得分, 标签编号)

        if 设备.type == "cuda":
            # 显卡上：用混合精度省显存
            with torch.autocast(device_type="cuda", dtype=训练精度):
                损失 = 算前向和损失()
        else:
            # CPU 上：不用混合精度，直接算
            损失 = 算前向和损失()
        # 除以梯度累积步数，让"多次小步"合成"一步大更新"
        return 损失 / self.配置.梯度累积步数


# ---------- 二、LoRA 辅助工具 ----------
def 探测线性层模块名(模型) -> list:
    """自动找出模型里适合挂 LoRA 的线性层（如 q_proj、v_proj）。

    说明：不同模型命名不同，写死名字容易报错，所以运行时动态探测：
    凡是名字以 _proj 结尾、且是 torch.nn.Linear 的，都算候选。
    """
    候选集合 = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    找到的 = set()
    for 名字, 模块 in 模型.named_modules():
        末段 = 名字.split(".")[-1]
        if 末段 in 候选集合 and isinstance(模块, torch.nn.Linear):
            找到的.add(末段)
    return sorted(找到的)


def 训练LoRA模型(配置, 模型, 类别token编号: torch.Tensor, 训练数据加载器, 秩: int):
    """用 LoRA 方式微调模型，返回训练好的 LoRA 模型和适配器参数量。

    参数：
      配置           ：配置项对象。
      模型           ：基座模型（会被包上一层 LoRA 旁路）。
      类别token编号  ：类别词 token 编号。
      训练数据加载器  ：训练 DataLoader。
      秩             ：LoRA 的秩 r（16 或 64，越大能力越强、文件越大）。
    返回：
      (训练好的LoRA模型, 可训练参数量)
    """
    # 延迟导入 peft（本地没装时不影响其它模块）
    from peft import LoraConfig, get_peft_model, TaskType

    # 1) 探测该模型有哪些线性层可以挂 LoRA
    目标模块 = 探测线性层模块名(模型)
    if not 目标模块:
        # 找不到任何候选层：给出明确提示（理论上 Qwen 都会有）
        raise ValueError("在模型中找不到可挂 LoRA 的线性层（q_proj/v_proj 等），请检查模型结构")
    print(f"  LoRA 将作用在这些层上：{目标模块}")

    # 2) 配置 LoRA 参数：秩 r、缩放系数 alpha（通常取 2 倍 r）、丢弃率
    lora配置 = LoraConfig(
        task_type=TaskType.CAUSAL_LM,   # 因果语言模型任务
        r=秩,                           # 秩
        lora_alpha=秩 * 2,              # 缩放系数
        target_modules=目标模块,        # 作用在哪几类层上
        lora_dropout=0.05,              # 防过拟合
        bias="none",                    # 不动偏置
    )
    # 3) 把基座模型包成 LoRA 模型（基座权重冻结，只训旁路）
    lora模型 = get_peft_model(模型, lora配置)
    可训练参数 = sum(参数.numel() for 参数 in lora模型.parameters() if 参数.requires_grad)
    print(f"  LoRA 秩 r={秩}，可训练参数 {可训练参数:,}（约占全模型 {可训练参数 / sum(p.numel() for p in 模型.parameters()):.2%}）")

    # 4) 用同一个分类训练器训练（只更新 LoRA 旁路）
    训练器 = 分类训练器(配置, lora模型, 类别token编号)
    lora模型 = 训练器.训练(训练数据加载器)
    return lora模型, 可训练参数
