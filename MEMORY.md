# 📌 MEMORY.md —— 钻石架构验证项目 · 持久记忆

> **本文件是项目的"长期记忆"**，专门给 AI 和人类看。
> 任何 AI 接手这个项目（无论是继续开发、修 bug、改需求），**请先完整读一遍本文件**，
> 再动手。这样即使上下文被清空，也能快速恢复对项目的完整认识。
> 每次重大变更后，记得更新本文件（尤其"文件目录""每个文件作用""架构流程"三节）。

---

## 0. 一句话定位

用 **Qwen3.5-2B** 在 **魔搭(modelscope) 免费 Notebook** 上，验证"钻石架构"白皮书的核心主张：
**全参数微调后只保存 Top 2%~6% 关键参数的差值（硬核补丁），推理时盖回常驻基座**，能否逼近全量微调效果。

- 模型：`Qwen/Qwen3.5-2B`（魔搭，约 2.27B 参数 / 4.57GB，**多模态**模型，架构 `qwen3_5`）
- 运行环境：modelscope Notebook（**必须 GPU 实例**，如 A10 24G；CPU 实例跑不动 2B 全参微调）
- 任务：自造的"三分类密语任务"（离线、可控、干净）
- 用户身份：非编程专家，代码要"能跑 + 能看懂"，本地只存改不运行

---

## 1. 目录结构（当前版本）

```
diamond-architecture/
├── README.md               # 人读的说明（快速开始/常见问题）
├── MEMORY.md               # ★本文件：AI/人 记忆
├── requirements.txt        # 依赖（transformers>=5.3.0、torch>=2.5.0 是硬约束！）
├── .gitignore              # 忽略 缓存/ 结果/ __pycache__/
│
├── 配置/
│   └── 配置.py             # ★参数总控：模型ID、数据规模、补丁清单、开关
├── 工具/
│   ├── 环境自检.py         # 体检：Python/显卡/依赖/模型缓存
│   └── 环境体检.py         # 深度诊断：解释器路径/库能否导入/pip记录/显卡
├── 数据/
│   ├── 合成数据.py         # 默认任务：三分类密语数据生成器
│   ├── 真实数据.py         # 可选：从魔搭下载真实分类数据（半成品，需填字段）
│   └── 数据接口.py         # 统一入口：构建数据 + 校验类别词单token
├── 模型/
│   ├── 模型加载.py         # 下载+自适应加载（自动兼容多模态）
│   ├── 模型工具.py         # 参数收集/状态存取/释放/DataLoader
│   ├── 微调.py             # 全参数微调训练器 + LoRA
│   └── 计算重要度.py       # Fisher 信息 → 参数重要性表
├── 补丁/
│   ├── 补丁格式.py         # 补丁数据结构 + 保存/读取
│   ├── 生成补丁.py         # 三种策略选位 + 计算 Δ 值 → 补丁文件
│   └── 应用补丁.py         # 盖回基座 + 切换耗时计时
├── 评估/
│   └── 评估.py             # 分类准确率
├── 实验/
│   ├── 实验总控.py         # ★总指挥：10 步流程
│   └── 汇总报告.py         # 生成 结果/报告.md
├── 运行/
│   └── 主入口.py           # ★唯一入口：python 运行/主入口.py
└── 结果/                   # 运行时生成：补丁/模型状态/报告（git 忽略）
```

---

## 2. 每个文件的作用与依赖条件（速查）

| 文件 | 干什么 | 依赖/条件 |
| --- | --- | --- |
| `运行/主入口.py` | 唯一启动点；自检→硬性检查(torch版本+GPU)→跑实验 | 无 |
| `配置/配置.py` | 全部参数；用户只改这里（含 优化器=adafactor 默认、梯度检查点默认关闭） | 无第三方库 |
| `工具/环境自检.py` | 检查 Python/库/显卡/模型缓存 | torch(可缺)、modelscope |
| `工具/环境体检.py` | 深度诊断（解释器/库导入/torch GPU状态/pip记录/显卡），出问题先跑它 | 仅标准库 |
| `数据/合成数据.py` | 生成训练/测试/校准三套密语数据 | 需要分词器、torch |
| `数据/真实数据.py` | 可选：魔搭真实数据 | datasets；需用户填字段 |
| `数据/数据接口.py` | 数据统一入口 + 类别词校验 | 无 |
| `模型/模型加载.py` | **先查本地缓存**（查找已下载模型路径）→没有才 snapshot_download；自适应加载（文本失败→多模态，全部失败给安装指引） | modelscope、transformers>=5.3 |
| `模型/模型工具.py` | 参数/状态/显存/加载器工具；保存前**自动创建目标目录**（结果/ 等） | torch |
| `模型/微调.py` | 分类训练器（全参/LoRA 通用）+ LoRA 包装；**优化器用 inspect 自动兼容不同 torch 版本**；**训练进度详细打印（每10batch+ROCm编译提示）** | torch、peft(LoRA) |
| `模型/计算重要度.py` | Fisher 信息（梯度平方累加） | torch |
| `补丁/补丁格式.py` | 补丁 = 元数据 + {参数名:{索引,差值}} | torch |
| `补丁/生成补丁.py` | 策略A/B/C 选位（两阶段全局topk）+ Δ 计算 | torch |
| `补丁/应用补丁.py` | 重置基座 / 盖补丁（新值=现值+Δ）/ 计时 | torch |
| `评估/评估.py` | 分类准确率 | torch |
| `实验/实验总控.py` | 10 步串流程、显存调度 | 全项目 |
| `实验/汇总报告.py` | 出表格 + 自动结论判定 | 无 |

---

## 3. 架构与运行流程（重要，务必理解）

```
┌─ 步骤 1: 加载基座 A + 分词器（Qwen/Qwen3.5-2B，bf16）
├─ 步骤 2: 构建数据（合成密语任务: 甲/乙/丙 三类，密语词是生造的）
├─ 步骤 3: 评估基座 A（预期≈33% 随机水平）
├─ 步骤 4: 在 A 上算 Fisher 重要度（校准集 60 条，梯度平方累加）→ 搬到 CPU 释放显存
├─ 步骤 5: 保存 A 完整权重到 结果/基座A状态.pt
├─ 步骤 6: 全参数微调 A→B（Adafactor, lr=3e-5, bf16, 梯度检查点）→ 评估（预期≈99%）→ 保存B状态
├─ 步骤 7: 生成补丁（策略A:2/4/6%、策略B:2%、策略C:6%）——只存被选位置的 Δ=B−A
├─ 步骤 8: 重新加载 A；对每个补丁: 重置A → 盖补丁 → 评估 + 计时切换
├─ 步骤 9: LoRA 对照组（r=16/64，训练→合并→评估）
└─ 步骤 10: 生成 结果/报告.md（恢复率 = (补丁−基座)/(全量−基座)）
```

**核心机制（对齐白皮书）**：
- 补丁里存的是**差值 Δ**，应用时 `新值 = 当前值 + Δ`（应用前模型是基座 A，结果等价于"物理替换为 B 值"）。
- 评估补丁采用"**一份模型 + 反复重置**"：加载一次 A，每个补丁前 `load_state_dict(基座状态)` 还原，再覆盖，避免反复加载大模型。

**三种策略（实现口径）**：
- 策略A：Fisher 重要度**全局 Top K**（两阶段 topk 省内存：每张量先取 10 倍候选 → 合并全局精筛）
- 策略B：Fisher 重要度**全局 Bottom K**
- 策略C：**每张量内部**取 高2% + 低2% + 中2%（"中"= 排序后正中间一段；注意 A/B 是全局、C 是局部，口径略不同，注释已说明）

---

## 4. 关键设计决策（为什么这样做）

1. **用"合成密语任务"而不是真实数据**：密语词是生造的，基座必然不认识 → 基座≈随机、微调≈满分，验证"补丁恢复多少提升"最干净；且离线可用，避免下载不稳定。
2. **只算类别 token 上的交叉熵**（不是整句 LM loss）：任务本质是分类，损失聚焦后 Fisher 打分更精准、微调更快。
3. **Fisher 在基座 A 上算**（对齐 FISH-Mask 标准做法/白皮书意图），配置里留了"专家"选项未实现。
4. **补丁评估 = 加载一次 + 反复 load_state_dict 重置**：省时省显存；显存调度顺序：Fisher 重要度(bf16 ≈4.5GB) 算完立刻搬 CPU → 再全参微调 → 微调完拿完专家参数就释放 → 再加载 A 评估补丁。
5. **加载自适应**：Qwen3.5 是多模态模型，`AutoModelForCausalLM` 失败会自动切 `AutoModelForImageTextToText`；配置 `加载方式="自动"`。
6. **transformers>=5.3.0 是硬约束**：qwen3_5 架构只有 5.3.0+ 才认识（曾误以为 5.2.0 可行导致用户报错，已修正；5.2.0 及以下报 "model type qwen3_5 无法识别"）。requirements 里已锁 >=5.3.0。
7. **torch>=2.5.0 且 GPU 版 是硬约束**：transformers 5.x 要求 torch>=2.5（否则禁用 PyTorch）；且 2B 全参微调必须 GPU（CPU 实例不可行）。主入口做三段硬检：能 import→版本≥2.5→有 CUDA。
8. **优化器默认 adafactor**：AdamW 对 2B 全参微调需要约 27GB fp32 优化器状态，24G 卡必爆（用户已实测 OOM，词嵌入一个状态张量就 1.89GB）；Adafactor 状态占用极小，效果对这个分类任务足够。配置 优化器 可切回 adamw。
9. **Fisher 重要度表用 bf16 累加**（比 fp32 省一半显存）并开启梯度检查点：老写法 `grad.float()**2` 每步生成 fp32 大临时张量（词嵌入 2GB），是 Fisher 阶段 OOM 主因。
10. **PYTORCH_ALLOC_CONF=expandable_segments:True**：主入口在 import torch 前设置，减少显存碎片。
11. **冒烟模式**：一键缩小规模，先验证流程，再跑正式实验。
12. **模型加载前先查本地缓存**：`模型/模型加载.py` 的 `查找已下载模型路径` 会按 modelscope 缓存规则（`缓存/模型/models/<ID中/换-->>/snapshots/<版本>/`，有 config.json/configuration.json 即完整）检查是否已下载；已有直接复用，没有才联网下载。配置 `是否强制重新下载=True` 可强制重下。
13. **所有写文件前自动建目录**：`结果/`、`结果/补丁/` 被 .gitignore 忽略、clone 下来不存在；`保存状态到磁盘` 等都在写文件前 `mkdir(parents=True, exist_ok=True)`（补丁格式/汇总报告本来就有，v0.1.5 给 保存状态到磁盘 补上，否则首次运行报 FileNotFoundError）。
14. **优化器参数用 inspect 自动兼容不同 torch 版本**：torch 2.11 开发版（ROCm 镜像预装）的 Adafactor 移除了 `scale_parameter` 等参数，写死传参会报 TypeError。`模型/微调.py` 的 `创建兼容优化器` 函数用 `inspect.signature` 检测当前版本支持哪些参数，只传支持的，不支持的自动忽略并打印提示。Adafactor 和 AdamW 都走这个兼容层。
15. **大显存默认关闭梯度检查点**：192G 显存跑 2B 模型完全不需要梯度检查点（它是用计算换显存的小显存技巧），开启反而更慢且可能与多模态模型前向有兼容性问题导致挂起。v0.1.9 起 `是否开启梯度检查点` 默认 False，24G 以下小显存才需要手动改 True。

---

## 5. 实验要回答的问题（验证目标）

| 白皮书论断 | 我们怎么验证 | 成功标准 |
| --- | --- | --- |
| Top 2~6% 参数替换逼近全量微调 | 策略A 补丁恢复率 | 恢复率≥90%（本任务上） |
| 策略B 只改风格不改能力 | 策略B 补丁准确率 | 准确率≈基座（差距<5%） |
| 无低秩瓶颈，优于 LoRA | 补丁 vs LoRA 准确率+体积 | 同体积下补丁更准 |
| 毫秒级切换 | 显存内覆盖计时 | 报告实测毫秒数 |
| 补丁体积小 | 文件大小统计 | 与 LoRA 适配器对比 |

---

## 6. 已知风险 / 待用户反馈确认的点

1. **Qwen3.5-2B 是多模态模型**，用纯文本方式喂文本应可行。加载已做多重兜底：AutoModelForCausalLM → AutoModelForImageTextToText → 全部失败时抛带安装指引的报错（提示升级 transformers>=5.3.0）。分词器同样有 AutoTokenizer → AutoProcessor 兜底。
2. **全参数微调 2B 模型显存压力**：默认 bf16 + 梯度检查点 + Adafactor 优化器 + batch=4 累积4（约 10~11GB，24G 卡够）；若 OOM，把 batch 降到 1~2。
3. **Fisher 全模型 bf16 buffer ≈ 4.5GB**（比早期 fp32 8GB 省一半）：加上模型本身约需 12GB 显存；A10 24G 够。
4. **LoRA 的 target_modules 是运行时探测的**（q_proj 等），若 Qwen3.5 命名不同会报错，用户反馈后调整。
5. **策略C 的"中段"是简化实现**（每张量局部而非全局），量级正确但口径与 A/B 不同。
6. 补丁评估/切换耗时是**显存内覆盖**，不含磁盘加载补丁的时间；报告里已注明。
7. **modelscope 镜像预装 torch 可能是 2.3.1（旧版/CPU 版）**：如果用户 `pip install -r requirements.txt` 只升级了 transformers 到 5.x 而 torch 仍是旧版，transformers 5.x 会"Disabling PyTorch"→ 模型加载报"PyTorch was not found"。这【不是】 transformers 版本问题，必须 `pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124` 装 GPU 版 torch>=2.5 并重启内核。模型加载.py 的报错已能区分"PyTorch 问题"和"transformers 版本问题"。
8. **魔搭 notebook 的终端 `python` 与 jupyter 内核可能是两套环境**：在终端跑 主入口.py 前先跑 `python 工具/环境体检.py` 确认 torch 状态；torch 版本不对时主入口会提前拦截给指引（v0.1.3+）。
9. **AMD GPU（ROCm）环境也能跑**：魔搭有"AMD GPU 环境"（如 8核/200GB/192G 显存，镜像 ubuntu22.04-rocm7.2.3-py312-torch2.11.0）。PyTorch ROCm 版把 AMD GPU 映射成 "cuda" 设备，`torch.cuda.is_available()` / `.to("cuda")` / `torch.cuda.memory_allocated()` 全部兼容，**代码无需修改**。唯一区别：没有 `nvidia-smi`，改用 `rocm-smi`；`torch.version.cuda` 为 None 但 `torch.version.hip` 有值。环境体检.py v0.1.7+ 已支持自动检测 ROCm。192G 显存非常充裕，不会 OOM。
10. **torch 2.11 开发版的 API 可能与稳定版不同**：ROCm 镜像预装的 torch 2.11.0+git 是开发版，Adafactor 等优化器的参数签名可能移除了旧参数（如 scale_parameter）。已用 `创建兼容优化器`（inspect 签名检测）解决。后续若遇到其他 API 不兼容，按同样思路处理。
11. **ROCm 首次运行大模型可能触发 MIOpen kernel 编译，耗时 5-15 分钟**：这是 AMD GPU 的正常现象，不是卡死。v0.1.9 起训练开始前会打印提示，每 10 个 batch 打印一次进度，方便判断是否在正常运行。如果 15 分钟后连第一个 batch 都没过去，再排查。

---

## 7. 协作流程（用户约定的工作方式）

1. **我（AI）**：在本地改代码 → 语法校验（本地不跑模型）→ git 提交 → 推送到 GitHub。
2. **用户**：把仓库拉到 modelscope Notebook → 自行运行 → 把报错/结果反馈给我。
3. **我**：根据反馈修 bug → 本地再提交 → 用户再拉取测试。
4. 本地只存改、不运行实验；实验只在用户侧的 modelscope 跑。

---

## 8. 项目进度日志（重要！每次变更在此追加）

- **[2026-09-02] v0.1 初版完成**
  - 搭建完整 10 步实验框架，全部代码语法校验通过（compileall）。
  - 已修的关键坑：① 中文句子里误用 ASCII 引号导致的 SyntaxError；② `模型.device` 属性不存在 → 改用 `next(模型.parameters()).device`；③ 补丁差值计算跨设备索引 → 先搬到专家张量设备再取回 CPU 减；④ CPU 无显卡时跳过 autocast；⑤ AdamW 只对 requires_grad 参数生效（适配 LoRA 冻结）。
  - 已知待用户反馈：模型加载方式、显存、LoRA 模块名是否 OK。
  - 下一步：用户跑冒烟测试 → 按反馈修 → 补正式实验报告解读。

- **[2026-09-02] v0.1.1 修复：Qwen3.5 加载报错（qwen3_5 架构无法识别）**
  - **根因**：transformers 版本。Qwen3.5 的 qwen3_5 架构要求 transformers>=5.3.0；v0.1 误锁 `==5.2.0`，导致 `AutoConfig` 报 `KeyError: 'qwen3_5'`。
  - **修复**：requirements.txt 改为 `transformers>=5.3.0`（并补 qwen-vl-utils>=0.0.14 备用）；环境自检最低版本改 5.3.0；README/MEMORY 同步修正。
  - **加载逻辑增强**：`模型/模型加载.py` 的 加载模型主干 改为"依次尝试文本→多模态→全部失败给明确安装指引"；加载分词器 增加 AutoProcessor 兜底。
  - **用户操作**：在 notebook 执行 `pip install "transformers>=5.3.0"`（或重装 requirements）后重跑。
  - **待观察**：Qwen3.5 多模态模型纯文本前向是否正常、LoRA 目标模块探测是否命中（Qwen3.5 是 Gated DeltaNet 混合架构，层名可能与 q_proj 等不同，报错再调）。

- **[2026-09-02] v0.1.2 修复：运行环境缺 PyTorch**
  - **用户报错**：transformers 已能加载（qwen3_5 错误消失），但报 "AutoModelForCausalLM requires the PyTorch library but it was not found in your environment"。
  - **根因**：当前 `python` 解释器环境里没有 torch（很可能 notebook 内核与终端 python 是两套环境，或 torch 未装/未重启内核）。
  - **修复**：`运行/主入口.py` 增加 torch 硬性前置检查——缺 torch 直接停下并打印安装指引（含 `pip install torch --index-url https://download.pytorch.org/whl/cu121` 与"装完重启内核"提示），不再一路崩到模型加载；`工具/环境自检.py` 的 torch 缺失提示也改为针对性安装命令。
  - **新增**：`工具/环境体检.py` —— 独立诊断脚本（只依赖标准库），一键输出：解释器路径/版本、关键库能否被当前解释器 import、pip 记录、nvidia-smi 显卡信息。用于定位"库装错环境/显卡看不到"问题。
  - **待用户反馈**：跑 环境体检.py 的输出（尤其 torch 是否 [有]、nvidia-smi 的 CUDA 版本），据此给精确安装命令。

- **[2026-09-02] v0.1.3 修复：CPU 实例 + torch 版本太旧（诊断结果确认）**
  - **用户环境体检结果（真实）**：解释器 `/usr/local/bin/python`，Python 3.11.11；`torch=2.3.1+cpu`（**CPU 版且版本旧**）；`transformers=5.16.1`；modelscope/accelerate/peft/datasets 齐全；**nvidia-smi 不存在 → 当前是 CPU 实例，无 GPU**。
  - **根因**：① transformers 5.x 要求 torch>=2.5，torch 2.3.1 太旧 → transformers 直接"Disabling PyTorch"（这就是报错第一行的来源）；② 当前 notebook 是 CPU 实例，即使 torch 装对，2B 全参微调在 CPU 上也不可行。
  - **修复**：
    - `requirements.txt`：`torch>=2.1.0` → `torch>=2.5.0`，注释写明"必须是 GPU 版"。
    - `工具/环境自检.py`：torch 最低版本改 2.5.0；版本不够时标【错误】；无 CUDA 时从"警告"升级为【错误】并提示换 GPU 实例。
    - `运行/主入口.py`：硬性检查升级为三段——①能 import torch ②版本>=2.5 ③有 GPU；任一不满足都停下并给明确指引。
    - `工具/环境体检.py`：新增【4.5】torch GPU 状态（torch.version.cuda 为 None 即 CPU 版）。
    - `README.md`：快速开始加"必须 GPU 实例"警告 + torch 常见问题两条。
  - **用户下一步（关键）**：到魔搭【新建 Notebook】选【GPU 免费实例】（A10 24G），在那里重新 clone + 装依赖 + 运行；不要在 CPU 实例上跑。

- **[2026-09-02] v0.1.4 修复：GPU 实例上爆显存（CUDA out of memory）**
  - **用户报错**：切到 24G 显卡后 `torch.OutOfMemoryError: Tried to allocate 1.89 GiB ... total capacity 22.18 GiB ... 16.42 GiB allocated by PyTorch`。
  - **根因（两处）**：① 微调用 `AdamW`，对 2B 全参微调要额外约 27GB fp32 优化器状态（词嵌入层 508M 参数，单个 fp32 状态张量就 ~2GB，正是报错那个 1.89GB）；② Fisher 用 fp32 累加缓冲（9GB）+ `grad.float()**2` 每步生成 fp32 大临时张量（词嵌入 2GB），加上模型与激活，22GB 卡装不下。
  - **修复**：
    - `配置/配置.py`：新增 `优化器` 字段，默认 `"adafactor"`（内存占用极小），可切回 `"adamw"`。
    - `模型/微调.py`：分类训练器按配置选择优化器——adafactor（scale_parameter=False/relative_step=False/warmup_init=False, lr 用配置值）或 adamw。
    - `模型/计算重要度.py`：重要度累加表改 bf16（省一半显存，卡不支持 bf16 时自动回 fp32）；累加改 `grad.detach()**2`（不再生成 fp32 大临时张量）；开启梯度检查点防激活 OOM。
    - `运行/主入口.py`：在 import torch 前设置 `PYTORCH_ALLOC_CONF=expandable_segments:True` 减少碎片。
    - `README.md`/`MEMORY.md`：同步说明（优化器选择、显存量级、FAQ）。
  - **预期显存**：全参微调约 10~11GB、Fisher 约 12GB，A10 24G 足够。
  - **待用户反馈**：重跑后下一个报错点（重点盯：多模态模型纯文本前向、LoRA target_modules 探测、正式模式的显存余量）。

- **[2026-09-02] v0.1.5 修复：首次运行报 FileNotFoundError（结果/ 目录不存在）+ 新增模型缓存判断**
  - **用户报错**：`FileNotFoundError: ... 结果/基座A状态.pt`（`保存状态到磁盘` 时，`结果/` 目录被 .gitignore 忽略、clone 后不存在，而写文件前没自动建目录）。
  - **修复**：`模型/模型工具.py` 的 `保存状态到磁盘` 在 torch.save 前 `Path(保存路径).parent.mkdir(parents=True, exist_ok=True)`（补丁格式.py/汇总报告.py 原本已有，这处是唯一遗漏）。
  - **新增（用户要求）**：模型加载前先判断本地是否有缓存。
    - `模型/模型加载.py` 新增 `查找已下载模型路径`：按 modelscope 缓存规则检查 `缓存/模型/models/<ID中/换-->>/snapshots/<版本>/`，存在 config.json/configuration.json 即视为已下载完整。
    - `下载模型` 改为：先查缓存，已有直接复用（打印"跳过下载"）；没有才 snapshot_download；`配置/配置.py` 新增 `是否强制重新下载`（默认 False）可强制重下。
  - **待用户反馈**：重跑后下一个报错点（重点盯：多模态模型纯文本前向、LoRA target_modules 探测、补丁生成/评估阶段）。

- **[2026-09-02] v0.1.6 增强：模型加载报错区分"torch 不可用"和"transformers 版本问题"**
  - **用户报错**：模型下载成功（缓存逻辑生效），但加载时报 `AutoModelForCausalLM requires the PyTorch library but it was not found`（transformers 5.x 检测到 torch 缺失/版本太旧→Disabling PyTorch）。**且用户运行的是旧版代码**（下载文案/主入口行号可证，未包含 v0.1.5 的缓存判断与 torch 硬检）。
  - **诊断**：报错直接原因是当前 python 环境 torch 不可用（modelscope 镜像预装 torch 2.3.1 旧版/CPU 版 + transformers 5.x 要求 torch>=2.5 → 禁用 PyTorch）。不是 transformers 版本问题。
  - **修复**：`模型/模型加载.py` 的 `加载模型主干` 收集每个方式的失败原因，全部失败后判断：失败含 "PyTorch"/"Disabling PyTorch" → 给"装 GPU 版 torch>=2.5 + 重启内核"的准确指引；否则才提示升级 transformers。MEMORY 风险清单补充第 7/8 条。
  - **用户操作（关键）**：① 确认在 GPU 实例；② `git pull` 拉最新 v0.1.6（旧代码会绕过 torch 硬检）；③ `python 工具/环境体检.py` 看 torch 是否 GPU 版；④ 若 torch 旧/CPU 版：`pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124` 后重启内核再跑。

- **[2026-09-02] v0.1.7 适配：支持 AMD GPU（ROCm）环境检测**
  - **用户选择**：魔搭"AMD GPU 环境"（8核/200GB/192G 显存，镜像 ubuntu22.04-rocm7.2.3-py312-torch2.11.0-1.39.0）。
  - **核心结论**：PyTorch ROCm 版把 AMD GPU 映射成 "cuda" 设备，代码**无需修改**即可运行（`torch.cuda.is_available()` / `.to("cuda")` 全部兼容）。192G 显存充裕，不会 OOM。
  - **修改**：`工具/环境体检.py` 的【4.5】增加 `torch.version.hip` 检测 + GPU 设备名/显存显示；【7】显卡信息在 nvidia-smi 不存在时自动改用 rocm-smi，并提示"可能是 AMD GPU 环境"。MEMORY 风险清单补充第 9 条。
  - **用户操作**：直接 `git pull` 后运行即可，不需要装 torch（镜像已预装 2.11.0 ROCm 版）；先跑 `python 工具/环境体检.py` 确认 GPU 状态，再跑 `python 运行/主入口.py`。

- **[2026-09-02] v0.1.8 修复：torch 2.11 开发版 Adafactor 参数不兼容（scale_parameter 被移除）**
  - **用户进展**：AMD ROCm 环境上前 5 步全部跑通！环境自检通过（torch 2.11.0 ROCm 版，191.7GB 显存）；模型加载成功（1.88B 参数，cuda 设备）；基座 A 准确率 37.5%（符合随机预期）；Fisher 计算完成；基座 A 状态保存成功。
  - **报错**：步骤 6 全参数微调时 `TypeError: Adafactor.__init__() got an unexpected keyword argument 'scale_parameter'`。
  - **根因**：torch 2.11.0 开发版（ROCm 镜像预装）的 Adafactor 参数签名变了，移除了 `scale_parameter`（可能还有 `relative_step`/`warmup_init`）。老版本 torch 这些参数都存在，写死传参在新版上报 TypeError。
  - **修复**：`模型/微调.py` 新增 `创建兼容优化器` 函数——用 `inspect.signature` 检测优化器支持哪些参数，只传当前版本支持的，不支持的自动忽略并打印提示。Adafactor 和 AdamW 都改用这个兼容函数创建。这样不管 torch 哪个版本都能跑。
  - **用户操作**：`git pull` 后直接重跑 `python 运行/主入口.py` 即可（模型已缓存，不会重新下载；前 5 步会快速重跑，重点看第 6 步微调是否开始）。

- **[2026-09-02] v0.1.9 修复：训练阶段卡住无输出（梯度检查点兼容性 + ROCm 首次 kernel 编译无提示）**
  - **用户现象**：步骤 6 全参数微调时，打印完 `[transformers] use_cache=True is incompatible with gradient checkpointing. Setting use_cache=False` 后长时间卡住，无任何输出。
  - **根因（两个）**：① 梯度检查点默认开启，但 192G 显存完全不需要（它是用计算换显存的小显存技巧），开启后训练更慢且可能与多模态模型前向有兼容性问题导致挂起；② 训练循环只在每 4 个 batch（梯度累积步）才打印一次，且没有"首次 ROCm 编译 kernel 可能很慢"的提示，用户无法判断是死了还是在跑。
  - **修复**：
    - `配置/配置.py`：`是否开启梯度检查点` 默认从 True 改为 False（192G 大显存不需要，关闭更快更稳定），注释说明 24G 以下小显存才需要开启。
    - `模型/微调.py`：训练方法增加详细进度打印——训练前打印总轮数/每轮 batch 数/总步数 + "首次 ROCm 运行前几个 batch 可能很慢（编译 kernel，5-15 分钟）"提示；每轮开始/结束打印；每 10 个 batch 打印一次进度（即使没到梯度更新步也能看到在跑）；训练完成打印总 batch 数。
  - **用户操作**：`git pull` 后重跑 `python 运行/主入口.py`。这次会在步骤 6 开始时看到"开始训练：共 3 轮，每轮 N 个 batch"+"首次 ROCm 可能 5-15 分钟编译 kernel"提示，然后每 10 个 batch 打印一次进度。如果前 15 分钟内有"已处理 10/XX 个 batch"出现，说明在正常跑；如果 15 分钟后连第一个 batch 都没过去，再反馈。
