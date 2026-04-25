# MultiSync 产品定位与技术路线 v3.0

*更新于 2026-04-24，基于四轮审查反馈（三轮大模型审查 + 科研导师 PM 反馈）后的重大架构调整*

---

## 一句话定位

**MultiSync 是一个面向人际同步性研究者的"动态过程分析与可视化环境"。**

它不处理原始信号（那是 HyPyP、MNE、OpenFace 的工作），而是接收已经提取好的同步性特征序列，回答一个现有工具都无法回答的问题：

> 同步性是如何随时间发生、如何在不同模态间级联、是否可以被早期信号预测、在不同情境下呈现什么模式？

---

## 架构核心：Separation of Concerns（算法与表现分离）

### 两个组件，两种语言

MultiSync 由两个独立组件组成，从 v1.0 开始就严格分离：

| 组件 | 语言 | 职责 | 交付物 |
|------|------|------|--------|
| **multisync-core** | Python | 所有数学计算和统计分析 | `results.json` / Parquet 结果包 |
| **MultiSync Web** | React + TypeScript | 交互式可视化审阅 | 可分享的可视化界面 |

```
┌─────────────────────────────────────────┐
│  multisync-core (Python)                 │
│                                          │
│  输入：预处理特征序列 CSV                 │
│  输出：results.json                      │
│                                          │
│  核心计算：                              │
│  ├─ Within-dyad Z-score Normalization    │
│  ├─ 10 个动态特征提取                    │
│  ├─ CCF + Bartlett 置信区间              │
│  ├─ Surrogate Testing (×100)             │
│  ├─ Rolling Origin Time-Series CV        │
│  ├─ LODO (Leave-One-Dyad-Out)            │
│  └─ 朴素基线 vs. 动态特征对比框架        │
│                                          │
│  依赖：scipy, numpy, sklearn, pandas     │
│  调用方式：Jupyter / CLI / pipeline      │
└──────────────┬──────────────────────────┘
               │ results.json
               ↓
┌─────────────────────────────────────────┐
│  MultiSync Web Viewer (React)            │
│                                          │
│  输入：results.json + 原始时间序列 CSV    │
│  输出：交互式可视化                       │
│                                          │
│  核心功能：                              │
│  ├─ Score View（情境标注轨道）           │
│  ├─ Cascade Map（级联有向图）            │
│  ├─ 动态特征时间轴可视化                 │
│  ├─ 情境切片对比图                       │
│  └─ 图表/报告导出                        │
│                                          │
│  定位：审阅器（Viewer），不是计算器      │
│  部署：GitHub Pages / Vercel（零安装）   │
└─────────────────────────────────────────┘
```

### 为什么从一开始就分离

"先在 JS 里验证算法，发 Paper 后再移植 Python"是一个致命的战略误判，原因有三：

1. **审稿人不会接受前端代码作为可复现的研究代码。** 计算神经科学和心理学方法学论文的审稿人期望的是 Python/R 代码——可以跑、可以复现、可以集成到自己的 pipeline。一个 React 仓库里的 `.tsx` 文件不是他们认可的交付物。

2. **Python 有成熟的统计基础设施，JS 没有。** Time-Series Split + Logistic Regression 在 sklearn 里 5 行，在 TypeScript 里要手搓矩阵运算和统计分布。Surrogate Testing 在 scipy 里 10 行，在前端要手写。花三个月在前端 debug 统计轮子，换来的是审稿人根本不会看的东西。

3. **双语言陷阱（Two-Language Problem）。** 如果同一套算法逻辑同时存在于 JS 和 Python 中，任何修改都要改两遍。先在 JS 里写再移植 Python，意味着移植时可能引入不一致，而且移植成本远高于"一开始就用 Python 写"。

**正确的做法**：Python 负责所有"可以被审稿人检查的计算"，React 只负责"可以被审稿人看到但不需要检查逻辑的可视化"。这就是 TensorBoard、Weights & Biases、fMRIPrep 的模式。

---

## 核心判断：MultiSync 做什么，不做什么

### 做什么（Dynamic Analysis Layer — 动态分析层）

这是 MultiSync 的核心贡献，没有任何现有工具覆盖：

1. **动态特征提取**：从同步性时间序列中提取 10 个时间动态特征（onset latency、build-up rate、peak amplitude、maintenance duration、breakdown rate、entropy 等），将同步性从"一个数"变成"一个过程"
2. **级联分析**：CCF 交叉相关 + Surrogate Testing，检测不同模态间同步性的 onset 顺序，识别"前奏信号"
3. **预测窗口**：朴素基线 vs. 动态特征的假设验证框架（Rolling Origin CV / LODO），量化"动态过程特征比静态描述多提供了多少信息"
4. **情境切片分析**：研究者标注情境段落，比较不同情境下同步性动态特征的差异

### 不做什么（Signal Processing Layer — 信号处理层）

这些是 HyPyP、multiSyncPy、MNE、OpenFace、pyHRV、R crqa 等工具已经做好的：

- EEG 预处理（滤波、ICA 去伪迹、源定位）
- 连接性指标计算（PLV、PLI、相干性、功率谱）
- 行为特征提取（运动能量、AU 检测、头部朝向）
- 生理信号处理（HRV、EDA 峰值检测、呼吸频率提取）
- 递归量化分析（CRQA/MdCRQA — 参数空间巨大，应在 R/MATLAB 中完成）
- 量表/问卷数据的采集和评分

**MultiSync 不关心上游用了什么算法——PLV 是一条线，MdCRQA 的 Recurrence Rate 也是一条线，multisync-core 只负责从这条线上提取动态特征、做级联分析、做预测验证。**

---

## 数据流架构

```
                    信号处理层（外部工具）              动态分析层（MultiSync）
                    ─────────────────────              ──────────────────────

原始数据              预处理 → 特征提取                  特征序列输入
───────             ─────────────────                  ──────────────
EEG (.edf)    →     MNE/HyPyP → PLV/相干性/PSD    →   同步性时间序列
    .bdf              (Python)    (per pair, per     │  (per pair, per
    .vhdr                         frequency band)   │   modality)
                                                  │
视频      →     OpenFace → AU强度/运动能量      →   同步性时间序列
                                                  │
生理信号    →     pyHRV/BioPac → HRV/EDA/RESP   →   同步性时间序列
                                                  │
量表/问卷    →     SPSS/Excel → 量表得分         →   情境标签（Context）
                                                  │
                                                  ↓
                                           multisync-core (Python)
                                           ┌─────────────────┐
                                           │  Z-score 归一化   │
                                           │  动态特征提取     │
                                           │  CCF + Surrogate  │
                                           │  Time-Series CV   │
                                           │  朴素基线对比     │
                                           └────────┬────────┘
                                                    │
                                                    ↓
                                             results.json
                                                    │
                                                    ↓
                                           ┌─────────────────┐
                                           │  MultiSync Web   │
                                           │  (React Viewer)  │
                                           │                  │
                                           │  Score View      │
                                           │  Cascade Map     │
                                           │  动态曲线        │
                                           │  情境切片对比    │
                                           │  图表导出        │
                                           └─────────────────┘
```

**输入格式要求**：
- 主输入：预处理后的同步性时间序列（CSV）
  - 每行一个时间点，列 = 各模态各指标（如 `neural_plv_alpha`, `bio_hrv_rmssd`, `behavioral_motion_energy`）
  - 推荐采样率：1-10 Hz
  - 时间粒度与信号处理工具的 epoch/滑动窗口一致
  - 量纲不限（multisync-core 在进入任何分析之前自动执行 Z-score 归一化）
- 辅助输入：情境标签（CSV）
  - 每行一个情境段落：`start_time, end_time, label`

**multisync-core 预处理管线（自动执行）**：

1. **Within-dyad Z-score Normalization**：对每个 dyad 的每个指标独立做 Z 变换。硬性步骤，不可跳过。理由：不同模态的原始量纲完全不同，归一化后所有后续分析操作的是"偏离基线多少个标准差"。
2. 缺失值处理：线性插值（短缺口 ≤ 3 epochs）或标记为 NaN（长缺口）

---

## 与现有工具的关系

| 工具 | 定位 | 与 MultiSync 的关系 |
|------|------|-------------------|
| **HyPyP** | EEG 超扫描连接性分析（Python） | 上游。multisync-core 接收其输出的 PLV 时间序列 |
| **multiSyncPy** | 多元同步性信息论指标（Python） | 上游/互补 |
| **MNE-Python** | EEG 预处理与分析（Python） | 上游 |
| **OpenFace** | 面部行为特征提取（C++/Python） | 上游 |
| **pyHRV** | 心率变异性分析（Python） | 上游 |
| **rMEA** | 行为运动能量分析（R） | 上游/互补 |

**核心差异**：所有现有工具解决的问题是"同步了多少"，MultiSync 解决的问题是"同步是如何发生的"。

---

## 技术约束与设计决策

### 1. 算法与表现分离（从 v1.0 开始）

- **multisync-core（Python）**：所有计算逻辑。调用 scipy/numpy/sklearn/pandas。可在 Jupyter Notebook 中交互式使用，也可写进自动化 pipeline。
- **MultiSync Web（React）**：纯可视化审阅器。不运行任何统计计算。读取 multisync-core 输出的 results.json，用 D3/Recharts 渲染。
- **为什么前端不做计算**：避免"双语言陷阱"、避免手搓统计轮子、确保审稿人能检查的代码在 Python 中、确保前端 UI 不会因计算阻塞而卡顿。

### 2. 输入数据粒度

推荐 1-10Hz 特征序列，理由：
- 与信号处理层的 epoch 粒度对齐（常见 epoch: 1s, 2s, 5s, 10s）
- 动态特征的时序分辨率足够区分有意义的模式
- Time-Series Split CV 在这个粒度上样本量适中、自相关性可控

### 3. Psycho 模态的重新定位

120s 采样的量表数据不能作为连续时间序列。**Psycho 降级为情境标签层（Context Layer）**：
- 不参与 WCC 计算、级联分析、预测窗口
- 作为情境分段输入，定义情境区间
- 研究逻辑变为：**在 psycho 定义的情境区间内，三模态的动态特征有什么差异**

这意味着核心分析从"四模态"缩减为**"三模态动态分析 + 情境标签"**。

**Psycho as Context Layer 是整个方案中最有论文潜力的 Idea**：如果数据显示"被试自评为高默契的区间里，行为→生理的级联发生率是低默契区间的 3 倍"，这就是一篇 Study 1 的核心发现。

### 4. 两个 deltaT 的严格区分

- **Window deltaT（特征分辨率）**：数据的时间粒度（epoch 大小）。由输入数据决定，用户不可修改。影响：onset latency 精度 ≤ Window deltaT。
- **Lag deltaT（级联容差）**：判定"级联"而非"独立事件"的最大允许时间差。由用户设定（默认 = 数据总时长的 10%）。

### 5. 伪级联的统计控制（Surrogate Testing）

CCF 分析必须配合 Surrogate Testing：
1. 对其中一个序列做 epoch 级相位打乱（circular shift），破坏时间结构
2. 重复 100 次，每次计算 CCF 峰值，得到零分布
3. 真实 CCF 峰值 vs. 零分布 → p 值
4. p < 0.05 → Cascade Map 上画**实线箭头**（统计显著）
5. p ≥ 0.05 → 画**虚线箭头**或不画

### 6. 预测窗口：假设验证，不是预测器

核心输出不是绝对 AUC，而是：
- 朴素基线（前 N 秒同步性均值 → 后 N 秒）的 AUC
- 动态特征模型（velocity、acceleration、entropy 等）的 AUC
- 两者差异 + 效应量 + 置信区间

参考表述："引入动态过程特征显著提升了对级联现象的预判能力（ΔAUC = X.XX, p < .05）"

---

## 模块映射

### multisync-core (Python) — 核心计算模块

| 模块 | 功能 | 依赖 |
|------|------|------|
| **normalization** | Within-dyad Z-score Normalization | numpy, pandas |
| **dynamic_features** | 10 个动态特征提取 | numpy, scipy |
| **cascade** | CCF + Bartlett CI + Surrogate Testing | scipy.signal, numpy |
| **prediction** | Rolling Origin CV + LODO + 朴素基线对比 | sklearn, numpy |
| **cli** | 命令行入口 | click / argparse |
| **io** | CSV 读取 + results.json 导出 | pandas, json |

### MultiSync Web (React) — 可视化模块

| 模块 | 功能 | 状态 |
|------|------|------|
| **DataLoader** | 拖入 results.json + 原始时间序列 CSV | 需重写 |
| **ScoreView** | 多模态时间轴 + 情境标注轨道 + 段内统计 | 已实现（需适配新数据格式） |
| **CascadeMap** | 级联有向图（实线 = p<0.05，虚线 = 不显著） | 已实现（需适配新数据格式） |
| **DynamicTimeline** | 动态特征时间轴可视化 | 已实现（需适配新数据格式） |
| **ContextCompare** | 情境切片对比图 | 部分实现 |
| **ExportPanel** | 图表/报告导出 | 需新建 |

---

## v1.0 里程碑（2026 年 Q2）

### 目标：一个可发布、可复现、可审稿的动态分析方法学工具

### P0 — multisync-core Python 包（第一步）

- [ ] 用 Python/Jupyter Notebook 跑通核心计算链路
- [ ] 输入：Koul 2023 OSF 公开数据（或其他公开数据集）的 CSV
- [ ] 输出：标准 `results.json`
  - 10 个动态特征的数值（per dyad, per modality）
  - CCF 交叉相关结果（peak lag, peak value, Bartlett CI）
  - Surrogate Testing p 值
  - 级联顺序和方向
  - Rolling Origin CV / LODO 结果（朴素基线 vs. 动态特征的 AUC 对比）
- [ ] Within-dyad Z-score Normalization（硬性前置）
- [ ] Window deltaT vs Lag deltaT 严格区分
- [ ] CLI 入口：`python -m multisync analyze --input data.csv --output results.json`
- [ ] 单元测试覆盖核心算法

### P0 — MultiSync Web Viewer 重构（第二步）

- [ ] 删除前端所有重度计算逻辑（`cascade-analysis.ts` 的 CCF/Surrogate、`prediction-window.ts` 的 CV/回归、`dynamic-features.ts` 的核心特征提取）
- [ ] DataLoader 重写：接受 results.json + 原始时间序列 CSV
- [ ] 所有面板（Score View / Cascade Map / Dynamic Timeline）适配 results.json 数据格式
- [ ] 数据导入引导：特征序列 CSV → multisync-core → results.json → Web Viewer
- [ ] 移除原始数据处理相关 UI 入口

### P1 — 用户体验

- [ ] Cascade Map：实线箭头（p < 0.05）vs 虚线箭头（不显著），标注 CCF peak lag + p value
- [ ] README 重写：双组件架构说明 + Quick Start（Python 计算 → Web 可视化）
- [ ] 输入格式文档（CSV schema + results.json schema）
- [ ] 演示数据：Koul 2023 CSV + 预计算的 results.json

### P2 — 学术输出

- [ ] 方法论文：动态特征操作化 + 情境切片 + 假设验证框架
- [ ] "和而不同"概念从量化指标转为理论讨论
- [ ] Paper 中明确写明双组件架构

---

## 算法整改清单

### 必须改的（科学严谨性）

| 项目 | 当前状态 | 整改方案 | 优先级 |
|------|---------|---------|--------|
| 计算语言 | TypeScript 前端实现 | **迁移到 Python**（scipy/numpy/sklearn） | P0 |
| 预测窗口 CV | LOO-CV | **删除**，替换为 Rolling Origin Time-Series Split（含 buffer gap）+ LODO | P0 |
| 因果分析 | VAR + BIC + ADF | **删除**，替换为 CCF + Bartlett CI + Surrogate Testing | P0 |
| HDI 指标 | 自定义公式，无文献背书 | **完全删除** | P0 |
| Psycho 模态 | 作为第四个并列时序 | 降级为情境标签层 | P0 |
| 特征归一化 | 无 | Within-dyad Z-score（硬性前置） | P0 |
| 级联统计检验 | onset 时间差直接作为证据 | CCF + Surrogate Testing (×100) | P0 |
| 预测窗口定位 | 报告绝对 AUC | 朴素基线 vs. 动态特征对比框架 | P0 |

### 保留但标注的

| 项目 | 处理方式 |
|------|---------|
| DTW 路径不对称性 | 保留为 CCF 的辅助方向性指标，标注 [Exploratory] |

**标签原则**：有根本性偏差的方法不该以任何形式保留。[Exploratory] 仅用于方法本身合理但未经充分验证的场景。

### 已删除

- ~~Granger/VAR 因果分析~~（WCC 序列上 VAR 建模是范畴错误）
- ~~LOO-CV~~（WCC 平滑序列上独立性假设彻底失效）
- ~~HDI 指标~~（无文献背书）
- ~~Psycho 模态时序分析~~（降级为情境标签层）
- ~~ADF 检验~~（对 WCC 序列无意义）
- ~~Symphony Report / Sonification / Conductor Analysis~~

---

## 产品命名与对外表述

### 名称
**MultiSync**（保持不变）

### 一句话描述
> MultiSync: Dynamics of Interpersonal Synchrony — 从同步性时间序列中提取动态特征、检测模态级联、验证预测假设

### README 核心段落

```markdown
## What MultiSync Does

MultiSync is a **two-component dynamic process analyzer** for interpersonal synchrony research:

1. **multisync-core (Python)**: Computes dynamic features, cascade analysis, and prediction 
   validation from pre-computed synchrony timeseries. Designed for reproducibility and 
   integration into existing research pipelines.
   
2. **MultiSync Web (React)**: Interactive visualization viewer for exploring multi-modal 
   synchrony dynamics, cascade patterns, and context-sliced comparisons.

It takes pre-computed synchrony timeseries (from tools like HyPyP, MNE, OpenFace, pyHRV) 
and answers questions that no existing tool can:

- How does synchrony unfold over time? — 10 dynamic features capture onset, build-up, 
  maintenance, and breakdown of synchrony episodes
- Which modality leads and which follows? — CCF + surrogate testing identifies 
  cross-modal lead-lag patterns with statistical rigor
- Do dynamic features outperform static descriptions? — Rolling-origin CV with naive 
  baseline comparison quantifies incremental predictive value
- How does context shape synchrony dynamics? — Context-sliced analysis compares dynamic 
  features across experimentally defined episodes

## What MultiSync Does NOT Do

MultiSync does **not** process raw signals:
- EEG preprocessing & connectivity → use HyPyP or MNE-Python
- Facial behavior → use OpenFace
- Peripheral physiology → use pyHRV or BioPac
- Questionnaire scoring → use SPSS, R, or any standard tool

Feed multisync-core your pre-computed feature timeseries (recommended: 1-10 Hz), 
then explore the results in the Web Viewer.
```

---

## 已知边界

### v1.0 明确不覆盖
- 纯前端计算（所有计算由 multisync-core Python 包完成）
- Group-level 批处理 UI（multisync-core 支持批处理，但 Web Viewer v1.0 专注单 dyad 审阅）
- 临床数据合规（IRB/HIPAA 要求本地部署 → 未来 Electron/Tauri 方案）

### v1.0 明确覆盖
- 单 dyad 的完整动态分析链路（Python 计算 → Web 可视化）
- 可复现的研究代码交付（Python，审稿人可检查）
- 方法学 Paper 的概念验证（使用 OSF 公开数据）

---

*MultiSync 从 v1.0 开始就严格分离算法（Python）与表现（React），避免双语言陷阱。Python Core 负责所有可被审稿人检查的计算，React Viewer 负责零安装的交互式可视化审阅。放弃信号处理不是退缩，是聚焦。放弃前端计算不是退缩，是专业。*
