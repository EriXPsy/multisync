# MultiSync 代码整改清单 v4.0

*基于四轮审查反馈后的最终执行计划（架构重大调整：算法与表现分离）*
*更新于 2026-04-24*

---

## 架构变更概览

**核心变更**：从"纯前端 + 未来移植 Python"改为"从一开始就分离 Python Core + React Viewer"。

**原因**：
- 审稿人不会接受前端代码作为可复现的研究代码
- Python 有成熟的统计基础设施（scipy/sklearn），JS 没有
- 避免双语言陷阱（同一算法维护两套实现）
- 前端不做计算 → UI 不会因计算阻塞而卡顿

**执行顺序**：先写 multisync-core Python 包（P0-1），再重构 React Viewer（P0-2）。

---

## P0-1：multisync-core Python 包（第一步，优先级最高）

### 1. 项目初始化

- [ ] 创建 `multisync-core/` 目录
- [ ] 初始化 Python 包结构：
  ```
  multisync-core/
  ├── multisync/
  │   ├── __init__.py
  │   ├── normalization.py    # Z-score 归一化
  │   ├── dynamic_features.py # 10 个动态特征提取
  │   ├── cascade.py          # CCF + Bartlett CI + Surrogate Testing
  │   ├── prediction.py       # Rolling Origin CV + LODO + 朴素基线
  │   ├── io.py               # CSV 读取 + results.json 导出
  │   └── cli.py              # 命令行入口
  ├── tests/
  ├── pyproject.toml
  └── README.md
  ```
- [ ] 依赖声明：scipy, numpy, sklearn, pandas, click (或 argparse)
- [ ] CI：基础单元测试

### 2. normalization.py — Within-dyad Z-score Normalization

- [ ] 对每个 dyad 的每个指标独立做 Z 变换：z = (x - mean) / std
- [ ] 在所有动态特征提取之前执行，硬性步骤不可跳过
- [ ] 处理全零/全相同值序列（std = 0 → 跳过归一化，标记 warning）
- [ ] 单元测试

### 3. dynamic_features.py — 10 个动态特征提取

- [ ] 从当前 TypeScript 实现移植特征定义（不移植计算逻辑，用 numpy/scipy 重写）：
  1. Onset Latency（首次超过阈值的时间）
  2. Build-up Rate（从 onset 到 peak 的攀升速度）
  3. Peak Amplitude（最大值）
  4. Time to Peak（onset 到 peak 的时间）
  5. Maintenance Duration（维持在 peak 附近的时间）
  6. Breakdown Rate（从 maintenance 到 baseline 的下降速度）
  7. Episode Duration（整个同步性事件的持续时间）
  8. Entropy（时间序列的信息熵）
  9. Variability（标准差 / 均值比率）
  10. Recovery Time（breakdown 后回到 baseline 的时间）
- [ ] 阈值参数：默认 = mean + 1 SD（归一化后）
- [ ] 所有特征在归一化后的序列上计算（单位 = z-score / epoch）
- [ ] 单元测试（用合成数据验证每个特征的输出）

### 4. cascade.py — CCF + Surrogate Testing

- [ ] **删除移植**：不移植任何 VAR/Granger/ADF/BIC 代码
- [ ] 实现 CCF（Cross-Correlation Function）：
  - `scipy.signal.correlate` 计算交叉相关
  - Bartlett's Formula 估计置信区间
  - 提取最大相关峰值对应的 lag
- [ ] 实现 Surrogate Testing：
  - epoch 级相位打乱（`numpy.roll` circular shift）
  - 重复 N 次（默认 100），每次计算 CCF 峰值
  - 真实峰值 vs. 零分布 → p 值（经验 p-value）
  - 返回：CCF peak value, CCF peak lag, p value, significance flag
- [ ] 实现 Lag deltaT 过滤：
  - 如果 |onset_A - onset_B| > lag_deltaT → 不构成级联
- [ ] 可选：DTW 路径不对称性（标注 [Exploratory]）
- [ ] 单元测试

### 5. prediction.py — 假设验证框架

- [ ] **删除移植**：不移植 LOO-CV
- [ ] 实现 Rolling Origin Time-Series Split：
  - train: [0 : t - buffer], test: [t : t + horizon]
  - buffer ≥ 1 个 Window deltaT
  - 多个 fold 的结果聚合
- [ ] 实现 LODO (Leave-One-Dyad-Out)：
  - 跨被试对验证，默认策略
- [ ] 实现朴素基线模型：
  - 用前 N 秒同步性均值预测后 N 秒
  - sklearn DummyRegressor 或手动实现
- [ ] 实现动态特征模型：
  - 用 velocity, acceleration, entropy 等作为特征
  - sklearn LogisticRegression（或等效）
- [ ] 输出对比框架：
  - Baseline AUC vs. Dynamic Features AUC
  - ΔAUC + 效应量 + 置信区间
- [ ] 单元测试

### 6. io.py — 输入输出

- [ ] CSV 读取：自动识别时间列 + 数值列 + dyad ID 列
- [ ] Window deltaT 自动检测：从数据的时间戳间隔推断
- [ ] results.json 导出：标准格式，包含所有计算结果
- [ ] 情境标签 CSV 读取

### 7. cli.py — 命令行入口

- [ ] `python -m multisync analyze --input data.csv --output results.json`
- [ ] 参数：`--surrogates N`, `--cv-strategy {rolling,lodo}`, `--lag-delta-t SECONDS`
- [ ] 进度条（tqdm）

### 8. 端到端验证

- [ ] 用 Koul 2023 OSF 公开数据跑通完整管线
- [ ] 验证 results.json 包含所有预期字段
- [ ] 验证数值合理（与手动计算对比）

---

## P0-2：MultiSync Web Viewer 重构（第二步）

### 9. 删除前端计算逻辑

- [ ] **删除** `src/lib/cascade-analysis.ts` 中的 CCF/Surrogate/VAR/Granger 计算代码
- [ ] **删除** `src/lib/prediction-window.ts` 中的 CV/回归计算代码
- [ ] **删除** `src/lib/dynamic-features.ts` 中的核心特征提取逻辑
- [ ] 保留文件作为**类型定义**（TypeScript interfaces for results.json schema）
- [ ] 保留纯 UI 渲染逻辑（组件、样式、图表配置）

### 10. DataLoader 重写

- [ ] 接受两种输入：
  1. `results.json`（multisync-core 的输出）
  2. 原始时间序列 CSV（仅用于渲染时间轴背景）
- [ ] 不再在前端做任何统计计算
- [ ] 如果用户直接上传 CSV 而没有 results.json → 显示引导信息："请先用 multisync-core 生成结果"

### 11. 面板适配

- [ ] **ScoreView**：从 results.json 读取动态特征数据，渲染情境标注轨道
- [ ] **CascadeMap**：从 results.json 读取 CCF 结果 + p 值，实线/虚线箭头
- [ ] **DynamicTimeline**：从 results.json 读取动态特征时间序列
- [ ] **ContextCompare**：从 results.json 读取情境切片统计

### 12. 数据导入引导

- [ ] 移除原始数据处理相关 UI 入口
- [ ] 新用户引导流程：上传特征序列 CSV → 提示运行 multisync-core → 上传 results.json → 可视化审阅

---

## P1：体验优化

### 13. Cascade Map 可视化
- [ ] 实线箭头 = Surrogate Testing p < 0.05
- [ ] 虚线箭头 = p ≥ 0.05（可选显示）
- [ ] 箭头标注：CCF peak lag + p value

### 14. 文档
- [ ] README 重写：双组件架构 + Quick Start
- [ ] multisync-core README：安装、使用、API 文档
- [ ] results.json schema 文档
- [ ] 输入 CSV 格式文档

### 15. 演示数据
- [ ] Koul 2023 CSV + 预计算的 results.json
- [ ] 情境标签 CSV 示例

---

## P2：后续迭代

### 16. 学术输出
- [ ] 方法论文：动态特征操作化 + 情境切片 + 假设验证框架
- [ ] Paper 中明确双组件架构
- [ ] "和而不同"从量化指标转为理论讨论
- [ ] Psycho as Context Layer 的实证范例

### 17. 架构演进
- [ ] `pip install multisync`（正式 PyPI 发布）
- [ ] Web Viewer 支持多 dyad 结果切换
- [ ] 评估 Electron/Tauri 桌面客户端（v3.0，解决 IRB 合规）

---

## 已删除的规划（不再执行）

- ~~前端 CCF/Surrogate Testing 计算~~ → 移至 multisync-core
- ~~前端 Time-Series CV / LODO 计算~~ → 移至 multisync-core
- ~~前端动态特征提取计算~~ → 移至 multisync-core
- ~~Granger/VAR 因果分析~~（范畴错误，完全删除）
- ~~LOO-CV~~（独立性假设失效，完全删除）
- ~~HDI 指标~~（无文献背书，完全删除）
- ~~ADF 检验~~（对 WCC 序列无意义）
- ~~BIC 滞后阶数选择~~（随 VAR 删除）
- ~~Symphony Report / Sonification / Conductor Analysis~~
- ~~[Experimental] 标签~~
- ~~MdCRQA 前端实现~~（在 R/MATLAB 中完成）
- ~~"先 JS 验证再移植 Python"的策略~~（双语言陷阱，改为从一开始就分离）

---

*v4.0 的核心变化：承认"在 TS 里手搓统计轮子是无效劳动"，将所有计算逻辑迁移到 Python（scipy/numpy/sklearn），React 前端彻底变为纯可视化审阅器。这不是退缩，是专业。*
