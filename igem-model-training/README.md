# iGEM Model 实战课

给第一次做 iGEM 建模的同学。课程来自往届生物动力学与数学建模讲义，使用公开的 iGEM 项目、研究论文和小规模练习。每个案例从作者要做的决定出发，跟到输入、方程或算法、软件、计算量、结果核对和下一步验证。

## 第一次上手

建议从[拟合与插值](lessons/03-fitting-interpolation.md)开始。它有团队公开数据和可以复算的四个峰位。接着学[统计](lessons/10-statistics.md)和[生物动力学](lessons/01-dynamics.md)，就能理解多数 iGEM Model 页如何把测量变成预测。

1. 从 GitHub 仓库页面的 **Code → Download ZIP** 下载并解压，进入 `igem-model-training` 文件夹。也可用 Git：

   ```bash
   git clone https://github.com/gouyutong2-hue/2027-model-.git
   cd 2027-model-/igem-model-training
   ```

2. 用终端进入这个课程文件夹。运行 `ls`（macOS/Linux）或 `dir`（Windows），看到 `requirements.txt` 和 `examples` 后再继续。需要一份 Python；课程在 Python 3.10.19 上实测。
3. 首次安装依赖需要联网。macOS / Linux：

   ```bash
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   .venv/bin/python examples/curve_fitting.py
   ```

   Windows PowerShell：

   ```powershell
   py -3.10 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe examples/curve_fitting.py
   ```

4. 在 `build/curve_fitting` 打开 `activity.png`，用 Excel、Numbers 或文本编辑器打开 `model_comparison.csv`。控制台应打印两个温度峰位 55.5 °C、57.5 °C，以及两个 pH 峰位 5.5、5.8。
5. 后续章节用 `python` 简写。可换成刚用过的虚拟环境解释器路径。检查全套时运行 `python run_examples.py`，它会把各章输出放在 `build`。除选做的外部模型下载，基础例题安装依赖后可以离线跑。普通本机 CPU 足够，无需 MATLAB、GPU、付费云算力或连接实验仪器。

## 课程地图

| 课题 | 真实项目或原始研究 | 跟做产物 |
|---|---|---|
| [生物动力学](lessons/01-dynamics.md) | TAS Taipei、Cambridge、Manchester、XMU-China、ZJU-China 与 Gause 历史实验等 | 米氏曲线、报告信号、受限增长、竞争与捕食、数值积分比较 |
| [蒙特卡洛](lessons/02-monte-carlo.md) | NMU-China 搜救可见性 | 阈值覆盖图、抽样区间和解析对照 |
| [拟合与插值](lessons/03-fitting-interpolation.md) | Fujian-United 植酸酶实测曲线 | 四个峰位、插值对比和留点误差 |
| [小波](lessons/04-wavelets.md) | 君子兰叶面电位原始研究 | 模拟波形去噪、突变定位和已知真值检查 |
| [LP、MILP 与 NLP](lessons/05-optimization.md) | Düsseldorf Fungilyzer、托莱多教学医院、Fudan 酵母生长 | 稳态通量、手术间容量分配、非线性最小二乘 |
| [排队论](lessons/06-queueing.md) | iGEM Toronto 自动化工程、医院采血和床位研究 | 开窗表复算、拥堵或损失率、离散事件模拟和排班 |
| [对策论](lessons/07-game-theory.md) | 捕食者搜寻、酵母合作、CJUH-JLU-China 模型解释 | 零和随机策略、Nash 均衡、演化频率和 Shapley 值 |
| [层次分析法](lessons/08-ahp.md) | UESTC Nebula、微藻收获技术选型 | 判断矩阵、总排序、一致性和权重敏感性 |
| [遗传算法](lessons/09-genetic-algorithm.md) | NMU-China 的模型优化 | 等预算随机基线、20 次复跑和穷举验收 |
| [统计描述与推断](lessons/10-statistics.md) | 2016 iGEM InterLab 的同行评议分析 | 校准后质控、几何统计量、配对推断与经典检验 |
| [微分方程建模](lessons/13-differential-equations.md) | Cornell 补料反应器、Fudan 酵母、Gause 混合培养和天然微藻 | 质量衡算、Logistic、物种互作、沉降与事件求解 |

## 怎么跟做

每个案例先打开原链接确认作者问题，再关掉脚本，写出输入、输出和一个可验证的预测。运行后核对结果，改一个输入重跑。最后提交一张图、一个小表和两段话。第一段写模型会帮助你决定什么，第二段写当前证据支持到哪里。

原项目的曲线、表格、参数和代码按各章标注。能复算的给数据或转录位置。只能重画公布曲线时注明未重新估计参数。输入或实现缺失的案例提供有验证目标的教学仿真。合成输入、数字化观测和团队报告数字都有独立来源说明。[旧讲义与课程对照](sources.md) · [数据与许可](data/README.md)

## 第一次改代码

打开 `examples/curve_fitting.py`。把绘图或打印的内容改一项，重跑确认结果有变化。做完整复现记录时，请保存：

- 来源及具体图、表或公式，输入文件、单位、处理步骤和重复数。
- 目标函数、约束、初值、参数含义、种子与依赖版本。
- 独立检查、留出误差、典型失败案例和剩余限制。
- 一条从干净环境可以重跑的命令。

如果出现 `ModuleNotFoundError`，先确认安装依赖和执行脚本用的是同一个 Python。如果提示找不到 `examples` 或 `requirements.txt`，检查是否进入了课程目录。图片会写到文件夹，终端运行后无需等待图窗。
