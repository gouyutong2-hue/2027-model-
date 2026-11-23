# 往届讲义与课程对照

课程根据附件里的一个动力学 PDF、五章数学讲义和一个 HTML 编写。同一方法的 HTML 提纲与 PDF 详稿合在一课。附件内的操作提示帮助定位章节和例题。这里把训练目标落实到可运行的案例和作业。

| 附件位置 | 对应教学内容 |
|---|---|
| `chap01.pdf`，14 页 | [反应动力学](lessons/01-dynamics.md)含质量作用、可逆反应、米氏、Hill、mRNA 与蛋白质、布尔规则、指数、Logistic、Monod、混合模型、活死细胞、Contois、竞争与捕食、双倒数回归、Euler、梯形、Heun、RK4、相图及 Wiki 公式 |
| `数学建模算法.html` §1 | [蒙特卡洛](lessons/02-monte-carlo.md)和随机数 |
| HTML §2 | [拟合及四种插值](lessons/03-fitting-interpolation.md)、[小波及异常检测](lessons/04-wavelets.md) |
| HTML §3 | [线性、整数线性与非线性规划](lessons/05-optimization.md) |
| HTML §4 | [遗传算法](lessons/09-genetic-algorithm.md) |
| HTML §5 与 PDF 第六章 | [排队论](lessons/06-queueing.md) |
| HTML §6 与 PDF 第七章 | [对策论](lessons/07-game-theory.md) |
| HTML §7 与 PDF 第八章 | [层次分析](lessons/08-ahp.md) |
| HTML 统计部分与 PDF 第十章 | [统计描述与分析](lessons/10-statistics.md) |
| HTML 微分方程部分与 PDF 第十三章 | [微分方程建模](lessons/13-differential-equations.md)。用补料质量衡算、多物种竞争和藻细胞沉降练习原章的分阶段方程、联立方程、力平衡及阈值事件 |

## 检索与复现记录

2026-11-23 核对公开 iGEM 队伍 Wiki、原作者项目仓库、期刊正文、附表与官方软件文档。检索按训练方法加问题组合，例如 `iGEM model cubic spline phytase`、`iGEM flux balance analysis phosphate`、`iGEM Monte Carlo modeling illumination`、`InterLab independent calibrants fluorescence`、`wavelet plant electrophysiology`、`hospital M/M/c phlebotomy`、`predator prey zero-sum game`、`microalgae harvesting AHP` 和 `gauseR Lotka Volterra`。选入有明确项目决策、可定位的方法步骤和至少一种数值检查的来源。队伍材料的公开程度不一，课程优先利用原作者公开数据和代码，原资料缺口按各课列出。

| 代表性案例 | 可检查的来源及复现范围 |
|---|---|
| 植酸酶 | 原 Model PDF §3–4 给出完整数组和代码。按原流程重算峰位；AppA 的两处峰高差异记在课内 |
| InterLab | 原论文 S1 Table 含 65 行。运行后对比正文筛选比例，保留 27 与 30 的差异；配对统计为课堂二次分析 |
| 海上可见性与模型优化 | NMU-China Wiki 的 Fig. 13–16 描述流程和团队输出；课程用通用点光源重做步骤 |
| 植物信号 | 原论文 Fig. 8–9 提供波形示例和处理流程；原始录波需向作者索取，课堂用可核验的模拟序列 |
| 酵母网络、医院排程 | Düsseldorf Wiki 列出代谢网络、酶约束示例与软件；论文表 2–3 列出排班输入。课堂复算底层 LP 和聚合排程 |
| 历史生态数据与增长 | 论文表中参数和作者数字化数据用于模型重画、重新拟合及留出检验；每个小节标明具体证据 |
| 排队、对策与 AHP | 论文表格、收益设定或判断矩阵可以复算；iGEM 工程案例补足真实应用背景，代码可用性见各章 |

优先使用可以解释算法价值且有验收目标的项目。只有方法名称而缺少模型、数据和结果的 Wiki 没有入选主案例。序列设计和实验操作流程不属于本课程的复现内容。
