# 生物动力学：从反应、读数到预测

本章对应旧讲义 `chap01.pdf` 第 1–10 页和第 11–14 页的 Wiki 公式。完成后，你能把过程写成反应速率或微分方程，从读数求初始速率，用守恒检查仿真，并用留出的时间点判断模型是否值得外推。先动手跑一次，再选一项练习修改。

从课程目录运行。依赖安装见首页，全部实验只用 Python、NumPy、SciPy、Matplotlib 与随附 CSV。普通笔记本 CPU 足够，数据与依赖就绪后可以离线，整章实测约数秒。

```bash
python examples/dynamics.py --out build/dynamics
```

只想跑一种实验时，添加 `--lab enzyme`、`--lab reporter`、`--lab growth`、`--lab community` 或 `--lab solvers`。每项生成一张 PNG，终端会打印估计参数、轨迹终点或误差。

| 旧讲义内容 | 亲手完成的任务 | 可追溯案例 |
| --- | --- | --- |
| pp. 1–2 基元反应、质量作用、可逆反应和米氏动力学 | 用守恒检查反应系统，画出测量范围内的饱和曲线 | iGEM 2021 TAS Taipei 酶动力学 |
| pp. 2–3 Hill、mRNA 与蛋白质、布尔调控 | 模拟三种刺激下的荧光延迟，从真值表生成连续响应 | iGEM 2024 Cambridge、iGEM 2022 KU Leuven、BEELINE 论文 |
| pp. 4–5 指数、Logistic、单底物与多底物 Monod、混合模型、活死细胞、Contois | 画生物量与底物曲线，辨别饱和成因 | iGEM 2022 Manchester、iGEM 2024 XMU-China、小球藻论文 |
| pp. 5、7–9 多种群竞争及捕食、相图 | 拟合历史实验，检验留出期并画相图 | Gause 1935 与 Mühlbauer 等 2020 |
| pp. 5–8 双倒数回归、Euler、梯形、Heun、RK4 | 比较速率误差与步长误差 | TAS Taipei、iGEM 2025 ZJU-China、SciPy |
| pp. 10–14 MATLAB、LaTeX 与 MathJax | 辨认求解器，并在 GitHub 写出可渲染公式 | MathWorks、GitHub 和 iGEM 模板文档 |

## 1. 酶动力学与曲线拟合

### 作者如何把比色读数变成速率

TAS Taipei 在 2021 年获得高中组 Best Model。团队想从酶活性测量得到可用于工程预测的参数。读原 Wiki 的 **Enzyme Kinetics Model**，沿 Figure 4、5、6 追踪同一条 β-半乳糖苷酶路径。其建模工作分四步。 [iGEM 获奖记录](https://blog.igem.org/blog/2023/9/13/communicating-with-the-world-through-igem-team-wikis) · [TAS Taipei 原项目](https://2021.igem.org/Team:TAS_Taipei/Model#Enzyme_Kinetics_Model)

1. 用浓度已知的显色物做标定。在 Figure 4 的情景里，吸光度是仪器观察量，显色产物浓度才进入速率计算。
2. 对每组初始底物浓度，把吸光度转换为产物浓度。初始近线性区间内用浓度对时间的斜率估计 $v_0$，每组得到一个 $(S_0,v_0)$。
3. 用 Lineweaver–Burk 图做回归。横轴 $1/S_0$、纵轴 $1/v_0$，斜率 $a=K_m/V_{max}$、截距 $b=1/V_{max}$。求得 $V_{max}=1/b$、$K_m=a/b$，再转回原来的坐标。
4. Figure 6 公布了拟合曲线。横轴是比色底物浓度（mM），纵轴是初始速率（mM/min）。$K_m=0.3762$ mM，$V_{max}=0.0345$ mM/min；原图误差棒为一倍标准误。正文未找到可直接重跑的逐点表和拟合源码。本章用已公布参数重画曲线，用另设的模拟读数完成拟合练习。 [原 Wiki Figure 4–6 与 Eq. 8](https://2021.igem.org/Team:TAS_Taipei/Model)

### 为什么能写成双曲线

先列基元反应 $E+S\rightleftharpoons C\rightarrow E+P$。正向结合速率 $k_{on}ES$、解离速率 $k_{off}C$，合成产物速率 $k_{cat}C$。这里 $E,S,C,P$ 都表示浓度，$C$ 就是酶与底物的复合物。程序把状态排序为 `[E, S, C, P]`。正、逆结合速率的差记为 $J=k_{on}ES-k_{off}C$，产物生成通量是 $Q=k_{cat}C$。

$$
(\dot E,\dot S,\dot C,\dot P)=(-J+Q,-J,J-Q,Q).
$$

因此 $E+C$ 和 $S+C+P$ 在闭合模型里恒定。若复合物先快速调整、随后变化慢，可令 $\dot C\approx0$，得到

$$
v=V_{max}\frac{S}{K_m+S},\qquad
K_m=\frac{k_{off}+k_{cat}}{k_{on}},\quad V_{max}=k_{cat}(E+C).
$$

$K_m$ 是半最大速率对应的底物浓度，式中的 $k_{cat}$ 说明它一般与只由结合和解离定义的解离常数不同。可逆结合要写两条通量。质量作用定律按基元反应写，整条多步反应的化学计量系数通常不足以定出速率律。

### 运行与验收

- `dynamics-enzyme.png` 左图是 Figure 6 的拟合曲线重绘，计算点 $(0.3762,0.01725)$ 应落在曲线上。用高底物极限检查曲线应接近 $0.0345$ mM/min。
- 中图使用 12 个模拟底物浓度、每组 7 个模拟时间点、固定随机种子。代码先扣空白再除以标定斜率，求得 12 个初速。合成真值 $K_m=0.4$ mM、$V_{max}=0.036$ mM/min。本机双倒数法估计 $0.2741$、$0.03120$，原尺度最小二乘给出 $0.3506$、$0.03482$。速率 RMSE 分别约 $0.00167$、$0.00106$ mM/min。
- 右图用一组独立的模拟基元速率，比较完整反应和米氏近似的产物时间曲线。改变 `e_total` 后重新检查两个守恒式，再比较初期的产物误差。减小初始底物或让酶用量接近底物时，准稳态近似的误差可能显著增加。

**选型。** 米氏模型仍适合解释简单饱和初速。双倒数图有直观的截距含义，适合读历史文献和教学。倒数变换使低速率上的误差显著放大，估参数通常在原始速率尺度做正值约束的非线性最小二乘。有重复测量时可用方差信息加权，先看残差随底物浓度是否变化。若已有整段时间序列且出现底物耗尽、产物抑制或反向反应，完整 ODE 的联合拟合可以利用更多信息，同时要检查多出来的参数能否识别。全部计算可用本地 SciPy。 [Lineweaver–Burk 原论文](https://pubs.acs.org/doi/10.1021/ja01318a036) · [原尺度及全程数据对照，Figure 1–2](https://pmc.ncbi.nlm.nih.gov/articles/PMC4624389/)

## 2. 从 Hill 响应到有延迟的荧光信号

### 原项目的设计思路

KU Leuven 2022 希望把输入分子浓度接到荧光输出。作者先把荧光除以 OD600，当作归一化读数，用 SciPy 拟合含基线、上限、半效输入和陡峭度的 Hill 曲线。随后按反应链拆模块，用 Simulink 组合它们。Figure 8 的 $R^2=0.9935$ 描述这次拟合，进一步评估跨条件预测需要独立数据。本章使用它说明 Hill 的观测标定。 [KU Leuven 原 Wiki，Model Description 和 Results Step 1](https://2022.igem.wiki/ku-leuven/model)

Cambridge 2024 的酵母荧光模块关注**达到可观测信号需要多久**。他们把刺激写到转录速率中，再分开模拟 mRNA、未成熟荧光蛋白和成熟荧光蛋白。项目用热模型给定不同位置的情景，考察输出曲线随位置的变化。作者说明这一酵母路径没有得到可用构建体，页面上的时间曲线是计算情景。本章采用相同的状态划分，用无量纲刺激和教学参数演示。 [Cambridge 原 Wiki，Thermal: S. cerevisiae (HSF)](https://2024.igem.wiki/cambridge/model)

### 从方程到输出

$H(u)=u^n/(K^n+u^n)$ 在 $u=K$ 处等于 $1/2$，$n$ 越大，转折附近越陡。抑制模型可用 $1-H(u)$。如有光学背景，在饱和函数外加基线并调整幅度。经验型 Hill 指数描述曲线形状，分子结合数需要独立机制证据。模拟采用三个状态，成熟速率在未成熟状态中作为流出，在成熟状态中作为流入。

$$
\begin{aligned}
\dot m &= aH(u)-\lambda_m m,\\
\dot D &= b m-\lambda_DD-kD,\\
\dot F &= kD-\lambda_FF.
\end{aligned}
$$

$m$ 是相对 mRNA 量，$D,F$ 是以同一相对量标度表示的未成熟和成熟报告蛋白。在程序的观测模型里只画 $F$，光学仪器的比例因子取为 1。时间用教学单位。即使 $u$ 一开始便改变，也要经过生成和成熟过程，信号才会累积。这里只有末端荧光时，多种转录、翻译与成熟参数组合可能给出类似曲线。

### 布尔规则如何进入 ODE

BEELINE 的研究者要检验单细胞表达数据能否反推出调控网络，先让正确答案已知，才有办法给不同推断算法打分。BoolODE 把表格里的布尔规则变为连续调控函数，为每个基因写转录、翻译与降解方程，加入随机项产生多条轨迹，最后重采样为模拟表达矩阵。作者还提供网络边表，作为从模拟数据反推调控关系的对照。GeneNetWeaver 是培训讲义给出的另一种原研究基准，用公开工具生成并评测调控网络。 [BEELINE 原论文，Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC7098173/) · [BoolODE 原代码](https://github.com/murali-group/BoolODE) · [GeneNetWeaver 原论文](https://academic.oup.com/bioinformatics/article/27/16/2263/254752)

`reporter` 练习取独立活性 $h_A,h_B,h_C\in[0,1]$，对 8 种开关状态逐一算权重。ON 用 $h_i$，OFF 用 $1-h_i$，将满足 $(A\lor B)\land\neg C$ 的状态权重相加，得到 $f=(h_A+h_B-h_Ah_B)(1-h_C)$。把 $h_i$ 写成 $z_i/(1+z_i)$，其中 $z_i$ 是一个无量纲占据权重，就得到讲义第 3 页同结构的有理式。本地代码验证了所有 8 个端点。论文完整目标需要随机轨迹，原仓库标注 BEELINE 发表版本对应 v0.1，教程的脚本入口与后续 YAML 版有差异。要尝试作者的例子，按相应版本的教程准备制表符分隔的 `Gene/Rule` 表和初始条件，先跑少量细胞，查看 `refNetwork.csv` 与 `ExpressionData.csv`。这一步是选做，本章的确定性示例已实测，上游论文环境未运行。 [BoolODE 作者教程](https://beeline-docs.readthedocs.io/en/latest/BoolODE.html)

**验收与选型。** `dynamics-reporter.png` 左图的三个 Hill 指数曲线会在 $(1,0.5)$ 交会。中图的末时刻模拟荧光约为 $0.196$、$3.728$、$7.190$ a.u.。右图提高 C 会压低整条曲线。练习将成熟速率调为两倍，比较达到各自末值一半的时间；再令 `stimulus = 1`，得到讲义中的组成型表达。Hill 和低维 ODE 仍是解释饱和与延迟的良好起点。只有逻辑证据时可先用布尔模型，有定量时间序列后再拟合连续模型。需要比较细胞间分布时，再引入随机模型或测量噪声。

## 3. 生长模型如何支持团队做决策

### iGEM 2022 Manchester：先问哪种资源最限制藻类

作者把微藻培养环境拆成氮、磷、光三个因子，用乘积修正最大比生长速率。给定初始营养与光条件后，改变一项输入，观察初始比生长率变化，据此讨论团队系统下一步该测什么。原项目给出约 $0.32\,\mathrm{h}^{-1}$ 的估计，并把氮列为该情景下的初始限制因素。该数值来自模型情景，作者给出的 MATLAB 下载归于同页另一条泄漏模型。本章练习用同一乘积结构重做局部敏感性分析。 [Manchester 原 Wiki，Microalgae Growth Model](https://2022.igem.wiki/manchester/model)

单底物 Monod 将最大增速乘以 $S/(K_S+S)$。多底物写成 $\mu_{max}H_N H_Pf_I$，其中 $H_N=N/(K_N+N)$，磷同理，$f_I$ 为光照因子。某种营养远高于它的半饱和常数时，继续增加的边际效果变小。乘积需要数据支持，若候选假说是完全由最稀缺资源决定，则可评估两个饱和因子取最小值的版本。

### iGEM 2024 XMU-China：OD 可能混合了不同状态

作者把活细胞、死亡细胞和裂解组分拆成状态，在活细胞的增长中同时加入底物饱和和容纳量。模型区分了生长所用物质、细胞状态和生成物读数。这个案例值得逐项检查通量符号与单位。例如页面中 $K_S$ 和 $S$ 相加，二者应有相同的浓度单位。为数值实现检查边界时，可把某种群设为零，确认流出项也为零。本章用一个守恒的转移模型做同类分析，合成生物量单位统一为 g/L。 [XMU-China 原 Wiki，Macro-Level Model](https://2024.igem.wiki/xmu-china/model)

- 指数增长用 $\dot X=\mu X$。早期且条件近似不变时，可对 $\log X$ 与时间做线性回归估计增长率。
- Logistic 用 $\dot X=rX(1-X/K)$，$K$ 是当前条件下的表观承载量。数据有平台期时，可先拟合这两个参数。
- Monod 可与底物耗竭耦合。若每单位消耗底物产生 $Y$ 单位生物量，生长通量 $G=\mu(S)X$ 满足 $\dot S=-G/Y$。出现平台时，瓶颈可能来自资源耗尽，也可能来自其他容量限制。
- 混合模型把 Monod 与 Logistic 因子相乘。引入死亡率 $k_d$ 和裂解率 $k_l$ 后，本例用 $\dot X=G-k_dX$、$\dot D=k_dX-k_lD$、$\dot L=k_lD$。于是 $X+D+L+YS$ 守恒。旧讲义式 (24) 用总量的 Logistic 约束反推死亡组分，实用时应检查该差值在死亡组分为零时是否产生负值。原混合模型论文以批式与连续搅拌釜的形式推导解析关系，Supplementary files 有 MATLAB 代码说明。 [Xu 2020，Table 1–2 与 Supporting Information](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/bit.27230)

### 论文补充：用 Contois 表达底物和生物量的联合限制

Contois 用 $\mu=\mu_{max}S/(K_C X+S)$，$K_C$ 的量纲是底物浓度比生物量浓度。固定同一个 $S$ 时，较多生物量可能对应较低的每单位增速。Sudibyo 等用 MATLAB 对小球藻的 sCOD 与干生物量比较 Monod、Contois、Haldane，在论文 Figure 4 的条件下 Contois 报告的 SSE 较小，Table 1 给出参数与误差。这个拟合比较只支持该数据情景，sCOD 是代理读数。可用另外一组初始生物量的时间序列检查 $X$ 进入速率律是否改善预测。 [Contois 原论文](https://www.microbiologyresearch.org/content/journal/micro/10.1099/00221287-21-1-40) · [小球藻原研究，Figure 4、Table 1](https://www.sciencedirect.com/science/article/pii/S1876610217303478/pdf)

**运行与验收。** `dynamics-growth.png` 的曲线与参数均为教学设定。本机模拟 30 h 后，Logistic 活生物量约 $1.60$ g/L，Monod 与 Contois 都约 $1.01$ g/L，混合加死亡约 $0.54$ g/L。Monod 与 Contois 的终点接近，因为该模拟里底物基本耗尽且二者都用相同的转化率。观察中期曲线可见两者抵达平台的时间不同。右下是 Manchester 式的初始敏感性推演。$N/K_N=0.5$ 而 $P/K_P=3$ 时，先算分别将氮或磷加倍的预期增长率变化，再用代码验证。

**选型。** 指数、Logistic、Monod、Contois 都适合做基线，选哪一个取决于能测到什么与要预测什么。只有一条 OD 曲线时，先用简模型。要区分资源和死亡，应同步观察底物和活性信号。底物过量后增速反而下降时，可比较 Andrews/Haldane 抑制形式；延滞期明显时，可比较带延滞参数的经验曲线。完整模型与基线应共享相同的留出条件和误差尺度。 [实际藻类模型比较，Table 1](https://link.springer.com/article/10.1186/s40643-026-01145-x)

## 4. Lotka–Volterra：用真实时间序列检查预测

### 先区分竞争和捕食

竞争模型让每个种群按自身承载量减速，并叠加对手的抑制。对 A 来说，写成 $r_A X(1-(X+\alpha_{BA}Y)/K_A)$。$\alpha_{BA}$ 表示每单位 B 对 A 的等效竞争压力，反方向可以用不同系数。在捕食模型里，猎物的交互项为负、捕食者的交互项为正。经典简式用：

$$
\dot X=aX-bXY,\qquad \dot Y=-cY+dXY .
$$

$X$ 是猎物，$Y$ 是捕食者。遇见的机会用 $XY$ 表示，模型给出捕食者无猎物时减少、猎物无捕食者时增加两个方向性检查。相图以猎物为横轴、捕食者为纵轴，每点对应同一时刻的两个状态。需结合时间曲线查看走向与峰值间隔。

### 作者做了什么

Gause 在 1935 年 Figure 3 比较酵母与草履虫混合系统的数量波动，在受控实验中观察近似周期和峰值错位。其论文指出系统含草履虫稀释操作，实验中的持续波动与这些条件有关。随仓库附的 CSV 来自公开项目 `predator_prey` 对该图的数字化，许可为 CC0。数据有两个种群各 18 个时间点，横轴是日。酵母以每 $0.1\ \mathrm{mm}^3$ 计，草履虫以每 $15\ \mathrm{cm}^3$ 计。第三列 `Individuals` 是从图读出的密度标度，保留小数；仓库和原图未提供逐点数字化误差或平行样本数。这里使用公开逐点数据，未重新读图。 [Gause 原论文全文，p. 47 Figure 3](https://www.math.toronto.edu/afenyes/teaching/mat244-fall2018/G.%20F.%20Gause%20-%20Experimental%20Demonstration%20of%20Volterra%27s%20Periodic%20Oscillations%20in%20the%20Numbers%20of%20Animals.pdf) · [公开数字化仓库及许可](https://github.com/gugushvili/predator_prey) · [原数字化 CSV](https://raw.githubusercontent.com/gugushvili/predator_prey/master/gause_fig3.csv)

Mühlbauer 等的研究给出了另一组草履虫的经典竞争数据与 R 流程。Figure 2 用 `get_logistic` 拟合单培养，Figure 3 用 `lv_interaction` 画混培养的竞争曲线。作者的方法先用相邻时刻的观察值估计个体平均增速，再回归增速与两种群丰度的关系。需要时把得到的参数作为 ODE 优化的起点。Table 2(a,b) 有估计值和一倍近似标准误，互作矩阵的第 $i$ 行第 $j$ 列对应 $j$ 对 $i$ 的作用。复跑原 R 示例可在本机 R 中安装 `gauseR`，载入 `data("gause_1934_science_f02_03", package="gauseR")`，对照 `help("get_logistic", package="gauseR")` 与 `help("lv_interaction", package="gauseR")` 的 Examples。该 R 路线未在本机执行，本章 Python 执行的是下文的 Gause 捕食数据和独立模拟竞争情景。 [gauseR 原论文 Figure 2–3 和 Table 2](https://doi.org/10.1002/ece3.6926) · [作者 R 源码、数据说明](https://github.com/adamtclark/gauseR)

### 跑本章的最小拟合

1. 将两条时间序列分别除以各自在训练期的最大值，避免量尺较大的序列主导目标函数。以 day 0 的实测值设初值。
2. 用 day 1–11 的点拟合 $a,b,c,d$。每次试参数先调用 `solve_ivp`，再将两种群在观测时间的归一化预测减去数据，用有正值边界的 `least_squares`，比较三组初始猜测。
3. 保持最佳参数不动，积分到 day 19。将结果换回两个种群各自原尺度，分别在训练期和 day 13–19 留出期算 RMSE。画时间曲线与相图。
4. 观察独立的模拟竞争面板。增大 B 对 A 的竞争压力后，两者共存变为 B 排除 A。对照 B 在承载量附近时 A 的入侵增长率 $r_A(1-\alpha_{BA})$，它在两个示范系数 $0.45$ 和 $1.3$ 下有相反符号。

`dynamics-community.png` 的真实数据部分，本机训练 RMSE 约为酵母 $14.05$、草履虫 $20.63$，留出期约为 $68.40$ 与 $77.37$，各自沿用原图的密度标度。查看留出期的峰值误差，解释为什么捕捉到训练期的相位关系仍可能外推失准。这个实例的替代模型可显式表示稀释、猎物容量或捕食饱和，再用留出时段判断改进是否稳定。只有相对丰度时也应考虑总生物量是否可观测，避免把相对比例的变化直接解释为个体数量变化。

## 5. 如何挑求解器，确认结果来自模型

iGEM 2025 ZJU-China 在藻菌共培养模块中把共培养拆为藻生物量、菌生物量和糖三个状态，利用 Monod 描述糖与生长的关系，并在网页 JavaScript 中手写 RK4 供交互演示。该文档 §4.3.4 列出每一步实现，§4.2 明示其校准数据表省略。首页与优化小节对高级版算法的表述存在差异，本章读取基础版 RK4 步骤。若只重写求解器，任何小系统都可当输入，因此用可直接算真解的衰减问题来检查，再应用到藻类或竞争模型。 [ZJU-China 原 Wiki](https://2025.igem.wiki/zju-china/model) · [团队原技术文档，§4.2–4.4](https://static.igem.wiki/teams/5869/model/mkm.pdf)

- **显式 Euler。** iGEM 2023 Washington 用它逐步推进氮浓度驱动的生物膜厚度模型。作者先从文献构造反应与物料平衡，再外借其他蓝藻参数得到情景曲线。若要评估可信度，可先用本章 `fixed_step` 对一条增长曲线减半步长，再独立讨论外借参数的偏差。原团队没有说明该模块的具体软件或算力配置。 [Washington 原 Wiki 的模型总结](https://2023.igem.wiki/washington/model)
- **Heun。** Bernard 等先将微藻跑道池分层以近似水流和生物反应，再用经典 Heun 使时间离散达到二阶。他们在 §4.1 先构造存在解析解的简化耦合问题检查稳态，再模拟搅拌与静水等情景。本章从这个设计借用“先验算，再扩大系统”的实验顺序，完整二维求解还需要空间离散。 [Bernard 等原论文，§3.3.5、§4](https://www.numdam.org/item/10.1051/m2an/2013072.pdf)
- **后向 Euler 与隐式梯形。** 房水生成研究把跨膜组分和体积写成低维动力学，用 MATLAB 比较两种 $\theta$ 方法及现成刚性求解器。$\theta=1$ 是后向 Euler，$\theta=0.5$ 对应隐式梯形。求解步骤和比较任务见论文的 Numerical Approximation 部分。本章复跑两种更新方式在有真解的衰减方程上的行为，便于看清稳定性与误差的区别。 [细胞体积原研究](https://www.mdpi.com/2297-8747/30/1/13)

| 方法 | 每一步如何决定 | 平滑问题总体误差常见阶数 | 适合做什么 |
| --- | --- | --- | --- |
| 显式 Euler | 用当前斜率走一步 | 一阶 | 手算与检验模型符号 |
| 后向 Euler | 用下一时刻的未知斜率，解隐式方程 | 一阶 | 演示快衰减对稳定性的影响 |
| 隐式梯形 | 平均当前和下一时刻的未知斜率，解隐式方程 | 二阶 | 平滑系统，在可接受隐式求解开销时保持二阶精度 |
| Heun／改进 Euler | 先用显式 Euler 试走一步，平均当前和试走得到的斜率 | 二阶 | 入门和固定步长对照 |
| 经典 RK4 | 在起点、两个中间预测点、终点取四个斜率并加权 | 四阶 | 平滑非刚性过程和前端固定步长仿真 |

跑 `--lab solvers`。用 $y'=-y$、$y(0)=1$ 测 $t=4$ 的末值误差，依次把步长减半，终端的估计阶数约 $0.98, 2.05, 4.05$。再看右图 $y'=-20y$ 取 $h=0.2$，显式 Euler 一步乘 $1-20h=-3$，幅值反而增长。后向 Euler 和隐式梯形的标量更新因子可直接解出，因此代码用解析更新来画图。手写 `fixed_step` 可将 RHS 换成第四节的竞争系统，重复步长收敛检查。

**选型。** 固定步长 RK4 仍方便透明地控制浏览器帧步或对齐简单模型，发布结果时应报告所用步长并确认减半后结论一致。平滑非刚性应用通常先用自适应 SciPy `solve_ivp` 的 RK45，刚性时可比较 Radau 或 BDF 并检查容差。MATLAB 的 `ode45` 是自适应 Dormand–Prince 4/5 阶对，本章的固定步长 RK4 有独立实现。旧讲义第 10 页 MATLAB 小节是预留位置，其余练习已在 Python 可运行。 [SciPy 求解器文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) · [MathWorks ode45 文档](https://www.mathworks.com/help/matlab/ref/ode45.html)

## 6. 把公式和结果写进 Wiki

本仓库的 GitHub Markdown 可直接写行内 `$v=V_{max}S/(K_m+S)$`，独立公式写在成对的 `$$` 之间。下方写公式的状态、变量、单位和来源。比赛 Wiki 属于另一套发布环境。iGEM 模板要求承载 Wiki 的代码放在团队仓库，外部 CDN 资源按该模板说明改用平台托管。使用 MathJax 时可将需要的组件按比赛当年的规则本地托管，并在提交预览中检查公式和字体。 [GitHub 数学表达式文档](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions) · [iGEM 官方 Wiki 模板](https://gitlab.igem.org/templates/wiki-frozen-flask-markdown/-/blob/main/README.md) · [MathJax 自托管文档](https://docs.mathjax.org/en/latest/web/hosting.html)

提交练习时带上三样内容：你改变的输入或模型假设、对应的输出图和数值、一个可失败的验收条件。例子是把捕食训练结束日改为第 9 天，重算留出期误差，并检查增加预测提前量是否加大相位误差。与这次原始留出结果并列可以看出数据划分的影响。

案例检索与核对日期为 2026-11-23。查询围绕 `site:igem.wiki Michaelis-Menten model`、`Monod microalgae growth iGEM`、`Hill expression iGEM`、`BoolODE BEELINE`、`Contois Chlorella growth kinetics` 与 `Gause 1935 Figure 3 digitized`，优先使用团队公开 Wiki、作者论文与公开代码。在选例时优先保留能定位到图、表和公式的模型。
