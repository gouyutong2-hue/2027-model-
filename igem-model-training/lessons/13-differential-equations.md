# 13 微分方程建模：从变化率到项目决策

学完这一课，你可以写出一个状态随时间变化的模型，用 Python 求解，做一次守恒或解析解检查，再用留出的数据检验模型。四个案例分别回答实际问题：补料后发酵罐里有多少生物量，怎么比较两条生长曲线，混合培养为什么不能把两条单培养曲线简单叠在一起，以及怎么用体积和质量推算微藻的理论沉降速度。

| 原讲义位置 | 要学会的数学方法 | 本课案例与可复现程度 |
|---|---|---|
| §1，印刷页 265–269 | 直接列方程、处理变化的总量、分阶段求解 | iGEM Cornell 2022。迁移原四条质量衡算式，本课自定输入 |
| §2，印刷页 270–272 | Malthus 指数、Logistic、平衡点、非线性回归 | iGEM Fudan 2025。用队伍公布的 BY4741 参数重画拟合线，再用模拟观测练习拟合 |
| §3，印刷页 272–277 | 联立状态、区分线性和乘积项、从时间序列验证 | Gause 草履虫竞争与 Mühlbauer 等的再分析。原论文数字化数据、表中参数近似复算和独立的训练/留出检验 |
| §4，印刷页 277–279 | 力平衡、线性和平方阻力、阈值事件 | Miettinen 等的天然微藻研究。复算 Stokes 关系，并用教学参数求速度达到阈值的时刻 |

页码按《13第十三章 微分方程建模.pdf》的印刷页标注。第 1、3、4 部分采用应用问题不同、数学技能对应的案例。讲义中的战争例题在这里对应生态竞争教学，应用背景与参数均来自生态学研究。

## 先跑一次

在课程目录执行。文档按初次运行约 60–90 分钟的练习量组织，电脑计算只需几秒。首次运行按仓库 [README](../README.md) 配置 Python。脚本依赖 NumPy、SciPy、Matplotlib，普通 Windows、macOS 或 Linux 笔记本 CPU 足够。装好依赖后，脚本和所需 CSV 都能离线运行。

```bash
python examples/differential_equations.py --outdir build/13-differential-equations
```

打开输出文件夹，把 `.png` 拖到图片查看器，把 `.csv` 用表格软件打开。如果没有 `python` 命令，macOS 和 Linux 可试 `python3`，Windows 可试 `py -3`。四个案例分别对应脚本中的 `fed_batch`、`yeast_growth`、`competition` 和 `sinking`。[教学脚本](../examples/differential_equations.py)

### 先写状态，再求解

1. 写问题和读数。例如「我要比较总生物量」，同时确认仪器是否只读到了浓度。
2. 定义状态，就是方程追踪的未知量。把自变量、初值和每一项的单位列出来。写出「进入 − 离开 + 产生 − 消耗」，或从平衡关系、数据假设写出变化率。
3. 把公式右端写成 `rhs(t, y)`。例如 `y=[X,P,S,V]`，函数输出对应的四个变化率。给求解器起止时间和初值，就能算出一串轨迹。
4. 交叉检查。优先用守恒关系、已知解析解、极端输入或单位找出错误。有实验数据时，另留后期时间点或独立批次做检验。小时间步 $\Delta t$ 下把 $y$ 加上 $\Delta t f(t,y)$ 是欧拉法。脚本用 `solve_ivp` 的默认自适应 RK45 控制步长，用 `t_eval` 指定输出时间。刚性问题可考虑 Radau 或 BDF，先对比误差容限和结果再选。[SciPy 初值问题求解器文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)

## 13.1 Cornell 2022：补料罐中，总量和浓度可以走向不同方向

**队伍的问题和做法。** Cornell 对色蛋白项目的补料分批反应器作模拟，想把细胞、产物和底物浓度随时间的变化联系起来。Wiki 的 **Bioreactor Modeling** 从质量衡算列出式 (1)–(4)，再用乘积求导转成浓度形式式 (1a)–(4a)。底物多时生长速率趋近上限，底物少时生长受限。队伍用 Python、NumPy、SciPy、Matplotlib 和 Bokeh 展示 Figure 1 的模拟。下一节的色蛋白表达模型明确用 SciPy 的 `odeint`，并报告缺少新产物的产率 $Y_{P/X}$。Figure 1 的完整起始条件、流量时程和运行脚本未在该模型页找到。计算只涉及少数状态，笔记本即可。[Cornell 模型原页，Bioreactor Modeling 与 Chromoprotein Expression Modeling](https://2022.igem.wiki/cornell/model)

**为什么先写浓度乘体积。** 设 $X,P,S$ 分别为生物量、产物、底物浓度，三个量乘 $V$ 才是罐内总量。本课统一采用 g/L 来表达三种浓度，以匹配用质量比表示的产率。Wiki 对 $X$ 的解释写成 cells/L，照此口径计算时，要把产率定义为与细胞数相容的单位，或先独立把细胞数转换成生物量。每小时加入 $F$ L 液体，底物进料浓度为 $S_f$。无出料的理想罐里，用 Monod 关系 $r_g=\mu_{\max}SX/(K_S+S)$ 表示单位体积的生物量增长率：

$$
\frac{d(XV)}{dt}=Vr_g,\qquad
\frac{d(PV)}{dt}=VY_{P/X}r_g,\qquad
\frac{d(SV)}{dt}=FS_f-\frac{Vr_g}{Y_{X/S}},\qquad
\frac{dV}{dt}=F.
$$

例如将第一式左端换成 $V\dot X+X\dot V$，得到 $\dot X=r_g-(F/V)X$。新流入液体会把物质稀释，所以即使 $XV$ 增加，$X$ 也可能下降。补料切换点前后的速率可跳变。把上一段末的状态作为下一段初值即可让状态连续。

**跟着复现。** 在 `fed_batch` 找到下表各值，先照跑一遍，再每次只改一个值。所有参数是计算教学设定，表示一个通用过程。把 [脚本](../examples/differential_equations.py) 输出的 `cornell-fed-batch.csv` 与 `cornell-fed-batch.png` 对照。

| 输入 | 教学数值 | 意义 |
|---|---:|---|
| $X(0),P(0),S(0),V(0)$ | 0.20 g/L，0 g/L，4.0 g/L，1.0 L | 起始浓度与体积 |
| $\mu_{\max},K_S$ | 0.44 h⁻¹，0.45 g/L | 最大比生长率、半饱和底物浓度 |
| $Y_{X/S},Y_{P/X}$ | 0.50 g 生物量/g 底物，0.15 g 产物/g 生物量 | 模拟用质量产率 |
| $S_f$ 与 $F$ | 15.0 g/L，8–14 h 为 0.45 L/h，其余时间为 0 | 教学分段输入 |

1. 找到图中 8 h 的切换点。8 h 到 8.05 h，$X$ 从约 2.20 降到 2.16 g/L，$XV$ 从约 2.20 升至 2.21 g。解释这两个变化为什么能同时发生。
2. 检查 20 h 的体积为 $1+0.45\times 6=3.70$ L，总生物量约 22.45 g，产物约 3.34 g。用「期初底物总量 + 累计进料底物 − 当前底物总量」得到已消耗底物，再乘 $Y_{X/S}$ 应等于新增生物量。代码检查了完整时间序列的这项守恒。
3. 把补料段推迟两小时而保持累计补料体积不变，比较 8–20 h 的曲线和终点。可以产生相似的终点却经历不同的中间状态。

**评价。** 质量衡算加 Monod 依然适合把工艺假设转成可算的模型。原案例最值得借鉴的是从守恒推出稀释项、并公开生产率缺口。真实过程如果受溶氧、pH 或维持能耗约束，先看残差随这些变量是否有规律，再按数据增加状态。参数多时需要专门的时间序列，否则仅有一个终产物读数很难同时估计初始生物量、增长率和产率。[Cornell 模型及团队对改进的讨论](https://2022.igem.wiki/cornell/model)

## 13.2 Fudan 2025：为什么酵母的指数增长预测会失效

**队伍的问题和做法。** Fudan 对酵母读出系统比较荧光表现与对生长的影响。他们记录每小时的光密度 OD₆₀₀，用 R 的 `nls` 拟合自启始 Logistic 模型 `SSlogis`，读取平台值、拐点和曲线的时间尺度。随后将生长表现与读出性能一同用于比较。Figure 3 和 Table 4 提供结果。供本课复算的是野生型酿酒酵母 BY4741 对照。团队从光密度用了基于纳米颗粒的近似换算，因此按他们报告的密度代理单位画图，绝对细胞计数需另做测量校准。模型页的增长小节给了汇总参数，逐小时的对照原始观测和拟合不确定性未在该小节找到。[Fudan Measurement，Growth Curve 小节，Figure 3、Table 4](https://2025.igem.wiki/fudan/measurement/)

**算法。** Malthus 模型 $\dot N=rN$ 假定单位个体的增长率恒定，解为 $N(t)=N_0e^{rt}$。Logistic 用 $\dot N=rN(1-N/K)$ 让单位个体增长率随密度上升而降。它也可写成讲义中的 $\dot N=aN-bN^2$，其中 $a=r$，$b=r/K$。在 $r,K>0$ 且初值大于零时，解向稳定平衡 $K$ 靠近。$N=K/2$ 时总增长速度最大，值为 $rK/4$。R 的形式是

$$N(t)=\frac{K}{1+\exp(-(t-t_{\rm mid})/s)}$$

其中 $s=1/r$ 的单位是小时。把 $t_{\rm mid}$ 看成达到半平台所需的时间，有助于区分「何时长得最快」和「那时每小时能长多少」。[R 官方 SSlogis 定义](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/SSlogis.html)

| 输入或量 | 值 | 来源 |
|---|---:|---|
| $K$ | 3.67，单位为队伍报告的 $10^8$/mL 密度代理 | BY4741，Table 4 |
| $t_{\rm mid}$ | 5.76 h | BY4741，Table 4 |
| $s$ | 1.3649 h | BY4741，Table 4 |
| $r$，$N(0)$ | 0.7327 h⁻¹，0.0532（同 $K$ 的单位） | 本课根据上述四舍五入参数换算 |
| $R^2$ | 0.9953 | 队伍 Table 4 的拟合优度，复算线无需用它 |

**跟着复现。** `fudan-growth.png` 的实线是公布参数的重新计算。脚本先从解析曲线读出 $N(0)$，再在同一初值下用 RK45 求解，断言两条结果重合。在程序 `yeast_growth` 中，`start` 就是该初值。另外脚本用固定随机种子给表中拟合线加小噪声，产生 0–16 h 的**模拟观测**。用 `curve_fit` 在正的参数范围内最小化平方残差，得到 $K\approx3.674$、$t_{\rm mid}\approx5.741$ h、$s\approx1.380$ h。虚线是该教学拟合，黑点为模拟观测。点线用模拟观测前五点（0–4 h）在对数尺度上拟合早期指数；后段脱离显示范围，可在 CSV 看到具体数值。这是生长期外推的反例，Logistic 用的是全时间段，两条线的拟合误差不宜当作公平的模型排名。[教学脚本](../examples/differential_equations.py)

**练习与评价。** 只拿早期数据时，试着把 $K$ 改成 5 或 10，再调整 $N_0$ 与 $r$，观察早期吻合程度和最终预测的差别。因为此时 $N\ll K$，指数足以近似局部增长，数据很难约束 $K$。Logistic 适合资源不变、曲线有清楚平台且整体近似对称的经验比较。观测到明显滞后或上下不对称时，可以比较 Gompertz 或带滞后项的增长模型，对多次完整实验分批留出再挑模型。有底物时间序列时可尝试上节的资源限制模型。较高的 $R^2$ 只描述相应数据上的拟合效果。[Fudan 原测量页](https://2025.igem.wiki/fudan/measurement/) · [Zwietering 等对不同生长曲线的实证比较](https://pmc.ncbi.nlm.nih.gov/articles/PMC184525/)

## 13.3 Gause 草履虫竞争：两条时间曲线怎样互相牵制

**作者的问题和做法。** Gause 对比两种草履虫的单独培养和混合培养。Mühlbauer、Schulze、Harpole 和 Clark 用 R 开源包 `gauseR` 整理数据，并以该实验展示 Lotka–Volterra 竞争方程。先根据相邻时间点的体积指数估算人均增长率 $[\log N(t_2)-\log N(t_1)]/(t_2-t_1)$，再用两种草履虫各自的体积指数做线性回归，估计自限和交叉作用。其通用流程还可用数值 ODE 模拟和优化细调系数。公开的 `lv_interaction` 示例展示从原图数字化数据、回归到 `deSolve::ode` 的代码。论文给出 Figure 3 和 Table 2(b)。用 R 按帮助运行 `example("lv_interaction", package="gauseR")` 可查看作者提供的示例，它的初始化与论文表中估计值可以不同。两维 ODE 和一次小回归无需租用资源。[Gause 1934 原研究 DOI](https://doi.org/10.1126/science.79.2036.16-a) · [Mühlbauer 等论文 Eq. 1–2、Figure 3、Table 2(b)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7713957/) · [gauseR 官方包示例](https://search.r-project.org/CRAN/refmans/gauseR/html/lv_interaction.html) · [作者源码](https://github.com/adamtclark/gauseR)

**算法。** 令 $N_i$ 分别是 *P. caudatum* 和 *P. aurelia* 的体积指数，$r_i$ 是低密度且无对方时的净增长率。系数 $a_{ij}$ 表示第 $j$ 种对第 $i$ 种单位个体增长率的影响。因此 $a_{ij}<0$ 可以用于表示竞争。两条式子要同时求解，因为当前 $N_1$ 会改变第二条的速度，当前 $N_2$ 会改变第一条的速度。

$$\dot N_1=N_1(r_1+a_{11}N_1+a_{12}N_2),\qquad
\dot N_2=N_2(r_2+a_{21}N_1+a_{22}N_2).$$

当 $N_2=0$ 时第一式退回单物种 Logistic。在 CSV 里，时间是日，体积是用于比较两种生物大小的标准化指数，无绝对体积单位。由原论文 Fig. 2–3 数字化的 `gause_1934_science_f02_03` 中，本课抽取 Treatment 为 Mixture 的 23 个时间点（第 2–24 天）。CSV 中的四位小数保留作者数据文件中的精度，原图读数的不确定性未随此数据提供，无法据小数位推断测量精度。[数据字典](https://search.r-project.org/CRAN/refmans/gauseR/html/gause_1934_science_f02_03.html) · [作者数据 JSON](https://adamtclark.r-universe.dev/gauseR/data/gause_1934_science_f02_03/json) · [仓库原图数字化子集](../data/gause-competition-mixture.csv)

**跟着复现。** `gause-competition.png` 灰虚线代入 Table 2(b) 的初值 1.60、1.63，$r_1=1.26$、$r_2=1.03$（日⁻¹）与交互矩阵各行 $(-0.005,-0.008)$、$(-0.002,-0.007)$。该线从第 0 天数值积分后按观测时刻叠图，用来重算论文表中四舍五入系数给出的竞争趋势。论文表中的约数和这里采用的时间起点会影响曲线，原 R 示例是进一步比对的入口。蓝线是本课重新拟合版本，将首个混合培养观测设为第 2 天初值，拟合第 2–17 天的两个物种体积指数，规定竞争系数非正，最小化观察值与 ODE 模拟值的平方差。点线使用相同训练时段，固定交叉项为零作对照。空心点对应从未参与拟合的第 18–24 天。

| 第 18–24 天 RMSE，单位同体积指数 | 有交叉项 | 交叉项固定为 0 |
|---|---:|---:|
| *P. caudatum* | 9.78 | 40.63 |
| *P. aurelia* | 10.75 | 9.59 |

**练习与评价。** 两个物种的留出结果不同：当前这批数据中，交叉项明显改善了 *P. caudatum* 的后段预测，*P. aurelia* 用简单模型略好。可把 `fit_through_day` 从 17 改成 14，重新比较。经典竞争 ODE 适合作为短期、双种群、受控环境的可解释基线。论文给出的部分交互项标准误大于点估计绝对值，因此用一次混合培养轨迹难以精确判断相互作用强度。加上不同初始比例的独立时间序列，或同时拟合单培养和混合培养，能让自限与交叉作用更容易区分。遇到共享资源显著波动时，可将资源浓度也写成状态；计数噪声大时，考虑把测量误差单独写进状态空间模型。[Mühlbauer 等 Table 2、Discussion](https://pmc.ncbi.nlm.nih.gov/articles/PMC7713957/)

## 13.4 天然微藻沉降：把力平衡写成可计算的模型

**作者的问题和做法。** Miettinen 等想知道海洋真核微藻的细胞大小、密度怎样影响重力沉降。研究者先用悬浮微通道谐振器（SMR）测浮力质量，用荧光排阻或 Coulter 计数器获得体积，再从这两项计算密度、等效球半径和 Stokes 终端速度。文中的仪器 SMR 信号分析用了自编 MATLAB 代码，沉降公式本身只需表格或几行 Python。Figure 1C 对 *Tetraselmis* sp. 给出代表性的 1,027 个单细胞分布，摘要统计来自 5 个独立培养。Figure 1D 是跨物种展示，Figure 2B 的跨物种回归在 17 个种的尺度上给出体积指数约 $0.73\pm0.04$（估计值 ± 标准误）。这是分析由质量和体积推出的理论重力沉降速度。研究中提到的主动游动是另外一种贡献。[Miettinen 等 2024，Eqs. 1–2、Figures 1–2、Materials and Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC11225777/)

**算法。** 向下为正。$m_b=V(\rho_{\rm cell}-\rho_{\rm water})$ 是浮力质量，代表抵消浮力后还剩下的有效重量除以 $g$。对近似球形、低雷诺数、静水环境中的细胞，用线性 Stokes 阻力 $kv$，其中 $k=6\pi\mu R$。在给速度加上惯性的课堂模型里，$m$ 是细胞质量、$z$ 是深度，因而

$$
\dot z=v,\qquad m\dot v=m_bg-kv,\qquad
v_\infty=\frac{m_bg}{k}=\frac{2g(\rho_{\rm cell}-\rho_{\rm water})R^2}{9\mu}.
$$

从静止出发，$v(t)=v_\infty(1-e^{-t/\tau})$，其中 $\tau=m/k$。在固定密度差下 $R\propto V^{1/3}$，所以 $v_\infty\propto V^{2/3}$。另一种阻力近似是 $-k_2v|v|$，用于物理条件处在不同惯性区间的对照。本课把两者按各自的时间和速度尺度画在一张图里作数学对比。$k$ 的单位为 kg/s，$k_2$ 的单位为 kg/m，不能直接用同一个数值代表两者。对近似微米级藻细胞，先通过 $Re=\rho_{\rm water}|v|\,(2R)/\mu$ 核验低雷诺数假设。[原文 Eq. 1 与 Sinking velocity calculations](https://pmc.ncbi.nlm.nih.gov/articles/PMC11225777/)

**跟着复现。** `algae-sinking.png` 左图比较线性和平方阻力的无量纲曲线，右图复算固定密度差的理论 $2/3$ 斜率。脚本采用论文海水参数 $\rho_{\rm water}=1020$ kg/m³ 和 $\mu=1.07\times10^{-3}$ Pa·s，用教学球半径 5 μm、浮力质量 40 pg，算出终端速度约 3.891 μm/s、$Re\approx3.71\times10^{-5}$。事件函数在 $v/v_\infty-0.95=0$ 时触发，本简化模型给出的时间约 17.05 μs。按终端速度估算沉降 1 mm 约 257 秒，远长于达到阈值所需的时间。论文报告的实证斜率和右图理论值并排阅读。补充数据包在本次检索中未能核对单细胞表，因此本地脚本复算方程和理论斜率，微藻输入是明确标注的教学量。[论文原始补充数据入口](https://pmc.ncbi.nlm.nih.gov/articles/PMC11225777/) · [教学脚本](../examples/differential_equations.py)

**练习与评价。** 将细胞半径变为两倍，同时保持密度差恒定，代码里须让浮力质量变为八倍。先用公式猜终端速度，再运行程序检查。维持低雷诺数边界内时，四倍预测成立。真实沉降还可受形状、壁面、细胞聚集、游动和流动影响。Stokes 关系适合先估算纯重力沉降的量级。若目标是解释实时轨迹，需要额外轨迹观测并区分重力分量与水体流动。读原始数据时，可从 Dataset S1–S2 选择同一独立培养的浮力质量和体积，按论文各行的单位换成 SI，再逐样本计算，再查 Figure 1 对应的统计口径。[Miettinen 等 Figure 1 与补充材料](https://pmc.ncbi.nlm.nih.gov/articles/PMC11225777/)

## 小作业

- **建立方程。** 为一个输入率随时间变化的理想容器定义体积 $V$、物质浓度 $C$，写出 $d(VC)/dt$，再用乘积法则反推 $dC/dt$。检查式子中每项的单位。参考 13.1 的拆分。
- **给三个模型一个相同检验。** 用同一组模拟酵母时间点，比较训练截止时间变化时指数、Logistic 和选做的 Gompertz 拟合。三者用同一训练点、同一测试点和同一误差指标。
- **解释失败。** 挑一张生态竞争曲线，在报告里分别写一条「可能的真实机制偏差」、一条「测量上的不确定性」和一条「最能区分二者的新观测」。
- **检查单位。** 用 $1\text{ pg}=10^{-15}\text{ kg}$、$1\text{ fL}=10^{-18}\text{ m}^3$ 重新计算 13.4 的速度，解释为什么将 pg 当作 ng 会使式中估计的速度差三个数量级。

## 来源筛选记录

2026-11-23 查阅 iGEM 原队伍 Wiki、PubMed Central、文章附表和 R 官方包文档。检索词包含 `iGEM fed-batch bioreactor mass balance`、`iGEM yeast Logistic growth SSlogis`、`Gause Paramecium competition Lotka Volterra gauseR` 和 `phytoplankton sinking velocity Stokes buoyant mass primary`。优先采用能接到公式、公开参数或数据的项目。BGI-MammothEdu-East 2024 有酵母 Logistic 曲线与重复，Fudan 的独立对照参数表更适合本课现场复算。Cornell 的公开质量衡算式最适合分段补料教学，公开信息足以复算方程但不足以锁定其 Figure 1。Gause 原始论文出版页本次返回访问限制，使用作者再分析和数据集元数据核对该图的数字化来源。[备选 iGEM 模型页](https://2024.igem.wiki/bgi-mammothedu-east/model) · [Fudan 参数表](https://2025.igem.wiki/fudan/measurement/) · [gauseR 资料及代码存档](https://doi.org/10.5281/zenodo.4058426)
