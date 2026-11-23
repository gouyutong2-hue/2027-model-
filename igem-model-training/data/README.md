# 课程随附数据

| 文件 | 来源与证据位置 | 内容和单位 |
|---|---|---|
| [fujian_phytase_activity.csv](fujian_phytase_activity.csv) | Fujian-United 2023 Model PDF，§3 两段 MATLAB 数值数组。见[原报告](https://static.igem.wiki/teams/4870/wiki/model.pdf) | 15 个模型输入位置，温度为 °C，pH 无量纲，输出是队伍报告的相对活性。三次平行测量原表在报告 §2 |
| [beal_2018_s1_bulk.csv](beal_2018_s1_bulk.csv) | Beal 等，2018，S1 Table。见[论文](https://doi.org/10.1371/journal.pone.0199432)和[原表](https://doi.org/10.1371/journal.pone.0199432.s006) | 65 个板读数据集的校准和输出摘要。原 CSV 按 CC BY 保留，列名附单位或统计说明。文件 SHA-256 为 `440ed208e11d933aacf66fc660144f93ccce14f52b4fb3b565bc73736f92281c` |
| [gause-1935-figure3.csv](gause-1935-figure3.csv) | Gause 等 1935 Figure 3，Mühlbauer 等 `gauseR` 作者数字化。见[代码与数据仓库](https://github.com/adamtclark/gauseR)和[存档](https://doi.org/10.5281/zenodo.4058426) | 36 条数字化的酵母与草履虫时间序列，字段按作者数据。数字化误差没有报告 |
| [gause-competition-mixture.csv](gause-competition-mixture.csv) | Gause 1934 Figure 2–3，Mühlbauer 等 `gause_1934_science_f02_03` 的 Mixture 子集。见[数据字典](https://search.r-project.org/CRAN/refmans/gauseR/html/gause_1934_science_f02_03.html)和[作者数据](https://adamtclark.r-universe.dev/gauseR/data/gause_1934_science_f02_03/json) | 23 个混合培养时间点，时间为天，两个输出是两种草履虫的体积指数 |

两组 Gause 数据经数字化整理，论文的数据可用性说明指向 CC0 存档。Fujian 表只含团队公开数值的教学转录。其他小型输入在对应脚本中声明，无需另找文件。模拟数据由脚本生成。
