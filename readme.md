# Experimentation(如何运行该程序)

本项目需要的库包括numpy, pandas, sklearn, math, zipfile, urllib3, xgboost, lightgbm, catboost, os, itertools.
运行本程序可能需要50分钟，如果您有GPU，请将第164行的task_type='GPU'反注释。本程序将会从kaggle网站上自行下载所需要的数据集，如果下载地址变动导致下载错误，请前往网页下载，并解压所有CSV文件到py同目录下。

如果您需要查看本项目各个模型以及stacking后的rmse，请将72-100行的used for parameters selection部分、172-174行的fitting and check the R_MSE of each model部分、176-184行的test stacking performance部分的注释取消。这样就可以看到rmse，但是也会增加很多运行时间。



# 预测结果分析

当程序运行结束后，在同文件夹下，可以看到有一个submission.csv文件。这即为我们预测的结果。stacking后的rmse为0.91825，我们的成绩是kaggle前26.55%（2173/8183）名。值得注意的是，在一个时长为5年的项目里，排名20%的选手中，不少代码都是same。而我们在几周时间里，完全自己学习与运用的情况下还能得到这个成绩，应该是值得肯定的。

打开submission.csv，您可以看到一栏“ID”与一栏“item_cnt_month”。item_cnt_month即为我们最后得到的预测值。从数据详情中，我们可以看到，数据集的平均值在0.3134上下（附图）。因为通过观察原始数据集中的预测值，我们发现合理的最大值是在20左右，而价格的最小值基本上不可能为负数，所以说在算法中，对于最小值和最大值都做出了界定，取值范围为[0, 20]。从jupyter notebook的频率直方图中（附图），我们也可以发现，预测的价格基本上都在[0, 2.5]的区间内，所以说我们对于最大值的界定增加了数据的可信度。

