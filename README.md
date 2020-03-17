# 第三届融360智能金融算法大赛“笔架山下小书童”团队

多金融场景下的模型训练 赛道0.7793（22th)
 
## 一、赛题介绍

融360平台积累了大量不同时期、不同金额、不同期限、不同利率、不同市场环境下的借贷订单。在风控建模过程中，如何选择最合适的样本针对当前市场环境下特定金融产品建模。

训练样本：包括从2017.4.1到2018.5.1不同金额、不同期限、不同利率的金融产品样本，并给出每个样本的类型（属于大额分期贷或小额现金贷产品）是否逾期。（约10万样本） 

验证样本：2018.1.1到2018.5.1机构A的产品，验证集不提供样本是否逾期，参赛选手自行完成是否逾期预测后，可以提交至比赛平台评估结果。（约2万样本） 

测试样本：与验证样本来源相同且同分布。测试集不提供样本是否逾期，参赛选手只能在比赛最后的评比阶段将预测结果提交至比赛平台评估，且只能提交一次。（约2万样本）

## 二、特征工程

（1）缺失值处理

删除缺失值大的样本

（2）构建特征

原始特征；
排序特征：对数值特征进行排序；
离散特征：对连续特征进行分区间离散化； 
统计特征：统计最大值、最小值、平均值、标准差、四分位数；

（3）特征选择

使用方差最小化、互信息、person 系数法

## 三、模型

使用lightgbm五折交叉验证以及DNN模型，两者加权融合。
