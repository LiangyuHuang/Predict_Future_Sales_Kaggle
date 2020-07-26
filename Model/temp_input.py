#import some necessary librairies

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
def input_data():
    color = sns.color_palette()
    sns.set_style('darkgrid')
    import warnings
    def ignore_warn(*args, **kwargs):
        pass
    # ignore annoying warning (from sklearn and seaborn)
    warnings.warn = ignore_warn
    from sklearn.preprocessing import LabelEncoder
    from scipy.stats import norm, skew #for some statistics

    # 设置输出的浮点数的格式
    pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

    from subprocess import check_output
    # check the files available in the directory
    # print(check_output(["ls", "/Users/liudong/Desktop/house_price/train.csv"]).decode("utf8"))
    # 加载数据
    train = pd.read_csv('/Users/liangyu/Documents/GitHub/Predict_Future_Sales_Kaggle/Data/train.csv')
    test = pd.read_csv('/Users/liangyu/Documents/GitHub/Predict_Future_Sales_Kaggle/Data/test.csv')
    # 查看训练数据的特征
    # print(train.head(5))
    # 查看测试数据的特征
    # print(test.head(5))

    # 查看未删除ID之前数据的shape （1460， 81） （1459， 80）
    # print("The train data size before dropping Id feature is : {} ".format(train.shape))
    # print("The test data size before dropping Id feature is : {} ".format(test.shape))

    # 保存 Id列的值
    train_ID = train['Id']
    test_ID = test['Id']

    # 删除ID列的值，因为它对于预测结果没有太大的影响
    train.drop("Id", axis = 1, inplace = True)
    test.drop("Id", axis = 1, inplace = True)

    # 检查删除ID以后的数据的shape （1460， 80） （1459， 79）
    # print("\nThe train data size after dropping Id feature is : {} ".format(train.shape))
    # print("The test data size after dropping Id feature is : {} ".format(test.shape))

    # 删除那些异常数据值   异常值的处理对应于那些极端情况
    '''
    fig, ax = plt.subplots()
    ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
    plt.ylabel('SalePrice', fontsize=13)
    plt.xlabel('GrLivArea', fontsize=13)
    plt.show()
    这里举的是异常值的处理  Example：房子面积很大，但是价格很低 其他的这种异常并不是需要全部删除
    还要保持模型的健壮性
    '''
    train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

    # 对SalePrice使用log(1+x)的形式来处理
    train["SalePrice"] = np.log1p(train["SalePrice"])


    # 特征工程
    # 将train数据集和test数据集结合在一起
    # ntrain ntest 各自行数
    ntrain = train.shape[0]
    ntest = test.shape[0]
    y_train = train.SalePrice.values
    # contact连接以后index只是重复，会出现逻辑错误  需要使用reset_index处理 drop为True
    all_data = pd.concat((train, test)).reset_index(drop=True)
    all_data.drop(['SalePrice'], axis=1, inplace=True)
    # print("all_data size is : {}".format(all_data.shape))

    # 特征工程  对特征进行处理
    # 处理缺失数据
    # print(all_data.isnull().sum())
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
    # print(missing_data.head(20))

    all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
    all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
    all_data["Alley"] = all_data["Alley"].fillna("None")
    all_data["Fence"] = all_data["Fence"].fillna("None")
    all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

    # Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        all_data[col] = all_data[col].fillna('None')
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        all_data[col] = all_data[col].fillna(0)
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        all_data[col] = all_data[col].fillna(0)
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        all_data[col] = all_data[col].fillna('None')
    all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
    all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
    # all records are "AllPub", except for one "NoSeWa" and 2 NA  remove
    all_data = all_data.drop(['Utilities'], axis=1)
    all_data["Functional"] = all_data["Functional"].fillna("Typ")
    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
    all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
    all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
    all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
    all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
    all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
    # 检查是否还有缺失值存在 输出结果是没有缺失值存在
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
    # print(missing_data.head())
    # 附加的特征工程
    # 把一些数值变量分类
    #MSSubClass=建筑物的类别
    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

    #Changing OverallCond into a categorical variable
    all_data['OverallCond'] = all_data['OverallCond'].astype(str)

    #Year and month sold are transformed into categorical features.
    all_data['YrSold'] = all_data['YrSold'].astype(str)
    all_data['MoSold'] = all_data['MoSold'].astype(str)

    # 这些类别型的数据需要转换为数值型的数据
    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
            'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
            'YrSold', 'MoSold')
    # 将这些有类别的特征值(例如英文表示的额) 使用LabelEncoder变成分类的数值数据
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(all_data[c].values))
        all_data[c] = lbl.transform(list(all_data[c].values))


    # shape
    # print('Shape all_data: {}'.format(all_data.shape))

    # 增加更多重要的特征
    # 计算出房子总的面积  包含上下两层的面积 基础的面积  一层面积 二层面积
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

    # Skewed features 将内容为数值型的特征列找出来  skewed为偏斜度的设置
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    # 检查所有数值特征的偏斜度
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False) #降序排列
    # print("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    # print(skewness.head(10))

    # Box Cox Transformation of (highly) skewed features
    # We use the scipy function boxcox1p which computes the Box-Cox transformation of  1+x .
    # Note that setting  λ=0  is equivalent to log1p used above for the target variable.
    skewness = skewness[abs (skewness) > 0.75]
    # print ("There are {} skewed numerical features to Box Cox transform".format (skewness.shape[0]))

    # 转换数据位正态分布，避免长尾分布现象的出现
    from scipy.special import boxcox1p

    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        # all_data[feat] += 1
        all_data[feat] = boxcox1p(all_data[feat], lam)
    # one-hot编码
    all_data = pd.get_dummies(all_data)
    # print(all_data.shape)
    train = all_data[:ntrain]
    test = all_data[ntrain:]
    return train, test