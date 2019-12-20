import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler, OneHotEncoder
from sklearn.svm import SVR
from scipy.stats import skew

warnings.filterwarnings('ignore')  # 关闭警告
auto_grid = False  # 是否启用下面自动调参的过程，如果为了快速得到结果，这里设置为False

# 加载训练数据与测试数据
print('row21--加载训练数据与测试数据')
training_data = pd.read_csv('/HousePrice_Predict/dataset/train.csv')
test_data = pd.read_csv('/HousePrice_Predict/dataset/test.csv')

# 数据初探可视化
'''
# plt.figure(figsize=(20, 6))
sns.boxplot(training_data.YearBuilt, training_data.SalePrice)
plt.show()

'''
# 参阅https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python 一文，作者总结了SalaPrices有重要影响的
# 四个变量。分别为：1.OverallQual 2.YearBuilt 3.TotalBsmtSF 4.GrlivArea
# 地上居住面积与出售价格的关系图表中，能够发现右下角有两个离群点
'''
plt.figure()
plt.scatter(x=training_data["GrLivArea"], y=training_data["SalePrice"])
plt.ylim(0, 800000)#y轴范围
plt.xlim(0, 5000)#x轴范围
plt.show()
'''
# 它们满足地上居住面积大于4000但是销售价格小于200000。把这两个点去掉
training_data.drop(training_data[(training_data['GrLivArea'] > 4000) & (training_data["SalePrice"] < 200000)].index, inplace=True)

# 观察OverallQual, YearBuilt, TotalBSmtSf, GrLivArea和SalePrice之间的关系
'''
sns.pairplot(data=training_data, x_vars=['GrLivArea', 'YearBuilt', 'TotalBsmtSF', 'OverallQual'], y_vars=['SalePrice'], dropna=True, size=5)
plt.show()
'''

# 观察散点图可得仍存在部分离群点，对这些离群点进行处理
training_data.drop(training_data[(training_data['OverallQual'] < 5) & (training_data['SalePrice'] > 200000)].index, inplace=True)
training_data.drop(training_data[(training_data['OverallQual'] == 8) & (training_data['SalePrice'] > 500000)].index, inplace=True)
training_data.drop(training_data[(training_data['TotalBsmtSF'] > 3000)].index, inplace=True)
training_data.drop(training_data[(training_data['YearBuilt']<1900) & (training_data['SalePrice']>400000)].index,inplace=True)
'''
这里我们先将训练数据集和测试数据集合并为一个数据集，这样做除了方便之后可以同时对训练数据集和测试数据集进行数据清洗和特征工程，此外，也考虑在之后对类别型变量（category variable）需要进行标签编码（LabelEncoder）和独热编码(OneHotEncoder）
 标签编码和独热编码主要是基于类别变量的特征值进行编码，为了避免测试集的类别变量存在训练集所不具有的特征值，而影响模型的性能，因此这里先将两个数据集进行合并，在最后对模型进行训练时再将合并的数据集按照索引重新分割为训练集和测试集。
'''
print('row60--合并训练集和测试集方便进行数据处理')
full_data = pd.concat([training_data, test_data], ignore_index=True)
full_data.drop(['Id'], axis=1, inplace=True) #去掉'Id'属性，降维
train_index = training_data.index
test_index = list(set(full_data.index).difference(set(train_index)))

'''STEP1:数据清洗阶段-----------------------------------------------------------------------------------'''
# 查看缺失值
print('row68--统计各属性缺失值个数，以下为缺失值表格')
miss_data = full_data.isnull().sum().sort_values(ascending=False)
miss_count = miss_data[miss_data > 0]
ratio = miss_count/len(full_data)
null_data = pd.concat([miss_count, ratio], axis=1, keys=['count', 'ratio'])
print(null_data) #查看缺失值表格
'''
对缺失值进行填补，采用两种方法，一种是采取众数填充，一种是定义一个新值进行填充。实际上，有一部分特征值缺失，是因为一部分房屋确实不存在此种特种（如PoolQC为空，房屋可能本身就没有泳池）。对于这种情况，类别特征采用一种新值进行填充，数值特征用‘0’填充。
'''
# 填充缺失值
print('row78--填充特征值')


def fill_values(res):
    none_attr = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish",
                 "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1",
                 "MasVnrType"]
    for attr in none_attr:  # 对于可以NA表示的属性，用None值填补
        res[attr].fillna('None', inplace=True)

    zero_attr = ['MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFinSF2', 'BsmtFinSF1', 'GarageCars', 'GarageArea']
    for attr in zero_attr:  # 对于房子可能没有的数值属性，用0填补
        res[attr].fillna(0, inplace=True)

    mode_attr = ["MSZoning", "BsmtFullBath", "BsmtHalfBath", "Utilities", "Functional", "Electrical", "KitchenQual",
                 "SaleType", "Exterior1st", "Exterior2nd", 'LotFrontage', 'GarageYrBlt']
    for attr in mode_attr:  # 对于缺失值不多的属性及'LotFrontage'用众数填补
        res[attr].fillna(res[attr].mode()[0], inplace=True)
    return res


''' 以下是数值型变量
['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
                     'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
                     'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
                     'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',
                     'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal']
'''
fill_values(full_data)
miss_data = full_data.isnull().sum().sort_values(ascending=False)
miss_data = miss_data[miss_data > 0]
print(miss_data)  # 检查是否还有缺失值

'''STEP2:特征工程-----------------------------------------------------------------------------------'''
# 有一些属性虽然是用数据表示的，但实际上是每个离散值对应一个类别，因此，将这些属性转换成类别属性（category feature），用字符串表示其值
print('row113--将用离散数字表示属性取值转换为用字符串表示')
NumStr = ["MSSubClass", "BsmtFullBath", "BsmtHalfBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "MoSold", "YrSold",
          "YearBuilt", "YearRemodAdd", "LowQualFinSF", "GarageYrBlt"]
for attr in NumStr:
    full_data[attr] = full_data[attr].astype(str)


def ord_to_int(x):  # 将顺序变量（ordinal variable），例如（BsmtQual：Ex，Gd，TA，Fa，Po），转成离散数字表示，LabelEncoder不能很好的表示值之间的顺序关系，因此使用自定义函数
    if x == 'Ex':
        x = 0
    elif x == 'Gd':
        x = 1
    elif x == 'TA':
        x = 2
    elif x == 'Fa':
        x = 3
    elif x == 'Po':
        x = 4
    else:
        x = 5
    return x

    if x == 'GLQ':
        x == 0
    elif x == 'ALQ':
        x = 1
    elif x == 'BLQ':
        x = 2
    elif x == 'Rec':
        x = 3
    elif x == 'LwQ':
        x = 4
    elif x == 'Unf':
        x = 5
    else:
        x = 6
    return x

    if x == 'Typ':
        x == 0
    elif x == 'Min1':
        x = 1
    elif x == 'Min2':
        x = 2
    elif x == 'Mod':
        x = 3
    elif x == 'Maj1':
        x = 4
    elif x == 'Maj2':
        x = 5
    elif x == 'Sev':
        x = 6
    else:
        x = 7
    return x

    if x == 'Gd':
        x = 0
    elif x == 'Av':
        x = 1
    elif x == 'Mn':
        x = 2
    elif x == 'No':
        x = 3
    else:
        x = 4
    return x

    if x == 'Y':
        x = 0
    elif x == 'P':
        x = 1
    else:
        x = 2
    return x

    if x == 'Fin':
        x = 0
    elif x == 'RFn':
        x = 1
    elif x == 'Unf':
        x = 2
    else:
        x = 3
    return x

    if x == 'Grvl':
        x == 0
    else:
        x ==1
    return x




trans_attr = ['BsmtFinType1','MasVnrType','Foundation','HouseStyle','Functional','BsmtExposure','GarageFinish','PavedDrive','Street','ExterQual','PavedDrive','ExterQual','ExterCond','KitchenQual','HeatingQC','BsmtQual','FireplaceQu','GarageQual','PoolQC']
for attr in trans_attr:
    full_data[attr] = full_data[attr].apply(ord_to_int)
    full_data[attr] = full_data[attr].astype(str)  # 转换成字符串
# 添加一个跟房价相关的特征，地下室面积+一楼面积+二楼面积
full_data['TotalSF'] = full_data.TotalBsmtSF + full_data['1stFlrSF'] + full_data['2ndFlrSF']
# 房屋有些区域的有无也是影响房屋价格的重要因素
full_data['HasWoodDeck'] = (full_data['WoodDeckSF'] == 0) * 1
full_data['HasOpenPorch'] = (full_data['OpenPorchSF'] == 0) * 1
full_data['HasEnclosedPorch'] = (full_data['EnclosedPorch'] == 0) * 1
full_data['Has3SsnPorch'] = (full_data['3SsnPorch'] == 0) * 1
full_data['HasScreenPorch'] = (full_data['ScreenPorch'] == 0) * 1
full_data['YearsSinceRemodel'] = full_data['YrSold'].astype(int) - full_data['YearRemodAdd'].astype(int)  # 房屋改造时间
full_data['Total_Home_Quality'] = full_data['OverallQual'] + full_data['OverallCond']  # 房屋整体质量

# 进行数据转换，采用对数转换，取对数之后数据的性质和相关关系不会发生改变，但压缩了变量的尺度，大大方便了计算。
quantitative = [f for f in full_data.columns if full_data.dtypes[f] != 'object' and full_data.dtypes[f] != 'str']
quantitative.remove('SalePrice')
# f = pd.melt(training_data, value_vars=quantitative)
# g = sns.FacetGrid(f, col="variable",  col_wrap=5, sharex=False, sharey=False)
# g = g.map(sns.distplot, "value")
# plt.show()
# 计算各定量变量的偏度
full_data[quantitative].skew(axis=0).sort_values(ascending=False)


def add_logs(res, ls):
    m = res.shape[1]
    for la in ls:
        res = res.assign(newcol=pd.Series(np.log(1.01+res[la])).values)
        res.columns.values[m] = la + '_log'
        m += 1
    return res


trans_attr = ['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'EnclosedPorch', 'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars', 'GrLivArea', 'LotArea', 'LotFrontage', 'MasVnrArea', 'MiscVal', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'PoolArea', 'ScreenPorch', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF', 'TotalSF', 'HasWoodDeck', 'HasOpenPorch', 'HasEnclosedPorch', 'Has3SsnPorch', 'HasScreenPorch', 'YearsSinceRemodel', 'Total_Home_Quality']
# 对于偏度大于0.15的定量变量，我们可以对其进行log操作取对数以提升质量。
log_list = [attr for attr in trans_attr if abs(full_data[attr].skew(axis=0).astype(int)) > 0.15]
full_data = add_logs(full_data, log_list)

# One-hot-encoding，找出所有需要编码的定型变量，去除经过ord_to_int函数处理过的顺序变量
qualitative = [f for f in training_data.columns if training_data.dtypes[f] == 'object' or training_data.dtypes[f] == 'str']
oridnals=['BsmtFinType1','MasVnrType','Foundation','HouseStyle','Functional','BsmtExposure','GarageFinish','PavedDrive','Street',
                   'ExterQual', 'PavedDrive','ExterQua','ExterCond','KitchenQual','HeatingQC','BsmtQual','FireplaceQu','GarageQual','PoolQC']
qualitative = list(set(qualitative).difference(set(oridnals)))
qualitative.append('GarageYrBlt')


def encode(encode_df):
    encode_df = np.array(encode_df)
    ohc = OneHotEncoder()
    lbl = LabelEncoder()
    lbl.fit(encode_df)
    res1 = lbl.transform(encode_df).reshape(-1, 1)  # reshape(-1,1)转换成二位array
    return pd.DataFrame(ohc.fit_transform(res1).toarray()), lbl, ohc


def getdummies(res, ls):
    decoder = []
    outers = pd.DataFrame({'A': []})

    for la in ls:
        cat, lbl, ohc = encode(res[la])
        cat.columns = [la + str(x) for x in cat.columns]
        outers.reset_index(drop=True, inplace=True)
        outers = pd.concat([outers, cat], axis=1)
        decoder.append([lbl, ohc])
    return outers, decoder


catpredlist = qualitative
res = getdummies(full_data[catpredlist], catpredlist)
df = res[0]
decoder = res[1]
float_andordinal = list(set(full_data.columns.values).difference(set(qualitative)))
full_data.columns.values
print(df.shape)
df = pd.concat([df, full_data[float_andordinal]], axis=1)
df.drop(['SalePrice', 'A'], axis=1, inplace=True)
df.to_csv('df.csv')
# 特征降维 PCA
pca = PCA(n_components=295)
df = pd.DataFrame(pca.fit_transform(df))
df_train=df.iloc[train_index]
df_score=df.iloc[test_index]
'''STEP3:建模-----------------------------------------------------------------------------------'''
n_train = training_data.shape[0]
X = df[:n_train]
X_test = df[n_train:]
Y = training_data.SalePrice
scaler = RobustScaler()
X_scaled = scaler.fit(X).transform(X)
Y_log = np.log(Y)
X_test_scaled2 = scaler.transform(X_test)

# 先定义比赛使用的RMSE评估指标，使用交叉验证策略
def rmse_cv(model, X, y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse


# 选择几个模型来做（初始未调参）
models = [
    Ridge(),
    Lasso(alpha=0.01, max_iter=10000),
    SVR(),
    ElasticNet(alpha=0.001, max_iter=10000),
    BayesianRidge(),
    KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
]

names = ['1.ridge', '2.lasso', '3.svr', '4.ela_net', '5.bay_ridge', '6.ker_ridge']
print('row343--使用给定的6个未调参的初始模型，依次评估其得分')
for name, model in zip(names, models):
    score = rmse_cv(model, X_scaled, Y_log)
    print("\t{}\t\t\tmean={:.6f}\tstd={:.4f}".format(name, score.mean(), score.std()))


class grid:
    """
    对选的的模型自动调参
    """

    def __init__(self, model):
        self.model = model

    def grid_get(self, X, y, param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(X, y)
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score', 'std_test_score']])


# 对选定的模型进行自动调参（除BayesianRidge其他五种都需要调参）
if auto_grid:
    print('row366--开始对5个选定的模型进行自动调参（Bayesian Ridge无需调参）')
    print('\t\t1.Ridge-------------------')
    grid(Ridge()).grid_get(X_scaled, Y_log,
                           {'alpha': [35, 40, 45, 50, 55, 60, 65, 70, 80, 90]})
    print('\t\t2.LASSO-------------------')
    grid(Lasso()).grid_get(X_scaled, Y_log,
                           {'alpha': [0.0004, 0.0005, 0.0007, 0.0009], 'max_iter': [10000]})
    print('\t\t3.SVR-------------------')
    grid(SVR()).grid_get(X_scaled, Y_log,
                         {'C': [11, 13, 15], 'kernel': ["rbf"], "gamma": [0.0003, 0.0004], "epsilon": [0.008, 0.009]})
    print('\t\t4.Elastic Net-------------------')
    grid(ElasticNet()).grid_get(X_scaled, Y_log,
                                {'alpha': [0.0008, 0.004, 0.005], 'l1_ratio': [0.08, 0.1, 0.3], 'max_iter': [10000]})
    print('\t\t5.Kernel Ridge-------------------')
    grid(KernelRidge()).grid_get(X_scaled, Y_log,
                                 {'alpha': [0.2, 0.3, 0.4], 'kernel': ["polynomial"], 'degree': [3], 'coef0': [0.8, 1]})

# 以下是调好参数的模型
model_ridge = Ridge(alpha=35)
model_lasso = Lasso(alpha=0.0009, max_iter=10000)
model_svr = SVR(gamma=0.0004, kernel='rbf', C=11, epsilon=0.008)
model_elasticnet = ElasticNet(alpha=0.005, l1_ratio=0.3, max_iter=10000)
model_bayesianridge = BayesianRidge()
model_kernelridge = KernelRidge(alpha=0.3, kernel='polynomial', degree=3, coef0=1)

"""STEP7:集成多个模型-------------------------------------------------------------------------------------"""


class AverageWeight(BaseEstimator, RegressorMixin):
    def __init__(self, mod, weight):
        self.mod = mod
        self.weight = weight

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.mod]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        w = list()
        pred = np.array([model.predict(X) for model in self.models_])
        # for every data point, single model prediction times weight, then add them together
        for data in range(pred.shape[1]):
            single = [pred[model, data] * weight for model, weight in zip(range(pred.shape[0]), self.weight)]
            w.append(np.sum(single))
        return w


# 对全部六个模型进行加权评分
print('row416--对全部6个模型进行加权评分')
weight_avg1 = AverageWeight(
    mod=[model_lasso, model_ridge, model_svr, model_kernelridge, model_elasticnet, model_bayesianridge],
    weight=[0.02, 0.2, 0.25, 0.3, 0.03, 0.2])
score1 = rmse_cv(weight_avg1, X_scaled, Y_log)
print('\tScore=' + str(score1.mean()))

# 只选择两个比较好的模型进行加权评分
print('row424--对表现最好的2个模型进行加权评分')
weight_avg2 = AverageWeight(
    mod=[model_svr, model_kernelridge],
    weight=[0.5, 0.5]
)
score2 = rmse_cv(weight_avg2, X_scaled, Y_log)
print('\tScore=' + str(score2.mean()))

"""STEP8:Stacking----------------------------------------------------------------------------------------"""


class stacking(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, mod, meta_model):
        self.mod = mod
        self.meta_model = meta_model
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)

    def fit(self, X, y):
        self.saved_model = [list() for i in self.mod]
        oof_train = np.zeros((X.shape[0], len(self.mod)))

        for i, model in enumerate(self.mod):
            for train_index, val_index in self.kf.split(X, y):
                renew_model = clone(model)
                renew_model.fit(X[train_index], y[train_index])
                self.saved_model[i].append(renew_model)
                oof_train[val_index, i] = renew_model.predict(X[val_index])

        self.meta_model.fit(oof_train, y)
        return self

    def predict(self, X):
        whole_test = np.column_stack([np.column_stack(model.predict(X) for model in single_model).mean(axis=1)
                                      for single_model in self.saved_model])
        return self.meta_model.predict(whole_test)

    def get_oof(self, X, y, test_X):
        oof = np.zeros((X.shape[0], len(self.mod)))
        test_single = np.zeros((test_X.shape[0], 5))
        test_mean = np.zeros((test_X.shape[0], len(self.mod)))
        for i, model in enumerate(self.mod):
            for j, (train_index, val_index) in enumerate(self.kf.split(X, y)):
                clone_model = clone(model)
                clone_model.fit(X[train_index], y[train_index])
                oof[val_index, i] = clone_model.predict(X[val_index])
                test_single[:, j] = clone_model.predict(test_X)
            test_mean[:, i] = test_single.mean(axis=1)
        return oof, test_mean


# 第一次小尝试。注意：必须先进行imputer否则stacking不会工作，原作者也不清楚原因
print('row475--try once，使用stacking进行训练并预测')
a = SimpleImputer().fit_transform(X_scaled)
b = SimpleImputer().fit_transform(Y_log.values.reshape(-1, 1)).ravel()
stack_model1 = stacking(
    mod=[model_lasso, model_ridge, model_svr, model_kernelridge, model_elasticnet, model_bayesianridge],
    meta_model=model_kernelridge
)
score3 = rmse_cv(stack_model1, a, b)
print('\tScore=' + str(score3.mean()))

# 再次尝试。这里从stacking中选取一些特征来和原始特征组合在一起
print('row486--try twice，再次使用stacking进行训练并预测，这次使用stacking中的特征与原始特征组合在一起')
X_train_stack, X_test_stack = stack_model1.get_oof(a, b, X_test_scaled2)
X_train_add = np.hstack((a, X_train_stack))
X_test_add = np.hstack((X_test_scaled2, X_test_stack))
score4 = rmse_cv(stack_model1, X_train_add, b)
print('\tScore=' + str(score4.mean()))

"""STEP9（FINAL）:取得最终的预测数据----------------------------------------------------------------------"""
# 最终使用的模型就是上面的stack_model1，为了避免上面几步对这个对象的改变，这里重新定义一个
stack_model2 = stacking(
    mod=[model_lasso, model_ridge, model_svr, model_kernelridge, model_elasticnet, model_bayesianridge],
    meta_model=model_kernelridge
)
# 再定义一个新的model，这里只选取原作者觉得最好的两个模型
stack_model3 = stacking(
    mod=[model_svr, model_kernelridge],
    meta_model=model_kernelridge
)
# 用它来进行预测，取得预测结果并导出
print('row500--try final，最终预测')
stack_model2.fit(a, b)
prediction = np.exp(stack_model2.predict(X_test_scaled2))
result = pd.DataFrame({'Id': test_data.Id, 'SalePrice': prediction})
print('row504--输出预测结果到文件')
result.to_csv("Prediction/submission.csv", index=False)
print('DONE!')
