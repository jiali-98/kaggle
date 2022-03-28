"""
创建时间：2022/3/23
创建人：JiaLi-Wu
泰坦尼克号之灾：越策泰坦尼克号上的生存情况

训练集：11列
['PassengerId':乘客, 'Pclass'：乘客等级, 'Name'：姓名, 'Sex'：性别,
# 'Age'：年龄, 'SibSp'：堂兄弟/妹个数, 'Parch'： 父母与小孩个数,
#  'Ticket'：船票信息, 'Fare'：票价, 'Cabin'：客舱, 'Embarked'：登船港口]
"""

import pandas as pd
pd.set_option('display.expand_frame_repr', False)

"""===============S1.数据读取&数据预处理==============="""
# 1.1导入数据
data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")
data_full = data_train.append(data_test, ignore_index=True)

# 1.2查看数据信息
data_full.head() #查看数据集信息
data_full.describe() # 查看数据集的描述统计信息
data_train.info()  # 查看每列信息
data_full.info()
data_full.isnull().sum() #统计缺失的数据

# 1.3 数据预处理

# 1.3.1 数字型缺失值处理 ->  [Age, Fare票价] 利用平均值进行填充
# Age 年龄
data_full['Age'] = data_full['Age'].fillna(data_full['Age'].mean())
# Fare 票价
data_full['Fare'] = data_full['Fare'].fillna(data_full['Fare'].mean())
# 查看缺失值处理后的数据
data_full['Age'].shape
data_full['Fare'].shape
data_full.isnull().sum()

# 1.3.2 字符型缺失值处理-> [Cabin客舱号, Embarked登船港口]
data_full['Embarked'].value_counts() # 计算各字符的数量
data_full['Embarked'] = data_full['Embarked'].fillna('S') #用数量最多的字符填充
data_full.isnull().sum()

# Cabin的缺失值为NaN->用Unkow的缩写U填充
data_full['Cabin'] = data_full['Cabin'].fillna('U')
data_full.isnull().sum()
data_full.info()

"""===============S2.特征工程==============="""
#特征工程：特征提取+特征选择

# 2.1 特征提取
# 01 分类数据：性别Sex、登船港口Embarked、客舱等级Pclass
# 02　字符串类型：乘客姓名Name、客舱号Cabin
# 03  家庭类别：堂兄弟/妹个数SibSp、父母与小孩个数Parch

# 2.1.1分类数据
#性别分类{male, female}：男1 女0
sex_dict = {'male':1, 'female':0}
#Series的map方法可以接受一个函数或含有映射关系的字典型对象
data_full['Sex'] = data_full['Sex'].map(sex_dict)
data_full['Sex'].head()

# 登船港口Embarked{S出发地点,C途径地点1,Q出发地点2}->One-Hot编码
data_full['Embarked'].value_counts()
embarkedDf = pd.DataFrame()
# 采用get_dummies实现one-hot编码
embarkedDf = pd.get_dummies(data_full['Embarked'],prefix='Embarked')
embarkedDf.head()
# 添加obe-hot编码到data_full并删除原Embarked列
data_full = pd.concat([data_full, embarkedDf], axis=1)
data_full.drop('Embarked',axis=1,inplace=True)
data_full.head()

#客舱等级 Pclass->One-ot编码
pclassDf = pd.DataFrame()
pclassDf = pd.get_dummies(data_full['Pclass'],prefix='Pclass')
pclassDf.head()
# 添加obe-hot编码到data_full并删除原Pclass列
data_full = pd.concat([data_full, pclassDf], axis=1)
data_full.drop('Pclass',axis=1,inplace=True)
data_full.info()

# 2.1.2 字符串数据
# 姓名name
def getTitle(name):
    #从姓名中获取头衔
    str1 = name.split(',')[1]
    str2 = str1.split('.')[0]
    str3 = str2.strip()#移除字符串首尾空格
    return str3
titleDf = pd.DataFrame()
titleDf['Title'] = data_full['Name'].map(getTitle)
titleDf.value_counts()
# 头衔类别
# officer:政府官员， Royalty皇室，
# Mr已婚男士， Mrs已婚女士， Miss未婚女士， Master有技能的人
title_mapDict = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }
#map函数：对Series每个数据应用自定义的函数计算
titleDf['Title'] = titleDf['Title'].map(title_mapDict)
#使用get_dummies进行one-hot编码
titleDf = pd.get_dummies(titleDf['Title'])
titleDf.head()
# 添加obe-hot编码到data_full并删除原Name列
data_full = pd.concat([data_full, titleDf], axis=1)
data_full.drop('Name',axis=1,inplace=True)
data_full.info()

#客舱号Cabin
data_full['Cabin'].value_counts()
cabinDf = pd.DataFrame()
cabinDf['Cabin'] = data_full['Cabin'].map(lambda c:c[0])#提取首字母
#使用get_dummies进行one-hot编码
cabinDf = pd.get_dummies(cabinDf['Cabin'],prefix='Cabin')
cabinDf.head()
# 添加obe-hot编码到data_full并删除原Name列
data_full = pd.concat([data_full, cabinDf], axis=1)
data_full.drop('Cabin',axis=1,inplace=True)
data_full.info()


# 2.1.3 家庭类别数据
familyDf = pd.DataFrame()
familyDf['familySize'] = data_full['Parch'] + data_full['SibSp']+1
"""
fam_samll:1 fam_middle:2-4 fam_big:>=5
"""
familyDf['fam_small'] = familyDf['familySize'].map(lambda s:1 if s==1 else 0)
familyDf['fam_middle'] = familyDf['familySize'].map(lambda s:1 if 2<=s<5 else 0)
familyDf['fam_big'] = familyDf['familySize'].map(lambda s:1 if s>=5 else 0)
familyDf.head(100)

# 2.2 特征选择
# 2.2.1 相关系数法： 计算各个特征的相关系数
corrDf = data_full.corr()
corrDf['Survived'].sort_values(ascending=False) #查看各特征与生存情况的相关系数
# 根据各特征与Survived的相关系数的大小，选择以下特征作为模型的输入：
#头衔（titleDf）、客舱等级（pclassDf）、家庭大小（familyDf）、船票价格（Fare）、船舱号（cabinDf）、登船港口（embarkedDf）、性别（Sex）
#特征选择
full_X = pd.concat([titleDf,pclassDf, familyDf, data_full['Fare'],
                    cabinDf, embarkedDf, data_full['Sex']], axis=1)
full_X.head()

"""===============S3.构建模型&训练模型&模型评估==============="""
# 3.1 建立训练数据集和测试集
# 3.1.1 得到原始数据集和预测数据集
sourceRow = data_train.shape[0] #原始训练集包括891
source_X = full_X.loc[0: sourceRow-1, :] #原始数据集：特征
source_Y = data_full.loc[0: sourceRow-1, 'Survived'] #原始数据集：标签
pred_X = full_X.loc[sourceRow:,:] #预测数据集：特征

#3.1.2 得到训练数据和测试数据
from sklearn.model_selection import train_test_split
#train_test_split:按比例划分测试集与训练集
#建立模型用的训练数据集和测试数据集
[train_X, test_X, train_Y, test_Y] = train_test_split(source_X, source_Y, test_size= 0.2)

print('原始特征数据集: ', source_X.shape,
      '特征训练数据: ', train_X.shape,
      '特征测试数据: ', test_X.shape)
print('原始标签数据集: ', source_Y.shape,
      '标签训练数据：',train_Y.shape,
      '标签测试数据：', test_Y.shape)

# 3.1.3 训练模型
model = 0

if model == 0:
    # 逻辑回归
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()#创建模型
    model.fit(train_X, train_Y) #训练模型
    # model.score(test_X, test_Y) #模型评估
    print("逻辑回归：", model.score(test_X, test_Y))

elif model == 1:
    #随机森林
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100)
    model.fit( train_X, train_Y)
    print("随机森林：", model.score(test_X, test_Y))

elif model == 2:
    #SVM
    from sklearn.svm import SVC
    model = SVC()
    model.fit(train_X, train_Y)
    print("SVM：", model.score(test_X, test_Y))

elif model == 3:
    #KNN
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors = 3)
    model.fit(train_X, train_Y)
    print("KNN：", model.score(test_X, test_Y))

elif model == 4:
    #朴素贝叶斯
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(train_X, train_Y)
    print("朴素贝叶斯：", model.score(test_X, test_Y))

"""===============S4.方案实施==============="""
Pred_Y = model.predict(pred_X)
Pred_Y = Pred_Y.astype(int)
#生成的预测值为浮点数,但kaggle为整数
passenger_id = data_full.loc[sourceRow:, 'PassengerId']
predDf = pd.DataFrame({'PassengerId': passenger_id,
    'Survived':Pred_Y})
predDf.shape
predDf.head()
#保存csv文件
predDf.to_csv('titanic_pred.csv', index=False)
a=1