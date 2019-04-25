#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
通过对tag进行fillNan之后，并且选择6279个特征。
效果提升了，现在是0.7850
我感觉离前面相差这麽大的原因，很有可能就是模型或者样本的原因了
样本，现在主要是欠采样；
模型，现在是树模型
一方面，可以选择逗DNN模型；
另一方面，可以选择把所有样本都训练
_____
使用处理缺失值多的样本
_
使用测试集

'''


# In[2]:


import gc 
gc.collect()


# In[3]:



import pandas as pd 
import numpy as np

import random
random.seed(1028)

import lightgbm as lgb
import math


import os 
try:
    os.chdir(r'F:\比赛\融360 3th')
except:
    os.chdir('/data/anonym2/data/r')
version='_1109_test_'
useGpu = True
print('useGpu',useGpu)
if useGpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    
import warnings
warnings.filterwarnings("ignore")


from sklearn import metrics
from sklearn.metrics import log_loss
try:
    from sklearn.cross_validation import train_test_split
except:
    from sklearn.model_selection import train_test_split
import time

from sklearn import preprocessing
import time
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import StratifiedKFold
import pickle
selectFeature = False
print('selectFea',selectFeature)


# In[4]:


def MIS(data, percentage=0.8 ):
    '''
    使用互信息进行特征选择
    '''
    print('互信息选择特征')
    if os.path.exists('./feature/misCol(tag).txt') and True:
        print('load misCol.txt')
        with open("./feature/misCol(tag).txt", "rb") as fp:
            newCol = pickle.load(fp)
    else:
        corr_values = []
        allCol =data.columns.tolist()
        for col in ['label','id']:
            try:
                allCol.remove(col)
            except:
                pass
        y_train =data['label']
        
        for col in allCol:
            corr_values.append(abs(mutual_info_score( y_train, data[col])))
        tempDf = pd.DataFrame({'col': allCol,'corr_value':corr_values})
        tempDf2 = tempDf.nlargest(int(len(tempDf) * percentage), 'corr_value')
    
        newCol = tempDf2['col'].values.tolist()
        with open("./feature/misCol(tag).txt", "wb") as fp:  
            pickle.dump(newCol, fp)
    print('mis col len', len(newCol))
    return newCol


# In[5]:


def calCorrcoef(data, percentage=0.8, method = 'percentage',threshold=0.01):
    '''
    使用相关系数进行特征选择
    '''
    if os.path.exists('./feature/corrCoefCol(tag).txt') and True:
        print('load CorrCoefCol(tag) file')
        with open('./feature/corrCoefCol(tag).txt','rb') as f:
            allCoefCol = pickle.load(f)
    else:     
        print('cal corrcoef')
        allCol = data.columns.tolist()
        for col in ['label','id','loan_dt','tag']:
            try:
                allCol.remove(col)
            except:
                pass
        dataOne = data[data['label']==1]
        dataZero = data[data['label']==0]
        print('dataZero',dataZero.shape)
        print('dataOne',dataOne.shape)
        allCoefCol = []
#        for i in range(math.floor(len(dataZero)/len(dataOne))):
        data.sample(frac=1)
        data.reset_index(drop=True,inplace=True)
        data.fillna(data.mean(),inplace=True)
        for i in range(math.floor(len(data)*2/data.shape[1])):
            print('calcorrcoef,step:',i)
#            tempData = pd.concat([dataZero[i:(i+1)*len(dataOne)],dataOne],axis = 0)
            tempData = data[int(i*data.shape[1]/2):int((i+1)*data.shape[1]/2)]
            corr_values = []
            y_train =tempData['label'].values
    
            for col in allCol:
                corr_values.append(abs(np.corrcoef(tempData[col].values,y_train)[0,1]))
                
            corr_df = pd.DataFrame({'col':allCol,'corr_value':corr_values})
            if method == 'percentage':
                corr_df = corr_df.nlargest(int(len(corr_df) * percentage), 'corr_value')
                newCol =corr_df['col'].values.tolist()
            else:
                print('cal corrcoef with method larger than:',threshold)
                corr_df = corr_df[corr_df['corr_value']> threshold]
                newCol = corr_df['col'].values.tolist()
            print('newCol length',len(newCol))
            allCoefCol= list(set(newCol)-set(allCoefCol))
        with open('./feature/corrCoefCol(tag).txt','wb') as f :
            pickle.dump(allCoefCol,f)
    
    print('get corrcoef col',len(allCoefCol))
    return allCoefCol


# In[6]:


def lassoSelect(data, percentage=0.6, method='percentage'):
    '''
    使用lasso进行特征选择
    '''
    if os.path.exists('./feature/lassoCol(tag).txt') and True:
        print('load lassoCV file')
        with open("./feature/lassoCol(tag).txt", "rb") as fp:
            lassoCol = pickle.load(fp)
    else:
        x= data[[i for i in data.columns if i not in ['label','id']]]
        y= data['label']
        x.fillna(x.mean(),inplace=True)

        lassocv = LassoCV()
        lassocv.fit(x,y)
        alpha = lassocv.alpha_
        print('alpha',alpha)
        
        lasso = Lasso(alpha)
        lasso.fit(x,y)
        coefList = lasso.coef_
        coefList = [ abs(item) for item in coefList]
        
        tempDf =pd.DataFrame({'name':x.columns.tolist(),'coef': coefList})
        if method == 'percentage':
            print('select method',method)
            tempDf = tempDf.nlargest(int(len(tempDf) * percentage), 'coef')
        else:
            print('select method',method)
            tempDf = tempDf[tempDf['coef']>0]
        lassoCol =tempDf['name'].values.tolist()
        with open("./feature/lassoCol(tag).txt", "wb") as fp:
            pickle.dump(lassoCol, fp)
    print('lassoCol len',len(lassoCol))
    return lassoCol 

def lassoCVSelect(data, nfolds=3):
    '''
    使用CV进行选择，最终进行取系数大于0的并集(或者交集)
    个人认为，交集可以具有更高的鲁棒性
    '''
    if os.path.exists('./feature/lassoCVCol.txt') and True:
        print('load lassoCV file')
        with open("./feature/lassoCVCol.txt", "rb") as fp:
            lassoCol = pickle.load(fp)
    else:
        lassoCol = []
        data.reset_index(drop=True,inplace=True)
        x= data[[i for i in data.columns if i not in ['label','id']]]
        y= data['label']
    
        skf = StratifiedKFold(n_splits = nfolds , random_state=1028, shuffle= True)
        x.fillna(x.mean(),inplace=True)
        for index, (train_index, test_index) in enumerate(skf.split(x,y)):
            print("Fold", index)
            xtr =x.loc[train_index,:]
            ytr =y.loc[train_index]
            xte = x.loc[test_index,:]
            yte = y.loc[test_index]

    
            lassocv = LassoCV()
            lassocv.fit(xtr,ytr)
            alpha = lassocv.alpha_
            print('alpha',alpha)
        
            lasso = Lasso(alpha)
            lasso.fit(xtr,ytr)
            coefList = lasso.coef_
            coefList = [ abs(item) for item in coefList]
            print('valid score:',lasso.score(xte,yte))
            tempDf =pd.DataFrame({'name':xte.columns.tolist(),'coef': coefList})
#            tempDf = tempDf.nlargest(int(len(tempDf) * percentage), 'coef')
            tempDf = tempDf[tempDf['coef'] > 0]
            newCol =tempDf['name'].values.tolist()
            if lassoCol==[]:
                lassoCol = newCol
            else:
                lassoCol= list(set(lassoCol)-set(newCol))
        with open("./feature/lassoCVCol.txt", "wb") as fp:
            pickle.dump(lassoCol, fp)
    print('lassoCol length',len(lassoCol))
    return lassoCol


# In[7]:


def underSample(data):
    '''
    对数据进行欠采样，平衡数据分布
    '''
    data.reset_index(drop=True,inplace=True)
    indexs=data.index.values
    labels =data.label.tolist()
    zeroIx=[ix for ix in indexs if labels[ix]==0]
    oneIx= [ix for ix in indexs if labels[ix]==1]
    
    newZeroIx= random.sample(zeroIx, int(len(oneIx)))
    balanceIx= newZeroIx+ oneIx
    return data.loc[balanceIx,:]


# In[8]:


def selectFea(data):
    '''
    使用多种方法进行特征选择
    '''
    print('selectFea')
    misCol, coefCol,lassoCol, lassoCvCol= [],[],[],[]
    print('互信息特征选择,percentage: 0.6')
    misCol = MIS(data[data['label'].notnull()], 0.7)
#    misCol = list(set(misCol+['label','id']) & set(data.columns))
    print('MIS col len:', len(misCol))
    
    print('相关系数特征选择,percentage: 0.6')
    coefCol = calCorrcoef(data[data['label'].notnull()], 0.7,'other',0.01 )
#    coefCol = list(set(coefCol+['label','id']) & set(data.columns))
    print('coefCol len',len(coefCol))

    print('lasso 选择特征 perc:0.6')
    lassoCol = lassoSelect(data[data['label'].notnull()], 0.6, 'percentage')
#    lassoCol = list(set(lassoCol+['label','id']) & set(data.columns))
    print('lasso fea len', len(lassoCol))
    
#    print('lassoCVselect 选择特征 nfolds 2')
#    lassoCvCol = lassoCVSelect(data[data['label'].notnull()], nfolds=2)
#    print('lasso cv fea len', len(lassoCvCol))
    
    return misCol, coefCol,lassoCol, lassoCvCol
    
    
def preProcess(data):
    '''
    对数据进行数据清洗，特征选择
    '''
    print('drop unique ==1 ')
    dataCol = data.columns.tolist()
    dataCol.remove('label')
    for col in dataCol:
        if data[col].nunique()==1: 
            data.drop(col,axis=1,inplace=True)
    print('drop nan percentage >0.8')
    data = data.loc[:, data.isnull().mean()<0.8]
    print('把缺失值都填补为mean')
    dataCol = data.columns.tolist()
    dataCol.remove('label')
    data.loc[:,dataCol] = data.loc[:,dataCol].fillna(data.mean())
            
    misCol, coefCol,lassoCol, lassoCvCol = selectFea(data)
    if len(lassoCvCol)==0:
        #allFeaCol = list(set(misCol)|set(coefCol)|set(lassoCol))
        allFeaCol = list(set(misCol)&set(coefCol)&set(lassoCol))
    else:
        allFeaCol = list(set(misCol)&set(coefCol)&set(lassoCol)&set(lassoCvCol))
    allFeaCol = list(set(allFeaCol+['label','id']) & set(data.columns))
    print('allFeaCol',len(allFeaCol))
    
    data = data[allFeaCol]
    trainCol= data.columns.tolist()
    trainCol.remove('label')
    Xtrain = data[data['label'].notnull()][trainCol]
    Ytrain = data[data['label'].notnull()]['label']
    valid = data[data['label'].isnull()][trainCol]
    Xtrain.reset_index(drop =True,inplace =True)
    Ytrain.reset_index(drop= True,inplace =True)
    valid.reset_index(drop = True ,inplace= True)
    return Xtrain, Ytrain, valid


# In[9]:


if os.path.exists('./medFile/train(underSample).pickle') and True:
        print('load Xtrain Ytrain test pickle')
        trainU =pd.read_pickle('./medFile/train(underSample).pickle')
        print('read file done!')
else: 
    train=pd.read_pickle('./data/train.pickle')
    print('read file done!')
    trainU =underSample(train)
    trainU.to_pickle('./medFile/train(underSample).pickle')
    print('write to pickle sucessfully')


# In[10]:


train=pd.read_pickle('./data/train.pickle')
print(train.shape)


# In[11]:


# valid = pd.read_pickle('./data/valid.pickle')
# print(valid.shape)


# In[12]:


test = pd.read_csv('./data/test.txt',sep='\t',engine='python',encoding='utf-8',)
print(test.head())
print(test.shape)


# In[13]:


if os.path.exists('./feature/misCol(tag).txt') and True:
    print('load Tag misCol.txt')
    with open("./feature/misCol(tag).txt", "rb") as fp:
        tagFea = pickle.load(fp)


# In[14]:


valid = test 


# In[15]:


Xtrain = train[tagFea]
test = valid[tagFea]
Ytrain = train[['tag']]

le = preprocessing.LabelEncoder()
Ytrain['tag'] = le.fit_transform(Ytrain['tag'])


# In[16]:


print(Xtrain.shape)
print(test.shape)


# In[17]:


Xtrain.head()


# In[18]:


Xtrain.reset_index(drop = True,inplace =True)
Ytrain.reset_index(drop = True, inplace = True)


# In[19]:


print('先对tag进行预测')


nfolds = 5
skf = StratifiedKFold(n_splits= nfolds, random_state= 1028, shuffle=True)
best_score = []
bestAuc = []

train['tag_0'] = 0
train['tag_1'] = 0
valid['tag_0'] = 0
valid['tag_1'] = 0

lgb_model = lgb.LGBMClassifier(
    boosting_type='gbdt', 
    num_leaves=32, 
    reg_alpha=0, 
    reg_lambda=0.1,
    max_depth=-1, 
    n_estimators=5000, 
    objective='binary',
    metric = ['auc', 'binary_logloss'],
    colsample_bytree=0.7, 
    subsample_freq=1,
    learning_rate=0.05, 
    random_state= 1028,
    n_jobs=-1,
    device = 'gpu',
    verbose_eval =10,
)
    
for index, (train_index, test_index) in enumerate(skf.split(Xtrain, Ytrain)):
    print('Folds:',index)
    lgb_model.fit(Xtrain.loc[train_index,:], Ytrain.loc[train_index],
                  eval_set=[(Xtrain.loc[train_index,:], Ytrain.loc[train_index]),
                            (Xtrain.loc[test_index,:], Ytrain.loc[test_index])], 
                            early_stopping_rounds=50, )
    best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
    print(best_score)
    bestAuc.append(lgb_model.best_score_['valid_1']['auc'])
    print(bestAuc)
    test_pred = lgb_model.predict_proba(test, num_iteration=lgb_model.best_iteration_)[:, 0]
    valid['tag_0'] = valid['tag_0'] + test_pred
    test_pred = lgb_model.predict_proba(test, num_iteration=lgb_model.best_iteration_)[:, 1]
    valid['tag_1'] = valid['tag_1'] + test_pred

    train_pred = lgb_model.predict_proba(Xtrain, num_iteration=lgb_model.best_iteration_)[:, 0]
    train['tag_0'] = train['tag_0'] + train_pred
    train_pred = lgb_model.predict_proba(Xtrain, num_iteration=lgb_model.best_iteration_)[:, 1]
    train['tag_1'] = train['tag_1'] + train_pred
        
print('best_score mean:',np.mean(best_score))

valid['tag_0'] = valid['tag_0'] / nfolds 
valid['tag_1'] = valid['tag_1'] / nfolds 
train['tag_0'] = train['tag_0'] / nfolds 
train['tag_1'] = train['tag_1'] / nfolds 


# In[20]:


train.tail(10)


# In[21]:


valid.head()


# In[22]:


train.to_pickle('./medFile/train_1(tag).pickle')
valid.to_pickle('./medFile/test_1(tag).pickle')


# In[23]:


#groupby tag fillna
valid['tag']= valid['tag_0'].apply(lambda x: 'fq' if x>=0.5 else 'pd')


# In[24]:


valid.head()


# In[25]:


trainNanCol = train.columns[train.isna().any()].tolist()


# In[26]:


len(trainNanCol)


# In[27]:


list(set(train.columns)-set(trainNanCol))


# In[28]:


trainNanCol05 = train.columns[train.isnull().mean()<0.5].tolist()


# In[29]:


len(trainNanCol05)


# In[30]:


trainNanCol05


# In[31]:


trainNanCol05 = list(set(trainNanCol05)-set(['id','loan_dt','label','tag','tag_0','tag_1']))


# In[32]:


for col in trainNanCol05:
    train[col] =train.groupby(['tag'])[col].apply(lambda x: x.fillna(x.mean()))


# In[33]:


validNanCol05 = valid.columns[valid.isnull().mean()<0.5].tolist()


# In[34]:


validNanCol05 = list(set(trainNanCol05)-set(['id','loan_dt','label','tag','tag_0','tag_1']))


# In[35]:


for col in validNanCol05:
    valid[col] = valid.groupby(['tag'])[col].apply(lambda x: x.fillna(x.mean()))


# In[36]:


misCol, coefCol,lassoCol, lassoCvCol = selectFea(train[list(set(train.columns)-set(['loan_dt','tag','tag']))])
labelFea = list(set(misCol+ coefCol +lassoCol +lassoCvCol))
labelFea = list(set(labelFea+ ['tag_0','tag_1']))


# In[37]:


len(labelFea)


# In[38]:


train.to_pickle('./medFile/train(tag_fillNan_1).pickle')
valid.to_pickle('./medFile/test(tag_fillNan_1).pickle')


# In[39]:


def underSampleFold(data):
    '''
    对数据进行欠采样，平衡数据分布
    '''
    data.reset_index(drop=True,inplace=True)
    indexs=data.index.values
    labels =data.label.tolist()
    zeroIx=[ix for ix in indexs if labels[ix]==0]
    oneIx= [ix for ix in indexs if labels[ix]==1]
    print('len one Ix',len(oneIx))
    
    dataZeroDf = data.loc[zeroIx,:]
    dataZeroDf = dataZeroDf.sample(frac=1).reset_index(drop=True)
    
    data1 = pd.concat([dataZeroDf.loc[0:17860,:],data.loc[oneIx,:]],axis = 0)
    data2 = pd.concat([dataZeroDf.loc[17861:35722,:],data.loc[oneIx,:]],axis = 0)
    data3 = pd.concat([dataZeroDf.loc[35723:53583,:],data.loc[oneIx,:]],axis = 0)
    data4 = pd.concat([dataZeroDf.loc[53584:71445,:],data.loc[oneIx,:]],axis = 0)
    data5 = pd.concat([dataZeroDf.loc[71446:,:],data.loc[oneIx,:]],axis = 0)
    
#     newZeroIx= random.sample(zeroIx, int(len(oneIx)))
#     balanceIx= newZeroIx+ oneIx
    return data1, data2, data3, data4,data5 


# In[40]:


print(train.shape)

train1, train2, train3, train4, train5 = underSampleFold(train)
print(train1.shape)
print(train2.shape)
print(train3.shape)
print(train4.shape)
print(train5.shape)


# In[41]:


lgb_model = lgb.LGBMClassifier(
    boosting_type='gbdt', 
    num_leaves=32, 
    reg_alpha=0, 
    reg_lambda=0.1,
    max_depth=-1, 
    n_estimators=5000, 
    objective='binary',
    metric = ['auc', 'binary_logloss'],
    colsample_bytree=0.7, 
    subsample_freq=1,
    learning_rate=0.05, 
    random_state= 1028,
    n_jobs=-1,
    device = 'gpu',
    verbose_eval =10,
)
    


# In[42]:


i = 1
totalPred = valid[['id']]
totalPred['prob']= 0
trainList = [ train1, train2, train3, train4, train5]
for item in trainList:
    print('train ',i)
    i += 1
    train = item 
    train.reset_index(drop = True, inplace= True)
    print('train shape:',train.shape)
    Xtrain = train[labelFea]
    test = valid[labelFea]
    Ytrain = train[['label']]
    print('预测label prob')
    
    pred= valid[['id']]
    pred['prob'] = 0
    nfolds = 5
    skf = StratifiedKFold(n_splits= nfolds, random_state= 1028, shuffle=True)
    best_score = []
    bestAuc = []

    for index, (train_index, test_index) in enumerate(skf.split(Xtrain, Ytrain)):
        print('Folds:',index)
        lgb_model.fit(Xtrain.loc[train_index,:], Ytrain.loc[train_index],
                      eval_set=[(Xtrain.loc[train_index,:], Ytrain.loc[train_index]),
                                (Xtrain.loc[test_index,:], Ytrain.loc[test_index])], 
                                early_stopping_rounds=50,)
        best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
        print(best_score)
        bestAuc.append(lgb_model.best_score_['valid_1']['auc'])
        print(bestAuc)
        test_pred = lgb_model.predict_proba(test, num_iteration=lgb_model.best_iteration_)[:, 1]
        print('test mean:', test_pred.mean())
        pred['prob'] = pred['prob'] + test_pred

    print('best_score mean:',np.mean(best_score))
    print('best auc mean',np.mean(bestAuc))
    pred['prob'] = pred['prob'] / nfolds 
    print('best auc mean',np.mean(bestAuc))
    totalPred['prob'] +=  pred['prob']

totalPred['prob']= totalPred['prob']/len(trainList)


# In[ ]:





# In[43]:


# totalPred['prob']= totalPred['prob']/5


# In[45]:


print ("sub result")

totalPred=totalPred[['id','prob']]

totalPred.to_csv("./result/2lgb_test_1_"+version+"_(tag_fillNan).txt",index=None)
print('write result sucessfully！')


# In[44]:


totalPred.head(50)


# In[ ]:





# In[ ]:




