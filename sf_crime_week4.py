
### Load Dataset
import pandas as pd
import numpy as np

train = pd.read_csv("train.csv")

print(train.shape)
train.head()




### Configuration
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline


## Dates 컬럼의 분석
train["Dates"] = pd.to_datetime(train["Dates"])

train["Dates-year"] = train["Dates"].dt.year
train["Dates-month"] = train["Dates"].dt.month
train["Dates-day"] = train["Dates"].dt.day
train["Dates-hour"] = train["Dates"].dt.hour
train["Dates-minute"] = train["Dates"].dt.minute
train["Dates-second"] = train["Dates"].dt.second

print(train.shape)
train[["Dates","Dates-year","Dates-month","Dates-day","Dates-hour","Dates-minute","Dates-second"]].head()

# 시각화
figure, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)

figure.set_size_inches(18,8)

sns.countplot(data=train, x="Dates-year", ax=ax1)
sns.countplot(data=train, x="Dates-month", ax=ax2)
sns.countplot(data=train, x="Dates-day", ax=ax3)
sns.countplot(data=train, x="Dates-hour", ax=ax4)
sns.countplot(data=train, x="Dates-minute", ax=ax5)
sns.countplot(data=train, x="Dates-second", ax=ax6)


## X,Y 컬럼의 분석
sns.lmplot(data=train, x="X", y="Y", fit_reg = False)

# outlier
train["X"].max(), train["Y"].max()

X_outliers = (train["X"] == train["X"].max())
Y_outliers = (train["Y"] == train["Y"].max())
outlier = train[X_outliers & Y_outliers]

non_outliers = train[~(X_outliers&Y_outliers)]

# 시각화
sns.lmplot(data=non_outliers, x="X", y="Y", fit_reg=False)


## DayOfWeek
plt.figure(figsize = (12, 4))
dayofweek_list = ["Monday", "Tuesday", "Wednesday", "Thrusday", "Friday", "Saturday", "Sunday"] # 요일의 리스트를 미리 만들어 두면, 시각화 그래프를 이 순서대로 출력 가능하다
sns.countplot(data = train, x = "DayOfWeek", order = dayofweek_list)

# 시각화
figure, axes = plt.subplots(nrows = 10, ncols = 4)
figure.set_size_inches(30, 48)

category_list = train["Category"].value_counts().index

for row in range(10):
    for column in range(4):
        index = row * 4 + column
        
        if index < len(category_list):
            ax = axes[row][column]
            category = category_list[index]
        
            target = train[train["Category"] == category]
            sns.countplot(data = target, x = "DayOfWeek", order = dayofweek_list, ax = ax)

            ax.set(xlabel = category)


## PdDistrict
plt.figure(figsize = (12, 4))
sns.countplot(data = train, x = "PdDistrict")

figure, axes = plt.subplots(nrows = 10, ncols = 4)
figure.set_size_inches(30, 48)

category_list = train["Category"].value_counts().index

for row in range(10):
    for column in range(4):
        index = row * 4 + column
        
        if index < len(category_list):
            ax = axes[row][column]
            category = category_list[index]
        
            target = train[train["Category"] == category]
            sns.countplot(data = target, x = "PdDistrict", ax = ax)

            ax.set(xlabel = category)


## Address
train["Crossroad"] = train["Address"].str.contains("/")
sns.countplot(data=train, x="Crossroad")

plt.figure(figsize = (18, 64))
sns.countplot(data=train, hue="Crossroad", y="Category")
            



### Preprocessing
test = pd.read_csv("test.csv", index_col = "Id")


## Dates
# train은 이미 진행함
test["Dates"] = pd.to_datetime(test["Dates"])

test["Dates-year"] = test["Dates"].dt.year
test["Dates-month"] = test["Dates"].dt.month
test["Dates-day"] = test["Dates"].dt.day
test["Dates-hour"] = test["Dates"].dt.hour
test["Dates-minute"] = test["Dates"].dt.minute
test["Dates-second"] = test["Dates"].dt.second

# 분에서 30을 뺀후 절댓값
train["Dates-minute(abs)"] = np.abs(train["Dates-minute"]-30)
test["Dates-minute(abs)"] = np.abs(test["Dates-minute"]-30)


## DayOfWeek
# one hot encoding
train_dayofweek = pd.get_dummies(train["DayOfWeek"], prefix = "DayOfWeek")
train = pd.concat([train, train_dayofweek], axis = 1)

test_dayofweek = pd.get_dummies(test["DayOfWeek"], prefix = "DayOfWeek")
test = pd.concat([test, test_dayofweek], axis = 1)


## PdDistrict
train_pddistrict = pd.get_dummies(train["PdDistrict"], prefix = "PdDistrict")
train = pd.concat([train, train_pddistrict], axis = 1)

test_pddistrict = pd.get_dummies(test["PdDistrict"], prefix = "PdDistrict")
test = pd.concat([test, test_pddistrict], axis = 1)


## Crossroad
train["Crossroad"] = train["Address"].str.contains("/")     # 특정 문자가 포함되는지 확인
test["Crossroad"] = test["Address"].str.contains("/")

# crossroad 순서 통일
def clean_address(address):
    if "/" not in address:
        return address
    
    address1, address2 = address.split("/")
    address1, address2 = address1.strip(), address2.strip()    # 공백 제거
    
    if address1<address2:
        address = "{} / {}".format(address1, address2)
    else:
        address = "{} / {}".format(address2, address1)
    
    return address

train["Address(clean)"] = train["Address"].apply(clean_address)
test["Address(clean)"] = test["Address"].apply(clean_address)

# 발생횟수가 작은 주소에 대해 Others 처리
address_counts = train["Address(clean)"].value_counts()

top_address_counts = address_counts[address_counts >= 100]
top_address_counts = top_address_counts.index

train.loc[~train["Address(clean)"].isin(top_address_counts), "Address(clean)"] = "Others"
test.loc[~test["Address(clean)"].isin(top_address_counts), "Address(clean)"] = "Others"

# one-hot encoding -> CSR Matrix
train_address = pd.get_dummies(train["Address(clean)"])
test_address = pd.get_dummies(test["Address(clean)"])

from scipy.sparse import csr_matrix
train_address = csr_matrix(train_address)
test_address = csr_matrix(test_address)
train.head()




### Train
feature_names = ["X", "Y", "Dates-year", "Dates-month", "Dates-day", "Dates-hour", "Dates-minute(abs)", "Dates-second"]
feature_names = feature_names + list(train_dayofweek.columns)
feature_names = feature_names + list(train_pddistrict.columns)
label_name = "Category"

X_train = train[feature_names]
X_test = test[feature_names]
y_train = train[label_name]

# CSR Matrix 합치기 - hstack
from scipy.sparse import hstack

X_train = hstack([X_train.astype('float'), train_address])
X_train = csr_matrix(X_train)
X_test = hstack([X_test.astype('float'), test_address])
X_test = csr_matrix(X_test)




### Hyperparameter Tuning
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# Coarse Search
X_train_kf, X_test_kf, y_train_kf, y_test_kf = \
    train_test_split(X_train, y_train, test_size = 0.3, random_state = 37)
n_estimators = 100
num_loop = 100
early_stopping_rounds = 20

coarse_hyperparameters_list = []

for loop in range(num_loop):
    learning_rate = 10 ** np.random.uniform(low = -10, high = 1)
    num_leaves = np.random.randint(2, 500)
    max_bin = np.random.randint(2, 500)
    min_child_samples = np.random.randint(2, 500) 
    subsample = np.random.uniform(low = 0.1, high = 1.0)
    colsample_bytree = np.random.uniform(low = 0.1, high = 1.0)
    
    model = LGBMClassifier(n_estimators = n_estimators,
                           learning_rate = learning_rate,
                           num_leaves = num_leaves,
                           max_bin = max_bin,
                           min_child_samples = min_child_samples,
                           subsample = subsample,
                           subsample_freq = 1,
                           colsample_bytree = colsample_bytree,
                           class_type = 'balanced',
                           random_state = 37)
    
    model.fit(X_train_kf, y_train_kf,
              eval_set = [(X_test_kf, y_test_kf)],
              verbose = 0,
              early_stopping_rounds = early_stopping_rounds)
    
    best_iteration = model.best_iteration_
    score = model.best_score_['valid_0']['multi_logloss']
    
    coarse_hyperparameters_list.append({
        'loop': loop,
        'n_estimators': best_iteration,
        'learning_rate': learning_rate,
        'num_leaves': num_leaves,
        'max_bin': max_bin,
        'min_child_samples': min_child_samples,
        'subsample': subsample,
        'subsample_freq': 1,
        'colsample_bytree': colsample_bytree,
        'class_type': 'balanced',
        'random_state': 37,
        'score': score,
    })
    
    print(f"{loop:2} best iteration = {best_iteration} Score = {score:.5f}")
    
coarse_hyperparameters_list = pd.DataFrame(coarse_hyperparameters_list)
coarse_hyperparameters_list = coarse_hyperparameters_list.sort_values(by = "score")
coarse_hyperparameters_list.head()
    

# Finer Search (coarse와 앞부분 동일)
finer_hyperparameters_list = []

for loop in range(num_loop):
    # coarse search에서 찾은 값
    learning_rate = np.random.uniform(low = 0.025100, high = 0.030819)
    num_leaves = np.random.randint(158, 278)
    max_bin = np.random.randint(173, 418)
    min_child_samples = np.random.randint(228, 448) 
    subsample = np.random.uniform(low = 0.504958, high = 0.901055)
    colsample_bytree = np.random.uniform(low = 0.860466, high = 0.989937)
    
    model = LGBMClassifier(n_estimators = n_estimators,
                           learning_rate = learning_rate,
                           num_leaves = num_leaves,
                           max_bin = max_bin,
                           min_child_samples = min_child_samples,
                           subsample = subsample,
                           subsample_freq = 1,
                           colsample_bytree = colsample_bytree,
                           class_type = 'balanced',
                           random_state = 37)
    
    model.fit(X_train_kf, y_train_kf,
              eval_set = [(X_test_kf, y_test_kf)],
              verbose = 0,
              early_stopping_rounds = early_stopping_rounds)
    
    best_iteration = model.best_iteration_
    score = model.best_score_['valid_0']['multi_logloss']
    
    finer_hyperparameters_list.append({
        'loop': loop,
        'n_estimators': best_iteration,
        'learning_rate': learning_rate,
        'num_leaves': num_leaves,
        'max_bin': max_bin,
        'min_child_samples': min_child_samples,
        'subsample': subsample,
        'subsample_freq': 1,
        'colsample_bytree': colsample_bytree,
        'class_type': 'balanced',
        'random_state': 37,
        'score': score,
    })

    print(f"{loop:2} best iteration = {best_iteration} Score = {score:.5f}")   

finer_hyperparameters_list = pd.DataFrame(finer_hyperparameters_list)
finer_hyperparameters_list = finer_hyperparameters_list.sort_values(by = "score")
finer_hyperparameters_list.head()

best_hyperparameters = finer_hyperparameters_list.iloc[0]

    
    

# Random Forest
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 10,
                              n_jobs = -1,
                              random_state = 37)
                              

# Gradient Boosting Machine(LightGBM)
#!conda install -c conda-forge -y lightgbm
import lightgbm as lgb
#from lightgbm import LGBMModel,LGBMClassifier
from sklearn import metrics

# best hyperparameter 적용
Lgb = LGBMClassifier(n_estimators = best_hyperparameters['n_estimators'],
                       learning_rate = best_hyperparameters['learning_rate'],
                       num_leaves = best_hyperparameters['num_leaves'],
                       max_bin = best_hyperparameters['max_bin'],
                       min_child_samples = best_hyperparameters['min_child_samples'],
                       subsample = best_hyperparameters['subsample'],
                       subsample_freq = best_hyperparameters['subsample_freq'],
                       colsample_bytree = best_hyperparameters['colsample_bytree'],
                       class_type = best_hyperparameters['class_type'],
                       random_state = best_hyperparameters['random_state'])
                    



### Evaluate
# train 데이터를 train_kf, test_kf로 쪼개서 모델성능 평가
X_train_kf, X_test_kf, y_train_kf, y_test_kf = train_test_split(X_train, y_train, test_size = 0.3, random_state = 37)

# 학습(RF)
%time model.fit(X_train_kf, y_train_kf, early_stopping_rounds = 100, eval_metric = "logloss", eval_set = [(X_test_kf, y_test_kf)], verbose = True)
# 학습(LightGBM)
%time Lgb.fit(X_train_kf, y_train_kf)

# 범죄가 발생할 확률(RF)
y_predict_test_kf = model.predict_proba(X_test_kf)
# 범죄가 발생할 확률(LightGBM)
y_predict_test_kf = Lgb_model.predict_proba(X_test_kf)

# score 계산: 캐글에 업로드하지 않아도 순위 예상 가능
from sklearn.metrics import log_loss

score = log_loss(y_test_kf, y_predict_test_kf)
print(f"Score = {score:.5f}")




### Predict
%time model.fit(X_train, y_train)
prediction_list = model.predict_proba(X_test)   # 진짜 test로 예측




### Submit
sample_submission = pd.read_csv("sampleSubmission.csv", index_col = "Id")

submission = pd.DataFrame(prediction_list,
                          index = sample_submission.index,
                          columns = model.classes_)

submission.to_csv("baseline-script.csv")
