
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

sns.lmplot(data=non_outliers, x="X", y="Y", fit_reg=False)


## DayOfWeek
plt.figure(figsize = (12, 4))
dayofweek_list = ["Monday", "Tuesday", "Wednesday", "Thrusday", "Friday", "Saturday", "Sunday"] # 요일의 리스트를 미리 만들어 두면, 시각화 그래프를 이 순서대로 출력 가능하다
sns.countplot(data = train, x = "DayOfWeek", order = dayofweek_list)

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




### Preprocessing
test = pd.read_csv("test.csv", index_col = "Id")


## Dates
# train은 이미 진행함
test["Dates"] = pd.to_datetime(train["Dates"])

train["Dates-year"] = train["Dates"].dt.year
train["Dates-month"] = train["Dates"].dt.month
train["Dates-day"] = train["Dates"].dt.day
train["Dates-hour"] = train["Dates"].dt.hour
train["Dates-minute"] = train["Dates"].dt.minute
train["Dates-second"] = train["Dates"].dt.second




### Train
feature_names = ["X", "Y", "Dates-year", "Dates-month", "Dates-day", "Dates-hour", "Dates-minute", "Dates-second"]
label_name = "Category"

X_train = train[feature_names]
X_test = test[feature_names]
y_train = train[label_name]

# Random Forest
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 10,
                              n_jobs = -1,
                              random_state = 37)
                              

# Gradient Boosting Machine(LightGBM)
!conda install -c conda-forge -y lightgbm
import lightgbm as lgb
from lightgbm import LGBMModel,LGBMClassifier
from sklearn import metrics

Lgb = LGBMClassifier(n_estimators=90,
                    silent=False,
                    random_state=37,
                    max_depth=5,
                    num_leaves=31,
                    metrics='auc')
                    



### Evaluate
from sklearn.model_selection import train_test_split    # 데이터를 일정 비율로 두 개로 쪼개줌

# train 데이터를 train_kf, test_kf로 쪼개서 모델성능 평가
X_train_kf, X_test_kf, y_train_kf, y_test_kf = train_test_split(X_train, y_train, test_size = 0.3, random_state = 37)

# 학습(RF)
%time model.fit(X_train_kf, y_train_kf)
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
