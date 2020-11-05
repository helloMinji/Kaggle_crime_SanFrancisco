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
