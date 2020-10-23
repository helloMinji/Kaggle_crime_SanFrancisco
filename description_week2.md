# DS School – week2
2주차의 목표: 
- 1주차의 고도화 (탐험적 데이터분석(EDA))
- 전처리 일부
- Train & Predict
</br>
</br>

### 1주차 평가
- Dates에 대한 시각화 결과에서, ‘초’는 중요하지 않고 ‘시간’과 ‘분’은 중요하다고 했으므로 2가지 외에 나머지는 모델에서 제외하기로 한다.
- x, y는 이상치를 제거해야 모델의 성능이 더 좋아지기 때문에, 모델 생성 이전에 outlier를 제거한다.
</br>
</br>

### 1. DayOfWeek 컬럼의 분석
- **요일별 범죄발생수 확인**
``` python
plt.figure(figsize = (12, 4))
dayofweek_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
sns.countplot(data = train, x = "DayOfWeek", order = dayofweek_list)
```
![image](https://user-images.githubusercontent.com/41939828/97017654-262b3200-1589-11eb-92e8-9a08e3fa9440.png)

위의 결과로는 큰 차이가 보이지 않는다.</br>
→ 발생하는 모든 종류의 범죄를 합친 상태이므로, 범죄별로 요일별 발생 비율을 확인하자.

- **범죄별 요일별 발생수**
``` python
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
```
![image](https://user-images.githubusercontent.com/41939828/97017692-32af8a80-1589-11eb-96f0-ccedee037330.png)
![image](https://user-images.githubusercontent.com/41939828/97017724-3c38f280-1589-11eb-8bfd-43db27cf8a6f.png)

 
범죄별로 나눠서 요일별 발생수를 확인하니 주말에 많이 발생하는 범죄가 있고, 주중에 많이 발생하는 범죄가 있다.</br>
주말을 금토로 하느냐, 토일로 하느냐에 따라서 그 분류가 달라진다.</br>
주중에는 전체적으로 발생수가 많은지, 특정 요일에 많은지에 따라서도 그 분류가 달라진다.</br>
분류에 따라 가중치를 줄 수 있는 컬럼을 추가 생성하면 모델이 더욱 개선된다.
</br>
</br>
</br>
</br>
</br>


### 2. PdDistrict 컬럼의 분석
DayOfWeek에서처럼 각 범죄마다 관할경찰서별 범죄발생수를 확인한다.
``` python
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
```
![image](https://user-images.githubusercontent.com/41939828/97017763-465af100-1589-11eb-991d-6ced81b4d8c9.png)
![image](https://user-images.githubusercontent.com/41939828/97017799-4fe45900-1589-11eb-8041-0f4e39f0ceca.png)
 
관할경찰서마다 발생빈도가 높은 범죄종류가 명확하게 나오는 곳이 많다. 그래서 단순히 One Hot Encoding해서 넣어줘도 효과가 좋을 것이다.
</br>
</br>
</br>
</br>
</br>

### 3. 전처리
- **Dates**
시각화를 위해서 나눴던 것처럼 연, 월, 일, 시, 분, 초로 Dates 컬럼을 쪼개서 각각의 컬럼으로 생성한다.
``` python
train["Dates-year"] = train["Dates"].dt.year
train["Dates-month"] = train["Dates"].dt.month
train["Dates-day"] = train["Dates"].dt.day
train["Dates-hour"] = train["Dates"].dt.hour
train["Dates-minute"] = train["Dates"].dt.minute
train["Dates-second"] = train["Dates"].dt.second
```
</br>
</br>


### 4. Train
- **변수 세팅**
``` python
feature_names = ["X", "Y", "Dates-year", "Dates-month", "Dates-day", "Dates-hour", "Dates-minute", "Dates-second"]
label_name = “Category”

X_train = train[feature_names]
X_test = test[feature_names]
y_train = train[label_name]
```

- **모델 생성**
``` python
# Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 10, n_jobs = -1, random_state = 37)
```
- n_estimators: 트리의 개수. 더 만들었더니 컴퓨팅 파워가 딸려서 시간이 오래 걸리고, 모델의 성능이 굉장히 좋아지는 것도 아니어서 10으로 유지.
- n_jobs: 병렬처리 여부. -1이면 컴퓨터에 존재하는 모든 코어를 활용한다.
- random_state: 랜덤포레스트의 결과가 랜덤하게 나오는 것을 고정. seed number라고도 한다. 고정해야 score의 비교가 의미있는 과정이 된다. 여기서는 37로 고정한 것으로 숫자 자체는 큰 의미 없음.

``` python
# Gradient Boosting Machine 중에서도 “LightGBM”
# 설치법: !conda install -c comda-forge -y lightgbm

import lightgbm as lgb
from lightgbm import LGBMModel, LGBMClassifier
from sklean import metrics
Lgb = LGBMClassifier(n_estimators-90, silent=False, random_state=37, max_depth=5, num_leaves=31, metrics=’auc’)
```
</br>
</br>


### 5. Evaluate
컴페티션의 score 방식을 확인하고 이를 코드로 구현하여 파일을 submission하지 않고 대략적인 score를 확인한다.

- **평가를 위한 변수 세팅**
``` python
from sklearn.model_selection import train_test_split
X_train_kf, test_kf, y_train_kf, y_test_kf = train_test_split(X_train, y_train, test_size = 0.3, random_state = 37)
```

- **모델 학습**
``` python
# random forest 모델 학습
model.fit(X_train_kf, y_train_kf)
```
앞에 %time을 붙여주면 해당 코드를 실행하는데 얼마나 걸렸는지 알 수 있다.

- **평가를 위한 예측**
``` python
y_predict_test_kf = model.predict_proba(X_test_kf)
```
:question: 예측을 predict가 아닌 predict_proba로 하는 이유? </br>
:exclamation: 이 컴페티션에서 요구하는 예측 결과가 하나만 예측해라!가 아닌 각각이 될 확률로 되어있기 때문!

- **평가**
``` python
from sklearn.metrics import log_loss
score = log_loss(y_test_kf, y_predict_test_kf)
```
</br>
</br>


### 6. Predict
score가 만족스럽게 나왔다면 진짜 train으로 모델을 학습시켜서 진짜 test를 예측한다.
``` python
model.fit(X_train, y_train)
prediction_list = model.predict_proba(X_test)
```
</br>
</br>


### 7. Submit
``` python
sample_submisson = pd.read_csv(“sampleSubmission.csv”, index_col = “Id”)
# index를 가져다 쓰면 쉽게 모양을 만들 수 있다.

submission = pd.DataFrame(prediction_list, index = sample_submission.index, columns = model.classes)
submission.to_csv(“파일이름.csv”)
```
