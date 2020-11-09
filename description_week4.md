# DS School - week4

### 1. 전처리 추가 진행
- **Dates-minute**
``` python 
train["Dates-minute(abs)"] = np.abs(train["Dates-minute"] - 30)
test["Dates-minute(abs)"] = np.abs(test["Dates-minute"] - 30)
```
분은 자세히 기록되어 있지 않기 때문에, 분에 30을 뺀 후 절대값(absolute)을 씌운다. </br>
이렇게 하면 0분을 기존으로 상대적인 플러스/마이너스치를 알 수 있다. </br>
</br>
- **Address**
``` python
def clean_address(address):
    if "/" not in address:
        return address

    address1, address2 = address.split("/")
    address1, address2 = address1.strip(), address2.strip()    # strip(): 공백 제거

    if address1 < address2:
        address = "{} / {}".format(address1, address2)
    else:
        address = "{} / {}".format(address2, address1)

    return address

train["Address(clean)"] = train["Address"].apply(clean_address)
test["Address(clean)"] = test["Address"].apply(clean_address)
```
**crossroad 이름 통일** </br>
교차로 이름이 a/b, b/a로 되어있는 경우가 있기 때문에 </br>
주소값을 비교하여 알파벳이 빠를수록 작은 값이라고 가정하고, 작은 값을 앞으로, 큰 값을 뒤로 배치한다. </br>


``` python
address_counts = train["Address(clean)"].value_counts()

top_address_counts = address_counts[address_counts >= 100]
top_address_counts = top_address_counts.index

train.loc[~train["Address(clean)"].isin(top_address_counts), "Address(clean)"] = "Others"
test.loc[~test["Address(clean)"].isin(top_address_counts), "Address(clean)"] = "Others"
```
**발생횟수가 작은 주소에 대해 Others 처리** </br>
주소값을 많은 순으로 정렬하고, 주소 발생 횟수가 100회 이상인 데이터만 가져온다. </br>
100회 미만 주소들은 Others로 값을 통일한다.
</br>
</br>
</br>
</br>

### 2. CSR Matrix
- 전처리
``` python
train_address = pd.get_dummies(train["Address(clean)"])
test_address = pd.get_dummies(test["Address(clean)"])

from scipy.sparse import csr_matrix
train_address = csr_matrix(train_address)
test_address = csr_matrix(test_address)
```
메모리의 효율적인 사용을 위해 one-hot encoding 후 CSR Matrix로 변환한다.
</br>
</br>
- 모델 생성
``` python
from scipy.sparse import hstack

X_train = hstack([X_train.astype('float'), train_address])
X_train = csr_matrix(X_train)

X_test = hstack([X_test.astype('float'), test_address])
X_test = csr_matrix(X_test)
```
hstack: CSR Matrix를 하나로 합친다. </br>
-> 이를 이용해서 원래 train 데이터와 CSR Matrix를 합친다. </br>
-> 이를 다시 CSR Matrix로 변환한다. </br>
</br>
</br>
</br>

### 3. Hyperparameter Tuning: Coarse & Fine Search
- Coarse Search </br>
Random Search를 하되, 이론상으로 존재 가능한 모든 하이퍼패러미터 범위를 집어넣는다.
가장 좋은 하이퍼패러미터를 찾는 것은 어렵지만, 좋지 않은 하이퍼패러미터를 정렬해서 후순위로 놓을 수 있다.
- Fine Search </br>
Coarse Search를 통해 좋지 않은 하이퍼패러미터를 버린 뒤 다시 한 번 Random Search 진행.
</br>

**■ Coarse Search**
</br>  1. 필요한 변수 세팅
``` python
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss   # 해당 대회의 측정공식

X_train_kf, X_test_kf, y_train_kf, y_test_kf = \
    train_test_split(X_train, y_train, test_size = 0.3, random_state = 37)

n_estimators = 100	# 트리의 개수
num_loop = 100		# 랜덤서치 반복횟수
early_stopping_rounds = 20
```

2. 여러 파라미터로 모델 생성 및 학습
``` python
coarse_hyperparameters_list = []

for loop in range(num_loop):
    # 이론 상으로 존재하는 모든 하이퍼패러미터 범위 탐색
    learning_rate = 10 ** np.random.uniform(low = -10, high = 1)
    num_leaves = np.random.randint(2, 500)
    max_bin = np.random.randint(2, 500)
    min_child_samples = np.random.randint(2, 500) 
    subsample = np.random.uniform(low = 0.1, high = 1.0)
    colsample_bytree = np.random.uniform(low = 0.1, high = 1.0)
    
    # LGBMClassifier
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
    
    # 모델 학습
    model.fit(X_train_kf, y_train_kf,
              eval_set = [(X_test_kf, y_test_kf)],
              verbose = 0,
              early_stopping_rounds = early_stopping_rounds)

    # 가장 좋은 점수와 이에 해당하는 n_estimators를 저장
    best_iteration = model.best_iteration_
    score = model.best_score_['valid_0']['multi_logloss']
    
    # hyperparameter 탐색 결과를 리스트에 저장
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

    # hyperparameter 탐색 결과 출력
    print(f"{loop:2} best iteration = {best_iteration} Score = {score:.5f}")
```
  
3. 최적 파라미터 확인
```
coarse_hyperparameters_list = pd.DataFrame(coarse_hyperparameters_list)
coarse_hyperparameters_list = coarse_hyperparameters_list.sort_values(by = "score")
coarse_hyperparameters_list.head()
```
</br>
</br>

**■ Fine Search**
</br>1. 필요한 변수 세팅: Coarse Search와 동일
</br>2. 여러 파라미터로 모델 생성 및 학습
``` python
finer_hyperparameters_list = []

for loop in range(num_loop):
    # Coarse Search를 통해 범위를 좁힌 하이퍼패러미터
    learning_rate = np.random.uniform(low = 0.030977, high = 0.047312)
    num_leaves = np.random.randint(16, 483)
    max_bin = np.random.randint(135, 454)
    min_child_samples = np.random.randint(111, 482) 
    subsample = np.random.uniform(low = 0.411598, high = 0.944035)
    colsample_bytree = np.random.uniform(low = 0.603785, high = 0.929522)
    
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
```

3. 최적 파라미터 확인
``` python
finer_hyperparameters_list = pd.DataFrame(finer_hyperparameters_list)
finer_hyperparameters_list = finer_hyperparameters_list.sort_values(by = "score")
finer_hyperparameters_list.head()
```
</br>
</br>

**fine search로 찾은 가장 좋은 파라미터를 이용하여 최종모델을 생성**

``` python
best_hyperparameters = finer_hyperparameters_list.iloc[0]

model = LGBMClassifier(n_estimators = best_hyperparameters['n_estimators'],
                       learning_rate = best_hyperparameters['learning_rate'],
                       num_leaves = best_hyperparameters['num_leaves'],
                       max_bin = best_hyperparameters['max_bin'],
                       min_child_samples = best_hyperparameters['min_child_samples'],
                       subsample = best_hyperparameters['subsample'],
                       subsample_freq = best_hyperparameters['subsample_freq'],
                       colsample_bytree = best_hyperparameters['colsample_bytree'],
                       class_type = best_hyperparameters['class_type'],
                       random_state = best_hyperparameters['random_state'])
```
