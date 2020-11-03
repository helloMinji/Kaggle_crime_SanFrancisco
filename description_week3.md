# DS School - week3

### 1. EDA 추가 진행
- **Address**
``` python
train["Crossroad"] = train["Address"].str.contains("/")
sns.countplot(data=train, x="Crossroad")

plt.figure(figsize = (18, 64))
sns.countplot(data=train, hue="Crossroad", y="Category")
```
![add01](https://user-images.githubusercontent.com/41939828/97961654-11576580-1df7-11eb-89b9-f8599b1fd0fa.JPG)

특정 도로에 집중적으로 발생하는 범죄를 알 수 있다.</br>
교차점(Crossroad)과 그렇지 않은 부분(Block)이 특정 범죄에 따라 발생빈도의 차이가 있다.
</br>
</br>
</br>

### 2. 전처리 추가 진행
- **DayOfWeek**
``` python
train_dayofweek = pd.get_dummies(train["DayOfWeek"], prefix = "DayOfWeek")
train = pd.concat([train, train_dayofweek], axis = 1)

test_dayofweek = pd.get_dummies(test["DayOfWeek"], prefix = "DayOfWeek")
test = pd.concat([test, test_dayofweek], axis = 1)
```

- **PdDistrict**
``` python
train_pddistrict = pd.get_dummies(train["PdDistrict"], prefix = "PdDistrict")
train = pd.concat([train, train_pddistrict], axis = 1)

test_pddistrict = pd.get_dummies(test["PdDistrict"], prefix = "PdDistrict")
test = pd.concat([test, test_pddistrict], axis = 1)
```
pd.get_dummies()를 이용하여 **one-hot encoding** 을 더 간단한 방법으로 실행한다.</br>
옵션 중 prefix를 이용하여 새로 생성되는 더미 변수명을 지정할 수 있다.

- **Crossroad**
``` python
train["Crossroad"] = train["Address"].str.contains("/")

test["Crossroad"] = test["Address"].str.contains("/")
```
교차점(Crossroad)인지 아닌지에 따라 특정 범죄의 발생률에 차이가 있기 때문에 변수로 활용하기 위하여 새로운 컬럼 생성.

</br>
</br>

### 3. lightGBM
``` python
Lgb = LGBMClassifier(n_estimators=10, random_state=37)
```

기존의 코드로 돌렸더니 오류. </br>
parameter에 대해 더 알아본 후 수정하기로 하고, 제일 기본 상태로 수정.</br>
</br>
해당 모델로 변경 후 score가 눈에 띄게 좋아졌다! **= 현재 상위 40%**
