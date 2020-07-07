# DS School - week1
1주차의 목표 : 탐험적 데이터 분석(Exploratory Data Analysis)을 통해서 데이터를 완벽히 이해한다!     
→ 그 중 Dates, X, Y에 대해 EDA 진행하는 방식을 설명해주고, DayOfWeek와 PdDistrict를 과제로 제시했다.
</br>
</br>

### 컬럼 설명
- Dates - 범죄가 발생한 날짜와 시간
- Category - 범죄의 세부적인 종류. train에만 존재 → label!
- Descript - 범죄의 세부정보. train에만 존재
- DayOfWeek - 범죄가 발생한 요일
- PdDistrict - 범죄를 관할하는 경찰서의 이름
- Resolution - 범죄의 상태, 범죄가 해결되었는지 여부. train에만 존재
- Address - 범죄가 발생한 구체적인 주소
- X - 범죄가 발생한 좌표의 경도
- Y - 범죄가 발생한 좌표의 위도
</br>
</br>

### Load Dataset
```python
import pandas as pd		# 분석용 패키지
import numpy as np		# 수학연산 패키지

train = pd.read_csv("train.csv")	# 데이터 읽어오기

print(train.shape)	# 데이터의 행렬 사이즈 출력. (row, column)
train.head()		# 데이터의 상위 5개 출력.
```
</br>

### Configuration
```python
import seaborn as sns	 	        # 시각화 패키지
import matplotlib.pyplot as plt 	# 시각화 패키지

%matplotlib inline	# 해당 코드로 jupyter에서 바로 시각화결과 확인가능
```
</br>

### 1. Dates 컬럼의 분석     
- 연,월,일,시,분,초를 나타내는 새로운 컬럼 생성. **dt 옵션 사용.**
```python
train["Dates"] = pd.to_datetime(train["Dates"])	# 데이터 타입을 DateTime으로 변환

train["Dates-year"] = train["Dates"].dt.year
train["Dates-month"] = train["Dates"].dt.month
train["Dates-day"] = train["Dates"].dt.day
train["Dates-hour"] = train["Dates"].dt.hour
train["Dates-minute"] = train["Dates"].dt.minute
train["Dates-second"] = train["Dates"].dt.second
```

- subplots: 여러개의 시각화를 한 화면에 띄운다
```python
figure, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
```

- 사이즈 설정: 18x8
```python
figure.set_size_inches(18,8)
```

**barplot**    
: date별 범죄발생빈도 
```python
sns.countplot(data=train, x="Dates-year", ax=ax1)
sns.countplot(data=train, x="Dates-month", ax=ax2)
sns.countplot(data=train, x="Dates-day", ax=ax3)
sns.countplot(data=train, x="Dates-hour", ax=ax4)
sns.countplot(data=train, x="Dates-minute", ax=ax5)
sns.countplot(data=train, x="Dates-second", ax=ax6)
```
![week1_01](https://user-images.githubusercontent.com/41939828/86716055-c9e4e680-c05b-11ea-888e-227a99e4ed0f.JPG)   
**> 시각화 결과해석**     
분, 초, 일(31일을 제외하면)은 범죄의 발생빈도를 판가름하는데 별 영향이 없다.     
**시간**은 범죄 발생빈도에 큰 영향이 있다.
</br>
</br>
</br>

### 2. X,Y 컬럼의 분석
**:exclamation: 좌표 시각화에는 (seaborn)lmplot 을 사용하는 것이 일반적**
```python
sns.lmplot(data=train, x="X", y="Y", fit_reg = False)
```
- fit_reg: 추세선 여부(True,False)
</br>

![week1_02](https://user-images.githubusercontent.com/41939828/86716079-d10bf480-c05b-11ea-8165-6786d2be77b0.JPG)        

outlier :     
오른쪽 위에 한 점으로 몰려있는 것으로 보아, 경도와 위도가 가장 높은 값이 outlier일 것으로 예상
```python
train["X"].max(), train["Y"].max()	# 가장 높은 값 확인

# outlier들 확인
X_outliers = (train["X"] == train["X"].max())
Y_outliers = (train["Y"] == train["Y"].max())
outlier = train[X_outliers & Y_outliers]    # x좌표와 y좌표 모두 outlier인 것을 확인

# outlier의 개수 확인. 매우 작은 비중을 차지하기 때문에 개선보다는 제거.
non_outliers = train[~(X_outliers & Y_outliers)]

sns.lmplot(data=non_outliers, x="X", y="Y", fit_reg=False)
```
![week1_03](https://user-images.githubusercontent.com/41939828/86716103-d6693f00-c05b-11ea-8fbe-f36ec4aef4b8.JPG)           
