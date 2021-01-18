![image](https://user-images.githubusercontent.com/41939828/104958720-dcc97200-5a13-11eb-80b7-99214b44f7d8.png)

### Kaggle Competition | San Francisco Crime Classification
1934년부터 1963년까지, 샌프란시스코는 알카트라즈 섬에 세계에서 가장 악명 높은 범죄자들을 수용한 것으로 악명이 높았습니다.

오늘날, 이 도시는 범죄 과거보다 기술 분야로 더 잘 알려져 있습니다. 하지만 부의 불평등 증가, 주택 부족, 그리고 BART를 타고 출근하는 값비싼 디지털 장난감의 확산으로, 만 옆 도시에는 범죄의 부족이 없습니다.

Sunset에서 SOMA까지, Marina에서 Excelsior까지, 이 경쟁사의 데이터셋은 샌프란시스코의 모든 지역에서 약 12년간의 범죄 보고서를 제공합니다. 시간과 장소에 따라 발생한 범죄의 범주를 예측해야 합니다.

또한 데이터 세트를 시각적으로 살펴보시기 바랍니다. Top Criminals Map과 같은 시각화를 통해 이 도시에 대해 알 수 있는 것은 무엇일까요?

ㅡ[홈페이지](https://www.kaggle.com/c/sf-crime)에서 발췌

**ㅡDS School 온라인 심화반 학습내용**
</br>     
</br>     
</br>     

### 설치
1. 저장소 전체를 zip 파일로 다운로드하거나 오른쪽 코드를 터미널에서 실행 ```git clone https://github.com/helloMinji/Kaggle_crime_SanFrancisco.git```
2. 파이썬3 [설치](http://www.python.org/downloads)
3. cmd창에서 필요한 패키지 설치 ```pip install 패키지명```
4. sf_crime_course.py를 실행

#### 패키지
- pandas
- numpy
- seaborn : 시각화
- matplotlib.pyplot : 시각화
- lightgbm
- sklearn.model_selection
- sklearn.metrics : score 계산
- sklearn.ensemble
</br>     

### 코드 정보
#### 데이터 처리
- Pandas로 데이터 가져 오기
- 데이터 정리(전처리, one-hot encoding, CSR Matrix)
- matplotlib로 시각화
#### 데이터 분석
- 하이퍼파라미터 튜닝
- LightGBM
#### 분석의 평가
- 로컬에서 결과를 평가하기위한 log_loss 계산
- 결과를 제출용 파일로 출력
</br>                 

분석 과정에 대한 자세한 내용은 [EDA](https://hellominji.tistory.com/42)와 [분석](https://hellominji.tistory.com/43)을 참고하세요.
