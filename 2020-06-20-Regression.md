# 회귀
=> 2개의 변수(열, feature)간의 예측 관계에 있어서 한 변수에 의해서 예측되는 다른 변수의 예측치들이 그 변수의 평균치로 회귀하는 경향이 있다

=> 하나의 종속 변수(label, target)와 한 개 이상의 독립 변수(feature, 설명 변수)와의 관계를 모델링 하는데 종속 변수의 값이
=> 범주형이면 Classification(분류)이라고 하고 연속형 숫자 형태이면 Regression(회귀)이라고 합니다.
=> 설명 변수의 개수가 1개이면 단변량 회귀라고 하고 2개 이상이면 다변량 회귀 라고 합니다.
=> 결정 경계(수식)의 모양이 선형인지 비선형인지로 분류합니다.

# 선형회귀 - 실제 값과 예측 값의 차이(오류의 제곱값)를 최소화하는 직선형 회귀선을 최적화하는 회귀

- 선형 회귀의 과적합 문제를 해결하기 위해서 규제(Regularization)를 가하는 방법에 따라 3가지로 나눌 수 있습니다.

- Ridge(L2 규제 - 회귀 계수의 예측 영향도를 감소시키는 방식), Lasso(L1규제 - 회귀 계수의 예측 영향도를 제거하는 방식), ElasticNet(L1과 L2를 모두 사용하는 방식) - 변수가 많을 때는 제거하기도 하고 영향력을 줄이기도 함)

- MAE(Mean Absolute Error 평균 제곱 오차): 실제 값과 오차 값 제곱의 평균
=> Sklearn 에서는 숫자의 값이 높은 것이 좋다고 판정하는 것이 일반적이므로 음의 MSE를 취함 - neg_mean_sqared_error

- RMSE(Root MSE): MSE 의 값이 너무 커서 MSE 에 루트를 취한 값
=> MSE나 RMSE에 로그를 취한 값을 사용하는 MSLE나 RMSLE도 있습니다.

- R**2(결정 계수): 타겟의 분산을 측정 - 실제 값과 예측값 사이의 제곱을 해서 분산을 계산 : 1.0에 가까울 수록 좋은 모델입니다.

## 공통 코드


```python
# 파이썬 ≥3.5 필수
import sys
assert sys.version_info >= (3, 5)

# 사이킷런 ≥0.20 필수
import sklearn
assert sklearn.__version__ >= "0.20"

# 공통 모듈 임포트
import numpy as np
import pandas as pd
import seaborn as sns

import os

# 노트북 실행 결과를 동일하게 유지하기 위해
np.random.seed(42)

# 깔끔한 그래프 출력을 위해
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# 그림을 저장할 위치
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "training_linear_models"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

import platform
from matplotlib import font_manager, rc

#매킨토시의 경우
if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
#윈도우의 경우
elif platform.system() == 'Windows':
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)

mpl.rcParams['axes.unicode_minus'] = False


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("그림 저장:", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
import warnings
warnings.filterwarnings(action='ignore')

```

## 평가 지표 사용해보기


```python
#데이터 생성을 위한 함수
from sklearn.datasets import make_regression
#교차 검증을 위한 함수
from sklearn.model_selection import cross_val_score
#선형 회귀 모델을 만들기 위한 클래스
from sklearn.linear_model import LinearRegression

#회귀 모델의 샘플 데이터 생성 - 특성 행렬 과 타겟 벡터를 생성
#n_samples 는 샘플 데이터 개수
#n_features는 독립 변수 개수
#n_informative는 상관 관계를 가질 독립 변수의 개수
#n_targets 는 종속 변수 개수
#noise 는 정규 분포에 적용할 편차
#coef 는 상관 관계를 출력할 것인 지 여부

#결과는 2개나 3개(coef 가 True 인 경우)
#첫번째는 독립 변수 샘플 이고 두번째 종속 변수 샘플
features, target = make_regression(n_samples=100, n_features=3, n_informative=3, 
                                  n_targets=1, noise=50, coef=False, 
                                   random_state = 1)

#선형 회귀 모델을 생성
ols = LinearRegression()

#MSE 를 출력
print(cross_val_score(ols, features, target, scoring='neg_mean_squared_error'))
#r2를 출력
print(cross_val_score(ols, features, target, scoring='r2'))
```

    [-1974.65337976 -2004.54137625 -3935.19355723 -1060.04361386
     -1598.74104702]
    [0.8622399  0.85838075 0.74723548 0.91354743 0.84469331]


## 정규 방정식을 이용한 회귀 계수 찾기 - MSE 를 최소화하는 방식

- 정규 방정식을 이용하는 선형 회귀 클래스 - sklearn.linear_model.LiearRegression
=> 모델을 학습하고 intercept_ 와 coef_ 를 이용하면 절편과 기울기를 확인할 수 있습니다. / fit 메소드로 훈련하고 predict 로 예측

- Sklearn 에서의 선형 회귀 - 특잇값 분해(ingular Value Decomposition - SVD): 표준 행렬 분해기법을 이용해서 계산

- 복잡도 - 역행렬을 구하는 복잡도는 (n+1) 이고 정규 방정식은 역행렬에 제곱이라고 (n+1)을 제곱합니다. n**2 + 2n + 1
=> 복잡도는 차수가 가장 높은 항만 취합니다. - n**2
=> 복잡도를 계산할 떄 계수는 무시합니다.
=> 특성의 수가 2배 늘어나면 4 ~ 8 배 정도의 시간이 증가합니다.
=> 특성의 수가 늘어나면 느려집니다.

- 샘플의 개수가 늘어나면 늘어나는 것은 선형 - n : 메모리만 허락하면 데이터가 많은 것은 별 문제가 되지 않습니다.


```python
#샘플 데이터 생성
X = 2 * np.random.rand(100, 1)
#타겟 데이터 생성
#절편은 4 이고 기울기는 3을 적용해서 데이터를 생성하는데 잡음을 섞어 줍니다.
y = 4 + 3 * X + np.random.randn(100, 1) 
#데이터 분포 확인
plt.plot(X, y, "b.")
plt.xlabel("x_1", fontsize=16)
plt.ylabel("y", fontsize=16, rotation=0)
plt.axis([0,2, 0, 15])
save_fig('잡음이 섞인 선형회귀 분포')
plt.show()
```

    그림 저장: 잡음이 섞인 선형회귀 분포



    
![png](output_7_1.png)
    



```python
#모든 샘플 데이터에 1을 추가
X_b = np.c_[np.ones((100, 1)), X]
#print(X_b)

#정규 방정식 수행
#독립 변수의 역행렬을 구해서 종속 변수 와 행렬의 곱을 합니다.
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(theta_best)
#첫번째 데이터가 절편(intercept) 이고 두번째 데이터가 기울기(slope)
```

    [[4.21509616]
     [2.77011339]]



```python
#예측
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
print(y_predict)
```

    [[4.21509616]
     [9.75532293]]



```python
# 사이킷런을 이용해서 모델을 학습하고 결과를 확인하고 예측
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print("절편:", lin_reg.intercept_)
print("기울기:", lin_reg.coef_)

#예측
print(lin_reg.predict(X_new))
```

    절편: [4.21509616]
    기울기: [[2.77011339]]
    [[4.21509616]
     [9.75532293]]


## 경사 하강법 직접 구현 - 기본적인 아이디어는 비용 함수를 최소화 하기 위해서 파리마터를 조정해 가는 것

- 가장 중요한 파라미터가 학습률인데 학습률은 step 의 간격입니다.
- 이 값이 너무 작으면 최적의 모델을 찾을 가능성은 높아지지만 알고리즘을 학습하는데 걸리는 시간이 길어집니다.
- 이 값이 너무 크면 학습하는 시간은 짧아지겠지만 최적의 모델을 찾지 못할 수 있습니다.

- 모든 feature 들의 범위가 같아야 합니다.
- 피처들의 범위가 다르다는 의미는 여러 봉우리가 있는 것과 같습니다.
- 경사 하강법을 적용하기 전에 scailig 을 해주어야 합니다.

### 배치(Batch - 일괄) 경사 하강법 - 모든 데이터를 이용해서 gradient 를 구함 - 학습률에 따라 순서대로 적용해서 최적의 비용함수를 찾아가는 방식


```python
#학습률
eta = 0.1
#반복 횟수
n_iterations = 1000
#임의 기울기
m = 100

#결과를 저장하기 위한 배열 - 처음에는 임의로 생성
theta = np.random.randn(2, 1)

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

print(theta)
```

    [[4.21509616]
     [2.77011339]]


### 확률적 경사 하강법 - 샘플을 추출해서 gradient 를 구함 - 매 스텝에서 모든 데이터를 가지고 그라디언트를 구하지 않고 1개의 샘플만 적용해서 최적의 비용함수를 찾아가는 방식

- 모든 데이터를 가지고 그라디언트를 구하지 않기 때문에 알고리즘 학습 속도는 굉장히 빠름

- 배치 경사 하강법에 비해 매우 불안정 - 최적의 알고리즘을 못찾을 수 있습

- 비용 함수가 최솟값에 도달 할 때 까지 부드럽게 감소하지 않고 위 아래로 요동치면서 찾아갑니다.

- 샘플을 선택할 때 복원 추출을 못하도록 해야 합니다.
=> 하나의 스텝에서 선택한 데이터가 다음에 다시 선택하면 과대적합이 발생

- Sklearn 에서는 SGDRegressor 클래스가 확률적 경사 하강법을 사용할 수 있도록 지원합니다.


```python
from sklearn.linear_model import SGDRegressor

#확률적 경사 하강법을 이용하기 위한 객체를 생성
#max_iter 는 최대 반복 횟수
#tol는 최소 오차 범위: 0.001 - 오차가 이 값보다 작으면 스톱
#penalty 는 규제, eta 는 학습률
sgd_reg = SGDRegressor(max_iter = 1000, tol=1e-3, 
                       penalty=None, eta0=0.1, random_state=42)

sgd_reg.fit(X, y.ravel())

print("절편:", sgd_reg.intercept_)
print("기울기:", sgd_reg.coef_)
```

    절편: [4.24365286]
    기울기: [2.8250878]


## 보스톤 주택 데이터 셋에 선형회귀 적용
### 집값 예측

### 데이터 가져오기 


```python
from sklearn.datasets import load_boston

#seaborn 패키지에서는 데이터를 가져오면 데이터프레임으로 리턴
#sklearn 패키지에서는 데이터를 가져오면 dict 
#data 가 실제 데이터이고 feature_names 가 피처의 이름입니다.
#target 이 레이블입니다.
boston = load_boston()

bostonDF = pd.DataFrame(boston.data, columns=boston.feature_names)
#레이블 추가
bostonDF['PRICE'] = boston.target
print(bostonDF.head())
```

          CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD    TAX  \
    0  0.00632  18.0   2.31   0.0  0.538  6.575  65.2  4.0900  1.0  296.0   
    1  0.02731   0.0   7.07   0.0  0.469  6.421  78.9  4.9671  2.0  242.0   
    2  0.02729   0.0   7.07   0.0  0.469  7.185  61.1  4.9671  2.0  242.0   
    3  0.03237   0.0   2.18   0.0  0.458  6.998  45.8  6.0622  3.0  222.0   
    4  0.06905   0.0   2.18   0.0  0.458  7.147  54.2  6.0622  3.0  222.0   
    
       PTRATIO       B  LSTAT  PRICE  
    0     15.3  396.90   4.98   24.0  
    1     17.8  396.90   9.14   21.6  
    2     17.8  392.83   4.03   34.7  
    3     18.7  394.63   2.94   33.4  
    4     18.7  396.90   5.33   36.2  



```python
### 속성 간의 상관 관계를 파악
sns.pairplot(bostonDF, height=2.5)
plt.tight_layout()
plt.show()
```


    
![png](output_19_0.png)
    



```python
#상관관계를 몇 개의 피처만 추출해서 확인
cols = ['RM', 'ZN', 'INDUS', 'NOX', 'AGE', 'PTRATIO', 'RAD', 'PRICE']
sns.pairplot(bostonDF[cols], height=2.5)
plt.tight_layout()
plt.show()
```


    
![png](output_20_0.png)
    



```python
#PRICE 와 분포만 확인
lm_features = ['RM', 'ZN', 'INDUS', 'NOX', 'AGE', 'PTRATIO', 'LSTAT', 'RAD']

#그래프를 8개 만들기 위한 영역을 생성
fig, ax = plt.subplots(figsize=(16,8), ncols=4, nrows=2)

#반복문을 이용해서 8개의 항목과 PRICE 의 산포도를 출력
for i, feature in enumerate(lm_features):
    row = int(i/4)
    col = i%4
    sns.regplot(x = feature, y='PRICE', data=bostonDF, ax=ax[row][col])
```


    
![png](output_21_0.png)
    



```python
#상관 계수를 heatmap 으로 시각화
cols = ['RM', 'ZN', 'INDUS', 'NOX', 'AGE', 'PTRATIO', 'LSTAT', 'RAD', 'PRICE']
cm = np.corrcoef(bostonDF[cols].values.T)
#print(cm)

plt.figure(figsize=(16,8))
sns.set(font_scale=1.5)
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
           annot_kws={'size':15}, yticklabels=cols, xticklabels=cols)
plt.show()

#RM(방의 개수) 과 PRICE 는 강한 양의 상관 관계
#LSTAT(하위 계층의 비율) 과 PRICE 는 강한 음의 상관관계
```


    
![png](output_22_0.png)
    


### scipy 의 stats 패키지의 linregress 를 이용해서 RM 과 PRICE 의 선형 회귀 모델 훈련

- 기초 통계, 회귀 분석, 시계열 분석 등에 이용할 수 있습니다.

- 기초 통계 - 검정 기능, 커널 밀도 추정, moment 값

- 회귀 분석 - 선형 회귀, 일반화 선형 회귀, 강인 선형 회귀, 선형 혼합 효과 모형, ANOVA, Discrete Dependent Variable(로지스틱 회귀), 시계열 분석 등


```python
from scipy import stats
#lingress 는 기본적으로 단변량 선형 회귀에 사용
#데이터도 일차원 배열 - Series 로 대입
slope, intercept, r_value, p_value, stderr = stats.linregress(bostonDF['RM'], bostonDF['PRICE'])
    
print('기울기:', slope)
print('절편:', intercept)
print('상관계수:', r_value)
#유의확률이 0.05 보다 낮으므로 의미있는 결과로 간주
print('유의확률', p_value)
print('에러값:', stderr)

#방이 4개 일 때의 주택 가격은?
print(slope * 4 + intercept)
```

    기울기: 9.102108981180306
    절편: -34.67062077643854
    상관계수: 0.695359947071539
    유의확률 2.487228871008377e-74
    에러값: 0.41902656012134054
    1.7378151482826851



```python
#기존 데이터 와의 차이를 시각화
from scipy import polyval

#예측 값 구하기
ry = polyval([slope, intercept], bostonDF['RM'])

from pylab import plot, title, show, legend

#실제 값을 가지고 그래프를 그림
plot(bostonDF['RM'], bostonDF['PRICE'], 'k.')
#예측 값을 가지고 그래프를 그림
plot(bostonDF['RM'], ry, 'r')

title('주택 가격 과 방 개수 와의 관계')
legend(['실제 데이터', '예측 데이터'])
plt.xlabel('방 개수[RM]')
plt.ylabel('주택 가격[PRICE]')
show()
```


    
![png](output_25_0.png)
    


### sklearn 의 LinearRegression 을 이용한 선형 회귀 - 다변량 선형 회귀 가능 - 독립변수(설명변수, feature 등)가 2개 이상인 선형 회귀

- statsmodels.formula.api.ols(formula = '종속변수 ~ 독립변수 + 독립변수...', data=DataFrame(객체).fit()
=> 함수의 호출 결과

- params : y절편과 각 독립변수의 기울기를 저장하고 있는 Series

- pvalue : 유의확률

- predict() : 예측한 값을 리턴

- rsquared : 결정 계수


```python
from sklearn.linear_model import LinearRegression

#학습할 데이터 생성
#피쳐 데이터는 2차원 배열
X = bostonDF[['RM']].values
y = bostonDF['PRICE'].values

slr = LinearRegression()
slr.fit(X, y)

#기울기와 절편 확인
print('기울기:', slr.coef_[0])
print('절편:', slr.intercept_)
```

    기울기: 9.10210898118031
    절편: -34.67062077643857



```python
plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
plt.plot(X, slr.predict(X), color='black', lw=2)
plt.xlabel('방의 개수[RM]')
plt.ylabel('주택 가격[PRICE]')
plt.show()
```


    
![png](output_28_0.png)
    


### RANSAC 를 이용한 선형 회귀
#### 이상치를 제거하지 않고 옵션만 설정하면 그 옵션을 이용해서 데이터를 제거하면서
#### 선형 회귀를 수행하는 방식


```python
from sklearn.linear_model import RANSACRegressor

#수행할 회귀는 선형 회귀
#최대 반복 횟수는 100번
#최소 샘플의 개수는 50개
#잔차 계산 방법은 absolute_loss
#잔차의 허용치는 5.0
ransac = RANSACRegressor(LinearRegression(), 
                        max_trials=100,
                        min_samples=50,
                        loss='absolute_loss',
                        residual_threshold=5.0,
                        random_state = 42)

#훈련
ransac.fit(X, y)

#데이터가 오차 범위 내에 있는지 확인 - 오차 범위 내에 있는 데이터
inlier_mask = ransac.inlier_mask_
#배열의 데이터를 반대로 생성 - 오차 범위 바깥에 존재하는 데이터
outlier_mask = np.logical_not(inlier_mask)

#print(outlier_mask)

#시각화 작업
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])

#실제 데이터를 산점도로 표현
plt.scatter(X[inlier_mask], y[inlier_mask], c='steelblue', edgecolor='white',
           marker='o', label='Inlier')
plt.scatter(X[outlier_mask], y[outlier_mask], c='limegreen', edgecolor='white',
           marker='s', label='Outlier')
#예측한 데이터를 선 그래프로 표현
plt.plot(line_X, line_y_ransac, color='black', lw=2)
plt.xlabel('방 개수[RM]')
plt.ylabel('주택 가격[PRICE]')
plt.legend(loc = 'upper left')
plt.show()

#앞의 방식과 차이가 있음 - 몇 개의 데이터는 예측하는데에서 제외
print('기울기:', ransac.estimator_.coef_[0])
print('절편:', ransac.estimator_.intercept_)
```


    
![png](output_30_0.png)
    


    기울기: 9.52017496724595
    절편: -36.492768109126345


### 훈련 데이터 와 검증 데이터를 나누어서 선형 회귀
훈련 데이터는 모델의 학습에 사용하고 검증 데이터를 이용해서 확인


```python
#데이터 분할을 위한 함수
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_test.shape)

#모델 생성 - train 데이터 이용
slr = LinearRegression()
slr.fit(X_train, y_train)

#예측
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

#평가
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


#오차 출력
print('훈련 데이터의 MSE:', mean_squared_error(y_train, y_train_pred))
print('검증 데이터의 MSE:', mean_squared_error(y_test, y_test_pred))

#결정 계수 출력
print('훈련 데이터의 R2:', r2_score(y_train, y_train_pred))
print('검증 데이터의 R2:', r2_score(y_test, y_test_pred))

```

    (404, 1)
    (102, 1)
    훈련 데이터의 MSE: 42.82047894172751
    검증 데이터의 MSE: 46.907351627395315
    훈련 데이터의 R2: 0.4970800097843844
    검증 데이터의 R2: 0.42394386816456275


## 다변량 회귀 분석
독립 변수의 개수가 여러 개인 회귀 분석

### 다변량 선형 회귀에서 발생할 수 있는 문제 - 다중 공선성 문제
상관관계가 높은 독립 변수들이 포함되어 있어서 이 변수들의 영향을 많이 받게 되는 문제입니다.

- 부정확한한 회귀 결과가 도출되는 경우가 많음

- 종속 변수와 독립 변수간에는 상관관계가 높아야 하지만 독립 변수끼리는 독립적이어야 합니다.

- 다중 공선성을 파악하는 방법 : 시각화를 이용(산점도나 heatmap(상관계수를 출력) 등을 이용, VIF(Fariance Infaltion Factors - 분산 팽창 요인) 을 이용 : (1/(1 - 결정계수 - R2)) 로 계산


```python
import statsmodels.formula.api as sm

#데이터 읽어오기
df = pd.read_csv('./data/score.csv')
print(df.head())
df.info()
```

      name  score   iq  academy  game  tv
    0    A     90  140        2     1   0
    1    B     75  125        1     3   3
    2    C     77  120        1     0   4
    3    D     83  135        2     3   2
    4    E     65  105        0     4   4
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10 entries, 0 to 9
    Data columns (total 6 columns):
     #   Column   Non-Null Count  Dtype 
    ---  ------   --------------  ----- 
     0   name     10 non-null     object
     1   score    10 non-null     int64 
     2   iq       10 non-null     int64 
     3   academy  10 non-null     int64 
     4   game     10 non-null     int64 
     5   tv       10 non-null     int64 
    dtypes: int64(5), object(1)
    memory usage: 608.0+ bytes



```python
#iq  academy  game  tv 개를 독립변수로 하고 score를 종속변수로 하는 선형 회귀
result = sm.ols('score ~ iq + academy + game + tv', data=df).fit()
print('절편 과 기울기:', result.params)
print('유의확률:', result.pvalues)
print('결정계수:', result.rsquared)
print('각 학생의 예측 점수:', result.predict())
```

    절편 과 기울기: Intercept    23.299232
    iq            0.468422
    academy       0.717901
    game         -0.838955
    tv           -1.385408
    dtype: float64
    유의확률: Intercept    0.117508
    iq           0.003376
    academy      0.534402
    game         0.131001
    tv           0.184269
    dtype: float64
    결정계수: 0.9608351062148871
    각 학생의 예측 점수: [89.47516936 75.89680196 74.68614801 82.68433242 63.58609587 80.84448722
     82.54342107 72.89049111 86.12886227 80.2641907 ]



```python
#IQ 가 130 이고 학원을 3개 다니고 게임을 2시간하고 TV를 1시간 보는 학생의 예측 점수는
y = result.params.Intercept + 130 * result.params.iq + 3 * result.params.academy + 2 * result.params.game + 1 * result.params.tv
print(y)
```

    83.28448678034155



```python
#그래프를 그려셔 정확도 확인
plt.figure()
plt.plot(df['score'], label='실제 성적')
plt.plot(result.predict(), label='예측 성적')
plt.xticks(range(0,10,1), df['name'])
plt.legend()
plt.show()
```


    
![png](output_37_0.png)
    


## 투수들의 연봉 예측
데이터는 http://www.statiz.co.kr 에 있습니다.


```python
#데이터 가져오기
pitcher = pd.read_csv('./data/picher_stats_2017.csv')
print(pitcher.head())
pitcher.info()
```

       선수명   팀명   승   패  세  홀드  블론  경기  선발     이닝  ...  홈런/9  BABIP  LOB%   ERA  \
    0   켈리   SK  16   7  0   0   0  30  30  190.0  ...  0.76  0.342  73.7  3.60   
    1   소사   LG  11  11  1   0   0  30  29  185.1  ...  0.53  0.319  67.1  3.88   
    2  양현종  KIA  20   6  0   0   0  31  31  193.1  ...  0.79  0.332  72.1  3.44   
    3  차우찬   LG  10   7  0   0   0  28  28  175.2  ...  1.02  0.298  75.0  3.43   
    4  레일리   롯데  13   7  0   0   0  30  30  187.1  ...  0.91  0.323  74.1  3.80   
    
       RA9-WAR   FIP  kFIP   WAR  연봉(2018)  연봉(2017)  
    0     6.91  3.69  3.44  6.62    140000     85000  
    1     6.80  3.52  3.41  6.08    120000     50000  
    2     6.54  3.94  3.82  5.64    230000    150000  
    3     6.11  4.20  4.03  4.63    100000    100000  
    4     6.13  4.36  4.31  4.38    111000     85000  
    
    [5 rows x 22 columns]
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 152 entries, 0 to 151
    Data columns (total 22 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   선수명       152 non-null    object 
     1   팀명        152 non-null    object 
     2   승         152 non-null    int64  
     3   패         152 non-null    int64  
     4   세         152 non-null    int64  
     5   홀드        152 non-null    int64  
     6   블론        152 non-null    int64  
     7   경기        152 non-null    int64  
     8   선발        152 non-null    int64  
     9   이닝        152 non-null    float64
     10  삼진/9      152 non-null    float64
     11  볼넷/9      152 non-null    float64
     12  홈런/9      152 non-null    float64
     13  BABIP     152 non-null    float64
     14  LOB%      152 non-null    float64
     15  ERA       152 non-null    float64
     16  RA9-WAR   152 non-null    float64
     17  FIP       152 non-null    float64
     18  kFIP      152 non-null    float64
     19  WAR       152 non-null    float64
     20  연봉(2018)  152 non-null    int64  
     21  연봉(2017)  152 non-null    int64  
    dtypes: float64(11), int64(9), object(2)
    memory usage: 26.2+ KB



```python
pitcher['연봉(2018)'].hist(bins=100)
```




    <AxesSubplot:>




    
![png](output_40_1.png)
    



```python
pitcher.boxplot(column=['연봉(2018)'])
```




    <AxesSubplot:>




    
![png](output_41_1.png)
    



```python
#필요한 데이터만 추출
pitcher_feature_df = pitcher[['승','패','세','홀드','블론','경기','선발',
                              '이닝', '삼진/9', '볼넷/9','홈런/9','BABIP',
                              'LOB%','ERA','RA9-WAR', 'FIP', 'kFIP', 'WAR',
                              '연봉(2018)','연봉(2017)']]
print(pitcher_feature_df.head())
```

        승   패  세  홀드  블론  경기  선발     이닝  삼진/9  볼넷/9  홈런/9  BABIP  LOB%   ERA  \
    0  16   7  0   0   0  30  30  190.0  8.95  2.13  0.76  0.342  73.7  3.60   
    1  11  11  1   0   0  30  29  185.1  7.43  1.85  0.53  0.319  67.1  3.88   
    2  20   6  0   0   0  31  31  193.1  7.36  2.09  0.79  0.332  72.1  3.44   
    3  10   7  0   0   0  28  28  175.2  8.04  1.95  1.02  0.298  75.0  3.43   
    4  13   7  0   0   0  30  30  187.1  7.49  2.11  0.91  0.323  74.1  3.80   
    
       RA9-WAR   FIP  kFIP   WAR  연봉(2018)  연봉(2017)  
    0     6.91  3.69  3.44  6.62    140000     85000  
    1     6.80  3.52  3.41  6.08    120000     50000  
    2     6.54  3.94  3.82  5.64    230000    150000  
    3     6.11  4.20  4.03  4.63    100000    100000  
    4     6.13  4.36  4.31  4.38    111000     85000  



```python
#스케일링을 해주는 함수
def standard_scaling(df, scale_columns):
    for col in scale_columns:
        series_mean = df[col].mean()
        series_std = df[col].std()
        df[col] = df[col].apply(lambda x:(x-series_mean)/series_std)
    return df

scale_columns = ['승','패','세','홀드','블론','경기','선발',
                    '이닝', '삼진/9', '볼넷/9','홈런/9','BABIP',
                    'LOB%','ERA','RA9-WAR', 'FIP', 'kFIP', 'WAR','연봉(2017)']


pitcher_df = standard_scaling(pitcher, scale_columns)
#print(pitcher_df.head())

#컬럼 이름 변경
pitcher_df = pitcher_df.rename(columns={'연봉(2018)':'y'})
print(pitcher_df.head())

```

       선수명   팀명         승         패         세        홀드        블론        경기  \
    0   켈리   SK  3.313623  1.227145 -0.306452 -0.585705 -0.543592  0.059433   
    1   소사   LG  2.019505  2.504721 -0.098502 -0.585705 -0.543592  0.059433   
    2  양현종  KIA  4.348918  0.907751 -0.306452 -0.585705 -0.543592  0.111056   
    3  차우찬   LG  1.760682  1.227145 -0.306452 -0.585705 -0.543592 -0.043811   
    4  레일리   롯데  2.537153  1.227145 -0.306452 -0.585705 -0.543592  0.059433   
    
             선발        이닝  ...      홈런/9     BABIP      LOB%       ERA   RA9-WAR  \
    0  2.452068  2.645175  ... -0.442382  0.016783  0.446615 -0.587056  3.174630   
    1  2.349505  2.547755  ... -0.668521 -0.241686 -0.122764 -0.519855  3.114968   
    2  2.554632  2.706808  ... -0.412886 -0.095595  0.308584 -0.625456  2.973948   
    3  2.246942  2.350927  ... -0.186746 -0.477680  0.558765 -0.627856  2.740722   
    4  2.452068  2.587518  ... -0.294900 -0.196735  0.481122 -0.539055  2.751570   
    
            FIP      kFIP       WAR       y  연봉(2017)  
    0 -0.971030 -1.058125  4.503142  140000  2.734705  
    1 -1.061888 -1.073265  4.094734  120000  1.337303  
    2 -0.837415 -0.866361  3.761956  230000  5.329881  
    3 -0.698455 -0.760385  2.998081  100000  3.333592  
    4 -0.612941 -0.619085  2.809003  111000  2.734705  
    
    [5 rows x 22 columns]



```python
#팀명을 숫자로 변경 - 문자열은 숫자로 변경하고자 할 때 one-hot encoding 사용
team_encoding = pd.get_dummies(pitcher_df['팀명'])
#print(team_encoding)

#팀명을 삭제
pitcher_df = pitcher_df.drop('팀명', axis=1)

pitcher_df = pitcher_df.join(team_encoding)
print(pitcher_df.head())
```

       선수명         승         패         세        홀드        블론        경기        선발  \
    0   켈리  3.313623  1.227145 -0.306452 -0.585705 -0.543592  0.059433  2.452068   
    1   소사  2.019505  2.504721 -0.098502 -0.585705 -0.543592  0.059433  2.349505   
    2  양현종  4.348918  0.907751 -0.306452 -0.585705 -0.543592  0.111056  2.554632   
    3  차우찬  1.760682  1.227145 -0.306452 -0.585705 -0.543592 -0.043811  2.246942   
    4  레일리  2.537153  1.227145 -0.306452 -0.585705 -0.543592  0.059433  2.452068   
    
             이닝      삼진/9  ...  연봉(2017)  KIA  KT  LG  NC  SK  두산  롯데  삼성  한화  
    0  2.645175  0.672099  ...  2.734705    0   0   0   0   1   0   0   0   0  
    1  2.547755  0.134531  ...  1.337303    0   0   1   0   0   0   0   0   0  
    2  2.706808  0.109775  ...  5.329881    1   0   0   0   0   0   0   0   0  
    3  2.350927  0.350266  ...  3.333592    0   0   1   0   0   0   0   0   0  
    4  2.587518  0.155751  ...  2.734705    0   0   0   0   0   0   1   0   0  
    
    [5 rows x 30 columns]



```python
#피처 와 레이블을 분리
X = pitcher_df[pitcher_df.columns.difference(['선수명', 'y'])]
y = pitcher_df['y']

#훈련 데이터 와 검증 데이터를 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                test_size=0.2, random_state=42)

#선형 회귀 모델을 생성
lr = LinearRegression()
model = lr.fit(X_train, y_train)

#컬럼 이름을 같이 출력
print(pitcher_df.columns)
#회귀 계수 출력
print(lr.coef_)
```

    Index(['선수명', '승', '패', '세', '홀드', '블론', '경기', '선발', '이닝', '삼진/9', '볼넷/9',
           '홈런/9', 'BABIP', 'LOB%', 'ERA', 'RA9-WAR', 'FIP', 'kFIP', 'WAR', 'y',
           '연봉(2017)', 'KIA', 'KT', 'LG', 'NC', 'SK', '두산', '롯데', '삼성', '한화'],
          dtype='object')
    [ -1863.27167152   1147.15608757 -52147.32574794   5915.51391759
       2299.44885884  -1744.6150334     397.17996335   -249.60365919
      -1024.27838506    399.1396206   12274.79760529  44088.31585257
      -3602.91866901  -5319.02202278    617.34035282   4644.19380296
        879.30541662  -3936.74747195   1521.68382584 -10999.04385918
       -700.8303505    4526.7078132   21785.5776696    6965.59101874
        154.91380911   2018.54543747  -1217.59759673   9090.86143072]



```python
#훈련 데이터 와 테스트 데이터로 예측
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

#평가 지표 출력
print('훈련 데이터 MSE:', mean_squared_error(y_train, y_train_pred))
print('검증 데이터 MSE:', mean_squared_error(y_test, y_test_pred))
print()

print('훈련 데이터 RMSE:', np.sqrt(mean_squared_error(y_train, y_train_pred)))
print('검증 데이터 RMSE:', np.sqrt(mean_squared_error(y_test, y_test_pred)))
print()

print('훈련 데이터 R2:', r2_score(y_train, y_train_pred))
print('검증 데이터 R2:', r2_score(y_test, y_test_pred))


#훈련 데이터 와 검증 데이터에서의 차이가 많이 나면 과대 적합
#피처가 너무 많거나 샘플의 데이터가 너무 많은 경우 발생할 수 있음
#피처의 개수를 줄여줄 필요가 있음 
#다른 피처 와의 상관 관계가 높은 피처들을 일부분 제거
#직접 상관계수를 확인해도 되지만 VIF(분산 팽창 요인)으로도 확인히 가능
```

    훈련 데이터 MSE: 74824945.3662621
    검증 데이터 MSE: 81760179.59222916
    
    훈련 데이터 RMSE: 8650.141349496093
    검증 데이터 RMSE: 9042.133575225991
    
    훈련 데이터 R2: 0.9322281224002374
    검증 데이터 R2: 0.755303925817233



```python
#현재 가지고 있는 데이터 프레임 확인
pitcher_df.info()

#정규화 한 데이터 컬럼 이름 확인
print(scale_columns)
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 152 entries, 0 to 151
    Data columns (total 30 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   선수명       152 non-null    object 
     1   승         152 non-null    float64
     2   패         152 non-null    float64
     3   세         152 non-null    float64
     4   홀드        152 non-null    float64
     5   블론        152 non-null    float64
     6   경기        152 non-null    float64
     7   선발        152 non-null    float64
     8   이닝        152 non-null    float64
     9   삼진/9      152 non-null    float64
     10  볼넷/9      152 non-null    float64
     11  홈런/9      152 non-null    float64
     12  BABIP     152 non-null    float64
     13  LOB%      152 non-null    float64
     14  ERA       152 non-null    float64
     15  RA9-WAR   152 non-null    float64
     16  FIP       152 non-null    float64
     17  kFIP      152 non-null    float64
     18  WAR       152 non-null    float64
     19  y         152 non-null    int64  
     20  연봉(2017)  152 non-null    float64
     21  KIA       152 non-null    uint8  
     22  KT        152 non-null    uint8  
     23  LG        152 non-null    uint8  
     24  NC        152 non-null    uint8  
     25  SK        152 non-null    uint8  
     26  두산        152 non-null    uint8  
     27  롯데        152 non-null    uint8  
     28  삼성        152 non-null    uint8  
     29  한화        152 non-null    uint8  
    dtypes: float64(19), int64(1), object(1), uint8(9)
    memory usage: 26.4+ KB
    ['승', '패', '세', '홀드', '블론', '경기', '선발', '이닝', '삼진/9', '볼넷/9', '홈런/9', 'BABIP', 'LOB%', 'ERA', 'RA9-WAR', 'FIP', 'kFIP', 'WAR', '연봉(2017)']


### 상관 계수 확인


```python
#피어슨 상관 계수 구하기 - 원핫인코딩한 컬럼(상관계수나 정규화에서 제외)은 제외
corr = pitcher_df[scale_columns].corr(method='pearson')
print(corr)
```

                     승         패         세        홀드        블론        경기  \
    승         1.000000  0.710749  0.053747  0.092872  0.105281  0.397074   
    패         0.710749  1.000000  0.066256  0.098617  0.121283  0.343147   
    세         0.053747  0.066256  1.000000  0.112716  0.605229  0.434290   
    홀드        0.092872  0.098617  0.112716  1.000000  0.490076  0.715527   
    블론        0.105281  0.121283  0.605229  0.490076  1.000000  0.630526   
    경기        0.397074  0.343147  0.434290  0.715527  0.630526  1.000000   
    선발        0.773560  0.771395 -0.177069 -0.285204 -0.264160 -0.037443   
    이닝        0.906093  0.829018  0.020278  0.024631  0.014176  0.376378   
    삼진/9      0.078377  0.031755  0.170436  0.186790  0.188423  0.192487   
    볼넷/9     -0.404710 -0.386313 -0.131394 -0.146806 -0.137019 -0.364293   
    홈런/9     -0.116147 -0.064467 -0.073111 -0.076475 -0.064804 -0.113545   
    BABIP    -0.171111 -0.133354 -0.089212 -0.104307 -0.112480 -0.241608   
    LOB%      0.131178 -0.020994  0.167557  0.048123  0.100633  0.105762   
    ERA      -0.271086 -0.188036 -0.150348 -0.155712 -0.160761 -0.320177   
    RA9-WAR   0.851350  0.595989  0.167669  0.003526  0.008766  0.281595   
    FIP      -0.303133 -0.233416 -0.199746 -0.211515 -0.209014 -0.345351   
    kFIP     -0.314159 -0.238688 -0.225259 -0.237353 -0.237815 -0.373777   
    WAR       0.821420  0.625641  0.084151 -0.038613 -0.058213  0.197836   
    연봉(2017)  0.629710  0.429227  0.262664 -0.001213  0.146584  0.225357   
    
                    선발        이닝      삼진/9      볼넷/9      홈런/9     BABIP  \
    승         0.773560  0.906093  0.078377 -0.404710 -0.116147 -0.171111   
    패         0.771395  0.829018  0.031755 -0.386313 -0.064467 -0.133354   
    세        -0.177069  0.020278  0.170436 -0.131394 -0.073111 -0.089212   
    홀드       -0.285204  0.024631  0.186790 -0.146806 -0.076475 -0.104307   
    블론       -0.264160  0.014176  0.188423 -0.137019 -0.064804 -0.112480   
    경기       -0.037443  0.376378  0.192487 -0.364293 -0.113545 -0.241608   
    선발        1.000000  0.894018 -0.055364 -0.312935 -0.058120 -0.098909   
    이닝        0.894018  1.000000  0.037343 -0.451101 -0.107063 -0.191514   
    삼진/9     -0.055364  0.037343  1.000000  0.109345  0.216017  0.457523   
    볼넷/9     -0.312935 -0.451101  0.109345  1.000000  0.302251  0.276009   
    홈런/9     -0.058120 -0.107063  0.216017  0.302251  1.000000  0.362614   
    BABIP    -0.098909 -0.191514  0.457523  0.276009  0.362614  1.000000   
    LOB%      0.041819  0.103369 -0.071284 -0.150837 -0.274543 -0.505478   
    ERA      -0.157775 -0.285392  0.256840  0.521039  0.629912  0.733109   
    RA9-WAR   0.742258  0.853354  0.102963 -0.398586 -0.187210 -0.187058   
    FIP      -0.151040 -0.296768 -0.154857  0.629833  0.831042  0.251126   
    kFIP     -0.142685 -0.302288 -0.317594  0.605008  0.743623  0.166910   
    WAR       0.758846  0.832609  0.151791 -0.394131 -0.205014 -0.082995   
    연봉(2017)  0.488559  0.586874  0.104948 -0.332379 -0.100896 -0.088754   
    
                  LOB%       ERA   RA9-WAR       FIP      kFIP       WAR  연봉(2017)  
    승         0.131178 -0.271086  0.851350 -0.303133 -0.314159  0.821420  0.629710  
    패        -0.020994 -0.188036  0.595989 -0.233416 -0.238688  0.625641  0.429227  
    세         0.167557 -0.150348  0.167669 -0.199746 -0.225259  0.084151  0.262664  
    홀드        0.048123 -0.155712  0.003526 -0.211515 -0.237353 -0.038613 -0.001213  
    블론        0.100633 -0.160761  0.008766 -0.209014 -0.237815 -0.058213  0.146584  
    경기        0.105762 -0.320177  0.281595 -0.345351 -0.373777  0.197836  0.225357  
    선발        0.041819 -0.157775  0.742258 -0.151040 -0.142685  0.758846  0.488559  
    이닝        0.103369 -0.285392  0.853354 -0.296768 -0.302288  0.832609  0.586874  
    삼진/9     -0.071284  0.256840  0.102963 -0.154857 -0.317594  0.151791  0.104948  
    볼넷/9     -0.150837  0.521039 -0.398586  0.629833  0.605008 -0.394131 -0.332379  
    홈런/9     -0.274543  0.629912 -0.187210  0.831042  0.743623 -0.205014 -0.100896  
    BABIP    -0.505478  0.733109 -0.187058  0.251126  0.166910 -0.082995 -0.088754  
    LOB%      1.000000 -0.720091  0.286893 -0.288050 -0.269536  0.144191  0.110424  
    ERA      -0.720091  1.000000 -0.335584  0.648004  0.582057 -0.261508 -0.203305  
    RA9-WAR   0.286893 -0.335584  1.000000 -0.366308 -0.377679  0.917299  0.643375  
    FIP      -0.288050  0.648004 -0.366308  1.000000  0.984924 -0.391414 -0.268005  
    kFIP     -0.269536  0.582057 -0.377679  0.984924  1.000000 -0.408283 -0.282666  
    WAR       0.144191 -0.261508  0.917299 -0.391414 -0.408283  1.000000  0.675794  
    연봉(2017)  0.110424 -0.203305  0.643375 -0.268005 -0.282666  0.675794  1.000000  



```python
#상관 계수를 heatmap 으로 표현
plt.figure(figsize=(30,8))
show_cols = ['win', 'lose', 'save', 'hold', 'blon', 'match', 'start', 
            'inning', 'strike3', 'ball4', 'homerun', 'BABIP', 'LOB', 
            'ERA', 'RA9-WAR', 'FIP', 'kFIP', 'WAR', '2017']
sns.set(font_scale=1.5)
sns.heatmap(corr.values, cbar=True, annot=True, square=True, fmt='.2f',
           annot_kws={'size':10}, yticklabels=show_cols, 
            xticklabels=show_cols)
plt.show()
```


    
![png](output_50_0.png)
    


### VIF 확인


```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
#데이터 프레임 생성
vif = pd.DataFrame()
#X 가 가지고 있는 데이터들을 이용해서 분산 팽창 요인을 계산
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) 
                     for i in range(X.shape[1])]
#컬럼 이름을 복사해서 생성
vif["features"] = X.columns
#소수 첫번째 짜리까지 반올림
vif.round(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIF Factor</th>
      <th>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.2</td>
      <td>BABIP</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.6</td>
      <td>ERA</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14238.3</td>
      <td>FIP</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.1</td>
      <td>KIA</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.1</td>
      <td>KT</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.1</td>
      <td>LG</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4.3</td>
      <td>LOB%</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.1</td>
      <td>NC</td>
    </tr>
    <tr>
      <th>8</th>
      <td>13.6</td>
      <td>RA9-WAR</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.1</td>
      <td>SK</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10.4</td>
      <td>WAR</td>
    </tr>
    <tr>
      <th>11</th>
      <td>10264.1</td>
      <td>kFIP</td>
    </tr>
    <tr>
      <th>12</th>
      <td>14.6</td>
      <td>경기</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1.2</td>
      <td>두산</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.1</td>
      <td>롯데</td>
    </tr>
    <tr>
      <th>15</th>
      <td>57.8</td>
      <td>볼넷/9</td>
    </tr>
    <tr>
      <th>16</th>
      <td>3.0</td>
      <td>블론</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1.2</td>
      <td>삼성</td>
    </tr>
    <tr>
      <th>18</th>
      <td>89.5</td>
      <td>삼진/9</td>
    </tr>
    <tr>
      <th>19</th>
      <td>39.6</td>
      <td>선발</td>
    </tr>
    <tr>
      <th>20</th>
      <td>3.1</td>
      <td>세</td>
    </tr>
    <tr>
      <th>21</th>
      <td>8.0</td>
      <td>승</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2.5</td>
      <td>연봉(2017)</td>
    </tr>
    <tr>
      <th>23</th>
      <td>63.8</td>
      <td>이닝</td>
    </tr>
    <tr>
      <th>24</th>
      <td>5.9</td>
      <td>패</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1.1</td>
      <td>한화</td>
    </tr>
    <tr>
      <th>26</th>
      <td>3.8</td>
      <td>홀드</td>
    </tr>
    <tr>
      <th>27</th>
      <td>425.6</td>
      <td>홈런/9</td>
    </tr>
  </tbody>
</table>
</div>




```python
#피처의 개수를 줄여서 다중 공선성 문제를 해결
X = pitcher_df[['FIP', 'WAR', '볼넷/9', '삼진/9', '연봉(2017)']]
#데이터 프레임 생성
vif = pd.DataFrame()
#X 가 가지고 있는 데이터들을 이용해서 분산 팽창 요인을 계산
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) 
                     for i in range(X.shape[1])]
#컬럼 이름을 복사해서 생성
vif["features"] = X.columns
#소수 첫번째 짜리까지 반올림
vif.round(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIF Factor</th>
      <th>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.9</td>
      <td>FIP</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.1</td>
      <td>WAR</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.9</td>
      <td>볼넷/9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.1</td>
      <td>삼진/9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.9</td>
      <td>연봉(2017)</td>
    </tr>
  </tbody>
</table>
</div>




```python
#피처를 재선택해서 다시 학습하고 평가 지표를 출력
y = pitcher_df['y']

#train 데이터 와 test 데이터를 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#선형 회귀 모델(OLS) 학습
lr = LinearRegression()
model = lr.fit(X_train, y_train)

#평가
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

#평가 지표 출력
print('훈련 데이터 MSE:', mean_squared_error(y_train, y_train_pred))
print('검증 데이터 MSE:', mean_squared_error(y_test, y_test_pred))
print()

print('훈련 데이터 RMSE:', np.sqrt(mean_squared_error(y_train, y_train_pred)))
print('검증 데이터 RMSE:', np.sqrt(mean_squared_error(y_test, y_test_pred)))
print()

print('훈련 데이터 R2:', r2_score(y_train, y_train_pred))
print('검증 데이터 R2:', r2_score(y_test, y_test_pred))
```

    훈련 데이터 MSE: 91567288.5717807
    검증 데이터 MSE: 40695452.09679336
    
    훈련 데이터 RMSE: 9569.079818445485
    검증 데이터 RMSE: 6379.298715124834
    
    훈련 데이터 R2: 0.9170639277736519
    검증 데이터 R2: 0.8782045561195827


## 다항 회귀 - 선형회귀인데 slope 가 여러 개의 형태로 나타내는 회귀식

- 단항식 feature 를 다항식 feature 로 변경해서 회귀를 수행

- [1,2] -> [1(상수), 1(첫항그대로), 2(두번째항그대로), 1(첫번째항의제곱), 2(첫번째와두번째의곱), 4(두번째항의제곱)]
=> 단순 선형이 아닌 경우는 이런식으로 데이터를 늘리면 상관성을 만들어 낼 수 있음

- PolynomiaFeatures 클래스를 통해서 제공하며 객체를 생성할 때 degree 옵션에 다항식 feature 로 전환할 숫자를 설정합니다.
=> 2를 대입하면 제곱까지 3을 대입하면 3제곱까지 구해줍니다.

### 피처 변환

### 편향/분산 트레이드 오프(Bias-Variance Trade Off) 

- 1)오차의 종류
- 편향 : 잘못된 가정으로 인한 오차 - 데이터가 실제로는 2차 다항식의 형태로 만들어져야 하는데 1차 다항식으로 가정하는 경우로 편향이 큰 모델은 과속적합될 가능성이 높음

- 분산 : 분산은 훈련 데이터에 있는 작은 벼동에 모델이 과도하게 민감하기 때문에 나타나는 것으로 자유도가 높은 모델(고차 다항 회귀 모델)이 높은 분산을 가지기 쉬우며 과대 적합될 가능성이 높음

- 잡음 : 줄일수 없는 오차로 측정할 떄 측정된 데이터의 오차 - 데이터 소스 자체를 수정하거나 이상치 감지 기법을 해결

- 2)모델의 복잡도가 커지면 분산이 늘어나고 편향은 줄어들지만 반대로 모델의 복잡도가 줄어들면 편향이 커지고 분산이 작아지는 것을 분산/편향 트레이드 오프라고 합니다.


```python
from sklearn.preprocessing import PolynomialFeatures

X = np.arange(4).reshape(2,2)
print('일차 항:', X)

poly = PolynomialFeatures(degree=2)
poly.fit(X)
poly_ftr = poly.transform(X)
print('2차 항:', poly_ftr)
```

    일차 항: [[0 1]
     [2 3]]
    2차 항: [[1. 0. 1. 0. 0. 1.]
     [1. 2. 3. 4. 6. 9.]]



```python
#비선형 데이터 생성
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)
```


```python
#비선형 데이터 시각화
plt.plot(X, y, "b.")
plt.axis([-3, 3, 0, 10])
plt.show()
```


    
![png](output_59_0.png)
    



```python
#훈련 데이터를 다항 데이터 로 변환
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
print(X[0])
print(X_poly[0])
```

    [1.01354436]
    [1.01354436 1.02727217]



```python
lin_reg = LinearRegression()
#단항 데이터는 기울기가 1개
lin_reg.fit(X, y)
print("절편:", lin_reg.intercept_)
print("기울기:", lin_reg.coef_)

#다항 변환된 데이터는 기울기가 1개가 아님
lin_reg.fit(X_poly, y)
print("절편:", lin_reg.intercept_)
print("기울기:", lin_reg.coef_)
```

    절편: [3.58220213]
    기울기: [[1.02719252]]
    절편: [2.03155672]
    기울기: [[1.06186926 0.49982575]]



```python
#다항 회귀 식의 시각화
X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)

#예측
y_new = lin_reg.predict(X_new_poly)


#그래프 그리기
plt.plot(X, y, "b.")
#다항회귀라서 직선이 아니고 곡선의 형태
plt.plot(X_new, y_new, "r-", linewidth=2, label="예측값")
plt.axis([-3, 3, 0, 10])
plt.show()
```


    
![png](output_62_0.png)
    


## 규제가 있는 선형 회귀

 1.Loss

- 1)L1 Loss - 실제 값과 예측치 차이의 절대값 합
- 2)L2 Loss - 실제값과 예측치 차이의 제곱 합
- 3)차이
- L2 Loss 가 Outlier(이상치)에 더 민감합니다.
- L1 Loss 가 L2 Loss 에 비해 Outlier 에 더 Robust 하다라고 합니다.
- Outlier 가 적당히 무시되기를 바라면 L1 Los 를 사용하고 그렇지 않다면 L2 Loss 를 사용합니다.
- L1 Loss 는 미분이 안됨

2.일반화(정규화 - regularzation) - 공선성을 다루거나 데이터에서 잡음을 제거하여 과대적합을 방지할 수 있는 방법
- 모델 복잡도에 대한 패널티로 정규하는 Overfitting 을 예방
- 선형 회귀 규제 방법 : Ridge, Lasso, Elasting Net

1)L1 정규화 - 각각의 회귀 변수에 1/n 이나 1/2 등의 가중치의 절대값을 더해주는 방식
- 학습률이라는 하이퍼 파라미터를 이용해서 설정
- 0.0 ~ 1.0 사이로 설정
- 0에 가까운 수를 대입하며 정규화의 효과는 사라집니다. 너무 높게 설정하면 가중치가 없어질 수 있습니다.
- L1 정규화를 이용하는 방법이 Lasso입니다.

2)L2 정규화 - 분수에 가중치의 제곱을 더해주는 방식
- L2 정규화를 이용하는 방법이 Ridge 입니다.

3.Ridge 선형 회귀 - 분산은 줄이고 편향을 늘리는 방식으로 동작(Ridge 클래스를 이용)
- alpha 라는 하이퍼 파라미터를 이용
- 0.0 ~ 1.0 사이로 설정(알파값이 1.0 에 가까워지면 선형에 가까워짐, 0.0 에 가까워지면 고차 다항식을 만들어 냅니다.)
- 피처를 제거하지는 않음
- 피처의 회귀 계수를 줄이는 방식으로 영향력을 최소화 하는 방식

4.Lasso 선형 회귀 - L1 규제를 회귀에 적용
- 무의미한 피처들에 대해 계수를 0을 만들어서 제거하는 효과를 이용하는 방식
- 학습률이라는 파라미터를 이용해서 설정(이름은 alpha)
- 학습률이 작으면 고차 다항식을 만들어 냅니다.
- alpha 값이 높아지면 회귀 계수가 0 이 되는 feature 들이 만들어 집니다.

5.Elastic Net - Ridge 와 Lasso 회귀를 절충하는 방식
- r 이라는 하이퍼 파라미터를 이용하는데 0 이면 Ridge 모델이 되고 1이면 Lasso 모델이 됩니다.

6.선형 회귀 모델의 선택
- 기본적으로 제공되는 선형 회귀의 사용은 권장하지 않으며 기본은 Ridge 를 사용하고 피처의 개수는 많은데 몇 개의 피처만 사용되는 경우에는 Lasso 나 Elastic Net 을 고민
- 피처의 개수가 샘플의 개수보다 많거나 피처 몇개가 강한 상관관계를 갖는다면 Lasso 보다는 Elastic Net 를 선호

### 데이터 생성


```python
boston = load_boston()
bostonDF = pd.DataFrame(boston.data, columns=boston.feature_names)

bostonDF['PRICE'] = boston.target

y_target = bostonDF['PRICE']
X_data = bostonDF.drop(['PRICE'], axis=1, inplace=False)
```

### Ridge 모델 - L2 정규화, 분산은 줄이고 편향을 늘리는 방식으로 동작


```python
from sklearn.linear_model import Ridge

#사용할 alpha 값의 리스트
alphas = [0, 0.1, 1, 10, 100]


for alpha in alphas:
    ridge = Ridge(alpha = alpha)
    neg_mse_scores = cross_val_score(ridge, X_data, y_target, 
                            scoring='neg_mean_squared_error', cv=5)
    avg_rmse = np.mean(np.sqrt(-1 * neg_mse_scores))
    print('alpha가 {0} 일 때 평균 RMSE:{1:.3f}'.format(alpha, avg_rmse))
```

    alpha가 0 일 때 평균 RMSE:5.829
    alpha가 0.1 일 때 평균 RMSE:5.788
    alpha가 1 일 때 평균 RMSE:5.653
    alpha가 10 일 때 평균 RMSE:5.518
    alpha가 100 일 때 평균 RMSE:5.330


## 정규화 와 규제를 적용한 선형 회귀 모델

### 데이터 가져오기


```python
from sklearn.datasets import load_boston
## 정규화 와 규제를 적용한 선형 회귀 모델
boston = load_boston()
bostonDF = pd.DataFrame(boston.data, columns=boston.feature_names)

bostonDF['PRICE'] = boston.target

y_target = bostonDF['PRICE']
X_data = bostonDF.drop(['PRICE'], axis=1, inplace=False)
```


```python
#평가 지표를 구해주는 함수
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

#훈련 데이터 와 시험 데이터를 분할해주는 함수
from sklearn.model_selection import train_test_split

#규제가 있는 선형 회귀 모델
from sklearn.linear_model import Ridge, Lasso, ElasticNet


#회귀 모델 과 데이터를 받아서 검증을 수행하고 RMSE 를 출력해주는 함수
#첫번째 회귀 모델 이름, 두번째는 파라미터, 세번째 와 네번째는 데이터
#다섯번째는 모델 이름 출력 여부, 여섯번째는 회귀모델에서 intercept 출력 여부
def get_linear_reg_eval(model_name, params=None, X_data_n=None, y_target_n=None,
                       verbose=True, return_coeff = True):
    #결과를 만들기 위한 데이터프레임
    coeff_df = pd.DataFrame()
    #모델 이름 출력
    if verbose:
        print("##### ", model_name, " #####")
    
    #회귀 객체 생성
    for param in params:
        if model_name == 'Ridge':
            model = Ridge(alpha = param)
        elif model_name == 'Lasso':
            model = Lasso(alpha = param)
        elif model_name == 'ElasticNet':
            model = ElasticNet(alpha=param, l1_ratio = 0.7)
    
        #평가 지표 찾아오기
        #score는 높은 점수가 좋게 만들기 위해서 
        #평균 제곱 오차에 -를 취합니다.
        neg_mse_scores = cross_val_score(model, X_data_n, y_target_n, 
                                        scoring='neg_mean_squared_error', cv=5)
        #-를 취하고 제곱근을 해서 결과를 가져오기
        avg_rmse = np.mean(np.sqrt(-1 * neg_mse_scores))
        #출력하기
        print('alpha {0} 일 때 평균 RMSE:{1:.3f}'.format(param, avg_rmse))
```


```python
#정규화를 위한 클래스
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#다항식 데이터를 생성해주는 클래스
from sklearn.preprocessing import PolynomialFeatures


#데이터를 받아서 정규화를 수행한 후 리턴하는 함수
#method 는 정규화 방식
#p_degree는 다항식을 이용할 때 차수
#input_data 는 변환할 데이터
def get_scaled_data(method='None', p_degree=None, input_data=None):
    scaled_data = None
    if method == 'Standard':
        scaled_data = StandardScaler().fit_transform(input_data)
    elif method == 'MinMax':
        scaled_data = MinMaxScaler().fit_transform(input_data)   
    elif method == 'Log':
        scaled_data = np.log1p(input_data)
    else:
        scaled_data = input_data
        
    
    if p_degree != None:
        scaled_data = PolynomialFeatures(degree = p_degree,
                        include_bias=False).fit_transform(scaled_data)


    return scaled_data
```


```python
#파라미터 생성
alphas = [0, 0.1, 1, 10, 100]

#정규화 할 옵션 생성
scale_methods = [(None, None), ('Standard', None), ('Standard', 2), 
                ('MinMax', None), ('MinMax', 2), ('Log', None)]

for scale_method in scale_methods:
    #정규화 한 데이터 가져오기
    X_data_scaled = get_scaled_data(method=scale_method[0],
                                    p_degree=scale_method[1],
                                   input_data = X_data)
    print("\n### 변환 유형:{0} Polynomial Degree:{1}".format(scale_method[0], 
                                                        scale_method[1]))
    
    
    get_linear_reg_eval("Ridge", params=alphas, X_data_n = X_data_scaled, 
                       y_target_n = y_target, verbose=True, return_coeff=False)
    
#선형 회귀에서는 일반적인 정규화보다는 로그 변환하는 것이 우수한 회귀 모델을 
#만들 가능성이 높습니다.
```

    
    ### 변환 유형:None Polynomial Degree:None
    #####  Ridge  #####
    alpha 0 일 때 평균 RMSE:5.829
    alpha 0.1 일 때 평균 RMSE:5.788
    alpha 1 일 때 평균 RMSE:5.653
    alpha 10 일 때 평균 RMSE:5.518
    alpha 100 일 때 평균 RMSE:5.330
    
    ### 변환 유형:Standard Polynomial Degree:None
    #####  Ridge  #####
    alpha 0 일 때 평균 RMSE:5.829
    alpha 0.1 일 때 평균 RMSE:5.826
    alpha 1 일 때 평균 RMSE:5.803
    alpha 10 일 때 평균 RMSE:5.637
    alpha 100 일 때 평균 RMSE:5.421
    
    ### 변환 유형:Standard Polynomial Degree:2
    #####  Ridge  #####
    alpha 0 일 때 평균 RMSE:87121381995238.125
    alpha 0.1 일 때 평균 RMSE:8.827
    alpha 1 일 때 평균 RMSE:6.871
    alpha 10 일 때 평균 RMSE:5.485
    alpha 100 일 때 평균 RMSE:4.634
    
    ### 변환 유형:MinMax Polynomial Degree:None
    #####  Ridge  #####
    alpha 0 일 때 평균 RMSE:5.829
    alpha 0.1 일 때 평균 RMSE:5.764
    alpha 1 일 때 평균 RMSE:5.465
    alpha 10 일 때 평균 RMSE:5.754
    alpha 100 일 때 평균 RMSE:7.635
    
    ### 변환 유형:MinMax Polynomial Degree:2
    #####  Ridge  #####
    alpha 0 일 때 평균 RMSE:69966645063581.828
    alpha 0.1 일 때 평균 RMSE:5.298
    alpha 1 일 때 평균 RMSE:4.323
    alpha 10 일 때 평균 RMSE:5.185
    alpha 100 일 때 평균 RMSE:6.538
    
    ### 변환 유형:Log Polynomial Degree:None
    #####  Ridge  #####
    alpha 0 일 때 평균 RMSE:4.829
    alpha 0.1 일 때 평균 RMSE:4.770
    alpha 1 일 때 평균 RMSE:4.676
    alpha 10 일 때 평균 RMSE:4.836
    alpha 100 일 때 평균 RMSE:6.241


### 선형 회귀 모델을 위한 데이터 변환

1.선형 회귀의 가정
- 1)피처들과 레이블 간에는 선형의 관계가 있다.
- 2)피처들간에는 서로 독립적이어야 한다.
- 3)피처들과 레이블은 정규 분포 형태를 선호
- 레이블(타겟값)은 왜곡된 형태를 가지게 되면 예측 성능에 부정적인 영향을 미칠 가능성이 높음, 피처들은 레이블보다 작은 영향을 미침

2.변환 과정
- 1)회귀 모델로 학습
- 2)StandardScaler 나 MinMaxScaler 를 이용해서 데이터를 변환한 후 다시 학습하고 평가지표를 확인
- 3)별다른 효과가 없으면 피처들에 다항식을 적용한 데이터를 가지고 다시 학습하고 평가지표를 확인
- 일반적으로 이 방법을 사용하면 성능이 향상되는 경우가 많음
- 피처가 많아지면 생성되는 데이터도 많아집니다. Ex) 피처 2개 -> 2차 다항식을 이용 : 6개 -> 3차 다항식을 이용하면 피처가 10개가 넘음
- 4)피처의 개수가 많은 경우에는 로그 변환을 고려
- 레이블은 정규 분포냐 아니냐가 성능을 많은 영향을 미치기 때문에 정규 분포가 아닌 경우 거의 무조건 로그 변환을 해야 합니다.
- 로그 변환을 할 떄 numpy 의 log 함수를 잘 이용하지 않는데 언더 플로우가 발생할 가능성이 있기 때문입니다.
- 로그 변환 후 1을 더해주는 log1p() 를 이용하는 경우가 많습니다.
