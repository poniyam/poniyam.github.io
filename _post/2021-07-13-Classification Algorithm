# 기본 분류 알고리즘

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
CHAPTER_ID = "Classification_Basic_Algorithm"
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

## LDA - 선형 판별 분석
### 선형 회귀처럼 하나의 수직선을 만들어서 분류


```python
#데이터 가져오기
loan3000 = pd.read_csv('./data/loan3000.csv')

#outcome 을 레이블로 사용하기 위해서 category로 변환
loan3000.outcome = loan3000.outcome.astype('category')

#독립 변수와 종속 변수 생성
predictors = ['borrower_score', 'payment_inc_ratio']
outcome = 'outcome'

X = loan3000[predictors]
y = loan3000[outcome]

print(X.head())
print()
print(y.head())
```

       borrower_score  payment_inc_ratio
    0            0.40            5.11135
    1            0.40            5.43165
    2            0.70            9.23003
    3            0.40            2.33482
    4            0.45           12.10320
    
    0    paid off
    1     default
    2    paid off
    3    paid off
    4     default
    Name: outcome, dtype: category
    Categories (2, object): ['default', 'paid off']



```python
#LDA 훈련
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

load_lda = LinearDiscriminantAnalysis()
#borrower_score, payment_inc_ratio 에 따른 outcome 분류기 생성
load_lda.fit(X, y)

#최적의 값 찾아오기
print(pd.DataFrame(load_lda.scalings_, index=X.columns))
```

                              0
    borrower_score     7.175839
    payment_inc_ratio -0.099676



```python
#예측에 대한 확률을 확인
pred = pd.DataFrame(load_lda.predict_proba(loan3000[predictors]), 
                                           columns=load_lda.classes_)
print(pred)
```

           default  paid off
    0     0.553544  0.446456
    1     0.558953  0.441047
    2     0.272696  0.727304
    3     0.506254  0.493746
    4     0.609952  0.390048
    ...        ...       ...
    2995  0.652343  0.347657
    2996  0.269632  0.730368
    2997  0.870534  0.129466
    2998  0.557133  0.442867
    2999  0.678458  0.321542
    
    [3000 rows x 2 columns]


## iris 데이터 가져오기

### 붓꽃 품종 관련 데이터로 분류를 설명할 때 가장 많이 사용되는 데이터 중에서 하나


```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


iris = load_iris()

features = iris.data
target = iris.target

#훈련 데이터 와 테스트 데이터 나누기
features_train, features_test, target_train, target_test = train_test_split(
                features, target, random_state = 0)

print(features_train.shape)
print(features_test.shape)
```

    (112, 4)
    (38, 4)


### Dummy 분류기 - 레이블의 비율을 보고 분류를 수행하는 랜덤 분류기


```python
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy='uniform', random_state=1)
dummy.fit(features_train, target_train)

#정확도 점수를 계산
print(dummy.score(features_test, target_test ))
```

    0.42105263157894735


## MNIST 데이터 가져오기
### 70000 개의 숫자 손글씨 데이터


```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target
y = y.astype(np.uint8)
```


```python
#훈련 데이터 와 테스트 데이터 나누기
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


#데이터가 5인지 구분
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

#이진 분류
dmy_clf = DummyClassifier(strategy = 'stratified')
dummy.fit(X_train, y_train_5)

#정확도 점수
print(dummy.score(X_test, y_test_5))
```

    0.4986


## KNN을 이용한 대출 상환 여부 예측


```python
#payment_inc_ratio: 소득에서 부채를 상환하는 비율
#dti: 소득에 대한 부채
#이 2개의 피처를 독립변수로 하고 outcome 을 레이블로 설정

loan200 = pd.read_csv('./data/loan200.csv')

predictors = ['payment_inc_ratio', 'dti']
outcome = 'outcome'

newloan = loan200.loc[0:0, predictors]
#독립 변수 - 설명 변수, feature
X = loan200.loc[1:, predictors]
#레이블 - 종속 변수, target
y = loan200.loc[1:, outcome]

print(X)
print()
print(y)
```

         payment_inc_ratio    dti
    1              5.46933  21.33
    2              6.90294   8.97
    3             11.14800   1.83
    4              3.72120  10.81
    5              1.89923  11.34
    ..                 ...    ...
    196           11.66300  26.09
    197            2.97641  16.41
    198            1.96846  19.98
    199            9.64200  20.17
    200            7.03924  13.88
    
    [200 rows x 2 columns]
    
    1       default
    2      paid off
    3      paid off
    4       default
    5      paid off
             ...   
    196     default
    197     default
    198     default
    199    paid off
    200    paid off
    Name: outcome, Length: 200, dtype: object



```python
from sklearn.neighbors import KNeighborsClassifier
#분류기 생성
knn = KNeighborsClassifier(n_neighbors=20)
#훈련
knn.fit(X, y)
#예측
knn.predict(newloan)
```




    array(['paid off'], dtype=object)




```python
#예축 확률 출력 - 더 높은 쪽으로 판정
print(knn.predict_proba(newloan))
```

    [[0.45 0.55]]


### 표준화를 한 경우 와 그렇지 않은 경우의 비교


```python
#확장자가 gz 이면 압축파일인데 파이썬이 압축을 해제하고 엽니다.
#이런 확장자의 종류로는 tar, zip 등이 있습니다.
loan_data = pd.read_csv('./data/loan_data.csv.gz')
loan_data = loan_data.drop(columns=['Unnamed: 0', 'status'])

#카테고리 화
loan_data['outcome'] = pd.Categorical(loan_data['outcome'], 
                                     categories=['paid off', 'default'],
                                     ordered=True)

predictors = ['payment_inc_ratio', 'dti', 'revol_bal', 'revol_util']
outcome = 'outcome'

newloan = loan_data.loc[0:0, predictors]
print(newloan)
```

       payment_inc_ratio  dti  revol_bal  revol_util
    0             2.3932  1.0       1687         9.4



```python
X = loan_data.loc[1:, predictors]
y = loan_data.loc[1:, outcome]

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X, y)
```




    KNeighborsClassifier()




```python
#newloan 에서 가장 가까운 다섯개의 데이터 확인
nbrs = knn.kneighbors(newloan)
print(X.iloc[nbrs[1][0], :])
#revol_bal 은 거의 유사성을 갖음
#나머지 데이터는 유사성이거의 없습니다.
#나머지 3개의 feature 와 revol_bal 의 피처의 데이터 범위가 너무 다르기 때문에
#revol_bal 의 값의 반영이 많이 되어서 이런 현상이 벌어짐
```

           payment_inc_ratio   dti  revol_bal  revol_util
    35536            1.47212  1.46       1686        10.0
    33651            3.38178  6.37       1688         8.4
    25863            2.36303  1.39       1691         3.5
    42953            1.28160  7.14       1684         3.9
    43599            4.12244  8.98       1684         7.2



```python
#정규화를 수행해서 5개의 이웃을 가져오기
from sklearn import preprocessing

newloan = loan_data.loc[0:0, predictors]

X = loan_data.loc[1:, predictors]
y = loan_data.loc[1:, outcome]

#정규화 수행
scaler = preprocessing.StandardScaler()
scaler.fit(X * 1.0)
X_std = scaler.transform(X*1.0)

newloan_std = scaler.transform(newloan*1.0)

#정규화를 수행한 피처를 가지고 학습
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_std, y)

nbrs = knn.kneighbors(newloan_std)
print(X.iloc[nbrs[1][0], :])
```

           payment_inc_ratio   dti  revol_bal  revol_util
    2080             2.61091  1.03       1218         9.7
    1438             2.34343  0.51        278         9.9
    30215            2.71200  1.34       1075         8.5
    28542            2.39760  0.74       2917         7.4
    44737            2.34309  1.37        488         7.2


## 타이타닉 데이터에서의 생존자 예측 - KNN 알고리즘 사용

### 데이터 가져오기 및 확인


```python
titanic = sns.load_dataset('titanic')
print(titanic.head())
titanic.info()
```

       survived  pclass     sex   age  sibsp  parch     fare embarked  class  \
    0         0       3    male  22.0      1      0   7.2500        S  Third   
    1         1       1  female  38.0      1      0  71.2833        C  First   
    2         1       3  female  26.0      0      0   7.9250        S  Third   
    3         1       1  female  35.0      1      0  53.1000        S  First   
    4         0       3    male  35.0      0      0   8.0500        S  Third   
    
         who  adult_male deck  embark_town alive  alone  
    0    man        True  NaN  Southampton    no  False  
    1  woman       False    C    Cherbourg   yes  False  
    2  woman       False  NaN  Southampton   yes   True  
    3  woman       False    C  Southampton   yes  False  
    4    man        True  NaN  Southampton    no   True  
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 15 columns):
     #   Column       Non-Null Count  Dtype   
    ---  ------       --------------  -----   
     0   survived     891 non-null    int64   
     1   pclass       891 non-null    int64   
     2   sex          891 non-null    object  
     3   age          714 non-null    float64 
     4   sibsp        891 non-null    int64   
     5   parch        891 non-null    int64   
     6   fare         891 non-null    float64 
     7   embarked     889 non-null    object  
     8   class        891 non-null    category
     9   who          891 non-null    object  
     10  adult_male   891 non-null    bool    
     11  deck         203 non-null    category
     12  embark_town  889 non-null    object  
     13  alive        891 non-null    object  
     14  alone        891 non-null    bool    
    dtypes: bool(2), category(2), float64(2), int64(4), object(5)
    memory usage: 80.7+ KB


### 확인한 결과 
age 열에는 167개의 NaN 이 존재 - 행 제거
embarked 열에는 2개의 NaN이 존재 - 치환
deck 열에는 688 개의 NaN이 존재 - 열 제거

embarked 와 embark_town 은 동일한 의미
동일한 의미를 갖는 feature 가 여러 개 존재하면 영향력이 너무 커짐
둘 중에 하나는 제거

### 결측치 처리


```python
#불필요한 열 제거
rdf = titanic.drop(['deck', 'embark_town'], axis=1)

#불필요한 행 제거
rdf = rdf.dropna(subset = ['age'], how='any', axis=0)

#가장 많이 나온 값으로 치환
most_freq = rdf['embarked'].value_counts(dropna = True).idxmax()
rdf['embarked'].fillna(most_freq, inplace=True)

print(rdf.info())
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 714 entries, 0 to 890
    Data columns (total 13 columns):
     #   Column      Non-Null Count  Dtype   
    ---  ------      --------------  -----   
     0   survived    714 non-null    int64   
     1   pclass      714 non-null    int64   
     2   sex         714 non-null    object  
     3   age         714 non-null    float64 
     4   sibsp       714 non-null    int64   
     5   parch       714 non-null    int64   
     6   fare        714 non-null    float64 
     7   embarked    714 non-null    object  
     8   class       714 non-null    category
     9   who         714 non-null    object  
     10  adult_male  714 non-null    bool    
     11  alive       714 non-null    object  
     12  alone       714 non-null    bool    
    dtypes: bool(2), category(1), float64(2), int64(4), object(4)
    memory usage: 63.6+ KB
    None


### 필요한 속성 만 선택


```python
print(rdf.corr())
```

                survived    pclass       age     sibsp     parch      fare  \
    survived    1.000000 -0.359653 -0.077221 -0.017358  0.093317  0.268189   
    pclass     -0.359653  1.000000 -0.369226  0.067247  0.025683 -0.554182   
    age        -0.077221 -0.369226  1.000000 -0.308247 -0.189119  0.096067   
    sibsp      -0.017358  0.067247 -0.308247  1.000000  0.383820  0.138329   
    parch       0.093317  0.025683 -0.189119  0.383820  1.000000  0.205119   
    fare        0.268189 -0.554182  0.096067  0.138329  0.205119  1.000000   
    adult_male -0.552936  0.099021  0.280328 -0.310463 -0.363079 -0.179740   
    alone      -0.196140  0.146320  0.198270 -0.629818 -0.577524 -0.260136   
    
                adult_male     alone  
    survived     -0.552936 -0.196140  
    pclass        0.099021  0.146320  
    age           0.280328  0.198270  
    sibsp        -0.310463 -0.629818  
    parch        -0.363079 -0.577524  
    fare         -0.179740 -0.260136  
    adult_male    1.000000  0.396632  
    alone         0.396632  1.000000  



```python
ndf = rdf[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'embarked']]
```

### 범주형 데이터의 원핫 인코딩
#### 회귀나 분류를 할 때 범주형 데이터는 크기를 가지면 안됩니다.


```python
onehot_sex = pd.get_dummies(ndf['sex'])
ndf = pd.concat([ndf, onehot_sex], axis=1)

onehot_embarked = pd.get_dummies(ndf['embarked'])
ndf = pd.concat([ndf, onehot_embarked], axis=1)

ndf.drop(['sex', 'embarked'], axis=1, inplace=True)

ndf.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 714 entries, 0 to 890
    Data columns (total 10 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   survived  714 non-null    int64  
     1   pclass    714 non-null    int64  
     2   age       714 non-null    float64
     3   sibsp     714 non-null    int64  
     4   parch     714 non-null    int64  
     5   female    714 non-null    uint8  
     6   male      714 non-null    uint8  
     7   C         714 non-null    uint8  
     8   Q         714 non-null    uint8  
     9   S         714 non-null    uint8  
    dtypes: float64(1), int64(4), uint8(5)
    memory usage: 37.0 KB


### 독립 변수 와 종속 변수를 분리
#### sklearn 에서는 이미 분리가 되어 있음(data 와 target)
#### seaborn 은 DataFrame 으로 가져오기 때문에 분리를 해야 합니다.


```python
X = ndf[['pclass', 'age', 'sibsp', 'parch', 'female', 'male', 
             'C', 'Q', 'S' ]]

y = ndf['survived']

print(X.head())
print()
print(y.head())
```

       pclass   age  sibsp  parch  female  male  C  Q  S
    0       3  22.0      1      0       0     1  0  0  1
    1       1  38.0      1      0       1     0  1  0  0
    2       3  26.0      0      0       1     0  0  0  1
    3       1  35.0      1      0       1     0  0  0  1
    4       3  35.0      0      0       0     1  0  0  1
    
    0    0
    1    1
    2    1
    3    1
    4    0
    Name: survived, dtype: int64


### 독립 변수 들의 기술 통계량을 확인 - boxplot 이라도 확인
#### 숫자 자료형의 데이터들의 범위를 확인 
#### 정규화 여부를 결정하기 위해서


```python
print(X.describe())
```

               pclass         age       sibsp       parch      female        male  \
    count  714.000000  714.000000  714.000000  714.000000  714.000000  714.000000   
    mean     2.236695   29.699118    0.512605    0.431373    0.365546    0.634454   
    std      0.838250   14.526497    0.929783    0.853289    0.481921    0.481921   
    min      1.000000    0.420000    0.000000    0.000000    0.000000    0.000000   
    25%      1.000000   20.125000    0.000000    0.000000    0.000000    0.000000   
    50%      2.000000   28.000000    0.000000    0.000000    0.000000    1.000000   
    75%      3.000000   38.000000    1.000000    1.000000    1.000000    1.000000   
    max      3.000000   80.000000    5.000000    6.000000    1.000000    1.000000   
    
                    C           Q           S  
    count  714.000000  714.000000  714.000000  
    mean     0.182073    0.039216    0.778711  
    std      0.386175    0.194244    0.415405  
    min      0.000000    0.000000    0.000000  
    25%      0.000000    0.000000    1.000000  
    50%      0.000000    0.000000    1.000000  
    75%      0.000000    0.000000    1.000000  
    max      1.000000    1.000000    1.000000  


### 확인 결과
#### age 열과 다른 열의 범위가 너무 많이 차이나므로 Scaling 을 해주는 것이 좋음


```python
#스케일링

from sklearn import preprocessing

X = preprocessing.StandardScaler().fit(X).transform(X)

print(X)
```

    [[ 0.91123237 -0.53037664  0.52457013 ... -0.47180795 -0.20203051
       0.53307848]
     [-1.47636364  0.57183099  0.52457013 ...  2.11950647 -0.20203051
      -1.87589641]
     [ 0.91123237 -0.25482473 -0.55170307 ... -0.47180795 -0.20203051
       0.53307848]
     ...
     [-1.47636364 -0.73704057 -0.55170307 ... -0.47180795 -0.20203051
       0.53307848]
     [-1.47636364 -0.25482473 -0.55170307 ...  2.11950647 -0.20203051
      -1.87589641]
     [ 0.91123237  0.15850313 -0.55170307 ... -0.47180795  4.94974747
      -1.87589641]]


### 훈련 데이터 와 테스트 데이터 분리
#### 데이터 개수를 알고 데이터가 랜덤하게 분포되어 있다면 직접 해도 됩니다.
#### 분포를 잘 모른다면 함수를 이용하는 것이 좋습니다.
#### 비율은 8:2 나 7:3 을 권장


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                    test_size=0.2, random_state = 42)
#데이터의 구조를 확인하면 비율을 확인할 수 있음
print(X_train.shape)
print()
print(X_test.shape)
```

    (571, 9)
    
    (143, 9)


### KNN 알고리즘을 이용한 훈련 과 예측


```python
from sklearn.neighbors import KNeighborsClassifier

#k 의 값을 5로 해서 생성
knn = KNeighborsClassifier(n_neighbors = 5)

#모델 학습
knn.fit(X_train, y_train)

#예측
y_hat = knn.predict(X_test)

print(y_hat[0:10])
print(y_test.values[0:10])
```

    [0 1 1 1 0 1 0 1 1 1]
    [0 1 1 1 0 1 1 1 0 0]


### 모형 평가


```python
# 오차 행렬
from sklearn import metrics
knn_matrix = metrics.confusion_matrix(y_test, y_hat)
print(knn_matrix)
```

    [[74 13]
     [16 40]]



```python
#평가지표
knn_report = metrics.classification_report(y_test, y_hat)
print(knn_report)
```

                  precision    recall  f1-score   support
    
               0       0.82      0.85      0.84        87
               1       0.75      0.71      0.73        56
    
        accuracy                           0.80       143
       macro avg       0.79      0.78      0.79       143
    weighted avg       0.80      0.80      0.80       143
    


### 다른 알고리즘을 이용해서 수행
#### 더 나은 평가 지표를 가진 알고리즘이 있는지 확인


```python

```

### 최적의 하이퍼파라미터 찾기 - 하이퍼파라미터 튜닝
#### 가장 좋은 하이퍼파라미터를 찾아서 모델을 다시 생성해서 평가지표 확인
#### 그 후 모델을 선택


```python
from sklearn.model_selection import GridSearchCV

#파라미터 설정 : 총 6번 수행
param_grid = [{'weights':['uniform', 'distance'], 'n_neighbors':[3,4,5]}]

#분류기 생성
knn = KNeighborsClassifier()

#시작 - cv 는 교차검증할 서브셋의 개수
grid_search = GridSearchCV(knn, param_grid, cv=5, verbose=3)
grid_search.fit(X_train, y_train)
```

    Fitting 5 folds for each of 6 candidates, totalling 30 fits
    [CV 1/5] END .................n_neighbors=3, weights=uniform; total time=   0.0s
    [CV 2/5] END .................n_neighbors=3, weights=uniform; total time=   0.0s
    [CV 3/5] END .................n_neighbors=3, weights=uniform; total time=   0.0s
    [CV 4/5] END .................n_neighbors=3, weights=uniform; total time=   0.0s
    [CV 5/5] END .................n_neighbors=3, weights=uniform; total time=   0.0s
    [CV 1/5] END ................n_neighbors=3, weights=distance; total time=   0.0s
    [CV 2/5] END ................n_neighbors=3, weights=distance; total time=   0.0s
    [CV 3/5] END ................n_neighbors=3, weights=distance; total time=   0.0s
    [CV 4/5] END ................n_neighbors=3, weights=distance; total time=   0.0s
    [CV 5/5] END ................n_neighbors=3, weights=distance; total time=   0.0s
    [CV 1/5] END .................n_neighbors=4, weights=uniform; total time=   0.0s
    [CV 2/5] END .................n_neighbors=4, weights=uniform; total time=   0.0s
    [CV 3/5] END .................n_neighbors=4, weights=uniform; total time=   0.0s
    [CV 4/5] END .................n_neighbors=4, weights=uniform; total time=   0.0s
    [CV 5/5] END .................n_neighbors=4, weights=uniform; total time=   0.0s
    [CV 1/5] END ................n_neighbors=4, weights=distance; total time=   0.0s
    [CV 2/5] END ................n_neighbors=4, weights=distance; total time=   0.0s
    [CV 3/5] END ................n_neighbors=4, weights=distance; total time=   0.0s
    [CV 4/5] END ................n_neighbors=4, weights=distance; total time=   0.0s
    [CV 5/5] END ................n_neighbors=4, weights=distance; total time=   0.0s
    [CV 1/5] END .................n_neighbors=5, weights=uniform; total time=   0.0s
    [CV 2/5] END .................n_neighbors=5, weights=uniform; total time=   0.0s
    [CV 3/5] END .................n_neighbors=5, weights=uniform; total time=   0.0s
    [CV 4/5] END .................n_neighbors=5, weights=uniform; total time=   0.0s
    [CV 5/5] END .................n_neighbors=5, weights=uniform; total time=   0.0s
    [CV 1/5] END ................n_neighbors=5, weights=distance; total time=   0.0s
    [CV 2/5] END ................n_neighbors=5, weights=distance; total time=   0.0s
    [CV 3/5] END ................n_neighbors=5, weights=distance; total time=   0.0s
    [CV 4/5] END ................n_neighbors=5, weights=distance; total time=   0.0s
    [CV 5/5] END ................n_neighbors=5, weights=distance; total time=   0.0s





    GridSearchCV(cv=5, estimator=KNeighborsClassifier(),
                 param_grid=[{'n_neighbors': [3, 4, 5],
                              'weights': ['uniform', 'distance']}],
                 verbose=3)




```python
#최적의 파라미터 확인
print(grid_search.best_params_)
```

    {'n_neighbors': 5, 'weights': 'uniform'}



```python
#정확도 확인
print(grid_search.best_score_)
```

    0.8249122807017544


## 나이브베이즈 모델 적용


```python
#가우시안 나이브베이즈 객체를 생성
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

#데이터 훈련
gnb.fit(X_train, y_train)

#예측
y_hat = gnb.predict(X_test)

#10개의 예측 결과 확인
print(y_hat[0:10])
print(y_test.values[0:10])
```

    [0 0 1 1 0 0 1 1 1 1]
    [0 1 1 1 0 1 1 1 0 0]



```python
nb_matrix = metrics.confusion_matrix(y_test, y_hat)
print(nb_matrix)
```

    [[72 15]
     [18 38]]



```python
#평가지표
nb_report = metrics.classification_report(y_test, y_hat)
print(nb_report)
```

                  precision    recall  f1-score   support
    
               0       0.80      0.83      0.81        87
               1       0.72      0.68      0.70        56
    
        accuracy                           0.77       143
       macro avg       0.76      0.75      0.76       143
    weighted avg       0.77      0.77      0.77       143
    



```python

```
