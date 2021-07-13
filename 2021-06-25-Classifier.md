# 분류의 개념 과 평가지표

## 분류 - 분류에서 사용되는 개념은 회귀에서도 동일하게 적용됩니다.
- 회귀에서는 회귀계수와 절편을 만들어내지만 분류에서는 확률을 만들어 냅니다.
- 클래스 이름은 분류는 Clasifier 인데 회귀는 Regressor 가 들어갑니다.
- 지도학습 : 출력(Output)이 미리 존재하는 경우

1.분류
- 1)분류해야할 타겟의 개수
- 2개 이상이면 이진 분류
- 3개 이상이면 다항 분류 - 비선형을 이용하면 조건부 확률을 이용한 이진 분류 문제로 해결 가능

- 2)분류를 할 떄 결정 결계의 모양을 가지고 하느지 그렇지 않은지 
- 선형
- 결정 경계가 아니고 다른 방법을 이용하면 비선형 - 트리

2.Sklearn 에서는 predict 메소드를 이용해서 예측하고 prodict_probal 메소드를 이용해서 확률을 제시

3.알고리즘 종류
- 판별 분류, 랜덤 분류, KNN 알고리즘, SVM(upport Vector Machine), 나이브 베이즈, 로지스틱 회귀 - 이름은 회귀 지만 이진 분류에 사용
- Decision Tree(의사 결정 나무), 최소 근접, 신경망 모형, 여러가지 알고리즘을 섞은 앙상블(RandomForest, Bossting, Bagging 등)


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
CHAPTER_ID = "Classification"
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

## MNIST 데이터 가져오기

4.MNIST - 미국 고등학생과 인구 조사국 직원들이 손으로 쓴 약 70000개의 숫자 이미지 데이터
- 각 이미지에는 어떤 숫자를 나타내는지 레이블 되어 있음
- Sklearn 에서 데이터를 내려 받을 수 있음
- data : 피처 데이터, target : 레이블, DESCR : 데이터 셋을 설명하는 데이터


```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version = 1, as_frame=False)
mnist.keys()
```




    dict_keys(['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'])




```python
#데이터 확인
X, y = mnist['data'], mnist['target']
print(X.shape)
print()
print(y.shape)
```

    (70000, 784)
    
    (70000,)



```python
#feature 를 1개 출력
import matplotlib as mpl

#데이터 1개 출력
some_digit = X[0]
#여러 개의 피처를 정사각형으로 변환
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis('off')
plt.show()
```


    
![png](output_5_0.png)
    



```python
#레이블 1개 출력
print(y[0])
```

    5



```python
#이미지 여러 개를 출력하는 함수
#데이터 와 하나의 행에 출력할 개수 그리고 옵션을 받아서 출력해주는 함수

#매개변수 앞에 * 이 붙으면 set
#매개변수 앞에 ** 이 붙으면 dict - 여러 가지 옵션을 묶을 때 사용
def plot_digits(instances, images_per_row=10, **options):
    # 이미지 크기 설정 : MNIST는 28 * 28
    size = 28
    #열의 개수 구하기
    images_per_row = min(len(instances), images_per_row)
    #데이터를 순회하면서 28*28로 변환
    images = [instance.reshape(size, size) for instance in instances]
    #행의 개수 구하기
    n_rows = (len(instances) - 1) // images_per_row + 1
    
    #이미지를 저장할 리스트
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    #0으로 가득채운 행렬을 만들어서 채우기
    images.append(np.zeros((size, size * n_empty)))
    
    #순회하면서 이미지 채우기
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row+1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    
    image = np.concatenate(row_images, axis = 0)
    
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")
    
```


```python
plt.figure(figsize = (9,9))
example_images=X[:100]
plot_digits(example_images, images_per_row = 10)
plt.show()
```


    
![png](output_8_0.png)
    



```python
print(y.dtype)

#레이블의 자료형을 정수로 변환 - np.uint8(부호없는 양의 정수 8byte)
#메모리를 적게 사용하기 위해서 uint8로 변환
y = y.astype(np.uint8)

print(y.dtype)
```

    object
    uint8



```python
#훈련 세트 와 테스트 세트로 분류
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```

## 이진 분류하기

5.Sklearn 의 linear_model 의 SGDClassfier
- 확률적 경사 하강법을 이용해서 이진 분류를 해주는 클래스
- 아주 큰 데이터 셋을 효율적으로 처리
- 온라인 학습에 유리
- 객체 생성 시 주요한 하이퍼 파라미터ㅓ
- max_iter: 최대 반복 횟수, tol: 최대 허용 오차


```python
#이진 분류를 위해서 데이터를 구분
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
print(y_train[:5])
print(y_train_5[:5])

```

    [5 0 4 1 9]
    [ True False False False False]



```python
#확률적 경사하강법을 이용하는 이진분류기 생성
from sklearn.linear_model import SGDClassifier

#분류기 생성
sgd_clf = SGDClassifier(max_iter = 1000, tol = 1e-3, random_state=42)
#훈련
sgd_clf.fit(X_train, y_train_5)

```




    SGDClassifier(random_state=42)




```python
#예측
sgd_clf.predict([X[0]])
```




    array([ True])



### 교차 검증

6.평가지표

1)교차검증을 위한 정확도 측정 - 교차 검증은 클래스 별 비율이 유지되도록(계층적 샘플링) 폴드를 만들어서 검증을 수행, 폴드는 전체 데이터를 복원 추출하지 않고 생성
- 하이퍼파라미터는 첫번째로 평가를 수행할 분류나 회귀 객체가 되고 두번째는 훈련 데이터가 되고 세번째는 레이블이고 cv 옵션에 생성할 폴드의 개수를 설정하며 scoring 에 원하는 평가지표를 설정합니다.
- 회귀와 분류가 설정하는 지표가 다릅니다.
- 회귀에서는 평균 오차를 설정하지만 분류에서는 accuracy 와 같은 평가 지표를 이용합니다.

2)분류의 평가지표를 위한 Metrix - 오차 행렬
        실제 정답
         True            False
분류결과   True        True Positive(TP-정답)    False Positive(FP)
        False      False Negative(FN)         True Negative(TN)

3)Accuracy(정확도) - (TP+TN) / 전체 합
- 전체 데이터에서 정확하게 판정한 데이터의 비율
- 이 평가지표의 문제점은 데이터의 편중입니다.
- True 인 데이터나 False 인 데이터가 압도적으로 많은 경우 반대편은 틀려도 영향을 별로 받지 않습니다.

날씨 예측의 경우 
- 맑은 날 50일 흐린 날이 50일
- 맑은 날은 전부 예측하고 흐린 날은 절반만 맞춘 경우 - (50 + 25)/100 : 75%
- 맑은 날 98일 흐린 날이 2일
- 맑은 날은 전부 예측하고 흐린 날은 절반만 맞춘 경우 - (98 + 1)/100 : 99%

4)Precision(정밀도) - TP/(TP + FP) 
- True 라고 판정한 것 중에서 True 인 비율

5)Recall(재현율) - TP/(TP + FN)
- Sensitivity(민감도) 라고 하거나 Hit Rate 라고 합니다.
- 실제 True 인 데이터를 True 라고 예측한 비율

6)재현율과 정밀도를 같이 고려해서 평가를 해야 합니다.

7)F1 Score - Preciion 과 Recall 의 조화 평균
- 2 * ((Precision * Recall)/(precision + Recall))

8)sklearn 에서의 평가지표
- 오차행렬은 sklearn.model_selection 의 cross_val_predict 와 sklearn.metrics 의 confusion_matrix 를 이용해서 생성
- sklearn.metrics 의 accuracy_score, preciion_score, recall_score, f1_score 를 이용해서 구할 수 있습니다.

9)정밀도와 재현율의 트레이드 오프
- 정밀도를 높이면 재현율이 낮아지고 정밀도를 낮추면 재현율이 높아집니다.
- SGDClassifier 클래스에서는 predict 대신에 점수를 리턴하는 decision_function()메소드를 제공하는데 이 점수가 임계값을 설정해서 재현율과 정밀도를 조절할 수 있습니다.
- 정밀도와 재현율을 시각화 해서 원하는 결정 점수를 얻어낼 수 있습니다.
- 급격하게 값이 변화하는 지점을 결정 점수로 설정하는 경우가 많습니다.


```python
#교차 검증을 위한 함수
from sklearn.model_selection import cross_val_score

#3겹으로 정확도를 측정
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy' )
```




    array([0.95035, 0.96035, 0.9604 ])




```python
#오차 행렬 출력
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=5)

confusion_matrix(y_train_5, y_train_pred)

#5를 5라고 분류한 것이 53115 개 
#5가 아닌 것을 5라고 분류한 것이 1464개
#5를 5가 아니라고 분류한 것이 916개
#5가 아닌 것을 5가 아니라고 분류한 것이 4505개
```




    array([[53115,  1464],
           [  916,  4505]])




```python
#평가 지표들을 출력
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("정확도:", accuracy_score(y_train_5, y_train_pred))
print("정밀도:", precision_score(y_train_5, y_train_pred))
print("재현율:", recall_score(y_train_5, y_train_pred))
print("f1_score:", f1_score(y_train_5, y_train_pred))
```

    정확도: 0.9603333333333334
    정밀도: 0.7547327860613168
    재현율: 0.8310274857037447
    f1_score: 0.791044776119403



```python
#점수를 출력
y_scores = sgd_clf.decision_function([X[0]])
print(y_scores)
```

    [2164.22030239]



```python
#임계값을 설정해서 예측
threshold = 0

y_zero_digit_pred = (y_scores > threshold)
print(y_zero_digit_pred)

#임계값을 3000으로 설정
threshold = 3000

y_zero_digit_pred = (y_scores > threshold)
print(y_zero_digit_pred)
```

    [ True]
    [False]


## ROC 곡선

** 분류의 평가지표
1.Accuracy(정확도): 전체 분류한 개수 중에서 옳게 분류한 (T->T, F->F) 개수의 비율

2.Precision(정밀도):True 로 분류한 것 중에서 True 가 맞는 개수의 비율
- TP/(TP + FP - True 로 분류했는데 False 로 분류)

3.Recall(재현율): 실제 True 인 것 중에서 True 로 분류한 비율
- TP/(TP + FN - True 인데 False 로 분류)

4.F1 Value
- 정밀도와 재혀율은 트레이드 오프
- 정밀도와 재현율의 조화 평균
- 2 * (정밀도 X 재현율)/(정밀도 + 재현율)

5.ROC(Receiver Operating Characteristic) 곡선 - 정밀도 / 재현율 곡선과 비슷하지만 거짓 양성 비율(FPR)에 대한 진짜 양성 비율(재현율) 곡선
- FPR 은 True 로 잘못 분류된 False 데이터의 비율로 1에서 False 로 정확하게 분류에 False 샘플 비율인 진짜 음성 비율을 뺀 값으로 이 값을 Specificity(특이도)라고도 함
- ROC 곡선은 미감도에 대한 1-특이도 그래프
- fpr, tpr, thresholds 는 roc_curve 라는 함수를 이용해서 구할 수 있음

- 곡선 아래 면적을 AUC(area under the curve)라고 하면 이 면적은 바르게 분류한 영역의 면적입니다.
- 이 면적이 1이면 최적의 완변한 분류기이며 0.5이면 랜덤 분류기 입니다.
- sklearn.metrics.roc_score 함수에 훈련된 데이터와 예측한 점수를 대입하면 구해줍니다.

- 원래 데이터에서 True 인 데이터의 비율이 좋으면 점수가 높게 나올 가능성이 높습니다. - 이에 대한 보완책으로 PR을 사용하기도 합니다.
- PR 은 분류기의 성능 개선 여지가 얼마나 되는지 보여주는 곡선입니다.
- True 와 False 의 비율이 차이가 많거나 거짓 양성이 중요할 때 PR 곡선을 사용합니다.


```python
#평가지표와 관련된 함수는 대부분 sklearn.metrics 에 존재
from sklearn.metrics import roc_curve

#훈련 데이터의 예측 점수를 구함
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, 
                            method='decision_function')
print(y_scores)
```

    [  1200.93051237 -26883.79202424 -33072.03475406 ...  13272.12718981
      -7258.47203373 -16877.50840447]



```python
#fpr, tpr, thresholds 값 구하기
fpr, tpr, threholds = roc_curve(y_train_5, y_scores)
```


```python
#ROC 곡선 그리는 함수
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, linewidth=2, label = label)
    plt.plot([0,1], [0,1], 'k--') #대각선
    plt.axis([0,1, 0,1])
    plt.xlabel('F P R(Fall Out)', fontsize=16)
    plt.ylabel('T P R(Recall)', fontsize=16)
    plt.grid(True)
```


```python
#정밀도가 90% 가 되는 지점의 재현율 인덱스 찾기
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
print(recall_90_precision)
```

    0.4799852425751706



```python
#ROC 곡선 그리기
plt.figure(figsize = (8,6))
plot_roc_curve(fpr, tpr)
fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]
plt.plot([fpr_90, fpr_90], [0, recall_90_precision], "r:")
plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")
plt.plot([fpr_90], [recall_90_precision], "ro:")
plt.show()
```


    
![png](output_26_0.png)
    



```python
#훈련 데이터의 예측 점수를 구함
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, 
                            method='decision_function')

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_train_5, y_scores))
```

    0.9604938554008616



```python
#이진 분류 : 오차행렬, 정확도, 정밀도, 재현율, F1-Score, ROC곡선
#=> 분류할 대상이 2가지 값만 가지는 범주형인 경우 사용하는 분류

#다중 분류(Multi Calss Classifier) - 분류할 대상이 3가지 값 이상을 갖는 범주형인 경우 사용하는 분류

#if ~ elif 구조는 if ~ else 구문으로 전부 치환이 가능합니다.
```

## 다중 분류(Multi Class Classifier)

### 다중 분류기의 결과 확인

다중분류 - 이진분류는 분류할 대상이 2가지 값만 가지는 범주형인 경우 사용하는 분류이고 다중 분류는 분류할 대상이 3가지 값 이상을 갖는 범류형인 경우 사용하는 분류
- 로지스틱 회귀나 SVM 은 이진 분류만 가능하고 SGD 나 DecisionTree, RandomForest, Naive Bayes 등의 분류기는 다중 분류가 가능합니다.
- 이진 분류기를 이용해서 다중 분류도 가능합니다.

if ~ elif 구조는 if ~ else 구문으로 전부 치환이 가능합니다.

1.전략
- 다중 클래스 들의 결정 점수를 구해서 가장 점수가 높은 것을 선택하는 방식이 있는데 OvR(One-verus-the-Rest) 또는 OvA(One-versus-All) 전략이라고 합니다.
- OvO(One-Versus-One) 전략: 모든 클래스에 대해서 2개씩 비교를 해서 선택해 나가는 방식으로 10개의 클래스가 있다면 45번 비교를 수행
- 0 -1,0 -2, 0 -3...... 9 -8 순서대로 비교해 나가는 방식
- 이진 분류에서는 OvR 을 많이 사용하고 SVM 은 OvO를 선호

2.sklearn 에서 OvO 나 OvR 을 강제 
- 특별한 경우가 아니면 잘 사용하지 않음
- sklearn.multicalss.OneVsRestClaifier 나 OveVsOveClafier 클래스의 객체를 만들 때 분류기를 대입하면 됩니다.

3.SGD 분류기
- 다중 분류를 지원하기 때문에 별도의 전략을 사용할 필요가 없습니다.


```python
from sklearn.svm import SVC

print(y_train[:5])

some_digit = X[2]

svm_clf = SVC(gamma = 'auto', random_state=42)
svm_clf.fit(X_train[:1000], y_train[:1000])
#다중 분류는 결과가 레이블의 값이 범주형일 때 사용합니다.
svm_clf.predict([some_digit])
```

    [5 0 4 1 9]





    array([4], dtype=uint8)




```python
#타겟의 목록 확인
print(svm_clf.classes_)

#결정 점수를 출력
some_digit_scores = svm_clf.decision_function([some_digit])
print(some_digit_scores)

#결정 점수가 가장 높은 인덱스를 찾아서 clasess_에서 값을 찾은 후 리턴
print(np.argmax(some_digit_scores))

#인덱스를 가지고 claes_를 찾아가서 인덱스에 해당하는 값을 리턴
print(svm_clf.classes_[np.argmax(some_digit_scores)])
```

    [0 1 2 3 4 5 6 7 8 9]
    [[ 3.82111996  7.09167958  4.83444983  1.79943469  9.29932174  0.79485736
       2.80437474  8.10392157 -0.22417259  5.84182891]]
    4
    4


## SVM 이 OVR 전략을 사용하도록 강제하기


```python
from sklearn.multiclass import OneVsRestClassifier

#sklearn 이 왜 개발자들이 선호하는지 - 변환이 간단함
ovr_clf = OneVsRestClassifier(SVC(gamma = 'auto', random_state=42))
ovr_clf.fit(X_train[:1000], y_train[:1000])
ovr_clf.predict([some_digit])
```




    array([4], dtype=uint8)



### SGD 분류기


```python
#확률적 경사하강법을 이용하는 이진분류기 생성
from sklearn.linear_model import SGDClassifier

#분류기 생성
sgd_clf = SGDClassifier(max_iter = 1000, tol = 1e-3, random_state=42)

#훈련 데이터 와 타겟 데이터를 이용해서 알고리즘 생성
sgd_clf.fit(X_train[:1000], y_train[:1000])
#예측
print(sgd_clf.predict([some_digit]))
#결정점수 기반(decision function 기반)
print(sgd_clf.decision_function([some_digit]))
```

    [4]
    [[-5204706.98947443 -4954638.35555353 -1409644.85952414  -249333.37139651
       1424883.39627106 -2604110.05301226 -3639676.47119989 -3844853.63309997
      -2229525.57021486 -2339588.94079138]]


### 교차 검증


```python
#cv 는 교차검증할 때 데이터를 나눌 서브셋의 개수입니다.
#scoring 은 구하고자 하는 평가지표입니다.
print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy'))
```

    [0.87365 0.85835 0.8689 ]



```python
print(cross_val_score(svm_clf, X_train, y_train, cv=3, scoring='accuracy'))
```

    [0.11235 0.11235 0.1124 ]



```python
#스케일링을 하고 다시 학습한 후 평가지표를 확인
from sklearn.preprocsssing import StandardScaler
#훈련 데이터를 평균은 0이고 표준 편차는 1인 데이터로 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
print(cross_val_score(svm_clf, X_train_scaled, y_train, cv=3, scoring='accuracy'))
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-31-ed92a75b43d0> in <module>
          1 #스케일링을 하고 다시 학습한 후 평가지표를 확인
    ----> 2 from sklearn.preprocsssing import StandardScaler
          3 #훈련 데이터를 평균은 0이고 표준 편차는 1인 데이터로 표준화
          4 scaler = StandardScaler()
          5 X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))


    ModuleNotFoundError: No module named 'sklearn.preprocsssing'



```python
#오차 행렬(맞게 분류한 것과 그렇지 않게 분류한 것을 행렬로 표현)을 출력
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv3)
conf_mx = confusion_matrix(y_train, y_train_pred)
print(conf_mx)
```


```python
#오차 행렬을 matplotlib 의 matshow 로 시각화 하는 것이 가능
#밝게 나타나면 많은 것이 그렇지 않으면 적은것
#클래스 별로 데이터의 개수가 다르면 같이 비교하기가 어렵습니다.
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
```


```python
#주 대각선의 내용을 검정색으로 반영하고 각 클래스의 합계를 시각화
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

np.fill_diagonal(norm_conf_mx, 0)

plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
```

## 다중 레이블 분류 : 여러 개의 레이블을 한꺼번에 출력해야 하는 분류

1.작업순서
- 두개 이상의 타겟 레이블이 담긴 y_multilabel 배열을 생성, 첫번째와 두번째가 서로 독립적인 클래스
- 다중 레이블 분류를 지원하는 KNeighborClassifier 와 같은 클래스를 이용해서 분류

2.다중 레이블은 일반적으로 F1-Score 로 평가
- 모든 레이블의 존재 비율이 거의 동일하다고 판정합니다.
- 레이블의 비율이 다르다면 average="weightrf"로 설정해서 가중치를 부여해야 합니다.

**다중 출력 다중 클래스 분류
- 다중 레이블 분류에서 한 레이블이 다중 클래스가 될 수 있도록 변환한 것


```python
#다중 레이블 생성

#y_train 의 값을 2가지 조건을 이용해서 분해
y_train_large = (y_train >= 7) #7보다 크거나 같은 경우 True
y_train_doo = (y_train % 2 == 1) #홀수이면 True
y_multilabel = np.c_[y_train_large, y_train_odd]

#다중 레이블 분류기 생성
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsclassifier()
knn_clf.fit(X_train, y_mutilabel)

some_digit = X[0]
knn_clf.predict([some_digit])
```


```python
#다중 레이블 분류기의 평가 - F1 Score 이용
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_mulilabel, cv=3)
#macro 는 모든 클래스의 비율이 동일하다고 가정
#weighted 를 이용하면 데이터 개수에 따른 가중치를 부여
#micro 를 설정하면 FP, TP, FN 의 총합을 이용해서 계산
#f1_score 이외에서 accureacy_score, precision_score, recall_score 에도 설정이 가능
print(f1_score(y_multilabel, y_train_knn_pred, average='macro'))
```
