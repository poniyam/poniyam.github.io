# 앙상블


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
CHAPTER_ID = "emsemble"
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

## 직접 생성하는 투표 기반 분류기

### 비선형 데이터 생성하기


```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
print(X_train.shape)
print(X_test.shape)
```

    (375, 2)
    (125, 2)



```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

#개별 분류기 생성
log_clf = LogisticRegression(solver='lbfgs', random_state=42)
svm_clf = SVC(gamma='scale', random_state=42)
rnd_clf = RandomForestClassifier(n_estimators = 100, random_state=42)

#개별 분류기를 다수결로 동작시키기 위한 앙상블 분류기 생성
#분류된 클래스의 레이블을 가지고 다수결로 투표를 하는 방식을 직접 투표 방식이라고 합니다.
#voting 을 hard 라고 설정하면 됨
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('svc', svm_clf), ('rf', rnd_clf)],
    voting='hard')

#분류기 훈련
voting_clf.fit(X_train, y_train)
```




    VotingClassifier(estimators=[('lr', LogisticRegression(random_state=42)),
                                 ('svc', SVC(random_state=42)),
                                 ('rf', RandomForestClassifier(random_state=42))])




```python
#정확도 확인
from sklearn.metrics import accuracy_score

for clf in (log_clf, svm_clf, rnd_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, " 정확도:", accuracy_score(y_test, y_pred))

#투표 기반 분류기가 더 높은 정확도를 나타냄
```

    LogisticRegression  정확도: 0.864
    SVC  정확도: 0.896
    RandomForestClassifier  정확도: 0.896
    VotingClassifier  정확도: 0.912



```python
#개별 분류기 생성
log_clf = LogisticRegression(solver='lbfgs', random_state=42)
#svm 이 확률을 사용할 수 있도록 probability를 True로 설정 - 속도는 느려짐
svm_clf = SVC(gamma='scale', random_state=42, probability=True)
rnd_clf = RandomForestClassifier(n_estimators = 100, random_state=42)

#개별 분류기를 다수결로 동작시키기 위한 앙상블 분류기 생성
#확률을 이용하는 방식을 간접 투표 방식이라고 합니다.
#voting 을 soft 로 설정하면 됩니다.
#간접 투표 방식은 predict_proba()를 사용할 수 있는 분류기만 가능
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('svc', svm_clf), ('rf', rnd_clf)],
    voting='soft')

#분류기 훈련
voting_clf.fit(X_train, y_train)
```




    VotingClassifier(estimators=[('lr', LogisticRegression(random_state=42)),
                                 ('svc', SVC(probability=True, random_state=42)),
                                 ('rf', RandomForestClassifier(random_state=42))],
                     voting='soft')




```python
#정확도 확인
from sklearn.metrics import accuracy_score

for clf in (log_clf, svm_clf, rnd_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, " 정확도:", accuracy_score(y_test, y_pred))

#투표 기반 분류기가 더 높은 정확도를 나타냄
```

    LogisticRegression  정확도: 0.864
    SVC  정확도: 0.896
    RandomForestClassifier  정확도: 0.896
    VotingClassifier  정확도: 0.92


### 결정 트리 500개를 배깅 방식으로 훈련하는 앙상블


```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

#결정 트리를 이용한 앙상블
#결정 트리 개수는 500개를 사용
#각 서브셋의 데이터 개수는 100개
#bootstrap=True 을 설정하면 배깅
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators = 500, 
                           max_samples=100, bootstrap=True, random_state=42)

#훈련
bag_clf.fit(X_train, y_train)
#예측
y_pred = bag_clf.predict(X_test)

#평가 지표 확인
print(accuracy_score(y_test, y_pred))
```

    0.904



```python
tree_clf = DecisionTreeClassifier(random_state=42)

#훈련
tree_clf.fit(X_train, y_train)
#예측
y_pred = tree_clf.predict(X_test)

#평가 지표 확인
print(accuracy_score(y_test, y_pred))
```

    0.856


### oob 평가
훈련에 선택되지 않은 데이터를 가지고 평가


```python
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, 
                           bootstrap=True, oob_score=True, random_state=42)
bag_clf.fit(X_train, y_train)
print('oob 평가:', bag_clf.oob_score_)
```

    oob 평가: 0.896



```python
# 각 샘플의 클래스 확률
print(bag_clf.oob_decision_function_)
```

    [[0.32352941 0.67647059]
     [0.3375     0.6625    ]
     [1.         0.        ]
     [0.         1.        ]
     [0.         1.        ]
     [0.06145251 0.93854749]
     [0.35465116 0.64534884]
     [0.01142857 0.98857143]
     [0.98930481 0.01069519]
     [0.97927461 0.02072539]
     [0.75586854 0.24413146]
     [0.0049505  0.9950495 ]
     [0.75520833 0.24479167]
     [0.82122905 0.17877095]
     [0.98461538 0.01538462]
     [0.06315789 0.93684211]
     [0.00490196 0.99509804]
     [0.99004975 0.00995025]
     [0.92513369 0.07486631]
     [1.         0.        ]
     [0.03409091 0.96590909]
     [0.35087719 0.64912281]
     [0.91111111 0.08888889]
     [1.         0.        ]
     [0.96319018 0.03680982]
     [0.         1.        ]
     [1.         0.        ]
     [1.         0.        ]
     [0.         1.        ]
     [0.6635514  0.3364486 ]
     [0.         1.        ]
     [1.         0.        ]
     [0.0049505  0.9950495 ]
     [0.         1.        ]
     [0.19680851 0.80319149]
     [1.         0.        ]
     [0.00591716 0.99408284]
     [0.38509317 0.61490683]
     [0.         1.        ]
     [1.         0.        ]
     [0.22346369 0.77653631]
     [0.32777778 0.67222222]
     [1.         0.        ]
     [1.         0.        ]
     [0.         1.        ]
     [1.         0.        ]
     [1.         0.        ]
     [0.02906977 0.97093023]
     [1.         0.        ]
     [0.01183432 0.98816568]
     [0.97916667 0.02083333]
     [0.88829787 0.11170213]
     [0.94594595 0.05405405]
     [0.96132597 0.03867403]
     [0.         1.        ]
     [0.05612245 0.94387755]
     [0.98026316 0.01973684]
     [0.         1.        ]
     [0.         1.        ]
     [0.01005025 0.98994975]
     [0.98857143 0.01142857]
     [0.80769231 0.19230769]
     [0.44751381 0.55248619]
     [1.         0.        ]
     [0.         1.        ]
     [0.71523179 0.28476821]
     [1.         0.        ]
     [1.         0.        ]
     [0.84745763 0.15254237]
     [1.         0.        ]
     [0.61578947 0.38421053]
     [0.12777778 0.87222222]
     [0.63387978 0.36612022]
     [0.9132948  0.0867052 ]
     [0.         1.        ]
     [0.16129032 0.83870968]
     [0.90607735 0.09392265]
     [1.         0.        ]
     [0.         1.        ]
     [1.         0.        ]
     [0.00540541 0.99459459]
     [0.06358382 0.93641618]
     [0.02312139 0.97687861]
     [0.34939759 0.65060241]
     [1.         0.        ]
     [0.         1.        ]
     [0.84090909 0.15909091]
     [0.         1.        ]
     [0.         1.        ]
     [0.         1.        ]
     [0.22340426 0.77659574]
     [1.         0.        ]
     [0.         1.        ]
     [0.         1.        ]
     [0.         1.        ]
     [0.96335079 0.03664921]
     [0.80628272 0.19371728]
     [0.         1.        ]
     [1.         0.        ]
     [0.25274725 0.74725275]
     [0.58235294 0.41764706]
     [0.         1.        ]
     [0.03723404 0.96276596]
     [0.50543478 0.49456522]
     [1.         0.        ]
     [0.01704545 0.98295455]
     [0.99456522 0.00543478]
     [0.2259887  0.7740113 ]
     [0.45810056 0.54189944]
     [1.         0.        ]
     [0.01570681 0.98429319]
     [0.98901099 0.01098901]
     [0.25698324 0.74301676]
     [0.88304094 0.11695906]
     [1.         0.        ]
     [1.         0.        ]
     [0.         1.        ]
     [0.         1.        ]
     [0.78285714 0.21714286]
     [1.         0.        ]
     [0.00578035 0.99421965]
     [1.         0.        ]
     [1.         0.        ]
     [1.         0.        ]
     [0.98895028 0.01104972]
     [0.99447514 0.00552486]
     [0.         1.        ]
     [0.96534653 0.03465347]
     [0.99497487 0.00502513]
     [0.01092896 0.98907104]
     [0.1686747  0.8313253 ]
     [0.95408163 0.04591837]
     [0.31515152 0.68484848]
     [0.98924731 0.01075269]
     [0.         1.        ]
     [0.         1.        ]
     [0.71929825 0.28070175]
     [0.36206897 0.63793103]
     [0.4        0.6       ]
     [0.85326087 0.14673913]
     [0.95918367 0.04081633]
     [0.06349206 0.93650794]
     [0.81313131 0.18686869]
     [0.         1.        ]
     [0.         1.        ]
     [0.03508772 0.96491228]
     [0.98342541 0.01657459]
     [1.         0.        ]
     [1.         0.        ]
     [0.0052356  0.9947644 ]
     [0.         1.        ]
     [0.01932367 0.98067633]
     [0.         1.        ]
     [1.         0.        ]
     [1.         0.        ]
     [0.94827586 0.05172414]
     [1.         0.        ]
     [1.         0.        ]
     [1.         0.        ]
     [0.         1.        ]
     [0.38674033 0.61325967]
     [0.27717391 0.72282609]
     [0.01117318 0.98882682]
     [0.         1.        ]
     [0.30208333 0.69791667]
     [1.         0.        ]
     [0.99378882 0.00621118]
     [0.         1.        ]
     [1.         0.        ]
     [0.         1.        ]
     [0.         1.        ]
     [0.98295455 0.01704545]
     [0.         1.        ]
     [0.         1.        ]
     [1.         0.        ]
     [0.00502513 0.99497487]
     [0.61271676 0.38728324]
     [0.91747573 0.08252427]
     [0.         1.        ]
     [1.         0.        ]
     [0.99516908 0.00483092]
     [1.         0.        ]
     [0.         1.        ]
     [0.         1.        ]
     [1.         0.        ]
     [0.06930693 0.93069307]
     [1.         0.        ]
     [0.05405405 0.94594595]
     [0.         1.        ]
     [1.         0.        ]
     [0.         1.        ]
     [0.04294479 0.95705521]
     [1.         0.        ]
     [0.93229167 0.06770833]
     [0.73295455 0.26704545]
     [0.62424242 0.37575758]
     [0.         1.        ]
     [0.12755102 0.87244898]
     [1.         0.        ]
     [0.96341463 0.03658537]
     [0.96226415 0.03773585]
     [1.         0.        ]
     [0.02272727 0.97727273]
     [0.         1.        ]
     [0.40223464 0.59776536]
     [0.85082873 0.14917127]
     [0.         1.        ]
     [0.         1.        ]
     [0.99408284 0.00591716]
     [0.00520833 0.99479167]
     [0.00540541 0.99459459]
     [0.96111111 0.03888889]
     [0.         1.        ]
     [0.30808081 0.69191919]
     [0.         1.        ]
     [1.         0.        ]
     [0.00595238 0.99404762]
     [0.         1.        ]
     [0.96236559 0.03763441]
     [0.80246914 0.19753086]
     [0.995      0.005     ]
     [0.00543478 0.99456522]
     [0.04411765 0.95588235]
     [1.         0.        ]
     [0.02808989 0.97191011]
     [0.         1.        ]
     [0.04864865 0.95135135]
     [1.         0.        ]
     [0.82777778 0.17222222]
     [0.         1.        ]
     [0.8988764  0.1011236 ]
     [0.99450549 0.00549451]
     [0.18857143 0.81142857]
     [0.20994475 0.79005525]
     [1.         0.        ]
     [0.         1.        ]
     [0.00549451 0.99450549]
     [0.         1.        ]
     [0.25789474 0.74210526]
     [0.95212766 0.04787234]
     [0.00507614 0.99492386]
     [1.         0.        ]
     [1.         0.        ]
     [0.         1.        ]
     [0.42857143 0.57142857]
     [1.         0.        ]
     [0.         1.        ]
     [1.         0.        ]
     [0.         1.        ]
     [0.         1.        ]
     [0.04188482 0.95811518]
     [0.10404624 0.89595376]
     [0.98469388 0.01530612]
     [0.02185792 0.97814208]
     [1.         0.        ]
     [0.34042553 0.65957447]
     [0.12315271 0.87684729]
     [0.52717391 0.47282609]
     [0.65151515 0.34848485]
     [0.00558659 0.99441341]
     [1.         0.        ]
     [0.         1.        ]
     [0.         1.        ]
     [0.63679245 0.36320755]
     [0.         1.        ]
     [1.         0.        ]
     [0.22680412 0.77319588]
     [0.82089552 0.17910448]
     [0.03012048 0.96987952]
     [1.         0.        ]
     [0.86910995 0.13089005]
     [0.         1.        ]
     [0.00518135 0.99481865]
     [0.09137056 0.90862944]
     [0.00598802 0.99401198]
     [0.         1.        ]
     [1.         0.        ]
     [0.94318182 0.05681818]
     [0.1627907  0.8372093 ]
     [0.95897436 0.04102564]
     [0.01630435 0.98369565]
     [0.59562842 0.40437158]
     [0.06989247 0.93010753]
     [1.         0.        ]
     [0.78723404 0.21276596]
     [0.         1.        ]
     [1.         0.        ]
     [0.93684211 0.06315789]
     [0.         1.        ]
     [0.         1.        ]
     [1.         0.        ]
     [0.         1.        ]
     [1.         0.        ]
     [0.23036649 0.76963351]
     [0.99418605 0.00581395]
     [1.         0.        ]
     [0.         1.        ]
     [0.         1.        ]
     [0.85955056 0.14044944]
     [0.         1.        ]
     [1.         0.        ]
     [0.79255319 0.20744681]
     [0.91847826 0.08152174]
     [1.         0.        ]
     [0.68306011 0.31693989]
     [0.50802139 0.49197861]
     [0.         1.        ]
     [0.87790698 0.12209302]
     [0.         1.        ]
     [1.         0.        ]
     [0.9010989  0.0989011 ]
     [1.         0.        ]
     [1.         0.        ]
     [0.72139303 0.27860697]
     [0.09090909 0.90909091]
     [0.48170732 0.51829268]
     [0.25263158 0.74736842]
     [0.         1.        ]
     [0.84782609 0.15217391]
     [0.81443299 0.18556701]
     [0.         1.        ]
     [1.         0.        ]
     [0.98930481 0.01069519]
     [1.         0.        ]
     [0.         1.        ]
     [0.03921569 0.96078431]
     [0.95959596 0.04040404]
     [0.95375723 0.04624277]
     [1.         0.        ]
     [0.52542373 0.47457627]
     [1.         0.        ]
     [0.0106383  0.9893617 ]
     [0.99453552 0.00546448]
     [0.02659574 0.97340426]
     [1.         0.        ]
     [1.         0.        ]
     [1.         0.        ]
     [0.         1.        ]
     [0.98275862 0.01724138]
     [0.         1.        ]
     [0.08421053 0.91578947]
     [0.         1.        ]
     [0.         1.        ]
     [1.         0.        ]
     [1.         0.        ]
     [0.         1.        ]
     [1.         0.        ]
     [0.00564972 0.99435028]
     [1.         0.        ]
     [0.14124294 0.85875706]
     [0.         1.        ]
     [0.         1.        ]
     [0.         1.        ]
     [0.37125749 0.62874251]
     [0.08695652 0.91304348]
     [0.23255814 0.76744186]
     [1.         0.        ]
     [0.98181818 0.01818182]
     [0.19444444 0.80555556]
     [0.9902439  0.0097561 ]
     [0.         1.        ]
     [0.         1.        ]
     [1.         0.        ]
     [0.95930233 0.04069767]
     [0.34285714 0.65714286]
     [0.98843931 0.01156069]
     [1.         0.        ]
     [0.         1.        ]
     [1.         0.        ]
     [0.00561798 0.99438202]
     [0.02139037 0.97860963]
     [1.         0.        ]
     [1.         0.        ]
     [0.02564103 0.97435897]
     [0.6344086  0.3655914 ]]



```python
#정확도 확인
y_pred = bag_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

    0.92


## 랜덤 포레스트(Random Forest)
DecisionTree 여러 개의 동시에 수행시키는 방식 - bagging, bootstrapping

### 배깅과 랜덤 포레스트 비교


```python
#배깅
bag_clf = BaggingClassifier(DecisionTreeClassifier(max_features='sqrt', 
                             max_leaf_nodes=16), n_estimators=500, random_state=42)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)


#랜덤 포레스트 분류기
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, 
                                random_state=42)

rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)

#각각 예측한 결과를 비교해서 비율을 출력
print(np.sum(y_pred == y_pred_rf)/len(y_pred))
print(y_pred)
print(y_pred_rf)
```

    1.0
    [0 0 0 1 1 1 0 0 0 0 1 0 1 1 1 0 0 1 1 0 0 1 0 0 0 0 1 0 1 0 1 1 0 0 1 0 0
     1 1 1 1 1 0 0 0 0 1 0 1 1 1 1 0 0 1 0 1 1 0 1 0 1 1 0 1 0 0 0 0 1 0 0 1 1
     0 0 1 1 0 0 1 1 1 0 1 1 1 0 1 1 1 0 0 0 0 1 0 1 0 1 0 1 1 0 0 0 0 0 1 1 1
     0 0 1 1 0 0 0 0 1 1 1 0 0 0]
    [0 0 0 1 1 1 0 0 0 0 1 0 1 1 1 0 0 1 1 0 0 1 0 0 0 0 1 0 1 0 1 1 0 0 1 0 0
     1 1 1 1 1 0 0 0 0 1 0 1 1 1 1 0 0 1 0 1 1 0 1 0 1 1 0 1 0 0 0 0 1 0 0 1 1
     0 0 1 1 0 0 1 1 1 0 1 1 1 0 1 1 1 0 0 0 0 1 0 1 0 1 0 1 1 0 0 0 0 0 1 1 1
     0 0 1 1 0 0 0 0 1 1 1 0 0 0]


### iris 데이터를 RandomForest를 이용해서 분류하고 특성 중요도를 확인


```python
from sklearn.datasets import load_iris
iris = load_iris()

rnd_clf = RandomForestClassifier(n_estimators=500, random_state = 42)
rnd_clf.fit(iris.data, iris.target)

#print(rnd_clf.feature_importances_)

for name, score in zip(iris.feature_names, rnd_clf.feature_importances_):
    print(name, ':', score)
```

    sepal length (cm) : 0.11249225099876375
    sepal width (cm) : 0.02311928828251033
    petal length (cm) : 0.4410304643639577
    petal width (cm) : 0.4233579963547682


### 이미지의 피처 중요도 시각화


```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(np.uint8)
```


```python
rnd_clf = RandomForestClassifier(n_estimators=500, random_state = 42)
rnd_clf.fit(mnist.data, mnist.target)
```




    RandomForestClassifier(n_estimators=500, random_state=42)




```python
#중요도를 시각화 할 함수
def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.hot, interpolation='nearest')
    plt.axis('off')
```


```python
#이 경우는 외곽 부분은 별로 중요하지 않습니다.
plot_digit(rnd_clf.feature_importances_)

cbar = plt.colorbar(ticks = [rnd_clf.feature_importances_.min(),
                            rnd_clf.feature_importances_.max()])
cbar.ax.set_yticklabels(['Not Important', 'Very Important'])
plt.show()
#이를 적절히 이용하면 이미지를 축소해서 적은 양의 데이터만으로 분류를 할 수 있습니다.

```


    
![png](output_25_0.png)
    


### 사용자 행동 인식에 랜덤 포레스트 적용


```python
feature_name_df = pd.read_csv('./data/UCIHARDataset/features.txt', sep='\s+',
                             header=None, names=['column_index', 'column_name'])
print(feature_name_df)

feature_name_df = pd.read_csv('./data/UCIHARDataset/features.txt', sep='\s+',
                             header=None, names=['column_index', 'column_name'])
print(feature_name_df)

#데이터프레임을 매개변수로 받아서 처리하는 함수
def get_new_feature_name_df(old_feature_name_df):
    #컬럼 이름 별 누적 개수로 데이터 프레임을 생성
    feature_dup_df = pd.DataFrame(data=old_feature_name_df.groupby('column_name').cumcount(),columns=['dup_cnt'])
    #인덱스를 리셋
    feature_dup_df = feature_dup_df.reset_index()
    #원본 피처이름 데이터프레임과 병합
    new_feature_name_df = pd.merge(old_feature_name_df.reset_index(),
                                   feature_dup_df, how='outer')
    
    #데이터의 개수가 1이하이면 그 이름을 그대로 사용하고
    #누적된 데이터의 개수가 2이상이면 그 개수를 이름 뒤에 추가
    new_feature_name_df['column_name'] = new_feature_name_df[['column_name', 'dup_cnt']].apply(lambda x:x[0] + '_' + str(x[1]) if x[1] > 0 else x[0], axis=1)
    
    #index 컬럼 삭제
    new_feature_name_df = new_feature_name_df.drop(['index'], axis=1)
    return new_feature_name_df

# 피처 이름의 중복을 제거하기
new_feature_name_df = get_new_feature_name_df(feature_name_df)
#데이터 프레임에 사용하기 위해 list로 변경하기
feature_name = new_feature_name_df.iloc[:,1].values.tolist()
print(feature_name)

### 훈련 데이터 와 테스트 데이터 만들기
X_train = pd.read_csv('./data/UCIHARDataset/train/X_train.txt',
                      sep='\s+', names=feature_name)
X_test = pd.read_csv('./data/UCIHARDataset/test/X_test.txt',
                      sep='\s+', names=feature_name)

y_train = pd.read_csv('./data/UCIHARDataset/train/y_train.txt',
                      sep='\s+', header=None, names=['action'])
y_test = pd.read_csv('./data/UCIHARDataset/test/y_test.txt',
                      sep='\s+', header=None, names=['action'])
```

         column_index                           column_name
    0               1                     tBodyAcc-mean()-X
    1               2                     tBodyAcc-mean()-Y
    2               3                     tBodyAcc-mean()-Z
    3               4                      tBodyAcc-std()-X
    4               5                      tBodyAcc-std()-Y
    ..            ...                                   ...
    556           557      angle(tBodyGyroMean,gravityMean)
    557           558  angle(tBodyGyroJerkMean,gravityMean)
    558           559                  angle(X,gravityMean)
    559           560                  angle(Y,gravityMean)
    560           561                  angle(Z,gravityMean)
    
    [561 rows x 2 columns]
         column_index                           column_name
    0               1                     tBodyAcc-mean()-X
    1               2                     tBodyAcc-mean()-Y
    2               3                     tBodyAcc-mean()-Z
    3               4                      tBodyAcc-std()-X
    4               5                      tBodyAcc-std()-Y
    ..            ...                                   ...
    556           557      angle(tBodyGyroMean,gravityMean)
    557           558  angle(tBodyGyroJerkMean,gravityMean)
    558           559                  angle(X,gravityMean)
    559           560                  angle(Y,gravityMean)
    560           561                  angle(Z,gravityMean)
    
    [561 rows x 2 columns]
    ['tBodyAcc-mean()-X', 'tBodyAcc-mean()-Y', 'tBodyAcc-mean()-Z', 'tBodyAcc-std()-X', 'tBodyAcc-std()-Y', 'tBodyAcc-std()-Z', 'tBodyAcc-mad()-X', 'tBodyAcc-mad()-Y', 'tBodyAcc-mad()-Z', 'tBodyAcc-max()-X', 'tBodyAcc-max()-Y', 'tBodyAcc-max()-Z', 'tBodyAcc-min()-X', 'tBodyAcc-min()-Y', 'tBodyAcc-min()-Z', 'tBodyAcc-sma()', 'tBodyAcc-energy()-X', 'tBodyAcc-energy()-Y', 'tBodyAcc-energy()-Z', 'tBodyAcc-iqr()-X', 'tBodyAcc-iqr()-Y', 'tBodyAcc-iqr()-Z', 'tBodyAcc-entropy()-X', 'tBodyAcc-entropy()-Y', 'tBodyAcc-entropy()-Z', 'tBodyAcc-arCoeff()-X,1', 'tBodyAcc-arCoeff()-X,2', 'tBodyAcc-arCoeff()-X,3', 'tBodyAcc-arCoeff()-X,4', 'tBodyAcc-arCoeff()-Y,1', 'tBodyAcc-arCoeff()-Y,2', 'tBodyAcc-arCoeff()-Y,3', 'tBodyAcc-arCoeff()-Y,4', 'tBodyAcc-arCoeff()-Z,1', 'tBodyAcc-arCoeff()-Z,2', 'tBodyAcc-arCoeff()-Z,3', 'tBodyAcc-arCoeff()-Z,4', 'tBodyAcc-correlation()-X,Y', 'tBodyAcc-correlation()-X,Z', 'tBodyAcc-correlation()-Y,Z', 'tGravityAcc-mean()-X', 'tGravityAcc-mean()-Y', 'tGravityAcc-mean()-Z', 'tGravityAcc-std()-X', 'tGravityAcc-std()-Y', 'tGravityAcc-std()-Z', 'tGravityAcc-mad()-X', 'tGravityAcc-mad()-Y', 'tGravityAcc-mad()-Z', 'tGravityAcc-max()-X', 'tGravityAcc-max()-Y', 'tGravityAcc-max()-Z', 'tGravityAcc-min()-X', 'tGravityAcc-min()-Y', 'tGravityAcc-min()-Z', 'tGravityAcc-sma()', 'tGravityAcc-energy()-X', 'tGravityAcc-energy()-Y', 'tGravityAcc-energy()-Z', 'tGravityAcc-iqr()-X', 'tGravityAcc-iqr()-Y', 'tGravityAcc-iqr()-Z', 'tGravityAcc-entropy()-X', 'tGravityAcc-entropy()-Y', 'tGravityAcc-entropy()-Z', 'tGravityAcc-arCoeff()-X,1', 'tGravityAcc-arCoeff()-X,2', 'tGravityAcc-arCoeff()-X,3', 'tGravityAcc-arCoeff()-X,4', 'tGravityAcc-arCoeff()-Y,1', 'tGravityAcc-arCoeff()-Y,2', 'tGravityAcc-arCoeff()-Y,3', 'tGravityAcc-arCoeff()-Y,4', 'tGravityAcc-arCoeff()-Z,1', 'tGravityAcc-arCoeff()-Z,2', 'tGravityAcc-arCoeff()-Z,3', 'tGravityAcc-arCoeff()-Z,4', 'tGravityAcc-correlation()-X,Y', 'tGravityAcc-correlation()-X,Z', 'tGravityAcc-correlation()-Y,Z', 'tBodyAccJerk-mean()-X', 'tBodyAccJerk-mean()-Y', 'tBodyAccJerk-mean()-Z', 'tBodyAccJerk-std()-X', 'tBodyAccJerk-std()-Y', 'tBodyAccJerk-std()-Z', 'tBodyAccJerk-mad()-X', 'tBodyAccJerk-mad()-Y', 'tBodyAccJerk-mad()-Z', 'tBodyAccJerk-max()-X', 'tBodyAccJerk-max()-Y', 'tBodyAccJerk-max()-Z', 'tBodyAccJerk-min()-X', 'tBodyAccJerk-min()-Y', 'tBodyAccJerk-min()-Z', 'tBodyAccJerk-sma()', 'tBodyAccJerk-energy()-X', 'tBodyAccJerk-energy()-Y', 'tBodyAccJerk-energy()-Z', 'tBodyAccJerk-iqr()-X', 'tBodyAccJerk-iqr()-Y', 'tBodyAccJerk-iqr()-Z', 'tBodyAccJerk-entropy()-X', 'tBodyAccJerk-entropy()-Y', 'tBodyAccJerk-entropy()-Z', 'tBodyAccJerk-arCoeff()-X,1', 'tBodyAccJerk-arCoeff()-X,2', 'tBodyAccJerk-arCoeff()-X,3', 'tBodyAccJerk-arCoeff()-X,4', 'tBodyAccJerk-arCoeff()-Y,1', 'tBodyAccJerk-arCoeff()-Y,2', 'tBodyAccJerk-arCoeff()-Y,3', 'tBodyAccJerk-arCoeff()-Y,4', 'tBodyAccJerk-arCoeff()-Z,1', 'tBodyAccJerk-arCoeff()-Z,2', 'tBodyAccJerk-arCoeff()-Z,3', 'tBodyAccJerk-arCoeff()-Z,4', 'tBodyAccJerk-correlation()-X,Y', 'tBodyAccJerk-correlation()-X,Z', 'tBodyAccJerk-correlation()-Y,Z', 'tBodyGyro-mean()-X', 'tBodyGyro-mean()-Y', 'tBodyGyro-mean()-Z', 'tBodyGyro-std()-X', 'tBodyGyro-std()-Y', 'tBodyGyro-std()-Z', 'tBodyGyro-mad()-X', 'tBodyGyro-mad()-Y', 'tBodyGyro-mad()-Z', 'tBodyGyro-max()-X', 'tBodyGyro-max()-Y', 'tBodyGyro-max()-Z', 'tBodyGyro-min()-X', 'tBodyGyro-min()-Y', 'tBodyGyro-min()-Z', 'tBodyGyro-sma()', 'tBodyGyro-energy()-X', 'tBodyGyro-energy()-Y', 'tBodyGyro-energy()-Z', 'tBodyGyro-iqr()-X', 'tBodyGyro-iqr()-Y', 'tBodyGyro-iqr()-Z', 'tBodyGyro-entropy()-X', 'tBodyGyro-entropy()-Y', 'tBodyGyro-entropy()-Z', 'tBodyGyro-arCoeff()-X,1', 'tBodyGyro-arCoeff()-X,2', 'tBodyGyro-arCoeff()-X,3', 'tBodyGyro-arCoeff()-X,4', 'tBodyGyro-arCoeff()-Y,1', 'tBodyGyro-arCoeff()-Y,2', 'tBodyGyro-arCoeff()-Y,3', 'tBodyGyro-arCoeff()-Y,4', 'tBodyGyro-arCoeff()-Z,1', 'tBodyGyro-arCoeff()-Z,2', 'tBodyGyro-arCoeff()-Z,3', 'tBodyGyro-arCoeff()-Z,4', 'tBodyGyro-correlation()-X,Y', 'tBodyGyro-correlation()-X,Z', 'tBodyGyro-correlation()-Y,Z', 'tBodyGyroJerk-mean()-X', 'tBodyGyroJerk-mean()-Y', 'tBodyGyroJerk-mean()-Z', 'tBodyGyroJerk-std()-X', 'tBodyGyroJerk-std()-Y', 'tBodyGyroJerk-std()-Z', 'tBodyGyroJerk-mad()-X', 'tBodyGyroJerk-mad()-Y', 'tBodyGyroJerk-mad()-Z', 'tBodyGyroJerk-max()-X', 'tBodyGyroJerk-max()-Y', 'tBodyGyroJerk-max()-Z', 'tBodyGyroJerk-min()-X', 'tBodyGyroJerk-min()-Y', 'tBodyGyroJerk-min()-Z', 'tBodyGyroJerk-sma()', 'tBodyGyroJerk-energy()-X', 'tBodyGyroJerk-energy()-Y', 'tBodyGyroJerk-energy()-Z', 'tBodyGyroJerk-iqr()-X', 'tBodyGyroJerk-iqr()-Y', 'tBodyGyroJerk-iqr()-Z', 'tBodyGyroJerk-entropy()-X', 'tBodyGyroJerk-entropy()-Y', 'tBodyGyroJerk-entropy()-Z', 'tBodyGyroJerk-arCoeff()-X,1', 'tBodyGyroJerk-arCoeff()-X,2', 'tBodyGyroJerk-arCoeff()-X,3', 'tBodyGyroJerk-arCoeff()-X,4', 'tBodyGyroJerk-arCoeff()-Y,1', 'tBodyGyroJerk-arCoeff()-Y,2', 'tBodyGyroJerk-arCoeff()-Y,3', 'tBodyGyroJerk-arCoeff()-Y,4', 'tBodyGyroJerk-arCoeff()-Z,1', 'tBodyGyroJerk-arCoeff()-Z,2', 'tBodyGyroJerk-arCoeff()-Z,3', 'tBodyGyroJerk-arCoeff()-Z,4', 'tBodyGyroJerk-correlation()-X,Y', 'tBodyGyroJerk-correlation()-X,Z', 'tBodyGyroJerk-correlation()-Y,Z', 'tBodyAccMag-mean()', 'tBodyAccMag-std()', 'tBodyAccMag-mad()', 'tBodyAccMag-max()', 'tBodyAccMag-min()', 'tBodyAccMag-sma()', 'tBodyAccMag-energy()', 'tBodyAccMag-iqr()', 'tBodyAccMag-entropy()', 'tBodyAccMag-arCoeff()1', 'tBodyAccMag-arCoeff()2', 'tBodyAccMag-arCoeff()3', 'tBodyAccMag-arCoeff()4', 'tGravityAccMag-mean()', 'tGravityAccMag-std()', 'tGravityAccMag-mad()', 'tGravityAccMag-max()', 'tGravityAccMag-min()', 'tGravityAccMag-sma()', 'tGravityAccMag-energy()', 'tGravityAccMag-iqr()', 'tGravityAccMag-entropy()', 'tGravityAccMag-arCoeff()1', 'tGravityAccMag-arCoeff()2', 'tGravityAccMag-arCoeff()3', 'tGravityAccMag-arCoeff()4', 'tBodyAccJerkMag-mean()', 'tBodyAccJerkMag-std()', 'tBodyAccJerkMag-mad()', 'tBodyAccJerkMag-max()', 'tBodyAccJerkMag-min()', 'tBodyAccJerkMag-sma()', 'tBodyAccJerkMag-energy()', 'tBodyAccJerkMag-iqr()', 'tBodyAccJerkMag-entropy()', 'tBodyAccJerkMag-arCoeff()1', 'tBodyAccJerkMag-arCoeff()2', 'tBodyAccJerkMag-arCoeff()3', 'tBodyAccJerkMag-arCoeff()4', 'tBodyGyroMag-mean()', 'tBodyGyroMag-std()', 'tBodyGyroMag-mad()', 'tBodyGyroMag-max()', 'tBodyGyroMag-min()', 'tBodyGyroMag-sma()', 'tBodyGyroMag-energy()', 'tBodyGyroMag-iqr()', 'tBodyGyroMag-entropy()', 'tBodyGyroMag-arCoeff()1', 'tBodyGyroMag-arCoeff()2', 'tBodyGyroMag-arCoeff()3', 'tBodyGyroMag-arCoeff()4', 'tBodyGyroJerkMag-mean()', 'tBodyGyroJerkMag-std()', 'tBodyGyroJerkMag-mad()', 'tBodyGyroJerkMag-max()', 'tBodyGyroJerkMag-min()', 'tBodyGyroJerkMag-sma()', 'tBodyGyroJerkMag-energy()', 'tBodyGyroJerkMag-iqr()', 'tBodyGyroJerkMag-entropy()', 'tBodyGyroJerkMag-arCoeff()1', 'tBodyGyroJerkMag-arCoeff()2', 'tBodyGyroJerkMag-arCoeff()3', 'tBodyGyroJerkMag-arCoeff()4', 'fBodyAcc-mean()-X', 'fBodyAcc-mean()-Y', 'fBodyAcc-mean()-Z', 'fBodyAcc-std()-X', 'fBodyAcc-std()-Y', 'fBodyAcc-std()-Z', 'fBodyAcc-mad()-X', 'fBodyAcc-mad()-Y', 'fBodyAcc-mad()-Z', 'fBodyAcc-max()-X', 'fBodyAcc-max()-Y', 'fBodyAcc-max()-Z', 'fBodyAcc-min()-X', 'fBodyAcc-min()-Y', 'fBodyAcc-min()-Z', 'fBodyAcc-sma()', 'fBodyAcc-energy()-X', 'fBodyAcc-energy()-Y', 'fBodyAcc-energy()-Z', 'fBodyAcc-iqr()-X', 'fBodyAcc-iqr()-Y', 'fBodyAcc-iqr()-Z', 'fBodyAcc-entropy()-X', 'fBodyAcc-entropy()-Y', 'fBodyAcc-entropy()-Z', 'fBodyAcc-maxInds-X', 'fBodyAcc-maxInds-Y', 'fBodyAcc-maxInds-Z', 'fBodyAcc-meanFreq()-X', 'fBodyAcc-meanFreq()-Y', 'fBodyAcc-meanFreq()-Z', 'fBodyAcc-skewness()-X', 'fBodyAcc-kurtosis()-X', 'fBodyAcc-skewness()-Y', 'fBodyAcc-kurtosis()-Y', 'fBodyAcc-skewness()-Z', 'fBodyAcc-kurtosis()-Z', 'fBodyAcc-bandsEnergy()-1,8', 'fBodyAcc-bandsEnergy()-9,16', 'fBodyAcc-bandsEnergy()-17,24', 'fBodyAcc-bandsEnergy()-25,32', 'fBodyAcc-bandsEnergy()-33,40', 'fBodyAcc-bandsEnergy()-41,48', 'fBodyAcc-bandsEnergy()-49,56', 'fBodyAcc-bandsEnergy()-57,64', 'fBodyAcc-bandsEnergy()-1,16', 'fBodyAcc-bandsEnergy()-17,32', 'fBodyAcc-bandsEnergy()-33,48', 'fBodyAcc-bandsEnergy()-49,64', 'fBodyAcc-bandsEnergy()-1,24', 'fBodyAcc-bandsEnergy()-25,48', 'fBodyAcc-bandsEnergy()-1,8_1', 'fBodyAcc-bandsEnergy()-9,16_1', 'fBodyAcc-bandsEnergy()-17,24_1', 'fBodyAcc-bandsEnergy()-25,32_1', 'fBodyAcc-bandsEnergy()-33,40_1', 'fBodyAcc-bandsEnergy()-41,48_1', 'fBodyAcc-bandsEnergy()-49,56_1', 'fBodyAcc-bandsEnergy()-57,64_1', 'fBodyAcc-bandsEnergy()-1,16_1', 'fBodyAcc-bandsEnergy()-17,32_1', 'fBodyAcc-bandsEnergy()-33,48_1', 'fBodyAcc-bandsEnergy()-49,64_1', 'fBodyAcc-bandsEnergy()-1,24_1', 'fBodyAcc-bandsEnergy()-25,48_1', 'fBodyAcc-bandsEnergy()-1,8_2', 'fBodyAcc-bandsEnergy()-9,16_2', 'fBodyAcc-bandsEnergy()-17,24_2', 'fBodyAcc-bandsEnergy()-25,32_2', 'fBodyAcc-bandsEnergy()-33,40_2', 'fBodyAcc-bandsEnergy()-41,48_2', 'fBodyAcc-bandsEnergy()-49,56_2', 'fBodyAcc-bandsEnergy()-57,64_2', 'fBodyAcc-bandsEnergy()-1,16_2', 'fBodyAcc-bandsEnergy()-17,32_2', 'fBodyAcc-bandsEnergy()-33,48_2', 'fBodyAcc-bandsEnergy()-49,64_2', 'fBodyAcc-bandsEnergy()-1,24_2', 'fBodyAcc-bandsEnergy()-25,48_2', 'fBodyAccJerk-mean()-X', 'fBodyAccJerk-mean()-Y', 'fBodyAccJerk-mean()-Z', 'fBodyAccJerk-std()-X', 'fBodyAccJerk-std()-Y', 'fBodyAccJerk-std()-Z', 'fBodyAccJerk-mad()-X', 'fBodyAccJerk-mad()-Y', 'fBodyAccJerk-mad()-Z', 'fBodyAccJerk-max()-X', 'fBodyAccJerk-max()-Y', 'fBodyAccJerk-max()-Z', 'fBodyAccJerk-min()-X', 'fBodyAccJerk-min()-Y', 'fBodyAccJerk-min()-Z', 'fBodyAccJerk-sma()', 'fBodyAccJerk-energy()-X', 'fBodyAccJerk-energy()-Y', 'fBodyAccJerk-energy()-Z', 'fBodyAccJerk-iqr()-X', 'fBodyAccJerk-iqr()-Y', 'fBodyAccJerk-iqr()-Z', 'fBodyAccJerk-entropy()-X', 'fBodyAccJerk-entropy()-Y', 'fBodyAccJerk-entropy()-Z', 'fBodyAccJerk-maxInds-X', 'fBodyAccJerk-maxInds-Y', 'fBodyAccJerk-maxInds-Z', 'fBodyAccJerk-meanFreq()-X', 'fBodyAccJerk-meanFreq()-Y', 'fBodyAccJerk-meanFreq()-Z', 'fBodyAccJerk-skewness()-X', 'fBodyAccJerk-kurtosis()-X', 'fBodyAccJerk-skewness()-Y', 'fBodyAccJerk-kurtosis()-Y', 'fBodyAccJerk-skewness()-Z', 'fBodyAccJerk-kurtosis()-Z', 'fBodyAccJerk-bandsEnergy()-1,8', 'fBodyAccJerk-bandsEnergy()-9,16', 'fBodyAccJerk-bandsEnergy()-17,24', 'fBodyAccJerk-bandsEnergy()-25,32', 'fBodyAccJerk-bandsEnergy()-33,40', 'fBodyAccJerk-bandsEnergy()-41,48', 'fBodyAccJerk-bandsEnergy()-49,56', 'fBodyAccJerk-bandsEnergy()-57,64', 'fBodyAccJerk-bandsEnergy()-1,16', 'fBodyAccJerk-bandsEnergy()-17,32', 'fBodyAccJerk-bandsEnergy()-33,48', 'fBodyAccJerk-bandsEnergy()-49,64', 'fBodyAccJerk-bandsEnergy()-1,24', 'fBodyAccJerk-bandsEnergy()-25,48', 'fBodyAccJerk-bandsEnergy()-1,8_1', 'fBodyAccJerk-bandsEnergy()-9,16_1', 'fBodyAccJerk-bandsEnergy()-17,24_1', 'fBodyAccJerk-bandsEnergy()-25,32_1', 'fBodyAccJerk-bandsEnergy()-33,40_1', 'fBodyAccJerk-bandsEnergy()-41,48_1', 'fBodyAccJerk-bandsEnergy()-49,56_1', 'fBodyAccJerk-bandsEnergy()-57,64_1', 'fBodyAccJerk-bandsEnergy()-1,16_1', 'fBodyAccJerk-bandsEnergy()-17,32_1', 'fBodyAccJerk-bandsEnergy()-33,48_1', 'fBodyAccJerk-bandsEnergy()-49,64_1', 'fBodyAccJerk-bandsEnergy()-1,24_1', 'fBodyAccJerk-bandsEnergy()-25,48_1', 'fBodyAccJerk-bandsEnergy()-1,8_2', 'fBodyAccJerk-bandsEnergy()-9,16_2', 'fBodyAccJerk-bandsEnergy()-17,24_2', 'fBodyAccJerk-bandsEnergy()-25,32_2', 'fBodyAccJerk-bandsEnergy()-33,40_2', 'fBodyAccJerk-bandsEnergy()-41,48_2', 'fBodyAccJerk-bandsEnergy()-49,56_2', 'fBodyAccJerk-bandsEnergy()-57,64_2', 'fBodyAccJerk-bandsEnergy()-1,16_2', 'fBodyAccJerk-bandsEnergy()-17,32_2', 'fBodyAccJerk-bandsEnergy()-33,48_2', 'fBodyAccJerk-bandsEnergy()-49,64_2', 'fBodyAccJerk-bandsEnergy()-1,24_2', 'fBodyAccJerk-bandsEnergy()-25,48_2', 'fBodyGyro-mean()-X', 'fBodyGyro-mean()-Y', 'fBodyGyro-mean()-Z', 'fBodyGyro-std()-X', 'fBodyGyro-std()-Y', 'fBodyGyro-std()-Z', 'fBodyGyro-mad()-X', 'fBodyGyro-mad()-Y', 'fBodyGyro-mad()-Z', 'fBodyGyro-max()-X', 'fBodyGyro-max()-Y', 'fBodyGyro-max()-Z', 'fBodyGyro-min()-X', 'fBodyGyro-min()-Y', 'fBodyGyro-min()-Z', 'fBodyGyro-sma()', 'fBodyGyro-energy()-X', 'fBodyGyro-energy()-Y', 'fBodyGyro-energy()-Z', 'fBodyGyro-iqr()-X', 'fBodyGyro-iqr()-Y', 'fBodyGyro-iqr()-Z', 'fBodyGyro-entropy()-X', 'fBodyGyro-entropy()-Y', 'fBodyGyro-entropy()-Z', 'fBodyGyro-maxInds-X', 'fBodyGyro-maxInds-Y', 'fBodyGyro-maxInds-Z', 'fBodyGyro-meanFreq()-X', 'fBodyGyro-meanFreq()-Y', 'fBodyGyro-meanFreq()-Z', 'fBodyGyro-skewness()-X', 'fBodyGyro-kurtosis()-X', 'fBodyGyro-skewness()-Y', 'fBodyGyro-kurtosis()-Y', 'fBodyGyro-skewness()-Z', 'fBodyGyro-kurtosis()-Z', 'fBodyGyro-bandsEnergy()-1,8', 'fBodyGyro-bandsEnergy()-9,16', 'fBodyGyro-bandsEnergy()-17,24', 'fBodyGyro-bandsEnergy()-25,32', 'fBodyGyro-bandsEnergy()-33,40', 'fBodyGyro-bandsEnergy()-41,48', 'fBodyGyro-bandsEnergy()-49,56', 'fBodyGyro-bandsEnergy()-57,64', 'fBodyGyro-bandsEnergy()-1,16', 'fBodyGyro-bandsEnergy()-17,32', 'fBodyGyro-bandsEnergy()-33,48', 'fBodyGyro-bandsEnergy()-49,64', 'fBodyGyro-bandsEnergy()-1,24', 'fBodyGyro-bandsEnergy()-25,48', 'fBodyGyro-bandsEnergy()-1,8_1', 'fBodyGyro-bandsEnergy()-9,16_1', 'fBodyGyro-bandsEnergy()-17,24_1', 'fBodyGyro-bandsEnergy()-25,32_1', 'fBodyGyro-bandsEnergy()-33,40_1', 'fBodyGyro-bandsEnergy()-41,48_1', 'fBodyGyro-bandsEnergy()-49,56_1', 'fBodyGyro-bandsEnergy()-57,64_1', 'fBodyGyro-bandsEnergy()-1,16_1', 'fBodyGyro-bandsEnergy()-17,32_1', 'fBodyGyro-bandsEnergy()-33,48_1', 'fBodyGyro-bandsEnergy()-49,64_1', 'fBodyGyro-bandsEnergy()-1,24_1', 'fBodyGyro-bandsEnergy()-25,48_1', 'fBodyGyro-bandsEnergy()-1,8_2', 'fBodyGyro-bandsEnergy()-9,16_2', 'fBodyGyro-bandsEnergy()-17,24_2', 'fBodyGyro-bandsEnergy()-25,32_2', 'fBodyGyro-bandsEnergy()-33,40_2', 'fBodyGyro-bandsEnergy()-41,48_2', 'fBodyGyro-bandsEnergy()-49,56_2', 'fBodyGyro-bandsEnergy()-57,64_2', 'fBodyGyro-bandsEnergy()-1,16_2', 'fBodyGyro-bandsEnergy()-17,32_2', 'fBodyGyro-bandsEnergy()-33,48_2', 'fBodyGyro-bandsEnergy()-49,64_2', 'fBodyGyro-bandsEnergy()-1,24_2', 'fBodyGyro-bandsEnergy()-25,48_2', 'fBodyAccMag-mean()', 'fBodyAccMag-std()', 'fBodyAccMag-mad()', 'fBodyAccMag-max()', 'fBodyAccMag-min()', 'fBodyAccMag-sma()', 'fBodyAccMag-energy()', 'fBodyAccMag-iqr()', 'fBodyAccMag-entropy()', 'fBodyAccMag-maxInds', 'fBodyAccMag-meanFreq()', 'fBodyAccMag-skewness()', 'fBodyAccMag-kurtosis()', 'fBodyBodyAccJerkMag-mean()', 'fBodyBodyAccJerkMag-std()', 'fBodyBodyAccJerkMag-mad()', 'fBodyBodyAccJerkMag-max()', 'fBodyBodyAccJerkMag-min()', 'fBodyBodyAccJerkMag-sma()', 'fBodyBodyAccJerkMag-energy()', 'fBodyBodyAccJerkMag-iqr()', 'fBodyBodyAccJerkMag-entropy()', 'fBodyBodyAccJerkMag-maxInds', 'fBodyBodyAccJerkMag-meanFreq()', 'fBodyBodyAccJerkMag-skewness()', 'fBodyBodyAccJerkMag-kurtosis()', 'fBodyBodyGyroMag-mean()', 'fBodyBodyGyroMag-std()', 'fBodyBodyGyroMag-mad()', 'fBodyBodyGyroMag-max()', 'fBodyBodyGyroMag-min()', 'fBodyBodyGyroMag-sma()', 'fBodyBodyGyroMag-energy()', 'fBodyBodyGyroMag-iqr()', 'fBodyBodyGyroMag-entropy()', 'fBodyBodyGyroMag-maxInds', 'fBodyBodyGyroMag-meanFreq()', 'fBodyBodyGyroMag-skewness()', 'fBodyBodyGyroMag-kurtosis()', 'fBodyBodyGyroJerkMag-mean()', 'fBodyBodyGyroJerkMag-std()', 'fBodyBodyGyroJerkMag-mad()', 'fBodyBodyGyroJerkMag-max()', 'fBodyBodyGyroJerkMag-min()', 'fBodyBodyGyroJerkMag-sma()', 'fBodyBodyGyroJerkMag-energy()', 'fBodyBodyGyroJerkMag-iqr()', 'fBodyBodyGyroJerkMag-entropy()', 'fBodyBodyGyroJerkMag-maxInds', 'fBodyBodyGyroJerkMag-meanFreq()', 'fBodyBodyGyroJerkMag-skewness()', 'fBodyBodyGyroJerkMag-kurtosis()', 'angle(tBodyAccMean,gravity)', 'angle(tBodyAccJerkMean),gravityMean)', 'angle(tBodyGyroMean,gravityMean)', 'angle(tBodyGyroJerkMean,gravityMean)', 'angle(X,gravityMean)', 'angle(Y,gravityMean)', 'angle(Z,gravityMean)']



```python
from sklearn.metrics import accuracy_score

rf_clf = RandomForestClassifier(n_estimators=500, random_state = 42)
rf_clf.fit(X_train, y_train)
pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print("정확도:", accuracy)
```

    정확도: 0.9267051238547676


### Gradient Boosting
결정 트리를 여러 개 만들어서 사용하는 앙상블 기법으로 트리를 발전시킬 때 경사 하강법을 이용


```python
from sklearn.ensemble import GradientBoostingClassifier
import time

#수행 시간을 측정하기 위해서는 알고리즘이 시작하기 전에 현재 시간을 기록
start_time = time.time()

gb_clf = GradientBoostingClassifier(random_state=42)
gb_clf.fit(X_train, y_train)
gb_pred = gb_clf.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)

print("정확도:", gb_accuracy)
#현재 시간에서 이전에 기록해둔 시간을 빼면 알고리즘이 수행되는데 걸린 시간 측정이 가능
print("수행 시간:", (time.time() - start_time))
```

    정확도: 0.9385816084153377
    수행 시간: 490.0083689689636


### 하이퍼파라미터 튜닝


```python
from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators':[100, 500],
    'learning_rate': [0.05, 1]
}

grid_cv = GridSearchCV(gb_clf, param_grid = params, cv=2, verbose=1)
grid_cv.fit(X_train, y_train)

print("최적의 하이퍼 파라미터:", grid_cv.best_params_)
print("정확도:", grid_cv.best_score_)
```

    Fitting 2 folds for each of 4 candidates, totalling 8 fits



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-54-9631960aa1a6> in <module>
          7 
          8 grid_cv = GridSearchCV(gb_clf, param_grid = params, cv=2, verbose=1)
    ----> 9 grid_cv.fit(X_train, y_train)
         10 
         11 print("최적의 하이퍼 파라미터:", grid_cv.best_params_)


    ~\anaconda3\lib\site-packages\sklearn\utils\validation.py in inner_f(*args, **kwargs)
         61             extra_args = len(args) - len(all_args)
         62             if extra_args <= 0:
    ---> 63                 return f(*args, **kwargs)
         64 
         65             # extra_args > 0


    ~\anaconda3\lib\site-packages\sklearn\model_selection\_search.py in fit(self, X, y, groups, **fit_params)
        839                 return results
        840 
    --> 841             self._run_search(evaluate_candidates)
        842 
        843             # multimetric is determined here because in the case of a callable


    ~\anaconda3\lib\site-packages\sklearn\model_selection\_search.py in _run_search(self, evaluate_candidates)
       1286     def _run_search(self, evaluate_candidates):
       1287         """Search all candidates in param_grid"""
    -> 1288         evaluate_candidates(ParameterGrid(self.param_grid))
       1289 
       1290 


    ~\anaconda3\lib\site-packages\sklearn\model_selection\_search.py in evaluate_candidates(candidate_params, cv, more_results)
        793                               n_splits, n_candidates, n_candidates * n_splits))
        794 
    --> 795                 out = parallel(delayed(_fit_and_score)(clone(base_estimator),
        796                                                        X, y,
        797                                                        train=train, test=test,


    ~\anaconda3\lib\site-packages\joblib\parallel.py in __call__(self, iterable)
       1042                 self._iterating = self._original_iterator is not None
       1043 
    -> 1044             while self.dispatch_one_batch(iterator):
       1045                 pass
       1046 


    ~\anaconda3\lib\site-packages\joblib\parallel.py in dispatch_one_batch(self, iterator)
        857                 return False
        858             else:
    --> 859                 self._dispatch(tasks)
        860                 return True
        861 


    ~\anaconda3\lib\site-packages\joblib\parallel.py in _dispatch(self, batch)
        775         with self._lock:
        776             job_idx = len(self._jobs)
    --> 777             job = self._backend.apply_async(batch, callback=cb)
        778             # A job can complete so quickly than its callback is
        779             # called before we get here, causing self._jobs to


    ~\anaconda3\lib\site-packages\joblib\_parallel_backends.py in apply_async(self, func, callback)
        206     def apply_async(self, func, callback=None):
        207         """Schedule a func to be run"""
    --> 208         result = ImmediateResult(func)
        209         if callback:
        210             callback(result)


    ~\anaconda3\lib\site-packages\joblib\_parallel_backends.py in __init__(self, batch)
        570         # Don't delay the application, to avoid keeping the input
        571         # arguments in memory
    --> 572         self.results = batch()
        573 
        574     def get(self):


    ~\anaconda3\lib\site-packages\joblib\parallel.py in __call__(self)
        260         # change the default number of processes to -1
        261         with parallel_backend(self._backend, n_jobs=self._n_jobs):
    --> 262             return [func(*args, **kwargs)
        263                     for func, args, kwargs in self.items]
        264 


    ~\anaconda3\lib\site-packages\joblib\parallel.py in <listcomp>(.0)
        260         # change the default number of processes to -1
        261         with parallel_backend(self._backend, n_jobs=self._n_jobs):
    --> 262             return [func(*args, **kwargs)
        263                     for func, args, kwargs in self.items]
        264 


    ~\anaconda3\lib\site-packages\sklearn\utils\fixes.py in __call__(self, *args, **kwargs)
        220     def __call__(self, *args, **kwargs):
        221         with config_context(**self.config):
    --> 222             return self.function(*args, **kwargs)
    

    ~\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py in _fit_and_score(estimator, X, y, scorer, train, test, verbose, parameters, fit_params, return_train_score, return_parameters, return_n_test_samples, return_times, return_estimator, split_progress, candidate_progress, error_score)
        591             estimator.fit(X_train, **fit_params)
        592         else:
    --> 593             estimator.fit(X_train, y_train, **fit_params)
        594 
        595     except Exception as e:


    ~\anaconda3\lib\site-packages\sklearn\ensemble\_gb.py in fit(self, X, y, sample_weight, monitor)
        502 
        503         # fit the boosting stages
    --> 504         n_stages = self._fit_stages(
        505             X, y, raw_predictions, sample_weight, self._rng, X_val, y_val,
        506             sample_weight_val, begin_at_stage, monitor)


    ~\anaconda3\lib\site-packages\sklearn\ensemble\_gb.py in _fit_stages(self, X, y, raw_predictions, sample_weight, random_state, X_val, y_val, sample_weight_val, begin_at_stage, monitor)
        559 
        560             # fit next stage of trees
    --> 561             raw_predictions = self._fit_stage(
        562                 i, X, y, raw_predictions, sample_weight, sample_mask,
        563                 random_state, X_csc, X_csr)


    ~\anaconda3\lib\site-packages\sklearn\ensemble\_gb.py in _fit_stage(self, i, X, y, raw_predictions, sample_weight, sample_mask, random_state, X_csc, X_csr)
        212 
        213             X = X_csr if X_csr is not None else X
    --> 214             tree.fit(X, residual, sample_weight=sample_weight,
        215                      check_input=False)
        216 


    ~\anaconda3\lib\site-packages\sklearn\tree\_classes.py in fit(self, X, y, sample_weight, check_input, X_idx_sorted)
       1245         """
       1246 
    -> 1247         super().fit(
       1248             X, y,
       1249             sample_weight=sample_weight,


    ~\anaconda3\lib\site-packages\sklearn\tree\_classes.py in fit(self, X, y, sample_weight, check_input, X_idx_sorted)
        387                                            min_impurity_split)
        388 
    --> 389         builder.build(self.tree_, X, y, sample_weight)
        390 
        391         if self.n_outputs_ == 1 and is_classifier(self):


    KeyboardInterrupt: 


### XGB 를 이용한 위스콘신 유방암 데이터 분류


```python
#데이터 가져오기
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()
X_features = dataset.data
y_label = dataset.target

cancer_df = pd.DataFrame(data = X_features, columns = dataset.feature_names)
cancer_df['target'] = y_label

print(cancer_df.head())
```

       mean radius  mean texture  mean perimeter  mean area  mean smoothness  \
    0        17.99         10.38          122.80     1001.0          0.11840   
    1        20.57         17.77          132.90     1326.0          0.08474   
    2        19.69         21.25          130.00     1203.0          0.10960   
    3        11.42         20.38           77.58      386.1          0.14250   
    4        20.29         14.34          135.10     1297.0          0.10030   
    
       mean compactness  mean concavity  mean concave points  mean symmetry  \
    0           0.27760          0.3001              0.14710         0.2419   
    1           0.07864          0.0869              0.07017         0.1812   
    2           0.15990          0.1974              0.12790         0.2069   
    3           0.28390          0.2414              0.10520         0.2597   
    4           0.13280          0.1980              0.10430         0.1809   
    
       mean fractal dimension  ...  worst texture  worst perimeter  worst area  \
    0                 0.07871  ...          17.33           184.60      2019.0   
    1                 0.05667  ...          23.41           158.80      1956.0   
    2                 0.05999  ...          25.53           152.50      1709.0   
    3                 0.09744  ...          26.50            98.87       567.7   
    4                 0.05883  ...          16.67           152.20      1575.0   
    
       worst smoothness  worst compactness  worst concavity  worst concave points  \
    0            0.1622             0.6656           0.7119                0.2654   
    1            0.1238             0.1866           0.2416                0.1860   
    2            0.1444             0.4245           0.4504                0.2430   
    3            0.2098             0.8663           0.6869                0.2575   
    4            0.1374             0.2050           0.4000                0.1625   
    
       worst symmetry  worst fractal dimension  target  
    0          0.4601                  0.11890       0  
    1          0.2750                  0.08902       0  
    2          0.3613                  0.08758       0  
    3          0.6638                  0.17300       0  
    4          0.2364                  0.07678       0  
    
    [5 rows x 31 columns]



```python
#레이블 확인
print(dataset.target_names)
print(cancer_df.target.value_counts())

#1번이 악성 종양이고 0번이 양성 종양
```

    ['malignant' 'benign']
    1    357
    0    212
    Name: target, dtype: int64



```python
#훈련 데이터 와 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_features, y_label, 
                                                    test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
```

    (455, 30)
    (114, 30)



```python
#없으면 설치
#!pip install xgboost
import xgboost as xgb
from xgboost import plot_importance
```


```python
#xgboost 에서 사용하기 위한 데이터를 생성 - DMatrix 를 사용
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test, label=y_test)
```


```python
#하이퍼 파라미터를 변수로 생성
params = {'max_depth': 3, 'eta':0.1, 'objective': 'binary:logistic', 
         'eval_metric':'logloss'}
num_rounds = 400
```


```python
wlist = [(dtrain, 'train'), (dtest, 'eval')]
xgb_model = xgb.train(params = params, dtrain=dtrain, 
                      num_boost_round = num_rounds, early_stopping_rounds=100,
                     evals=wlist)
```

    [0]	train-logloss:0.60961	eval-logloss:0.61443
    [1]	train-logloss:0.54076	eval-logloss:0.54721
    [2]	train-logloss:0.48407	eval-logloss:0.49559
    [3]	train-logloss:0.43324	eval-logloss:0.44659
    [4]	train-logloss:0.38985	eval-logloss:0.40682
    [5]	train-logloss:0.35213	eval-logloss:0.37082
    [6]	train-logloss:0.31949	eval-logloss:0.34032
    [7]	train-logloss:0.29139	eval-logloss:0.31559
    [8]	train-logloss:0.26606	eval-logloss:0.29181
    [9]	train-logloss:0.24392	eval-logloss:0.27323
    [10]	train-logloss:0.22408	eval-logloss:0.25490
    [11]	train-logloss:0.20697	eval-logloss:0.24217
    [12]	train-logloss:0.19128	eval-logloss:0.22577
    [13]	train-logloss:0.17705	eval-logloss:0.21283
    [14]	train-logloss:0.16451	eval-logloss:0.19947
    [15]	train-logloss:0.15319	eval-logloss:0.19169
    [16]	train-logloss:0.14312	eval-logloss:0.18069
    [17]	train-logloss:0.13410	eval-logloss:0.17432
    [18]	train-logloss:0.12522	eval-logloss:0.16573
    [19]	train-logloss:0.11687	eval-logloss:0.15517
    [20]	train-logloss:0.10980	eval-logloss:0.15131
    [21]	train-logloss:0.10320	eval-logloss:0.14297
    [22]	train-logloss:0.09739	eval-logloss:0.14038
    [23]	train-logloss:0.09170	eval-logloss:0.13659
    [24]	train-logloss:0.08657	eval-logloss:0.13153
    [25]	train-logloss:0.08167	eval-logloss:0.12885
    [26]	train-logloss:0.07742	eval-logloss:0.12498
    [27]	train-logloss:0.07337	eval-logloss:0.12312
    [28]	train-logloss:0.06981	eval-logloss:0.11902
    [29]	train-logloss:0.06634	eval-logloss:0.11769
    [30]	train-logloss:0.06332	eval-logloss:0.11719
    [31]	train-logloss:0.06013	eval-logloss:0.11469
    [32]	train-logloss:0.05760	eval-logloss:0.11483
    [33]	train-logloss:0.05490	eval-logloss:0.11278
    [34]	train-logloss:0.05249	eval-logloss:0.11100
    [35]	train-logloss:0.04981	eval-logloss:0.11050
    [36]	train-logloss:0.04789	eval-logloss:0.11044
    [37]	train-logloss:0.04577	eval-logloss:0.10774
    [38]	train-logloss:0.04361	eval-logloss:0.10780
    [39]	train-logloss:0.04212	eval-logloss:0.10597
    [40]	train-logloss:0.04050	eval-logloss:0.10549
    [41]	train-logloss:0.03885	eval-logloss:0.10659
    [42]	train-logloss:0.03726	eval-logloss:0.10633
    [43]	train-logloss:0.03566	eval-logloss:0.10575
    [44]	train-logloss:0.03442	eval-logloss:0.10479
    [45]	train-logloss:0.03314	eval-logloss:0.10498
    [46]	train-logloss:0.03222	eval-logloss:0.10376
    [47]	train-logloss:0.03105	eval-logloss:0.10413
    [48]	train-logloss:0.02994	eval-logloss:0.10416
    [49]	train-logloss:0.02903	eval-logloss:0.10393
    [50]	train-logloss:0.02820	eval-logloss:0.10280
    [51]	train-logloss:0.02727	eval-logloss:0.10118
    [52]	train-logloss:0.02638	eval-logloss:0.10162
    [53]	train-logloss:0.02567	eval-logloss:0.10110
    [54]	train-logloss:0.02491	eval-logloss:0.10166
    [55]	train-logloss:0.02414	eval-logloss:0.10151
    [56]	train-logloss:0.02350	eval-logloss:0.10074
    [57]	train-logloss:0.02298	eval-logloss:0.10201
    [58]	train-logloss:0.02233	eval-logloss:0.10194
    [59]	train-logloss:0.02177	eval-logloss:0.10158
    [60]	train-logloss:0.02120	eval-logloss:0.10135
    [61]	train-logloss:0.02065	eval-logloss:0.10024
    [62]	train-logloss:0.02019	eval-logloss:0.09891
    [63]	train-logloss:0.01973	eval-logloss:0.09916
    [64]	train-logloss:0.01924	eval-logloss:0.09932
    [65]	train-logloss:0.01888	eval-logloss:0.09914
    [66]	train-logloss:0.01844	eval-logloss:0.10027
    [67]	train-logloss:0.01805	eval-logloss:0.10109
    [68]	train-logloss:0.01771	eval-logloss:0.10167
    [69]	train-logloss:0.01730	eval-logloss:0.10187
    [70]	train-logloss:0.01700	eval-logloss:0.10171
    [71]	train-logloss:0.01671	eval-logloss:0.10082
    [72]	train-logloss:0.01637	eval-logloss:0.10062
    [73]	train-logloss:0.01602	eval-logloss:0.09954
    [74]	train-logloss:0.01570	eval-logloss:0.09882
    [75]	train-logloss:0.01534	eval-logloss:0.09954
    [76]	train-logloss:0.01504	eval-logloss:0.10005
    [77]	train-logloss:0.01470	eval-logloss:0.10143
    [78]	train-logloss:0.01453	eval-logloss:0.10187
    [79]	train-logloss:0.01435	eval-logloss:0.10158
    [80]	train-logloss:0.01412	eval-logloss:0.10115
    [81]	train-logloss:0.01394	eval-logloss:0.10017
    [82]	train-logloss:0.01367	eval-logloss:0.10086
    [83]	train-logloss:0.01349	eval-logloss:0.10086
    [84]	train-logloss:0.01321	eval-logloss:0.10222
    [85]	train-logloss:0.01302	eval-logloss:0.10156
    [86]	train-logloss:0.01281	eval-logloss:0.10266
    [87]	train-logloss:0.01268	eval-logloss:0.10308
    [88]	train-logloss:0.01256	eval-logloss:0.10235
    [89]	train-logloss:0.01243	eval-logloss:0.10207
    [90]	train-logloss:0.01230	eval-logloss:0.10142
    [91]	train-logloss:0.01215	eval-logloss:0.10146
    [92]	train-logloss:0.01199	eval-logloss:0.10100
    [93]	train-logloss:0.01187	eval-logloss:0.10126
    [94]	train-logloss:0.01171	eval-logloss:0.10212
    [95]	train-logloss:0.01161	eval-logloss:0.10146
    [96]	train-logloss:0.01149	eval-logloss:0.10140
    [97]	train-logloss:0.01139	eval-logloss:0.10092
    [98]	train-logloss:0.01128	eval-logloss:0.10118
    [99]	train-logloss:0.01119	eval-logloss:0.10055
    [100]	train-logloss:0.01108	eval-logloss:0.10097
    [101]	train-logloss:0.01098	eval-logloss:0.10075
    [102]	train-logloss:0.01090	eval-logloss:0.10061
    [103]	train-logloss:0.01079	eval-logloss:0.10059
    [104]	train-logloss:0.01065	eval-logloss:0.10141
    [105]	train-logloss:0.01054	eval-logloss:0.10146
    [106]	train-logloss:0.01046	eval-logloss:0.10101
    [107]	train-logloss:0.01040	eval-logloss:0.10102
    [108]	train-logloss:0.01032	eval-logloss:0.10086
    [109]	train-logloss:0.01022	eval-logloss:0.10091
    [110]	train-logloss:0.01009	eval-logloss:0.10111
    [111]	train-logloss:0.01000	eval-logloss:0.10136
    [112]	train-logloss:0.00989	eval-logloss:0.10216
    [113]	train-logloss:0.00981	eval-logloss:0.10155
    [114]	train-logloss:0.00974	eval-logloss:0.10113
    [115]	train-logloss:0.00966	eval-logloss:0.10114
    [116]	train-logloss:0.00957	eval-logloss:0.10119
    [117]	train-logloss:0.00954	eval-logloss:0.10109
    [118]	train-logloss:0.00946	eval-logloss:0.10053
    [119]	train-logloss:0.00935	eval-logloss:0.10016
    [120]	train-logloss:0.00931	eval-logloss:0.10015
    [121]	train-logloss:0.00922	eval-logloss:0.10021
    [122]	train-logloss:0.00913	eval-logloss:0.10025
    [123]	train-logloss:0.00907	eval-logloss:0.10012
    [124]	train-logloss:0.00900	eval-logloss:0.09960
    [125]	train-logloss:0.00897	eval-logloss:0.09952
    [126]	train-logloss:0.00888	eval-logloss:0.09983
    [127]	train-logloss:0.00880	eval-logloss:0.09989
    [128]	train-logloss:0.00872	eval-logloss:0.09994
    [129]	train-logloss:0.00869	eval-logloss:0.09986
    [130]	train-logloss:0.00864	eval-logloss:0.09939
    [131]	train-logloss:0.00861	eval-logloss:0.09932
    [132]	train-logloss:0.00854	eval-logloss:0.09938
    [133]	train-logloss:0.00847	eval-logloss:0.09982
    [134]	train-logloss:0.00840	eval-logloss:0.09988
    [135]	train-logloss:0.00835	eval-logloss:0.09976
    [136]	train-logloss:0.00830	eval-logloss:0.09929
    [137]	train-logloss:0.00827	eval-logloss:0.09899
    [138]	train-logloss:0.00822	eval-logloss:0.09917
    [139]	train-logloss:0.00820	eval-logloss:0.09910
    [140]	train-logloss:0.00814	eval-logloss:0.09917
    [141]	train-logloss:0.00807	eval-logloss:0.09923
    [142]	train-logloss:0.00803	eval-logloss:0.09918
    [143]	train-logloss:0.00801	eval-logloss:0.09889
    [144]	train-logloss:0.00798	eval-logloss:0.09905
    [145]	train-logloss:0.00794	eval-logloss:0.09894
    [146]	train-logloss:0.00788	eval-logloss:0.09934
    [147]	train-logloss:0.00785	eval-logloss:0.09892
    [148]	train-logloss:0.00782	eval-logloss:0.09894
    [149]	train-logloss:0.00780	eval-logloss:0.09867
    [150]	train-logloss:0.00773	eval-logloss:0.09900
    [151]	train-logloss:0.00769	eval-logloss:0.09896
    [152]	train-logloss:0.00764	eval-logloss:0.09902
    [153]	train-logloss:0.00762	eval-logloss:0.09903
    [154]	train-logloss:0.00760	eval-logloss:0.09892
    [155]	train-logloss:0.00753	eval-logloss:0.09960
    [156]	train-logloss:0.00751	eval-logloss:0.09934
    [157]	train-logloss:0.00749	eval-logloss:0.09949
    [158]	train-logloss:0.00746	eval-logloss:0.09919
    [159]	train-logloss:0.00741	eval-logloss:0.09909
    [160]	train-logloss:0.00739	eval-logloss:0.09904
    [161]	train-logloss:0.00737	eval-logloss:0.09879
    [162]	train-logloss:0.00734	eval-logloss:0.09851
    [163]	train-logloss:0.00732	eval-logloss:0.09860
    [164]	train-logloss:0.00730	eval-logloss:0.09849
    [165]	train-logloss:0.00728	eval-logloss:0.09851
    [166]	train-logloss:0.00726	eval-logloss:0.09859
    [167]	train-logloss:0.00725	eval-logloss:0.09815
    [168]	train-logloss:0.00721	eval-logloss:0.09793
    [169]	train-logloss:0.00720	eval-logloss:0.09809
    [170]	train-logloss:0.00718	eval-logloss:0.09798
    [171]	train-logloss:0.00716	eval-logloss:0.09767
    [172]	train-logloss:0.00714	eval-logloss:0.09763
    [173]	train-logloss:0.00712	eval-logloss:0.09752
    [174]	train-logloss:0.00710	eval-logloss:0.09760
    [175]	train-logloss:0.00709	eval-logloss:0.09749
    [176]	train-logloss:0.00707	eval-logloss:0.09726
    [177]	train-logloss:0.00705	eval-logloss:0.09741
    [178]	train-logloss:0.00703	eval-logloss:0.09730
    [179]	train-logloss:0.00702	eval-logloss:0.09689
    [180]	train-logloss:0.00700	eval-logloss:0.09697
    [181]	train-logloss:0.00698	eval-logloss:0.09687
    [182]	train-logloss:0.00696	eval-logloss:0.09688
    [183]	train-logloss:0.00695	eval-logloss:0.09696
    [184]	train-logloss:0.00693	eval-logloss:0.09697
    [185]	train-logloss:0.00692	eval-logloss:0.09656
    [186]	train-logloss:0.00690	eval-logloss:0.09668
    [187]	train-logloss:0.00688	eval-logloss:0.09682
    [188]	train-logloss:0.00687	eval-logloss:0.09673
    [189]	train-logloss:0.00685	eval-logloss:0.09669
    [190]	train-logloss:0.00683	eval-logloss:0.09631
    [191]	train-logloss:0.00682	eval-logloss:0.09633
    [192]	train-logloss:0.00680	eval-logloss:0.09647
    [193]	train-logloss:0.00679	eval-logloss:0.09649
    [194]	train-logloss:0.00677	eval-logloss:0.09627
    [195]	train-logloss:0.00676	eval-logloss:0.09634
    [196]	train-logloss:0.00674	eval-logloss:0.09649
    [197]	train-logloss:0.00673	eval-logloss:0.09651
    [198]	train-logloss:0.00671	eval-logloss:0.09641
    [199]	train-logloss:0.00670	eval-logloss:0.09649
    [200]	train-logloss:0.00668	eval-logloss:0.09612
    [201]	train-logloss:0.00667	eval-logloss:0.09608
    [202]	train-logloss:0.00666	eval-logloss:0.09598
    [203]	train-logloss:0.00664	eval-logloss:0.09606
    [204]	train-logloss:0.00663	eval-logloss:0.09620
    [205]	train-logloss:0.00661	eval-logloss:0.09584
    [206]	train-logloss:0.00660	eval-logloss:0.09598
    [207]	train-logloss:0.00658	eval-logloss:0.09588
    [208]	train-logloss:0.00657	eval-logloss:0.09584
    [209]	train-logloss:0.00656	eval-logloss:0.09564
    [210]	train-logloss:0.00654	eval-logloss:0.09566
    [211]	train-logloss:0.00653	eval-logloss:0.09569
    [212]	train-logloss:0.00652	eval-logloss:0.09582
    [213]	train-logloss:0.00650	eval-logloss:0.09548
    [214]	train-logloss:0.00649	eval-logloss:0.09539
    [215]	train-logloss:0.00648	eval-logloss:0.09535
    [216]	train-logloss:0.00647	eval-logloss:0.09537
    [217]	train-logloss:0.00645	eval-logloss:0.09550
    [218]	train-logloss:0.00644	eval-logloss:0.09516
    [219]	train-logloss:0.00643	eval-logloss:0.09508
    [220]	train-logloss:0.00641	eval-logloss:0.09521
    [221]	train-logloss:0.00640	eval-logloss:0.09507
    [222]	train-logloss:0.00639	eval-logloss:0.09483
    [223]	train-logloss:0.00638	eval-logloss:0.09486
    [224]	train-logloss:0.00636	eval-logloss:0.09488
    [225]	train-logloss:0.00635	eval-logloss:0.09490
    [226]	train-logloss:0.00634	eval-logloss:0.09487
    [227]	train-logloss:0.00633	eval-logloss:0.09488
    [228]	train-logloss:0.00632	eval-logloss:0.09500
    [229]	train-logloss:0.00631	eval-logloss:0.09468
    [230]	train-logloss:0.00629	eval-logloss:0.09470
    [231]	train-logloss:0.00628	eval-logloss:0.09472
    [232]	train-logloss:0.00627	eval-logloss:0.09454
    [233]	train-logloss:0.00626	eval-logloss:0.09457
    [234]	train-logloss:0.00625	eval-logloss:0.09469
    [235]	train-logloss:0.00624	eval-logloss:0.09472
    [236]	train-logloss:0.00623	eval-logloss:0.09469
    [237]	train-logloss:0.00622	eval-logloss:0.09471
    [238]	train-logloss:0.00620	eval-logloss:0.09472
    [239]	train-logloss:0.00619	eval-logloss:0.09454
    [240]	train-logloss:0.00618	eval-logloss:0.09456
    [241]	train-logloss:0.00617	eval-logloss:0.09459
    [242]	train-logloss:0.00616	eval-logloss:0.09462
    [243]	train-logloss:0.00615	eval-logloss:0.09429
    [244]	train-logloss:0.00614	eval-logloss:0.09441
    [245]	train-logloss:0.00613	eval-logloss:0.09427
    [246]	train-logloss:0.00612	eval-logloss:0.09439
    [247]	train-logloss:0.00611	eval-logloss:0.09442
    [248]	train-logloss:0.00610	eval-logloss:0.09439
    [249]	train-logloss:0.00609	eval-logloss:0.09410
    [250]	train-logloss:0.00608	eval-logloss:0.09412
    [251]	train-logloss:0.00607	eval-logloss:0.09413
    [252]	train-logloss:0.00606	eval-logloss:0.09396
    [253]	train-logloss:0.00605	eval-logloss:0.09408
    [254]	train-logloss:0.00604	eval-logloss:0.09394
    [255]	train-logloss:0.00603	eval-logloss:0.09397
    [256]	train-logloss:0.00602	eval-logloss:0.09399
    [257]	train-logloss:0.00601	eval-logloss:0.09408
    [258]	train-logloss:0.00600	eval-logloss:0.09411
    [259]	train-logloss:0.00599	eval-logloss:0.09422
    [260]	train-logloss:0.00598	eval-logloss:0.09426
    [261]	train-logloss:0.00597	eval-logloss:0.09409
    [262]	train-logloss:0.00596	eval-logloss:0.09411
    [263]	train-logloss:0.00595	eval-logloss:0.09409
    [264]	train-logloss:0.00594	eval-logloss:0.09396
    [265]	train-logloss:0.00593	eval-logloss:0.09407
    [266]	train-logloss:0.00592	eval-logloss:0.09410
    [267]	train-logloss:0.00592	eval-logloss:0.09382
    [268]	train-logloss:0.00591	eval-logloss:0.09362
    [269]	train-logloss:0.00590	eval-logloss:0.09373
    [270]	train-logloss:0.00589	eval-logloss:0.09375
    [271]	train-logloss:0.00588	eval-logloss:0.09377
    [272]	train-logloss:0.00587	eval-logloss:0.09364
    [273]	train-logloss:0.00586	eval-logloss:0.09375
    [274]	train-logloss:0.00585	eval-logloss:0.09346
    [275]	train-logloss:0.00584	eval-logloss:0.09355
    [276]	train-logloss:0.00583	eval-logloss:0.09357
    [277]	train-logloss:0.00583	eval-logloss:0.09359
    [278]	train-logloss:0.00582	eval-logloss:0.09357
    [279]	train-logloss:0.00581	eval-logloss:0.09355
    [280]	train-logloss:0.00580	eval-logloss:0.09329
    [281]	train-logloss:0.00579	eval-logloss:0.09332
    [282]	train-logloss:0.00578	eval-logloss:0.09343
    [283]	train-logloss:0.00577	eval-logloss:0.09331
    [284]	train-logloss:0.00577	eval-logloss:0.09341
    [285]	train-logloss:0.00576	eval-logloss:0.09341
    [286]	train-logloss:0.00575	eval-logloss:0.09339
    [287]	train-logloss:0.00574	eval-logloss:0.09347
    [288]	train-logloss:0.00573	eval-logloss:0.09322
    [289]	train-logloss:0.00573	eval-logloss:0.09324
    [290]	train-logloss:0.00572	eval-logloss:0.09327
    [291]	train-logloss:0.00571	eval-logloss:0.09324
    [292]	train-logloss:0.00570	eval-logloss:0.09324
    [293]	train-logloss:0.00570	eval-logloss:0.09322
    [294]	train-logloss:0.00569	eval-logloss:0.09310
    [295]	train-logloss:0.00568	eval-logloss:0.09321
    [296]	train-logloss:0.00567	eval-logloss:0.09323
    [297]	train-logloss:0.00566	eval-logloss:0.09296
    [298]	train-logloss:0.00566	eval-logloss:0.09306
    [299]	train-logloss:0.00565	eval-logloss:0.09294
    [300]	train-logloss:0.00564	eval-logloss:0.09294
    [301]	train-logloss:0.00563	eval-logloss:0.09297
    [302]	train-logloss:0.00563	eval-logloss:0.09307
    [303]	train-logloss:0.00562	eval-logloss:0.09305
    [304]	train-logloss:0.00561	eval-logloss:0.09305
    [305]	train-logloss:0.00560	eval-logloss:0.09279
    [306]	train-logloss:0.00560	eval-logloss:0.09282
    [307]	train-logloss:0.00559	eval-logloss:0.09280
    [308]	train-logloss:0.00558	eval-logloss:0.09279
    [309]	train-logloss:0.00558	eval-logloss:0.09272
    [310]	train-logloss:0.00557	eval-logloss:0.09272
    [311]	train-logloss:0.00556	eval-logloss:0.09274
    [312]	train-logloss:0.00555	eval-logloss:0.09251
    [313]	train-logloss:0.00555	eval-logloss:0.09254
    [314]	train-logloss:0.00554	eval-logloss:0.09264
    [315]	train-logloss:0.00553	eval-logloss:0.09262
    [316]	train-logloss:0.00553	eval-logloss:0.09251
    [317]	train-logloss:0.00552	eval-logloss:0.09261
    [318]	train-logloss:0.00551	eval-logloss:0.09250
    [319]	train-logloss:0.00551	eval-logloss:0.09250
    [320]	train-logloss:0.00550	eval-logloss:0.09248
    [321]	train-logloss:0.00549	eval-logloss:0.09223
    [322]	train-logloss:0.00549	eval-logloss:0.09233
    [323]	train-logloss:0.00548	eval-logloss:0.09222
    [324]	train-logloss:0.00547	eval-logloss:0.09220
    [325]	train-logloss:0.00547	eval-logloss:0.09222
    [326]	train-logloss:0.00546	eval-logloss:0.09225
    [327]	train-logloss:0.00545	eval-logloss:0.09234
    [328]	train-logloss:0.00545	eval-logloss:0.09224
    [329]	train-logloss:0.00544	eval-logloss:0.09223
    [330]	train-logloss:0.00543	eval-logloss:0.09224
    [331]	train-logloss:0.00543	eval-logloss:0.09222
    [332]	train-logloss:0.00542	eval-logloss:0.09198
    [333]	train-logloss:0.00542	eval-logloss:0.09208
    [334]	train-logloss:0.00541	eval-logloss:0.09197
    [335]	train-logloss:0.00540	eval-logloss:0.09197
    [336]	train-logloss:0.00540	eval-logloss:0.09187
    [337]	train-logloss:0.00539	eval-logloss:0.09196
    [338]	train-logloss:0.00539	eval-logloss:0.09196
    [339]	train-logloss:0.00538	eval-logloss:0.09186
    [340]	train-logloss:0.00537	eval-logloss:0.09195
    [341]	train-logloss:0.00537	eval-logloss:0.09182
    [342]	train-logloss:0.00536	eval-logloss:0.09181
    [343]	train-logloss:0.00536	eval-logloss:0.09183
    [344]	train-logloss:0.00535	eval-logloss:0.09173
    [345]	train-logloss:0.00535	eval-logloss:0.09182
    [346]	train-logloss:0.00534	eval-logloss:0.09162
    [347]	train-logloss:0.00534	eval-logloss:0.09164
    [348]	train-logloss:0.00533	eval-logloss:0.09162
    [349]	train-logloss:0.00532	eval-logloss:0.09162
    [350]	train-logloss:0.00532	eval-logloss:0.09163
    [351]	train-logloss:0.00531	eval-logloss:0.09153
    [352]	train-logloss:0.00531	eval-logloss:0.09162
    [353]	train-logloss:0.00530	eval-logloss:0.09164
    [354]	train-logloss:0.00530	eval-logloss:0.09166
    [355]	train-logloss:0.00529	eval-logloss:0.09165
    [356]	train-logloss:0.00529	eval-logloss:0.09143
    [357]	train-logloss:0.00528	eval-logloss:0.09152
    [358]	train-logloss:0.00528	eval-logloss:0.09142
    [359]	train-logloss:0.00527	eval-logloss:0.09143
    [360]	train-logloss:0.00527	eval-logloss:0.09143
    [361]	train-logloss:0.00526	eval-logloss:0.09145
    [362]	train-logloss:0.00526	eval-logloss:0.09144
    [363]	train-logloss:0.00525	eval-logloss:0.09153
    [364]	train-logloss:0.00524	eval-logloss:0.09143
    [365]	train-logloss:0.00524	eval-logloss:0.09143
    [366]	train-logloss:0.00524	eval-logloss:0.09130
    [367]	train-logloss:0.00523	eval-logloss:0.09129
    [368]	train-logloss:0.00523	eval-logloss:0.09108
    [369]	train-logloss:0.00522	eval-logloss:0.09117
    [370]	train-logloss:0.00522	eval-logloss:0.09119
    [371]	train-logloss:0.00521	eval-logloss:0.09120
    [372]	train-logloss:0.00521	eval-logloss:0.09111
    [373]	train-logloss:0.00520	eval-logloss:0.09119
    [374]	train-logloss:0.00520	eval-logloss:0.09119
    [375]	train-logloss:0.00519	eval-logloss:0.09110
    [376]	train-logloss:0.00519	eval-logloss:0.09118
    [377]	train-logloss:0.00518	eval-logloss:0.09120
    [378]	train-logloss:0.00518	eval-logloss:0.09119
    [379]	train-logloss:0.00517	eval-logloss:0.09120
    [380]	train-logloss:0.00517	eval-logloss:0.09120
    [381]	train-logloss:0.00516	eval-logloss:0.09111
    [382]	train-logloss:0.00516	eval-logloss:0.09119
    [383]	train-logloss:0.00515	eval-logloss:0.09121
    [384]	train-logloss:0.00515	eval-logloss:0.09101
    [385]	train-logloss:0.00515	eval-logloss:0.09100
    [386]	train-logloss:0.00514	eval-logloss:0.09100
    [387]	train-logloss:0.00514	eval-logloss:0.09101
    [388]	train-logloss:0.00513	eval-logloss:0.09103
    [389]	train-logloss:0.00513	eval-logloss:0.09111
    [390]	train-logloss:0.00512	eval-logloss:0.09102
    [391]	train-logloss:0.00512	eval-logloss:0.09110
    [392]	train-logloss:0.00511	eval-logloss:0.09101
    [393]	train-logloss:0.00511	eval-logloss:0.09082
    [394]	train-logloss:0.00511	eval-logloss:0.09090
    [395]	train-logloss:0.00510	eval-logloss:0.09092
    [396]	train-logloss:0.00510	eval-logloss:0.09091
    [397]	train-logloss:0.00509	eval-logloss:0.09091
    [398]	train-logloss:0.00509	eval-logloss:0.09092
    [399]	train-logloss:0.00508	eval-logloss:0.09083



```python

```
