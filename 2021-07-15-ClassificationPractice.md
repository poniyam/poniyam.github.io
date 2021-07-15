# 캐글 산탄데르 고객 만족 예측

### 370개의 피처와 1개의 레이블로 구성
### 레이블의 이름은 TARGET 이고 1이면 불만인 고객이고 0이면 만족한 고객
### 데이터 다운로드: https://www.kaggle.com/c/santander-customer-satisfaction/data

### 데이터 읽어오기
=>캐글의 유럽에서도 문제를 많이 제시
=>인코딩이 utf-8 이나 euc-kr, ms949(cp949) 가 아닐 수 있음
서유럽에서는 iso-latin-1(iso-8859_1) 을 많이 사용합니다. 
pandas 에서는 latin-1 이라고 합니다.


```python
import numpy as np
import pandas as pd
import matplotlib as plt

cust_df = pd.read_csv('./data/santander/train.csv', encoding='latin-1')
print(cust_df.head())
cust_df.info()

#데이터의 개수를 확인해보니 10,000 개의 데이터가 넘으니 LGBM 을 사용해도 된다고 판단

```

       ID  var3  var15  imp_ent_var16_ult1  imp_op_var39_comer_ult1  \
    0   1     2     23                 0.0                      0.0   
    1   3     2     34                 0.0                      0.0   
    2   4     2     23                 0.0                      0.0   
    3   8     2     37                 0.0                    195.0   
    4  10     2     39                 0.0                      0.0   
    
       imp_op_var39_comer_ult3  imp_op_var40_comer_ult1  imp_op_var40_comer_ult3  \
    0                      0.0                      0.0                      0.0   
    1                      0.0                      0.0                      0.0   
    2                      0.0                      0.0                      0.0   
    3                    195.0                      0.0                      0.0   
    4                      0.0                      0.0                      0.0   
    
       imp_op_var40_efect_ult1  imp_op_var40_efect_ult3  ...  \
    0                      0.0                      0.0  ...   
    1                      0.0                      0.0  ...   
    2                      0.0                      0.0  ...   
    3                      0.0                      0.0  ...   
    4                      0.0                      0.0  ...   
    
       saldo_medio_var33_hace2  saldo_medio_var33_hace3  saldo_medio_var33_ult1  \
    0                      0.0                      0.0                     0.0   
    1                      0.0                      0.0                     0.0   
    2                      0.0                      0.0                     0.0   
    3                      0.0                      0.0                     0.0   
    4                      0.0                      0.0                     0.0   
    
       saldo_medio_var33_ult3  saldo_medio_var44_hace2  saldo_medio_var44_hace3  \
    0                     0.0                      0.0                      0.0   
    1                     0.0                      0.0                      0.0   
    2                     0.0                      0.0                      0.0   
    3                     0.0                      0.0                      0.0   
    4                     0.0                      0.0                      0.0   
    
       saldo_medio_var44_ult1  saldo_medio_var44_ult3          var38  TARGET  
    0                     0.0                     0.0   39205.170000       0  
    1                     0.0                     0.0   49278.030000       0  
    2                     0.0                     0.0   67333.770000       0  
    3                     0.0                     0.0   64007.970000       0  
    4                     0.0                     0.0  117310.979016       0  
    
    [5 rows x 371 columns]
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 76020 entries, 0 to 76019
    Columns: 371 entries, ID to TARGET
    dtypes: float64(111), int64(260)
    memory usage: 215.2 MB


## 타겟의 분포 확인


```python
#타겟의 빈도 수 확인
print(cust_df['TARGET'].value_counts())

#불만족의 개수
unsatisfied_cnt = cust_df[cust_df['TARGET'] == 1].TARGET.count()

#전체 데이터 개수
total_cnt = cust_df['TARGET'].count()

#비율 확인
print("불만족 비율:", unsatisfied_cnt/total_cnt)

#출력 결과를 보고 타겟의 분포가 불균형 하다는 것을 알게 됨
#이런 경우에는 층화 추출을 하던가 오버 나 언더 샘플링을 하기도 합니다.
```

    0    73012
    1     3008
    Name: TARGET, dtype: int64
    불만족 비율: 0.0395685345961589


## 기술 통계량 확인
### 대락적인 데이터의 분포 와 이상치를 탐색할 수 있고 각 피처의 데이터 크기를 확인


```python
cust_df.describe()

#각 피처들의 데이터 크기 확인 - 정규화가 필요할 지 모름
#min 과 max 값이 다른 값들과 차이가 많이 나는지 확인 - 이상치일 가능성이 있음
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
      <th>ID</th>
      <th>var3</th>
      <th>var15</th>
      <th>imp_ent_var16_ult1</th>
      <th>imp_op_var39_comer_ult1</th>
      <th>imp_op_var39_comer_ult3</th>
      <th>imp_op_var40_comer_ult1</th>
      <th>imp_op_var40_comer_ult3</th>
      <th>imp_op_var40_efect_ult1</th>
      <th>imp_op_var40_efect_ult3</th>
      <th>...</th>
      <th>saldo_medio_var33_hace2</th>
      <th>saldo_medio_var33_hace3</th>
      <th>saldo_medio_var33_ult1</th>
      <th>saldo_medio_var33_ult3</th>
      <th>saldo_medio_var44_hace2</th>
      <th>saldo_medio_var44_hace3</th>
      <th>saldo_medio_var44_ult1</th>
      <th>saldo_medio_var44_ult3</th>
      <th>var38</th>
      <th>TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>...</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>7.602000e+04</td>
      <td>76020.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>75964.050723</td>
      <td>-1523.199277</td>
      <td>33.212865</td>
      <td>86.208265</td>
      <td>72.363067</td>
      <td>119.529632</td>
      <td>3.559130</td>
      <td>6.472698</td>
      <td>0.412946</td>
      <td>0.567352</td>
      <td>...</td>
      <td>7.935824</td>
      <td>1.365146</td>
      <td>12.215580</td>
      <td>8.784074</td>
      <td>31.505324</td>
      <td>1.858575</td>
      <td>76.026165</td>
      <td>56.614351</td>
      <td>1.172358e+05</td>
      <td>0.039569</td>
    </tr>
    <tr>
      <th>std</th>
      <td>43781.947379</td>
      <td>39033.462364</td>
      <td>12.956486</td>
      <td>1614.757313</td>
      <td>339.315831</td>
      <td>546.266294</td>
      <td>93.155749</td>
      <td>153.737066</td>
      <td>30.604864</td>
      <td>36.513513</td>
      <td>...</td>
      <td>455.887218</td>
      <td>113.959637</td>
      <td>783.207399</td>
      <td>538.439211</td>
      <td>2013.125393</td>
      <td>147.786584</td>
      <td>4040.337842</td>
      <td>2852.579397</td>
      <td>1.826646e+05</td>
      <td>0.194945</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>-999999.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.163750e+03</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>38104.750000</td>
      <td>2.000000</td>
      <td>23.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.787061e+04</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>76043.000000</td>
      <td>2.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.064092e+05</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>113748.750000</td>
      <td>2.000000</td>
      <td>40.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.187563e+05</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>151838.000000</td>
      <td>238.000000</td>
      <td>105.000000</td>
      <td>210000.000000</td>
      <td>12888.030000</td>
      <td>21024.810000</td>
      <td>8237.820000</td>
      <td>11073.570000</td>
      <td>6600.000000</td>
      <td>6600.000000</td>
      <td>...</td>
      <td>50003.880000</td>
      <td>20385.720000</td>
      <td>138831.630000</td>
      <td>91778.730000</td>
      <td>438329.220000</td>
      <td>24650.010000</td>
      <td>681462.900000</td>
      <td>397884.300000</td>
      <td>2.203474e+07</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 371 columns</p>
</div>



## 데이터 전처리

### var3 에서 이상한 데이터(-999999 - None 의 의미)를 최빈값으로 치환
### 결측치 처리는 치환(최빈값, 평균, 중앙값, 이전값, 이후값, 머신러닝의 결과) 이나 제거



```python
#빈도수 확인
print(cust_df['var3'].value_counts())

#가장 많이 나온 2로 치환
cust_df['var3'].replace(-999999, 2, inplace=True)
```

     2         74165
     8           138
    -999999      116
     9           110
     3           108
               ...  
     218           1
     215           1
     151           1
     87            1
     191           1
    Name: var3, Length: 208, dtype: int64


### ID 피처는 분석에 사용하지 않을 것이므로 제거


```python
cust_df.drop('ID', axis=1, inplace=True)
```

### feature 와 target 을 분리


```python
#가장 마지막 열이 target
X_features = cust_df.iloc[:, :-1]
y_labels = cust_df.iloc[:, -1]
print(X_features.shape)
```

    (76020, 369)


### 훈련 데이터 와 테스트 데이터 분할


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, 
                                                   test_size=0.2, random_state=42)
#훈련 데이터 와 테스트 데이터 비율 확인
print(y_train.shape)
print(y_test.shape)


#원본 데이터 와의 비율을 확인 
#원본 데이터의 비율에 맞춰서 샘플링 하는 것을 층화 표본 추출이라고 합니다.

#훈련 데이터의 레이블 비율 확인
print(y_train.value_counts()/y_train.count())
#테스트 데이터의 레이블 비율 확인
print(y_test.value_counts()/y_test.count())
```

    (60816,)
    (15204,)
    0    0.96052
    1    0.03948
    Name: TARGET, dtype: float64
    0    0.960076
    1    0.039924
    Name: TARGET, dtype: float64


## 분류 알고리즘 적용
### 알고리즘: XGBoost
#### 하이퍼 파라미터
##### n_estimators 는 500, early_stopping_rounds는 100
##### 평가지표는 roc_auc


```python
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

xgb_clf = XGBClassifier(n_estimators=500, random_state=42)
# 100번 이상 수행한 상태에서 roc_auc 가 더이상 좋아지지 않으면 훈련 중지
xgb_clf.fit(X_train, y_train, early_stopping_rounds=100, eval_metric='auc',
           eval_set=[(X_train, y_train), (X_test, y_test)])

```

    C:\Users\admin\anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


    [0]	validation_0-auc:0.82081	validation_1-auc:0.79925
    [1]	validation_0-auc:0.83547	validation_1-auc:0.81372
    [2]	validation_0-auc:0.84069	validation_1-auc:0.82139
    [3]	validation_0-auc:0.84305	validation_1-auc:0.82258
    [4]	validation_0-auc:0.84664	validation_1-auc:0.82587
    [5]	validation_0-auc:0.85097	validation_1-auc:0.82742
    [6]	validation_0-auc:0.85317	validation_1-auc:0.83037
    [7]	validation_0-auc:0.85800	validation_1-auc:0.83326
    [8]	validation_0-auc:0.86224	validation_1-auc:0.83377
    [9]	validation_0-auc:0.86449	validation_1-auc:0.83608
    [10]	validation_0-auc:0.86850	validation_1-auc:0.83748
    [11]	validation_0-auc:0.87104	validation_1-auc:0.83952
    [12]	validation_0-auc:0.87384	validation_1-auc:0.83965
    [13]	validation_0-auc:0.87689	validation_1-auc:0.83826
    [14]	validation_0-auc:0.87870	validation_1-auc:0.83901
    [15]	validation_0-auc:0.88027	validation_1-auc:0.83961
    [16]	validation_0-auc:0.88249	validation_1-auc:0.84033
    [17]	validation_0-auc:0.88374	validation_1-auc:0.83983
    [18]	validation_0-auc:0.88570	validation_1-auc:0.83958
    [19]	validation_0-auc:0.88743	validation_1-auc:0.83861
    [20]	validation_0-auc:0.88821	validation_1-auc:0.83886
    [21]	validation_0-auc:0.88979	validation_1-auc:0.83901
    [22]	validation_0-auc:0.89021	validation_1-auc:0.84002
    [23]	validation_0-auc:0.89064	validation_1-auc:0.83973
    [24]	validation_0-auc:0.89293	validation_1-auc:0.83991
    [25]	validation_0-auc:0.89351	validation_1-auc:0.83949
    [26]	validation_0-auc:0.89532	validation_1-auc:0.83921
    [27]	validation_0-auc:0.89729	validation_1-auc:0.83851
    [28]	validation_0-auc:0.89866	validation_1-auc:0.83783
    [29]	validation_0-auc:0.89879	validation_1-auc:0.83772
    [30]	validation_0-auc:0.89938	validation_1-auc:0.83743
    [31]	validation_0-auc:0.89978	validation_1-auc:0.83765
    [32]	validation_0-auc:0.90084	validation_1-auc:0.83701
    [33]	validation_0-auc:0.90112	validation_1-auc:0.83710
    [34]	validation_0-auc:0.90134	validation_1-auc:0.83698
    [35]	validation_0-auc:0.90200	validation_1-auc:0.83646
    [36]	validation_0-auc:0.90222	validation_1-auc:0.83630
    [37]	validation_0-auc:0.90363	validation_1-auc:0.83625
    [38]	validation_0-auc:0.90388	validation_1-auc:0.83570
    [39]	validation_0-auc:0.90463	validation_1-auc:0.83649
    [40]	validation_0-auc:0.90585	validation_1-auc:0.83678
    [41]	validation_0-auc:0.90682	validation_1-auc:0.83586
    [42]	validation_0-auc:0.90719	validation_1-auc:0.83617
    [43]	validation_0-auc:0.90911	validation_1-auc:0.83536
    [44]	validation_0-auc:0.90946	validation_1-auc:0.83548
    [45]	validation_0-auc:0.90957	validation_1-auc:0.83537
    [46]	validation_0-auc:0.91103	validation_1-auc:0.83587
    [47]	validation_0-auc:0.91125	validation_1-auc:0.83568
    [48]	validation_0-auc:0.91158	validation_1-auc:0.83574
    [49]	validation_0-auc:0.91306	validation_1-auc:0.83481
    [50]	validation_0-auc:0.91369	validation_1-auc:0.83473
    [51]	validation_0-auc:0.91387	validation_1-auc:0.83470
    [52]	validation_0-auc:0.91400	validation_1-auc:0.83418
    [53]	validation_0-auc:0.91470	validation_1-auc:0.83364
    [54]	validation_0-auc:0.91506	validation_1-auc:0.83388
    [55]	validation_0-auc:0.91555	validation_1-auc:0.83303
    [56]	validation_0-auc:0.91627	validation_1-auc:0.83266
    [57]	validation_0-auc:0.91642	validation_1-auc:0.83279
    [58]	validation_0-auc:0.91659	validation_1-auc:0.83247
    [59]	validation_0-auc:0.91681	validation_1-auc:0.83249
    [60]	validation_0-auc:0.91800	validation_1-auc:0.83181
    [61]	validation_0-auc:0.91826	validation_1-auc:0.83175
    [62]	validation_0-auc:0.91839	validation_1-auc:0.83176
    [63]	validation_0-auc:0.91909	validation_1-auc:0.83185
    [64]	validation_0-auc:0.91993	validation_1-auc:0.83184
    [65]	validation_0-auc:0.92016	validation_1-auc:0.83141
    [66]	validation_0-auc:0.92054	validation_1-auc:0.83094
    [67]	validation_0-auc:0.92101	validation_1-auc:0.83028
    [68]	validation_0-auc:0.92118	validation_1-auc:0.83029
    [69]	validation_0-auc:0.92214	validation_1-auc:0.82986
    [70]	validation_0-auc:0.92317	validation_1-auc:0.82945
    [71]	validation_0-auc:0.92342	validation_1-auc:0.82895
    [72]	validation_0-auc:0.92362	validation_1-auc:0.82869
    [73]	validation_0-auc:0.92411	validation_1-auc:0.82822
    [74]	validation_0-auc:0.92488	validation_1-auc:0.82812
    [75]	validation_0-auc:0.92521	validation_1-auc:0.82828
    [76]	validation_0-auc:0.92532	validation_1-auc:0.82834
    [77]	validation_0-auc:0.92601	validation_1-auc:0.82821
    [78]	validation_0-auc:0.92638	validation_1-auc:0.82776
    [79]	validation_0-auc:0.92662	validation_1-auc:0.82713
    [80]	validation_0-auc:0.92680	validation_1-auc:0.82682
    [81]	validation_0-auc:0.92687	validation_1-auc:0.82677
    [82]	validation_0-auc:0.92710	validation_1-auc:0.82656
    [83]	validation_0-auc:0.92760	validation_1-auc:0.82630
    [84]	validation_0-auc:0.92762	validation_1-auc:0.82608
    [85]	validation_0-auc:0.92825	validation_1-auc:0.82591
    [86]	validation_0-auc:0.92897	validation_1-auc:0.82598
    [87]	validation_0-auc:0.92918	validation_1-auc:0.82529
    [88]	validation_0-auc:0.92927	validation_1-auc:0.82488
    [89]	validation_0-auc:0.92967	validation_1-auc:0.82436
    [90]	validation_0-auc:0.92973	validation_1-auc:0.82416
    [91]	validation_0-auc:0.92995	validation_1-auc:0.82363
    [92]	validation_0-auc:0.93007	validation_1-auc:0.82343
    [93]	validation_0-auc:0.93020	validation_1-auc:0.82369
    [94]	validation_0-auc:0.93032	validation_1-auc:0.82373
    [95]	validation_0-auc:0.93054	validation_1-auc:0.82363
    [96]	validation_0-auc:0.93124	validation_1-auc:0.82391
    [97]	validation_0-auc:0.93155	validation_1-auc:0.82341
    [98]	validation_0-auc:0.93170	validation_1-auc:0.82312
    [99]	validation_0-auc:0.93198	validation_1-auc:0.82302
    [100]	validation_0-auc:0.93216	validation_1-auc:0.82277
    [101]	validation_0-auc:0.93226	validation_1-auc:0.82267
    [102]	validation_0-auc:0.93228	validation_1-auc:0.82260
    [103]	validation_0-auc:0.93234	validation_1-auc:0.82262
    [104]	validation_0-auc:0.93253	validation_1-auc:0.82219
    [105]	validation_0-auc:0.93266	validation_1-auc:0.82203
    [106]	validation_0-auc:0.93282	validation_1-auc:0.82185
    [107]	validation_0-auc:0.93381	validation_1-auc:0.82203
    [108]	validation_0-auc:0.93398	validation_1-auc:0.82210
    [109]	validation_0-auc:0.93413	validation_1-auc:0.82211
    [110]	validation_0-auc:0.93459	validation_1-auc:0.82179
    [111]	validation_0-auc:0.93554	validation_1-auc:0.82109
    [112]	validation_0-auc:0.93588	validation_1-auc:0.82124
    [113]	validation_0-auc:0.93622	validation_1-auc:0.82142
    [114]	validation_0-auc:0.93670	validation_1-auc:0.82122
    [115]	validation_0-auc:0.93719	validation_1-auc:0.82093





    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                  importance_type='gain', interaction_constraints='',
                  learning_rate=0.300000012, max_delta_step=0, max_depth=6,
                  min_child_weight=1, missing=nan, monotone_constraints='()',
                  n_estimators=500, n_jobs=12, num_parallel_tree=1, random_state=42,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                  tree_method='exact', validate_parameters=1, verbosity=None)




```python
#평가 점수를 확인
#레이블이 불균형을 이룰 때는 average를 micro 로 설정하는 것이 좋고
#레이블이 균형을 이루는 경우에는 macro
xgb_roc_auc_score = roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:,1], 
                                 average='micro')
print('roc_auc_score:', xgb_roc_auc_score)
```

    roc_auc_score: 0.8403312093083152


## 하이퍼파라미터 튜닝

### max_depth 와 min_child_weight, colsample_bytree 만 조정


```python
from sklearn.model_selection import GridSearchCV

xgb_clf = XGBClassifier(n_estimators=100)
#파라미터 생성 - 시간 관계상 파라미터의 개수를 2개씩으로 설정한 것이고
#실제 모델을 가지고 학습할 때는 더 다양한 값을 설정해야 합니다.
params = {'max_depth':[5,7], 'min_child_weight':[1, 3],
          'colsamples_bytree':[0.5,0.75]}

#cv를 3으로 설정했으므로 24번 수행
gridcv = GridSearchCV(xgb_clf, param_grid = params, cv=3)
#early_stopping_rounds 를 조금 더 높여주어도 됩니다.
gridcv.fit(X_train, y_train, early_stopping_rounds=30, eval_metric='auc',
          eval_set=[(X_train, y_train), (X_test, y_test)])

#최적의 파라미터 출력
print("최적의 파라미터:", gridcv.best_params_)
xgb_roc_auc_score = roc_auc_score(y_test, gridcv.predict_proba(X_test)[:,1], 
                                 average='micro')
print('roc_auc_score:', xgb_roc_auc_score)
```

    C:\Users\admin\anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


    [15:06:01] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "colsamples_bytree" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [0]	validation_0-auc:0.81574	validation_1-auc:0.80177
    [1]	validation_0-auc:0.82540	validation_1-auc:0.81029
    [2]	validation_0-auc:0.82868	validation_1-auc:0.81288
    [3]	validation_0-auc:0.83075	validation_1-auc:0.81497
    [4]	validation_0-auc:0.83462	validation_1-auc:0.81745
    [5]	validation_0-auc:0.83689	validation_1-auc:0.81972
    [6]	validation_0-auc:0.84038	validation_1-auc:0.82364
    [7]	validation_0-auc:0.84344	validation_1-auc:0.82645
    [8]	validation_0-auc:0.84649	validation_1-auc:0.82893
    [9]	validation_0-auc:0.84847	validation_1-auc:0.82918
    [10]	validation_0-auc:0.85051	validation_1-auc:0.83116
    [11]	validation_0-auc:0.85273	validation_1-auc:0.83180
    [12]	validation_0-auc:0.85442	validation_1-auc:0.83355
    [13]	validation_0-auc:0.85649	validation_1-auc:0.83482
    [14]	validation_0-auc:0.85823	validation_1-auc:0.83621
    [15]	validation_0-auc:0.85954	validation_1-auc:0.83830
    [16]	validation_0-auc:0.86074	validation_1-auc:0.83913
    [17]	validation_0-auc:0.86165	validation_1-auc:0.83968
    [18]	validation_0-auc:0.86301	validation_1-auc:0.84009
    [19]	validation_0-auc:0.86368	validation_1-auc:0.84039
    [20]	validation_0-auc:0.86468	validation_1-auc:0.83951
    [21]	validation_0-auc:0.86507	validation_1-auc:0.83948
    [22]	validation_0-auc:0.86534	validation_1-auc:0.83950
    [23]	validation_0-auc:0.86576	validation_1-auc:0.83925
    [24]	validation_0-auc:0.86597	validation_1-auc:0.83934
    [25]	validation_0-auc:0.86601	validation_1-auc:0.83938
    [26]	validation_0-auc:0.86646	validation_1-auc:0.83931
    [27]	validation_0-auc:0.86740	validation_1-auc:0.83927
    [28]	validation_0-auc:0.86785	validation_1-auc:0.83883
    [29]	validation_0-auc:0.86792	validation_1-auc:0.83874
    [30]	validation_0-auc:0.86819	validation_1-auc:0.83836
    [31]	validation_0-auc:0.86837	validation_1-auc:0.83813
    [32]	validation_0-auc:0.86863	validation_1-auc:0.83813
    [33]	validation_0-auc:0.86884	validation_1-auc:0.83821
    [34]	validation_0-auc:0.86894	validation_1-auc:0.83812
    [35]	validation_0-auc:0.86956	validation_1-auc:0.83800
    [36]	validation_0-auc:0.87065	validation_1-auc:0.83748
    [37]	validation_0-auc:0.87130	validation_1-auc:0.83762
    [38]	validation_0-auc:0.87160	validation_1-auc:0.83788
    [39]	validation_0-auc:0.87162	validation_1-auc:0.83793
    [40]	validation_0-auc:0.87176	validation_1-auc:0.83768
    [41]	validation_0-auc:0.87184	validation_1-auc:0.83748
    [42]	validation_0-auc:0.87208	validation_1-auc:0.83708
    [43]	validation_0-auc:0.87237	validation_1-auc:0.83601
    [44]	validation_0-auc:0.87333	validation_1-auc:0.83501
    [45]	validation_0-auc:0.87382	validation_1-auc:0.83428
    [46]	validation_0-auc:0.87421	validation_1-auc:0.83381
    [47]	validation_0-auc:0.87443	validation_1-auc:0.83373
    [48]	validation_0-auc:0.87508	validation_1-auc:0.83378


    C:\Users\admin\anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


    [15:06:06] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "colsamples_bytree" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [0]	validation_0-auc:0.80809	validation_1-auc:0.80034
    [1]	validation_0-auc:0.82399	validation_1-auc:0.81539
    [2]	validation_0-auc:0.82956	validation_1-auc:0.81486
    [3]	validation_0-auc:0.83139	validation_1-auc:0.81403
    [4]	validation_0-auc:0.83306	validation_1-auc:0.81454
    [5]	validation_0-auc:0.84142	validation_1-auc:0.82495
    [6]	validation_0-auc:0.84361	validation_1-auc:0.82716
    [7]	validation_0-auc:0.84624	validation_1-auc:0.83084
    [8]	validation_0-auc:0.84905	validation_1-auc:0.83408
    [9]	validation_0-auc:0.85071	validation_1-auc:0.83490
    [10]	validation_0-auc:0.85460	validation_1-auc:0.83796
    [11]	validation_0-auc:0.85558	validation_1-auc:0.83924
    [12]	validation_0-auc:0.85662	validation_1-auc:0.84002
    [13]	validation_0-auc:0.85819	validation_1-auc:0.84012
    [14]	validation_0-auc:0.85922	validation_1-auc:0.83988
    [15]	validation_0-auc:0.86078	validation_1-auc:0.84000
    [16]	validation_0-auc:0.86266	validation_1-auc:0.83954
    [17]	validation_0-auc:0.86361	validation_1-auc:0.83981
    [18]	validation_0-auc:0.86487	validation_1-auc:0.83898
    [19]	validation_0-auc:0.86534	validation_1-auc:0.83947
    [20]	validation_0-auc:0.86580	validation_1-auc:0.83974
    [21]	validation_0-auc:0.86619	validation_1-auc:0.83977
    [22]	validation_0-auc:0.86717	validation_1-auc:0.83980
    [23]	validation_0-auc:0.86748	validation_1-auc:0.84010
    [24]	validation_0-auc:0.86815	validation_1-auc:0.83852
    [25]	validation_0-auc:0.86823	validation_1-auc:0.83825
    [26]	validation_0-auc:0.86867	validation_1-auc:0.83812
    [27]	validation_0-auc:0.86914	validation_1-auc:0.83887
    [28]	validation_0-auc:0.86968	validation_1-auc:0.83838
    [29]	validation_0-auc:0.87068	validation_1-auc:0.83889
    [30]	validation_0-auc:0.87071	validation_1-auc:0.83860
    [31]	validation_0-auc:0.87131	validation_1-auc:0.83881
    [32]	validation_0-auc:0.87156	validation_1-auc:0.83848
    [33]	validation_0-auc:0.87252	validation_1-auc:0.83838
    [34]	validation_0-auc:0.87288	validation_1-auc:0.83836
    [35]	validation_0-auc:0.87313	validation_1-auc:0.83811
    [36]	validation_0-auc:0.87471	validation_1-auc:0.83786
    [37]	validation_0-auc:0.87481	validation_1-auc:0.83782
    [38]	validation_0-auc:0.87479	validation_1-auc:0.83746
    [39]	validation_0-auc:0.87558	validation_1-auc:0.83762
    [40]	validation_0-auc:0.87622	validation_1-auc:0.83708
    [41]	validation_0-auc:0.87645	validation_1-auc:0.83683
    [42]	validation_0-auc:0.87676	validation_1-auc:0.83642


    C:\Users\admin\anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


    [15:06:11] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "colsamples_bytree" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [0]	validation_0-auc:0.81210	validation_1-auc:0.79994
    [1]	validation_0-auc:0.82550	validation_1-auc:0.81441
    [2]	validation_0-auc:0.82904	validation_1-auc:0.81624
    [3]	validation_0-auc:0.83135	validation_1-auc:0.81725
    [4]	validation_0-auc:0.83364	validation_1-auc:0.81958
    [5]	validation_0-auc:0.83526	validation_1-auc:0.81995
    [6]	validation_0-auc:0.84052	validation_1-auc:0.82608
    [7]	validation_0-auc:0.84169	validation_1-auc:0.82640
    [8]	validation_0-auc:0.84668	validation_1-auc:0.82951
    [9]	validation_0-auc:0.85018	validation_1-auc:0.83051
    [10]	validation_0-auc:0.85098	validation_1-auc:0.83139
    [11]	validation_0-auc:0.85283	validation_1-auc:0.83208
    [12]	validation_0-auc:0.85505	validation_1-auc:0.83195
    [13]	validation_0-auc:0.85650	validation_1-auc:0.83383
    [14]	validation_0-auc:0.85832	validation_1-auc:0.83448
    [15]	validation_0-auc:0.85956	validation_1-auc:0.83520
    [16]	validation_0-auc:0.86117	validation_1-auc:0.83456
    [17]	validation_0-auc:0.86261	validation_1-auc:0.83449
    [18]	validation_0-auc:0.86506	validation_1-auc:0.83349
    [19]	validation_0-auc:0.86595	validation_1-auc:0.83467
    [20]	validation_0-auc:0.86696	validation_1-auc:0.83440
    [21]	validation_0-auc:0.86797	validation_1-auc:0.83422
    [22]	validation_0-auc:0.86898	validation_1-auc:0.83369
    [23]	validation_0-auc:0.86978	validation_1-auc:0.83349
    [24]	validation_0-auc:0.87052	validation_1-auc:0.83261
    [25]	validation_0-auc:0.87188	validation_1-auc:0.83254
    [26]	validation_0-auc:0.87246	validation_1-auc:0.83303
    [27]	validation_0-auc:0.87279	validation_1-auc:0.83310
    [28]	validation_0-auc:0.87333	validation_1-auc:0.83366
    [29]	validation_0-auc:0.87356	validation_1-auc:0.83360
    [30]	validation_0-auc:0.87408	validation_1-auc:0.83312
    [31]	validation_0-auc:0.87426	validation_1-auc:0.83324
    [32]	validation_0-auc:0.87448	validation_1-auc:0.83285
    [33]	validation_0-auc:0.87456	validation_1-auc:0.83239
    [34]	validation_0-auc:0.87548	validation_1-auc:0.83268
    [35]	validation_0-auc:0.87692	validation_1-auc:0.83215
    [36]	validation_0-auc:0.87747	validation_1-auc:0.83166
    [37]	validation_0-auc:0.87830	validation_1-auc:0.83235
    [38]	validation_0-auc:0.87857	validation_1-auc:0.83224
    [39]	validation_0-auc:0.87890	validation_1-auc:0.83193
    [40]	validation_0-auc:0.87907	validation_1-auc:0.83212
    [41]	validation_0-auc:0.87940	validation_1-auc:0.83185
    [42]	validation_0-auc:0.87955	validation_1-auc:0.83175
    [43]	validation_0-auc:0.87985	validation_1-auc:0.83157
    [44]	validation_0-auc:0.88060	validation_1-auc:0.83189
    [45]	validation_0-auc:0.88067	validation_1-auc:0.83186


    C:\Users\admin\anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


    [15:06:17] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "colsamples_bytree" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [0]	validation_0-auc:0.81467	validation_1-auc:0.80080
    [1]	validation_0-auc:0.82467	validation_1-auc:0.81106
    [2]	validation_0-auc:0.82894	validation_1-auc:0.81460
    [3]	validation_0-auc:0.82965	validation_1-auc:0.81369
    [4]	validation_0-auc:0.83378	validation_1-auc:0.81819
    [5]	validation_0-auc:0.83661	validation_1-auc:0.82072
    [6]	validation_0-auc:0.83864	validation_1-auc:0.82244
    [7]	validation_0-auc:0.84405	validation_1-auc:0.82609
    [8]	validation_0-auc:0.84660	validation_1-auc:0.82685
    [9]	validation_0-auc:0.84855	validation_1-auc:0.82874
    [10]	validation_0-auc:0.84945	validation_1-auc:0.82948
    [11]	validation_0-auc:0.85203	validation_1-auc:0.83241
    [12]	validation_0-auc:0.85382	validation_1-auc:0.83378
    [13]	validation_0-auc:0.85483	validation_1-auc:0.83436
    [14]	validation_0-auc:0.85615	validation_1-auc:0.83478
    [15]	validation_0-auc:0.85671	validation_1-auc:0.83603
    [16]	validation_0-auc:0.85791	validation_1-auc:0.83570
    [17]	validation_0-auc:0.85915	validation_1-auc:0.83758
    [18]	validation_0-auc:0.86048	validation_1-auc:0.83673
    [19]	validation_0-auc:0.86134	validation_1-auc:0.83488
    [20]	validation_0-auc:0.86147	validation_1-auc:0.83540
    [21]	validation_0-auc:0.86236	validation_1-auc:0.83601
    [22]	validation_0-auc:0.86312	validation_1-auc:0.83590
    [23]	validation_0-auc:0.86404	validation_1-auc:0.83589
    [24]	validation_0-auc:0.86413	validation_1-auc:0.83588
    [25]	validation_0-auc:0.86457	validation_1-auc:0.83556
    [26]	validation_0-auc:0.86492	validation_1-auc:0.83554
    [27]	validation_0-auc:0.86581	validation_1-auc:0.83597
    [28]	validation_0-auc:0.86589	validation_1-auc:0.83582
    [29]	validation_0-auc:0.86608	validation_1-auc:0.83584
    [30]	validation_0-auc:0.86646	validation_1-auc:0.83564
    [31]	validation_0-auc:0.86680	validation_1-auc:0.83523
    [32]	validation_0-auc:0.86742	validation_1-auc:0.83504
    [33]	validation_0-auc:0.86765	validation_1-auc:0.83522
    [34]	validation_0-auc:0.86784	validation_1-auc:0.83497
    [35]	validation_0-auc:0.86803	validation_1-auc:0.83398
    [36]	validation_0-auc:0.86817	validation_1-auc:0.83403
    [37]	validation_0-auc:0.86813	validation_1-auc:0.83355
    [38]	validation_0-auc:0.86888	validation_1-auc:0.83290
    [39]	validation_0-auc:0.86898	validation_1-auc:0.83303
    [40]	validation_0-auc:0.86996	validation_1-auc:0.83286
    [41]	validation_0-auc:0.87016	validation_1-auc:0.83262
    [42]	validation_0-auc:0.87128	validation_1-auc:0.83234
    [43]	validation_0-auc:0.87132	validation_1-auc:0.83260
    [44]	validation_0-auc:0.87203	validation_1-auc:0.83215
    [45]	validation_0-auc:0.87223	validation_1-auc:0.83206
    [46]	validation_0-auc:0.87246	validation_1-auc:0.83211
    [47]	validation_0-auc:0.87342	validation_1-auc:0.83168


    C:\Users\admin\anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


    [15:06:22] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "colsamples_bytree" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [0]	validation_0-auc:0.81267	validation_1-auc:0.80112
    [1]	validation_0-auc:0.82759	validation_1-auc:0.81306
    [2]	validation_0-auc:0.83099	validation_1-auc:0.81471
    [3]	validation_0-auc:0.83497	validation_1-auc:0.82058
    [4]	validation_0-auc:0.83631	validation_1-auc:0.82090
    [5]	validation_0-auc:0.84166	validation_1-auc:0.82807
    [6]	validation_0-auc:0.84374	validation_1-auc:0.82842
    [7]	validation_0-auc:0.84539	validation_1-auc:0.83106
    [8]	validation_0-auc:0.84799	validation_1-auc:0.83523
    [9]	validation_0-auc:0.85035	validation_1-auc:0.83786
    [10]	validation_0-auc:0.85175	validation_1-auc:0.83851
    [11]	validation_0-auc:0.85362	validation_1-auc:0.83876
    [12]	validation_0-auc:0.85524	validation_1-auc:0.83980
    [13]	validation_0-auc:0.85709	validation_1-auc:0.84089
    [14]	validation_0-auc:0.85796	validation_1-auc:0.84129
    [15]	validation_0-auc:0.85958	validation_1-auc:0.84412
    [16]	validation_0-auc:0.86014	validation_1-auc:0.84456
    [17]	validation_0-auc:0.86141	validation_1-auc:0.84425
    [18]	validation_0-auc:0.86165	validation_1-auc:0.84433
    [19]	validation_0-auc:0.86221	validation_1-auc:0.84433
    [20]	validation_0-auc:0.86266	validation_1-auc:0.84403
    [21]	validation_0-auc:0.86385	validation_1-auc:0.84400
    [22]	validation_0-auc:0.86482	validation_1-auc:0.84421
    [23]	validation_0-auc:0.86491	validation_1-auc:0.84408
    [24]	validation_0-auc:0.86585	validation_1-auc:0.84386
    [25]	validation_0-auc:0.86726	validation_1-auc:0.84382
    [26]	validation_0-auc:0.86753	validation_1-auc:0.84380
    [27]	validation_0-auc:0.86767	validation_1-auc:0.84346
    [28]	validation_0-auc:0.86813	validation_1-auc:0.84377
    [29]	validation_0-auc:0.86814	validation_1-auc:0.84351
    [30]	validation_0-auc:0.86868	validation_1-auc:0.84385
    [31]	validation_0-auc:0.86890	validation_1-auc:0.84409
    [32]	validation_0-auc:0.86935	validation_1-auc:0.84411
    [33]	validation_0-auc:0.86975	validation_1-auc:0.84376
    [34]	validation_0-auc:0.87040	validation_1-auc:0.84363
    [35]	validation_0-auc:0.87062	validation_1-auc:0.84324
    [36]	validation_0-auc:0.87089	validation_1-auc:0.84268
    [37]	validation_0-auc:0.87158	validation_1-auc:0.84237
    [38]	validation_0-auc:0.87286	validation_1-auc:0.84211
    [39]	validation_0-auc:0.87317	validation_1-auc:0.84220
    [40]	validation_0-auc:0.87330	validation_1-auc:0.84198
    [41]	validation_0-auc:0.87356	validation_1-auc:0.84162
    [42]	validation_0-auc:0.87360	validation_1-auc:0.84155
    [43]	validation_0-auc:0.87382	validation_1-auc:0.84133
    [44]	validation_0-auc:0.87435	validation_1-auc:0.84094
    [45]	validation_0-auc:0.87429	validation_1-auc:0.84079
    [46]	validation_0-auc:0.87482	validation_1-auc:0.84034


    C:\Users\admin\anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


    [15:06:27] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "colsamples_bytree" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [0]	validation_0-auc:0.81897	validation_1-auc:0.80631
    [1]	validation_0-auc:0.82510	validation_1-auc:0.81305
    [2]	validation_0-auc:0.82775	validation_1-auc:0.81513
    [3]	validation_0-auc:0.83074	validation_1-auc:0.81791
    [4]	validation_0-auc:0.83267	validation_1-auc:0.81753
    [5]	validation_0-auc:0.83487	validation_1-auc:0.82178
    [6]	validation_0-auc:0.84024	validation_1-auc:0.82724
    [7]	validation_0-auc:0.84425	validation_1-auc:0.82813
    [8]	validation_0-auc:0.84612	validation_1-auc:0.83280
    [9]	validation_0-auc:0.84972	validation_1-auc:0.83280
    [10]	validation_0-auc:0.85120	validation_1-auc:0.83198
    [11]	validation_0-auc:0.85270	validation_1-auc:0.83408
    [12]	validation_0-auc:0.85475	validation_1-auc:0.83382
    [13]	validation_0-auc:0.85627	validation_1-auc:0.83412
    [14]	validation_0-auc:0.85766	validation_1-auc:0.83472
    [15]	validation_0-auc:0.85900	validation_1-auc:0.83438
    [16]	validation_0-auc:0.86011	validation_1-auc:0.83483
    [17]	validation_0-auc:0.86062	validation_1-auc:0.83474
    [18]	validation_0-auc:0.86192	validation_1-auc:0.83488
    [19]	validation_0-auc:0.86296	validation_1-auc:0.83429
    [20]	validation_0-auc:0.86376	validation_1-auc:0.83418
    [21]	validation_0-auc:0.86479	validation_1-auc:0.83389
    [22]	validation_0-auc:0.86566	validation_1-auc:0.83368
    [23]	validation_0-auc:0.86626	validation_1-auc:0.83362
    [24]	validation_0-auc:0.86675	validation_1-auc:0.83419
    [25]	validation_0-auc:0.86720	validation_1-auc:0.83443
    [26]	validation_0-auc:0.86804	validation_1-auc:0.83452
    [27]	validation_0-auc:0.86852	validation_1-auc:0.83491
    [28]	validation_0-auc:0.86904	validation_1-auc:0.83447
    [29]	validation_0-auc:0.87063	validation_1-auc:0.83339
    [30]	validation_0-auc:0.87091	validation_1-auc:0.83365
    [31]	validation_0-auc:0.87228	validation_1-auc:0.83280
    [32]	validation_0-auc:0.87234	validation_1-auc:0.83263
    [33]	validation_0-auc:0.87272	validation_1-auc:0.83223
    [34]	validation_0-auc:0.87293	validation_1-auc:0.83213
    [35]	validation_0-auc:0.87312	validation_1-auc:0.83179
    [36]	validation_0-auc:0.87336	validation_1-auc:0.83159
    [37]	validation_0-auc:0.87355	validation_1-auc:0.83155
    [38]	validation_0-auc:0.87398	validation_1-auc:0.83126
    [39]	validation_0-auc:0.87433	validation_1-auc:0.83090
    [40]	validation_0-auc:0.87461	validation_1-auc:0.83041
    [41]	validation_0-auc:0.87495	validation_1-auc:0.83058
    [42]	validation_0-auc:0.87545	validation_1-auc:0.83025
    [43]	validation_0-auc:0.87551	validation_1-auc:0.83018
    [44]	validation_0-auc:0.87587	validation_1-auc:0.83058
    [45]	validation_0-auc:0.87638	validation_1-auc:0.83062
    [46]	validation_0-auc:0.87732	validation_1-auc:0.82946
    [47]	validation_0-auc:0.87760	validation_1-auc:0.82974
    [48]	validation_0-auc:0.87783	validation_1-auc:0.82965
    [49]	validation_0-auc:0.87819	validation_1-auc:0.82896
    [50]	validation_0-auc:0.87832	validation_1-auc:0.82860
    [51]	validation_0-auc:0.87855	validation_1-auc:0.82865
    [52]	validation_0-auc:0.87914	validation_1-auc:0.82864
    [53]	validation_0-auc:0.87937	validation_1-auc:0.82831
    [54]	validation_0-auc:0.87975	validation_1-auc:0.82803
    [55]	validation_0-auc:0.88018	validation_1-auc:0.82852
    [56]	validation_0-auc:0.88026	validation_1-auc:0.82826
    [57]	validation_0-auc:0.88057	validation_1-auc:0.82812


    C:\Users\admin\anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


    [15:06:34] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "colsamples_bytree" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [0]	validation_0-auc:0.82483	validation_1-auc:0.80939
    [1]	validation_0-auc:0.83013	validation_1-auc:0.80990
    [2]	validation_0-auc:0.83360	validation_1-auc:0.81395
    [3]	validation_0-auc:0.83644	validation_1-auc:0.81362
    [4]	validation_0-auc:0.84346	validation_1-auc:0.81794
    [5]	validation_0-auc:0.84742	validation_1-auc:0.82094
    [6]	validation_0-auc:0.85282	validation_1-auc:0.82503
    [7]	validation_0-auc:0.85855	validation_1-auc:0.82940
    [8]	validation_0-auc:0.86260	validation_1-auc:0.82851
    [9]	validation_0-auc:0.86553	validation_1-auc:0.83110
    [10]	validation_0-auc:0.86882	validation_1-auc:0.83099
    [11]	validation_0-auc:0.87167	validation_1-auc:0.83240
    [12]	validation_0-auc:0.87447	validation_1-auc:0.83186
    [13]	validation_0-auc:0.87639	validation_1-auc:0.83144
    [14]	validation_0-auc:0.87763	validation_1-auc:0.83091
    [15]	validation_0-auc:0.87946	validation_1-auc:0.83345
    [16]	validation_0-auc:0.88112	validation_1-auc:0.83342
    [17]	validation_0-auc:0.88198	validation_1-auc:0.83373
    [18]	validation_0-auc:0.88330	validation_1-auc:0.83340
    [19]	validation_0-auc:0.88567	validation_1-auc:0.83390
    [20]	validation_0-auc:0.88588	validation_1-auc:0.83373
    [21]	validation_0-auc:0.88716	validation_1-auc:0.83240
    [22]	validation_0-auc:0.88791	validation_1-auc:0.83177
    [23]	validation_0-auc:0.88855	validation_1-auc:0.83211
    [24]	validation_0-auc:0.88887	validation_1-auc:0.83215
    [25]	validation_0-auc:0.88928	validation_1-auc:0.83215
    [26]	validation_0-auc:0.88953	validation_1-auc:0.83235
    [27]	validation_0-auc:0.88958	validation_1-auc:0.83252
    [28]	validation_0-auc:0.88960	validation_1-auc:0.83247
    [29]	validation_0-auc:0.89014	validation_1-auc:0.83173
    [30]	validation_0-auc:0.89045	validation_1-auc:0.83123
    [31]	validation_0-auc:0.89131	validation_1-auc:0.83141
    [32]	validation_0-auc:0.89179	validation_1-auc:0.83088
    [33]	validation_0-auc:0.89199	validation_1-auc:0.83074
    [34]	validation_0-auc:0.89237	validation_1-auc:0.82990
    [35]	validation_0-auc:0.89232	validation_1-auc:0.83001
    [36]	validation_0-auc:0.89243	validation_1-auc:0.83012
    [37]	validation_0-auc:0.89289	validation_1-auc:0.83034
    [38]	validation_0-auc:0.89308	validation_1-auc:0.83047
    [39]	validation_0-auc:0.89336	validation_1-auc:0.83035
    [40]	validation_0-auc:0.89415	validation_1-auc:0.82991
    [41]	validation_0-auc:0.89513	validation_1-auc:0.82878
    [42]	validation_0-auc:0.89550	validation_1-auc:0.82868
    [43]	validation_0-auc:0.89617	validation_1-auc:0.82897
    [44]	validation_0-auc:0.89610	validation_1-auc:0.82891
    [45]	validation_0-auc:0.89638	validation_1-auc:0.82890
    [46]	validation_0-auc:0.89688	validation_1-auc:0.82795
    [47]	validation_0-auc:0.89720	validation_1-auc:0.82675
    [48]	validation_0-auc:0.89794	validation_1-auc:0.82668
    [49]	validation_0-auc:0.89807	validation_1-auc:0.82642


    C:\Users\admin\anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


    [15:06:40] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "colsamples_bytree" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [0]	validation_0-auc:0.81430	validation_1-auc:0.80241
    [1]	validation_0-auc:0.83069	validation_1-auc:0.81451
    [2]	validation_0-auc:0.83796	validation_1-auc:0.81684
    [3]	validation_0-auc:0.84068	validation_1-auc:0.81668
    [4]	validation_0-auc:0.84335	validation_1-auc:0.81661
    [5]	validation_0-auc:0.85263	validation_1-auc:0.82610
    [6]	validation_0-auc:0.85698	validation_1-auc:0.82862
    [7]	validation_0-auc:0.86088	validation_1-auc:0.83080
    [8]	validation_0-auc:0.86482	validation_1-auc:0.83230
    [9]	validation_0-auc:0.86966	validation_1-auc:0.83181
    [10]	validation_0-auc:0.87207	validation_1-auc:0.83388
    [11]	validation_0-auc:0.87596	validation_1-auc:0.83439
    [12]	validation_0-auc:0.87907	validation_1-auc:0.83463
    [13]	validation_0-auc:0.88080	validation_1-auc:0.83517
    [14]	validation_0-auc:0.88249	validation_1-auc:0.83511
    [15]	validation_0-auc:0.88493	validation_1-auc:0.83479
    [16]	validation_0-auc:0.88651	validation_1-auc:0.83422
    [17]	validation_0-auc:0.88770	validation_1-auc:0.83381
    [18]	validation_0-auc:0.88812	validation_1-auc:0.83391
    [19]	validation_0-auc:0.88953	validation_1-auc:0.83347
    [20]	validation_0-auc:0.89026	validation_1-auc:0.83281
    [21]	validation_0-auc:0.89114	validation_1-auc:0.83292
    [22]	validation_0-auc:0.89176	validation_1-auc:0.83286
    [23]	validation_0-auc:0.89226	validation_1-auc:0.83281
    [24]	validation_0-auc:0.89271	validation_1-auc:0.83266
    [25]	validation_0-auc:0.89281	validation_1-auc:0.83240
    [26]	validation_0-auc:0.89295	validation_1-auc:0.83150
    [27]	validation_0-auc:0.89357	validation_1-auc:0.83174
    [28]	validation_0-auc:0.89469	validation_1-auc:0.83103
    [29]	validation_0-auc:0.89478	validation_1-auc:0.83102
    [30]	validation_0-auc:0.89568	validation_1-auc:0.83003
    [31]	validation_0-auc:0.89710	validation_1-auc:0.82869
    [32]	validation_0-auc:0.89772	validation_1-auc:0.82897
    [33]	validation_0-auc:0.89783	validation_1-auc:0.82930
    [34]	validation_0-auc:0.89829	validation_1-auc:0.82880
    [35]	validation_0-auc:0.89820	validation_1-auc:0.82889
    [36]	validation_0-auc:0.89849	validation_1-auc:0.82822
    [37]	validation_0-auc:0.89874	validation_1-auc:0.82781
    [38]	validation_0-auc:0.89921	validation_1-auc:0.82776
    [39]	validation_0-auc:0.89931	validation_1-auc:0.82742
    [40]	validation_0-auc:0.89958	validation_1-auc:0.82717
    [41]	validation_0-auc:0.89989	validation_1-auc:0.82679
    [42]	validation_0-auc:0.90007	validation_1-auc:0.82664
    [43]	validation_0-auc:0.90034	validation_1-auc:0.82666


    C:\Users\admin\anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


    [15:06:46] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "colsamples_bytree" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [0]	validation_0-auc:0.82346	validation_1-auc:0.80827
    [1]	validation_0-auc:0.83229	validation_1-auc:0.81327
    [2]	validation_0-auc:0.83781	validation_1-auc:0.81714
    [3]	validation_0-auc:0.84400	validation_1-auc:0.82109
    [4]	validation_0-auc:0.84735	validation_1-auc:0.82371
    [5]	validation_0-auc:0.85010	validation_1-auc:0.82568
    [6]	validation_0-auc:0.85285	validation_1-auc:0.82809
    [7]	validation_0-auc:0.85776	validation_1-auc:0.83283
    [8]	validation_0-auc:0.86287	validation_1-auc:0.83337
    [9]	validation_0-auc:0.86719	validation_1-auc:0.83305
    [10]	validation_0-auc:0.87058	validation_1-auc:0.83230
    [11]	validation_0-auc:0.87355	validation_1-auc:0.83134
    [12]	validation_0-auc:0.87600	validation_1-auc:0.83188
    [13]	validation_0-auc:0.87916	validation_1-auc:0.83188
    [14]	validation_0-auc:0.88075	validation_1-auc:0.83207
    [15]	validation_0-auc:0.88261	validation_1-auc:0.83149
    [16]	validation_0-auc:0.88417	validation_1-auc:0.83074
    [17]	validation_0-auc:0.88507	validation_1-auc:0.83097
    [18]	validation_0-auc:0.88599	validation_1-auc:0.83098
    [19]	validation_0-auc:0.88742	validation_1-auc:0.83025
    [20]	validation_0-auc:0.88823	validation_1-auc:0.83029
    [21]	validation_0-auc:0.88903	validation_1-auc:0.82953
    [22]	validation_0-auc:0.88919	validation_1-auc:0.82913
    [23]	validation_0-auc:0.89128	validation_1-auc:0.82913
    [24]	validation_0-auc:0.89148	validation_1-auc:0.82926
    [25]	validation_0-auc:0.89261	validation_1-auc:0.82915
    [26]	validation_0-auc:0.89308	validation_1-auc:0.82916
    [27]	validation_0-auc:0.89344	validation_1-auc:0.82900
    [28]	validation_0-auc:0.89394	validation_1-auc:0.82856
    [29]	validation_0-auc:0.89416	validation_1-auc:0.82833
    [30]	validation_0-auc:0.89436	validation_1-auc:0.82802
    [31]	validation_0-auc:0.89458	validation_1-auc:0.82809
    [32]	validation_0-auc:0.89578	validation_1-auc:0.82676
    [33]	validation_0-auc:0.89611	validation_1-auc:0.82676
    [34]	validation_0-auc:0.89621	validation_1-auc:0.82672
    [35]	validation_0-auc:0.89668	validation_1-auc:0.82628
    [36]	validation_0-auc:0.89689	validation_1-auc:0.82616
    [37]	validation_0-auc:0.89733	validation_1-auc:0.82598
    [38]	validation_0-auc:0.89751	validation_1-auc:0.82581


    C:\Users\admin\anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


    [15:06:52] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "colsamples_bytree" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [0]	validation_0-auc:0.82342	validation_1-auc:0.80583
    [1]	validation_0-auc:0.82923	validation_1-auc:0.81038
    [2]	validation_0-auc:0.83377	validation_1-auc:0.81499
    [3]	validation_0-auc:0.83813	validation_1-auc:0.81923
    [4]	validation_0-auc:0.84381	validation_1-auc:0.82164
    [5]	validation_0-auc:0.84556	validation_1-auc:0.82039
    [6]	validation_0-auc:0.85191	validation_1-auc:0.82747
    [7]	validation_0-auc:0.85585	validation_1-auc:0.82956
    [8]	validation_0-auc:0.85897	validation_1-auc:0.83026
    [9]	validation_0-auc:0.86094	validation_1-auc:0.83100
    [10]	validation_0-auc:0.86319	validation_1-auc:0.83167
    [11]	validation_0-auc:0.86571	validation_1-auc:0.83233
    [12]	validation_0-auc:0.86818	validation_1-auc:0.83203
    [13]	validation_0-auc:0.87043	validation_1-auc:0.83226
    [14]	validation_0-auc:0.87234	validation_1-auc:0.83186
    [15]	validation_0-auc:0.87375	validation_1-auc:0.83339
    [16]	validation_0-auc:0.87459	validation_1-auc:0.83390
    [17]	validation_0-auc:0.87505	validation_1-auc:0.83497
    [18]	validation_0-auc:0.87630	validation_1-auc:0.83457
    [19]	validation_0-auc:0.87776	validation_1-auc:0.83492
    [20]	validation_0-auc:0.87789	validation_1-auc:0.83479
    [21]	validation_0-auc:0.87899	validation_1-auc:0.83416
    [22]	validation_0-auc:0.87928	validation_1-auc:0.83374
    [23]	validation_0-auc:0.88035	validation_1-auc:0.83396
    [24]	validation_0-auc:0.88077	validation_1-auc:0.83408
    [25]	validation_0-auc:0.88168	validation_1-auc:0.83376
    [26]	validation_0-auc:0.88189	validation_1-auc:0.83311
    [27]	validation_0-auc:0.88252	validation_1-auc:0.83323
    [28]	validation_0-auc:0.88280	validation_1-auc:0.83358
    [29]	validation_0-auc:0.88371	validation_1-auc:0.83360
    [30]	validation_0-auc:0.88452	validation_1-auc:0.83240
    [31]	validation_0-auc:0.88507	validation_1-auc:0.83242
    [32]	validation_0-auc:0.88523	validation_1-auc:0.83241
    [33]	validation_0-auc:0.88524	validation_1-auc:0.83241
    [34]	validation_0-auc:0.88572	validation_1-auc:0.83191
    [35]	validation_0-auc:0.88594	validation_1-auc:0.83157
    [36]	validation_0-auc:0.88621	validation_1-auc:0.83069
    [37]	validation_0-auc:0.88675	validation_1-auc:0.83129
    [38]	validation_0-auc:0.88776	validation_1-auc:0.83041
    [39]	validation_0-auc:0.88805	validation_1-auc:0.83066
    [40]	validation_0-auc:0.88857	validation_1-auc:0.82985
    [41]	validation_0-auc:0.88904	validation_1-auc:0.82941
    [42]	validation_0-auc:0.88976	validation_1-auc:0.82897
    [43]	validation_0-auc:0.89038	validation_1-auc:0.82860
    [44]	validation_0-auc:0.89040	validation_1-auc:0.82857
    [45]	validation_0-auc:0.89074	validation_1-auc:0.82821
    [46]	validation_0-auc:0.89095	validation_1-auc:0.82776
    [47]	validation_0-auc:0.89097	validation_1-auc:0.82796


    C:\Users\admin\anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


    [15:06:59] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "colsamples_bytree" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [0]	validation_0-auc:0.81540	validation_1-auc:0.80333
    [1]	validation_0-auc:0.82988	validation_1-auc:0.81844
    [2]	validation_0-auc:0.83719	validation_1-auc:0.81927
    [3]	validation_0-auc:0.83923	validation_1-auc:0.81795
    [4]	validation_0-auc:0.84766	validation_1-auc:0.82617
    [5]	validation_0-auc:0.85230	validation_1-auc:0.82934
    [6]	validation_0-auc:0.85552	validation_1-auc:0.83191
    [7]	validation_0-auc:0.85821	validation_1-auc:0.83278
    [8]	validation_0-auc:0.86110	validation_1-auc:0.83417
    [9]	validation_0-auc:0.86309	validation_1-auc:0.83485
    [10]	validation_0-auc:0.86722	validation_1-auc:0.83738
    [11]	validation_0-auc:0.86984	validation_1-auc:0.83728
    [12]	validation_0-auc:0.87247	validation_1-auc:0.84042
    [13]	validation_0-auc:0.87433	validation_1-auc:0.83984
    [14]	validation_0-auc:0.87531	validation_1-auc:0.84036
    [15]	validation_0-auc:0.87679	validation_1-auc:0.84015
    [16]	validation_0-auc:0.87827	validation_1-auc:0.84131
    [17]	validation_0-auc:0.87899	validation_1-auc:0.84145
    [18]	validation_0-auc:0.87962	validation_1-auc:0.84172
    [19]	validation_0-auc:0.88009	validation_1-auc:0.84181
    [20]	validation_0-auc:0.88092	validation_1-auc:0.84214
    [21]	validation_0-auc:0.88101	validation_1-auc:0.84220
    [22]	validation_0-auc:0.88126	validation_1-auc:0.84183
    [23]	validation_0-auc:0.88130	validation_1-auc:0.84152
    [24]	validation_0-auc:0.88180	validation_1-auc:0.84179
    [25]	validation_0-auc:0.88243	validation_1-auc:0.84161
    [26]	validation_0-auc:0.88251	validation_1-auc:0.84139
    [27]	validation_0-auc:0.88276	validation_1-auc:0.84098
    [28]	validation_0-auc:0.88327	validation_1-auc:0.84075
    [29]	validation_0-auc:0.88372	validation_1-auc:0.84118
    [30]	validation_0-auc:0.88424	validation_1-auc:0.84089
    [31]	validation_0-auc:0.88457	validation_1-auc:0.84072
    [32]	validation_0-auc:0.88572	validation_1-auc:0.84148
    [33]	validation_0-auc:0.88681	validation_1-auc:0.83925
    [34]	validation_0-auc:0.88701	validation_1-auc:0.83923
    [35]	validation_0-auc:0.88704	validation_1-auc:0.83937
    [36]	validation_0-auc:0.88723	validation_1-auc:0.83891
    [37]	validation_0-auc:0.88736	validation_1-auc:0.83828
    [38]	validation_0-auc:0.88824	validation_1-auc:0.83732
    [39]	validation_0-auc:0.88913	validation_1-auc:0.83553
    [40]	validation_0-auc:0.88923	validation_1-auc:0.83504
    [41]	validation_0-auc:0.88960	validation_1-auc:0.83479
    [42]	validation_0-auc:0.88971	validation_1-auc:0.83484
    [43]	validation_0-auc:0.88988	validation_1-auc:0.83479
    [44]	validation_0-auc:0.89029	validation_1-auc:0.83432
    [45]	validation_0-auc:0.89067	validation_1-auc:0.83426
    [46]	validation_0-auc:0.89052	validation_1-auc:0.83413
    [47]	validation_0-auc:0.89146	validation_1-auc:0.83376
    [48]	validation_0-auc:0.89148	validation_1-auc:0.83342
    [49]	validation_0-auc:0.89207	validation_1-auc:0.83350
    [50]	validation_0-auc:0.89262	validation_1-auc:0.83315


    C:\Users\admin\anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


    [15:07:06] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "colsamples_bytree" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [0]	validation_0-auc:0.82298	validation_1-auc:0.80807
    [1]	validation_0-auc:0.83122	validation_1-auc:0.81350
    [2]	validation_0-auc:0.83770	validation_1-auc:0.82016
    [3]	validation_0-auc:0.84037	validation_1-auc:0.82036
    [4]	validation_0-auc:0.84438	validation_1-auc:0.82282
    [5]	validation_0-auc:0.84728	validation_1-auc:0.82268
    [6]	validation_0-auc:0.85074	validation_1-auc:0.82916
    [7]	validation_0-auc:0.85626	validation_1-auc:0.82908
    [8]	validation_0-auc:0.86247	validation_1-auc:0.82979
    [9]	validation_0-auc:0.86488	validation_1-auc:0.83060
    [10]	validation_0-auc:0.86721	validation_1-auc:0.83031
    [11]	validation_0-auc:0.86923	validation_1-auc:0.83168
    [12]	validation_0-auc:0.87132	validation_1-auc:0.83148
    [13]	validation_0-auc:0.87335	validation_1-auc:0.83341
    [14]	validation_0-auc:0.87635	validation_1-auc:0.83367
    [15]	validation_0-auc:0.87767	validation_1-auc:0.83412
    [16]	validation_0-auc:0.87908	validation_1-auc:0.83417
    [17]	validation_0-auc:0.87984	validation_1-auc:0.83401
    [18]	validation_0-auc:0.88133	validation_1-auc:0.83476
    [19]	validation_0-auc:0.88225	validation_1-auc:0.83374
    [20]	validation_0-auc:0.88276	validation_1-auc:0.83390
    [21]	validation_0-auc:0.88319	validation_1-auc:0.83366
    [22]	validation_0-auc:0.88348	validation_1-auc:0.83380
    [23]	validation_0-auc:0.88362	validation_1-auc:0.83355
    [24]	validation_0-auc:0.88410	validation_1-auc:0.83297
    [25]	validation_0-auc:0.88444	validation_1-auc:0.83261
    [26]	validation_0-auc:0.88505	validation_1-auc:0.83276
    [27]	validation_0-auc:0.88555	validation_1-auc:0.83270
    [28]	validation_0-auc:0.88575	validation_1-auc:0.83232
    [29]	validation_0-auc:0.88602	validation_1-auc:0.83251
    [30]	validation_0-auc:0.88710	validation_1-auc:0.83213
    [31]	validation_0-auc:0.88736	validation_1-auc:0.83184
    [32]	validation_0-auc:0.88781	validation_1-auc:0.83168
    [33]	validation_0-auc:0.88789	validation_1-auc:0.83114
    [34]	validation_0-auc:0.88880	validation_1-auc:0.83109
    [35]	validation_0-auc:0.88941	validation_1-auc:0.83072
    [36]	validation_0-auc:0.88962	validation_1-auc:0.83012
    [37]	validation_0-auc:0.88982	validation_1-auc:0.82995
    [38]	validation_0-auc:0.89055	validation_1-auc:0.83012
    [39]	validation_0-auc:0.89113	validation_1-auc:0.83034
    [40]	validation_0-auc:0.89162	validation_1-auc:0.82992
    [41]	validation_0-auc:0.89165	validation_1-auc:0.82982
    [42]	validation_0-auc:0.89201	validation_1-auc:0.82988
    [43]	validation_0-auc:0.89215	validation_1-auc:0.82954
    [44]	validation_0-auc:0.89246	validation_1-auc:0.82929
    [45]	validation_0-auc:0.89282	validation_1-auc:0.82917
    [46]	validation_0-auc:0.89309	validation_1-auc:0.82927
    [47]	validation_0-auc:0.89334	validation_1-auc:0.82854


    C:\Users\admin\anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


    [15:07:13] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "colsamples_bytree" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [0]	validation_0-auc:0.81574	validation_1-auc:0.80177
    [1]	validation_0-auc:0.82540	validation_1-auc:0.81029
    [2]	validation_0-auc:0.82868	validation_1-auc:0.81288
    [3]	validation_0-auc:0.83075	validation_1-auc:0.81497
    [4]	validation_0-auc:0.83462	validation_1-auc:0.81745
    [5]	validation_0-auc:0.83689	validation_1-auc:0.81972
    [6]	validation_0-auc:0.84038	validation_1-auc:0.82364
    [7]	validation_0-auc:0.84344	validation_1-auc:0.82645
    [8]	validation_0-auc:0.84649	validation_1-auc:0.82893
    [9]	validation_0-auc:0.84847	validation_1-auc:0.82918
    [10]	validation_0-auc:0.85051	validation_1-auc:0.83116
    [11]	validation_0-auc:0.85273	validation_1-auc:0.83180
    [12]	validation_0-auc:0.85442	validation_1-auc:0.83355
    [13]	validation_0-auc:0.85649	validation_1-auc:0.83482
    [14]	validation_0-auc:0.85823	validation_1-auc:0.83621
    [15]	validation_0-auc:0.85954	validation_1-auc:0.83830
    [16]	validation_0-auc:0.86074	validation_1-auc:0.83913
    [17]	validation_0-auc:0.86165	validation_1-auc:0.83968
    [18]	validation_0-auc:0.86301	validation_1-auc:0.84009
    [19]	validation_0-auc:0.86368	validation_1-auc:0.84039
    [20]	validation_0-auc:0.86468	validation_1-auc:0.83951
    [21]	validation_0-auc:0.86507	validation_1-auc:0.83948
    [22]	validation_0-auc:0.86534	validation_1-auc:0.83950
    [23]	validation_0-auc:0.86576	validation_1-auc:0.83925
    [24]	validation_0-auc:0.86597	validation_1-auc:0.83934
    [25]	validation_0-auc:0.86601	validation_1-auc:0.83938
    [26]	validation_0-auc:0.86646	validation_1-auc:0.83931
    [27]	validation_0-auc:0.86740	validation_1-auc:0.83927
    [28]	validation_0-auc:0.86785	validation_1-auc:0.83883
    [29]	validation_0-auc:0.86792	validation_1-auc:0.83874
    [30]	validation_0-auc:0.86819	validation_1-auc:0.83836
    [31]	validation_0-auc:0.86837	validation_1-auc:0.83813
    [32]	validation_0-auc:0.86863	validation_1-auc:0.83813
    [33]	validation_0-auc:0.86884	validation_1-auc:0.83821
    [34]	validation_0-auc:0.86894	validation_1-auc:0.83812
    [35]	validation_0-auc:0.86956	validation_1-auc:0.83800
    [36]	validation_0-auc:0.87065	validation_1-auc:0.83748
    [37]	validation_0-auc:0.87130	validation_1-auc:0.83762
    [38]	validation_0-auc:0.87160	validation_1-auc:0.83788
    [39]	validation_0-auc:0.87162	validation_1-auc:0.83793
    [40]	validation_0-auc:0.87176	validation_1-auc:0.83768
    [41]	validation_0-auc:0.87184	validation_1-auc:0.83748
    [42]	validation_0-auc:0.87208	validation_1-auc:0.83708
    [43]	validation_0-auc:0.87237	validation_1-auc:0.83601
    [44]	validation_0-auc:0.87333	validation_1-auc:0.83501
    [45]	validation_0-auc:0.87382	validation_1-auc:0.83428
    [46]	validation_0-auc:0.87421	validation_1-auc:0.83381
    [47]	validation_0-auc:0.87443	validation_1-auc:0.83373
    [48]	validation_0-auc:0.87508	validation_1-auc:0.83378
    [49]	validation_0-auc:0.87565	validation_1-auc:0.83293


    C:\Users\admin\anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


    [15:07:19] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "colsamples_bytree" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [0]	validation_0-auc:0.80809	validation_1-auc:0.80034
    [1]	validation_0-auc:0.82399	validation_1-auc:0.81539
    [2]	validation_0-auc:0.82956	validation_1-auc:0.81486
    [3]	validation_0-auc:0.83139	validation_1-auc:0.81403
    [4]	validation_0-auc:0.83306	validation_1-auc:0.81454
    [5]	validation_0-auc:0.84142	validation_1-auc:0.82495
    [6]	validation_0-auc:0.84361	validation_1-auc:0.82716
    [7]	validation_0-auc:0.84624	validation_1-auc:0.83084
    [8]	validation_0-auc:0.84905	validation_1-auc:0.83408
    [9]	validation_0-auc:0.85071	validation_1-auc:0.83490
    [10]	validation_0-auc:0.85460	validation_1-auc:0.83796
    [11]	validation_0-auc:0.85558	validation_1-auc:0.83924
    [12]	validation_0-auc:0.85662	validation_1-auc:0.84002
    [13]	validation_0-auc:0.85819	validation_1-auc:0.84012
    [14]	validation_0-auc:0.85922	validation_1-auc:0.83988
    [15]	validation_0-auc:0.86078	validation_1-auc:0.84000
    [16]	validation_0-auc:0.86266	validation_1-auc:0.83954
    [17]	validation_0-auc:0.86361	validation_1-auc:0.83981
    [18]	validation_0-auc:0.86487	validation_1-auc:0.83898
    [19]	validation_0-auc:0.86534	validation_1-auc:0.83947
    [20]	validation_0-auc:0.86580	validation_1-auc:0.83974
    [21]	validation_0-auc:0.86619	validation_1-auc:0.83977
    [22]	validation_0-auc:0.86717	validation_1-auc:0.83980
    [23]	validation_0-auc:0.86748	validation_1-auc:0.84010
    [24]	validation_0-auc:0.86815	validation_1-auc:0.83852
    [25]	validation_0-auc:0.86823	validation_1-auc:0.83825
    [26]	validation_0-auc:0.86867	validation_1-auc:0.83812
    [27]	validation_0-auc:0.86914	validation_1-auc:0.83887
    [28]	validation_0-auc:0.86968	validation_1-auc:0.83838
    [29]	validation_0-auc:0.87068	validation_1-auc:0.83889
    [30]	validation_0-auc:0.87071	validation_1-auc:0.83860
    [31]	validation_0-auc:0.87131	validation_1-auc:0.83881
    [32]	validation_0-auc:0.87156	validation_1-auc:0.83848
    [33]	validation_0-auc:0.87252	validation_1-auc:0.83838
    [34]	validation_0-auc:0.87288	validation_1-auc:0.83836
    [35]	validation_0-auc:0.87313	validation_1-auc:0.83811
    [36]	validation_0-auc:0.87471	validation_1-auc:0.83786
    [37]	validation_0-auc:0.87481	validation_1-auc:0.83782
    [38]	validation_0-auc:0.87479	validation_1-auc:0.83746
    [39]	validation_0-auc:0.87558	validation_1-auc:0.83762
    [40]	validation_0-auc:0.87622	validation_1-auc:0.83708
    [41]	validation_0-auc:0.87645	validation_1-auc:0.83683
    [42]	validation_0-auc:0.87676	validation_1-auc:0.83642
    [43]	validation_0-auc:0.87717	validation_1-auc:0.83668


    C:\Users\admin\anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


    [15:07:24] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "colsamples_bytree" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [0]	validation_0-auc:0.81210	validation_1-auc:0.79994
    [1]	validation_0-auc:0.82550	validation_1-auc:0.81441
    [2]	validation_0-auc:0.82904	validation_1-auc:0.81624
    [3]	validation_0-auc:0.83135	validation_1-auc:0.81725
    [4]	validation_0-auc:0.83364	validation_1-auc:0.81958
    [5]	validation_0-auc:0.83526	validation_1-auc:0.81995
    [6]	validation_0-auc:0.84052	validation_1-auc:0.82608
    [7]	validation_0-auc:0.84169	validation_1-auc:0.82640
    [8]	validation_0-auc:0.84668	validation_1-auc:0.82951
    [9]	validation_0-auc:0.85018	validation_1-auc:0.83051
    [10]	validation_0-auc:0.85098	validation_1-auc:0.83139
    [11]	validation_0-auc:0.85283	validation_1-auc:0.83208
    [12]	validation_0-auc:0.85505	validation_1-auc:0.83195
    [13]	validation_0-auc:0.85650	validation_1-auc:0.83383
    [14]	validation_0-auc:0.85832	validation_1-auc:0.83448
    [15]	validation_0-auc:0.85956	validation_1-auc:0.83520
    [16]	validation_0-auc:0.86117	validation_1-auc:0.83456
    [17]	validation_0-auc:0.86261	validation_1-auc:0.83449
    [18]	validation_0-auc:0.86506	validation_1-auc:0.83349
    [19]	validation_0-auc:0.86595	validation_1-auc:0.83467
    [20]	validation_0-auc:0.86696	validation_1-auc:0.83440
    [21]	validation_0-auc:0.86797	validation_1-auc:0.83422
    [22]	validation_0-auc:0.86898	validation_1-auc:0.83369
    [23]	validation_0-auc:0.86978	validation_1-auc:0.83349
    [24]	validation_0-auc:0.87052	validation_1-auc:0.83261
    [25]	validation_0-auc:0.87188	validation_1-auc:0.83254
    [26]	validation_0-auc:0.87246	validation_1-auc:0.83303
    [27]	validation_0-auc:0.87279	validation_1-auc:0.83310
    [28]	validation_0-auc:0.87333	validation_1-auc:0.83366
    [29]	validation_0-auc:0.87356	validation_1-auc:0.83360
    [30]	validation_0-auc:0.87408	validation_1-auc:0.83312
    [31]	validation_0-auc:0.87426	validation_1-auc:0.83324
    [32]	validation_0-auc:0.87448	validation_1-auc:0.83285
    [33]	validation_0-auc:0.87456	validation_1-auc:0.83239
    [34]	validation_0-auc:0.87548	validation_1-auc:0.83268
    [35]	validation_0-auc:0.87692	validation_1-auc:0.83215
    [36]	validation_0-auc:0.87747	validation_1-auc:0.83166
    [37]	validation_0-auc:0.87830	validation_1-auc:0.83235
    [38]	validation_0-auc:0.87857	validation_1-auc:0.83224
    [39]	validation_0-auc:0.87890	validation_1-auc:0.83193
    [40]	validation_0-auc:0.87907	validation_1-auc:0.83212
    [41]	validation_0-auc:0.87940	validation_1-auc:0.83185
    [42]	validation_0-auc:0.87955	validation_1-auc:0.83175
    [43]	validation_0-auc:0.87985	validation_1-auc:0.83157
    [44]	validation_0-auc:0.88060	validation_1-auc:0.83189
    [45]	validation_0-auc:0.88067	validation_1-auc:0.83186


    C:\Users\admin\anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


    [15:07:30] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "colsamples_bytree" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [0]	validation_0-auc:0.81467	validation_1-auc:0.80080
    [1]	validation_0-auc:0.82467	validation_1-auc:0.81106
    [2]	validation_0-auc:0.82894	validation_1-auc:0.81460
    [3]	validation_0-auc:0.82965	validation_1-auc:0.81369
    [4]	validation_0-auc:0.83378	validation_1-auc:0.81819
    [5]	validation_0-auc:0.83661	validation_1-auc:0.82072
    [6]	validation_0-auc:0.83864	validation_1-auc:0.82244
    [7]	validation_0-auc:0.84405	validation_1-auc:0.82609
    [8]	validation_0-auc:0.84660	validation_1-auc:0.82685
    [9]	validation_0-auc:0.84855	validation_1-auc:0.82874
    [10]	validation_0-auc:0.84945	validation_1-auc:0.82948
    [11]	validation_0-auc:0.85203	validation_1-auc:0.83241
    [12]	validation_0-auc:0.85382	validation_1-auc:0.83378
    [13]	validation_0-auc:0.85483	validation_1-auc:0.83436
    [14]	validation_0-auc:0.85615	validation_1-auc:0.83478
    [15]	validation_0-auc:0.85671	validation_1-auc:0.83603
    [16]	validation_0-auc:0.85791	validation_1-auc:0.83570
    [17]	validation_0-auc:0.85915	validation_1-auc:0.83758
    [18]	validation_0-auc:0.86048	validation_1-auc:0.83673
    [19]	validation_0-auc:0.86134	validation_1-auc:0.83488
    [20]	validation_0-auc:0.86147	validation_1-auc:0.83540
    [21]	validation_0-auc:0.86236	validation_1-auc:0.83601
    [22]	validation_0-auc:0.86312	validation_1-auc:0.83590
    [23]	validation_0-auc:0.86404	validation_1-auc:0.83589
    [24]	validation_0-auc:0.86413	validation_1-auc:0.83588
    [25]	validation_0-auc:0.86457	validation_1-auc:0.83556
    [26]	validation_0-auc:0.86492	validation_1-auc:0.83554
    [27]	validation_0-auc:0.86581	validation_1-auc:0.83597
    [28]	validation_0-auc:0.86589	validation_1-auc:0.83582
    [29]	validation_0-auc:0.86608	validation_1-auc:0.83584
    [30]	validation_0-auc:0.86646	validation_1-auc:0.83564
    [31]	validation_0-auc:0.86680	validation_1-auc:0.83523
    [32]	validation_0-auc:0.86742	validation_1-auc:0.83504
    [33]	validation_0-auc:0.86765	validation_1-auc:0.83522
    [34]	validation_0-auc:0.86784	validation_1-auc:0.83497
    [35]	validation_0-auc:0.86803	validation_1-auc:0.83398
    [36]	validation_0-auc:0.86817	validation_1-auc:0.83403
    [37]	validation_0-auc:0.86813	validation_1-auc:0.83355
    [38]	validation_0-auc:0.86888	validation_1-auc:0.83290
    [39]	validation_0-auc:0.86898	validation_1-auc:0.83303
    [40]	validation_0-auc:0.86996	validation_1-auc:0.83286
    [41]	validation_0-auc:0.87016	validation_1-auc:0.83262
    [42]	validation_0-auc:0.87128	validation_1-auc:0.83234
    [43]	validation_0-auc:0.87132	validation_1-auc:0.83260
    [44]	validation_0-auc:0.87203	validation_1-auc:0.83215
    [45]	validation_0-auc:0.87223	validation_1-auc:0.83206
    [46]	validation_0-auc:0.87246	validation_1-auc:0.83211
    [47]	validation_0-auc:0.87342	validation_1-auc:0.83168


    C:\Users\admin\anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


    [15:07:35] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "colsamples_bytree" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [0]	validation_0-auc:0.81267	validation_1-auc:0.80112
    [1]	validation_0-auc:0.82759	validation_1-auc:0.81306
    [2]	validation_0-auc:0.83099	validation_1-auc:0.81471
    [3]	validation_0-auc:0.83497	validation_1-auc:0.82058
    [4]	validation_0-auc:0.83631	validation_1-auc:0.82090
    [5]	validation_0-auc:0.84166	validation_1-auc:0.82807
    [6]	validation_0-auc:0.84374	validation_1-auc:0.82842
    [7]	validation_0-auc:0.84539	validation_1-auc:0.83106
    [8]	validation_0-auc:0.84799	validation_1-auc:0.83523
    [9]	validation_0-auc:0.85035	validation_1-auc:0.83786
    [10]	validation_0-auc:0.85175	validation_1-auc:0.83851
    [11]	validation_0-auc:0.85362	validation_1-auc:0.83876
    [12]	validation_0-auc:0.85524	validation_1-auc:0.83980
    [13]	validation_0-auc:0.85709	validation_1-auc:0.84089
    [14]	validation_0-auc:0.85796	validation_1-auc:0.84129
    [15]	validation_0-auc:0.85958	validation_1-auc:0.84412
    [16]	validation_0-auc:0.86014	validation_1-auc:0.84456
    [17]	validation_0-auc:0.86141	validation_1-auc:0.84425
    [18]	validation_0-auc:0.86165	validation_1-auc:0.84433
    [19]	validation_0-auc:0.86221	validation_1-auc:0.84433
    [20]	validation_0-auc:0.86266	validation_1-auc:0.84403
    [21]	validation_0-auc:0.86385	validation_1-auc:0.84400
    [22]	validation_0-auc:0.86482	validation_1-auc:0.84421
    [23]	validation_0-auc:0.86491	validation_1-auc:0.84408
    [24]	validation_0-auc:0.86585	validation_1-auc:0.84386
    [25]	validation_0-auc:0.86726	validation_1-auc:0.84382
    [26]	validation_0-auc:0.86753	validation_1-auc:0.84380
    [27]	validation_0-auc:0.86767	validation_1-auc:0.84346
    [28]	validation_0-auc:0.86813	validation_1-auc:0.84377
    [29]	validation_0-auc:0.86814	validation_1-auc:0.84351
    [30]	validation_0-auc:0.86868	validation_1-auc:0.84385
    [31]	validation_0-auc:0.86890	validation_1-auc:0.84409
    [32]	validation_0-auc:0.86935	validation_1-auc:0.84411
    [33]	validation_0-auc:0.86975	validation_1-auc:0.84376
    [34]	validation_0-auc:0.87040	validation_1-auc:0.84363
    [35]	validation_0-auc:0.87062	validation_1-auc:0.84324
    [36]	validation_0-auc:0.87089	validation_1-auc:0.84268
    [37]	validation_0-auc:0.87158	validation_1-auc:0.84237
    [38]	validation_0-auc:0.87286	validation_1-auc:0.84211
    [39]	validation_0-auc:0.87317	validation_1-auc:0.84220
    [40]	validation_0-auc:0.87330	validation_1-auc:0.84198
    [41]	validation_0-auc:0.87356	validation_1-auc:0.84162
    [42]	validation_0-auc:0.87360	validation_1-auc:0.84155
    [43]	validation_0-auc:0.87382	validation_1-auc:0.84133
    [44]	validation_0-auc:0.87435	validation_1-auc:0.84094
    [45]	validation_0-auc:0.87429	validation_1-auc:0.84079


    C:\Users\admin\anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


    [15:07:41] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "colsamples_bytree" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [0]	validation_0-auc:0.81897	validation_1-auc:0.80631
    [1]	validation_0-auc:0.82510	validation_1-auc:0.81305
    [2]	validation_0-auc:0.82775	validation_1-auc:0.81513
    [3]	validation_0-auc:0.83074	validation_1-auc:0.81791
    [4]	validation_0-auc:0.83267	validation_1-auc:0.81753
    [5]	validation_0-auc:0.83487	validation_1-auc:0.82178
    [6]	validation_0-auc:0.84024	validation_1-auc:0.82724
    [7]	validation_0-auc:0.84425	validation_1-auc:0.82813
    [8]	validation_0-auc:0.84612	validation_1-auc:0.83280
    [9]	validation_0-auc:0.84972	validation_1-auc:0.83280
    [10]	validation_0-auc:0.85120	validation_1-auc:0.83198
    [11]	validation_0-auc:0.85270	validation_1-auc:0.83408
    [12]	validation_0-auc:0.85475	validation_1-auc:0.83382
    [13]	validation_0-auc:0.85627	validation_1-auc:0.83412
    [14]	validation_0-auc:0.85766	validation_1-auc:0.83472
    [15]	validation_0-auc:0.85900	validation_1-auc:0.83438
    [16]	validation_0-auc:0.86011	validation_1-auc:0.83483
    [17]	validation_0-auc:0.86062	validation_1-auc:0.83474
    [18]	validation_0-auc:0.86192	validation_1-auc:0.83488
    [19]	validation_0-auc:0.86296	validation_1-auc:0.83429
    [20]	validation_0-auc:0.86376	validation_1-auc:0.83418
    [21]	validation_0-auc:0.86479	validation_1-auc:0.83389
    [22]	validation_0-auc:0.86566	validation_1-auc:0.83368
    [23]	validation_0-auc:0.86626	validation_1-auc:0.83362
    [24]	validation_0-auc:0.86675	validation_1-auc:0.83419
    [25]	validation_0-auc:0.86720	validation_1-auc:0.83443
    [26]	validation_0-auc:0.86804	validation_1-auc:0.83452
    [27]	validation_0-auc:0.86852	validation_1-auc:0.83491
    [28]	validation_0-auc:0.86904	validation_1-auc:0.83447
    [29]	validation_0-auc:0.87063	validation_1-auc:0.83339
    [30]	validation_0-auc:0.87091	validation_1-auc:0.83365
    [31]	validation_0-auc:0.87228	validation_1-auc:0.83280
    [32]	validation_0-auc:0.87234	validation_1-auc:0.83263
    [33]	validation_0-auc:0.87272	validation_1-auc:0.83223
    [34]	validation_0-auc:0.87293	validation_1-auc:0.83213
    [35]	validation_0-auc:0.87312	validation_1-auc:0.83179
    [36]	validation_0-auc:0.87336	validation_1-auc:0.83159
    [37]	validation_0-auc:0.87355	validation_1-auc:0.83155
    [38]	validation_0-auc:0.87398	validation_1-auc:0.83126
    [39]	validation_0-auc:0.87433	validation_1-auc:0.83090
    [40]	validation_0-auc:0.87461	validation_1-auc:0.83041
    [41]	validation_0-auc:0.87495	validation_1-auc:0.83058
    [42]	validation_0-auc:0.87545	validation_1-auc:0.83025
    [43]	validation_0-auc:0.87551	validation_1-auc:0.83018
    [44]	validation_0-auc:0.87587	validation_1-auc:0.83058
    [45]	validation_0-auc:0.87638	validation_1-auc:0.83062
    [46]	validation_0-auc:0.87732	validation_1-auc:0.82946
    [47]	validation_0-auc:0.87760	validation_1-auc:0.82974
    [48]	validation_0-auc:0.87783	validation_1-auc:0.82965
    [49]	validation_0-auc:0.87819	validation_1-auc:0.82896
    [50]	validation_0-auc:0.87832	validation_1-auc:0.82860
    [51]	validation_0-auc:0.87855	validation_1-auc:0.82865
    [52]	validation_0-auc:0.87914	validation_1-auc:0.82864
    [53]	validation_0-auc:0.87937	validation_1-auc:0.82831
    [54]	validation_0-auc:0.87975	validation_1-auc:0.82803
    [55]	validation_0-auc:0.88018	validation_1-auc:0.82852
    [56]	validation_0-auc:0.88026	validation_1-auc:0.82826


    C:\Users\admin\anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


    [15:07:47] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "colsamples_bytree" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [0]	validation_0-auc:0.82483	validation_1-auc:0.80939
    [1]	validation_0-auc:0.83013	validation_1-auc:0.80990
    [2]	validation_0-auc:0.83360	validation_1-auc:0.81395
    [3]	validation_0-auc:0.83644	validation_1-auc:0.81362
    [4]	validation_0-auc:0.84346	validation_1-auc:0.81794
    [5]	validation_0-auc:0.84742	validation_1-auc:0.82094
    [6]	validation_0-auc:0.85282	validation_1-auc:0.82503
    [7]	validation_0-auc:0.85855	validation_1-auc:0.82940
    [8]	validation_0-auc:0.86260	validation_1-auc:0.82851
    [9]	validation_0-auc:0.86553	validation_1-auc:0.83110
    [10]	validation_0-auc:0.86882	validation_1-auc:0.83099
    [11]	validation_0-auc:0.87167	validation_1-auc:0.83240
    [12]	validation_0-auc:0.87447	validation_1-auc:0.83186
    [13]	validation_0-auc:0.87639	validation_1-auc:0.83144
    [14]	validation_0-auc:0.87763	validation_1-auc:0.83091
    [15]	validation_0-auc:0.87946	validation_1-auc:0.83345
    [16]	validation_0-auc:0.88112	validation_1-auc:0.83342
    [17]	validation_0-auc:0.88198	validation_1-auc:0.83373
    [18]	validation_0-auc:0.88330	validation_1-auc:0.83340
    [19]	validation_0-auc:0.88567	validation_1-auc:0.83390
    [20]	validation_0-auc:0.88588	validation_1-auc:0.83373
    [21]	validation_0-auc:0.88716	validation_1-auc:0.83240
    [22]	validation_0-auc:0.88791	validation_1-auc:0.83177
    [23]	validation_0-auc:0.88855	validation_1-auc:0.83211
    [24]	validation_0-auc:0.88887	validation_1-auc:0.83215
    [25]	validation_0-auc:0.88928	validation_1-auc:0.83215
    [26]	validation_0-auc:0.88953	validation_1-auc:0.83235
    [27]	validation_0-auc:0.88958	validation_1-auc:0.83252
    [28]	validation_0-auc:0.88960	validation_1-auc:0.83247
    [29]	validation_0-auc:0.89014	validation_1-auc:0.83173
    [30]	validation_0-auc:0.89045	validation_1-auc:0.83123
    [31]	validation_0-auc:0.89131	validation_1-auc:0.83141
    [32]	validation_0-auc:0.89179	validation_1-auc:0.83088
    [33]	validation_0-auc:0.89199	validation_1-auc:0.83074
    [34]	validation_0-auc:0.89237	validation_1-auc:0.82990
    [35]	validation_0-auc:0.89232	validation_1-auc:0.83001
    [36]	validation_0-auc:0.89243	validation_1-auc:0.83012
    [37]	validation_0-auc:0.89289	validation_1-auc:0.83034
    [38]	validation_0-auc:0.89308	validation_1-auc:0.83047
    [39]	validation_0-auc:0.89336	validation_1-auc:0.83035
    [40]	validation_0-auc:0.89415	validation_1-auc:0.82991
    [41]	validation_0-auc:0.89513	validation_1-auc:0.82878
    [42]	validation_0-auc:0.89550	validation_1-auc:0.82868
    [43]	validation_0-auc:0.89617	validation_1-auc:0.82897
    [44]	validation_0-auc:0.89610	validation_1-auc:0.82891
    [45]	validation_0-auc:0.89638	validation_1-auc:0.82890
    [46]	validation_0-auc:0.89688	validation_1-auc:0.82795
    [47]	validation_0-auc:0.89720	validation_1-auc:0.82675
    [48]	validation_0-auc:0.89794	validation_1-auc:0.82668
    [49]	validation_0-auc:0.89807	validation_1-auc:0.82642


    C:\Users\admin\anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


    [15:07:54] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "colsamples_bytree" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [0]	validation_0-auc:0.81430	validation_1-auc:0.80241
    [1]	validation_0-auc:0.83069	validation_1-auc:0.81451
    [2]	validation_0-auc:0.83796	validation_1-auc:0.81684
    [3]	validation_0-auc:0.84068	validation_1-auc:0.81668
    [4]	validation_0-auc:0.84335	validation_1-auc:0.81661
    [5]	validation_0-auc:0.85263	validation_1-auc:0.82610
    [6]	validation_0-auc:0.85698	validation_1-auc:0.82862
    [7]	validation_0-auc:0.86088	validation_1-auc:0.83080
    [8]	validation_0-auc:0.86482	validation_1-auc:0.83230
    [9]	validation_0-auc:0.86966	validation_1-auc:0.83181
    [10]	validation_0-auc:0.87207	validation_1-auc:0.83388
    [11]	validation_0-auc:0.87596	validation_1-auc:0.83439
    [12]	validation_0-auc:0.87907	validation_1-auc:0.83463
    [13]	validation_0-auc:0.88080	validation_1-auc:0.83517
    [14]	validation_0-auc:0.88249	validation_1-auc:0.83511
    [15]	validation_0-auc:0.88493	validation_1-auc:0.83479
    [16]	validation_0-auc:0.88651	validation_1-auc:0.83422
    [17]	validation_0-auc:0.88770	validation_1-auc:0.83381
    [18]	validation_0-auc:0.88812	validation_1-auc:0.83391
    [19]	validation_0-auc:0.88953	validation_1-auc:0.83347
    [20]	validation_0-auc:0.89026	validation_1-auc:0.83281
    [21]	validation_0-auc:0.89114	validation_1-auc:0.83292
    [22]	validation_0-auc:0.89176	validation_1-auc:0.83286
    [23]	validation_0-auc:0.89226	validation_1-auc:0.83281
    [24]	validation_0-auc:0.89271	validation_1-auc:0.83266
    [25]	validation_0-auc:0.89281	validation_1-auc:0.83240
    [26]	validation_0-auc:0.89295	validation_1-auc:0.83150
    [27]	validation_0-auc:0.89357	validation_1-auc:0.83174
    [28]	validation_0-auc:0.89469	validation_1-auc:0.83103
    [29]	validation_0-auc:0.89478	validation_1-auc:0.83102
    [30]	validation_0-auc:0.89568	validation_1-auc:0.83003
    [31]	validation_0-auc:0.89710	validation_1-auc:0.82869
    [32]	validation_0-auc:0.89772	validation_1-auc:0.82897
    [33]	validation_0-auc:0.89783	validation_1-auc:0.82930
    [34]	validation_0-auc:0.89829	validation_1-auc:0.82880
    [35]	validation_0-auc:0.89820	validation_1-auc:0.82889
    [36]	validation_0-auc:0.89849	validation_1-auc:0.82822
    [37]	validation_0-auc:0.89874	validation_1-auc:0.82781
    [38]	validation_0-auc:0.89921	validation_1-auc:0.82776
    [39]	validation_0-auc:0.89931	validation_1-auc:0.82742
    [40]	validation_0-auc:0.89958	validation_1-auc:0.82717
    [41]	validation_0-auc:0.89989	validation_1-auc:0.82679
    [42]	validation_0-auc:0.90007	validation_1-auc:0.82664
    [43]	validation_0-auc:0.90034	validation_1-auc:0.82666


    C:\Users\admin\anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


    [15:08:01] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "colsamples_bytree" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [0]	validation_0-auc:0.82346	validation_1-auc:0.80827
    [1]	validation_0-auc:0.83229	validation_1-auc:0.81327
    [2]	validation_0-auc:0.83781	validation_1-auc:0.81714
    [3]	validation_0-auc:0.84400	validation_1-auc:0.82109
    [4]	validation_0-auc:0.84735	validation_1-auc:0.82371
    [5]	validation_0-auc:0.85010	validation_1-auc:0.82568
    [6]	validation_0-auc:0.85285	validation_1-auc:0.82809
    [7]	validation_0-auc:0.85776	validation_1-auc:0.83283
    [8]	validation_0-auc:0.86287	validation_1-auc:0.83337
    [9]	validation_0-auc:0.86719	validation_1-auc:0.83305
    [10]	validation_0-auc:0.87058	validation_1-auc:0.83230
    [11]	validation_0-auc:0.87355	validation_1-auc:0.83134
    [12]	validation_0-auc:0.87600	validation_1-auc:0.83188
    [13]	validation_0-auc:0.87916	validation_1-auc:0.83188
    [14]	validation_0-auc:0.88075	validation_1-auc:0.83207
    [15]	validation_0-auc:0.88261	validation_1-auc:0.83149
    [16]	validation_0-auc:0.88417	validation_1-auc:0.83074
    [17]	validation_0-auc:0.88507	validation_1-auc:0.83097
    [18]	validation_0-auc:0.88599	validation_1-auc:0.83098
    [19]	validation_0-auc:0.88742	validation_1-auc:0.83025
    [20]	validation_0-auc:0.88823	validation_1-auc:0.83029
    [21]	validation_0-auc:0.88903	validation_1-auc:0.82953
    [22]	validation_0-auc:0.88919	validation_1-auc:0.82913
    [23]	validation_0-auc:0.89128	validation_1-auc:0.82913
    [24]	validation_0-auc:0.89148	validation_1-auc:0.82926
    [25]	validation_0-auc:0.89261	validation_1-auc:0.82915
    [26]	validation_0-auc:0.89308	validation_1-auc:0.82916
    [27]	validation_0-auc:0.89344	validation_1-auc:0.82900
    [28]	validation_0-auc:0.89394	validation_1-auc:0.82856
    [29]	validation_0-auc:0.89416	validation_1-auc:0.82833
    [30]	validation_0-auc:0.89436	validation_1-auc:0.82802
    [31]	validation_0-auc:0.89458	validation_1-auc:0.82809
    [32]	validation_0-auc:0.89578	validation_1-auc:0.82676
    [33]	validation_0-auc:0.89611	validation_1-auc:0.82676
    [34]	validation_0-auc:0.89621	validation_1-auc:0.82672
    [35]	validation_0-auc:0.89668	validation_1-auc:0.82628
    [36]	validation_0-auc:0.89689	validation_1-auc:0.82616
    [37]	validation_0-auc:0.89733	validation_1-auc:0.82598
    [38]	validation_0-auc:0.89751	validation_1-auc:0.82581


    C:\Users\admin\anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


    [15:08:07] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "colsamples_bytree" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [0]	validation_0-auc:0.82342	validation_1-auc:0.80583
    [1]	validation_0-auc:0.82923	validation_1-auc:0.81038
    [2]	validation_0-auc:0.83377	validation_1-auc:0.81499
    [3]	validation_0-auc:0.83813	validation_1-auc:0.81923
    [4]	validation_0-auc:0.84381	validation_1-auc:0.82164
    [5]	validation_0-auc:0.84556	validation_1-auc:0.82039
    [6]	validation_0-auc:0.85191	validation_1-auc:0.82747
    [7]	validation_0-auc:0.85585	validation_1-auc:0.82956
    [8]	validation_0-auc:0.85897	validation_1-auc:0.83026
    [9]	validation_0-auc:0.86094	validation_1-auc:0.83100
    [10]	validation_0-auc:0.86319	validation_1-auc:0.83167
    [11]	validation_0-auc:0.86571	validation_1-auc:0.83233
    [12]	validation_0-auc:0.86818	validation_1-auc:0.83203
    [13]	validation_0-auc:0.87043	validation_1-auc:0.83226
    [14]	validation_0-auc:0.87234	validation_1-auc:0.83186
    [15]	validation_0-auc:0.87375	validation_1-auc:0.83339
    [16]	validation_0-auc:0.87459	validation_1-auc:0.83390
    [17]	validation_0-auc:0.87505	validation_1-auc:0.83497
    [18]	validation_0-auc:0.87630	validation_1-auc:0.83457
    [19]	validation_0-auc:0.87776	validation_1-auc:0.83492
    [20]	validation_0-auc:0.87789	validation_1-auc:0.83479
    [21]	validation_0-auc:0.87899	validation_1-auc:0.83416
    [22]	validation_0-auc:0.87928	validation_1-auc:0.83374
    [23]	validation_0-auc:0.88035	validation_1-auc:0.83396
    [24]	validation_0-auc:0.88077	validation_1-auc:0.83408
    [25]	validation_0-auc:0.88168	validation_1-auc:0.83376
    [26]	validation_0-auc:0.88189	validation_1-auc:0.83311
    [27]	validation_0-auc:0.88252	validation_1-auc:0.83323
    [28]	validation_0-auc:0.88280	validation_1-auc:0.83358
    [29]	validation_0-auc:0.88371	validation_1-auc:0.83360
    [30]	validation_0-auc:0.88452	validation_1-auc:0.83240
    [31]	validation_0-auc:0.88507	validation_1-auc:0.83242
    [32]	validation_0-auc:0.88523	validation_1-auc:0.83241
    [33]	validation_0-auc:0.88524	validation_1-auc:0.83241
    [34]	validation_0-auc:0.88572	validation_1-auc:0.83191
    [35]	validation_0-auc:0.88594	validation_1-auc:0.83157
    [36]	validation_0-auc:0.88621	validation_1-auc:0.83069
    [37]	validation_0-auc:0.88675	validation_1-auc:0.83129
    [38]	validation_0-auc:0.88776	validation_1-auc:0.83041
    [39]	validation_0-auc:0.88805	validation_1-auc:0.83066
    [40]	validation_0-auc:0.88857	validation_1-auc:0.82985
    [41]	validation_0-auc:0.88904	validation_1-auc:0.82941
    [42]	validation_0-auc:0.88976	validation_1-auc:0.82897
    [43]	validation_0-auc:0.89038	validation_1-auc:0.82860
    [44]	validation_0-auc:0.89040	validation_1-auc:0.82857
    [45]	validation_0-auc:0.89074	validation_1-auc:0.82821
    [46]	validation_0-auc:0.89095	validation_1-auc:0.82776
    [47]	validation_0-auc:0.89097	validation_1-auc:0.82796


    C:\Users\admin\anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


    [15:08:14] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "colsamples_bytree" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [0]	validation_0-auc:0.81540	validation_1-auc:0.80333
    [1]	validation_0-auc:0.82988	validation_1-auc:0.81844
    [2]	validation_0-auc:0.83719	validation_1-auc:0.81927
    [3]	validation_0-auc:0.83923	validation_1-auc:0.81795
    [4]	validation_0-auc:0.84766	validation_1-auc:0.82617
    [5]	validation_0-auc:0.85230	validation_1-auc:0.82934
    [6]	validation_0-auc:0.85552	validation_1-auc:0.83191
    [7]	validation_0-auc:0.85821	validation_1-auc:0.83278
    [8]	validation_0-auc:0.86110	validation_1-auc:0.83417
    [9]	validation_0-auc:0.86309	validation_1-auc:0.83485
    [10]	validation_0-auc:0.86722	validation_1-auc:0.83738
    [11]	validation_0-auc:0.86984	validation_1-auc:0.83728
    [12]	validation_0-auc:0.87247	validation_1-auc:0.84042
    [13]	validation_0-auc:0.87433	validation_1-auc:0.83984
    [14]	validation_0-auc:0.87531	validation_1-auc:0.84036
    [15]	validation_0-auc:0.87679	validation_1-auc:0.84015
    [16]	validation_0-auc:0.87827	validation_1-auc:0.84131
    [17]	validation_0-auc:0.87899	validation_1-auc:0.84145
    [18]	validation_0-auc:0.87962	validation_1-auc:0.84172
    [19]	validation_0-auc:0.88009	validation_1-auc:0.84181
    [20]	validation_0-auc:0.88092	validation_1-auc:0.84214
    [21]	validation_0-auc:0.88101	validation_1-auc:0.84220
    [22]	validation_0-auc:0.88126	validation_1-auc:0.84183
    [23]	validation_0-auc:0.88130	validation_1-auc:0.84152
    [24]	validation_0-auc:0.88180	validation_1-auc:0.84179
    [25]	validation_0-auc:0.88243	validation_1-auc:0.84161
    [26]	validation_0-auc:0.88251	validation_1-auc:0.84139
    [27]	validation_0-auc:0.88276	validation_1-auc:0.84098
    [28]	validation_0-auc:0.88327	validation_1-auc:0.84075
    [29]	validation_0-auc:0.88372	validation_1-auc:0.84118
    [30]	validation_0-auc:0.88424	validation_1-auc:0.84089
    [31]	validation_0-auc:0.88457	validation_1-auc:0.84072
    [32]	validation_0-auc:0.88572	validation_1-auc:0.84148
    [33]	validation_0-auc:0.88681	validation_1-auc:0.83925
    [34]	validation_0-auc:0.88701	validation_1-auc:0.83923
    [35]	validation_0-auc:0.88704	validation_1-auc:0.83937
    [36]	validation_0-auc:0.88723	validation_1-auc:0.83891
    [37]	validation_0-auc:0.88736	validation_1-auc:0.83828
    [38]	validation_0-auc:0.88824	validation_1-auc:0.83732
    [39]	validation_0-auc:0.88913	validation_1-auc:0.83553
    [40]	validation_0-auc:0.88923	validation_1-auc:0.83504
    [41]	validation_0-auc:0.88960	validation_1-auc:0.83479
    [42]	validation_0-auc:0.88971	validation_1-auc:0.83484
    [43]	validation_0-auc:0.88988	validation_1-auc:0.83479
    [44]	validation_0-auc:0.89029	validation_1-auc:0.83432
    [45]	validation_0-auc:0.89067	validation_1-auc:0.83426
    [46]	validation_0-auc:0.89052	validation_1-auc:0.83413
    [47]	validation_0-auc:0.89146	validation_1-auc:0.83376
    [48]	validation_0-auc:0.89148	validation_1-auc:0.83342
    [49]	validation_0-auc:0.89207	validation_1-auc:0.83350
    [50]	validation_0-auc:0.89262	validation_1-auc:0.83315
    [51]	validation_0-auc:0.89296	validation_1-auc:0.83274


    C:\Users\admin\anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


    [15:08:21] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "colsamples_bytree" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [0]	validation_0-auc:0.82298	validation_1-auc:0.80807
    [1]	validation_0-auc:0.83122	validation_1-auc:0.81350
    [2]	validation_0-auc:0.83770	validation_1-auc:0.82016
    [3]	validation_0-auc:0.84037	validation_1-auc:0.82036
    [4]	validation_0-auc:0.84438	validation_1-auc:0.82282
    [5]	validation_0-auc:0.84728	validation_1-auc:0.82268
    [6]	validation_0-auc:0.85074	validation_1-auc:0.82916
    [7]	validation_0-auc:0.85626	validation_1-auc:0.82908
    [8]	validation_0-auc:0.86247	validation_1-auc:0.82979
    [9]	validation_0-auc:0.86488	validation_1-auc:0.83060
    [10]	validation_0-auc:0.86721	validation_1-auc:0.83031
    [11]	validation_0-auc:0.86923	validation_1-auc:0.83168
    [12]	validation_0-auc:0.87132	validation_1-auc:0.83148
    [13]	validation_0-auc:0.87335	validation_1-auc:0.83341
    [14]	validation_0-auc:0.87635	validation_1-auc:0.83367
    [15]	validation_0-auc:0.87767	validation_1-auc:0.83412
    [16]	validation_0-auc:0.87908	validation_1-auc:0.83417
    [17]	validation_0-auc:0.87984	validation_1-auc:0.83401
    [18]	validation_0-auc:0.88133	validation_1-auc:0.83476
    [19]	validation_0-auc:0.88225	validation_1-auc:0.83374
    [20]	validation_0-auc:0.88276	validation_1-auc:0.83390
    [21]	validation_0-auc:0.88319	validation_1-auc:0.83366
    [22]	validation_0-auc:0.88348	validation_1-auc:0.83380
    [23]	validation_0-auc:0.88362	validation_1-auc:0.83355
    [24]	validation_0-auc:0.88410	validation_1-auc:0.83297
    [25]	validation_0-auc:0.88444	validation_1-auc:0.83261
    [26]	validation_0-auc:0.88505	validation_1-auc:0.83276
    [27]	validation_0-auc:0.88555	validation_1-auc:0.83270
    [28]	validation_0-auc:0.88575	validation_1-auc:0.83232
    [29]	validation_0-auc:0.88602	validation_1-auc:0.83251
    [30]	validation_0-auc:0.88710	validation_1-auc:0.83213
    [31]	validation_0-auc:0.88736	validation_1-auc:0.83184
    [32]	validation_0-auc:0.88781	validation_1-auc:0.83168
    [33]	validation_0-auc:0.88789	validation_1-auc:0.83114
    [34]	validation_0-auc:0.88880	validation_1-auc:0.83109
    [35]	validation_0-auc:0.88941	validation_1-auc:0.83072
    [36]	validation_0-auc:0.88962	validation_1-auc:0.83012
    [37]	validation_0-auc:0.88982	validation_1-auc:0.82995
    [38]	validation_0-auc:0.89055	validation_1-auc:0.83012
    [39]	validation_0-auc:0.89113	validation_1-auc:0.83034
    [40]	validation_0-auc:0.89162	validation_1-auc:0.82992
    [41]	validation_0-auc:0.89165	validation_1-auc:0.82982
    [42]	validation_0-auc:0.89201	validation_1-auc:0.82988
    [43]	validation_0-auc:0.89215	validation_1-auc:0.82954
    [44]	validation_0-auc:0.89246	validation_1-auc:0.82929
    [45]	validation_0-auc:0.89282	validation_1-auc:0.82917
    [46]	validation_0-auc:0.89309	validation_1-auc:0.82927
    [47]	validation_0-auc:0.89334	validation_1-auc:0.82854


    C:\Users\admin\anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


    [15:08:28] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "colsamples_bytree" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [0]	validation_0-auc:0.81864	validation_1-auc:0.79881
    [1]	validation_0-auc:0.83198	validation_1-auc:0.81627
    [2]	validation_0-auc:0.83510	validation_1-auc:0.81907
    [3]	validation_0-auc:0.83783	validation_1-auc:0.82001
    [4]	validation_0-auc:0.83934	validation_1-auc:0.82147
    [5]	validation_0-auc:0.84160	validation_1-auc:0.82284
    [6]	validation_0-auc:0.84409	validation_1-auc:0.82698
    [7]	validation_0-auc:0.84764	validation_1-auc:0.82806
    [8]	validation_0-auc:0.85094	validation_1-auc:0.83185
    [9]	validation_0-auc:0.85352	validation_1-auc:0.83494
    [10]	validation_0-auc:0.85546	validation_1-auc:0.83492
    [11]	validation_0-auc:0.85689	validation_1-auc:0.83601
    [12]	validation_0-auc:0.85927	validation_1-auc:0.83636
    [13]	validation_0-auc:0.86134	validation_1-auc:0.83651
    [14]	validation_0-auc:0.86292	validation_1-auc:0.83732
    [15]	validation_0-auc:0.86466	validation_1-auc:0.83818
    [16]	validation_0-auc:0.86695	validation_1-auc:0.83821
    [17]	validation_0-auc:0.86828	validation_1-auc:0.83778
    [18]	validation_0-auc:0.87093	validation_1-auc:0.83742
    [19]	validation_0-auc:0.87290	validation_1-auc:0.83859
    [20]	validation_0-auc:0.87344	validation_1-auc:0.83867
    [21]	validation_0-auc:0.87415	validation_1-auc:0.83895
    [22]	validation_0-auc:0.87482	validation_1-auc:0.83894
    [23]	validation_0-auc:0.87543	validation_1-auc:0.83949
    [24]	validation_0-auc:0.87686	validation_1-auc:0.83895
    [25]	validation_0-auc:0.87945	validation_1-auc:0.83842
    [26]	validation_0-auc:0.87999	validation_1-auc:0.83827
    [27]	validation_0-auc:0.88084	validation_1-auc:0.83840
    [28]	validation_0-auc:0.88191	validation_1-auc:0.83823
    [29]	validation_0-auc:0.88215	validation_1-auc:0.83832
    [30]	validation_0-auc:0.88261	validation_1-auc:0.83810
    [31]	validation_0-auc:0.88286	validation_1-auc:0.83808
    [32]	validation_0-auc:0.88446	validation_1-auc:0.83821
    [33]	validation_0-auc:0.88552	validation_1-auc:0.83784
    [34]	validation_0-auc:0.88582	validation_1-auc:0.83753
    [35]	validation_0-auc:0.88711	validation_1-auc:0.83710
    [36]	validation_0-auc:0.88747	validation_1-auc:0.83725
    [37]	validation_0-auc:0.88802	validation_1-auc:0.83735
    [38]	validation_0-auc:0.88830	validation_1-auc:0.83723
    [39]	validation_0-auc:0.88845	validation_1-auc:0.83725
    [40]	validation_0-auc:0.88903	validation_1-auc:0.83734
    [41]	validation_0-auc:0.88964	validation_1-auc:0.83757
    [42]	validation_0-auc:0.88998	validation_1-auc:0.83747
    [43]	validation_0-auc:0.89059	validation_1-auc:0.83711
    [44]	validation_0-auc:0.89082	validation_1-auc:0.83707
    [45]	validation_0-auc:0.89088	validation_1-auc:0.83709
    [46]	validation_0-auc:0.89201	validation_1-auc:0.83681
    [47]	validation_0-auc:0.89320	validation_1-auc:0.83590
    [48]	validation_0-auc:0.89326	validation_1-auc:0.83580
    [49]	validation_0-auc:0.89342	validation_1-auc:0.83578
    [50]	validation_0-auc:0.89372	validation_1-auc:0.83586
    [51]	validation_0-auc:0.89418	validation_1-auc:0.83531
    [52]	validation_0-auc:0.89531	validation_1-auc:0.83503
    [53]	validation_0-auc:0.89627	validation_1-auc:0.83462
    최적의 파라미터: {'colsamples_bytree': 0.5, 'max_depth': 5, 'min_child_weight': 1}
    roc_auc_score: 0.8394866066112973



```python
### 다른 파라미터도 수행을 해봐야 합니다.
### 튜닝의 결과로 나온 하이퍼파라미터를 적용해서 학습
xgb_clf = XGBClassifier(n_estimators=1000, learning_rate=0.02, reg_alpha=0.03,
                        max_depth=5, min_child_weight=1, colsample_bytree=0.5,
                       random_state = 42)
# 200번 이상 수행한 상태에서 roc_auc 가 더이상 좋아지지 않으면 훈련 중지
xgb_clf.fit(X_train, y_train, early_stopping_rounds=200, eval_metric='auc',
           eval_set=[(X_train, y_train), (X_test, y_test)])

xgb_roc_auc_score = roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:,1], 
                                 average='micro')
print('roc_auc_score:', xgb_roc_auc_score)
```

    C:\Users\admin\anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


    [0]	validation_0-auc:0.71675	validation_1-auc:0.68527
    [1]	validation_0-auc:0.80460	validation_1-auc:0.78356
    [2]	validation_0-auc:0.82434	validation_1-auc:0.80652
    [3]	validation_0-auc:0.81679	validation_1-auc:0.79745
    [4]	validation_0-auc:0.81834	validation_1-auc:0.79729
    [5]	validation_0-auc:0.81785	validation_1-auc:0.79425
    [6]	validation_0-auc:0.81575	validation_1-auc:0.79298
    [7]	validation_0-auc:0.81435	validation_1-auc:0.79189
    [8]	validation_0-auc:0.81573	validation_1-auc:0.79402
    [9]	validation_0-auc:0.81569	validation_1-auc:0.79272
    [10]	validation_0-auc:0.81407	validation_1-auc:0.79187
    [11]	validation_0-auc:0.81782	validation_1-auc:0.79524
    [12]	validation_0-auc:0.81844	validation_1-auc:0.79503
    [13]	validation_0-auc:0.81769	validation_1-auc:0.79273
    [14]	validation_0-auc:0.82239	validation_1-auc:0.79873
    [15]	validation_0-auc:0.82362	validation_1-auc:0.80001
    [16]	validation_0-auc:0.82508	validation_1-auc:0.80182
    [17]	validation_0-auc:0.82409	validation_1-auc:0.79959
    [18]	validation_0-auc:0.82354	validation_1-auc:0.79804
    [19]	validation_0-auc:0.82522	validation_1-auc:0.80065
    [20]	validation_0-auc:0.82663	validation_1-auc:0.80231
    [21]	validation_0-auc:0.82562	validation_1-auc:0.80095
    [22]	validation_0-auc:0.82679	validation_1-auc:0.80296
    [23]	validation_0-auc:0.82594	validation_1-auc:0.80146
    [24]	validation_0-auc:0.82743	validation_1-auc:0.80308
    [25]	validation_0-auc:0.82717	validation_1-auc:0.80213
    [26]	validation_0-auc:0.82827	validation_1-auc:0.80355
    [27]	validation_0-auc:0.82945	validation_1-auc:0.80513
    [28]	validation_0-auc:0.83072	validation_1-auc:0.80640
    [29]	validation_0-auc:0.82995	validation_1-auc:0.80522
    [30]	validation_0-auc:0.82914	validation_1-auc:0.80422
    [31]	validation_0-auc:0.82879	validation_1-auc:0.80363
    [32]	validation_0-auc:0.82898	validation_1-auc:0.80331
    [33]	validation_0-auc:0.82968	validation_1-auc:0.80420
    [34]	validation_0-auc:0.82931	validation_1-auc:0.80371
    [35]	validation_0-auc:0.83060	validation_1-auc:0.80508
    [36]	validation_0-auc:0.83130	validation_1-auc:0.80626
    [37]	validation_0-auc:0.83076	validation_1-auc:0.80523
    [38]	validation_0-auc:0.83157	validation_1-auc:0.80690
    [39]	validation_0-auc:0.83178	validation_1-auc:0.80672
    [40]	validation_0-auc:0.83114	validation_1-auc:0.80583
    [41]	validation_0-auc:0.83110	validation_1-auc:0.80568
    [42]	validation_0-auc:0.83159	validation_1-auc:0.80632
    [43]	validation_0-auc:0.83143	validation_1-auc:0.80630
    [44]	validation_0-auc:0.83096	validation_1-auc:0.80589
    [45]	validation_0-auc:0.83058	validation_1-auc:0.80488
    [46]	validation_0-auc:0.83134	validation_1-auc:0.80612
    [47]	validation_0-auc:0.83232	validation_1-auc:0.80709
    [48]	validation_0-auc:0.83171	validation_1-auc:0.80630
    [49]	validation_0-auc:0.83154	validation_1-auc:0.80640
    [50]	validation_0-auc:0.83144	validation_1-auc:0.80581
    [51]	validation_0-auc:0.83137	validation_1-auc:0.80584
    [52]	validation_0-auc:0.83200	validation_1-auc:0.80645
    [53]	validation_0-auc:0.83298	validation_1-auc:0.80770
    [54]	validation_0-auc:0.83403	validation_1-auc:0.80882
    [55]	validation_0-auc:0.83387	validation_1-auc:0.80841
    [56]	validation_0-auc:0.83477	validation_1-auc:0.80968
    [57]	validation_0-auc:0.83413	validation_1-auc:0.80882
    [58]	validation_0-auc:0.83460	validation_1-auc:0.80960
    [59]	validation_0-auc:0.83437	validation_1-auc:0.80892
    [60]	validation_0-auc:0.83371	validation_1-auc:0.80805
    [61]	validation_0-auc:0.83449	validation_1-auc:0.80879
    [62]	validation_0-auc:0.83446	validation_1-auc:0.80862
    [63]	validation_0-auc:0.83387	validation_1-auc:0.80751
    [64]	validation_0-auc:0.83380	validation_1-auc:0.80739
    [65]	validation_0-auc:0.83497	validation_1-auc:0.80862
    [66]	validation_0-auc:0.83466	validation_1-auc:0.80829
    [67]	validation_0-auc:0.83422	validation_1-auc:0.80743
    [68]	validation_0-auc:0.83363	validation_1-auc:0.80691
    [69]	validation_0-auc:0.83371	validation_1-auc:0.80662
    [70]	validation_0-auc:0.83372	validation_1-auc:0.80633
    [71]	validation_0-auc:0.83435	validation_1-auc:0.80710
    [72]	validation_0-auc:0.83506	validation_1-auc:0.80802
    [73]	validation_0-auc:0.83459	validation_1-auc:0.80719
    [74]	validation_0-auc:0.83448	validation_1-auc:0.80699
    [75]	validation_0-auc:0.83545	validation_1-auc:0.80825
    [76]	validation_0-auc:0.83636	validation_1-auc:0.80924
    [77]	validation_0-auc:0.83703	validation_1-auc:0.81032
    [78]	validation_0-auc:0.83684	validation_1-auc:0.80991
    [79]	validation_0-auc:0.83769	validation_1-auc:0.81122
    [80]	validation_0-auc:0.83841	validation_1-auc:0.81226
    [81]	validation_0-auc:0.83821	validation_1-auc:0.81171
    [82]	validation_0-auc:0.83817	validation_1-auc:0.81147
    [83]	validation_0-auc:0.83782	validation_1-auc:0.81089
    [84]	validation_0-auc:0.83776	validation_1-auc:0.81034
    [85]	validation_0-auc:0.83851	validation_1-auc:0.81140
    [86]	validation_0-auc:0.83814	validation_1-auc:0.81089
    [87]	validation_0-auc:0.83781	validation_1-auc:0.81030
    [88]	validation_0-auc:0.83760	validation_1-auc:0.81001
    [89]	validation_0-auc:0.83732	validation_1-auc:0.80953
    [90]	validation_0-auc:0.83806	validation_1-auc:0.81061
    [91]	validation_0-auc:0.83774	validation_1-auc:0.80993
    [92]	validation_0-auc:0.83822	validation_1-auc:0.81071
    [93]	validation_0-auc:0.83811	validation_1-auc:0.81040
    [94]	validation_0-auc:0.83798	validation_1-auc:0.80993
    [95]	validation_0-auc:0.83900	validation_1-auc:0.81135
    [96]	validation_0-auc:0.83883	validation_1-auc:0.81100
    [97]	validation_0-auc:0.83882	validation_1-auc:0.81073
    [98]	validation_0-auc:0.83852	validation_1-auc:0.81028
    [99]	validation_0-auc:0.83835	validation_1-auc:0.81001
    [100]	validation_0-auc:0.83812	validation_1-auc:0.80964
    [101]	validation_0-auc:0.83796	validation_1-auc:0.80939
    [102]	validation_0-auc:0.83869	validation_1-auc:0.81051
    [103]	validation_0-auc:0.83847	validation_1-auc:0.81015
    [104]	validation_0-auc:0.83933	validation_1-auc:0.81128
    [105]	validation_0-auc:0.84005	validation_1-auc:0.81238
    [106]	validation_0-auc:0.83978	validation_1-auc:0.81200
    [107]	validation_0-auc:0.84064	validation_1-auc:0.81298
    [108]	validation_0-auc:0.84044	validation_1-auc:0.81276
    [109]	validation_0-auc:0.84029	validation_1-auc:0.81243
    [110]	validation_0-auc:0.84021	validation_1-auc:0.81210
    [111]	validation_0-auc:0.84018	validation_1-auc:0.81171
    [112]	validation_0-auc:0.84015	validation_1-auc:0.81141
    [113]	validation_0-auc:0.84005	validation_1-auc:0.81099
    [114]	validation_0-auc:0.84073	validation_1-auc:0.81208
    [115]	validation_0-auc:0.84141	validation_1-auc:0.81308
    [116]	validation_0-auc:0.84210	validation_1-auc:0.81386
    [117]	validation_0-auc:0.84273	validation_1-auc:0.81471
    [118]	validation_0-auc:0.84331	validation_1-auc:0.81550
    [119]	validation_0-auc:0.84366	validation_1-auc:0.81581
    [120]	validation_0-auc:0.84417	validation_1-auc:0.81622
    [121]	validation_0-auc:0.84461	validation_1-auc:0.81693
    [122]	validation_0-auc:0.84477	validation_1-auc:0.81675
    [123]	validation_0-auc:0.84546	validation_1-auc:0.81776
    [124]	validation_0-auc:0.84552	validation_1-auc:0.81734
    [125]	validation_0-auc:0.84541	validation_1-auc:0.81723
    [126]	validation_0-auc:0.84580	validation_1-auc:0.81812
    [127]	validation_0-auc:0.84606	validation_1-auc:0.81896
    [128]	validation_0-auc:0.84655	validation_1-auc:0.81976
    [129]	validation_0-auc:0.84653	validation_1-auc:0.81946
    [130]	validation_0-auc:0.84658	validation_1-auc:0.81942
    [131]	validation_0-auc:0.84691	validation_1-auc:0.82006
    [132]	validation_0-auc:0.84695	validation_1-auc:0.81990
    [133]	validation_0-auc:0.84733	validation_1-auc:0.82054
    [134]	validation_0-auc:0.84719	validation_1-auc:0.82017
    [135]	validation_0-auc:0.84762	validation_1-auc:0.82063
    [136]	validation_0-auc:0.84815	validation_1-auc:0.82117
    [137]	validation_0-auc:0.84809	validation_1-auc:0.82093
    [138]	validation_0-auc:0.84850	validation_1-auc:0.82131
    [139]	validation_0-auc:0.84838	validation_1-auc:0.82122
    [140]	validation_0-auc:0.84891	validation_1-auc:0.82170
    [141]	validation_0-auc:0.84956	validation_1-auc:0.82235
    [142]	validation_0-auc:0.84946	validation_1-auc:0.82200
    [143]	validation_0-auc:0.84933	validation_1-auc:0.82173
    [144]	validation_0-auc:0.84939	validation_1-auc:0.82178
    [145]	validation_0-auc:0.84980	validation_1-auc:0.82239
    [146]	validation_0-auc:0.85027	validation_1-auc:0.82299
    [147]	validation_0-auc:0.85006	validation_1-auc:0.82264
    [148]	validation_0-auc:0.85062	validation_1-auc:0.82342
    [149]	validation_0-auc:0.85100	validation_1-auc:0.82400
    [150]	validation_0-auc:0.85152	validation_1-auc:0.82437
    [151]	validation_0-auc:0.85153	validation_1-auc:0.82424
    [152]	validation_0-auc:0.85153	validation_1-auc:0.82420
    [153]	validation_0-auc:0.85150	validation_1-auc:0.82384
    [154]	validation_0-auc:0.85196	validation_1-auc:0.82429
    [155]	validation_0-auc:0.85235	validation_1-auc:0.82489
    [156]	validation_0-auc:0.85233	validation_1-auc:0.82470
    [157]	validation_0-auc:0.85265	validation_1-auc:0.82512
    [158]	validation_0-auc:0.85269	validation_1-auc:0.82495
    [159]	validation_0-auc:0.85261	validation_1-auc:0.82500
    [160]	validation_0-auc:0.85313	validation_1-auc:0.82546
    [161]	validation_0-auc:0.85324	validation_1-auc:0.82539
    [162]	validation_0-auc:0.85366	validation_1-auc:0.82560
    [163]	validation_0-auc:0.85351	validation_1-auc:0.82547
    [164]	validation_0-auc:0.85347	validation_1-auc:0.82530
    [165]	validation_0-auc:0.85355	validation_1-auc:0.82505
    [166]	validation_0-auc:0.85367	validation_1-auc:0.82490
    [167]	validation_0-auc:0.85409	validation_1-auc:0.82518
    [168]	validation_0-auc:0.85413	validation_1-auc:0.82515
    [169]	validation_0-auc:0.85452	validation_1-auc:0.82597
    [170]	validation_0-auc:0.85443	validation_1-auc:0.82550
    [171]	validation_0-auc:0.85447	validation_1-auc:0.82524
    [172]	validation_0-auc:0.85437	validation_1-auc:0.82510
    [173]	validation_0-auc:0.85438	validation_1-auc:0.82497
    [174]	validation_0-auc:0.85485	validation_1-auc:0.82550
    [175]	validation_0-auc:0.85491	validation_1-auc:0.82554
    [176]	validation_0-auc:0.85518	validation_1-auc:0.82592
    [177]	validation_0-auc:0.85578	validation_1-auc:0.82661
    [178]	validation_0-auc:0.85572	validation_1-auc:0.82633
    [179]	validation_0-auc:0.85565	validation_1-auc:0.82628
    [180]	validation_0-auc:0.85578	validation_1-auc:0.82637
    [181]	validation_0-auc:0.85585	validation_1-auc:0.82616
    [182]	validation_0-auc:0.85592	validation_1-auc:0.82581
    [183]	validation_0-auc:0.85595	validation_1-auc:0.82564
    [184]	validation_0-auc:0.85592	validation_1-auc:0.82557
    [185]	validation_0-auc:0.85634	validation_1-auc:0.82606
    [186]	validation_0-auc:0.85635	validation_1-auc:0.82596
    [187]	validation_0-auc:0.85673	validation_1-auc:0.82624
    [188]	validation_0-auc:0.85680	validation_1-auc:0.82631
    [189]	validation_0-auc:0.85722	validation_1-auc:0.82690
    [190]	validation_0-auc:0.85762	validation_1-auc:0.82766
    [191]	validation_0-auc:0.85765	validation_1-auc:0.82758
    [192]	validation_0-auc:0.85804	validation_1-auc:0.82805
    [193]	validation_0-auc:0.85841	validation_1-auc:0.82867
    [194]	validation_0-auc:0.85875	validation_1-auc:0.82912
    [195]	validation_0-auc:0.85899	validation_1-auc:0.82956
    [196]	validation_0-auc:0.85909	validation_1-auc:0.82939
    [197]	validation_0-auc:0.85916	validation_1-auc:0.82917
    [198]	validation_0-auc:0.85925	validation_1-auc:0.82900
    [199]	validation_0-auc:0.85966	validation_1-auc:0.82955
    [200]	validation_0-auc:0.85973	validation_1-auc:0.82948
    [201]	validation_0-auc:0.86000	validation_1-auc:0.82993
    [202]	validation_0-auc:0.86005	validation_1-auc:0.82984
    [203]	validation_0-auc:0.86033	validation_1-auc:0.83020
    [204]	validation_0-auc:0.86060	validation_1-auc:0.83068
    [205]	validation_0-auc:0.86067	validation_1-auc:0.83045
    [206]	validation_0-auc:0.86092	validation_1-auc:0.83089
    [207]	validation_0-auc:0.86095	validation_1-auc:0.83079
    [208]	validation_0-auc:0.86124	validation_1-auc:0.83118
    [209]	validation_0-auc:0.86124	validation_1-auc:0.83120
    [210]	validation_0-auc:0.86127	validation_1-auc:0.83093
    [211]	validation_0-auc:0.86131	validation_1-auc:0.83087
    [212]	validation_0-auc:0.86158	validation_1-auc:0.83129
    [213]	validation_0-auc:0.86193	validation_1-auc:0.83170
    [214]	validation_0-auc:0.86197	validation_1-auc:0.83155
    [215]	validation_0-auc:0.86205	validation_1-auc:0.83134
    [216]	validation_0-auc:0.86231	validation_1-auc:0.83165
    [217]	validation_0-auc:0.86260	validation_1-auc:0.83191
    [218]	validation_0-auc:0.86278	validation_1-auc:0.83212
    [219]	validation_0-auc:0.86302	validation_1-auc:0.83234
    [220]	validation_0-auc:0.86311	validation_1-auc:0.83224
    [221]	validation_0-auc:0.86322	validation_1-auc:0.83205
    [222]	validation_0-auc:0.86329	validation_1-auc:0.83198
    [223]	validation_0-auc:0.86337	validation_1-auc:0.83189
    [224]	validation_0-auc:0.86349	validation_1-auc:0.83182
    [225]	validation_0-auc:0.86377	validation_1-auc:0.83201
    [226]	validation_0-auc:0.86391	validation_1-auc:0.83187
    [227]	validation_0-auc:0.86397	validation_1-auc:0.83163
    [228]	validation_0-auc:0.86422	validation_1-auc:0.83195
    [229]	validation_0-auc:0.86428	validation_1-auc:0.83191
    [230]	validation_0-auc:0.86434	validation_1-auc:0.83182
    [231]	validation_0-auc:0.86448	validation_1-auc:0.83207
    [232]	validation_0-auc:0.86460	validation_1-auc:0.83208
    [233]	validation_0-auc:0.86481	validation_1-auc:0.83227
    [234]	validation_0-auc:0.86505	validation_1-auc:0.83243
    [235]	validation_0-auc:0.86526	validation_1-auc:0.83261
    [236]	validation_0-auc:0.86538	validation_1-auc:0.83278
    [237]	validation_0-auc:0.86552	validation_1-auc:0.83282
    [238]	validation_0-auc:0.86568	validation_1-auc:0.83309
    [239]	validation_0-auc:0.86580	validation_1-auc:0.83344
    [240]	validation_0-auc:0.86588	validation_1-auc:0.83332
    [241]	validation_0-auc:0.86609	validation_1-auc:0.83348
    [242]	validation_0-auc:0.86621	validation_1-auc:0.83331
    [243]	validation_0-auc:0.86629	validation_1-auc:0.83329
    [244]	validation_0-auc:0.86633	validation_1-auc:0.83328
    [245]	validation_0-auc:0.86649	validation_1-auc:0.83346
    [246]	validation_0-auc:0.86656	validation_1-auc:0.83333
    [247]	validation_0-auc:0.86674	validation_1-auc:0.83350
    [248]	validation_0-auc:0.86689	validation_1-auc:0.83371
    [249]	validation_0-auc:0.86691	validation_1-auc:0.83377
    [250]	validation_0-auc:0.86711	validation_1-auc:0.83381
    [251]	validation_0-auc:0.86725	validation_1-auc:0.83381
    [252]	validation_0-auc:0.86740	validation_1-auc:0.83396
    [253]	validation_0-auc:0.86745	validation_1-auc:0.83396
    [254]	validation_0-auc:0.86762	validation_1-auc:0.83415
    [255]	validation_0-auc:0.86770	validation_1-auc:0.83415
    [256]	validation_0-auc:0.86778	validation_1-auc:0.83416
    [257]	validation_0-auc:0.86788	validation_1-auc:0.83411
    [258]	validation_0-auc:0.86809	validation_1-auc:0.83431
    [259]	validation_0-auc:0.86819	validation_1-auc:0.83428
    [260]	validation_0-auc:0.86838	validation_1-auc:0.83445
    [261]	validation_0-auc:0.86855	validation_1-auc:0.83469
    [262]	validation_0-auc:0.86872	validation_1-auc:0.83468
    [263]	validation_0-auc:0.86884	validation_1-auc:0.83487
    [264]	validation_0-auc:0.86891	validation_1-auc:0.83490
    [265]	validation_0-auc:0.86904	validation_1-auc:0.83505
    [266]	validation_0-auc:0.86911	validation_1-auc:0.83503
    [267]	validation_0-auc:0.86917	validation_1-auc:0.83506
    [268]	validation_0-auc:0.86940	validation_1-auc:0.83526
    [269]	validation_0-auc:0.86943	validation_1-auc:0.83523
    [270]	validation_0-auc:0.86957	validation_1-auc:0.83514
    [271]	validation_0-auc:0.86972	validation_1-auc:0.83531
    [272]	validation_0-auc:0.86986	validation_1-auc:0.83553
    [273]	validation_0-auc:0.86998	validation_1-auc:0.83576
    [274]	validation_0-auc:0.87004	validation_1-auc:0.83576
    [275]	validation_0-auc:0.87016	validation_1-auc:0.83578
    [276]	validation_0-auc:0.87019	validation_1-auc:0.83575
    [277]	validation_0-auc:0.87032	validation_1-auc:0.83577
    [278]	validation_0-auc:0.87042	validation_1-auc:0.83576
    [279]	validation_0-auc:0.87054	validation_1-auc:0.83574
    [280]	validation_0-auc:0.87079	validation_1-auc:0.83590
    [281]	validation_0-auc:0.87097	validation_1-auc:0.83604
    [282]	validation_0-auc:0.87113	validation_1-auc:0.83622
    [283]	validation_0-auc:0.87128	validation_1-auc:0.83641
    [284]	validation_0-auc:0.87139	validation_1-auc:0.83656
    [285]	validation_0-auc:0.87150	validation_1-auc:0.83655
    [286]	validation_0-auc:0.87158	validation_1-auc:0.83671
    [287]	validation_0-auc:0.87169	validation_1-auc:0.83672
    [288]	validation_0-auc:0.87178	validation_1-auc:0.83673
    [289]	validation_0-auc:0.87189	validation_1-auc:0.83666
    [290]	validation_0-auc:0.87192	validation_1-auc:0.83665
    [291]	validation_0-auc:0.87198	validation_1-auc:0.83667
    [292]	validation_0-auc:0.87202	validation_1-auc:0.83668
    [293]	validation_0-auc:0.87205	validation_1-auc:0.83664
    [294]	validation_0-auc:0.87215	validation_1-auc:0.83680
    [295]	validation_0-auc:0.87235	validation_1-auc:0.83686
    [296]	validation_0-auc:0.87248	validation_1-auc:0.83680
    [297]	validation_0-auc:0.87254	validation_1-auc:0.83680
    [298]	validation_0-auc:0.87260	validation_1-auc:0.83679
    [299]	validation_0-auc:0.87271	validation_1-auc:0.83691
    [300]	validation_0-auc:0.87280	validation_1-auc:0.83713
    [301]	validation_0-auc:0.87289	validation_1-auc:0.83713
    [302]	validation_0-auc:0.87297	validation_1-auc:0.83728
    [303]	validation_0-auc:0.87307	validation_1-auc:0.83729
    [304]	validation_0-auc:0.87326	validation_1-auc:0.83740
    [305]	validation_0-auc:0.87345	validation_1-auc:0.83766
    [306]	validation_0-auc:0.87362	validation_1-auc:0.83782
    [307]	validation_0-auc:0.87367	validation_1-auc:0.83779
    [308]	validation_0-auc:0.87369	validation_1-auc:0.83782
    [309]	validation_0-auc:0.87372	validation_1-auc:0.83784
    [310]	validation_0-auc:0.87387	validation_1-auc:0.83782
    [311]	validation_0-auc:0.87393	validation_1-auc:0.83783
    [312]	validation_0-auc:0.87396	validation_1-auc:0.83789
    [313]	validation_0-auc:0.87408	validation_1-auc:0.83813
    [314]	validation_0-auc:0.87418	validation_1-auc:0.83809
    [315]	validation_0-auc:0.87423	validation_1-auc:0.83813
    [316]	validation_0-auc:0.87441	validation_1-auc:0.83823
    [317]	validation_0-auc:0.87452	validation_1-auc:0.83839
    [318]	validation_0-auc:0.87469	validation_1-auc:0.83856
    [319]	validation_0-auc:0.87474	validation_1-auc:0.83857
    [320]	validation_0-auc:0.87481	validation_1-auc:0.83870
    [321]	validation_0-auc:0.87486	validation_1-auc:0.83881
    [322]	validation_0-auc:0.87494	validation_1-auc:0.83899
    [323]	validation_0-auc:0.87498	validation_1-auc:0.83899
    [324]	validation_0-auc:0.87501	validation_1-auc:0.83907
    [325]	validation_0-auc:0.87512	validation_1-auc:0.83927
    [326]	validation_0-auc:0.87519	validation_1-auc:0.83931
    [327]	validation_0-auc:0.87522	validation_1-auc:0.83932
    [328]	validation_0-auc:0.87526	validation_1-auc:0.83933
    [329]	validation_0-auc:0.87539	validation_1-auc:0.83926
    [330]	validation_0-auc:0.87548	validation_1-auc:0.83935
    [331]	validation_0-auc:0.87561	validation_1-auc:0.83940
    [332]	validation_0-auc:0.87572	validation_1-auc:0.83933
    [333]	validation_0-auc:0.87584	validation_1-auc:0.83930
    [334]	validation_0-auc:0.87593	validation_1-auc:0.83934
    [335]	validation_0-auc:0.87601	validation_1-auc:0.83936
    [336]	validation_0-auc:0.87610	validation_1-auc:0.83949
    [337]	validation_0-auc:0.87614	validation_1-auc:0.83949
    [338]	validation_0-auc:0.87617	validation_1-auc:0.83950
    [339]	validation_0-auc:0.87621	validation_1-auc:0.83952
    [340]	validation_0-auc:0.87631	validation_1-auc:0.83957
    [341]	validation_0-auc:0.87638	validation_1-auc:0.83974
    [342]	validation_0-auc:0.87650	validation_1-auc:0.83979
    [343]	validation_0-auc:0.87653	validation_1-auc:0.83983
    [344]	validation_0-auc:0.87657	validation_1-auc:0.83990
    [345]	validation_0-auc:0.87667	validation_1-auc:0.83991
    [346]	validation_0-auc:0.87671	validation_1-auc:0.83994
    [347]	validation_0-auc:0.87681	validation_1-auc:0.83998
    [348]	validation_0-auc:0.87691	validation_1-auc:0.83999
    [349]	validation_0-auc:0.87703	validation_1-auc:0.84005
    [350]	validation_0-auc:0.87710	validation_1-auc:0.84006
    [351]	validation_0-auc:0.87722	validation_1-auc:0.84019
    [352]	validation_0-auc:0.87728	validation_1-auc:0.84026
    [353]	validation_0-auc:0.87733	validation_1-auc:0.84030
    [354]	validation_0-auc:0.87735	validation_1-auc:0.84039
    [355]	validation_0-auc:0.87744	validation_1-auc:0.84042
    [356]	validation_0-auc:0.87748	validation_1-auc:0.84042
    [357]	validation_0-auc:0.87757	validation_1-auc:0.84045
    [358]	validation_0-auc:0.87761	validation_1-auc:0.84052
    [359]	validation_0-auc:0.87770	validation_1-auc:0.84055
    [360]	validation_0-auc:0.87776	validation_1-auc:0.84061
    [361]	validation_0-auc:0.87780	validation_1-auc:0.84069
    [362]	validation_0-auc:0.87783	validation_1-auc:0.84065
    [363]	validation_0-auc:0.87791	validation_1-auc:0.84075
    [364]	validation_0-auc:0.87809	validation_1-auc:0.84070
    [365]	validation_0-auc:0.87815	validation_1-auc:0.84078
    [366]	validation_0-auc:0.87822	validation_1-auc:0.84086
    [367]	validation_0-auc:0.87834	validation_1-auc:0.84090
    [368]	validation_0-auc:0.87837	validation_1-auc:0.84092
    [369]	validation_0-auc:0.87843	validation_1-auc:0.84101
    [370]	validation_0-auc:0.87848	validation_1-auc:0.84099
    [371]	validation_0-auc:0.87852	validation_1-auc:0.84097
    [372]	validation_0-auc:0.87859	validation_1-auc:0.84098
    [373]	validation_0-auc:0.87872	validation_1-auc:0.84098
    [374]	validation_0-auc:0.87878	validation_1-auc:0.84096
    [375]	validation_0-auc:0.87883	validation_1-auc:0.84096
    [376]	validation_0-auc:0.87888	validation_1-auc:0.84096
    [377]	validation_0-auc:0.87897	validation_1-auc:0.84103
    [378]	validation_0-auc:0.87902	validation_1-auc:0.84109
    [379]	validation_0-auc:0.87909	validation_1-auc:0.84104
    [380]	validation_0-auc:0.87915	validation_1-auc:0.84118
    [381]	validation_0-auc:0.87919	validation_1-auc:0.84125
    [382]	validation_0-auc:0.87925	validation_1-auc:0.84131
    [383]	validation_0-auc:0.87929	validation_1-auc:0.84128
    [384]	validation_0-auc:0.87934	validation_1-auc:0.84142
    [385]	validation_0-auc:0.87940	validation_1-auc:0.84139
    [386]	validation_0-auc:0.87946	validation_1-auc:0.84137
    [387]	validation_0-auc:0.87960	validation_1-auc:0.84133
    [388]	validation_0-auc:0.87964	validation_1-auc:0.84140
    [389]	validation_0-auc:0.87965	validation_1-auc:0.84148
    [390]	validation_0-auc:0.87978	validation_1-auc:0.84147
    [391]	validation_0-auc:0.87982	validation_1-auc:0.84155
    [392]	validation_0-auc:0.87987	validation_1-auc:0.84155
    [393]	validation_0-auc:0.87990	validation_1-auc:0.84152
    [394]	validation_0-auc:0.88004	validation_1-auc:0.84151
    [395]	validation_0-auc:0.88006	validation_1-auc:0.84151
    [396]	validation_0-auc:0.88010	validation_1-auc:0.84150
    [397]	validation_0-auc:0.88019	validation_1-auc:0.84154
    [398]	validation_0-auc:0.88027	validation_1-auc:0.84154
    [399]	validation_0-auc:0.88033	validation_1-auc:0.84160
    [400]	validation_0-auc:0.88040	validation_1-auc:0.84168
    [401]	validation_0-auc:0.88054	validation_1-auc:0.84170
    [402]	validation_0-auc:0.88064	validation_1-auc:0.84170
    [403]	validation_0-auc:0.88072	validation_1-auc:0.84168
    [404]	validation_0-auc:0.88085	validation_1-auc:0.84171
    [405]	validation_0-auc:0.88090	validation_1-auc:0.84175
    [406]	validation_0-auc:0.88097	validation_1-auc:0.84183
    [407]	validation_0-auc:0.88102	validation_1-auc:0.84187
    [408]	validation_0-auc:0.88109	validation_1-auc:0.84187
    [409]	validation_0-auc:0.88116	validation_1-auc:0.84187
    [410]	validation_0-auc:0.88123	validation_1-auc:0.84191
    [411]	validation_0-auc:0.88128	validation_1-auc:0.84191
    [412]	validation_0-auc:0.88130	validation_1-auc:0.84189
    [413]	validation_0-auc:0.88132	validation_1-auc:0.84187
    [414]	validation_0-auc:0.88136	validation_1-auc:0.84187
    [415]	validation_0-auc:0.88138	validation_1-auc:0.84189
    [416]	validation_0-auc:0.88145	validation_1-auc:0.84192
    [417]	validation_0-auc:0.88150	validation_1-auc:0.84196
    [418]	validation_0-auc:0.88153	validation_1-auc:0.84196
    [419]	validation_0-auc:0.88163	validation_1-auc:0.84195
    [420]	validation_0-auc:0.88169	validation_1-auc:0.84200
    [421]	validation_0-auc:0.88179	validation_1-auc:0.84199
    [422]	validation_0-auc:0.88186	validation_1-auc:0.84197
    [423]	validation_0-auc:0.88187	validation_1-auc:0.84197
    [424]	validation_0-auc:0.88194	validation_1-auc:0.84200
    [425]	validation_0-auc:0.88203	validation_1-auc:0.84203
    [426]	validation_0-auc:0.88210	validation_1-auc:0.84199
    [427]	validation_0-auc:0.88215	validation_1-auc:0.84201
    [428]	validation_0-auc:0.88224	validation_1-auc:0.84206
    [429]	validation_0-auc:0.88231	validation_1-auc:0.84209
    [430]	validation_0-auc:0.88242	validation_1-auc:0.84207
    [431]	validation_0-auc:0.88244	validation_1-auc:0.84209
    [432]	validation_0-auc:0.88245	validation_1-auc:0.84209
    [433]	validation_0-auc:0.88254	validation_1-auc:0.84213
    [434]	validation_0-auc:0.88257	validation_1-auc:0.84210
    [435]	validation_0-auc:0.88263	validation_1-auc:0.84211
    [436]	validation_0-auc:0.88265	validation_1-auc:0.84210
    [437]	validation_0-auc:0.88267	validation_1-auc:0.84210
    [438]	validation_0-auc:0.88271	validation_1-auc:0.84209
    [439]	validation_0-auc:0.88281	validation_1-auc:0.84206
    [440]	validation_0-auc:0.88287	validation_1-auc:0.84204
    [441]	validation_0-auc:0.88293	validation_1-auc:0.84202
    [442]	validation_0-auc:0.88301	validation_1-auc:0.84206
    [443]	validation_0-auc:0.88308	validation_1-auc:0.84210
    [444]	validation_0-auc:0.88309	validation_1-auc:0.84209
    [445]	validation_0-auc:0.88316	validation_1-auc:0.84205
    [446]	validation_0-auc:0.88322	validation_1-auc:0.84205
    [447]	validation_0-auc:0.88328	validation_1-auc:0.84209
    [448]	validation_0-auc:0.88331	validation_1-auc:0.84207
    [449]	validation_0-auc:0.88342	validation_1-auc:0.84210
    [450]	validation_0-auc:0.88351	validation_1-auc:0.84211
    [451]	validation_0-auc:0.88356	validation_1-auc:0.84215
    [452]	validation_0-auc:0.88357	validation_1-auc:0.84213
    [453]	validation_0-auc:0.88364	validation_1-auc:0.84211
    [454]	validation_0-auc:0.88369	validation_1-auc:0.84209
    [455]	validation_0-auc:0.88373	validation_1-auc:0.84210
    [456]	validation_0-auc:0.88375	validation_1-auc:0.84208
    [457]	validation_0-auc:0.88379	validation_1-auc:0.84210
    [458]	validation_0-auc:0.88386	validation_1-auc:0.84211
    [459]	validation_0-auc:0.88387	validation_1-auc:0.84211
    [460]	validation_0-auc:0.88396	validation_1-auc:0.84219
    [461]	validation_0-auc:0.88409	validation_1-auc:0.84216
    [462]	validation_0-auc:0.88410	validation_1-auc:0.84215
    [463]	validation_0-auc:0.88417	validation_1-auc:0.84211
    [464]	validation_0-auc:0.88421	validation_1-auc:0.84210
    [465]	validation_0-auc:0.88431	validation_1-auc:0.84208
    [466]	validation_0-auc:0.88436	validation_1-auc:0.84214
    [467]	validation_0-auc:0.88443	validation_1-auc:0.84217
    [468]	validation_0-auc:0.88453	validation_1-auc:0.84218
    [469]	validation_0-auc:0.88461	validation_1-auc:0.84223
    [470]	validation_0-auc:0.88465	validation_1-auc:0.84224
    [471]	validation_0-auc:0.88474	validation_1-auc:0.84223
    [472]	validation_0-auc:0.88475	validation_1-auc:0.84222
    [473]	validation_0-auc:0.88477	validation_1-auc:0.84222
    [474]	validation_0-auc:0.88479	validation_1-auc:0.84220
    [475]	validation_0-auc:0.88482	validation_1-auc:0.84219
    [476]	validation_0-auc:0.88486	validation_1-auc:0.84220
    [477]	validation_0-auc:0.88488	validation_1-auc:0.84220
    [478]	validation_0-auc:0.88489	validation_1-auc:0.84220
    [479]	validation_0-auc:0.88491	validation_1-auc:0.84217
    [480]	validation_0-auc:0.88496	validation_1-auc:0.84220
    [481]	validation_0-auc:0.88506	validation_1-auc:0.84221
    [482]	validation_0-auc:0.88511	validation_1-auc:0.84221
    [483]	validation_0-auc:0.88519	validation_1-auc:0.84218
    [484]	validation_0-auc:0.88522	validation_1-auc:0.84220
    [485]	validation_0-auc:0.88523	validation_1-auc:0.84220
    [486]	validation_0-auc:0.88528	validation_1-auc:0.84224
    [487]	validation_0-auc:0.88535	validation_1-auc:0.84222
    [488]	validation_0-auc:0.88537	validation_1-auc:0.84223
    [489]	validation_0-auc:0.88539	validation_1-auc:0.84224
    [490]	validation_0-auc:0.88541	validation_1-auc:0.84223
    [491]	validation_0-auc:0.88546	validation_1-auc:0.84225
    [492]	validation_0-auc:0.88553	validation_1-auc:0.84230
    [493]	validation_0-auc:0.88558	validation_1-auc:0.84232
    [494]	validation_0-auc:0.88565	validation_1-auc:0.84231
    [495]	validation_0-auc:0.88568	validation_1-auc:0.84227
    [496]	validation_0-auc:0.88572	validation_1-auc:0.84224
    [497]	validation_0-auc:0.88576	validation_1-auc:0.84224
    [498]	validation_0-auc:0.88578	validation_1-auc:0.84223
    [499]	validation_0-auc:0.88585	validation_1-auc:0.84222
    [500]	validation_0-auc:0.88587	validation_1-auc:0.84219
    [501]	validation_0-auc:0.88592	validation_1-auc:0.84215
    [502]	validation_0-auc:0.88593	validation_1-auc:0.84216
    [503]	validation_0-auc:0.88595	validation_1-auc:0.84216
    [504]	validation_0-auc:0.88597	validation_1-auc:0.84217
    [505]	validation_0-auc:0.88604	validation_1-auc:0.84216
    [506]	validation_0-auc:0.88606	validation_1-auc:0.84214
    [507]	validation_0-auc:0.88611	validation_1-auc:0.84213
    [508]	validation_0-auc:0.88613	validation_1-auc:0.84210
    [509]	validation_0-auc:0.88616	validation_1-auc:0.84211
    [510]	validation_0-auc:0.88621	validation_1-auc:0.84211
    [511]	validation_0-auc:0.88627	validation_1-auc:0.84217
    [512]	validation_0-auc:0.88629	validation_1-auc:0.84218
    [513]	validation_0-auc:0.88632	validation_1-auc:0.84216
    [514]	validation_0-auc:0.88637	validation_1-auc:0.84218
    [515]	validation_0-auc:0.88643	validation_1-auc:0.84217
    [516]	validation_0-auc:0.88649	validation_1-auc:0.84215
    [517]	validation_0-auc:0.88652	validation_1-auc:0.84217
    [518]	validation_0-auc:0.88656	validation_1-auc:0.84215
    [519]	validation_0-auc:0.88657	validation_1-auc:0.84216
    [520]	validation_0-auc:0.88659	validation_1-auc:0.84217
    [521]	validation_0-auc:0.88664	validation_1-auc:0.84216
    [522]	validation_0-auc:0.88666	validation_1-auc:0.84215
    [523]	validation_0-auc:0.88673	validation_1-auc:0.84220
    [524]	validation_0-auc:0.88675	validation_1-auc:0.84217
    [525]	validation_0-auc:0.88679	validation_1-auc:0.84217
    [526]	validation_0-auc:0.88680	validation_1-auc:0.84215
    [527]	validation_0-auc:0.88682	validation_1-auc:0.84214
    [528]	validation_0-auc:0.88693	validation_1-auc:0.84214
    [529]	validation_0-auc:0.88697	validation_1-auc:0.84212
    [530]	validation_0-auc:0.88698	validation_1-auc:0.84213
    [531]	validation_0-auc:0.88700	validation_1-auc:0.84209
    [532]	validation_0-auc:0.88701	validation_1-auc:0.84208
    [533]	validation_0-auc:0.88703	validation_1-auc:0.84207
    [534]	validation_0-auc:0.88704	validation_1-auc:0.84206
    [535]	validation_0-auc:0.88706	validation_1-auc:0.84206
    [536]	validation_0-auc:0.88711	validation_1-auc:0.84207
    [537]	validation_0-auc:0.88713	validation_1-auc:0.84207
    [538]	validation_0-auc:0.88716	validation_1-auc:0.84205
    [539]	validation_0-auc:0.88721	validation_1-auc:0.84203
    [540]	validation_0-auc:0.88727	validation_1-auc:0.84207
    [541]	validation_0-auc:0.88728	validation_1-auc:0.84206
    [542]	validation_0-auc:0.88731	validation_1-auc:0.84208
    [543]	validation_0-auc:0.88733	validation_1-auc:0.84208
    [544]	validation_0-auc:0.88736	validation_1-auc:0.84207
    [545]	validation_0-auc:0.88745	validation_1-auc:0.84208
    [546]	validation_0-auc:0.88748	validation_1-auc:0.84208
    [547]	validation_0-auc:0.88752	validation_1-auc:0.84204
    [548]	validation_0-auc:0.88753	validation_1-auc:0.84204
    [549]	validation_0-auc:0.88760	validation_1-auc:0.84203
    [550]	validation_0-auc:0.88763	validation_1-auc:0.84201
    [551]	validation_0-auc:0.88765	validation_1-auc:0.84199
    [552]	validation_0-auc:0.88766	validation_1-auc:0.84199
    [553]	validation_0-auc:0.88768	validation_1-auc:0.84198
    [554]	validation_0-auc:0.88777	validation_1-auc:0.84197
    [555]	validation_0-auc:0.88777	validation_1-auc:0.84196
    [556]	validation_0-auc:0.88779	validation_1-auc:0.84198
    [557]	validation_0-auc:0.88780	validation_1-auc:0.84198
    [558]	validation_0-auc:0.88782	validation_1-auc:0.84197
    [559]	validation_0-auc:0.88783	validation_1-auc:0.84195
    [560]	validation_0-auc:0.88785	validation_1-auc:0.84193
    [561]	validation_0-auc:0.88788	validation_1-auc:0.84192
    [562]	validation_0-auc:0.88791	validation_1-auc:0.84193
    [563]	validation_0-auc:0.88793	validation_1-auc:0.84194
    [564]	validation_0-auc:0.88793	validation_1-auc:0.84193
    [565]	validation_0-auc:0.88796	validation_1-auc:0.84191
    [566]	validation_0-auc:0.88800	validation_1-auc:0.84193
    [567]	validation_0-auc:0.88802	validation_1-auc:0.84191
    [568]	validation_0-auc:0.88803	validation_1-auc:0.84189
    [569]	validation_0-auc:0.88806	validation_1-auc:0.84189
    [570]	validation_0-auc:0.88808	validation_1-auc:0.84189
    [571]	validation_0-auc:0.88811	validation_1-auc:0.84189
    [572]	validation_0-auc:0.88817	validation_1-auc:0.84188
    [573]	validation_0-auc:0.88817	validation_1-auc:0.84188
    [574]	validation_0-auc:0.88819	validation_1-auc:0.84188
    [575]	validation_0-auc:0.88821	validation_1-auc:0.84188
    [576]	validation_0-auc:0.88822	validation_1-auc:0.84188
    [577]	validation_0-auc:0.88826	validation_1-auc:0.84189
    [578]	validation_0-auc:0.88827	validation_1-auc:0.84188
    [579]	validation_0-auc:0.88835	validation_1-auc:0.84191
    [580]	validation_0-auc:0.88840	validation_1-auc:0.84192
    [581]	validation_0-auc:0.88844	validation_1-auc:0.84188
    [582]	validation_0-auc:0.88847	validation_1-auc:0.84185
    [583]	validation_0-auc:0.88848	validation_1-auc:0.84184
    [584]	validation_0-auc:0.88849	validation_1-auc:0.84185
    [585]	validation_0-auc:0.88850	validation_1-auc:0.84185
    [586]	validation_0-auc:0.88853	validation_1-auc:0.84184
    [587]	validation_0-auc:0.88856	validation_1-auc:0.84183
    [588]	validation_0-auc:0.88862	validation_1-auc:0.84182
    [589]	validation_0-auc:0.88866	validation_1-auc:0.84183
    [590]	validation_0-auc:0.88867	validation_1-auc:0.84182
    [591]	validation_0-auc:0.88876	validation_1-auc:0.84178
    [592]	validation_0-auc:0.88880	validation_1-auc:0.84179
    [593]	validation_0-auc:0.88881	validation_1-auc:0.84177
    [594]	validation_0-auc:0.88882	validation_1-auc:0.84176
    [595]	validation_0-auc:0.88884	validation_1-auc:0.84176
    [596]	validation_0-auc:0.88885	validation_1-auc:0.84175
    [597]	validation_0-auc:0.88886	validation_1-auc:0.84174
    [598]	validation_0-auc:0.88888	validation_1-auc:0.84175
    [599]	validation_0-auc:0.88893	validation_1-auc:0.84176
    [600]	validation_0-auc:0.88895	validation_1-auc:0.84173
    [601]	validation_0-auc:0.88897	validation_1-auc:0.84175
    [602]	validation_0-auc:0.88901	validation_1-auc:0.84173
    [603]	validation_0-auc:0.88902	validation_1-auc:0.84172
    [604]	validation_0-auc:0.88902	validation_1-auc:0.84171
    [605]	validation_0-auc:0.88904	validation_1-auc:0.84169
    [606]	validation_0-auc:0.88905	validation_1-auc:0.84168
    [607]	validation_0-auc:0.88907	validation_1-auc:0.84166
    [608]	validation_0-auc:0.88916	validation_1-auc:0.84161
    [609]	validation_0-auc:0.88917	validation_1-auc:0.84158
    [610]	validation_0-auc:0.88928	validation_1-auc:0.84161
    [611]	validation_0-auc:0.88933	validation_1-auc:0.84160
    [612]	validation_0-auc:0.88934	validation_1-auc:0.84159
    [613]	validation_0-auc:0.88937	validation_1-auc:0.84156
    [614]	validation_0-auc:0.88948	validation_1-auc:0.84157
    [615]	validation_0-auc:0.88950	validation_1-auc:0.84157
    [616]	validation_0-auc:0.88954	validation_1-auc:0.84154
    [617]	validation_0-auc:0.88956	validation_1-auc:0.84153
    [618]	validation_0-auc:0.88959	validation_1-auc:0.84153
    [619]	validation_0-auc:0.88968	validation_1-auc:0.84156
    [620]	validation_0-auc:0.88968	validation_1-auc:0.84156
    [621]	validation_0-auc:0.88971	validation_1-auc:0.84159
    [622]	validation_0-auc:0.88977	validation_1-auc:0.84159
    [623]	validation_0-auc:0.88982	validation_1-auc:0.84160
    [624]	validation_0-auc:0.88983	validation_1-auc:0.84159
    [625]	validation_0-auc:0.88984	validation_1-auc:0.84159
    [626]	validation_0-auc:0.88987	validation_1-auc:0.84158
    [627]	validation_0-auc:0.88987	validation_1-auc:0.84157
    [628]	validation_0-auc:0.88990	validation_1-auc:0.84154
    [629]	validation_0-auc:0.88996	validation_1-auc:0.84152
    [630]	validation_0-auc:0.89000	validation_1-auc:0.84152
    [631]	validation_0-auc:0.89001	validation_1-auc:0.84150
    [632]	validation_0-auc:0.89006	validation_1-auc:0.84150
    [633]	validation_0-auc:0.89006	validation_1-auc:0.84149
    [634]	validation_0-auc:0.89009	validation_1-auc:0.84148
    [635]	validation_0-auc:0.89013	validation_1-auc:0.84147
    [636]	validation_0-auc:0.89018	validation_1-auc:0.84147
    [637]	validation_0-auc:0.89019	validation_1-auc:0.84145
    [638]	validation_0-auc:0.89024	validation_1-auc:0.84147
    [639]	validation_0-auc:0.89024	validation_1-auc:0.84147
    [640]	validation_0-auc:0.89027	validation_1-auc:0.84149
    [641]	validation_0-auc:0.89031	validation_1-auc:0.84145
    [642]	validation_0-auc:0.89032	validation_1-auc:0.84145
    [643]	validation_0-auc:0.89034	validation_1-auc:0.84145
    [644]	validation_0-auc:0.89037	validation_1-auc:0.84143
    [645]	validation_0-auc:0.89039	validation_1-auc:0.84142
    [646]	validation_0-auc:0.89040	validation_1-auc:0.84142
    [647]	validation_0-auc:0.89042	validation_1-auc:0.84142
    [648]	validation_0-auc:0.89045	validation_1-auc:0.84145
    [649]	validation_0-auc:0.89048	validation_1-auc:0.84147
    [650]	validation_0-auc:0.89050	validation_1-auc:0.84146
    [651]	validation_0-auc:0.89052	validation_1-auc:0.84144
    [652]	validation_0-auc:0.89054	validation_1-auc:0.84144
    [653]	validation_0-auc:0.89055	validation_1-auc:0.84140
    [654]	validation_0-auc:0.89057	validation_1-auc:0.84142
    [655]	validation_0-auc:0.89059	validation_1-auc:0.84142
    [656]	validation_0-auc:0.89060	validation_1-auc:0.84141
    [657]	validation_0-auc:0.89063	validation_1-auc:0.84140
    [658]	validation_0-auc:0.89067	validation_1-auc:0.84138
    [659]	validation_0-auc:0.89074	validation_1-auc:0.84142
    [660]	validation_0-auc:0.89077	validation_1-auc:0.84142
    [661]	validation_0-auc:0.89079	validation_1-auc:0.84141
    [662]	validation_0-auc:0.89079	validation_1-auc:0.84141
    [663]	validation_0-auc:0.89084	validation_1-auc:0.84142
    [664]	validation_0-auc:0.89086	validation_1-auc:0.84144
    [665]	validation_0-auc:0.89092	validation_1-auc:0.84141
    [666]	validation_0-auc:0.89092	validation_1-auc:0.84140
    [667]	validation_0-auc:0.89094	validation_1-auc:0.84140
    [668]	validation_0-auc:0.89098	validation_1-auc:0.84137
    [669]	validation_0-auc:0.89102	validation_1-auc:0.84136
    [670]	validation_0-auc:0.89105	validation_1-auc:0.84137
    [671]	validation_0-auc:0.89108	validation_1-auc:0.84142
    [672]	validation_0-auc:0.89114	validation_1-auc:0.84143
    [673]	validation_0-auc:0.89116	validation_1-auc:0.84142
    [674]	validation_0-auc:0.89118	validation_1-auc:0.84144
    [675]	validation_0-auc:0.89121	validation_1-auc:0.84143
    [676]	validation_0-auc:0.89123	validation_1-auc:0.84145
    [677]	validation_0-auc:0.89128	validation_1-auc:0.84150
    [678]	validation_0-auc:0.89130	validation_1-auc:0.84150
    [679]	validation_0-auc:0.89132	validation_1-auc:0.84150
    [680]	validation_0-auc:0.89136	validation_1-auc:0.84151
    [681]	validation_0-auc:0.89138	validation_1-auc:0.84150
    [682]	validation_0-auc:0.89140	validation_1-auc:0.84150
    [683]	validation_0-auc:0.89140	validation_1-auc:0.84150
    [684]	validation_0-auc:0.89148	validation_1-auc:0.84151
    [685]	validation_0-auc:0.89151	validation_1-auc:0.84149
    [686]	validation_0-auc:0.89152	validation_1-auc:0.84150
    [687]	validation_0-auc:0.89154	validation_1-auc:0.84151
    [688]	validation_0-auc:0.89160	validation_1-auc:0.84148
    [689]	validation_0-auc:0.89163	validation_1-auc:0.84148
    [690]	validation_0-auc:0.89165	validation_1-auc:0.84148
    [691]	validation_0-auc:0.89166	validation_1-auc:0.84147
    [692]	validation_0-auc:0.89168	validation_1-auc:0.84147
    roc_auc_score: 0.8423159438213648


### 피처의 중요도 출력


```python
from xgboost import plot_importance
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,1, figsize=(10,8))
plot_importance(xgb_clf, ax=ax, max_num_features=20, height=0.4)
```




    <AxesSubplot:title={'center':'Feature importance'}, xlabel='F score', ylabel='Features'>




    
![png](output_24_1.png)
    


## 알고리즘: Light GBM 

### 학습


```python
from lightgbm import LGBMClassifier
#알고리즘 생성
lgbm_clf = LGBMClassifier(n_estimators=1000)

#평가할 데이터 - 평가할 데이터에는 훈련 데이터를 포함시키지 않아도 됩니다.
evals = [(X_test, y_test)]

lgbm_clf.fit(X_train, y_train, early_stopping_rounds=200, eval_metric='auc', 
            eval_set=evals, verbose=True)

```

    [1]	valid_0's auc: 0.809902	valid_0's binary_logloss: 0.159964
    Training until validation scores don't improve for 200 rounds
    [2]	valid_0's auc: 0.816899	valid_0's binary_logloss: 0.155283
    [3]	valid_0's auc: 0.821095	valid_0's binary_logloss: 0.151902
    [4]	valid_0's auc: 0.824407	valid_0's binary_logloss: 0.1494
    [5]	valid_0's auc: 0.830119	valid_0's binary_logloss: 0.147369
    [6]	valid_0's auc: 0.830039	valid_0's binary_logloss: 0.145835
    [7]	valid_0's auc: 0.832806	valid_0's binary_logloss: 0.144464
    [8]	valid_0's auc: 0.83413	valid_0's binary_logloss: 0.143287
    [9]	valid_0's auc: 0.83441	valid_0's binary_logloss: 0.142303
    [10]	valid_0's auc: 0.834318	valid_0's binary_logloss: 0.141564
    [11]	valid_0's auc: 0.834841	valid_0's binary_logloss: 0.140823
    [12]	valid_0's auc: 0.83506	valid_0's binary_logloss: 0.140228
    [13]	valid_0's auc: 0.836167	valid_0's binary_logloss: 0.139711
    [14]	valid_0's auc: 0.836397	valid_0's binary_logloss: 0.139345
    [15]	valid_0's auc: 0.836836	valid_0's binary_logloss: 0.138925
    [16]	valid_0's auc: 0.837185	valid_0's binary_logloss: 0.138574
    [17]	valid_0's auc: 0.837474	valid_0's binary_logloss: 0.138226
    [18]	valid_0's auc: 0.837552	valid_0's binary_logloss: 0.137971
    [19]	valid_0's auc: 0.838014	valid_0's binary_logloss: 0.137751
    [20]	valid_0's auc: 0.838538	valid_0's binary_logloss: 0.137566
    [21]	valid_0's auc: 0.838207	valid_0's binary_logloss: 0.137409
    [22]	valid_0's auc: 0.838061	valid_0's binary_logloss: 0.137294
    [23]	valid_0's auc: 0.837976	valid_0's binary_logloss: 0.137213
    [24]	valid_0's auc: 0.838873	valid_0's binary_logloss: 0.137
    [25]	valid_0's auc: 0.838653	valid_0's binary_logloss: 0.13695
    [26]	valid_0's auc: 0.838674	valid_0's binary_logloss: 0.136892
    [27]	valid_0's auc: 0.838863	valid_0's binary_logloss: 0.136776
    [28]	valid_0's auc: 0.838505	valid_0's binary_logloss: 0.13671
    [29]	valid_0's auc: 0.838766	valid_0's binary_logloss: 0.1366
    [30]	valid_0's auc: 0.838672	valid_0's binary_logloss: 0.13656
    [31]	valid_0's auc: 0.838612	valid_0's binary_logloss: 0.136543
    [32]	valid_0's auc: 0.839369	valid_0's binary_logloss: 0.136388
    [33]	valid_0's auc: 0.839257	valid_0's binary_logloss: 0.136371
    [34]	valid_0's auc: 0.839211	valid_0's binary_logloss: 0.136324
    [35]	valid_0's auc: 0.839573	valid_0's binary_logloss: 0.136222
    [36]	valid_0's auc: 0.839484	valid_0's binary_logloss: 0.136219
    [37]	valid_0's auc: 0.839052	valid_0's binary_logloss: 0.136249
    [38]	valid_0's auc: 0.839061	valid_0's binary_logloss: 0.136245
    [39]	valid_0's auc: 0.839339	valid_0's binary_logloss: 0.136175
    [40]	valid_0's auc: 0.839358	valid_0's binary_logloss: 0.136168
    [41]	valid_0's auc: 0.839147	valid_0's binary_logloss: 0.136185
    [42]	valid_0's auc: 0.839564	valid_0's binary_logloss: 0.136133
    [43]	valid_0's auc: 0.839272	valid_0's binary_logloss: 0.136176
    [44]	valid_0's auc: 0.839197	valid_0's binary_logloss: 0.136215
    [45]	valid_0's auc: 0.839223	valid_0's binary_logloss: 0.136175
    [46]	valid_0's auc: 0.839472	valid_0's binary_logloss: 0.136133
    [47]	valid_0's auc: 0.839482	valid_0's binary_logloss: 0.13611
    [48]	valid_0's auc: 0.839278	valid_0's binary_logloss: 0.136159
    [49]	valid_0's auc: 0.839372	valid_0's binary_logloss: 0.136131
    [50]	valid_0's auc: 0.839126	valid_0's binary_logloss: 0.136147
    [51]	valid_0's auc: 0.839331	valid_0's binary_logloss: 0.136023
    [52]	valid_0's auc: 0.839124	valid_0's binary_logloss: 0.136053
    [53]	valid_0's auc: 0.839513	valid_0's binary_logloss: 0.136012
    [54]	valid_0's auc: 0.839285	valid_0's binary_logloss: 0.136059
    [55]	valid_0's auc: 0.839144	valid_0's binary_logloss: 0.136078
    [56]	valid_0's auc: 0.839312	valid_0's binary_logloss: 0.136059
    [57]	valid_0's auc: 0.839311	valid_0's binary_logloss: 0.13603
    [58]	valid_0's auc: 0.839349	valid_0's binary_logloss: 0.136065
    [59]	valid_0's auc: 0.839282	valid_0's binary_logloss: 0.136078
    [60]	valid_0's auc: 0.839344	valid_0's binary_logloss: 0.136088
    [61]	valid_0's auc: 0.839269	valid_0's binary_logloss: 0.136098
    [62]	valid_0's auc: 0.839052	valid_0's binary_logloss: 0.136135
    [63]	valid_0's auc: 0.839102	valid_0's binary_logloss: 0.136118
    [64]	valid_0's auc: 0.839114	valid_0's binary_logloss: 0.136134
    [65]	valid_0's auc: 0.839066	valid_0's binary_logloss: 0.136138
    [66]	valid_0's auc: 0.838747	valid_0's binary_logloss: 0.136182
    [67]	valid_0's auc: 0.838754	valid_0's binary_logloss: 0.136215
    [68]	valid_0's auc: 0.838558	valid_0's binary_logloss: 0.136256
    [69]	valid_0's auc: 0.838321	valid_0's binary_logloss: 0.136299
    [70]	valid_0's auc: 0.8382	valid_0's binary_logloss: 0.136329
    [71]	valid_0's auc: 0.838589	valid_0's binary_logloss: 0.136282
    [72]	valid_0's auc: 0.838337	valid_0's binary_logloss: 0.136321
    [73]	valid_0's auc: 0.8381	valid_0's binary_logloss: 0.13638
    [74]	valid_0's auc: 0.837968	valid_0's binary_logloss: 0.136423
    [75]	valid_0's auc: 0.837739	valid_0's binary_logloss: 0.136491
    [76]	valid_0's auc: 0.837475	valid_0's binary_logloss: 0.136527
    [77]	valid_0's auc: 0.837277	valid_0's binary_logloss: 0.136585
    [78]	valid_0's auc: 0.837187	valid_0's binary_logloss: 0.136611
    [79]	valid_0's auc: 0.8372	valid_0's binary_logloss: 0.13664
    [80]	valid_0's auc: 0.837209	valid_0's binary_logloss: 0.136664
    [81]	valid_0's auc: 0.837205	valid_0's binary_logloss: 0.136659
    [82]	valid_0's auc: 0.836863	valid_0's binary_logloss: 0.136724
    [83]	valid_0's auc: 0.836756	valid_0's binary_logloss: 0.136715
    [84]	valid_0's auc: 0.836803	valid_0's binary_logloss: 0.136703
    [85]	valid_0's auc: 0.836776	valid_0's binary_logloss: 0.136733
    [86]	valid_0's auc: 0.836665	valid_0's binary_logloss: 0.136753
    [87]	valid_0's auc: 0.836441	valid_0's binary_logloss: 0.136788
    [88]	valid_0's auc: 0.836379	valid_0's binary_logloss: 0.136819
    [89]	valid_0's auc: 0.836421	valid_0's binary_logloss: 0.136795
    [90]	valid_0's auc: 0.836356	valid_0's binary_logloss: 0.136813
    [91]	valid_0's auc: 0.836151	valid_0's binary_logloss: 0.136875
    [92]	valid_0's auc: 0.836169	valid_0's binary_logloss: 0.136899
    [93]	valid_0's auc: 0.836074	valid_0's binary_logloss: 0.136939
    [94]	valid_0's auc: 0.836517	valid_0's binary_logloss: 0.136886
    [95]	valid_0's auc: 0.836375	valid_0's binary_logloss: 0.136918
    [96]	valid_0's auc: 0.836343	valid_0's binary_logloss: 0.136921
    [97]	valid_0's auc: 0.836636	valid_0's binary_logloss: 0.136938
    [98]	valid_0's auc: 0.836638	valid_0's binary_logloss: 0.13696
    [99]	valid_0's auc: 0.836467	valid_0's binary_logloss: 0.136974
    [100]	valid_0's auc: 0.836346	valid_0's binary_logloss: 0.137028
    [101]	valid_0's auc: 0.836249	valid_0's binary_logloss: 0.13707
    [102]	valid_0's auc: 0.836223	valid_0's binary_logloss: 0.1371
    [103]	valid_0's auc: 0.836101	valid_0's binary_logloss: 0.137111
    [104]	valid_0's auc: 0.836134	valid_0's binary_logloss: 0.137111
    [105]	valid_0's auc: 0.835986	valid_0's binary_logloss: 0.137146
    [106]	valid_0's auc: 0.835874	valid_0's binary_logloss: 0.137142
    [107]	valid_0's auc: 0.835975	valid_0's binary_logloss: 0.137141
    [108]	valid_0's auc: 0.836024	valid_0's binary_logloss: 0.137149
    [109]	valid_0's auc: 0.835998	valid_0's binary_logloss: 0.137167
    [110]	valid_0's auc: 0.835778	valid_0's binary_logloss: 0.137223
    [111]	valid_0's auc: 0.835579	valid_0's binary_logloss: 0.137272
    [112]	valid_0's auc: 0.835204	valid_0's binary_logloss: 0.137352
    [113]	valid_0's auc: 0.835105	valid_0's binary_logloss: 0.137352
    [114]	valid_0's auc: 0.834791	valid_0's binary_logloss: 0.137409
    [115]	valid_0's auc: 0.834625	valid_0's binary_logloss: 0.137434
    [116]	valid_0's auc: 0.834475	valid_0's binary_logloss: 0.137462
    [117]	valid_0's auc: 0.834371	valid_0's binary_logloss: 0.137483
    [118]	valid_0's auc: 0.834732	valid_0's binary_logloss: 0.137437
    [119]	valid_0's auc: 0.834871	valid_0's binary_logloss: 0.137451
    [120]	valid_0's auc: 0.83489	valid_0's binary_logloss: 0.13742
    [121]	valid_0's auc: 0.834937	valid_0's binary_logloss: 0.137447
    [122]	valid_0's auc: 0.834831	valid_0's binary_logloss: 0.137478
    [123]	valid_0's auc: 0.834741	valid_0's binary_logloss: 0.137541
    [124]	valid_0's auc: 0.834755	valid_0's binary_logloss: 0.137534
    [125]	valid_0's auc: 0.834507	valid_0's binary_logloss: 0.137584
    [126]	valid_0's auc: 0.834546	valid_0's binary_logloss: 0.137593
    [127]	valid_0's auc: 0.834477	valid_0's binary_logloss: 0.137612
    [128]	valid_0's auc: 0.834319	valid_0's binary_logloss: 0.137613
    [129]	valid_0's auc: 0.834265	valid_0's binary_logloss: 0.137632
    [130]	valid_0's auc: 0.834169	valid_0's binary_logloss: 0.137671
    [131]	valid_0's auc: 0.833941	valid_0's binary_logloss: 0.137747
    [132]	valid_0's auc: 0.833831	valid_0's binary_logloss: 0.13776
    [133]	valid_0's auc: 0.833877	valid_0's binary_logloss: 0.137748
    [134]	valid_0's auc: 0.83397	valid_0's binary_logloss: 0.137742
    [135]	valid_0's auc: 0.833866	valid_0's binary_logloss: 0.137752
    [136]	valid_0's auc: 0.833835	valid_0's binary_logloss: 0.137764
    [137]	valid_0's auc: 0.833701	valid_0's binary_logloss: 0.137777
    [138]	valid_0's auc: 0.833646	valid_0's binary_logloss: 0.137791
    [139]	valid_0's auc: 0.833375	valid_0's binary_logloss: 0.137861
    [140]	valid_0's auc: 0.83315	valid_0's binary_logloss: 0.137913
    [141]	valid_0's auc: 0.832933	valid_0's binary_logloss: 0.137964
    [142]	valid_0's auc: 0.832942	valid_0's binary_logloss: 0.137974
    [143]	valid_0's auc: 0.832897	valid_0's binary_logloss: 0.137989
    [144]	valid_0's auc: 0.832742	valid_0's binary_logloss: 0.138059
    [145]	valid_0's auc: 0.832534	valid_0's binary_logloss: 0.138103
    [146]	valid_0's auc: 0.83242	valid_0's binary_logloss: 0.138124
    [147]	valid_0's auc: 0.832373	valid_0's binary_logloss: 0.138123
    [148]	valid_0's auc: 0.832306	valid_0's binary_logloss: 0.138137
    [149]	valid_0's auc: 0.832482	valid_0's binary_logloss: 0.138105
    [150]	valid_0's auc: 0.832223	valid_0's binary_logloss: 0.138173
    [151]	valid_0's auc: 0.83217	valid_0's binary_logloss: 0.138198
    [152]	valid_0's auc: 0.83215	valid_0's binary_logloss: 0.138215
    [153]	valid_0's auc: 0.832222	valid_0's binary_logloss: 0.138224
    [154]	valid_0's auc: 0.832132	valid_0's binary_logloss: 0.138256
    [155]	valid_0's auc: 0.832084	valid_0's binary_logloss: 0.138307
    [156]	valid_0's auc: 0.831965	valid_0's binary_logloss: 0.138352
    [157]	valid_0's auc: 0.831992	valid_0's binary_logloss: 0.138382
    [158]	valid_0's auc: 0.832081	valid_0's binary_logloss: 0.138342
    [159]	valid_0's auc: 0.831996	valid_0's binary_logloss: 0.138394
    [160]	valid_0's auc: 0.831948	valid_0's binary_logloss: 0.13843
    [161]	valid_0's auc: 0.831816	valid_0's binary_logloss: 0.138491
    [162]	valid_0's auc: 0.831729	valid_0's binary_logloss: 0.138538
    [163]	valid_0's auc: 0.831627	valid_0's binary_logloss: 0.138599
    [164]	valid_0's auc: 0.831251	valid_0's binary_logloss: 0.138677
    [165]	valid_0's auc: 0.830997	valid_0's binary_logloss: 0.138723
    [166]	valid_0's auc: 0.831024	valid_0's binary_logloss: 0.13873
    [167]	valid_0's auc: 0.83079	valid_0's binary_logloss: 0.138795
    [168]	valid_0's auc: 0.830369	valid_0's binary_logloss: 0.138885
    [169]	valid_0's auc: 0.830449	valid_0's binary_logloss: 0.138878
    [170]	valid_0's auc: 0.83038	valid_0's binary_logloss: 0.138929
    [171]	valid_0's auc: 0.830474	valid_0's binary_logloss: 0.138943
    [172]	valid_0's auc: 0.830491	valid_0's binary_logloss: 0.138955
    [173]	valid_0's auc: 0.830465	valid_0's binary_logloss: 0.13896
    [174]	valid_0's auc: 0.830256	valid_0's binary_logloss: 0.139017
    [175]	valid_0's auc: 0.83014	valid_0's binary_logloss: 0.139044
    [176]	valid_0's auc: 0.830185	valid_0's binary_logloss: 0.139042
    [177]	valid_0's auc: 0.83006	valid_0's binary_logloss: 0.139059
    [178]	valid_0's auc: 0.829894	valid_0's binary_logloss: 0.139083
    [179]	valid_0's auc: 0.829842	valid_0's binary_logloss: 0.139116
    [180]	valid_0's auc: 0.829761	valid_0's binary_logloss: 0.139154
    [181]	valid_0's auc: 0.830233	valid_0's binary_logloss: 0.139043
    [182]	valid_0's auc: 0.830161	valid_0's binary_logloss: 0.139057
    [183]	valid_0's auc: 0.829973	valid_0's binary_logloss: 0.13909
    [184]	valid_0's auc: 0.830069	valid_0's binary_logloss: 0.139088
    [185]	valid_0's auc: 0.829932	valid_0's binary_logloss: 0.139131
    [186]	valid_0's auc: 0.829556	valid_0's binary_logloss: 0.13924
    [187]	valid_0's auc: 0.829658	valid_0's binary_logloss: 0.139231
    [188]	valid_0's auc: 0.82947	valid_0's binary_logloss: 0.139293
    [189]	valid_0's auc: 0.829197	valid_0's binary_logloss: 0.139365
    [190]	valid_0's auc: 0.829004	valid_0's binary_logloss: 0.139431
    [191]	valid_0's auc: 0.829059	valid_0's binary_logloss: 0.139454
    [192]	valid_0's auc: 0.828829	valid_0's binary_logloss: 0.139533
    [193]	valid_0's auc: 0.828999	valid_0's binary_logloss: 0.139497
    [194]	valid_0's auc: 0.828891	valid_0's binary_logloss: 0.139532
    [195]	valid_0's auc: 0.828924	valid_0's binary_logloss: 0.139536
    [196]	valid_0's auc: 0.828794	valid_0's binary_logloss: 0.139561
    [197]	valid_0's auc: 0.828571	valid_0's binary_logloss: 0.139646
    [198]	valid_0's auc: 0.828395	valid_0's binary_logloss: 0.13966
    [199]	valid_0's auc: 0.82802	valid_0's binary_logloss: 0.139742
    [200]	valid_0's auc: 0.827762	valid_0's binary_logloss: 0.139825
    [201]	valid_0's auc: 0.827773	valid_0's binary_logloss: 0.139853
    [202]	valid_0's auc: 0.827911	valid_0's binary_logloss: 0.139855
    [203]	valid_0's auc: 0.827869	valid_0's binary_logloss: 0.13987
    [204]	valid_0's auc: 0.827793	valid_0's binary_logloss: 0.139895
    [205]	valid_0's auc: 0.827715	valid_0's binary_logloss: 0.13992
    [206]	valid_0's auc: 0.827727	valid_0's binary_logloss: 0.139932
    [207]	valid_0's auc: 0.82773	valid_0's binary_logloss: 0.139956
    [208]	valid_0's auc: 0.827812	valid_0's binary_logloss: 0.139944
    [209]	valid_0's auc: 0.827785	valid_0's binary_logloss: 0.139978
    [210]	valid_0's auc: 0.827676	valid_0's binary_logloss: 0.140028
    [211]	valid_0's auc: 0.827725	valid_0's binary_logloss: 0.140049
    [212]	valid_0's auc: 0.827798	valid_0's binary_logloss: 0.14006
    [213]	valid_0's auc: 0.827883	valid_0's binary_logloss: 0.140051
    [214]	valid_0's auc: 0.82792	valid_0's binary_logloss: 0.140064
    [215]	valid_0's auc: 0.827817	valid_0's binary_logloss: 0.140123
    [216]	valid_0's auc: 0.827808	valid_0's binary_logloss: 0.140139
    [217]	valid_0's auc: 0.827573	valid_0's binary_logloss: 0.140222
    [218]	valid_0's auc: 0.827513	valid_0's binary_logloss: 0.140232
    [219]	valid_0's auc: 0.827572	valid_0's binary_logloss: 0.140242
    [220]	valid_0's auc: 0.827556	valid_0's binary_logloss: 0.140267
    [221]	valid_0's auc: 0.827336	valid_0's binary_logloss: 0.140324
    [222]	valid_0's auc: 0.827366	valid_0's binary_logloss: 0.140361
    [223]	valid_0's auc: 0.827317	valid_0's binary_logloss: 0.140389
    [224]	valid_0's auc: 0.827252	valid_0's binary_logloss: 0.140417
    [225]	valid_0's auc: 0.827196	valid_0's binary_logloss: 0.140464
    [226]	valid_0's auc: 0.827033	valid_0's binary_logloss: 0.140515
    [227]	valid_0's auc: 0.827116	valid_0's binary_logloss: 0.140512
    [228]	valid_0's auc: 0.827057	valid_0's binary_logloss: 0.14055
    [229]	valid_0's auc: 0.82688	valid_0's binary_logloss: 0.140601
    [230]	valid_0's auc: 0.826885	valid_0's binary_logloss: 0.140636
    [231]	valid_0's auc: 0.826922	valid_0's binary_logloss: 0.140636
    [232]	valid_0's auc: 0.826919	valid_0's binary_logloss: 0.14069
    [233]	valid_0's auc: 0.82695	valid_0's binary_logloss: 0.140695
    [234]	valid_0's auc: 0.82715	valid_0's binary_logloss: 0.140655
    [235]	valid_0's auc: 0.827149	valid_0's binary_logloss: 0.1407
    Early stopping, best iteration is:
    [35]	valid_0's auc: 0.839573	valid_0's binary_logloss: 0.136222





    LGBMClassifier(n_estimators=1000)



### 평가 지표 확인


```python
lgbm_roc_auc_score = roc_auc_score(y_test, lgbm_clf.predict_proba(X_test)[:,1], 
                                 average='micro')
print('roc_auc_score:',lgbm_roc_auc_score)
```

    roc_auc_score: 0.8395730588951105


### 최적의 하이퍼파라미터 찾기


```python
from sklearn.model_selection import GridSearchCV

lgbm_clf = LGBMClassifier(n_estimators=100)
#파라미터 생성 - 시간 관계상 파라미터의 개수를 2개씩으로 설정한 것이고
#실제 모델을 가지고 학습할 때는 더 다양한 값을 설정해야 합니다.
params = {'num_leaves': [32, 64],
          'max_depth':[130,140,150,160],
          'min_child_samples':[60, 100],
          'subsample':[0.8,1.0]}

#cv를 3으로 설정했으므로 96번 수행
gridcv = GridSearchCV(lgbm_clf, param_grid = params, cv=3)
#early_stopping_rounds 를 조금 더 높여주어도 됩니다.
gridcv.fit(X_train, y_train, early_stopping_rounds=30, eval_metric='auc',
          eval_set=[(X_train, y_train), (X_test, y_test)])

#최적의 파라미터 출력
print("최적의 파라미터:", gridcv.best_params_)
lgbm_roc_auc_score = roc_auc_score(y_test, gridcv.predict_proba(X_test)[:,1], 
                                 average='micro')
print('roc_auc_score:', lgbm_roc_auc_score)
```

    [1]	valid_0's auc: 0.822135	valid_0's binary_logloss: 0.157275	valid_1's auc: 0.80763	valid_1's binary_logloss: 0.159959
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.826629	valid_0's binary_logloss: 0.15186	valid_1's auc: 0.811377	valid_1's binary_logloss: 0.155197
    [3]	valid_0's auc: 0.830813	valid_0's binary_logloss: 0.147949	valid_1's auc: 0.813917	valid_1's binary_logloss: 0.151946
    [4]	valid_0's auc: 0.833387	valid_0's binary_logloss: 0.144809	valid_1's auc: 0.814641	valid_1's binary_logloss: 0.149358
    [5]	valid_0's auc: 0.838116	valid_0's binary_logloss: 0.142332	valid_1's auc: 0.816965	valid_1's binary_logloss: 0.147361
    [6]	valid_0's auc: 0.841055	valid_0's binary_logloss: 0.140267	valid_1's auc: 0.81978	valid_1's binary_logloss: 0.145867
    [7]	valid_0's auc: 0.842021	valid_0's binary_logloss: 0.138495	valid_1's auc: 0.819689	valid_1's binary_logloss: 0.144575
    [8]	valid_0's auc: 0.843828	valid_0's binary_logloss: 0.137029	valid_1's auc: 0.822496	valid_1's binary_logloss: 0.143422
    [9]	valid_0's auc: 0.845522	valid_0's binary_logloss: 0.135783	valid_1's auc: 0.823288	valid_1's binary_logloss: 0.142428
    [10]	valid_0's auc: 0.847654	valid_0's binary_logloss: 0.134654	valid_1's auc: 0.824658	valid_1's binary_logloss: 0.141712
    [11]	valid_0's auc: 0.849302	valid_0's binary_logloss: 0.133679	valid_1's auc: 0.826788	valid_1's binary_logloss: 0.140983
    [12]	valid_0's auc: 0.850949	valid_0's binary_logloss: 0.132827	valid_1's auc: 0.828316	valid_1's binary_logloss: 0.140403
    [13]	valid_0's auc: 0.852852	valid_0's binary_logloss: 0.132062	valid_1's auc: 0.830745	valid_1's binary_logloss: 0.139849
    [14]	valid_0's auc: 0.854299	valid_0's binary_logloss: 0.131363	valid_1's auc: 0.832741	valid_1's binary_logloss: 0.139383
    [15]	valid_0's auc: 0.855376	valid_0's binary_logloss: 0.13071	valid_1's auc: 0.833869	valid_1's binary_logloss: 0.138963
    [16]	valid_0's auc: 0.856265	valid_0's binary_logloss: 0.130166	valid_1's auc: 0.833925	valid_1's binary_logloss: 0.138745
    [17]	valid_0's auc: 0.85735	valid_0's binary_logloss: 0.129615	valid_1's auc: 0.834227	valid_1's binary_logloss: 0.138474
    [18]	valid_0's auc: 0.859279	valid_0's binary_logloss: 0.129153	valid_1's auc: 0.834146	valid_1's binary_logloss: 0.138306
    [19]	valid_0's auc: 0.860099	valid_0's binary_logloss: 0.128684	valid_1's auc: 0.833391	valid_1's binary_logloss: 0.138142
    [20]	valid_0's auc: 0.861239	valid_0's binary_logloss: 0.128272	valid_1's auc: 0.833651	valid_1's binary_logloss: 0.13795
    [21]	valid_0's auc: 0.862153	valid_0's binary_logloss: 0.127883	valid_1's auc: 0.834662	valid_1's binary_logloss: 0.137829
    [22]	valid_0's auc: 0.863466	valid_0's binary_logloss: 0.127439	valid_1's auc: 0.834505	valid_1's binary_logloss: 0.137762
    [23]	valid_0's auc: 0.86379	valid_0's binary_logloss: 0.127135	valid_1's auc: 0.834377	valid_1's binary_logloss: 0.137657
    [24]	valid_0's auc: 0.864548	valid_0's binary_logloss: 0.126788	valid_1's auc: 0.834289	valid_1's binary_logloss: 0.137499
    [25]	valid_0's auc: 0.865681	valid_0's binary_logloss: 0.126451	valid_1's auc: 0.834899	valid_1's binary_logloss: 0.137329
    [26]	valid_0's auc: 0.866659	valid_0's binary_logloss: 0.126121	valid_1's auc: 0.835946	valid_1's binary_logloss: 0.137177
    [27]	valid_0's auc: 0.867505	valid_0's binary_logloss: 0.125853	valid_1's auc: 0.836616	valid_1's binary_logloss: 0.137016
    [28]	valid_0's auc: 0.868089	valid_0's binary_logloss: 0.125584	valid_1's auc: 0.836565	valid_1's binary_logloss: 0.137
    [29]	valid_0's auc: 0.868709	valid_0's binary_logloss: 0.125327	valid_1's auc: 0.837028	valid_1's binary_logloss: 0.136878
    [30]	valid_0's auc: 0.87006	valid_0's binary_logloss: 0.12501	valid_1's auc: 0.837512	valid_1's binary_logloss: 0.136792
    [31]	valid_0's auc: 0.870612	valid_0's binary_logloss: 0.124704	valid_1's auc: 0.837541	valid_1's binary_logloss: 0.136757
    [32]	valid_0's auc: 0.871234	valid_0's binary_logloss: 0.124431	valid_1's auc: 0.837141	valid_1's binary_logloss: 0.136755
    [33]	valid_0's auc: 0.872027	valid_0's binary_logloss: 0.124167	valid_1's auc: 0.837258	valid_1's binary_logloss: 0.136728
    [34]	valid_0's auc: 0.87244	valid_0's binary_logloss: 0.123956	valid_1's auc: 0.836829	valid_1's binary_logloss: 0.136799
    [35]	valid_0's auc: 0.873259	valid_0's binary_logloss: 0.123713	valid_1's auc: 0.836871	valid_1's binary_logloss: 0.136775
    [36]	valid_0's auc: 0.873695	valid_0's binary_logloss: 0.123497	valid_1's auc: 0.836512	valid_1's binary_logloss: 0.136859
    [37]	valid_0's auc: 0.874365	valid_0's binary_logloss: 0.123259	valid_1's auc: 0.836421	valid_1's binary_logloss: 0.136872
    [38]	valid_0's auc: 0.874998	valid_0's binary_logloss: 0.123014	valid_1's auc: 0.83628	valid_1's binary_logloss: 0.136939
    [39]	valid_0's auc: 0.875666	valid_0's binary_logloss: 0.122811	valid_1's auc: 0.836577	valid_1's binary_logloss: 0.13689
    [40]	valid_0's auc: 0.875931	valid_0's binary_logloss: 0.122638	valid_1's auc: 0.836148	valid_1's binary_logloss: 0.13697
    [41]	valid_0's auc: 0.876403	valid_0's binary_logloss: 0.122474	valid_1's auc: 0.836008	valid_1's binary_logloss: 0.137005
    [42]	valid_0's auc: 0.87697	valid_0's binary_logloss: 0.122306	valid_1's auc: 0.835814	valid_1's binary_logloss: 0.137031
    [43]	valid_0's auc: 0.877305	valid_0's binary_logloss: 0.122115	valid_1's auc: 0.835467	valid_1's binary_logloss: 0.137112
    [44]	valid_0's auc: 0.87782	valid_0's binary_logloss: 0.121932	valid_1's auc: 0.835339	valid_1's binary_logloss: 0.137217
    [45]	valid_0's auc: 0.878559	valid_0's binary_logloss: 0.121701	valid_1's auc: 0.83506	valid_1's binary_logloss: 0.137256
    [46]	valid_0's auc: 0.878895	valid_0's binary_logloss: 0.121564	valid_1's auc: 0.834955	valid_1's binary_logloss: 0.137255
    [47]	valid_0's auc: 0.879643	valid_0's binary_logloss: 0.121306	valid_1's auc: 0.834987	valid_1's binary_logloss: 0.137275
    [48]	valid_0's auc: 0.88002	valid_0's binary_logloss: 0.121119	valid_1's auc: 0.834864	valid_1's binary_logloss: 0.137329
    [49]	valid_0's auc: 0.880318	valid_0's binary_logloss: 0.120965	valid_1's auc: 0.834547	valid_1's binary_logloss: 0.137394
    [50]	valid_0's auc: 0.880967	valid_0's binary_logloss: 0.12082	valid_1's auc: 0.834588	valid_1's binary_logloss: 0.137388
    [51]	valid_0's auc: 0.8813	valid_0's binary_logloss: 0.120647	valid_1's auc: 0.834288	valid_1's binary_logloss: 0.137467
    [52]	valid_0's auc: 0.881706	valid_0's binary_logloss: 0.120482	valid_1's auc: 0.834305	valid_1's binary_logloss: 0.137491
    [53]	valid_0's auc: 0.882192	valid_0's binary_logloss: 0.1203	valid_1's auc: 0.834662	valid_1's binary_logloss: 0.137492
    [54]	valid_0's auc: 0.882307	valid_0's binary_logloss: 0.120188	valid_1's auc: 0.834241	valid_1's binary_logloss: 0.13763
    [55]	valid_0's auc: 0.883062	valid_0's binary_logloss: 0.119929	valid_1's auc: 0.834145	valid_1's binary_logloss: 0.137674
    [56]	valid_0's auc: 0.883272	valid_0's binary_logloss: 0.119823	valid_1's auc: 0.833884	valid_1's binary_logloss: 0.137762
    [57]	valid_0's auc: 0.883628	valid_0's binary_logloss: 0.11968	valid_1's auc: 0.833563	valid_1's binary_logloss: 0.13782
    [58]	valid_0's auc: 0.884097	valid_0's binary_logloss: 0.119546	valid_1's auc: 0.833647	valid_1's binary_logloss: 0.137781
    [59]	valid_0's auc: 0.884337	valid_0's binary_logloss: 0.119387	valid_1's auc: 0.833138	valid_1's binary_logloss: 0.137878
    [60]	valid_0's auc: 0.884777	valid_0's binary_logloss: 0.119188	valid_1's auc: 0.832909	valid_1's binary_logloss: 0.13796
    [61]	valid_0's auc: 0.885076	valid_0's binary_logloss: 0.119054	valid_1's auc: 0.833025	valid_1's binary_logloss: 0.137959
    Early stopping, best iteration is:
    [31]	valid_0's auc: 0.870612	valid_0's binary_logloss: 0.124704	valid_1's auc: 0.837541	valid_1's binary_logloss: 0.136757
    [1]	valid_0's auc: 0.825036	valid_0's binary_logloss: 0.157424	valid_1's auc: 0.807618	valid_1's binary_logloss: 0.159921
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.829804	valid_0's binary_logloss: 0.152132	valid_1's auc: 0.815884	valid_1's binary_logloss: 0.155124
    [3]	valid_0's auc: 0.835123	valid_0's binary_logloss: 0.148284	valid_1's auc: 0.81743	valid_1's binary_logloss: 0.151751
    [4]	valid_0's auc: 0.841579	valid_0's binary_logloss: 0.145192	valid_1's auc: 0.823237	valid_1's binary_logloss: 0.149159
    [5]	valid_0's auc: 0.843972	valid_0's binary_logloss: 0.142753	valid_1's auc: 0.82747	valid_1's binary_logloss: 0.147042
    [6]	valid_0's auc: 0.846408	valid_0's binary_logloss: 0.140687	valid_1's auc: 0.830402	valid_1's binary_logloss: 0.145398
    [7]	valid_0's auc: 0.848617	valid_0's binary_logloss: 0.138978	valid_1's auc: 0.830319	valid_1's binary_logloss: 0.144054
    [8]	valid_0's auc: 0.849973	valid_0's binary_logloss: 0.137527	valid_1's auc: 0.831414	valid_1's binary_logloss: 0.142955
    [9]	valid_0's auc: 0.852151	valid_0's binary_logloss: 0.136279	valid_1's auc: 0.834577	valid_1's binary_logloss: 0.141825
    [10]	valid_0's auc: 0.852962	valid_0's binary_logloss: 0.135231	valid_1's auc: 0.835301	valid_1's binary_logloss: 0.140947
    [11]	valid_0's auc: 0.853769	valid_0's binary_logloss: 0.134247	valid_1's auc: 0.836379	valid_1's binary_logloss: 0.140121
    [12]	valid_0's auc: 0.855739	valid_0's binary_logloss: 0.133341	valid_1's auc: 0.83615	valid_1's binary_logloss: 0.139629
    [13]	valid_0's auc: 0.857549	valid_0's binary_logloss: 0.132607	valid_1's auc: 0.836232	valid_1's binary_logloss: 0.139209
    [14]	valid_0's auc: 0.858955	valid_0's binary_logloss: 0.131867	valid_1's auc: 0.836387	valid_1's binary_logloss: 0.138827
    [15]	valid_0's auc: 0.860018	valid_0's binary_logloss: 0.13125	valid_1's auc: 0.837299	valid_1's binary_logloss: 0.138441
    [16]	valid_0's auc: 0.861059	valid_0's binary_logloss: 0.130679	valid_1's auc: 0.838197	valid_1's binary_logloss: 0.138074
    [17]	valid_0's auc: 0.862112	valid_0's binary_logloss: 0.130128	valid_1's auc: 0.838835	valid_1's binary_logloss: 0.137791
    [18]	valid_0's auc: 0.862994	valid_0's binary_logloss: 0.129646	valid_1's auc: 0.838904	valid_1's binary_logloss: 0.137577
    [19]	valid_0's auc: 0.863568	valid_0's binary_logloss: 0.129195	valid_1's auc: 0.839028	valid_1's binary_logloss: 0.137319
    [20]	valid_0's auc: 0.86472	valid_0's binary_logloss: 0.128729	valid_1's auc: 0.83984	valid_1's binary_logloss: 0.137089
    [21]	valid_0's auc: 0.866051	valid_0's binary_logloss: 0.128315	valid_1's auc: 0.840385	valid_1's binary_logloss: 0.136889
    [22]	valid_0's auc: 0.866997	valid_0's binary_logloss: 0.127908	valid_1's auc: 0.840544	valid_1's binary_logloss: 0.136724
    [23]	valid_0's auc: 0.867998	valid_0's binary_logloss: 0.127516	valid_1's auc: 0.840471	valid_1's binary_logloss: 0.136628
    [24]	valid_0's auc: 0.868835	valid_0's binary_logloss: 0.127186	valid_1's auc: 0.84011	valid_1's binary_logloss: 0.136573
    [25]	valid_0's auc: 0.869535	valid_0's binary_logloss: 0.126855	valid_1's auc: 0.83987	valid_1's binary_logloss: 0.136502
    [26]	valid_0's auc: 0.870439	valid_0's binary_logloss: 0.126515	valid_1's auc: 0.840538	valid_1's binary_logloss: 0.136387
    [27]	valid_0's auc: 0.871466	valid_0's binary_logloss: 0.126179	valid_1's auc: 0.840326	valid_1's binary_logloss: 0.136301
    [28]	valid_0's auc: 0.872022	valid_0's binary_logloss: 0.125897	valid_1's auc: 0.840415	valid_1's binary_logloss: 0.136253
    [29]	valid_0's auc: 0.873022	valid_0's binary_logloss: 0.125624	valid_1's auc: 0.840531	valid_1's binary_logloss: 0.136185
    [30]	valid_0's auc: 0.873626	valid_0's binary_logloss: 0.125329	valid_1's auc: 0.840407	valid_1's binary_logloss: 0.136188
    [31]	valid_0's auc: 0.874227	valid_0's binary_logloss: 0.12509	valid_1's auc: 0.841102	valid_1's binary_logloss: 0.136046
    [32]	valid_0's auc: 0.874828	valid_0's binary_logloss: 0.124856	valid_1's auc: 0.841125	valid_1's binary_logloss: 0.135986
    [33]	valid_0's auc: 0.875492	valid_0's binary_logloss: 0.124583	valid_1's auc: 0.841558	valid_1's binary_logloss: 0.135907
    [34]	valid_0's auc: 0.876123	valid_0's binary_logloss: 0.124343	valid_1's auc: 0.841695	valid_1's binary_logloss: 0.135865
    [35]	valid_0's auc: 0.876893	valid_0's binary_logloss: 0.124064	valid_1's auc: 0.841945	valid_1's binary_logloss: 0.135831
    [36]	valid_0's auc: 0.877421	valid_0's binary_logloss: 0.123823	valid_1's auc: 0.841949	valid_1's binary_logloss: 0.135864
    [37]	valid_0's auc: 0.877945	valid_0's binary_logloss: 0.123612	valid_1's auc: 0.842028	valid_1's binary_logloss: 0.135826
    [38]	valid_0's auc: 0.878391	valid_0's binary_logloss: 0.123437	valid_1's auc: 0.841969	valid_1's binary_logloss: 0.135815
    [39]	valid_0's auc: 0.878886	valid_0's binary_logloss: 0.123211	valid_1's auc: 0.84193	valid_1's binary_logloss: 0.135809
    [40]	valid_0's auc: 0.879487	valid_0's binary_logloss: 0.123003	valid_1's auc: 0.841846	valid_1's binary_logloss: 0.135816
    [41]	valid_0's auc: 0.880192	valid_0's binary_logloss: 0.12276	valid_1's auc: 0.841501	valid_1's binary_logloss: 0.135878
    [42]	valid_0's auc: 0.880817	valid_0's binary_logloss: 0.122555	valid_1's auc: 0.841224	valid_1's binary_logloss: 0.135894
    [43]	valid_0's auc: 0.881168	valid_0's binary_logloss: 0.122344	valid_1's auc: 0.841189	valid_1's binary_logloss: 0.135895
    [44]	valid_0's auc: 0.88156	valid_0's binary_logloss: 0.122151	valid_1's auc: 0.841136	valid_1's binary_logloss: 0.135914
    [45]	valid_0's auc: 0.882438	valid_0's binary_logloss: 0.121855	valid_1's auc: 0.840553	valid_1's binary_logloss: 0.136015
    [46]	valid_0's auc: 0.883226	valid_0's binary_logloss: 0.121617	valid_1's auc: 0.840057	valid_1's binary_logloss: 0.136084
    [47]	valid_0's auc: 0.883701	valid_0's binary_logloss: 0.121428	valid_1's auc: 0.839905	valid_1's binary_logloss: 0.13611
    [48]	valid_0's auc: 0.884405	valid_0's binary_logloss: 0.121174	valid_1's auc: 0.840003	valid_1's binary_logloss: 0.136077
    [49]	valid_0's auc: 0.8847	valid_0's binary_logloss: 0.12101	valid_1's auc: 0.839683	valid_1's binary_logloss: 0.136145
    [50]	valid_0's auc: 0.885111	valid_0's binary_logloss: 0.120814	valid_1's auc: 0.839749	valid_1's binary_logloss: 0.136099
    [51]	valid_0's auc: 0.885425	valid_0's binary_logloss: 0.12065	valid_1's auc: 0.839416	valid_1's binary_logloss: 0.13609
    [52]	valid_0's auc: 0.885826	valid_0's binary_logloss: 0.120466	valid_1's auc: 0.83916	valid_1's binary_logloss: 0.136158
    [53]	valid_0's auc: 0.88609	valid_0's binary_logloss: 0.120311	valid_1's auc: 0.839381	valid_1's binary_logloss: 0.136151
    [54]	valid_0's auc: 0.886458	valid_0's binary_logloss: 0.120122	valid_1's auc: 0.839222	valid_1's binary_logloss: 0.136197
    [55]	valid_0's auc: 0.887239	valid_0's binary_logloss: 0.11994	valid_1's auc: 0.839191	valid_1's binary_logloss: 0.136207
    [56]	valid_0's auc: 0.887885	valid_0's binary_logloss: 0.119726	valid_1's auc: 0.839099	valid_1's binary_logloss: 0.136256
    [57]	valid_0's auc: 0.888068	valid_0's binary_logloss: 0.119592	valid_1's auc: 0.83885	valid_1's binary_logloss: 0.136275
    [58]	valid_0's auc: 0.888634	valid_0's binary_logloss: 0.119455	valid_1's auc: 0.838851	valid_1's binary_logloss: 0.136269
    [59]	valid_0's auc: 0.888873	valid_0's binary_logloss: 0.119334	valid_1's auc: 0.838683	valid_1's binary_logloss: 0.13629
    [60]	valid_0's auc: 0.889172	valid_0's binary_logloss: 0.119194	valid_1's auc: 0.838562	valid_1's binary_logloss: 0.136301
    [61]	valid_0's auc: 0.889511	valid_0's binary_logloss: 0.119036	valid_1's auc: 0.838472	valid_1's binary_logloss: 0.136335
    [62]	valid_0's auc: 0.889884	valid_0's binary_logloss: 0.118877	valid_1's auc: 0.838355	valid_1's binary_logloss: 0.136343
    [63]	valid_0's auc: 0.890436	valid_0's binary_logloss: 0.118679	valid_1's auc: 0.838056	valid_1's binary_logloss: 0.136425
    [64]	valid_0's auc: 0.890662	valid_0's binary_logloss: 0.118521	valid_1's auc: 0.837425	valid_1's binary_logloss: 0.136547
    [65]	valid_0's auc: 0.890862	valid_0's binary_logloss: 0.118389	valid_1's auc: 0.837399	valid_1's binary_logloss: 0.13655
    [66]	valid_0's auc: 0.890931	valid_0's binary_logloss: 0.118288	valid_1's auc: 0.837035	valid_1's binary_logloss: 0.13665
    [67]	valid_0's auc: 0.891015	valid_0's binary_logloss: 0.118183	valid_1's auc: 0.836707	valid_1's binary_logloss: 0.136783
    Early stopping, best iteration is:
    [37]	valid_0's auc: 0.877945	valid_0's binary_logloss: 0.123612	valid_1's auc: 0.842028	valid_1's binary_logloss: 0.135826
    [1]	valid_0's auc: 0.823507	valid_0's binary_logloss: 0.157501	valid_1's auc: 0.810415	valid_1's binary_logloss: 0.160155
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.829581	valid_0's binary_logloss: 0.152083	valid_1's auc: 0.815091	valid_1's binary_logloss: 0.155417
    [3]	valid_0's auc: 0.834308	valid_0's binary_logloss: 0.148148	valid_1's auc: 0.818622	valid_1's binary_logloss: 0.152027
    [4]	valid_0's auc: 0.836859	valid_0's binary_logloss: 0.145159	valid_1's auc: 0.819436	valid_1's binary_logloss: 0.149485
    [5]	valid_0's auc: 0.841846	valid_0's binary_logloss: 0.142678	valid_1's auc: 0.823324	valid_1's binary_logloss: 0.147372
    [6]	valid_0's auc: 0.844625	valid_0's binary_logloss: 0.140636	valid_1's auc: 0.824092	valid_1's binary_logloss: 0.145715
    [7]	valid_0's auc: 0.848223	valid_0's binary_logloss: 0.138877	valid_1's auc: 0.826887	valid_1's binary_logloss: 0.144253
    [8]	valid_0's auc: 0.849635	valid_0's binary_logloss: 0.137406	valid_1's auc: 0.829164	valid_1's binary_logloss: 0.143039
    [9]	valid_0's auc: 0.850865	valid_0's binary_logloss: 0.136105	valid_1's auc: 0.828916	valid_1's binary_logloss: 0.142109
    [10]	valid_0's auc: 0.852877	valid_0's binary_logloss: 0.134962	valid_1's auc: 0.829565	valid_1's binary_logloss: 0.141319
    [11]	valid_0's auc: 0.853498	valid_0's binary_logloss: 0.134001	valid_1's auc: 0.829604	valid_1's binary_logloss: 0.140625
    [12]	valid_0's auc: 0.854677	valid_0's binary_logloss: 0.133137	valid_1's auc: 0.828978	valid_1's binary_logloss: 0.140133
    [13]	valid_0's auc: 0.856296	valid_0's binary_logloss: 0.132367	valid_1's auc: 0.830184	valid_1's binary_logloss: 0.139696
    [14]	valid_0's auc: 0.858364	valid_0's binary_logloss: 0.131592	valid_1's auc: 0.831576	valid_1's binary_logloss: 0.139209
    [15]	valid_0's auc: 0.859632	valid_0's binary_logloss: 0.130943	valid_1's auc: 0.833147	valid_1's binary_logloss: 0.138759
    [16]	valid_0's auc: 0.86077	valid_0's binary_logloss: 0.130306	valid_1's auc: 0.833392	valid_1's binary_logloss: 0.138459
    [17]	valid_0's auc: 0.861722	valid_0's binary_logloss: 0.129793	valid_1's auc: 0.834279	valid_1's binary_logloss: 0.138085
    [18]	valid_0's auc: 0.862253	valid_0's binary_logloss: 0.129343	valid_1's auc: 0.834323	valid_1's binary_logloss: 0.137866
    [19]	valid_0's auc: 0.863766	valid_0's binary_logloss: 0.128824	valid_1's auc: 0.834736	valid_1's binary_logloss: 0.13763
    [20]	valid_0's auc: 0.8643	valid_0's binary_logloss: 0.128423	valid_1's auc: 0.834528	valid_1's binary_logloss: 0.137554
    [21]	valid_0's auc: 0.865267	valid_0's binary_logloss: 0.127958	valid_1's auc: 0.834478	valid_1's binary_logloss: 0.137406
    [22]	valid_0's auc: 0.866608	valid_0's binary_logloss: 0.127506	valid_1's auc: 0.834905	valid_1's binary_logloss: 0.137271
    [23]	valid_0's auc: 0.867804	valid_0's binary_logloss: 0.127104	valid_1's auc: 0.834293	valid_1's binary_logloss: 0.137211
    [24]	valid_0's auc: 0.86846	valid_0's binary_logloss: 0.12677	valid_1's auc: 0.833842	valid_1's binary_logloss: 0.137176
    [25]	valid_0's auc: 0.869535	valid_0's binary_logloss: 0.126409	valid_1's auc: 0.834057	valid_1's binary_logloss: 0.13708
    [26]	valid_0's auc: 0.870946	valid_0's binary_logloss: 0.126026	valid_1's auc: 0.833797	valid_1's binary_logloss: 0.137021
    [27]	valid_0's auc: 0.871863	valid_0's binary_logloss: 0.12567	valid_1's auc: 0.83361	valid_1's binary_logloss: 0.13702
    [28]	valid_0's auc: 0.872503	valid_0's binary_logloss: 0.125332	valid_1's auc: 0.833415	valid_1's binary_logloss: 0.136948
    [29]	valid_0's auc: 0.873256	valid_0's binary_logloss: 0.125049	valid_1's auc: 0.83344	valid_1's binary_logloss: 0.136909
    [30]	valid_0's auc: 0.874265	valid_0's binary_logloss: 0.124767	valid_1's auc: 0.833129	valid_1's binary_logloss: 0.136924
    [31]	valid_0's auc: 0.875083	valid_0's binary_logloss: 0.124478	valid_1's auc: 0.833207	valid_1's binary_logloss: 0.136915
    [32]	valid_0's auc: 0.875789	valid_0's binary_logloss: 0.124224	valid_1's auc: 0.833079	valid_1's binary_logloss: 0.136916
    [33]	valid_0's auc: 0.876187	valid_0's binary_logloss: 0.123994	valid_1's auc: 0.833172	valid_1's binary_logloss: 0.136856
    [34]	valid_0's auc: 0.876733	valid_0's binary_logloss: 0.123732	valid_1's auc: 0.833412	valid_1's binary_logloss: 0.136808
    [35]	valid_0's auc: 0.877384	valid_0's binary_logloss: 0.123461	valid_1's auc: 0.833287	valid_1's binary_logloss: 0.136785
    [36]	valid_0's auc: 0.877802	valid_0's binary_logloss: 0.123212	valid_1's auc: 0.83302	valid_1's binary_logloss: 0.13685
    [37]	valid_0's auc: 0.878799	valid_0's binary_logloss: 0.122947	valid_1's auc: 0.832295	valid_1's binary_logloss: 0.136905
    [38]	valid_0's auc: 0.879503	valid_0's binary_logloss: 0.122719	valid_1's auc: 0.832134	valid_1's binary_logloss: 0.136919
    [39]	valid_0's auc: 0.880358	valid_0's binary_logloss: 0.122488	valid_1's auc: 0.832266	valid_1's binary_logloss: 0.136938
    [40]	valid_0's auc: 0.881013	valid_0's binary_logloss: 0.122287	valid_1's auc: 0.832133	valid_1's binary_logloss: 0.136932
    [41]	valid_0's auc: 0.881424	valid_0's binary_logloss: 0.122107	valid_1's auc: 0.832209	valid_1's binary_logloss: 0.136934
    [42]	valid_0's auc: 0.881984	valid_0's binary_logloss: 0.121921	valid_1's auc: 0.831997	valid_1's binary_logloss: 0.136976
    [43]	valid_0's auc: 0.882635	valid_0's binary_logloss: 0.121694	valid_1's auc: 0.831527	valid_1's binary_logloss: 0.137038
    [44]	valid_0's auc: 0.883243	valid_0's binary_logloss: 0.121523	valid_1's auc: 0.831593	valid_1's binary_logloss: 0.137055
    [45]	valid_0's auc: 0.883616	valid_0's binary_logloss: 0.121345	valid_1's auc: 0.831225	valid_1's binary_logloss: 0.137134
    [46]	valid_0's auc: 0.884106	valid_0's binary_logloss: 0.121155	valid_1's auc: 0.831678	valid_1's binary_logloss: 0.137064
    [47]	valid_0's auc: 0.884462	valid_0's binary_logloss: 0.120968	valid_1's auc: 0.832127	valid_1's binary_logloss: 0.137012
    [48]	valid_0's auc: 0.884804	valid_0's binary_logloss: 0.120754	valid_1's auc: 0.832474	valid_1's binary_logloss: 0.136991
    [49]	valid_0's auc: 0.885391	valid_0's binary_logloss: 0.120518	valid_1's auc: 0.832375	valid_1's binary_logloss: 0.137054
    [50]	valid_0's auc: 0.885988	valid_0's binary_logloss: 0.120287	valid_1's auc: 0.832272	valid_1's binary_logloss: 0.137065
    [51]	valid_0's auc: 0.886393	valid_0's binary_logloss: 0.120131	valid_1's auc: 0.831867	valid_1's binary_logloss: 0.137159
    [52]	valid_0's auc: 0.886879	valid_0's binary_logloss: 0.119924	valid_1's auc: 0.831971	valid_1's binary_logloss: 0.137133
    Early stopping, best iteration is:
    [22]	valid_0's auc: 0.866608	valid_0's binary_logloss: 0.127506	valid_1's auc: 0.834905	valid_1's binary_logloss: 0.137271
    [1]	valid_0's auc: 0.822135	valid_0's binary_logloss: 0.157275	valid_1's auc: 0.80763	valid_1's binary_logloss: 0.159959
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.826629	valid_0's binary_logloss: 0.15186	valid_1's auc: 0.811377	valid_1's binary_logloss: 0.155197
    [3]	valid_0's auc: 0.830813	valid_0's binary_logloss: 0.147949	valid_1's auc: 0.813917	valid_1's binary_logloss: 0.151946
    [4]	valid_0's auc: 0.833387	valid_0's binary_logloss: 0.144809	valid_1's auc: 0.814641	valid_1's binary_logloss: 0.149358
    [5]	valid_0's auc: 0.838116	valid_0's binary_logloss: 0.142332	valid_1's auc: 0.816965	valid_1's binary_logloss: 0.147361
    [6]	valid_0's auc: 0.841055	valid_0's binary_logloss: 0.140267	valid_1's auc: 0.81978	valid_1's binary_logloss: 0.145867
    [7]	valid_0's auc: 0.842021	valid_0's binary_logloss: 0.138495	valid_1's auc: 0.819689	valid_1's binary_logloss: 0.144575
    [8]	valid_0's auc: 0.843828	valid_0's binary_logloss: 0.137029	valid_1's auc: 0.822496	valid_1's binary_logloss: 0.143422
    [9]	valid_0's auc: 0.845522	valid_0's binary_logloss: 0.135783	valid_1's auc: 0.823288	valid_1's binary_logloss: 0.142428
    [10]	valid_0's auc: 0.847654	valid_0's binary_logloss: 0.134654	valid_1's auc: 0.824658	valid_1's binary_logloss: 0.141712
    [11]	valid_0's auc: 0.849302	valid_0's binary_logloss: 0.133679	valid_1's auc: 0.826788	valid_1's binary_logloss: 0.140983
    [12]	valid_0's auc: 0.850949	valid_0's binary_logloss: 0.132827	valid_1's auc: 0.828316	valid_1's binary_logloss: 0.140403
    [13]	valid_0's auc: 0.852852	valid_0's binary_logloss: 0.132062	valid_1's auc: 0.830745	valid_1's binary_logloss: 0.139849
    [14]	valid_0's auc: 0.854299	valid_0's binary_logloss: 0.131363	valid_1's auc: 0.832741	valid_1's binary_logloss: 0.139383
    [15]	valid_0's auc: 0.855376	valid_0's binary_logloss: 0.13071	valid_1's auc: 0.833869	valid_1's binary_logloss: 0.138963
    [16]	valid_0's auc: 0.856265	valid_0's binary_logloss: 0.130166	valid_1's auc: 0.833925	valid_1's binary_logloss: 0.138745
    [17]	valid_0's auc: 0.85735	valid_0's binary_logloss: 0.129615	valid_1's auc: 0.834227	valid_1's binary_logloss: 0.138474
    [18]	valid_0's auc: 0.859279	valid_0's binary_logloss: 0.129153	valid_1's auc: 0.834146	valid_1's binary_logloss: 0.138306
    [19]	valid_0's auc: 0.860099	valid_0's binary_logloss: 0.128684	valid_1's auc: 0.833391	valid_1's binary_logloss: 0.138142
    [20]	valid_0's auc: 0.861239	valid_0's binary_logloss: 0.128272	valid_1's auc: 0.833651	valid_1's binary_logloss: 0.13795
    [21]	valid_0's auc: 0.862153	valid_0's binary_logloss: 0.127883	valid_1's auc: 0.834662	valid_1's binary_logloss: 0.137829
    [22]	valid_0's auc: 0.863466	valid_0's binary_logloss: 0.127439	valid_1's auc: 0.834505	valid_1's binary_logloss: 0.137762
    [23]	valid_0's auc: 0.86379	valid_0's binary_logloss: 0.127135	valid_1's auc: 0.834377	valid_1's binary_logloss: 0.137657
    [24]	valid_0's auc: 0.864548	valid_0's binary_logloss: 0.126788	valid_1's auc: 0.834289	valid_1's binary_logloss: 0.137499
    [25]	valid_0's auc: 0.865681	valid_0's binary_logloss: 0.126451	valid_1's auc: 0.834899	valid_1's binary_logloss: 0.137329
    [26]	valid_0's auc: 0.866659	valid_0's binary_logloss: 0.126121	valid_1's auc: 0.835946	valid_1's binary_logloss: 0.137177
    [27]	valid_0's auc: 0.867505	valid_0's binary_logloss: 0.125853	valid_1's auc: 0.836616	valid_1's binary_logloss: 0.137016
    [28]	valid_0's auc: 0.868089	valid_0's binary_logloss: 0.125584	valid_1's auc: 0.836565	valid_1's binary_logloss: 0.137
    [29]	valid_0's auc: 0.868709	valid_0's binary_logloss: 0.125327	valid_1's auc: 0.837028	valid_1's binary_logloss: 0.136878
    [30]	valid_0's auc: 0.87006	valid_0's binary_logloss: 0.12501	valid_1's auc: 0.837512	valid_1's binary_logloss: 0.136792
    [31]	valid_0's auc: 0.870612	valid_0's binary_logloss: 0.124704	valid_1's auc: 0.837541	valid_1's binary_logloss: 0.136757
    [32]	valid_0's auc: 0.871234	valid_0's binary_logloss: 0.124431	valid_1's auc: 0.837141	valid_1's binary_logloss: 0.136755
    [33]	valid_0's auc: 0.872027	valid_0's binary_logloss: 0.124167	valid_1's auc: 0.837258	valid_1's binary_logloss: 0.136728
    [34]	valid_0's auc: 0.87244	valid_0's binary_logloss: 0.123956	valid_1's auc: 0.836829	valid_1's binary_logloss: 0.136799
    [35]	valid_0's auc: 0.873259	valid_0's binary_logloss: 0.123713	valid_1's auc: 0.836871	valid_1's binary_logloss: 0.136775
    [36]	valid_0's auc: 0.873695	valid_0's binary_logloss: 0.123497	valid_1's auc: 0.836512	valid_1's binary_logloss: 0.136859
    [37]	valid_0's auc: 0.874365	valid_0's binary_logloss: 0.123259	valid_1's auc: 0.836421	valid_1's binary_logloss: 0.136872
    [38]	valid_0's auc: 0.874998	valid_0's binary_logloss: 0.123014	valid_1's auc: 0.83628	valid_1's binary_logloss: 0.136939
    [39]	valid_0's auc: 0.875666	valid_0's binary_logloss: 0.122811	valid_1's auc: 0.836577	valid_1's binary_logloss: 0.13689
    [40]	valid_0's auc: 0.875931	valid_0's binary_logloss: 0.122638	valid_1's auc: 0.836148	valid_1's binary_logloss: 0.13697
    [41]	valid_0's auc: 0.876403	valid_0's binary_logloss: 0.122474	valid_1's auc: 0.836008	valid_1's binary_logloss: 0.137005
    [42]	valid_0's auc: 0.87697	valid_0's binary_logloss: 0.122306	valid_1's auc: 0.835814	valid_1's binary_logloss: 0.137031
    [43]	valid_0's auc: 0.877305	valid_0's binary_logloss: 0.122115	valid_1's auc: 0.835467	valid_1's binary_logloss: 0.137112
    [44]	valid_0's auc: 0.87782	valid_0's binary_logloss: 0.121932	valid_1's auc: 0.835339	valid_1's binary_logloss: 0.137217
    [45]	valid_0's auc: 0.878559	valid_0's binary_logloss: 0.121701	valid_1's auc: 0.83506	valid_1's binary_logloss: 0.137256
    [46]	valid_0's auc: 0.878895	valid_0's binary_logloss: 0.121564	valid_1's auc: 0.834955	valid_1's binary_logloss: 0.137255
    [47]	valid_0's auc: 0.879643	valid_0's binary_logloss: 0.121306	valid_1's auc: 0.834987	valid_1's binary_logloss: 0.137275
    [48]	valid_0's auc: 0.88002	valid_0's binary_logloss: 0.121119	valid_1's auc: 0.834864	valid_1's binary_logloss: 0.137329
    [49]	valid_0's auc: 0.880318	valid_0's binary_logloss: 0.120965	valid_1's auc: 0.834547	valid_1's binary_logloss: 0.137394
    [50]	valid_0's auc: 0.880967	valid_0's binary_logloss: 0.12082	valid_1's auc: 0.834588	valid_1's binary_logloss: 0.137388
    [51]	valid_0's auc: 0.8813	valid_0's binary_logloss: 0.120647	valid_1's auc: 0.834288	valid_1's binary_logloss: 0.137467
    [52]	valid_0's auc: 0.881706	valid_0's binary_logloss: 0.120482	valid_1's auc: 0.834305	valid_1's binary_logloss: 0.137491
    [53]	valid_0's auc: 0.882192	valid_0's binary_logloss: 0.1203	valid_1's auc: 0.834662	valid_1's binary_logloss: 0.137492
    [54]	valid_0's auc: 0.882307	valid_0's binary_logloss: 0.120188	valid_1's auc: 0.834241	valid_1's binary_logloss: 0.13763
    [55]	valid_0's auc: 0.883062	valid_0's binary_logloss: 0.119929	valid_1's auc: 0.834145	valid_1's binary_logloss: 0.137674
    [56]	valid_0's auc: 0.883272	valid_0's binary_logloss: 0.119823	valid_1's auc: 0.833884	valid_1's binary_logloss: 0.137762
    [57]	valid_0's auc: 0.883628	valid_0's binary_logloss: 0.11968	valid_1's auc: 0.833563	valid_1's binary_logloss: 0.13782
    [58]	valid_0's auc: 0.884097	valid_0's binary_logloss: 0.119546	valid_1's auc: 0.833647	valid_1's binary_logloss: 0.137781
    [59]	valid_0's auc: 0.884337	valid_0's binary_logloss: 0.119387	valid_1's auc: 0.833138	valid_1's binary_logloss: 0.137878
    [60]	valid_0's auc: 0.884777	valid_0's binary_logloss: 0.119188	valid_1's auc: 0.832909	valid_1's binary_logloss: 0.13796
    [61]	valid_0's auc: 0.885076	valid_0's binary_logloss: 0.119054	valid_1's auc: 0.833025	valid_1's binary_logloss: 0.137959
    Early stopping, best iteration is:
    [31]	valid_0's auc: 0.870612	valid_0's binary_logloss: 0.124704	valid_1's auc: 0.837541	valid_1's binary_logloss: 0.136757
    [1]	valid_0's auc: 0.825036	valid_0's binary_logloss: 0.157424	valid_1's auc: 0.807618	valid_1's binary_logloss: 0.159921
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.829804	valid_0's binary_logloss: 0.152132	valid_1's auc: 0.815884	valid_1's binary_logloss: 0.155124
    [3]	valid_0's auc: 0.835123	valid_0's binary_logloss: 0.148284	valid_1's auc: 0.81743	valid_1's binary_logloss: 0.151751
    [4]	valid_0's auc: 0.841579	valid_0's binary_logloss: 0.145192	valid_1's auc: 0.823237	valid_1's binary_logloss: 0.149159
    [5]	valid_0's auc: 0.843972	valid_0's binary_logloss: 0.142753	valid_1's auc: 0.82747	valid_1's binary_logloss: 0.147042
    [6]	valid_0's auc: 0.846408	valid_0's binary_logloss: 0.140687	valid_1's auc: 0.830402	valid_1's binary_logloss: 0.145398
    [7]	valid_0's auc: 0.848617	valid_0's binary_logloss: 0.138978	valid_1's auc: 0.830319	valid_1's binary_logloss: 0.144054
    [8]	valid_0's auc: 0.849973	valid_0's binary_logloss: 0.137527	valid_1's auc: 0.831414	valid_1's binary_logloss: 0.142955
    [9]	valid_0's auc: 0.852151	valid_0's binary_logloss: 0.136279	valid_1's auc: 0.834577	valid_1's binary_logloss: 0.141825
    [10]	valid_0's auc: 0.852962	valid_0's binary_logloss: 0.135231	valid_1's auc: 0.835301	valid_1's binary_logloss: 0.140947
    [11]	valid_0's auc: 0.853769	valid_0's binary_logloss: 0.134247	valid_1's auc: 0.836379	valid_1's binary_logloss: 0.140121
    [12]	valid_0's auc: 0.855739	valid_0's binary_logloss: 0.133341	valid_1's auc: 0.83615	valid_1's binary_logloss: 0.139629
    [13]	valid_0's auc: 0.857549	valid_0's binary_logloss: 0.132607	valid_1's auc: 0.836232	valid_1's binary_logloss: 0.139209
    [14]	valid_0's auc: 0.858955	valid_0's binary_logloss: 0.131867	valid_1's auc: 0.836387	valid_1's binary_logloss: 0.138827
    [15]	valid_0's auc: 0.860018	valid_0's binary_logloss: 0.13125	valid_1's auc: 0.837299	valid_1's binary_logloss: 0.138441
    [16]	valid_0's auc: 0.861059	valid_0's binary_logloss: 0.130679	valid_1's auc: 0.838197	valid_1's binary_logloss: 0.138074
    [17]	valid_0's auc: 0.862112	valid_0's binary_logloss: 0.130128	valid_1's auc: 0.838835	valid_1's binary_logloss: 0.137791
    [18]	valid_0's auc: 0.862994	valid_0's binary_logloss: 0.129646	valid_1's auc: 0.838904	valid_1's binary_logloss: 0.137577
    [19]	valid_0's auc: 0.863568	valid_0's binary_logloss: 0.129195	valid_1's auc: 0.839028	valid_1's binary_logloss: 0.137319
    [20]	valid_0's auc: 0.86472	valid_0's binary_logloss: 0.128729	valid_1's auc: 0.83984	valid_1's binary_logloss: 0.137089
    [21]	valid_0's auc: 0.866051	valid_0's binary_logloss: 0.128315	valid_1's auc: 0.840385	valid_1's binary_logloss: 0.136889
    [22]	valid_0's auc: 0.866997	valid_0's binary_logloss: 0.127908	valid_1's auc: 0.840544	valid_1's binary_logloss: 0.136724
    [23]	valid_0's auc: 0.867998	valid_0's binary_logloss: 0.127516	valid_1's auc: 0.840471	valid_1's binary_logloss: 0.136628
    [24]	valid_0's auc: 0.868835	valid_0's binary_logloss: 0.127186	valid_1's auc: 0.84011	valid_1's binary_logloss: 0.136573
    [25]	valid_0's auc: 0.869535	valid_0's binary_logloss: 0.126855	valid_1's auc: 0.83987	valid_1's binary_logloss: 0.136502
    [26]	valid_0's auc: 0.870439	valid_0's binary_logloss: 0.126515	valid_1's auc: 0.840538	valid_1's binary_logloss: 0.136387
    [27]	valid_0's auc: 0.871466	valid_0's binary_logloss: 0.126179	valid_1's auc: 0.840326	valid_1's binary_logloss: 0.136301
    [28]	valid_0's auc: 0.872022	valid_0's binary_logloss: 0.125897	valid_1's auc: 0.840415	valid_1's binary_logloss: 0.136253
    [29]	valid_0's auc: 0.873022	valid_0's binary_logloss: 0.125624	valid_1's auc: 0.840531	valid_1's binary_logloss: 0.136185
    [30]	valid_0's auc: 0.873626	valid_0's binary_logloss: 0.125329	valid_1's auc: 0.840407	valid_1's binary_logloss: 0.136188
    [31]	valid_0's auc: 0.874227	valid_0's binary_logloss: 0.12509	valid_1's auc: 0.841102	valid_1's binary_logloss: 0.136046
    [32]	valid_0's auc: 0.874828	valid_0's binary_logloss: 0.124856	valid_1's auc: 0.841125	valid_1's binary_logloss: 0.135986
    [33]	valid_0's auc: 0.875492	valid_0's binary_logloss: 0.124583	valid_1's auc: 0.841558	valid_1's binary_logloss: 0.135907
    [34]	valid_0's auc: 0.876123	valid_0's binary_logloss: 0.124343	valid_1's auc: 0.841695	valid_1's binary_logloss: 0.135865
    [35]	valid_0's auc: 0.876893	valid_0's binary_logloss: 0.124064	valid_1's auc: 0.841945	valid_1's binary_logloss: 0.135831
    [36]	valid_0's auc: 0.877421	valid_0's binary_logloss: 0.123823	valid_1's auc: 0.841949	valid_1's binary_logloss: 0.135864
    [37]	valid_0's auc: 0.877945	valid_0's binary_logloss: 0.123612	valid_1's auc: 0.842028	valid_1's binary_logloss: 0.135826
    [38]	valid_0's auc: 0.878391	valid_0's binary_logloss: 0.123437	valid_1's auc: 0.841969	valid_1's binary_logloss: 0.135815
    [39]	valid_0's auc: 0.878886	valid_0's binary_logloss: 0.123211	valid_1's auc: 0.84193	valid_1's binary_logloss: 0.135809
    [40]	valid_0's auc: 0.879487	valid_0's binary_logloss: 0.123003	valid_1's auc: 0.841846	valid_1's binary_logloss: 0.135816
    [41]	valid_0's auc: 0.880192	valid_0's binary_logloss: 0.12276	valid_1's auc: 0.841501	valid_1's binary_logloss: 0.135878
    [42]	valid_0's auc: 0.880817	valid_0's binary_logloss: 0.122555	valid_1's auc: 0.841224	valid_1's binary_logloss: 0.135894
    [43]	valid_0's auc: 0.881168	valid_0's binary_logloss: 0.122344	valid_1's auc: 0.841189	valid_1's binary_logloss: 0.135895
    [44]	valid_0's auc: 0.88156	valid_0's binary_logloss: 0.122151	valid_1's auc: 0.841136	valid_1's binary_logloss: 0.135914
    [45]	valid_0's auc: 0.882438	valid_0's binary_logloss: 0.121855	valid_1's auc: 0.840553	valid_1's binary_logloss: 0.136015
    [46]	valid_0's auc: 0.883226	valid_0's binary_logloss: 0.121617	valid_1's auc: 0.840057	valid_1's binary_logloss: 0.136084
    [47]	valid_0's auc: 0.883701	valid_0's binary_logloss: 0.121428	valid_1's auc: 0.839905	valid_1's binary_logloss: 0.13611
    [48]	valid_0's auc: 0.884405	valid_0's binary_logloss: 0.121174	valid_1's auc: 0.840003	valid_1's binary_logloss: 0.136077
    [49]	valid_0's auc: 0.8847	valid_0's binary_logloss: 0.12101	valid_1's auc: 0.839683	valid_1's binary_logloss: 0.136145
    [50]	valid_0's auc: 0.885111	valid_0's binary_logloss: 0.120814	valid_1's auc: 0.839749	valid_1's binary_logloss: 0.136099
    [51]	valid_0's auc: 0.885425	valid_0's binary_logloss: 0.12065	valid_1's auc: 0.839416	valid_1's binary_logloss: 0.13609
    [52]	valid_0's auc: 0.885826	valid_0's binary_logloss: 0.120466	valid_1's auc: 0.83916	valid_1's binary_logloss: 0.136158
    [53]	valid_0's auc: 0.88609	valid_0's binary_logloss: 0.120311	valid_1's auc: 0.839381	valid_1's binary_logloss: 0.136151
    [54]	valid_0's auc: 0.886458	valid_0's binary_logloss: 0.120122	valid_1's auc: 0.839222	valid_1's binary_logloss: 0.136197
    [55]	valid_0's auc: 0.887239	valid_0's binary_logloss: 0.11994	valid_1's auc: 0.839191	valid_1's binary_logloss: 0.136207
    [56]	valid_0's auc: 0.887885	valid_0's binary_logloss: 0.119726	valid_1's auc: 0.839099	valid_1's binary_logloss: 0.136256
    [57]	valid_0's auc: 0.888068	valid_0's binary_logloss: 0.119592	valid_1's auc: 0.83885	valid_1's binary_logloss: 0.136275
    [58]	valid_0's auc: 0.888634	valid_0's binary_logloss: 0.119455	valid_1's auc: 0.838851	valid_1's binary_logloss: 0.136269
    [59]	valid_0's auc: 0.888873	valid_0's binary_logloss: 0.119334	valid_1's auc: 0.838683	valid_1's binary_logloss: 0.13629
    [60]	valid_0's auc: 0.889172	valid_0's binary_logloss: 0.119194	valid_1's auc: 0.838562	valid_1's binary_logloss: 0.136301
    [61]	valid_0's auc: 0.889511	valid_0's binary_logloss: 0.119036	valid_1's auc: 0.838472	valid_1's binary_logloss: 0.136335
    [62]	valid_0's auc: 0.889884	valid_0's binary_logloss: 0.118877	valid_1's auc: 0.838355	valid_1's binary_logloss: 0.136343
    [63]	valid_0's auc: 0.890436	valid_0's binary_logloss: 0.118679	valid_1's auc: 0.838056	valid_1's binary_logloss: 0.136425
    [64]	valid_0's auc: 0.890662	valid_0's binary_logloss: 0.118521	valid_1's auc: 0.837425	valid_1's binary_logloss: 0.136547
    [65]	valid_0's auc: 0.890862	valid_0's binary_logloss: 0.118389	valid_1's auc: 0.837399	valid_1's binary_logloss: 0.13655
    [66]	valid_0's auc: 0.890931	valid_0's binary_logloss: 0.118288	valid_1's auc: 0.837035	valid_1's binary_logloss: 0.13665
    [67]	valid_0's auc: 0.891015	valid_0's binary_logloss: 0.118183	valid_1's auc: 0.836707	valid_1's binary_logloss: 0.136783
    Early stopping, best iteration is:
    [37]	valid_0's auc: 0.877945	valid_0's binary_logloss: 0.123612	valid_1's auc: 0.842028	valid_1's binary_logloss: 0.135826
    [1]	valid_0's auc: 0.823507	valid_0's binary_logloss: 0.157501	valid_1's auc: 0.810415	valid_1's binary_logloss: 0.160155
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.829581	valid_0's binary_logloss: 0.152083	valid_1's auc: 0.815091	valid_1's binary_logloss: 0.155417
    [3]	valid_0's auc: 0.834308	valid_0's binary_logloss: 0.148148	valid_1's auc: 0.818622	valid_1's binary_logloss: 0.152027
    [4]	valid_0's auc: 0.836859	valid_0's binary_logloss: 0.145159	valid_1's auc: 0.819436	valid_1's binary_logloss: 0.149485
    [5]	valid_0's auc: 0.841846	valid_0's binary_logloss: 0.142678	valid_1's auc: 0.823324	valid_1's binary_logloss: 0.147372
    [6]	valid_0's auc: 0.844625	valid_0's binary_logloss: 0.140636	valid_1's auc: 0.824092	valid_1's binary_logloss: 0.145715
    [7]	valid_0's auc: 0.848223	valid_0's binary_logloss: 0.138877	valid_1's auc: 0.826887	valid_1's binary_logloss: 0.144253
    [8]	valid_0's auc: 0.849635	valid_0's binary_logloss: 0.137406	valid_1's auc: 0.829164	valid_1's binary_logloss: 0.143039
    [9]	valid_0's auc: 0.850865	valid_0's binary_logloss: 0.136105	valid_1's auc: 0.828916	valid_1's binary_logloss: 0.142109
    [10]	valid_0's auc: 0.852877	valid_0's binary_logloss: 0.134962	valid_1's auc: 0.829565	valid_1's binary_logloss: 0.141319
    [11]	valid_0's auc: 0.853498	valid_0's binary_logloss: 0.134001	valid_1's auc: 0.829604	valid_1's binary_logloss: 0.140625
    [12]	valid_0's auc: 0.854677	valid_0's binary_logloss: 0.133137	valid_1's auc: 0.828978	valid_1's binary_logloss: 0.140133
    [13]	valid_0's auc: 0.856296	valid_0's binary_logloss: 0.132367	valid_1's auc: 0.830184	valid_1's binary_logloss: 0.139696
    [14]	valid_0's auc: 0.858364	valid_0's binary_logloss: 0.131592	valid_1's auc: 0.831576	valid_1's binary_logloss: 0.139209
    [15]	valid_0's auc: 0.859632	valid_0's binary_logloss: 0.130943	valid_1's auc: 0.833147	valid_1's binary_logloss: 0.138759
    [16]	valid_0's auc: 0.86077	valid_0's binary_logloss: 0.130306	valid_1's auc: 0.833392	valid_1's binary_logloss: 0.138459
    [17]	valid_0's auc: 0.861722	valid_0's binary_logloss: 0.129793	valid_1's auc: 0.834279	valid_1's binary_logloss: 0.138085
    [18]	valid_0's auc: 0.862253	valid_0's binary_logloss: 0.129343	valid_1's auc: 0.834323	valid_1's binary_logloss: 0.137866
    [19]	valid_0's auc: 0.863766	valid_0's binary_logloss: 0.128824	valid_1's auc: 0.834736	valid_1's binary_logloss: 0.13763
    [20]	valid_0's auc: 0.8643	valid_0's binary_logloss: 0.128423	valid_1's auc: 0.834528	valid_1's binary_logloss: 0.137554
    [21]	valid_0's auc: 0.865267	valid_0's binary_logloss: 0.127958	valid_1's auc: 0.834478	valid_1's binary_logloss: 0.137406
    [22]	valid_0's auc: 0.866608	valid_0's binary_logloss: 0.127506	valid_1's auc: 0.834905	valid_1's binary_logloss: 0.137271
    [23]	valid_0's auc: 0.867804	valid_0's binary_logloss: 0.127104	valid_1's auc: 0.834293	valid_1's binary_logloss: 0.137211
    [24]	valid_0's auc: 0.86846	valid_0's binary_logloss: 0.12677	valid_1's auc: 0.833842	valid_1's binary_logloss: 0.137176
    [25]	valid_0's auc: 0.869535	valid_0's binary_logloss: 0.126409	valid_1's auc: 0.834057	valid_1's binary_logloss: 0.13708
    [26]	valid_0's auc: 0.870946	valid_0's binary_logloss: 0.126026	valid_1's auc: 0.833797	valid_1's binary_logloss: 0.137021
    [27]	valid_0's auc: 0.871863	valid_0's binary_logloss: 0.12567	valid_1's auc: 0.83361	valid_1's binary_logloss: 0.13702
    [28]	valid_0's auc: 0.872503	valid_0's binary_logloss: 0.125332	valid_1's auc: 0.833415	valid_1's binary_logloss: 0.136948
    [29]	valid_0's auc: 0.873256	valid_0's binary_logloss: 0.125049	valid_1's auc: 0.83344	valid_1's binary_logloss: 0.136909
    [30]	valid_0's auc: 0.874265	valid_0's binary_logloss: 0.124767	valid_1's auc: 0.833129	valid_1's binary_logloss: 0.136924
    [31]	valid_0's auc: 0.875083	valid_0's binary_logloss: 0.124478	valid_1's auc: 0.833207	valid_1's binary_logloss: 0.136915
    [32]	valid_0's auc: 0.875789	valid_0's binary_logloss: 0.124224	valid_1's auc: 0.833079	valid_1's binary_logloss: 0.136916
    [33]	valid_0's auc: 0.876187	valid_0's binary_logloss: 0.123994	valid_1's auc: 0.833172	valid_1's binary_logloss: 0.136856
    [34]	valid_0's auc: 0.876733	valid_0's binary_logloss: 0.123732	valid_1's auc: 0.833412	valid_1's binary_logloss: 0.136808
    [35]	valid_0's auc: 0.877384	valid_0's binary_logloss: 0.123461	valid_1's auc: 0.833287	valid_1's binary_logloss: 0.136785
    [36]	valid_0's auc: 0.877802	valid_0's binary_logloss: 0.123212	valid_1's auc: 0.83302	valid_1's binary_logloss: 0.13685
    [37]	valid_0's auc: 0.878799	valid_0's binary_logloss: 0.122947	valid_1's auc: 0.832295	valid_1's binary_logloss: 0.136905
    [38]	valid_0's auc: 0.879503	valid_0's binary_logloss: 0.122719	valid_1's auc: 0.832134	valid_1's binary_logloss: 0.136919
    [39]	valid_0's auc: 0.880358	valid_0's binary_logloss: 0.122488	valid_1's auc: 0.832266	valid_1's binary_logloss: 0.136938
    [40]	valid_0's auc: 0.881013	valid_0's binary_logloss: 0.122287	valid_1's auc: 0.832133	valid_1's binary_logloss: 0.136932
    [41]	valid_0's auc: 0.881424	valid_0's binary_logloss: 0.122107	valid_1's auc: 0.832209	valid_1's binary_logloss: 0.136934
    [42]	valid_0's auc: 0.881984	valid_0's binary_logloss: 0.121921	valid_1's auc: 0.831997	valid_1's binary_logloss: 0.136976
    [43]	valid_0's auc: 0.882635	valid_0's binary_logloss: 0.121694	valid_1's auc: 0.831527	valid_1's binary_logloss: 0.137038
    [44]	valid_0's auc: 0.883243	valid_0's binary_logloss: 0.121523	valid_1's auc: 0.831593	valid_1's binary_logloss: 0.137055
    [45]	valid_0's auc: 0.883616	valid_0's binary_logloss: 0.121345	valid_1's auc: 0.831225	valid_1's binary_logloss: 0.137134
    [46]	valid_0's auc: 0.884106	valid_0's binary_logloss: 0.121155	valid_1's auc: 0.831678	valid_1's binary_logloss: 0.137064
    [47]	valid_0's auc: 0.884462	valid_0's binary_logloss: 0.120968	valid_1's auc: 0.832127	valid_1's binary_logloss: 0.137012
    [48]	valid_0's auc: 0.884804	valid_0's binary_logloss: 0.120754	valid_1's auc: 0.832474	valid_1's binary_logloss: 0.136991
    [49]	valid_0's auc: 0.885391	valid_0's binary_logloss: 0.120518	valid_1's auc: 0.832375	valid_1's binary_logloss: 0.137054
    [50]	valid_0's auc: 0.885988	valid_0's binary_logloss: 0.120287	valid_1's auc: 0.832272	valid_1's binary_logloss: 0.137065
    [51]	valid_0's auc: 0.886393	valid_0's binary_logloss: 0.120131	valid_1's auc: 0.831867	valid_1's binary_logloss: 0.137159
    [52]	valid_0's auc: 0.886879	valid_0's binary_logloss: 0.119924	valid_1's auc: 0.831971	valid_1's binary_logloss: 0.137133
    Early stopping, best iteration is:
    [22]	valid_0's auc: 0.866608	valid_0's binary_logloss: 0.127506	valid_1's auc: 0.834905	valid_1's binary_logloss: 0.137271
    [1]	valid_0's auc: 0.825869	valid_0's binary_logloss: 0.156436	valid_1's auc: 0.803397	valid_1's binary_logloss: 0.159993
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.834474	valid_0's binary_logloss: 0.150597	valid_1's auc: 0.809972	valid_1's binary_logloss: 0.155122
    [3]	valid_0's auc: 0.840511	valid_0's binary_logloss: 0.146374	valid_1's auc: 0.819137	valid_1's binary_logloss: 0.15148
    [4]	valid_0's auc: 0.844283	valid_0's binary_logloss: 0.143027	valid_1's auc: 0.820472	valid_1's binary_logloss: 0.14904
    [5]	valid_0's auc: 0.85046	valid_0's binary_logloss: 0.140321	valid_1's auc: 0.82606	valid_1's binary_logloss: 0.146982
    [6]	valid_0's auc: 0.852881	valid_0's binary_logloss: 0.13813	valid_1's auc: 0.823869	valid_1's binary_logloss: 0.145517
    [7]	valid_0's auc: 0.857268	valid_0's binary_logloss: 0.136149	valid_1's auc: 0.827586	valid_1's binary_logloss: 0.144156
    [8]	valid_0's auc: 0.859401	valid_0's binary_logloss: 0.13452	valid_1's auc: 0.829428	valid_1's binary_logloss: 0.14303
    [9]	valid_0's auc: 0.861435	valid_0's binary_logloss: 0.133011	valid_1's auc: 0.82971	valid_1's binary_logloss: 0.142113
    [10]	valid_0's auc: 0.862282	valid_0's binary_logloss: 0.131742	valid_1's auc: 0.830527	valid_1's binary_logloss: 0.141392
    [11]	valid_0's auc: 0.86448	valid_0's binary_logloss: 0.130536	valid_1's auc: 0.83065	valid_1's binary_logloss: 0.140854
    [12]	valid_0's auc: 0.865909	valid_0's binary_logloss: 0.129495	valid_1's auc: 0.830763	valid_1's binary_logloss: 0.140425
    [13]	valid_0's auc: 0.867083	valid_0's binary_logloss: 0.128537	valid_1's auc: 0.830097	valid_1's binary_logloss: 0.140034
    [14]	valid_0's auc: 0.869164	valid_0's binary_logloss: 0.127593	valid_1's auc: 0.831242	valid_1's binary_logloss: 0.139589
    [15]	valid_0's auc: 0.869779	valid_0's binary_logloss: 0.126753	valid_1's auc: 0.830054	valid_1's binary_logloss: 0.13935
    [16]	valid_0's auc: 0.870798	valid_0's binary_logloss: 0.125998	valid_1's auc: 0.829854	valid_1's binary_logloss: 0.139147
    [17]	valid_0's auc: 0.872413	valid_0's binary_logloss: 0.125321	valid_1's auc: 0.829939	valid_1's binary_logloss: 0.138859
    [18]	valid_0's auc: 0.873656	valid_0's binary_logloss: 0.124646	valid_1's auc: 0.83033	valid_1's binary_logloss: 0.138678
    [19]	valid_0's auc: 0.874787	valid_0's binary_logloss: 0.124094	valid_1's auc: 0.829747	valid_1's binary_logloss: 0.138635
    [20]	valid_0's auc: 0.875793	valid_0's binary_logloss: 0.123547	valid_1's auc: 0.829455	valid_1's binary_logloss: 0.138492
    [21]	valid_0's auc: 0.87694	valid_0's binary_logloss: 0.122969	valid_1's auc: 0.830945	valid_1's binary_logloss: 0.13825
    [22]	valid_0's auc: 0.878632	valid_0's binary_logloss: 0.122407	valid_1's auc: 0.831483	valid_1's binary_logloss: 0.138097
    [23]	valid_0's auc: 0.8796	valid_0's binary_logloss: 0.121869	valid_1's auc: 0.831408	valid_1's binary_logloss: 0.138071
    [24]	valid_0's auc: 0.881298	valid_0's binary_logloss: 0.121363	valid_1's auc: 0.831984	valid_1's binary_logloss: 0.137912
    [25]	valid_0's auc: 0.882699	valid_0's binary_logloss: 0.120863	valid_1's auc: 0.831941	valid_1's binary_logloss: 0.137824
    [26]	valid_0's auc: 0.883799	valid_0's binary_logloss: 0.120379	valid_1's auc: 0.832897	valid_1's binary_logloss: 0.137649
    [27]	valid_0's auc: 0.884893	valid_0's binary_logloss: 0.119944	valid_1's auc: 0.832841	valid_1's binary_logloss: 0.13763
    [28]	valid_0's auc: 0.88588	valid_0's binary_logloss: 0.119545	valid_1's auc: 0.833413	valid_1's binary_logloss: 0.137495
    [29]	valid_0's auc: 0.886461	valid_0's binary_logloss: 0.11916	valid_1's auc: 0.833437	valid_1's binary_logloss: 0.137511
    [30]	valid_0's auc: 0.887592	valid_0's binary_logloss: 0.118739	valid_1's auc: 0.833429	valid_1's binary_logloss: 0.137514
    [31]	valid_0's auc: 0.888552	valid_0's binary_logloss: 0.118399	valid_1's auc: 0.834036	valid_1's binary_logloss: 0.137459
    [32]	valid_0's auc: 0.889448	valid_0's binary_logloss: 0.118019	valid_1's auc: 0.833754	valid_1's binary_logloss: 0.137498
    [33]	valid_0's auc: 0.890203	valid_0's binary_logloss: 0.117711	valid_1's auc: 0.833853	valid_1's binary_logloss: 0.137491
    [34]	valid_0's auc: 0.890868	valid_0's binary_logloss: 0.117395	valid_1's auc: 0.833648	valid_1's binary_logloss: 0.137532
    [35]	valid_0's auc: 0.891378	valid_0's binary_logloss: 0.117106	valid_1's auc: 0.833479	valid_1's binary_logloss: 0.137577
    [36]	valid_0's auc: 0.892061	valid_0's binary_logloss: 0.116741	valid_1's auc: 0.832608	valid_1's binary_logloss: 0.137799
    [37]	valid_0's auc: 0.892715	valid_0's binary_logloss: 0.116403	valid_1's auc: 0.83208	valid_1's binary_logloss: 0.137938
    [38]	valid_0's auc: 0.893146	valid_0's binary_logloss: 0.116126	valid_1's auc: 0.83196	valid_1's binary_logloss: 0.137994
    [39]	valid_0's auc: 0.893963	valid_0's binary_logloss: 0.115783	valid_1's auc: 0.831584	valid_1's binary_logloss: 0.138076
    [40]	valid_0's auc: 0.894274	valid_0's binary_logloss: 0.115535	valid_1's auc: 0.831346	valid_1's binary_logloss: 0.13818
    [41]	valid_0's auc: 0.894809	valid_0's binary_logloss: 0.115247	valid_1's auc: 0.831197	valid_1's binary_logloss: 0.138289
    [42]	valid_0's auc: 0.895332	valid_0's binary_logloss: 0.114916	valid_1's auc: 0.830893	valid_1's binary_logloss: 0.138343
    [43]	valid_0's auc: 0.895864	valid_0's binary_logloss: 0.114701	valid_1's auc: 0.83129	valid_1's binary_logloss: 0.138307
    [44]	valid_0's auc: 0.89623	valid_0's binary_logloss: 0.114411	valid_1's auc: 0.831004	valid_1's binary_logloss: 0.138408
    [45]	valid_0's auc: 0.896616	valid_0's binary_logloss: 0.11413	valid_1's auc: 0.830369	valid_1's binary_logloss: 0.138547
    [46]	valid_0's auc: 0.897076	valid_0's binary_logloss: 0.113919	valid_1's auc: 0.830246	valid_1's binary_logloss: 0.138644
    [47]	valid_0's auc: 0.897393	valid_0's binary_logloss: 0.113691	valid_1's auc: 0.829999	valid_1's binary_logloss: 0.138769
    [48]	valid_0's auc: 0.897745	valid_0's binary_logloss: 0.113465	valid_1's auc: 0.829593	valid_1's binary_logloss: 0.138908
    [49]	valid_0's auc: 0.898203	valid_0's binary_logloss: 0.113214	valid_1's auc: 0.829209	valid_1's binary_logloss: 0.139045
    [50]	valid_0's auc: 0.898546	valid_0's binary_logloss: 0.112978	valid_1's auc: 0.829064	valid_1's binary_logloss: 0.139115
    [51]	valid_0's auc: 0.898958	valid_0's binary_logloss: 0.112723	valid_1's auc: 0.82913	valid_1's binary_logloss: 0.139163
    [52]	valid_0's auc: 0.899296	valid_0's binary_logloss: 0.112541	valid_1's auc: 0.828532	valid_1's binary_logloss: 0.139358
    [53]	valid_0's auc: 0.899966	valid_0's binary_logloss: 0.112208	valid_1's auc: 0.828277	valid_1's binary_logloss: 0.139431
    [54]	valid_0's auc: 0.900225	valid_0's binary_logloss: 0.112001	valid_1's auc: 0.827866	valid_1's binary_logloss: 0.139573
    [55]	valid_0's auc: 0.90036	valid_0's binary_logloss: 0.111812	valid_1's auc: 0.827887	valid_1's binary_logloss: 0.139655
    [56]	valid_0's auc: 0.900528	valid_0's binary_logloss: 0.111638	valid_1's auc: 0.827809	valid_1's binary_logloss: 0.139702
    [57]	valid_0's auc: 0.900692	valid_0's binary_logloss: 0.111452	valid_1's auc: 0.827613	valid_1's binary_logloss: 0.139758
    [58]	valid_0's auc: 0.901275	valid_0's binary_logloss: 0.111262	valid_1's auc: 0.827848	valid_1's binary_logloss: 0.139758
    [59]	valid_0's auc: 0.90149	valid_0's binary_logloss: 0.111086	valid_1's auc: 0.827643	valid_1's binary_logloss: 0.139835
    [60]	valid_0's auc: 0.901669	valid_0's binary_logloss: 0.110873	valid_1's auc: 0.827187	valid_1's binary_logloss: 0.13998
    [61]	valid_0's auc: 0.901715	valid_0's binary_logloss: 0.110743	valid_1's auc: 0.826643	valid_1's binary_logloss: 0.140136
    Early stopping, best iteration is:
    [31]	valid_0's auc: 0.888552	valid_0's binary_logloss: 0.118399	valid_1's auc: 0.834036	valid_1's binary_logloss: 0.137459
    [1]	valid_0's auc: 0.828293	valid_0's binary_logloss: 0.156591	valid_1's auc: 0.803729	valid_1's binary_logloss: 0.159809
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.836848	valid_0's binary_logloss: 0.150955	valid_1's auc: 0.813468	valid_1's binary_logloss: 0.155406
    [3]	valid_0's auc: 0.842786	valid_0's binary_logloss: 0.146796	valid_1's auc: 0.817407	valid_1's binary_logloss: 0.151952
    [4]	valid_0's auc: 0.851348	valid_0's binary_logloss: 0.143395	valid_1's auc: 0.823949	valid_1's binary_logloss: 0.14934
    [5]	valid_0's auc: 0.855149	valid_0's binary_logloss: 0.140626	valid_1's auc: 0.827122	valid_1's binary_logloss: 0.147185
    [6]	valid_0's auc: 0.857679	valid_0's binary_logloss: 0.13837	valid_1's auc: 0.826758	valid_1's binary_logloss: 0.145708
    [7]	valid_0's auc: 0.86155	valid_0's binary_logloss: 0.136445	valid_1's auc: 0.828472	valid_1's binary_logloss: 0.144294
    [8]	valid_0's auc: 0.862588	valid_0's binary_logloss: 0.134808	valid_1's auc: 0.828623	valid_1's binary_logloss: 0.143251
    [9]	valid_0's auc: 0.865115	valid_0's binary_logloss: 0.133359	valid_1's auc: 0.831471	valid_1's binary_logloss: 0.142276
    [10]	valid_0's auc: 0.866894	valid_0's binary_logloss: 0.132073	valid_1's auc: 0.831985	valid_1's binary_logloss: 0.141408
    [11]	valid_0's auc: 0.869042	valid_0's binary_logloss: 0.130873	valid_1's auc: 0.834107	valid_1's binary_logloss: 0.140661
    [12]	valid_0's auc: 0.870884	valid_0's binary_logloss: 0.129808	valid_1's auc: 0.833841	valid_1's binary_logloss: 0.140137
    [13]	valid_0's auc: 0.87235	valid_0's binary_logloss: 0.128859	valid_1's auc: 0.834392	valid_1's binary_logloss: 0.139657
    [14]	valid_0's auc: 0.873638	valid_0's binary_logloss: 0.128012	valid_1's auc: 0.834569	valid_1's binary_logloss: 0.13928
    [15]	valid_0's auc: 0.874792	valid_0's binary_logloss: 0.127177	valid_1's auc: 0.83417	valid_1's binary_logloss: 0.138989
    [16]	valid_0's auc: 0.876361	valid_0's binary_logloss: 0.126304	valid_1's auc: 0.835986	valid_1's binary_logloss: 0.138606
    [17]	valid_0's auc: 0.878328	valid_0's binary_logloss: 0.125589	valid_1's auc: 0.836721	valid_1's binary_logloss: 0.138362
    [18]	valid_0's auc: 0.879835	valid_0's binary_logloss: 0.124908	valid_1's auc: 0.836975	valid_1's binary_logloss: 0.138199
    [19]	valid_0's auc: 0.881056	valid_0's binary_logloss: 0.124303	valid_1's auc: 0.837163	valid_1's binary_logloss: 0.137993
    [20]	valid_0's auc: 0.881684	valid_0's binary_logloss: 0.123706	valid_1's auc: 0.836767	valid_1's binary_logloss: 0.137922
    [21]	valid_0's auc: 0.882852	valid_0's binary_logloss: 0.123138	valid_1's auc: 0.835726	valid_1's binary_logloss: 0.137937
    [22]	valid_0's auc: 0.883782	valid_0's binary_logloss: 0.122621	valid_1's auc: 0.836033	valid_1's binary_logloss: 0.137832
    [23]	valid_0's auc: 0.884894	valid_0's binary_logloss: 0.122101	valid_1's auc: 0.835965	valid_1's binary_logloss: 0.137761
    [24]	valid_0's auc: 0.886129	valid_0's binary_logloss: 0.121539	valid_1's auc: 0.836276	valid_1's binary_logloss: 0.137662
    [25]	valid_0's auc: 0.887299	valid_0's binary_logloss: 0.121003	valid_1's auc: 0.836494	valid_1's binary_logloss: 0.137562
    [26]	valid_0's auc: 0.888167	valid_0's binary_logloss: 0.120533	valid_1's auc: 0.836735	valid_1's binary_logloss: 0.137471
    [27]	valid_0's auc: 0.889209	valid_0's binary_logloss: 0.120033	valid_1's auc: 0.837077	valid_1's binary_logloss: 0.137335
    [28]	valid_0's auc: 0.890083	valid_0's binary_logloss: 0.119628	valid_1's auc: 0.836646	valid_1's binary_logloss: 0.137396
    [29]	valid_0's auc: 0.891268	valid_0's binary_logloss: 0.1192	valid_1's auc: 0.836893	valid_1's binary_logloss: 0.137334
    [30]	valid_0's auc: 0.892026	valid_0's binary_logloss: 0.118779	valid_1's auc: 0.836987	valid_1's binary_logloss: 0.13734
    [31]	valid_0's auc: 0.892612	valid_0's binary_logloss: 0.1184	valid_1's auc: 0.835995	valid_1's binary_logloss: 0.137457
    [32]	valid_0's auc: 0.893409	valid_0's binary_logloss: 0.118028	valid_1's auc: 0.836329	valid_1's binary_logloss: 0.137426
    [33]	valid_0's auc: 0.894715	valid_0's binary_logloss: 0.117634	valid_1's auc: 0.836237	valid_1's binary_logloss: 0.137428
    [34]	valid_0's auc: 0.896129	valid_0's binary_logloss: 0.117239	valid_1's auc: 0.837278	valid_1's binary_logloss: 0.137234
    [35]	valid_0's auc: 0.897111	valid_0's binary_logloss: 0.116881	valid_1's auc: 0.837313	valid_1's binary_logloss: 0.137208
    [36]	valid_0's auc: 0.89791	valid_0's binary_logloss: 0.116525	valid_1's auc: 0.837094	valid_1's binary_logloss: 0.137227
    [37]	valid_0's auc: 0.898327	valid_0's binary_logloss: 0.116184	valid_1's auc: 0.836808	valid_1's binary_logloss: 0.137267
    [38]	valid_0's auc: 0.899162	valid_0's binary_logloss: 0.115859	valid_1's auc: 0.836473	valid_1's binary_logloss: 0.137323
    [39]	valid_0's auc: 0.899593	valid_0's binary_logloss: 0.115562	valid_1's auc: 0.836433	valid_1's binary_logloss: 0.137356
    [40]	valid_0's auc: 0.899942	valid_0's binary_logloss: 0.115297	valid_1's auc: 0.836299	valid_1's binary_logloss: 0.137382
    [41]	valid_0's auc: 0.900439	valid_0's binary_logloss: 0.114998	valid_1's auc: 0.835877	valid_1's binary_logloss: 0.137474
    [42]	valid_0's auc: 0.90094	valid_0's binary_logloss: 0.1147	valid_1's auc: 0.835016	valid_1's binary_logloss: 0.137628
    [43]	valid_0's auc: 0.901223	valid_0's binary_logloss: 0.11447	valid_1's auc: 0.834849	valid_1's binary_logloss: 0.137661
    [44]	valid_0's auc: 0.901438	valid_0's binary_logloss: 0.11423	valid_1's auc: 0.834743	valid_1's binary_logloss: 0.137699
    [45]	valid_0's auc: 0.902058	valid_0's binary_logloss: 0.113982	valid_1's auc: 0.834615	valid_1's binary_logloss: 0.13774
    [46]	valid_0's auc: 0.902431	valid_0's binary_logloss: 0.113713	valid_1's auc: 0.834207	valid_1's binary_logloss: 0.13784
    [47]	valid_0's auc: 0.902812	valid_0's binary_logloss: 0.113487	valid_1's auc: 0.833758	valid_1's binary_logloss: 0.137985
    [48]	valid_0's auc: 0.90306	valid_0's binary_logloss: 0.113301	valid_1's auc: 0.833219	valid_1's binary_logloss: 0.138122
    [49]	valid_0's auc: 0.903267	valid_0's binary_logloss: 0.113063	valid_1's auc: 0.832937	valid_1's binary_logloss: 0.138197
    [50]	valid_0's auc: 0.903705	valid_0's binary_logloss: 0.112863	valid_1's auc: 0.832815	valid_1's binary_logloss: 0.13824
    [51]	valid_0's auc: 0.904223	valid_0's binary_logloss: 0.112624	valid_1's auc: 0.832789	valid_1's binary_logloss: 0.138311
    [52]	valid_0's auc: 0.904867	valid_0's binary_logloss: 0.112372	valid_1's auc: 0.832399	valid_1's binary_logloss: 0.138416
    [53]	valid_0's auc: 0.90521	valid_0's binary_logloss: 0.112133	valid_1's auc: 0.832296	valid_1's binary_logloss: 0.1385
    [54]	valid_0's auc: 0.905689	valid_0's binary_logloss: 0.111851	valid_1's auc: 0.831911	valid_1's binary_logloss: 0.138563
    [55]	valid_0's auc: 0.905941	valid_0's binary_logloss: 0.111672	valid_1's auc: 0.831691	valid_1's binary_logloss: 0.138659
    [56]	valid_0's auc: 0.906184	valid_0's binary_logloss: 0.111482	valid_1's auc: 0.83155	valid_1's binary_logloss: 0.138723
    [57]	valid_0's auc: 0.906352	valid_0's binary_logloss: 0.111271	valid_1's auc: 0.830993	valid_1's binary_logloss: 0.138832
    [58]	valid_0's auc: 0.906874	valid_0's binary_logloss: 0.110997	valid_1's auc: 0.831151	valid_1's binary_logloss: 0.138806
    [59]	valid_0's auc: 0.906995	valid_0's binary_logloss: 0.110797	valid_1's auc: 0.830642	valid_1's binary_logloss: 0.138926
    [60]	valid_0's auc: 0.907239	valid_0's binary_logloss: 0.110639	valid_1's auc: 0.830434	valid_1's binary_logloss: 0.139017
    [61]	valid_0's auc: 0.907442	valid_0's binary_logloss: 0.110465	valid_1's auc: 0.829813	valid_1's binary_logloss: 0.13916
    [62]	valid_0's auc: 0.908053	valid_0's binary_logloss: 0.110189	valid_1's auc: 0.829969	valid_1's binary_logloss: 0.139161
    [63]	valid_0's auc: 0.908139	valid_0's binary_logloss: 0.110026	valid_1's auc: 0.829479	valid_1's binary_logloss: 0.139308
    [64]	valid_0's auc: 0.908183	valid_0's binary_logloss: 0.109892	valid_1's auc: 0.828768	valid_1's binary_logloss: 0.139496
    [65]	valid_0's auc: 0.908915	valid_0's binary_logloss: 0.109601	valid_1's auc: 0.828359	valid_1's binary_logloss: 0.139582
    Early stopping, best iteration is:
    [35]	valid_0's auc: 0.897111	valid_0's binary_logloss: 0.116881	valid_1's auc: 0.837313	valid_1's binary_logloss: 0.137208
    [1]	valid_0's auc: 0.833154	valid_0's binary_logloss: 0.156832	valid_1's auc: 0.809041	valid_1's binary_logloss: 0.159834
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.842696	valid_0's binary_logloss: 0.15097	valid_1's auc: 0.816072	valid_1's binary_logloss: 0.155054
    [3]	valid_0's auc: 0.848628	valid_0's binary_logloss: 0.146679	valid_1's auc: 0.821987	valid_1's binary_logloss: 0.151615
    [4]	valid_0's auc: 0.852189	valid_0's binary_logloss: 0.143409	valid_1's auc: 0.823257	valid_1's binary_logloss: 0.148971
    [5]	valid_0's auc: 0.85445	valid_0's binary_logloss: 0.140691	valid_1's auc: 0.826693	valid_1's binary_logloss: 0.146922
    [6]	valid_0's auc: 0.856663	valid_0's binary_logloss: 0.138477	valid_1's auc: 0.827879	valid_1's binary_logloss: 0.145076
    [7]	valid_0's auc: 0.858141	valid_0's binary_logloss: 0.136527	valid_1's auc: 0.828071	valid_1's binary_logloss: 0.143771
    [8]	valid_0's auc: 0.860436	valid_0's binary_logloss: 0.134845	valid_1's auc: 0.830521	valid_1's binary_logloss: 0.142677
    [9]	valid_0's auc: 0.861929	valid_0's binary_logloss: 0.133357	valid_1's auc: 0.831252	valid_1's binary_logloss: 0.14175
    [10]	valid_0's auc: 0.865268	valid_0's binary_logloss: 0.132002	valid_1's auc: 0.831065	valid_1's binary_logloss: 0.141032
    [11]	valid_0's auc: 0.86773	valid_0's binary_logloss: 0.130767	valid_1's auc: 0.831969	valid_1's binary_logloss: 0.140477
    [12]	valid_0's auc: 0.869797	valid_0's binary_logloss: 0.129639	valid_1's auc: 0.831747	valid_1's binary_logloss: 0.139964
    [13]	valid_0's auc: 0.871962	valid_0's binary_logloss: 0.128704	valid_1's auc: 0.832299	valid_1's binary_logloss: 0.13953
    [14]	valid_0's auc: 0.873656	valid_0's binary_logloss: 0.12785	valid_1's auc: 0.831874	valid_1's binary_logloss: 0.139277
    [15]	valid_0's auc: 0.875313	valid_0's binary_logloss: 0.127041	valid_1's auc: 0.831922	valid_1's binary_logloss: 0.139006
    [16]	valid_0's auc: 0.875813	valid_0's binary_logloss: 0.126291	valid_1's auc: 0.832339	valid_1's binary_logloss: 0.138679
    [17]	valid_0's auc: 0.877574	valid_0's binary_logloss: 0.125606	valid_1's auc: 0.832459	valid_1's binary_logloss: 0.138444
    [18]	valid_0's auc: 0.879134	valid_0's binary_logloss: 0.12489	valid_1's auc: 0.832409	valid_1's binary_logloss: 0.138268
    [19]	valid_0's auc: 0.880176	valid_0's binary_logloss: 0.124247	valid_1's auc: 0.832557	valid_1's binary_logloss: 0.138164
    [20]	valid_0's auc: 0.881608	valid_0's binary_logloss: 0.123609	valid_1's auc: 0.832053	valid_1's binary_logloss: 0.138136
    [21]	valid_0's auc: 0.882632	valid_0's binary_logloss: 0.123013	valid_1's auc: 0.831852	valid_1's binary_logloss: 0.138081
    [22]	valid_0's auc: 0.883684	valid_0's binary_logloss: 0.12245	valid_1's auc: 0.832034	valid_1's binary_logloss: 0.137978
    [23]	valid_0's auc: 0.885039	valid_0's binary_logloss: 0.121827	valid_1's auc: 0.831422	valid_1's binary_logloss: 0.138011
    [24]	valid_0's auc: 0.886247	valid_0's binary_logloss: 0.121291	valid_1's auc: 0.831253	valid_1's binary_logloss: 0.137942
    [25]	valid_0's auc: 0.887301	valid_0's binary_logloss: 0.120809	valid_1's auc: 0.831276	valid_1's binary_logloss: 0.137924
    [26]	valid_0's auc: 0.888621	valid_0's binary_logloss: 0.120287	valid_1's auc: 0.830735	valid_1's binary_logloss: 0.137975
    [27]	valid_0's auc: 0.889227	valid_0's binary_logloss: 0.119878	valid_1's auc: 0.830569	valid_1's binary_logloss: 0.137976
    [28]	valid_0's auc: 0.890381	valid_0's binary_logloss: 0.119445	valid_1's auc: 0.830281	valid_1's binary_logloss: 0.137968
    [29]	valid_0's auc: 0.891139	valid_0's binary_logloss: 0.119016	valid_1's auc: 0.830916	valid_1's binary_logloss: 0.137839
    [30]	valid_0's auc: 0.892417	valid_0's binary_logloss: 0.118535	valid_1's auc: 0.830189	valid_1's binary_logloss: 0.137909
    [31]	valid_0's auc: 0.893102	valid_0's binary_logloss: 0.118157	valid_1's auc: 0.830117	valid_1's binary_logloss: 0.137933
    [32]	valid_0's auc: 0.893844	valid_0's binary_logloss: 0.117764	valid_1's auc: 0.829877	valid_1's binary_logloss: 0.137962
    [33]	valid_0's auc: 0.894508	valid_0's binary_logloss: 0.117419	valid_1's auc: 0.829552	valid_1's binary_logloss: 0.138044
    [34]	valid_0's auc: 0.895157	valid_0's binary_logloss: 0.117051	valid_1's auc: 0.830486	valid_1's binary_logloss: 0.137893
    [35]	valid_0's auc: 0.89588	valid_0's binary_logloss: 0.116679	valid_1's auc: 0.83041	valid_1's binary_logloss: 0.137896
    [36]	valid_0's auc: 0.896611	valid_0's binary_logloss: 0.116295	valid_1's auc: 0.829969	valid_1's binary_logloss: 0.138026
    [37]	valid_0's auc: 0.89762	valid_0's binary_logloss: 0.115979	valid_1's auc: 0.829802	valid_1's binary_logloss: 0.138089
    [38]	valid_0's auc: 0.898028	valid_0's binary_logloss: 0.115667	valid_1's auc: 0.829346	valid_1's binary_logloss: 0.138207
    [39]	valid_0's auc: 0.89839	valid_0's binary_logloss: 0.115401	valid_1's auc: 0.829208	valid_1's binary_logloss: 0.138191
    [40]	valid_0's auc: 0.898968	valid_0's binary_logloss: 0.115042	valid_1's auc: 0.828576	valid_1's binary_logloss: 0.138317
    [41]	valid_0's auc: 0.899373	valid_0's binary_logloss: 0.114766	valid_1's auc: 0.828382	valid_1's binary_logloss: 0.138408
    [42]	valid_0's auc: 0.900042	valid_0's binary_logloss: 0.114467	valid_1's auc: 0.828023	valid_1's binary_logloss: 0.138489
    [43]	valid_0's auc: 0.900488	valid_0's binary_logloss: 0.114194	valid_1's auc: 0.827815	valid_1's binary_logloss: 0.138591
    [44]	valid_0's auc: 0.901004	valid_0's binary_logloss: 0.113865	valid_1's auc: 0.827321	valid_1's binary_logloss: 0.138736
    [45]	valid_0's auc: 0.901401	valid_0's binary_logloss: 0.113616	valid_1's auc: 0.826875	valid_1's binary_logloss: 0.138874
    [46]	valid_0's auc: 0.901795	valid_0's binary_logloss: 0.113348	valid_1's auc: 0.82643	valid_1's binary_logloss: 0.138955
    [47]	valid_0's auc: 0.902244	valid_0's binary_logloss: 0.11309	valid_1's auc: 0.825825	valid_1's binary_logloss: 0.13908
    [48]	valid_0's auc: 0.902756	valid_0's binary_logloss: 0.112891	valid_1's auc: 0.825639	valid_1's binary_logloss: 0.139117
    [49]	valid_0's auc: 0.903206	valid_0's binary_logloss: 0.112607	valid_1's auc: 0.825607	valid_1's binary_logloss: 0.139184
    Early stopping, best iteration is:
    [19]	valid_0's auc: 0.880176	valid_0's binary_logloss: 0.124247	valid_1's auc: 0.832557	valid_1's binary_logloss: 0.138164
    [1]	valid_0's auc: 0.825869	valid_0's binary_logloss: 0.156436	valid_1's auc: 0.803397	valid_1's binary_logloss: 0.159993
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.834474	valid_0's binary_logloss: 0.150597	valid_1's auc: 0.809972	valid_1's binary_logloss: 0.155122
    [3]	valid_0's auc: 0.840511	valid_0's binary_logloss: 0.146374	valid_1's auc: 0.819137	valid_1's binary_logloss: 0.15148
    [4]	valid_0's auc: 0.844283	valid_0's binary_logloss: 0.143027	valid_1's auc: 0.820472	valid_1's binary_logloss: 0.14904
    [5]	valid_0's auc: 0.85046	valid_0's binary_logloss: 0.140321	valid_1's auc: 0.82606	valid_1's binary_logloss: 0.146982
    [6]	valid_0's auc: 0.852881	valid_0's binary_logloss: 0.13813	valid_1's auc: 0.823869	valid_1's binary_logloss: 0.145517
    [7]	valid_0's auc: 0.857268	valid_0's binary_logloss: 0.136149	valid_1's auc: 0.827586	valid_1's binary_logloss: 0.144156
    [8]	valid_0's auc: 0.859401	valid_0's binary_logloss: 0.13452	valid_1's auc: 0.829428	valid_1's binary_logloss: 0.14303
    [9]	valid_0's auc: 0.861435	valid_0's binary_logloss: 0.133011	valid_1's auc: 0.82971	valid_1's binary_logloss: 0.142113
    [10]	valid_0's auc: 0.862282	valid_0's binary_logloss: 0.131742	valid_1's auc: 0.830527	valid_1's binary_logloss: 0.141392
    [11]	valid_0's auc: 0.86448	valid_0's binary_logloss: 0.130536	valid_1's auc: 0.83065	valid_1's binary_logloss: 0.140854
    [12]	valid_0's auc: 0.865909	valid_0's binary_logloss: 0.129495	valid_1's auc: 0.830763	valid_1's binary_logloss: 0.140425
    [13]	valid_0's auc: 0.867083	valid_0's binary_logloss: 0.128537	valid_1's auc: 0.830097	valid_1's binary_logloss: 0.140034
    [14]	valid_0's auc: 0.869164	valid_0's binary_logloss: 0.127593	valid_1's auc: 0.831242	valid_1's binary_logloss: 0.139589
    [15]	valid_0's auc: 0.869779	valid_0's binary_logloss: 0.126753	valid_1's auc: 0.830054	valid_1's binary_logloss: 0.13935
    [16]	valid_0's auc: 0.870798	valid_0's binary_logloss: 0.125998	valid_1's auc: 0.829854	valid_1's binary_logloss: 0.139147
    [17]	valid_0's auc: 0.872413	valid_0's binary_logloss: 0.125321	valid_1's auc: 0.829939	valid_1's binary_logloss: 0.138859
    [18]	valid_0's auc: 0.873656	valid_0's binary_logloss: 0.124646	valid_1's auc: 0.83033	valid_1's binary_logloss: 0.138678
    [19]	valid_0's auc: 0.874787	valid_0's binary_logloss: 0.124094	valid_1's auc: 0.829747	valid_1's binary_logloss: 0.138635
    [20]	valid_0's auc: 0.875793	valid_0's binary_logloss: 0.123547	valid_1's auc: 0.829455	valid_1's binary_logloss: 0.138492
    [21]	valid_0's auc: 0.87694	valid_0's binary_logloss: 0.122969	valid_1's auc: 0.830945	valid_1's binary_logloss: 0.13825
    [22]	valid_0's auc: 0.878632	valid_0's binary_logloss: 0.122407	valid_1's auc: 0.831483	valid_1's binary_logloss: 0.138097
    [23]	valid_0's auc: 0.8796	valid_0's binary_logloss: 0.121869	valid_1's auc: 0.831408	valid_1's binary_logloss: 0.138071
    [24]	valid_0's auc: 0.881298	valid_0's binary_logloss: 0.121363	valid_1's auc: 0.831984	valid_1's binary_logloss: 0.137912
    [25]	valid_0's auc: 0.882699	valid_0's binary_logloss: 0.120863	valid_1's auc: 0.831941	valid_1's binary_logloss: 0.137824
    [26]	valid_0's auc: 0.883799	valid_0's binary_logloss: 0.120379	valid_1's auc: 0.832897	valid_1's binary_logloss: 0.137649
    [27]	valid_0's auc: 0.884893	valid_0's binary_logloss: 0.119944	valid_1's auc: 0.832841	valid_1's binary_logloss: 0.13763
    [28]	valid_0's auc: 0.88588	valid_0's binary_logloss: 0.119545	valid_1's auc: 0.833413	valid_1's binary_logloss: 0.137495
    [29]	valid_0's auc: 0.886461	valid_0's binary_logloss: 0.11916	valid_1's auc: 0.833437	valid_1's binary_logloss: 0.137511
    [30]	valid_0's auc: 0.887592	valid_0's binary_logloss: 0.118739	valid_1's auc: 0.833429	valid_1's binary_logloss: 0.137514
    [31]	valid_0's auc: 0.888552	valid_0's binary_logloss: 0.118399	valid_1's auc: 0.834036	valid_1's binary_logloss: 0.137459
    [32]	valid_0's auc: 0.889448	valid_0's binary_logloss: 0.118019	valid_1's auc: 0.833754	valid_1's binary_logloss: 0.137498
    [33]	valid_0's auc: 0.890203	valid_0's binary_logloss: 0.117711	valid_1's auc: 0.833853	valid_1's binary_logloss: 0.137491
    [34]	valid_0's auc: 0.890868	valid_0's binary_logloss: 0.117395	valid_1's auc: 0.833648	valid_1's binary_logloss: 0.137532
    [35]	valid_0's auc: 0.891378	valid_0's binary_logloss: 0.117106	valid_1's auc: 0.833479	valid_1's binary_logloss: 0.137577
    [36]	valid_0's auc: 0.892061	valid_0's binary_logloss: 0.116741	valid_1's auc: 0.832608	valid_1's binary_logloss: 0.137799
    [37]	valid_0's auc: 0.892715	valid_0's binary_logloss: 0.116403	valid_1's auc: 0.83208	valid_1's binary_logloss: 0.137938
    [38]	valid_0's auc: 0.893146	valid_0's binary_logloss: 0.116126	valid_1's auc: 0.83196	valid_1's binary_logloss: 0.137994
    [39]	valid_0's auc: 0.893963	valid_0's binary_logloss: 0.115783	valid_1's auc: 0.831584	valid_1's binary_logloss: 0.138076
    [40]	valid_0's auc: 0.894274	valid_0's binary_logloss: 0.115535	valid_1's auc: 0.831346	valid_1's binary_logloss: 0.13818
    [41]	valid_0's auc: 0.894809	valid_0's binary_logloss: 0.115247	valid_1's auc: 0.831197	valid_1's binary_logloss: 0.138289
    [42]	valid_0's auc: 0.895332	valid_0's binary_logloss: 0.114916	valid_1's auc: 0.830893	valid_1's binary_logloss: 0.138343
    [43]	valid_0's auc: 0.895864	valid_0's binary_logloss: 0.114701	valid_1's auc: 0.83129	valid_1's binary_logloss: 0.138307
    [44]	valid_0's auc: 0.89623	valid_0's binary_logloss: 0.114411	valid_1's auc: 0.831004	valid_1's binary_logloss: 0.138408
    [45]	valid_0's auc: 0.896616	valid_0's binary_logloss: 0.11413	valid_1's auc: 0.830369	valid_1's binary_logloss: 0.138547
    [46]	valid_0's auc: 0.897076	valid_0's binary_logloss: 0.113919	valid_1's auc: 0.830246	valid_1's binary_logloss: 0.138644
    [47]	valid_0's auc: 0.897393	valid_0's binary_logloss: 0.113691	valid_1's auc: 0.829999	valid_1's binary_logloss: 0.138769
    [48]	valid_0's auc: 0.897745	valid_0's binary_logloss: 0.113465	valid_1's auc: 0.829593	valid_1's binary_logloss: 0.138908
    [49]	valid_0's auc: 0.898203	valid_0's binary_logloss: 0.113214	valid_1's auc: 0.829209	valid_1's binary_logloss: 0.139045
    [50]	valid_0's auc: 0.898546	valid_0's binary_logloss: 0.112978	valid_1's auc: 0.829064	valid_1's binary_logloss: 0.139115
    [51]	valid_0's auc: 0.898958	valid_0's binary_logloss: 0.112723	valid_1's auc: 0.82913	valid_1's binary_logloss: 0.139163
    [52]	valid_0's auc: 0.899296	valid_0's binary_logloss: 0.112541	valid_1's auc: 0.828532	valid_1's binary_logloss: 0.139358
    [53]	valid_0's auc: 0.899966	valid_0's binary_logloss: 0.112208	valid_1's auc: 0.828277	valid_1's binary_logloss: 0.139431
    [54]	valid_0's auc: 0.900225	valid_0's binary_logloss: 0.112001	valid_1's auc: 0.827866	valid_1's binary_logloss: 0.139573
    [55]	valid_0's auc: 0.90036	valid_0's binary_logloss: 0.111812	valid_1's auc: 0.827887	valid_1's binary_logloss: 0.139655
    [56]	valid_0's auc: 0.900528	valid_0's binary_logloss: 0.111638	valid_1's auc: 0.827809	valid_1's binary_logloss: 0.139702
    [57]	valid_0's auc: 0.900692	valid_0's binary_logloss: 0.111452	valid_1's auc: 0.827613	valid_1's binary_logloss: 0.139758
    [58]	valid_0's auc: 0.901275	valid_0's binary_logloss: 0.111262	valid_1's auc: 0.827848	valid_1's binary_logloss: 0.139758
    [59]	valid_0's auc: 0.90149	valid_0's binary_logloss: 0.111086	valid_1's auc: 0.827643	valid_1's binary_logloss: 0.139835
    [60]	valid_0's auc: 0.901669	valid_0's binary_logloss: 0.110873	valid_1's auc: 0.827187	valid_1's binary_logloss: 0.13998
    [61]	valid_0's auc: 0.901715	valid_0's binary_logloss: 0.110743	valid_1's auc: 0.826643	valid_1's binary_logloss: 0.140136
    Early stopping, best iteration is:
    [31]	valid_0's auc: 0.888552	valid_0's binary_logloss: 0.118399	valid_1's auc: 0.834036	valid_1's binary_logloss: 0.137459
    [1]	valid_0's auc: 0.828293	valid_0's binary_logloss: 0.156591	valid_1's auc: 0.803729	valid_1's binary_logloss: 0.159809
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.836848	valid_0's binary_logloss: 0.150955	valid_1's auc: 0.813468	valid_1's binary_logloss: 0.155406
    [3]	valid_0's auc: 0.842786	valid_0's binary_logloss: 0.146796	valid_1's auc: 0.817407	valid_1's binary_logloss: 0.151952
    [4]	valid_0's auc: 0.851348	valid_0's binary_logloss: 0.143395	valid_1's auc: 0.823949	valid_1's binary_logloss: 0.14934
    [5]	valid_0's auc: 0.855149	valid_0's binary_logloss: 0.140626	valid_1's auc: 0.827122	valid_1's binary_logloss: 0.147185
    [6]	valid_0's auc: 0.857679	valid_0's binary_logloss: 0.13837	valid_1's auc: 0.826758	valid_1's binary_logloss: 0.145708
    [7]	valid_0's auc: 0.86155	valid_0's binary_logloss: 0.136445	valid_1's auc: 0.828472	valid_1's binary_logloss: 0.144294
    [8]	valid_0's auc: 0.862588	valid_0's binary_logloss: 0.134808	valid_1's auc: 0.828623	valid_1's binary_logloss: 0.143251
    [9]	valid_0's auc: 0.865115	valid_0's binary_logloss: 0.133359	valid_1's auc: 0.831471	valid_1's binary_logloss: 0.142276
    [10]	valid_0's auc: 0.866894	valid_0's binary_logloss: 0.132073	valid_1's auc: 0.831985	valid_1's binary_logloss: 0.141408
    [11]	valid_0's auc: 0.869042	valid_0's binary_logloss: 0.130873	valid_1's auc: 0.834107	valid_1's binary_logloss: 0.140661
    [12]	valid_0's auc: 0.870884	valid_0's binary_logloss: 0.129808	valid_1's auc: 0.833841	valid_1's binary_logloss: 0.140137
    [13]	valid_0's auc: 0.87235	valid_0's binary_logloss: 0.128859	valid_1's auc: 0.834392	valid_1's binary_logloss: 0.139657
    [14]	valid_0's auc: 0.873638	valid_0's binary_logloss: 0.128012	valid_1's auc: 0.834569	valid_1's binary_logloss: 0.13928
    [15]	valid_0's auc: 0.874792	valid_0's binary_logloss: 0.127177	valid_1's auc: 0.83417	valid_1's binary_logloss: 0.138989
    [16]	valid_0's auc: 0.876361	valid_0's binary_logloss: 0.126304	valid_1's auc: 0.835986	valid_1's binary_logloss: 0.138606
    [17]	valid_0's auc: 0.878328	valid_0's binary_logloss: 0.125589	valid_1's auc: 0.836721	valid_1's binary_logloss: 0.138362
    [18]	valid_0's auc: 0.879835	valid_0's binary_logloss: 0.124908	valid_1's auc: 0.836975	valid_1's binary_logloss: 0.138199
    [19]	valid_0's auc: 0.881056	valid_0's binary_logloss: 0.124303	valid_1's auc: 0.837163	valid_1's binary_logloss: 0.137993
    [20]	valid_0's auc: 0.881684	valid_0's binary_logloss: 0.123706	valid_1's auc: 0.836767	valid_1's binary_logloss: 0.137922
    [21]	valid_0's auc: 0.882852	valid_0's binary_logloss: 0.123138	valid_1's auc: 0.835726	valid_1's binary_logloss: 0.137937
    [22]	valid_0's auc: 0.883782	valid_0's binary_logloss: 0.122621	valid_1's auc: 0.836033	valid_1's binary_logloss: 0.137832
    [23]	valid_0's auc: 0.884894	valid_0's binary_logloss: 0.122101	valid_1's auc: 0.835965	valid_1's binary_logloss: 0.137761
    [24]	valid_0's auc: 0.886129	valid_0's binary_logloss: 0.121539	valid_1's auc: 0.836276	valid_1's binary_logloss: 0.137662
    [25]	valid_0's auc: 0.887299	valid_0's binary_logloss: 0.121003	valid_1's auc: 0.836494	valid_1's binary_logloss: 0.137562
    [26]	valid_0's auc: 0.888167	valid_0's binary_logloss: 0.120533	valid_1's auc: 0.836735	valid_1's binary_logloss: 0.137471
    [27]	valid_0's auc: 0.889209	valid_0's binary_logloss: 0.120033	valid_1's auc: 0.837077	valid_1's binary_logloss: 0.137335
    [28]	valid_0's auc: 0.890083	valid_0's binary_logloss: 0.119628	valid_1's auc: 0.836646	valid_1's binary_logloss: 0.137396
    [29]	valid_0's auc: 0.891268	valid_0's binary_logloss: 0.1192	valid_1's auc: 0.836893	valid_1's binary_logloss: 0.137334
    [30]	valid_0's auc: 0.892026	valid_0's binary_logloss: 0.118779	valid_1's auc: 0.836987	valid_1's binary_logloss: 0.13734
    [31]	valid_0's auc: 0.892612	valid_0's binary_logloss: 0.1184	valid_1's auc: 0.835995	valid_1's binary_logloss: 0.137457
    [32]	valid_0's auc: 0.893409	valid_0's binary_logloss: 0.118028	valid_1's auc: 0.836329	valid_1's binary_logloss: 0.137426
    [33]	valid_0's auc: 0.894715	valid_0's binary_logloss: 0.117634	valid_1's auc: 0.836237	valid_1's binary_logloss: 0.137428
    [34]	valid_0's auc: 0.896129	valid_0's binary_logloss: 0.117239	valid_1's auc: 0.837278	valid_1's binary_logloss: 0.137234
    [35]	valid_0's auc: 0.897111	valid_0's binary_logloss: 0.116881	valid_1's auc: 0.837313	valid_1's binary_logloss: 0.137208
    [36]	valid_0's auc: 0.89791	valid_0's binary_logloss: 0.116525	valid_1's auc: 0.837094	valid_1's binary_logloss: 0.137227
    [37]	valid_0's auc: 0.898327	valid_0's binary_logloss: 0.116184	valid_1's auc: 0.836808	valid_1's binary_logloss: 0.137267
    [38]	valid_0's auc: 0.899162	valid_0's binary_logloss: 0.115859	valid_1's auc: 0.836473	valid_1's binary_logloss: 0.137323
    [39]	valid_0's auc: 0.899593	valid_0's binary_logloss: 0.115562	valid_1's auc: 0.836433	valid_1's binary_logloss: 0.137356
    [40]	valid_0's auc: 0.899942	valid_0's binary_logloss: 0.115297	valid_1's auc: 0.836299	valid_1's binary_logloss: 0.137382
    [41]	valid_0's auc: 0.900439	valid_0's binary_logloss: 0.114998	valid_1's auc: 0.835877	valid_1's binary_logloss: 0.137474
    [42]	valid_0's auc: 0.90094	valid_0's binary_logloss: 0.1147	valid_1's auc: 0.835016	valid_1's binary_logloss: 0.137628
    [43]	valid_0's auc: 0.901223	valid_0's binary_logloss: 0.11447	valid_1's auc: 0.834849	valid_1's binary_logloss: 0.137661
    [44]	valid_0's auc: 0.901438	valid_0's binary_logloss: 0.11423	valid_1's auc: 0.834743	valid_1's binary_logloss: 0.137699
    [45]	valid_0's auc: 0.902058	valid_0's binary_logloss: 0.113982	valid_1's auc: 0.834615	valid_1's binary_logloss: 0.13774
    [46]	valid_0's auc: 0.902431	valid_0's binary_logloss: 0.113713	valid_1's auc: 0.834207	valid_1's binary_logloss: 0.13784
    [47]	valid_0's auc: 0.902812	valid_0's binary_logloss: 0.113487	valid_1's auc: 0.833758	valid_1's binary_logloss: 0.137985
    [48]	valid_0's auc: 0.90306	valid_0's binary_logloss: 0.113301	valid_1's auc: 0.833219	valid_1's binary_logloss: 0.138122
    [49]	valid_0's auc: 0.903267	valid_0's binary_logloss: 0.113063	valid_1's auc: 0.832937	valid_1's binary_logloss: 0.138197
    [50]	valid_0's auc: 0.903705	valid_0's binary_logloss: 0.112863	valid_1's auc: 0.832815	valid_1's binary_logloss: 0.13824
    [51]	valid_0's auc: 0.904223	valid_0's binary_logloss: 0.112624	valid_1's auc: 0.832789	valid_1's binary_logloss: 0.138311
    [52]	valid_0's auc: 0.904867	valid_0's binary_logloss: 0.112372	valid_1's auc: 0.832399	valid_1's binary_logloss: 0.138416
    [53]	valid_0's auc: 0.90521	valid_0's binary_logloss: 0.112133	valid_1's auc: 0.832296	valid_1's binary_logloss: 0.1385
    [54]	valid_0's auc: 0.905689	valid_0's binary_logloss: 0.111851	valid_1's auc: 0.831911	valid_1's binary_logloss: 0.138563
    [55]	valid_0's auc: 0.905941	valid_0's binary_logloss: 0.111672	valid_1's auc: 0.831691	valid_1's binary_logloss: 0.138659
    [56]	valid_0's auc: 0.906184	valid_0's binary_logloss: 0.111482	valid_1's auc: 0.83155	valid_1's binary_logloss: 0.138723
    [57]	valid_0's auc: 0.906352	valid_0's binary_logloss: 0.111271	valid_1's auc: 0.830993	valid_1's binary_logloss: 0.138832
    [58]	valid_0's auc: 0.906874	valid_0's binary_logloss: 0.110997	valid_1's auc: 0.831151	valid_1's binary_logloss: 0.138806
    [59]	valid_0's auc: 0.906995	valid_0's binary_logloss: 0.110797	valid_1's auc: 0.830642	valid_1's binary_logloss: 0.138926
    [60]	valid_0's auc: 0.907239	valid_0's binary_logloss: 0.110639	valid_1's auc: 0.830434	valid_1's binary_logloss: 0.139017
    [61]	valid_0's auc: 0.907442	valid_0's binary_logloss: 0.110465	valid_1's auc: 0.829813	valid_1's binary_logloss: 0.13916
    [62]	valid_0's auc: 0.908053	valid_0's binary_logloss: 0.110189	valid_1's auc: 0.829969	valid_1's binary_logloss: 0.139161
    [63]	valid_0's auc: 0.908139	valid_0's binary_logloss: 0.110026	valid_1's auc: 0.829479	valid_1's binary_logloss: 0.139308
    [64]	valid_0's auc: 0.908183	valid_0's binary_logloss: 0.109892	valid_1's auc: 0.828768	valid_1's binary_logloss: 0.139496
    [65]	valid_0's auc: 0.908915	valid_0's binary_logloss: 0.109601	valid_1's auc: 0.828359	valid_1's binary_logloss: 0.139582
    Early stopping, best iteration is:
    [35]	valid_0's auc: 0.897111	valid_0's binary_logloss: 0.116881	valid_1's auc: 0.837313	valid_1's binary_logloss: 0.137208
    [1]	valid_0's auc: 0.833154	valid_0's binary_logloss: 0.156832	valid_1's auc: 0.809041	valid_1's binary_logloss: 0.159834
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.842696	valid_0's binary_logloss: 0.15097	valid_1's auc: 0.816072	valid_1's binary_logloss: 0.155054
    [3]	valid_0's auc: 0.848628	valid_0's binary_logloss: 0.146679	valid_1's auc: 0.821987	valid_1's binary_logloss: 0.151615
    [4]	valid_0's auc: 0.852189	valid_0's binary_logloss: 0.143409	valid_1's auc: 0.823257	valid_1's binary_logloss: 0.148971
    [5]	valid_0's auc: 0.85445	valid_0's binary_logloss: 0.140691	valid_1's auc: 0.826693	valid_1's binary_logloss: 0.146922
    [6]	valid_0's auc: 0.856663	valid_0's binary_logloss: 0.138477	valid_1's auc: 0.827879	valid_1's binary_logloss: 0.145076
    [7]	valid_0's auc: 0.858141	valid_0's binary_logloss: 0.136527	valid_1's auc: 0.828071	valid_1's binary_logloss: 0.143771
    [8]	valid_0's auc: 0.860436	valid_0's binary_logloss: 0.134845	valid_1's auc: 0.830521	valid_1's binary_logloss: 0.142677
    [9]	valid_0's auc: 0.861929	valid_0's binary_logloss: 0.133357	valid_1's auc: 0.831252	valid_1's binary_logloss: 0.14175
    [10]	valid_0's auc: 0.865268	valid_0's binary_logloss: 0.132002	valid_1's auc: 0.831065	valid_1's binary_logloss: 0.141032
    [11]	valid_0's auc: 0.86773	valid_0's binary_logloss: 0.130767	valid_1's auc: 0.831969	valid_1's binary_logloss: 0.140477
    [12]	valid_0's auc: 0.869797	valid_0's binary_logloss: 0.129639	valid_1's auc: 0.831747	valid_1's binary_logloss: 0.139964
    [13]	valid_0's auc: 0.871962	valid_0's binary_logloss: 0.128704	valid_1's auc: 0.832299	valid_1's binary_logloss: 0.13953
    [14]	valid_0's auc: 0.873656	valid_0's binary_logloss: 0.12785	valid_1's auc: 0.831874	valid_1's binary_logloss: 0.139277
    [15]	valid_0's auc: 0.875313	valid_0's binary_logloss: 0.127041	valid_1's auc: 0.831922	valid_1's binary_logloss: 0.139006
    [16]	valid_0's auc: 0.875813	valid_0's binary_logloss: 0.126291	valid_1's auc: 0.832339	valid_1's binary_logloss: 0.138679
    [17]	valid_0's auc: 0.877574	valid_0's binary_logloss: 0.125606	valid_1's auc: 0.832459	valid_1's binary_logloss: 0.138444
    [18]	valid_0's auc: 0.879134	valid_0's binary_logloss: 0.12489	valid_1's auc: 0.832409	valid_1's binary_logloss: 0.138268
    [19]	valid_0's auc: 0.880176	valid_0's binary_logloss: 0.124247	valid_1's auc: 0.832557	valid_1's binary_logloss: 0.138164
    [20]	valid_0's auc: 0.881608	valid_0's binary_logloss: 0.123609	valid_1's auc: 0.832053	valid_1's binary_logloss: 0.138136
    [21]	valid_0's auc: 0.882632	valid_0's binary_logloss: 0.123013	valid_1's auc: 0.831852	valid_1's binary_logloss: 0.138081
    [22]	valid_0's auc: 0.883684	valid_0's binary_logloss: 0.12245	valid_1's auc: 0.832034	valid_1's binary_logloss: 0.137978
    [23]	valid_0's auc: 0.885039	valid_0's binary_logloss: 0.121827	valid_1's auc: 0.831422	valid_1's binary_logloss: 0.138011
    [24]	valid_0's auc: 0.886247	valid_0's binary_logloss: 0.121291	valid_1's auc: 0.831253	valid_1's binary_logloss: 0.137942
    [25]	valid_0's auc: 0.887301	valid_0's binary_logloss: 0.120809	valid_1's auc: 0.831276	valid_1's binary_logloss: 0.137924
    [26]	valid_0's auc: 0.888621	valid_0's binary_logloss: 0.120287	valid_1's auc: 0.830735	valid_1's binary_logloss: 0.137975
    [27]	valid_0's auc: 0.889227	valid_0's binary_logloss: 0.119878	valid_1's auc: 0.830569	valid_1's binary_logloss: 0.137976
    [28]	valid_0's auc: 0.890381	valid_0's binary_logloss: 0.119445	valid_1's auc: 0.830281	valid_1's binary_logloss: 0.137968
    [29]	valid_0's auc: 0.891139	valid_0's binary_logloss: 0.119016	valid_1's auc: 0.830916	valid_1's binary_logloss: 0.137839
    [30]	valid_0's auc: 0.892417	valid_0's binary_logloss: 0.118535	valid_1's auc: 0.830189	valid_1's binary_logloss: 0.137909
    [31]	valid_0's auc: 0.893102	valid_0's binary_logloss: 0.118157	valid_1's auc: 0.830117	valid_1's binary_logloss: 0.137933
    [32]	valid_0's auc: 0.893844	valid_0's binary_logloss: 0.117764	valid_1's auc: 0.829877	valid_1's binary_logloss: 0.137962
    [33]	valid_0's auc: 0.894508	valid_0's binary_logloss: 0.117419	valid_1's auc: 0.829552	valid_1's binary_logloss: 0.138044
    [34]	valid_0's auc: 0.895157	valid_0's binary_logloss: 0.117051	valid_1's auc: 0.830486	valid_1's binary_logloss: 0.137893
    [35]	valid_0's auc: 0.89588	valid_0's binary_logloss: 0.116679	valid_1's auc: 0.83041	valid_1's binary_logloss: 0.137896
    [36]	valid_0's auc: 0.896611	valid_0's binary_logloss: 0.116295	valid_1's auc: 0.829969	valid_1's binary_logloss: 0.138026
    [37]	valid_0's auc: 0.89762	valid_0's binary_logloss: 0.115979	valid_1's auc: 0.829802	valid_1's binary_logloss: 0.138089
    [38]	valid_0's auc: 0.898028	valid_0's binary_logloss: 0.115667	valid_1's auc: 0.829346	valid_1's binary_logloss: 0.138207
    [39]	valid_0's auc: 0.89839	valid_0's binary_logloss: 0.115401	valid_1's auc: 0.829208	valid_1's binary_logloss: 0.138191
    [40]	valid_0's auc: 0.898968	valid_0's binary_logloss: 0.115042	valid_1's auc: 0.828576	valid_1's binary_logloss: 0.138317
    [41]	valid_0's auc: 0.899373	valid_0's binary_logloss: 0.114766	valid_1's auc: 0.828382	valid_1's binary_logloss: 0.138408
    [42]	valid_0's auc: 0.900042	valid_0's binary_logloss: 0.114467	valid_1's auc: 0.828023	valid_1's binary_logloss: 0.138489
    [43]	valid_0's auc: 0.900488	valid_0's binary_logloss: 0.114194	valid_1's auc: 0.827815	valid_1's binary_logloss: 0.138591
    [44]	valid_0's auc: 0.901004	valid_0's binary_logloss: 0.113865	valid_1's auc: 0.827321	valid_1's binary_logloss: 0.138736
    [45]	valid_0's auc: 0.901401	valid_0's binary_logloss: 0.113616	valid_1's auc: 0.826875	valid_1's binary_logloss: 0.138874
    [46]	valid_0's auc: 0.901795	valid_0's binary_logloss: 0.113348	valid_1's auc: 0.82643	valid_1's binary_logloss: 0.138955
    [47]	valid_0's auc: 0.902244	valid_0's binary_logloss: 0.11309	valid_1's auc: 0.825825	valid_1's binary_logloss: 0.13908
    [48]	valid_0's auc: 0.902756	valid_0's binary_logloss: 0.112891	valid_1's auc: 0.825639	valid_1's binary_logloss: 0.139117
    [49]	valid_0's auc: 0.903206	valid_0's binary_logloss: 0.112607	valid_1's auc: 0.825607	valid_1's binary_logloss: 0.139184
    Early stopping, best iteration is:
    [19]	valid_0's auc: 0.880176	valid_0's binary_logloss: 0.124247	valid_1's auc: 0.832557	valid_1's binary_logloss: 0.138164
    [1]	valid_0's auc: 0.822874	valid_0's binary_logloss: 0.157449	valid_1's auc: 0.804936	valid_1's binary_logloss: 0.160126
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.827064	valid_0's binary_logloss: 0.152115	valid_1's auc: 0.810262	valid_1's binary_logloss: 0.155403
    [3]	valid_0's auc: 0.830773	valid_0's binary_logloss: 0.148229	valid_1's auc: 0.814099	valid_1's binary_logloss: 0.151956
    [4]	valid_0's auc: 0.834585	valid_0's binary_logloss: 0.145242	valid_1's auc: 0.815291	valid_1's binary_logloss: 0.149379
    [5]	valid_0's auc: 0.839767	valid_0's binary_logloss: 0.142798	valid_1's auc: 0.819992	valid_1's binary_logloss: 0.147278
    [6]	valid_0's auc: 0.845277	valid_0's binary_logloss: 0.140696	valid_1's auc: 0.824997	valid_1's binary_logloss: 0.145614
    [7]	valid_0's auc: 0.84706	valid_0's binary_logloss: 0.139021	valid_1's auc: 0.827429	valid_1's binary_logloss: 0.144095
    [8]	valid_0's auc: 0.847609	valid_0's binary_logloss: 0.137638	valid_1's auc: 0.827987	valid_1's binary_logloss: 0.143002
    [9]	valid_0's auc: 0.848913	valid_0's binary_logloss: 0.136364	valid_1's auc: 0.82986	valid_1's binary_logloss: 0.141984
    [10]	valid_0's auc: 0.850178	valid_0's binary_logloss: 0.135236	valid_1's auc: 0.829651	valid_1's binary_logloss: 0.141228
    [11]	valid_0's auc: 0.85127	valid_0's binary_logloss: 0.134273	valid_1's auc: 0.830496	valid_1's binary_logloss: 0.140509
    [12]	valid_0's auc: 0.852586	valid_0's binary_logloss: 0.133411	valid_1's auc: 0.830525	valid_1's binary_logloss: 0.13997
    [13]	valid_0's auc: 0.853732	valid_0's binary_logloss: 0.132666	valid_1's auc: 0.829319	valid_1's binary_logloss: 0.13957
    [14]	valid_0's auc: 0.854775	valid_0's binary_logloss: 0.13201	valid_1's auc: 0.832116	valid_1's binary_logloss: 0.139106
    [15]	valid_0's auc: 0.855714	valid_0's binary_logloss: 0.131409	valid_1's auc: 0.833454	valid_1's binary_logloss: 0.138684
    [16]	valid_0's auc: 0.856099	valid_0's binary_logloss: 0.130853	valid_1's auc: 0.832243	valid_1's binary_logloss: 0.138485
    [17]	valid_0's auc: 0.857475	valid_0's binary_logloss: 0.130295	valid_1's auc: 0.832577	valid_1's binary_logloss: 0.138179
    [18]	valid_0's auc: 0.857908	valid_0's binary_logloss: 0.129866	valid_1's auc: 0.832652	valid_1's binary_logloss: 0.138084
    [19]	valid_0's auc: 0.859226	valid_0's binary_logloss: 0.129384	valid_1's auc: 0.832887	valid_1's binary_logloss: 0.137828
    [20]	valid_0's auc: 0.860296	valid_0's binary_logloss: 0.129003	valid_1's auc: 0.834111	valid_1's binary_logloss: 0.137643
    [21]	valid_0's auc: 0.860934	valid_0's binary_logloss: 0.128619	valid_1's auc: 0.834602	valid_1's binary_logloss: 0.137471
    [22]	valid_0's auc: 0.861712	valid_0's binary_logloss: 0.12825	valid_1's auc: 0.834828	valid_1's binary_logloss: 0.137291
    [23]	valid_0's auc: 0.862939	valid_0's binary_logloss: 0.127869	valid_1's auc: 0.835767	valid_1's binary_logloss: 0.137067
    [24]	valid_0's auc: 0.863968	valid_0's binary_logloss: 0.127525	valid_1's auc: 0.835485	valid_1's binary_logloss: 0.13701
    [25]	valid_0's auc: 0.865029	valid_0's binary_logloss: 0.127234	valid_1's auc: 0.835507	valid_1's binary_logloss: 0.136911
    [26]	valid_0's auc: 0.865758	valid_0's binary_logloss: 0.126952	valid_1's auc: 0.835743	valid_1's binary_logloss: 0.136823
    [27]	valid_0's auc: 0.866244	valid_0's binary_logloss: 0.126688	valid_1's auc: 0.836049	valid_1's binary_logloss: 0.136736
    [28]	valid_0's auc: 0.866607	valid_0's binary_logloss: 0.126455	valid_1's auc: 0.836043	valid_1's binary_logloss: 0.136687
    [29]	valid_0's auc: 0.867322	valid_0's binary_logloss: 0.126255	valid_1's auc: 0.836375	valid_1's binary_logloss: 0.136596
    [30]	valid_0's auc: 0.867997	valid_0's binary_logloss: 0.126022	valid_1's auc: 0.836518	valid_1's binary_logloss: 0.136563
    [31]	valid_0's auc: 0.868888	valid_0's binary_logloss: 0.125781	valid_1's auc: 0.836949	valid_1's binary_logloss: 0.13648
    [32]	valid_0's auc: 0.869544	valid_0's binary_logloss: 0.125524	valid_1's auc: 0.836774	valid_1's binary_logloss: 0.136506
    [33]	valid_0's auc: 0.870314	valid_0's binary_logloss: 0.125324	valid_1's auc: 0.837039	valid_1's binary_logloss: 0.13642
    [34]	valid_0's auc: 0.871207	valid_0's binary_logloss: 0.125096	valid_1's auc: 0.836859	valid_1's binary_logloss: 0.136441
    [35]	valid_0's auc: 0.871788	valid_0's binary_logloss: 0.124883	valid_1's auc: 0.836641	valid_1's binary_logloss: 0.136467
    [36]	valid_0's auc: 0.872403	valid_0's binary_logloss: 0.124635	valid_1's auc: 0.836917	valid_1's binary_logloss: 0.13642
    [37]	valid_0's auc: 0.872914	valid_0's binary_logloss: 0.124439	valid_1's auc: 0.836892	valid_1's binary_logloss: 0.136489
    [38]	valid_0's auc: 0.873469	valid_0's binary_logloss: 0.124233	valid_1's auc: 0.836816	valid_1's binary_logloss: 0.136502
    [39]	valid_0's auc: 0.873971	valid_0's binary_logloss: 0.124017	valid_1's auc: 0.83659	valid_1's binary_logloss: 0.136543
    [40]	valid_0's auc: 0.875013	valid_0's binary_logloss: 0.123715	valid_1's auc: 0.836381	valid_1's binary_logloss: 0.136593
    [41]	valid_0's auc: 0.875341	valid_0's binary_logloss: 0.123538	valid_1's auc: 0.8363	valid_1's binary_logloss: 0.136624
    [42]	valid_0's auc: 0.875886	valid_0's binary_logloss: 0.123285	valid_1's auc: 0.836043	valid_1's binary_logloss: 0.136665
    [43]	valid_0's auc: 0.87633	valid_0's binary_logloss: 0.12311	valid_1's auc: 0.836018	valid_1's binary_logloss: 0.136647
    [44]	valid_0's auc: 0.876816	valid_0's binary_logloss: 0.122878	valid_1's auc: 0.836061	valid_1's binary_logloss: 0.136664
    [45]	valid_0's auc: 0.877116	valid_0's binary_logloss: 0.12271	valid_1's auc: 0.836517	valid_1's binary_logloss: 0.136608
    [46]	valid_0's auc: 0.877567	valid_0's binary_logloss: 0.122536	valid_1's auc: 0.836051	valid_1's binary_logloss: 0.13667
    [47]	valid_0's auc: 0.878192	valid_0's binary_logloss: 0.122332	valid_1's auc: 0.835827	valid_1's binary_logloss: 0.136694
    [48]	valid_0's auc: 0.878371	valid_0's binary_logloss: 0.122208	valid_1's auc: 0.835831	valid_1's binary_logloss: 0.136717
    [49]	valid_0's auc: 0.878669	valid_0's binary_logloss: 0.12204	valid_1's auc: 0.835913	valid_1's binary_logloss: 0.136679
    [50]	valid_0's auc: 0.879288	valid_0's binary_logloss: 0.121825	valid_1's auc: 0.835468	valid_1's binary_logloss: 0.136734
    [51]	valid_0's auc: 0.879552	valid_0's binary_logloss: 0.121647	valid_1's auc: 0.835095	valid_1's binary_logloss: 0.13684
    [52]	valid_0's auc: 0.880033	valid_0's binary_logloss: 0.121459	valid_1's auc: 0.83496	valid_1's binary_logloss: 0.136927
    [53]	valid_0's auc: 0.880447	valid_0's binary_logloss: 0.121297	valid_1's auc: 0.834955	valid_1's binary_logloss: 0.136953
    [54]	valid_0's auc: 0.88087	valid_0's binary_logloss: 0.121116	valid_1's auc: 0.834926	valid_1's binary_logloss: 0.136994
    [55]	valid_0's auc: 0.881105	valid_0's binary_logloss: 0.12097	valid_1's auc: 0.834861	valid_1's binary_logloss: 0.137043
    [56]	valid_0's auc: 0.88161	valid_0's binary_logloss: 0.120772	valid_1's auc: 0.834629	valid_1's binary_logloss: 0.137125
    [57]	valid_0's auc: 0.882134	valid_0's binary_logloss: 0.120595	valid_1's auc: 0.834576	valid_1's binary_logloss: 0.137157
    [58]	valid_0's auc: 0.882429	valid_0's binary_logloss: 0.120441	valid_1's auc: 0.834319	valid_1's binary_logloss: 0.137209
    [59]	valid_0's auc: 0.882799	valid_0's binary_logloss: 0.120268	valid_1's auc: 0.834028	valid_1's binary_logloss: 0.1373
    [60]	valid_0's auc: 0.883147	valid_0's binary_logloss: 0.120094	valid_1's auc: 0.833999	valid_1's binary_logloss: 0.137339
    [61]	valid_0's auc: 0.883628	valid_0's binary_logloss: 0.119914	valid_1's auc: 0.834234	valid_1's binary_logloss: 0.137314
    [62]	valid_0's auc: 0.884034	valid_0's binary_logloss: 0.119753	valid_1's auc: 0.834179	valid_1's binary_logloss: 0.137356
    [63]	valid_0's auc: 0.884169	valid_0's binary_logloss: 0.119623	valid_1's auc: 0.834035	valid_1's binary_logloss: 0.137378
    Early stopping, best iteration is:
    [33]	valid_0's auc: 0.870314	valid_0's binary_logloss: 0.125324	valid_1's auc: 0.837039	valid_1's binary_logloss: 0.13642
    [1]	valid_0's auc: 0.826066	valid_0's binary_logloss: 0.157509	valid_1's auc: 0.810763	valid_1's binary_logloss: 0.160177
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.830558	valid_0's binary_logloss: 0.152317	valid_1's auc: 0.814406	valid_1's binary_logloss: 0.15553
    [3]	valid_0's auc: 0.833526	valid_0's binary_logloss: 0.148529	valid_1's auc: 0.815841	valid_1's binary_logloss: 0.152134
    [4]	valid_0's auc: 0.839653	valid_0's binary_logloss: 0.145555	valid_1's auc: 0.821767	valid_1's binary_logloss: 0.149519
    [5]	valid_0's auc: 0.842943	valid_0's binary_logloss: 0.143073	valid_1's auc: 0.826334	valid_1's binary_logloss: 0.14736
    [6]	valid_0's auc: 0.845929	valid_0's binary_logloss: 0.141096	valid_1's auc: 0.828882	valid_1's binary_logloss: 0.145649
    [7]	valid_0's auc: 0.847231	valid_0's binary_logloss: 0.139394	valid_1's auc: 0.830621	valid_1's binary_logloss: 0.144201
    [8]	valid_0's auc: 0.849609	valid_0's binary_logloss: 0.137956	valid_1's auc: 0.831324	valid_1's binary_logloss: 0.143105
    [9]	valid_0's auc: 0.851333	valid_0's binary_logloss: 0.136733	valid_1's auc: 0.83377	valid_1's binary_logloss: 0.142018
    [10]	valid_0's auc: 0.852692	valid_0's binary_logloss: 0.135654	valid_1's auc: 0.834493	valid_1's binary_logloss: 0.141149
    [11]	valid_0's auc: 0.854022	valid_0's binary_logloss: 0.134688	valid_1's auc: 0.834616	valid_1's binary_logloss: 0.140437
    [12]	valid_0's auc: 0.854419	valid_0's binary_logloss: 0.13393	valid_1's auc: 0.83589	valid_1's binary_logloss: 0.139765
    [13]	valid_0's auc: 0.854853	valid_0's binary_logloss: 0.133229	valid_1's auc: 0.835999	valid_1's binary_logloss: 0.139284
    [14]	valid_0's auc: 0.856241	valid_0's binary_logloss: 0.132532	valid_1's auc: 0.837392	valid_1's binary_logloss: 0.138795
    [15]	valid_0's auc: 0.85803	valid_0's binary_logloss: 0.131862	valid_1's auc: 0.83818	valid_1's binary_logloss: 0.138366
    [16]	valid_0's auc: 0.859289	valid_0's binary_logloss: 0.131311	valid_1's auc: 0.838523	valid_1's binary_logloss: 0.138137
    [17]	valid_0's auc: 0.860115	valid_0's binary_logloss: 0.130812	valid_1's auc: 0.839359	valid_1's binary_logloss: 0.137816
    [18]	valid_0's auc: 0.860807	valid_0's binary_logloss: 0.130408	valid_1's auc: 0.840179	valid_1's binary_logloss: 0.13749
    [19]	valid_0's auc: 0.861568	valid_0's binary_logloss: 0.129954	valid_1's auc: 0.839855	valid_1's binary_logloss: 0.137387
    [20]	valid_0's auc: 0.863194	valid_0's binary_logloss: 0.129482	valid_1's auc: 0.839715	valid_1's binary_logloss: 0.137226
    [21]	valid_0's auc: 0.864273	valid_0's binary_logloss: 0.12907	valid_1's auc: 0.840516	valid_1's binary_logloss: 0.136991
    [22]	valid_0's auc: 0.865334	valid_0's binary_logloss: 0.128648	valid_1's auc: 0.841302	valid_1's binary_logloss: 0.136769
    [23]	valid_0's auc: 0.866353	valid_0's binary_logloss: 0.128302	valid_1's auc: 0.841085	valid_1's binary_logloss: 0.136653
    [24]	valid_0's auc: 0.86699	valid_0's binary_logloss: 0.127957	valid_1's auc: 0.841235	valid_1's binary_logloss: 0.136611
    [25]	valid_0's auc: 0.867587	valid_0's binary_logloss: 0.127634	valid_1's auc: 0.840948	valid_1's binary_logloss: 0.136644
    [26]	valid_0's auc: 0.868507	valid_0's binary_logloss: 0.127298	valid_1's auc: 0.841033	valid_1's binary_logloss: 0.136526
    [27]	valid_0's auc: 0.869362	valid_0's binary_logloss: 0.126976	valid_1's auc: 0.840939	valid_1's binary_logloss: 0.136445
    [28]	valid_0's auc: 0.870126	valid_0's binary_logloss: 0.126688	valid_1's auc: 0.840452	valid_1's binary_logloss: 0.13645
    [29]	valid_0's auc: 0.870568	valid_0's binary_logloss: 0.126442	valid_1's auc: 0.840809	valid_1's binary_logloss: 0.136345
    [30]	valid_0's auc: 0.871323	valid_0's binary_logloss: 0.126166	valid_1's auc: 0.840599	valid_1's binary_logloss: 0.136334
    [31]	valid_0's auc: 0.872181	valid_0's binary_logloss: 0.125929	valid_1's auc: 0.840663	valid_1's binary_logloss: 0.136267
    [32]	valid_0's auc: 0.873089	valid_0's binary_logloss: 0.125641	valid_1's auc: 0.841679	valid_1's binary_logloss: 0.136141
    [33]	valid_0's auc: 0.873443	valid_0's binary_logloss: 0.125415	valid_1's auc: 0.841597	valid_1's binary_logloss: 0.136138
    [34]	valid_0's auc: 0.874111	valid_0's binary_logloss: 0.125163	valid_1's auc: 0.841194	valid_1's binary_logloss: 0.13614
    [35]	valid_0's auc: 0.874804	valid_0's binary_logloss: 0.124891	valid_1's auc: 0.841136	valid_1's binary_logloss: 0.136214
    [36]	valid_0's auc: 0.875123	valid_0's binary_logloss: 0.124721	valid_1's auc: 0.840883	valid_1's binary_logloss: 0.136268
    [37]	valid_0's auc: 0.875964	valid_0's binary_logloss: 0.124479	valid_1's auc: 0.840327	valid_1's binary_logloss: 0.136363
    [38]	valid_0's auc: 0.876484	valid_0's binary_logloss: 0.12422	valid_1's auc: 0.840317	valid_1's binary_logloss: 0.136378
    [39]	valid_0's auc: 0.877529	valid_0's binary_logloss: 0.123975	valid_1's auc: 0.840224	valid_1's binary_logloss: 0.136372
    [40]	valid_0's auc: 0.877916	valid_0's binary_logloss: 0.123803	valid_1's auc: 0.840099	valid_1's binary_logloss: 0.136418
    [41]	valid_0's auc: 0.878404	valid_0's binary_logloss: 0.123606	valid_1's auc: 0.839447	valid_1's binary_logloss: 0.136498
    [42]	valid_0's auc: 0.878787	valid_0's binary_logloss: 0.123458	valid_1's auc: 0.839737	valid_1's binary_logloss: 0.136466
    [43]	valid_0's auc: 0.879296	valid_0's binary_logloss: 0.123263	valid_1's auc: 0.838848	valid_1's binary_logloss: 0.136611
    [44]	valid_0's auc: 0.879561	valid_0's binary_logloss: 0.123111	valid_1's auc: 0.83939	valid_1's binary_logloss: 0.136509
    [45]	valid_0's auc: 0.880167	valid_0's binary_logloss: 0.122912	valid_1's auc: 0.83916	valid_1's binary_logloss: 0.136564
    [46]	valid_0's auc: 0.880652	valid_0's binary_logloss: 0.12273	valid_1's auc: 0.839133	valid_1's binary_logloss: 0.136588
    [47]	valid_0's auc: 0.881351	valid_0's binary_logloss: 0.122478	valid_1's auc: 0.839108	valid_1's binary_logloss: 0.136599
    [48]	valid_0's auc: 0.881786	valid_0's binary_logloss: 0.122308	valid_1's auc: 0.838894	valid_1's binary_logloss: 0.136636
    [49]	valid_0's auc: 0.88242	valid_0's binary_logloss: 0.122054	valid_1's auc: 0.838476	valid_1's binary_logloss: 0.136727
    [50]	valid_0's auc: 0.88279	valid_0's binary_logloss: 0.121862	valid_1's auc: 0.837735	valid_1's binary_logloss: 0.136833
    [51]	valid_0's auc: 0.883422	valid_0's binary_logloss: 0.121646	valid_1's auc: 0.837691	valid_1's binary_logloss: 0.136833
    [52]	valid_0's auc: 0.884035	valid_0's binary_logloss: 0.121461	valid_1's auc: 0.837503	valid_1's binary_logloss: 0.136856
    [53]	valid_0's auc: 0.884636	valid_0's binary_logloss: 0.121237	valid_1's auc: 0.837159	valid_1's binary_logloss: 0.136933
    [54]	valid_0's auc: 0.885019	valid_0's binary_logloss: 0.121086	valid_1's auc: 0.836758	valid_1's binary_logloss: 0.136995
    [55]	valid_0's auc: 0.88546	valid_0's binary_logloss: 0.120908	valid_1's auc: 0.836692	valid_1's binary_logloss: 0.137006
    [56]	valid_0's auc: 0.885711	valid_0's binary_logloss: 0.120745	valid_1's auc: 0.836703	valid_1's binary_logloss: 0.136998
    [57]	valid_0's auc: 0.886081	valid_0's binary_logloss: 0.120544	valid_1's auc: 0.836462	valid_1's binary_logloss: 0.137073
    [58]	valid_0's auc: 0.886576	valid_0's binary_logloss: 0.120352	valid_1's auc: 0.836241	valid_1's binary_logloss: 0.137127
    [59]	valid_0's auc: 0.886878	valid_0's binary_logloss: 0.120191	valid_1's auc: 0.8358	valid_1's binary_logloss: 0.137207
    [60]	valid_0's auc: 0.887158	valid_0's binary_logloss: 0.120005	valid_1's auc: 0.835688	valid_1's binary_logloss: 0.137272
    [61]	valid_0's auc: 0.88755	valid_0's binary_logloss: 0.119855	valid_1's auc: 0.835652	valid_1's binary_logloss: 0.137262
    [62]	valid_0's auc: 0.887817	valid_0's binary_logloss: 0.11971	valid_1's auc: 0.835355	valid_1's binary_logloss: 0.13732
    Early stopping, best iteration is:
    [32]	valid_0's auc: 0.873089	valid_0's binary_logloss: 0.125641	valid_1's auc: 0.841679	valid_1's binary_logloss: 0.136141
    [1]	valid_0's auc: 0.822527	valid_0's binary_logloss: 0.157702	valid_1's auc: 0.811735	valid_1's binary_logloss: 0.160053
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.828897	valid_0's binary_logloss: 0.152381	valid_1's auc: 0.815279	valid_1's binary_logloss: 0.155308
    [3]	valid_0's auc: 0.833978	valid_0's binary_logloss: 0.148529	valid_1's auc: 0.819863	valid_1's binary_logloss: 0.151959
    [4]	valid_0's auc: 0.837641	valid_0's binary_logloss: 0.145558	valid_1's auc: 0.821032	valid_1's binary_logloss: 0.149309
    [5]	valid_0's auc: 0.841306	valid_0's binary_logloss: 0.143184	valid_1's auc: 0.824926	valid_1's binary_logloss: 0.147249
    [6]	valid_0's auc: 0.843853	valid_0's binary_logloss: 0.141094	valid_1's auc: 0.825446	valid_1's binary_logloss: 0.145583
    [7]	valid_0's auc: 0.846589	valid_0's binary_logloss: 0.13939	valid_1's auc: 0.829069	valid_1's binary_logloss: 0.144172
    [8]	valid_0's auc: 0.847805	valid_0's binary_logloss: 0.137988	valid_1's auc: 0.829457	valid_1's binary_logloss: 0.143004
    [9]	valid_0's auc: 0.850007	valid_0's binary_logloss: 0.136683	valid_1's auc: 0.831298	valid_1's binary_logloss: 0.141959
    [10]	valid_0's auc: 0.850514	valid_0's binary_logloss: 0.135597	valid_1's auc: 0.830745	valid_1's binary_logloss: 0.141234
    [11]	valid_0's auc: 0.851694	valid_0's binary_logloss: 0.134644	valid_1's auc: 0.831218	valid_1's binary_logloss: 0.140531
    [12]	valid_0's auc: 0.854647	valid_0's binary_logloss: 0.133779	valid_1's auc: 0.832445	valid_1's binary_logloss: 0.140011
    [13]	valid_0's auc: 0.8552	valid_0's binary_logloss: 0.13303	valid_1's auc: 0.832525	valid_1's binary_logloss: 0.139542
    [14]	valid_0's auc: 0.85611	valid_0's binary_logloss: 0.132355	valid_1's auc: 0.832559	valid_1's binary_logloss: 0.139125
    [15]	valid_0's auc: 0.856886	valid_0's binary_logloss: 0.131779	valid_1's auc: 0.833401	valid_1's binary_logloss: 0.138708
    [16]	valid_0's auc: 0.857397	valid_0's binary_logloss: 0.13126	valid_1's auc: 0.833709	valid_1's binary_logloss: 0.138376
    [17]	valid_0's auc: 0.859038	valid_0's binary_logloss: 0.130686	valid_1's auc: 0.83451	valid_1's binary_logloss: 0.138079
    [18]	valid_0's auc: 0.859754	valid_0's binary_logloss: 0.130225	valid_1's auc: 0.834886	valid_1's binary_logloss: 0.137839
    [19]	valid_0's auc: 0.861354	valid_0's binary_logloss: 0.129689	valid_1's auc: 0.835687	valid_1's binary_logloss: 0.137572
    [20]	valid_0's auc: 0.862114	valid_0's binary_logloss: 0.129263	valid_1's auc: 0.83587	valid_1's binary_logloss: 0.137393
    [21]	valid_0's auc: 0.863196	valid_0's binary_logloss: 0.128834	valid_1's auc: 0.835611	valid_1's binary_logloss: 0.137244
    [22]	valid_0's auc: 0.864467	valid_0's binary_logloss: 0.128421	valid_1's auc: 0.836013	valid_1's binary_logloss: 0.137127
    [23]	valid_0's auc: 0.865716	valid_0's binary_logloss: 0.12802	valid_1's auc: 0.835915	valid_1's binary_logloss: 0.13701
    [24]	valid_0's auc: 0.866358	valid_0's binary_logloss: 0.127649	valid_1's auc: 0.835345	valid_1's binary_logloss: 0.136983
    [25]	valid_0's auc: 0.867017	valid_0's binary_logloss: 0.127326	valid_1's auc: 0.835203	valid_1's binary_logloss: 0.13697
    [26]	valid_0's auc: 0.867793	valid_0's binary_logloss: 0.126998	valid_1's auc: 0.835068	valid_1's binary_logloss: 0.13692
    [27]	valid_0's auc: 0.86892	valid_0's binary_logloss: 0.126634	valid_1's auc: 0.835127	valid_1's binary_logloss: 0.136839
    [28]	valid_0's auc: 0.869611	valid_0's binary_logloss: 0.126332	valid_1's auc: 0.835011	valid_1's binary_logloss: 0.136765
    [29]	valid_0's auc: 0.870502	valid_0's binary_logloss: 0.126032	valid_1's auc: 0.835017	valid_1's binary_logloss: 0.136691
    [30]	valid_0's auc: 0.87109	valid_0's binary_logloss: 0.125752	valid_1's auc: 0.835346	valid_1's binary_logloss: 0.136657
    [31]	valid_0's auc: 0.871834	valid_0's binary_logloss: 0.125492	valid_1's auc: 0.835469	valid_1's binary_logloss: 0.136553
    [32]	valid_0's auc: 0.8727	valid_0's binary_logloss: 0.125224	valid_1's auc: 0.834809	valid_1's binary_logloss: 0.136626
    [33]	valid_0's auc: 0.873327	valid_0's binary_logloss: 0.12497	valid_1's auc: 0.834899	valid_1's binary_logloss: 0.136576
    [34]	valid_0's auc: 0.873759	valid_0's binary_logloss: 0.124747	valid_1's auc: 0.835299	valid_1's binary_logloss: 0.136478
    [35]	valid_0's auc: 0.874436	valid_0's binary_logloss: 0.124498	valid_1's auc: 0.834852	valid_1's binary_logloss: 0.136556
    [36]	valid_0's auc: 0.875262	valid_0's binary_logloss: 0.124237	valid_1's auc: 0.834479	valid_1's binary_logloss: 0.13659
    [37]	valid_0's auc: 0.875938	valid_0's binary_logloss: 0.123998	valid_1's auc: 0.834359	valid_1's binary_logloss: 0.136644
    [38]	valid_0's auc: 0.876343	valid_0's binary_logloss: 0.123795	valid_1's auc: 0.834106	valid_1's binary_logloss: 0.136687
    [39]	valid_0's auc: 0.877139	valid_0's binary_logloss: 0.123529	valid_1's auc: 0.83359	valid_1's binary_logloss: 0.136724
    [40]	valid_0's auc: 0.877911	valid_0's binary_logloss: 0.123326	valid_1's auc: 0.833512	valid_1's binary_logloss: 0.136737
    [41]	valid_0's auc: 0.87875	valid_0's binary_logloss: 0.123086	valid_1's auc: 0.833202	valid_1's binary_logloss: 0.136752
    [42]	valid_0's auc: 0.879342	valid_0's binary_logloss: 0.122885	valid_1's auc: 0.832649	valid_1's binary_logloss: 0.136842
    [43]	valid_0's auc: 0.879629	valid_0's binary_logloss: 0.122717	valid_1's auc: 0.832179	valid_1's binary_logloss: 0.136927
    [44]	valid_0's auc: 0.880215	valid_0's binary_logloss: 0.122487	valid_1's auc: 0.832172	valid_1's binary_logloss: 0.13695
    [45]	valid_0's auc: 0.880851	valid_0's binary_logloss: 0.122245	valid_1's auc: 0.831665	valid_1's binary_logloss: 0.137053
    [46]	valid_0's auc: 0.881492	valid_0's binary_logloss: 0.121994	valid_1's auc: 0.83185	valid_1's binary_logloss: 0.137071
    [47]	valid_0's auc: 0.881818	valid_0's binary_logloss: 0.121836	valid_1's auc: 0.831946	valid_1's binary_logloss: 0.137043
    [48]	valid_0's auc: 0.882136	valid_0's binary_logloss: 0.121683	valid_1's auc: 0.832078	valid_1's binary_logloss: 0.137058
    [49]	valid_0's auc: 0.882509	valid_0's binary_logloss: 0.121512	valid_1's auc: 0.832046	valid_1's binary_logloss: 0.13711
    [50]	valid_0's auc: 0.882872	valid_0's binary_logloss: 0.121296	valid_1's auc: 0.83209	valid_1's binary_logloss: 0.137124
    [51]	valid_0's auc: 0.883328	valid_0's binary_logloss: 0.121099	valid_1's auc: 0.831958	valid_1's binary_logloss: 0.137201
    [52]	valid_0's auc: 0.883722	valid_0's binary_logloss: 0.120897	valid_1's auc: 0.831752	valid_1's binary_logloss: 0.13722
    Early stopping, best iteration is:
    [22]	valid_0's auc: 0.864467	valid_0's binary_logloss: 0.128421	valid_1's auc: 0.836013	valid_1's binary_logloss: 0.137127
    [1]	valid_0's auc: 0.822874	valid_0's binary_logloss: 0.157449	valid_1's auc: 0.804936	valid_1's binary_logloss: 0.160126
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.827064	valid_0's binary_logloss: 0.152115	valid_1's auc: 0.810262	valid_1's binary_logloss: 0.155403
    [3]	valid_0's auc: 0.830773	valid_0's binary_logloss: 0.148229	valid_1's auc: 0.814099	valid_1's binary_logloss: 0.151956
    [4]	valid_0's auc: 0.834585	valid_0's binary_logloss: 0.145242	valid_1's auc: 0.815291	valid_1's binary_logloss: 0.149379
    [5]	valid_0's auc: 0.839767	valid_0's binary_logloss: 0.142798	valid_1's auc: 0.819992	valid_1's binary_logloss: 0.147278
    [6]	valid_0's auc: 0.845277	valid_0's binary_logloss: 0.140696	valid_1's auc: 0.824997	valid_1's binary_logloss: 0.145614
    [7]	valid_0's auc: 0.84706	valid_0's binary_logloss: 0.139021	valid_1's auc: 0.827429	valid_1's binary_logloss: 0.144095
    [8]	valid_0's auc: 0.847609	valid_0's binary_logloss: 0.137638	valid_1's auc: 0.827987	valid_1's binary_logloss: 0.143002
    [9]	valid_0's auc: 0.848913	valid_0's binary_logloss: 0.136364	valid_1's auc: 0.82986	valid_1's binary_logloss: 0.141984
    [10]	valid_0's auc: 0.850178	valid_0's binary_logloss: 0.135236	valid_1's auc: 0.829651	valid_1's binary_logloss: 0.141228
    [11]	valid_0's auc: 0.85127	valid_0's binary_logloss: 0.134273	valid_1's auc: 0.830496	valid_1's binary_logloss: 0.140509
    [12]	valid_0's auc: 0.852586	valid_0's binary_logloss: 0.133411	valid_1's auc: 0.830525	valid_1's binary_logloss: 0.13997
    [13]	valid_0's auc: 0.853732	valid_0's binary_logloss: 0.132666	valid_1's auc: 0.829319	valid_1's binary_logloss: 0.13957
    [14]	valid_0's auc: 0.854775	valid_0's binary_logloss: 0.13201	valid_1's auc: 0.832116	valid_1's binary_logloss: 0.139106
    [15]	valid_0's auc: 0.855714	valid_0's binary_logloss: 0.131409	valid_1's auc: 0.833454	valid_1's binary_logloss: 0.138684
    [16]	valid_0's auc: 0.856099	valid_0's binary_logloss: 0.130853	valid_1's auc: 0.832243	valid_1's binary_logloss: 0.138485
    [17]	valid_0's auc: 0.857475	valid_0's binary_logloss: 0.130295	valid_1's auc: 0.832577	valid_1's binary_logloss: 0.138179
    [18]	valid_0's auc: 0.857908	valid_0's binary_logloss: 0.129866	valid_1's auc: 0.832652	valid_1's binary_logloss: 0.138084
    [19]	valid_0's auc: 0.859226	valid_0's binary_logloss: 0.129384	valid_1's auc: 0.832887	valid_1's binary_logloss: 0.137828
    [20]	valid_0's auc: 0.860296	valid_0's binary_logloss: 0.129003	valid_1's auc: 0.834111	valid_1's binary_logloss: 0.137643
    [21]	valid_0's auc: 0.860934	valid_0's binary_logloss: 0.128619	valid_1's auc: 0.834602	valid_1's binary_logloss: 0.137471
    [22]	valid_0's auc: 0.861712	valid_0's binary_logloss: 0.12825	valid_1's auc: 0.834828	valid_1's binary_logloss: 0.137291
    [23]	valid_0's auc: 0.862939	valid_0's binary_logloss: 0.127869	valid_1's auc: 0.835767	valid_1's binary_logloss: 0.137067
    [24]	valid_0's auc: 0.863968	valid_0's binary_logloss: 0.127525	valid_1's auc: 0.835485	valid_1's binary_logloss: 0.13701
    [25]	valid_0's auc: 0.865029	valid_0's binary_logloss: 0.127234	valid_1's auc: 0.835507	valid_1's binary_logloss: 0.136911
    [26]	valid_0's auc: 0.865758	valid_0's binary_logloss: 0.126952	valid_1's auc: 0.835743	valid_1's binary_logloss: 0.136823
    [27]	valid_0's auc: 0.866244	valid_0's binary_logloss: 0.126688	valid_1's auc: 0.836049	valid_1's binary_logloss: 0.136736
    [28]	valid_0's auc: 0.866607	valid_0's binary_logloss: 0.126455	valid_1's auc: 0.836043	valid_1's binary_logloss: 0.136687
    [29]	valid_0's auc: 0.867322	valid_0's binary_logloss: 0.126255	valid_1's auc: 0.836375	valid_1's binary_logloss: 0.136596
    [30]	valid_0's auc: 0.867997	valid_0's binary_logloss: 0.126022	valid_1's auc: 0.836518	valid_1's binary_logloss: 0.136563
    [31]	valid_0's auc: 0.868888	valid_0's binary_logloss: 0.125781	valid_1's auc: 0.836949	valid_1's binary_logloss: 0.13648
    [32]	valid_0's auc: 0.869544	valid_0's binary_logloss: 0.125524	valid_1's auc: 0.836774	valid_1's binary_logloss: 0.136506
    [33]	valid_0's auc: 0.870314	valid_0's binary_logloss: 0.125324	valid_1's auc: 0.837039	valid_1's binary_logloss: 0.13642
    [34]	valid_0's auc: 0.871207	valid_0's binary_logloss: 0.125096	valid_1's auc: 0.836859	valid_1's binary_logloss: 0.136441
    [35]	valid_0's auc: 0.871788	valid_0's binary_logloss: 0.124883	valid_1's auc: 0.836641	valid_1's binary_logloss: 0.136467
    [36]	valid_0's auc: 0.872403	valid_0's binary_logloss: 0.124635	valid_1's auc: 0.836917	valid_1's binary_logloss: 0.13642
    [37]	valid_0's auc: 0.872914	valid_0's binary_logloss: 0.124439	valid_1's auc: 0.836892	valid_1's binary_logloss: 0.136489
    [38]	valid_0's auc: 0.873469	valid_0's binary_logloss: 0.124233	valid_1's auc: 0.836816	valid_1's binary_logloss: 0.136502
    [39]	valid_0's auc: 0.873971	valid_0's binary_logloss: 0.124017	valid_1's auc: 0.83659	valid_1's binary_logloss: 0.136543
    [40]	valid_0's auc: 0.875013	valid_0's binary_logloss: 0.123715	valid_1's auc: 0.836381	valid_1's binary_logloss: 0.136593
    [41]	valid_0's auc: 0.875341	valid_0's binary_logloss: 0.123538	valid_1's auc: 0.8363	valid_1's binary_logloss: 0.136624
    [42]	valid_0's auc: 0.875886	valid_0's binary_logloss: 0.123285	valid_1's auc: 0.836043	valid_1's binary_logloss: 0.136665
    [43]	valid_0's auc: 0.87633	valid_0's binary_logloss: 0.12311	valid_1's auc: 0.836018	valid_1's binary_logloss: 0.136647
    [44]	valid_0's auc: 0.876816	valid_0's binary_logloss: 0.122878	valid_1's auc: 0.836061	valid_1's binary_logloss: 0.136664
    [45]	valid_0's auc: 0.877116	valid_0's binary_logloss: 0.12271	valid_1's auc: 0.836517	valid_1's binary_logloss: 0.136608
    [46]	valid_0's auc: 0.877567	valid_0's binary_logloss: 0.122536	valid_1's auc: 0.836051	valid_1's binary_logloss: 0.13667
    [47]	valid_0's auc: 0.878192	valid_0's binary_logloss: 0.122332	valid_1's auc: 0.835827	valid_1's binary_logloss: 0.136694
    [48]	valid_0's auc: 0.878371	valid_0's binary_logloss: 0.122208	valid_1's auc: 0.835831	valid_1's binary_logloss: 0.136717
    [49]	valid_0's auc: 0.878669	valid_0's binary_logloss: 0.12204	valid_1's auc: 0.835913	valid_1's binary_logloss: 0.136679
    [50]	valid_0's auc: 0.879288	valid_0's binary_logloss: 0.121825	valid_1's auc: 0.835468	valid_1's binary_logloss: 0.136734
    [51]	valid_0's auc: 0.879552	valid_0's binary_logloss: 0.121647	valid_1's auc: 0.835095	valid_1's binary_logloss: 0.13684
    [52]	valid_0's auc: 0.880033	valid_0's binary_logloss: 0.121459	valid_1's auc: 0.83496	valid_1's binary_logloss: 0.136927
    [53]	valid_0's auc: 0.880447	valid_0's binary_logloss: 0.121297	valid_1's auc: 0.834955	valid_1's binary_logloss: 0.136953
    [54]	valid_0's auc: 0.88087	valid_0's binary_logloss: 0.121116	valid_1's auc: 0.834926	valid_1's binary_logloss: 0.136994
    [55]	valid_0's auc: 0.881105	valid_0's binary_logloss: 0.12097	valid_1's auc: 0.834861	valid_1's binary_logloss: 0.137043
    [56]	valid_0's auc: 0.88161	valid_0's binary_logloss: 0.120772	valid_1's auc: 0.834629	valid_1's binary_logloss: 0.137125
    [57]	valid_0's auc: 0.882134	valid_0's binary_logloss: 0.120595	valid_1's auc: 0.834576	valid_1's binary_logloss: 0.137157
    [58]	valid_0's auc: 0.882429	valid_0's binary_logloss: 0.120441	valid_1's auc: 0.834319	valid_1's binary_logloss: 0.137209
    [59]	valid_0's auc: 0.882799	valid_0's binary_logloss: 0.120268	valid_1's auc: 0.834028	valid_1's binary_logloss: 0.1373
    [60]	valid_0's auc: 0.883147	valid_0's binary_logloss: 0.120094	valid_1's auc: 0.833999	valid_1's binary_logloss: 0.137339
    [61]	valid_0's auc: 0.883628	valid_0's binary_logloss: 0.119914	valid_1's auc: 0.834234	valid_1's binary_logloss: 0.137314
    [62]	valid_0's auc: 0.884034	valid_0's binary_logloss: 0.119753	valid_1's auc: 0.834179	valid_1's binary_logloss: 0.137356
    [63]	valid_0's auc: 0.884169	valid_0's binary_logloss: 0.119623	valid_1's auc: 0.834035	valid_1's binary_logloss: 0.137378
    Early stopping, best iteration is:
    [33]	valid_0's auc: 0.870314	valid_0's binary_logloss: 0.125324	valid_1's auc: 0.837039	valid_1's binary_logloss: 0.13642
    [1]	valid_0's auc: 0.826066	valid_0's binary_logloss: 0.157509	valid_1's auc: 0.810763	valid_1's binary_logloss: 0.160177
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.830558	valid_0's binary_logloss: 0.152317	valid_1's auc: 0.814406	valid_1's binary_logloss: 0.15553
    [3]	valid_0's auc: 0.833526	valid_0's binary_logloss: 0.148529	valid_1's auc: 0.815841	valid_1's binary_logloss: 0.152134
    [4]	valid_0's auc: 0.839653	valid_0's binary_logloss: 0.145555	valid_1's auc: 0.821767	valid_1's binary_logloss: 0.149519
    [5]	valid_0's auc: 0.842943	valid_0's binary_logloss: 0.143073	valid_1's auc: 0.826334	valid_1's binary_logloss: 0.14736
    [6]	valid_0's auc: 0.845929	valid_0's binary_logloss: 0.141096	valid_1's auc: 0.828882	valid_1's binary_logloss: 0.145649
    [7]	valid_0's auc: 0.847231	valid_0's binary_logloss: 0.139394	valid_1's auc: 0.830621	valid_1's binary_logloss: 0.144201
    [8]	valid_0's auc: 0.849609	valid_0's binary_logloss: 0.137956	valid_1's auc: 0.831324	valid_1's binary_logloss: 0.143105
    [9]	valid_0's auc: 0.851333	valid_0's binary_logloss: 0.136733	valid_1's auc: 0.83377	valid_1's binary_logloss: 0.142018
    [10]	valid_0's auc: 0.852692	valid_0's binary_logloss: 0.135654	valid_1's auc: 0.834493	valid_1's binary_logloss: 0.141149
    [11]	valid_0's auc: 0.854022	valid_0's binary_logloss: 0.134688	valid_1's auc: 0.834616	valid_1's binary_logloss: 0.140437
    [12]	valid_0's auc: 0.854419	valid_0's binary_logloss: 0.13393	valid_1's auc: 0.83589	valid_1's binary_logloss: 0.139765
    [13]	valid_0's auc: 0.854853	valid_0's binary_logloss: 0.133229	valid_1's auc: 0.835999	valid_1's binary_logloss: 0.139284
    [14]	valid_0's auc: 0.856241	valid_0's binary_logloss: 0.132532	valid_1's auc: 0.837392	valid_1's binary_logloss: 0.138795
    [15]	valid_0's auc: 0.85803	valid_0's binary_logloss: 0.131862	valid_1's auc: 0.83818	valid_1's binary_logloss: 0.138366
    [16]	valid_0's auc: 0.859289	valid_0's binary_logloss: 0.131311	valid_1's auc: 0.838523	valid_1's binary_logloss: 0.138137
    [17]	valid_0's auc: 0.860115	valid_0's binary_logloss: 0.130812	valid_1's auc: 0.839359	valid_1's binary_logloss: 0.137816
    [18]	valid_0's auc: 0.860807	valid_0's binary_logloss: 0.130408	valid_1's auc: 0.840179	valid_1's binary_logloss: 0.13749
    [19]	valid_0's auc: 0.861568	valid_0's binary_logloss: 0.129954	valid_1's auc: 0.839855	valid_1's binary_logloss: 0.137387
    [20]	valid_0's auc: 0.863194	valid_0's binary_logloss: 0.129482	valid_1's auc: 0.839715	valid_1's binary_logloss: 0.137226
    [21]	valid_0's auc: 0.864273	valid_0's binary_logloss: 0.12907	valid_1's auc: 0.840516	valid_1's binary_logloss: 0.136991
    [22]	valid_0's auc: 0.865334	valid_0's binary_logloss: 0.128648	valid_1's auc: 0.841302	valid_1's binary_logloss: 0.136769
    [23]	valid_0's auc: 0.866353	valid_0's binary_logloss: 0.128302	valid_1's auc: 0.841085	valid_1's binary_logloss: 0.136653
    [24]	valid_0's auc: 0.86699	valid_0's binary_logloss: 0.127957	valid_1's auc: 0.841235	valid_1's binary_logloss: 0.136611
    [25]	valid_0's auc: 0.867587	valid_0's binary_logloss: 0.127634	valid_1's auc: 0.840948	valid_1's binary_logloss: 0.136644
    [26]	valid_0's auc: 0.868507	valid_0's binary_logloss: 0.127298	valid_1's auc: 0.841033	valid_1's binary_logloss: 0.136526
    [27]	valid_0's auc: 0.869362	valid_0's binary_logloss: 0.126976	valid_1's auc: 0.840939	valid_1's binary_logloss: 0.136445
    [28]	valid_0's auc: 0.870126	valid_0's binary_logloss: 0.126688	valid_1's auc: 0.840452	valid_1's binary_logloss: 0.13645
    [29]	valid_0's auc: 0.870568	valid_0's binary_logloss: 0.126442	valid_1's auc: 0.840809	valid_1's binary_logloss: 0.136345
    [30]	valid_0's auc: 0.871323	valid_0's binary_logloss: 0.126166	valid_1's auc: 0.840599	valid_1's binary_logloss: 0.136334
    [31]	valid_0's auc: 0.872181	valid_0's binary_logloss: 0.125929	valid_1's auc: 0.840663	valid_1's binary_logloss: 0.136267
    [32]	valid_0's auc: 0.873089	valid_0's binary_logloss: 0.125641	valid_1's auc: 0.841679	valid_1's binary_logloss: 0.136141
    [33]	valid_0's auc: 0.873443	valid_0's binary_logloss: 0.125415	valid_1's auc: 0.841597	valid_1's binary_logloss: 0.136138
    [34]	valid_0's auc: 0.874111	valid_0's binary_logloss: 0.125163	valid_1's auc: 0.841194	valid_1's binary_logloss: 0.13614
    [35]	valid_0's auc: 0.874804	valid_0's binary_logloss: 0.124891	valid_1's auc: 0.841136	valid_1's binary_logloss: 0.136214
    [36]	valid_0's auc: 0.875123	valid_0's binary_logloss: 0.124721	valid_1's auc: 0.840883	valid_1's binary_logloss: 0.136268
    [37]	valid_0's auc: 0.875964	valid_0's binary_logloss: 0.124479	valid_1's auc: 0.840327	valid_1's binary_logloss: 0.136363
    [38]	valid_0's auc: 0.876484	valid_0's binary_logloss: 0.12422	valid_1's auc: 0.840317	valid_1's binary_logloss: 0.136378
    [39]	valid_0's auc: 0.877529	valid_0's binary_logloss: 0.123975	valid_1's auc: 0.840224	valid_1's binary_logloss: 0.136372
    [40]	valid_0's auc: 0.877916	valid_0's binary_logloss: 0.123803	valid_1's auc: 0.840099	valid_1's binary_logloss: 0.136418
    [41]	valid_0's auc: 0.878404	valid_0's binary_logloss: 0.123606	valid_1's auc: 0.839447	valid_1's binary_logloss: 0.136498
    [42]	valid_0's auc: 0.878787	valid_0's binary_logloss: 0.123458	valid_1's auc: 0.839737	valid_1's binary_logloss: 0.136466
    [43]	valid_0's auc: 0.879296	valid_0's binary_logloss: 0.123263	valid_1's auc: 0.838848	valid_1's binary_logloss: 0.136611
    [44]	valid_0's auc: 0.879561	valid_0's binary_logloss: 0.123111	valid_1's auc: 0.83939	valid_1's binary_logloss: 0.136509
    [45]	valid_0's auc: 0.880167	valid_0's binary_logloss: 0.122912	valid_1's auc: 0.83916	valid_1's binary_logloss: 0.136564
    [46]	valid_0's auc: 0.880652	valid_0's binary_logloss: 0.12273	valid_1's auc: 0.839133	valid_1's binary_logloss: 0.136588
    [47]	valid_0's auc: 0.881351	valid_0's binary_logloss: 0.122478	valid_1's auc: 0.839108	valid_1's binary_logloss: 0.136599
    [48]	valid_0's auc: 0.881786	valid_0's binary_logloss: 0.122308	valid_1's auc: 0.838894	valid_1's binary_logloss: 0.136636
    [49]	valid_0's auc: 0.88242	valid_0's binary_logloss: 0.122054	valid_1's auc: 0.838476	valid_1's binary_logloss: 0.136727
    [50]	valid_0's auc: 0.88279	valid_0's binary_logloss: 0.121862	valid_1's auc: 0.837735	valid_1's binary_logloss: 0.136833
    [51]	valid_0's auc: 0.883422	valid_0's binary_logloss: 0.121646	valid_1's auc: 0.837691	valid_1's binary_logloss: 0.136833
    [52]	valid_0's auc: 0.884035	valid_0's binary_logloss: 0.121461	valid_1's auc: 0.837503	valid_1's binary_logloss: 0.136856
    [53]	valid_0's auc: 0.884636	valid_0's binary_logloss: 0.121237	valid_1's auc: 0.837159	valid_1's binary_logloss: 0.136933
    [54]	valid_0's auc: 0.885019	valid_0's binary_logloss: 0.121086	valid_1's auc: 0.836758	valid_1's binary_logloss: 0.136995
    [55]	valid_0's auc: 0.88546	valid_0's binary_logloss: 0.120908	valid_1's auc: 0.836692	valid_1's binary_logloss: 0.137006
    [56]	valid_0's auc: 0.885711	valid_0's binary_logloss: 0.120745	valid_1's auc: 0.836703	valid_1's binary_logloss: 0.136998
    [57]	valid_0's auc: 0.886081	valid_0's binary_logloss: 0.120544	valid_1's auc: 0.836462	valid_1's binary_logloss: 0.137073
    [58]	valid_0's auc: 0.886576	valid_0's binary_logloss: 0.120352	valid_1's auc: 0.836241	valid_1's binary_logloss: 0.137127
    [59]	valid_0's auc: 0.886878	valid_0's binary_logloss: 0.120191	valid_1's auc: 0.8358	valid_1's binary_logloss: 0.137207
    [60]	valid_0's auc: 0.887158	valid_0's binary_logloss: 0.120005	valid_1's auc: 0.835688	valid_1's binary_logloss: 0.137272
    [61]	valid_0's auc: 0.88755	valid_0's binary_logloss: 0.119855	valid_1's auc: 0.835652	valid_1's binary_logloss: 0.137262
    [62]	valid_0's auc: 0.887817	valid_0's binary_logloss: 0.11971	valid_1's auc: 0.835355	valid_1's binary_logloss: 0.13732
    Early stopping, best iteration is:
    [32]	valid_0's auc: 0.873089	valid_0's binary_logloss: 0.125641	valid_1's auc: 0.841679	valid_1's binary_logloss: 0.136141
    [1]	valid_0's auc: 0.822527	valid_0's binary_logloss: 0.157702	valid_1's auc: 0.811735	valid_1's binary_logloss: 0.160053
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.828897	valid_0's binary_logloss: 0.152381	valid_1's auc: 0.815279	valid_1's binary_logloss: 0.155308
    [3]	valid_0's auc: 0.833978	valid_0's binary_logloss: 0.148529	valid_1's auc: 0.819863	valid_1's binary_logloss: 0.151959
    [4]	valid_0's auc: 0.837641	valid_0's binary_logloss: 0.145558	valid_1's auc: 0.821032	valid_1's binary_logloss: 0.149309
    [5]	valid_0's auc: 0.841306	valid_0's binary_logloss: 0.143184	valid_1's auc: 0.824926	valid_1's binary_logloss: 0.147249
    [6]	valid_0's auc: 0.843853	valid_0's binary_logloss: 0.141094	valid_1's auc: 0.825446	valid_1's binary_logloss: 0.145583
    [7]	valid_0's auc: 0.846589	valid_0's binary_logloss: 0.13939	valid_1's auc: 0.829069	valid_1's binary_logloss: 0.144172
    [8]	valid_0's auc: 0.847805	valid_0's binary_logloss: 0.137988	valid_1's auc: 0.829457	valid_1's binary_logloss: 0.143004
    [9]	valid_0's auc: 0.850007	valid_0's binary_logloss: 0.136683	valid_1's auc: 0.831298	valid_1's binary_logloss: 0.141959
    [10]	valid_0's auc: 0.850514	valid_0's binary_logloss: 0.135597	valid_1's auc: 0.830745	valid_1's binary_logloss: 0.141234
    [11]	valid_0's auc: 0.851694	valid_0's binary_logloss: 0.134644	valid_1's auc: 0.831218	valid_1's binary_logloss: 0.140531
    [12]	valid_0's auc: 0.854647	valid_0's binary_logloss: 0.133779	valid_1's auc: 0.832445	valid_1's binary_logloss: 0.140011
    [13]	valid_0's auc: 0.8552	valid_0's binary_logloss: 0.13303	valid_1's auc: 0.832525	valid_1's binary_logloss: 0.139542
    [14]	valid_0's auc: 0.85611	valid_0's binary_logloss: 0.132355	valid_1's auc: 0.832559	valid_1's binary_logloss: 0.139125
    [15]	valid_0's auc: 0.856886	valid_0's binary_logloss: 0.131779	valid_1's auc: 0.833401	valid_1's binary_logloss: 0.138708
    [16]	valid_0's auc: 0.857397	valid_0's binary_logloss: 0.13126	valid_1's auc: 0.833709	valid_1's binary_logloss: 0.138376
    [17]	valid_0's auc: 0.859038	valid_0's binary_logloss: 0.130686	valid_1's auc: 0.83451	valid_1's binary_logloss: 0.138079
    [18]	valid_0's auc: 0.859754	valid_0's binary_logloss: 0.130225	valid_1's auc: 0.834886	valid_1's binary_logloss: 0.137839
    [19]	valid_0's auc: 0.861354	valid_0's binary_logloss: 0.129689	valid_1's auc: 0.835687	valid_1's binary_logloss: 0.137572
    [20]	valid_0's auc: 0.862114	valid_0's binary_logloss: 0.129263	valid_1's auc: 0.83587	valid_1's binary_logloss: 0.137393
    [21]	valid_0's auc: 0.863196	valid_0's binary_logloss: 0.128834	valid_1's auc: 0.835611	valid_1's binary_logloss: 0.137244
    [22]	valid_0's auc: 0.864467	valid_0's binary_logloss: 0.128421	valid_1's auc: 0.836013	valid_1's binary_logloss: 0.137127
    [23]	valid_0's auc: 0.865716	valid_0's binary_logloss: 0.12802	valid_1's auc: 0.835915	valid_1's binary_logloss: 0.13701
    [24]	valid_0's auc: 0.866358	valid_0's binary_logloss: 0.127649	valid_1's auc: 0.835345	valid_1's binary_logloss: 0.136983
    [25]	valid_0's auc: 0.867017	valid_0's binary_logloss: 0.127326	valid_1's auc: 0.835203	valid_1's binary_logloss: 0.13697
    [26]	valid_0's auc: 0.867793	valid_0's binary_logloss: 0.126998	valid_1's auc: 0.835068	valid_1's binary_logloss: 0.13692
    [27]	valid_0's auc: 0.86892	valid_0's binary_logloss: 0.126634	valid_1's auc: 0.835127	valid_1's binary_logloss: 0.136839
    [28]	valid_0's auc: 0.869611	valid_0's binary_logloss: 0.126332	valid_1's auc: 0.835011	valid_1's binary_logloss: 0.136765
    [29]	valid_0's auc: 0.870502	valid_0's binary_logloss: 0.126032	valid_1's auc: 0.835017	valid_1's binary_logloss: 0.136691
    [30]	valid_0's auc: 0.87109	valid_0's binary_logloss: 0.125752	valid_1's auc: 0.835346	valid_1's binary_logloss: 0.136657
    [31]	valid_0's auc: 0.871834	valid_0's binary_logloss: 0.125492	valid_1's auc: 0.835469	valid_1's binary_logloss: 0.136553
    [32]	valid_0's auc: 0.8727	valid_0's binary_logloss: 0.125224	valid_1's auc: 0.834809	valid_1's binary_logloss: 0.136626
    [33]	valid_0's auc: 0.873327	valid_0's binary_logloss: 0.12497	valid_1's auc: 0.834899	valid_1's binary_logloss: 0.136576
    [34]	valid_0's auc: 0.873759	valid_0's binary_logloss: 0.124747	valid_1's auc: 0.835299	valid_1's binary_logloss: 0.136478
    [35]	valid_0's auc: 0.874436	valid_0's binary_logloss: 0.124498	valid_1's auc: 0.834852	valid_1's binary_logloss: 0.136556
    [36]	valid_0's auc: 0.875262	valid_0's binary_logloss: 0.124237	valid_1's auc: 0.834479	valid_1's binary_logloss: 0.13659
    [37]	valid_0's auc: 0.875938	valid_0's binary_logloss: 0.123998	valid_1's auc: 0.834359	valid_1's binary_logloss: 0.136644
    [38]	valid_0's auc: 0.876343	valid_0's binary_logloss: 0.123795	valid_1's auc: 0.834106	valid_1's binary_logloss: 0.136687
    [39]	valid_0's auc: 0.877139	valid_0's binary_logloss: 0.123529	valid_1's auc: 0.83359	valid_1's binary_logloss: 0.136724
    [40]	valid_0's auc: 0.877911	valid_0's binary_logloss: 0.123326	valid_1's auc: 0.833512	valid_1's binary_logloss: 0.136737
    [41]	valid_0's auc: 0.87875	valid_0's binary_logloss: 0.123086	valid_1's auc: 0.833202	valid_1's binary_logloss: 0.136752
    [42]	valid_0's auc: 0.879342	valid_0's binary_logloss: 0.122885	valid_1's auc: 0.832649	valid_1's binary_logloss: 0.136842
    [43]	valid_0's auc: 0.879629	valid_0's binary_logloss: 0.122717	valid_1's auc: 0.832179	valid_1's binary_logloss: 0.136927
    [44]	valid_0's auc: 0.880215	valid_0's binary_logloss: 0.122487	valid_1's auc: 0.832172	valid_1's binary_logloss: 0.13695
    [45]	valid_0's auc: 0.880851	valid_0's binary_logloss: 0.122245	valid_1's auc: 0.831665	valid_1's binary_logloss: 0.137053
    [46]	valid_0's auc: 0.881492	valid_0's binary_logloss: 0.121994	valid_1's auc: 0.83185	valid_1's binary_logloss: 0.137071
    [47]	valid_0's auc: 0.881818	valid_0's binary_logloss: 0.121836	valid_1's auc: 0.831946	valid_1's binary_logloss: 0.137043
    [48]	valid_0's auc: 0.882136	valid_0's binary_logloss: 0.121683	valid_1's auc: 0.832078	valid_1's binary_logloss: 0.137058
    [49]	valid_0's auc: 0.882509	valid_0's binary_logloss: 0.121512	valid_1's auc: 0.832046	valid_1's binary_logloss: 0.13711
    [50]	valid_0's auc: 0.882872	valid_0's binary_logloss: 0.121296	valid_1's auc: 0.83209	valid_1's binary_logloss: 0.137124
    [51]	valid_0's auc: 0.883328	valid_0's binary_logloss: 0.121099	valid_1's auc: 0.831958	valid_1's binary_logloss: 0.137201
    [52]	valid_0's auc: 0.883722	valid_0's binary_logloss: 0.120897	valid_1's auc: 0.831752	valid_1's binary_logloss: 0.13722
    Early stopping, best iteration is:
    [22]	valid_0's auc: 0.864467	valid_0's binary_logloss: 0.128421	valid_1's auc: 0.836013	valid_1's binary_logloss: 0.137127
    [1]	valid_0's auc: 0.829442	valid_0's binary_logloss: 0.156848	valid_1's auc: 0.801853	valid_1's binary_logloss: 0.159917
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.835044	valid_0's binary_logloss: 0.151129	valid_1's auc: 0.810636	valid_1's binary_logloss: 0.155191
    [3]	valid_0's auc: 0.84106	valid_0's binary_logloss: 0.146986	valid_1's auc: 0.81428	valid_1's binary_logloss: 0.151786
    [4]	valid_0's auc: 0.849104	valid_0's binary_logloss: 0.143746	valid_1's auc: 0.822369	valid_1's binary_logloss: 0.149174
    [5]	valid_0's auc: 0.852054	valid_0's binary_logloss: 0.14111	valid_1's auc: 0.825422	valid_1's binary_logloss: 0.147108
    [6]	valid_0's auc: 0.854219	valid_0's binary_logloss: 0.138941	valid_1's auc: 0.828128	valid_1's binary_logloss: 0.145314
    [7]	valid_0's auc: 0.857758	valid_0's binary_logloss: 0.136956	valid_1's auc: 0.828178	valid_1's binary_logloss: 0.144053
    [8]	valid_0's auc: 0.859433	valid_0's binary_logloss: 0.135341	valid_1's auc: 0.828907	valid_1's binary_logloss: 0.142933
    [9]	valid_0's auc: 0.860562	valid_0's binary_logloss: 0.133965	valid_1's auc: 0.830266	valid_1's binary_logloss: 0.141913
    [10]	valid_0's auc: 0.862253	valid_0's binary_logloss: 0.132732	valid_1's auc: 0.830436	valid_1's binary_logloss: 0.141159
    [11]	valid_0's auc: 0.863863	valid_0's binary_logloss: 0.131616	valid_1's auc: 0.830888	valid_1's binary_logloss: 0.140465
    [12]	valid_0's auc: 0.86535	valid_0's binary_logloss: 0.130642	valid_1's auc: 0.828473	valid_1's binary_logloss: 0.140089
    [13]	valid_0's auc: 0.866315	valid_0's binary_logloss: 0.129707	valid_1's auc: 0.828442	valid_1's binary_logloss: 0.139673
    [14]	valid_0's auc: 0.868091	valid_0's binary_logloss: 0.128809	valid_1's auc: 0.828803	valid_1's binary_logloss: 0.13925
    [15]	valid_0's auc: 0.869154	valid_0's binary_logloss: 0.128055	valid_1's auc: 0.829147	valid_1's binary_logloss: 0.138981
    [16]	valid_0's auc: 0.870086	valid_0's binary_logloss: 0.127376	valid_1's auc: 0.829188	valid_1's binary_logloss: 0.13878
    [17]	valid_0's auc: 0.870857	valid_0's binary_logloss: 0.126705	valid_1's auc: 0.829277	valid_1's binary_logloss: 0.138553
    [18]	valid_0's auc: 0.87202	valid_0's binary_logloss: 0.126023	valid_1's auc: 0.829614	valid_1's binary_logloss: 0.138375
    [19]	valid_0's auc: 0.873171	valid_0's binary_logloss: 0.125459	valid_1's auc: 0.830335	valid_1's binary_logloss: 0.13824
    [20]	valid_0's auc: 0.874593	valid_0's binary_logloss: 0.124876	valid_1's auc: 0.831752	valid_1's binary_logloss: 0.13798
    [21]	valid_0's auc: 0.875741	valid_0's binary_logloss: 0.124281	valid_1's auc: 0.832761	valid_1's binary_logloss: 0.137687
    [22]	valid_0's auc: 0.877448	valid_0's binary_logloss: 0.123659	valid_1's auc: 0.833679	valid_1's binary_logloss: 0.137492
    [23]	valid_0's auc: 0.878493	valid_0's binary_logloss: 0.123115	valid_1's auc: 0.833912	valid_1's binary_logloss: 0.137434
    [24]	valid_0's auc: 0.879193	valid_0's binary_logloss: 0.122682	valid_1's auc: 0.833363	valid_1's binary_logloss: 0.137364
    [25]	valid_0's auc: 0.880227	valid_0's binary_logloss: 0.122229	valid_1's auc: 0.833173	valid_1's binary_logloss: 0.137358
    [26]	valid_0's auc: 0.881585	valid_0's binary_logloss: 0.121752	valid_1's auc: 0.832904	valid_1's binary_logloss: 0.13741
    [27]	valid_0's auc: 0.882473	valid_0's binary_logloss: 0.1213	valid_1's auc: 0.832408	valid_1's binary_logloss: 0.137589
    [28]	valid_0's auc: 0.883437	valid_0's binary_logloss: 0.120929	valid_1's auc: 0.832986	valid_1's binary_logloss: 0.137512
    [29]	valid_0's auc: 0.884502	valid_0's binary_logloss: 0.120484	valid_1's auc: 0.833732	valid_1's binary_logloss: 0.137414
    [30]	valid_0's auc: 0.88539	valid_0's binary_logloss: 0.120132	valid_1's auc: 0.83355	valid_1's binary_logloss: 0.137429
    [31]	valid_0's auc: 0.886163	valid_0's binary_logloss: 0.11975	valid_1's auc: 0.833288	valid_1's binary_logloss: 0.137459
    [32]	valid_0's auc: 0.886867	valid_0's binary_logloss: 0.119396	valid_1's auc: 0.833051	valid_1's binary_logloss: 0.137503
    [33]	valid_0's auc: 0.887733	valid_0's binary_logloss: 0.119013	valid_1's auc: 0.832946	valid_1's binary_logloss: 0.137467
    [34]	valid_0's auc: 0.888681	valid_0's binary_logloss: 0.118651	valid_1's auc: 0.833519	valid_1's binary_logloss: 0.137391
    [35]	valid_0's auc: 0.889275	valid_0's binary_logloss: 0.118341	valid_1's auc: 0.833734	valid_1's binary_logloss: 0.137371
    [36]	valid_0's auc: 0.890084	valid_0's binary_logloss: 0.117965	valid_1's auc: 0.833231	valid_1's binary_logloss: 0.137505
    [37]	valid_0's auc: 0.890726	valid_0's binary_logloss: 0.11765	valid_1's auc: 0.833136	valid_1's binary_logloss: 0.13761
    [38]	valid_0's auc: 0.891158	valid_0's binary_logloss: 0.11734	valid_1's auc: 0.832938	valid_1's binary_logloss: 0.137668
    [39]	valid_0's auc: 0.89196	valid_0's binary_logloss: 0.117006	valid_1's auc: 0.832307	valid_1's binary_logloss: 0.13782
    [40]	valid_0's auc: 0.892449	valid_0's binary_logloss: 0.116703	valid_1's auc: 0.832354	valid_1's binary_logloss: 0.137858
    [41]	valid_0's auc: 0.892978	valid_0's binary_logloss: 0.116414	valid_1's auc: 0.831731	valid_1's binary_logloss: 0.137998
    [42]	valid_0's auc: 0.893542	valid_0's binary_logloss: 0.116071	valid_1's auc: 0.830942	valid_1's binary_logloss: 0.138189
    [43]	valid_0's auc: 0.894078	valid_0's binary_logloss: 0.115805	valid_1's auc: 0.830997	valid_1's binary_logloss: 0.138182
    [44]	valid_0's auc: 0.894576	valid_0's binary_logloss: 0.115516	valid_1's auc: 0.830968	valid_1's binary_logloss: 0.138257
    [45]	valid_0's auc: 0.895099	valid_0's binary_logloss: 0.115236	valid_1's auc: 0.830506	valid_1's binary_logloss: 0.13839
    [46]	valid_0's auc: 0.895608	valid_0's binary_logloss: 0.114971	valid_1's auc: 0.829892	valid_1's binary_logloss: 0.138556
    [47]	valid_0's auc: 0.895999	valid_0's binary_logloss: 0.1147	valid_1's auc: 0.829591	valid_1's binary_logloss: 0.138649
    [48]	valid_0's auc: 0.89641	valid_0's binary_logloss: 0.11442	valid_1's auc: 0.829768	valid_1's binary_logloss: 0.138698
    [49]	valid_0's auc: 0.896742	valid_0's binary_logloss: 0.114182	valid_1's auc: 0.830072	valid_1's binary_logloss: 0.138692
    [50]	valid_0's auc: 0.896929	valid_0's binary_logloss: 0.113989	valid_1's auc: 0.829948	valid_1's binary_logloss: 0.138769
    [51]	valid_0's auc: 0.897717	valid_0's binary_logloss: 0.113718	valid_1's auc: 0.830297	valid_1's binary_logloss: 0.138742
    [52]	valid_0's auc: 0.898093	valid_0's binary_logloss: 0.113472	valid_1's auc: 0.830332	valid_1's binary_logloss: 0.138776
    [53]	valid_0's auc: 0.898387	valid_0's binary_logloss: 0.113271	valid_1's auc: 0.830278	valid_1's binary_logloss: 0.138833
    Early stopping, best iteration is:
    [23]	valid_0's auc: 0.878493	valid_0's binary_logloss: 0.123115	valid_1's auc: 0.833912	valid_1's binary_logloss: 0.137434
    [1]	valid_0's auc: 0.834008	valid_0's binary_logloss: 0.156932	valid_1's auc: 0.806689	valid_1's binary_logloss: 0.159986
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.841271	valid_0's binary_logloss: 0.151528	valid_1's auc: 0.816568	valid_1's binary_logloss: 0.155337
    [3]	valid_0's auc: 0.843827	valid_0's binary_logloss: 0.147378	valid_1's auc: 0.818842	valid_1's binary_logloss: 0.151989
    [4]	valid_0's auc: 0.850477	valid_0's binary_logloss: 0.144112	valid_1's auc: 0.824908	valid_1's binary_logloss: 0.149225
    [5]	valid_0's auc: 0.853808	valid_0's binary_logloss: 0.141422	valid_1's auc: 0.826744	valid_1's binary_logloss: 0.147175
    [6]	valid_0's auc: 0.856663	valid_0's binary_logloss: 0.139222	valid_1's auc: 0.828432	valid_1's binary_logloss: 0.145499
    [7]	valid_0's auc: 0.859664	valid_0's binary_logloss: 0.137318	valid_1's auc: 0.829523	valid_1's binary_logloss: 0.14402
    [8]	valid_0's auc: 0.86186	valid_0's binary_logloss: 0.135665	valid_1's auc: 0.8314	valid_1's binary_logloss: 0.142812
    [9]	valid_0's auc: 0.863104	valid_0's binary_logloss: 0.134357	valid_1's auc: 0.831976	valid_1's binary_logloss: 0.141915
    [10]	valid_0's auc: 0.865027	valid_0's binary_logloss: 0.133117	valid_1's auc: 0.83369	valid_1's binary_logloss: 0.141045
    [11]	valid_0's auc: 0.866437	valid_0's binary_logloss: 0.132026	valid_1's auc: 0.834219	valid_1's binary_logloss: 0.140347
    [12]	valid_0's auc: 0.868039	valid_0's binary_logloss: 0.131061	valid_1's auc: 0.834388	valid_1's binary_logloss: 0.139858
    [13]	valid_0's auc: 0.869537	valid_0's binary_logloss: 0.130159	valid_1's auc: 0.835402	valid_1's binary_logloss: 0.139334
    [14]	valid_0's auc: 0.870971	valid_0's binary_logloss: 0.12932	valid_1's auc: 0.835583	valid_1's binary_logloss: 0.138962
    [15]	valid_0's auc: 0.872381	valid_0's binary_logloss: 0.128543	valid_1's auc: 0.836121	valid_1's binary_logloss: 0.138579
    [16]	valid_0's auc: 0.873871	valid_0's binary_logloss: 0.127833	valid_1's auc: 0.836592	valid_1's binary_logloss: 0.13832
    [17]	valid_0's auc: 0.87491	valid_0's binary_logloss: 0.127212	valid_1's auc: 0.836031	valid_1's binary_logloss: 0.138138
    [18]	valid_0's auc: 0.875677	valid_0's binary_logloss: 0.126616	valid_1's auc: 0.835779	valid_1's binary_logloss: 0.137944
    [19]	valid_0's auc: 0.876894	valid_0's binary_logloss: 0.126018	valid_1's auc: 0.836666	valid_1's binary_logloss: 0.137657
    [20]	valid_0's auc: 0.877834	valid_0's binary_logloss: 0.125465	valid_1's auc: 0.836513	valid_1's binary_logloss: 0.137603
    [21]	valid_0's auc: 0.878861	valid_0's binary_logloss: 0.124918	valid_1's auc: 0.836696	valid_1's binary_logloss: 0.137441
    [22]	valid_0's auc: 0.880257	valid_0's binary_logloss: 0.124391	valid_1's auc: 0.836449	valid_1's binary_logloss: 0.137422
    [23]	valid_0's auc: 0.8818	valid_0's binary_logloss: 0.123782	valid_1's auc: 0.836709	valid_1's binary_logloss: 0.137292
    [24]	valid_0's auc: 0.882923	valid_0's binary_logloss: 0.123257	valid_1's auc: 0.836765	valid_1's binary_logloss: 0.137199
    [25]	valid_0's auc: 0.883924	valid_0's binary_logloss: 0.122788	valid_1's auc: 0.836654	valid_1's binary_logloss: 0.137164
    [26]	valid_0's auc: 0.885046	valid_0's binary_logloss: 0.12229	valid_1's auc: 0.836564	valid_1's binary_logloss: 0.137112
    [27]	valid_0's auc: 0.886069	valid_0's binary_logloss: 0.121841	valid_1's auc: 0.836535	valid_1's binary_logloss: 0.137076
    [28]	valid_0's auc: 0.886927	valid_0's binary_logloss: 0.121432	valid_1's auc: 0.836263	valid_1's binary_logloss: 0.137134
    [29]	valid_0's auc: 0.887853	valid_0's binary_logloss: 0.121034	valid_1's auc: 0.836202	valid_1's binary_logloss: 0.137072
    [30]	valid_0's auc: 0.888728	valid_0's binary_logloss: 0.120631	valid_1's auc: 0.836619	valid_1's binary_logloss: 0.136982
    [31]	valid_0's auc: 0.889701	valid_0's binary_logloss: 0.120172	valid_1's auc: 0.836517	valid_1's binary_logloss: 0.136934
    [32]	valid_0's auc: 0.890993	valid_0's binary_logloss: 0.119756	valid_1's auc: 0.836361	valid_1's binary_logloss: 0.136931
    [33]	valid_0's auc: 0.892053	valid_0's binary_logloss: 0.119407	valid_1's auc: 0.836583	valid_1's binary_logloss: 0.136846
    [34]	valid_0's auc: 0.892821	valid_0's binary_logloss: 0.119049	valid_1's auc: 0.836327	valid_1's binary_logloss: 0.136961
    [35]	valid_0's auc: 0.893663	valid_0's binary_logloss: 0.118651	valid_1's auc: 0.836579	valid_1's binary_logloss: 0.136912
    [36]	valid_0's auc: 0.894523	valid_0's binary_logloss: 0.11827	valid_1's auc: 0.836209	valid_1's binary_logloss: 0.136964
    [37]	valid_0's auc: 0.895475	valid_0's binary_logloss: 0.117947	valid_1's auc: 0.835968	valid_1's binary_logloss: 0.13697
    [38]	valid_0's auc: 0.895896	valid_0's binary_logloss: 0.117661	valid_1's auc: 0.836021	valid_1's binary_logloss: 0.136984
    [39]	valid_0's auc: 0.896908	valid_0's binary_logloss: 0.117297	valid_1's auc: 0.835969	valid_1's binary_logloss: 0.137061
    [40]	valid_0's auc: 0.897428	valid_0's binary_logloss: 0.116993	valid_1's auc: 0.835781	valid_1's binary_logloss: 0.137135
    [41]	valid_0's auc: 0.89802	valid_0's binary_logloss: 0.116616	valid_1's auc: 0.834706	valid_1's binary_logloss: 0.137377
    [42]	valid_0's auc: 0.898553	valid_0's binary_logloss: 0.116307	valid_1's auc: 0.834495	valid_1's binary_logloss: 0.137447
    [43]	valid_0's auc: 0.898979	valid_0's binary_logloss: 0.11602	valid_1's auc: 0.834078	valid_1's binary_logloss: 0.137573
    [44]	valid_0's auc: 0.899439	valid_0's binary_logloss: 0.115752	valid_1's auc: 0.833605	valid_1's binary_logloss: 0.137706
    [45]	valid_0's auc: 0.899915	valid_0's binary_logloss: 0.115497	valid_1's auc: 0.833114	valid_1's binary_logloss: 0.137835
    [46]	valid_0's auc: 0.90021	valid_0's binary_logloss: 0.115245	valid_1's auc: 0.833104	valid_1's binary_logloss: 0.137879
    [47]	valid_0's auc: 0.90064	valid_0's binary_logloss: 0.114971	valid_1's auc: 0.83304	valid_1's binary_logloss: 0.137916
    [48]	valid_0's auc: 0.901303	valid_0's binary_logloss: 0.114665	valid_1's auc: 0.833281	valid_1's binary_logloss: 0.137925
    [49]	valid_0's auc: 0.901644	valid_0's binary_logloss: 0.114372	valid_1's auc: 0.83306	valid_1's binary_logloss: 0.138017
    [50]	valid_0's auc: 0.902036	valid_0's binary_logloss: 0.114104	valid_1's auc: 0.832675	valid_1's binary_logloss: 0.13811
    [51]	valid_0's auc: 0.902469	valid_0's binary_logloss: 0.113883	valid_1's auc: 0.832339	valid_1's binary_logloss: 0.138195
    [52]	valid_0's auc: 0.902667	valid_0's binary_logloss: 0.113693	valid_1's auc: 0.831957	valid_1's binary_logloss: 0.138298
    [53]	valid_0's auc: 0.902822	valid_0's binary_logloss: 0.113485	valid_1's auc: 0.831693	valid_1's binary_logloss: 0.138383
    [54]	valid_0's auc: 0.902977	valid_0's binary_logloss: 0.113301	valid_1's auc: 0.830964	valid_1's binary_logloss: 0.138566
    Early stopping, best iteration is:
    [24]	valid_0's auc: 0.882923	valid_0's binary_logloss: 0.123257	valid_1's auc: 0.836765	valid_1's binary_logloss: 0.137199
    [1]	valid_0's auc: 0.832048	valid_0's binary_logloss: 0.157225	valid_1's auc: 0.811916	valid_1's binary_logloss: 0.159885
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.843711	valid_0's binary_logloss: 0.151535	valid_1's auc: 0.821654	valid_1's binary_logloss: 0.154778
    [3]	valid_0's auc: 0.846336	valid_0's binary_logloss: 0.14743	valid_1's auc: 0.823358	valid_1's binary_logloss: 0.151371
    [4]	valid_0's auc: 0.849091	valid_0's binary_logloss: 0.144223	valid_1's auc: 0.826879	valid_1's binary_logloss: 0.148584
    [5]	valid_0's auc: 0.851041	valid_0's binary_logloss: 0.141653	valid_1's auc: 0.825653	valid_1's binary_logloss: 0.14656
    [6]	valid_0's auc: 0.853453	valid_0's binary_logloss: 0.139394	valid_1's auc: 0.828113	valid_1's binary_logloss: 0.144822
    [7]	valid_0's auc: 0.857868	valid_0's binary_logloss: 0.137439	valid_1's auc: 0.828786	valid_1's binary_logloss: 0.143565
    [8]	valid_0's auc: 0.860559	valid_0's binary_logloss: 0.135866	valid_1's auc: 0.829347	valid_1's binary_logloss: 0.142504
    [9]	valid_0's auc: 0.861868	valid_0's binary_logloss: 0.134472	valid_1's auc: 0.829546	valid_1's binary_logloss: 0.141596
    [10]	valid_0's auc: 0.862998	valid_0's binary_logloss: 0.133271	valid_1's auc: 0.828894	valid_1's binary_logloss: 0.140928
    [11]	valid_0's auc: 0.864761	valid_0's binary_logloss: 0.132122	valid_1's auc: 0.828717	valid_1's binary_logloss: 0.140351
    [12]	valid_0's auc: 0.865861	valid_0's binary_logloss: 0.131127	valid_1's auc: 0.827893	valid_1's binary_logloss: 0.139897
    [13]	valid_0's auc: 0.867377	valid_0's binary_logloss: 0.130201	valid_1's auc: 0.828411	valid_1's binary_logloss: 0.139452
    [14]	valid_0's auc: 0.868645	valid_0's binary_logloss: 0.129388	valid_1's auc: 0.828859	valid_1's binary_logloss: 0.139104
    [15]	valid_0's auc: 0.869811	valid_0's binary_logloss: 0.12864	valid_1's auc: 0.827872	valid_1's binary_logloss: 0.138919
    [16]	valid_0's auc: 0.870761	valid_0's binary_logloss: 0.127902	valid_1's auc: 0.827629	valid_1's binary_logloss: 0.138614
    [17]	valid_0's auc: 0.872392	valid_0's binary_logloss: 0.127247	valid_1's auc: 0.828475	valid_1's binary_logloss: 0.138424
    [18]	valid_0's auc: 0.873984	valid_0's binary_logloss: 0.126643	valid_1's auc: 0.828038	valid_1's binary_logloss: 0.13836
    [19]	valid_0's auc: 0.875277	valid_0's binary_logloss: 0.126018	valid_1's auc: 0.828307	valid_1's binary_logloss: 0.138229
    [20]	valid_0's auc: 0.876767	valid_0's binary_logloss: 0.125429	valid_1's auc: 0.828398	valid_1's binary_logloss: 0.138132
    [21]	valid_0's auc: 0.878127	valid_0's binary_logloss: 0.124871	valid_1's auc: 0.827975	valid_1's binary_logloss: 0.138032
    [22]	valid_0's auc: 0.879333	valid_0's binary_logloss: 0.12434	valid_1's auc: 0.828026	valid_1's binary_logloss: 0.137957
    [23]	valid_0's auc: 0.880379	valid_0's binary_logloss: 0.123879	valid_1's auc: 0.827567	valid_1's binary_logloss: 0.138018
    [24]	valid_0's auc: 0.881429	valid_0's binary_logloss: 0.123358	valid_1's auc: 0.827472	valid_1's binary_logloss: 0.137975
    [25]	valid_0's auc: 0.882335	valid_0's binary_logloss: 0.122902	valid_1's auc: 0.827456	valid_1's binary_logloss: 0.137997
    [26]	valid_0's auc: 0.883467	valid_0's binary_logloss: 0.122419	valid_1's auc: 0.827539	valid_1's binary_logloss: 0.137977
    [27]	valid_0's auc: 0.884573	valid_0's binary_logloss: 0.121954	valid_1's auc: 0.827608	valid_1's binary_logloss: 0.137962
    [28]	valid_0's auc: 0.885354	valid_0's binary_logloss: 0.12151	valid_1's auc: 0.827828	valid_1's binary_logloss: 0.137979
    [29]	valid_0's auc: 0.88633	valid_0's binary_logloss: 0.1211	valid_1's auc: 0.827743	valid_1's binary_logloss: 0.137916
    [30]	valid_0's auc: 0.887129	valid_0's binary_logloss: 0.120718	valid_1's auc: 0.82774	valid_1's binary_logloss: 0.13796
    [31]	valid_0's auc: 0.888235	valid_0's binary_logloss: 0.120283	valid_1's auc: 0.827163	valid_1's binary_logloss: 0.137963
    [32]	valid_0's auc: 0.889179	valid_0's binary_logloss: 0.119986	valid_1's auc: 0.826867	valid_1's binary_logloss: 0.13798
    [33]	valid_0's auc: 0.889826	valid_0's binary_logloss: 0.119626	valid_1's auc: 0.826594	valid_1's binary_logloss: 0.13808
    [34]	valid_0's auc: 0.890813	valid_0's binary_logloss: 0.119231	valid_1's auc: 0.826475	valid_1's binary_logloss: 0.13814
    [35]	valid_0's auc: 0.891526	valid_0's binary_logloss: 0.118839	valid_1's auc: 0.826365	valid_1's binary_logloss: 0.138135
    [36]	valid_0's auc: 0.892313	valid_0's binary_logloss: 0.1185	valid_1's auc: 0.825614	valid_1's binary_logloss: 0.138235
    [37]	valid_0's auc: 0.892917	valid_0's binary_logloss: 0.118143	valid_1's auc: 0.825345	valid_1's binary_logloss: 0.138329
    [38]	valid_0's auc: 0.89337	valid_0's binary_logloss: 0.117833	valid_1's auc: 0.824926	valid_1's binary_logloss: 0.138429
    [39]	valid_0's auc: 0.893935	valid_0's binary_logloss: 0.117548	valid_1's auc: 0.824655	valid_1's binary_logloss: 0.138523
    Early stopping, best iteration is:
    [9]	valid_0's auc: 0.861868	valid_0's binary_logloss: 0.134472	valid_1's auc: 0.829546	valid_1's binary_logloss: 0.141596
    [1]	valid_0's auc: 0.829442	valid_0's binary_logloss: 0.156848	valid_1's auc: 0.801853	valid_1's binary_logloss: 0.159917
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.835044	valid_0's binary_logloss: 0.151129	valid_1's auc: 0.810636	valid_1's binary_logloss: 0.155191
    [3]	valid_0's auc: 0.84106	valid_0's binary_logloss: 0.146986	valid_1's auc: 0.81428	valid_1's binary_logloss: 0.151786
    [4]	valid_0's auc: 0.849104	valid_0's binary_logloss: 0.143746	valid_1's auc: 0.822369	valid_1's binary_logloss: 0.149174
    [5]	valid_0's auc: 0.852054	valid_0's binary_logloss: 0.14111	valid_1's auc: 0.825422	valid_1's binary_logloss: 0.147108
    [6]	valid_0's auc: 0.854219	valid_0's binary_logloss: 0.138941	valid_1's auc: 0.828128	valid_1's binary_logloss: 0.145314
    [7]	valid_0's auc: 0.857758	valid_0's binary_logloss: 0.136956	valid_1's auc: 0.828178	valid_1's binary_logloss: 0.144053
    [8]	valid_0's auc: 0.859433	valid_0's binary_logloss: 0.135341	valid_1's auc: 0.828907	valid_1's binary_logloss: 0.142933
    [9]	valid_0's auc: 0.860562	valid_0's binary_logloss: 0.133965	valid_1's auc: 0.830266	valid_1's binary_logloss: 0.141913
    [10]	valid_0's auc: 0.862253	valid_0's binary_logloss: 0.132732	valid_1's auc: 0.830436	valid_1's binary_logloss: 0.141159
    [11]	valid_0's auc: 0.863863	valid_0's binary_logloss: 0.131616	valid_1's auc: 0.830888	valid_1's binary_logloss: 0.140465
    [12]	valid_0's auc: 0.86535	valid_0's binary_logloss: 0.130642	valid_1's auc: 0.828473	valid_1's binary_logloss: 0.140089
    [13]	valid_0's auc: 0.866315	valid_0's binary_logloss: 0.129707	valid_1's auc: 0.828442	valid_1's binary_logloss: 0.139673
    [14]	valid_0's auc: 0.868091	valid_0's binary_logloss: 0.128809	valid_1's auc: 0.828803	valid_1's binary_logloss: 0.13925
    [15]	valid_0's auc: 0.869154	valid_0's binary_logloss: 0.128055	valid_1's auc: 0.829147	valid_1's binary_logloss: 0.138981
    [16]	valid_0's auc: 0.870086	valid_0's binary_logloss: 0.127376	valid_1's auc: 0.829188	valid_1's binary_logloss: 0.13878
    [17]	valid_0's auc: 0.870857	valid_0's binary_logloss: 0.126705	valid_1's auc: 0.829277	valid_1's binary_logloss: 0.138553
    [18]	valid_0's auc: 0.87202	valid_0's binary_logloss: 0.126023	valid_1's auc: 0.829614	valid_1's binary_logloss: 0.138375
    [19]	valid_0's auc: 0.873171	valid_0's binary_logloss: 0.125459	valid_1's auc: 0.830335	valid_1's binary_logloss: 0.13824
    [20]	valid_0's auc: 0.874593	valid_0's binary_logloss: 0.124876	valid_1's auc: 0.831752	valid_1's binary_logloss: 0.13798
    [21]	valid_0's auc: 0.875741	valid_0's binary_logloss: 0.124281	valid_1's auc: 0.832761	valid_1's binary_logloss: 0.137687
    [22]	valid_0's auc: 0.877448	valid_0's binary_logloss: 0.123659	valid_1's auc: 0.833679	valid_1's binary_logloss: 0.137492
    [23]	valid_0's auc: 0.878493	valid_0's binary_logloss: 0.123115	valid_1's auc: 0.833912	valid_1's binary_logloss: 0.137434
    [24]	valid_0's auc: 0.879193	valid_0's binary_logloss: 0.122682	valid_1's auc: 0.833363	valid_1's binary_logloss: 0.137364
    [25]	valid_0's auc: 0.880227	valid_0's binary_logloss: 0.122229	valid_1's auc: 0.833173	valid_1's binary_logloss: 0.137358
    [26]	valid_0's auc: 0.881585	valid_0's binary_logloss: 0.121752	valid_1's auc: 0.832904	valid_1's binary_logloss: 0.13741
    [27]	valid_0's auc: 0.882473	valid_0's binary_logloss: 0.1213	valid_1's auc: 0.832408	valid_1's binary_logloss: 0.137589
    [28]	valid_0's auc: 0.883437	valid_0's binary_logloss: 0.120929	valid_1's auc: 0.832986	valid_1's binary_logloss: 0.137512
    [29]	valid_0's auc: 0.884502	valid_0's binary_logloss: 0.120484	valid_1's auc: 0.833732	valid_1's binary_logloss: 0.137414
    [30]	valid_0's auc: 0.88539	valid_0's binary_logloss: 0.120132	valid_1's auc: 0.83355	valid_1's binary_logloss: 0.137429
    [31]	valid_0's auc: 0.886163	valid_0's binary_logloss: 0.11975	valid_1's auc: 0.833288	valid_1's binary_logloss: 0.137459
    [32]	valid_0's auc: 0.886867	valid_0's binary_logloss: 0.119396	valid_1's auc: 0.833051	valid_1's binary_logloss: 0.137503
    [33]	valid_0's auc: 0.887733	valid_0's binary_logloss: 0.119013	valid_1's auc: 0.832946	valid_1's binary_logloss: 0.137467
    [34]	valid_0's auc: 0.888681	valid_0's binary_logloss: 0.118651	valid_1's auc: 0.833519	valid_1's binary_logloss: 0.137391
    [35]	valid_0's auc: 0.889275	valid_0's binary_logloss: 0.118341	valid_1's auc: 0.833734	valid_1's binary_logloss: 0.137371
    [36]	valid_0's auc: 0.890084	valid_0's binary_logloss: 0.117965	valid_1's auc: 0.833231	valid_1's binary_logloss: 0.137505
    [37]	valid_0's auc: 0.890726	valid_0's binary_logloss: 0.11765	valid_1's auc: 0.833136	valid_1's binary_logloss: 0.13761
    [38]	valid_0's auc: 0.891158	valid_0's binary_logloss: 0.11734	valid_1's auc: 0.832938	valid_1's binary_logloss: 0.137668
    [39]	valid_0's auc: 0.89196	valid_0's binary_logloss: 0.117006	valid_1's auc: 0.832307	valid_1's binary_logloss: 0.13782
    [40]	valid_0's auc: 0.892449	valid_0's binary_logloss: 0.116703	valid_1's auc: 0.832354	valid_1's binary_logloss: 0.137858
    [41]	valid_0's auc: 0.892978	valid_0's binary_logloss: 0.116414	valid_1's auc: 0.831731	valid_1's binary_logloss: 0.137998
    [42]	valid_0's auc: 0.893542	valid_0's binary_logloss: 0.116071	valid_1's auc: 0.830942	valid_1's binary_logloss: 0.138189
    [43]	valid_0's auc: 0.894078	valid_0's binary_logloss: 0.115805	valid_1's auc: 0.830997	valid_1's binary_logloss: 0.138182
    [44]	valid_0's auc: 0.894576	valid_0's binary_logloss: 0.115516	valid_1's auc: 0.830968	valid_1's binary_logloss: 0.138257
    [45]	valid_0's auc: 0.895099	valid_0's binary_logloss: 0.115236	valid_1's auc: 0.830506	valid_1's binary_logloss: 0.13839
    [46]	valid_0's auc: 0.895608	valid_0's binary_logloss: 0.114971	valid_1's auc: 0.829892	valid_1's binary_logloss: 0.138556
    [47]	valid_0's auc: 0.895999	valid_0's binary_logloss: 0.1147	valid_1's auc: 0.829591	valid_1's binary_logloss: 0.138649
    [48]	valid_0's auc: 0.89641	valid_0's binary_logloss: 0.11442	valid_1's auc: 0.829768	valid_1's binary_logloss: 0.138698
    [49]	valid_0's auc: 0.896742	valid_0's binary_logloss: 0.114182	valid_1's auc: 0.830072	valid_1's binary_logloss: 0.138692
    [50]	valid_0's auc: 0.896929	valid_0's binary_logloss: 0.113989	valid_1's auc: 0.829948	valid_1's binary_logloss: 0.138769
    [51]	valid_0's auc: 0.897717	valid_0's binary_logloss: 0.113718	valid_1's auc: 0.830297	valid_1's binary_logloss: 0.138742
    [52]	valid_0's auc: 0.898093	valid_0's binary_logloss: 0.113472	valid_1's auc: 0.830332	valid_1's binary_logloss: 0.138776
    [53]	valid_0's auc: 0.898387	valid_0's binary_logloss: 0.113271	valid_1's auc: 0.830278	valid_1's binary_logloss: 0.138833
    Early stopping, best iteration is:
    [23]	valid_0's auc: 0.878493	valid_0's binary_logloss: 0.123115	valid_1's auc: 0.833912	valid_1's binary_logloss: 0.137434
    [1]	valid_0's auc: 0.834008	valid_0's binary_logloss: 0.156932	valid_1's auc: 0.806689	valid_1's binary_logloss: 0.159986
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.841271	valid_0's binary_logloss: 0.151528	valid_1's auc: 0.816568	valid_1's binary_logloss: 0.155337
    [3]	valid_0's auc: 0.843827	valid_0's binary_logloss: 0.147378	valid_1's auc: 0.818842	valid_1's binary_logloss: 0.151989
    [4]	valid_0's auc: 0.850477	valid_0's binary_logloss: 0.144112	valid_1's auc: 0.824908	valid_1's binary_logloss: 0.149225
    [5]	valid_0's auc: 0.853808	valid_0's binary_logloss: 0.141422	valid_1's auc: 0.826744	valid_1's binary_logloss: 0.147175
    [6]	valid_0's auc: 0.856663	valid_0's binary_logloss: 0.139222	valid_1's auc: 0.828432	valid_1's binary_logloss: 0.145499
    [7]	valid_0's auc: 0.859664	valid_0's binary_logloss: 0.137318	valid_1's auc: 0.829523	valid_1's binary_logloss: 0.14402
    [8]	valid_0's auc: 0.86186	valid_0's binary_logloss: 0.135665	valid_1's auc: 0.8314	valid_1's binary_logloss: 0.142812
    [9]	valid_0's auc: 0.863104	valid_0's binary_logloss: 0.134357	valid_1's auc: 0.831976	valid_1's binary_logloss: 0.141915
    [10]	valid_0's auc: 0.865027	valid_0's binary_logloss: 0.133117	valid_1's auc: 0.83369	valid_1's binary_logloss: 0.141045
    [11]	valid_0's auc: 0.866437	valid_0's binary_logloss: 0.132026	valid_1's auc: 0.834219	valid_1's binary_logloss: 0.140347
    [12]	valid_0's auc: 0.868039	valid_0's binary_logloss: 0.131061	valid_1's auc: 0.834388	valid_1's binary_logloss: 0.139858
    [13]	valid_0's auc: 0.869537	valid_0's binary_logloss: 0.130159	valid_1's auc: 0.835402	valid_1's binary_logloss: 0.139334
    [14]	valid_0's auc: 0.870971	valid_0's binary_logloss: 0.12932	valid_1's auc: 0.835583	valid_1's binary_logloss: 0.138962
    [15]	valid_0's auc: 0.872381	valid_0's binary_logloss: 0.128543	valid_1's auc: 0.836121	valid_1's binary_logloss: 0.138579
    [16]	valid_0's auc: 0.873871	valid_0's binary_logloss: 0.127833	valid_1's auc: 0.836592	valid_1's binary_logloss: 0.13832
    [17]	valid_0's auc: 0.87491	valid_0's binary_logloss: 0.127212	valid_1's auc: 0.836031	valid_1's binary_logloss: 0.138138
    [18]	valid_0's auc: 0.875677	valid_0's binary_logloss: 0.126616	valid_1's auc: 0.835779	valid_1's binary_logloss: 0.137944
    [19]	valid_0's auc: 0.876894	valid_0's binary_logloss: 0.126018	valid_1's auc: 0.836666	valid_1's binary_logloss: 0.137657
    [20]	valid_0's auc: 0.877834	valid_0's binary_logloss: 0.125465	valid_1's auc: 0.836513	valid_1's binary_logloss: 0.137603
    [21]	valid_0's auc: 0.878861	valid_0's binary_logloss: 0.124918	valid_1's auc: 0.836696	valid_1's binary_logloss: 0.137441
    [22]	valid_0's auc: 0.880257	valid_0's binary_logloss: 0.124391	valid_1's auc: 0.836449	valid_1's binary_logloss: 0.137422
    [23]	valid_0's auc: 0.8818	valid_0's binary_logloss: 0.123782	valid_1's auc: 0.836709	valid_1's binary_logloss: 0.137292
    [24]	valid_0's auc: 0.882923	valid_0's binary_logloss: 0.123257	valid_1's auc: 0.836765	valid_1's binary_logloss: 0.137199
    [25]	valid_0's auc: 0.883924	valid_0's binary_logloss: 0.122788	valid_1's auc: 0.836654	valid_1's binary_logloss: 0.137164
    [26]	valid_0's auc: 0.885046	valid_0's binary_logloss: 0.12229	valid_1's auc: 0.836564	valid_1's binary_logloss: 0.137112
    [27]	valid_0's auc: 0.886069	valid_0's binary_logloss: 0.121841	valid_1's auc: 0.836535	valid_1's binary_logloss: 0.137076
    [28]	valid_0's auc: 0.886927	valid_0's binary_logloss: 0.121432	valid_1's auc: 0.836263	valid_1's binary_logloss: 0.137134
    [29]	valid_0's auc: 0.887853	valid_0's binary_logloss: 0.121034	valid_1's auc: 0.836202	valid_1's binary_logloss: 0.137072
    [30]	valid_0's auc: 0.888728	valid_0's binary_logloss: 0.120631	valid_1's auc: 0.836619	valid_1's binary_logloss: 0.136982
    [31]	valid_0's auc: 0.889701	valid_0's binary_logloss: 0.120172	valid_1's auc: 0.836517	valid_1's binary_logloss: 0.136934
    [32]	valid_0's auc: 0.890993	valid_0's binary_logloss: 0.119756	valid_1's auc: 0.836361	valid_1's binary_logloss: 0.136931
    [33]	valid_0's auc: 0.892053	valid_0's binary_logloss: 0.119407	valid_1's auc: 0.836583	valid_1's binary_logloss: 0.136846
    [34]	valid_0's auc: 0.892821	valid_0's binary_logloss: 0.119049	valid_1's auc: 0.836327	valid_1's binary_logloss: 0.136961
    [35]	valid_0's auc: 0.893663	valid_0's binary_logloss: 0.118651	valid_1's auc: 0.836579	valid_1's binary_logloss: 0.136912
    [36]	valid_0's auc: 0.894523	valid_0's binary_logloss: 0.11827	valid_1's auc: 0.836209	valid_1's binary_logloss: 0.136964
    [37]	valid_0's auc: 0.895475	valid_0's binary_logloss: 0.117947	valid_1's auc: 0.835968	valid_1's binary_logloss: 0.13697
    [38]	valid_0's auc: 0.895896	valid_0's binary_logloss: 0.117661	valid_1's auc: 0.836021	valid_1's binary_logloss: 0.136984
    [39]	valid_0's auc: 0.896908	valid_0's binary_logloss: 0.117297	valid_1's auc: 0.835969	valid_1's binary_logloss: 0.137061
    [40]	valid_0's auc: 0.897428	valid_0's binary_logloss: 0.116993	valid_1's auc: 0.835781	valid_1's binary_logloss: 0.137135
    [41]	valid_0's auc: 0.89802	valid_0's binary_logloss: 0.116616	valid_1's auc: 0.834706	valid_1's binary_logloss: 0.137377
    [42]	valid_0's auc: 0.898553	valid_0's binary_logloss: 0.116307	valid_1's auc: 0.834495	valid_1's binary_logloss: 0.137447
    [43]	valid_0's auc: 0.898979	valid_0's binary_logloss: 0.11602	valid_1's auc: 0.834078	valid_1's binary_logloss: 0.137573
    [44]	valid_0's auc: 0.899439	valid_0's binary_logloss: 0.115752	valid_1's auc: 0.833605	valid_1's binary_logloss: 0.137706
    [45]	valid_0's auc: 0.899915	valid_0's binary_logloss: 0.115497	valid_1's auc: 0.833114	valid_1's binary_logloss: 0.137835
    [46]	valid_0's auc: 0.90021	valid_0's binary_logloss: 0.115245	valid_1's auc: 0.833104	valid_1's binary_logloss: 0.137879
    [47]	valid_0's auc: 0.90064	valid_0's binary_logloss: 0.114971	valid_1's auc: 0.83304	valid_1's binary_logloss: 0.137916
    [48]	valid_0's auc: 0.901303	valid_0's binary_logloss: 0.114665	valid_1's auc: 0.833281	valid_1's binary_logloss: 0.137925
    [49]	valid_0's auc: 0.901644	valid_0's binary_logloss: 0.114372	valid_1's auc: 0.83306	valid_1's binary_logloss: 0.138017
    [50]	valid_0's auc: 0.902036	valid_0's binary_logloss: 0.114104	valid_1's auc: 0.832675	valid_1's binary_logloss: 0.13811
    [51]	valid_0's auc: 0.902469	valid_0's binary_logloss: 0.113883	valid_1's auc: 0.832339	valid_1's binary_logloss: 0.138195
    [52]	valid_0's auc: 0.902667	valid_0's binary_logloss: 0.113693	valid_1's auc: 0.831957	valid_1's binary_logloss: 0.138298
    [53]	valid_0's auc: 0.902822	valid_0's binary_logloss: 0.113485	valid_1's auc: 0.831693	valid_1's binary_logloss: 0.138383
    [54]	valid_0's auc: 0.902977	valid_0's binary_logloss: 0.113301	valid_1's auc: 0.830964	valid_1's binary_logloss: 0.138566
    Early stopping, best iteration is:
    [24]	valid_0's auc: 0.882923	valid_0's binary_logloss: 0.123257	valid_1's auc: 0.836765	valid_1's binary_logloss: 0.137199
    [1]	valid_0's auc: 0.832048	valid_0's binary_logloss: 0.157225	valid_1's auc: 0.811916	valid_1's binary_logloss: 0.159885
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.843711	valid_0's binary_logloss: 0.151535	valid_1's auc: 0.821654	valid_1's binary_logloss: 0.154778
    [3]	valid_0's auc: 0.846336	valid_0's binary_logloss: 0.14743	valid_1's auc: 0.823358	valid_1's binary_logloss: 0.151371
    [4]	valid_0's auc: 0.849091	valid_0's binary_logloss: 0.144223	valid_1's auc: 0.826879	valid_1's binary_logloss: 0.148584
    [5]	valid_0's auc: 0.851041	valid_0's binary_logloss: 0.141653	valid_1's auc: 0.825653	valid_1's binary_logloss: 0.14656
    [6]	valid_0's auc: 0.853453	valid_0's binary_logloss: 0.139394	valid_1's auc: 0.828113	valid_1's binary_logloss: 0.144822
    [7]	valid_0's auc: 0.857868	valid_0's binary_logloss: 0.137439	valid_1's auc: 0.828786	valid_1's binary_logloss: 0.143565
    [8]	valid_0's auc: 0.860559	valid_0's binary_logloss: 0.135866	valid_1's auc: 0.829347	valid_1's binary_logloss: 0.142504
    [9]	valid_0's auc: 0.861868	valid_0's binary_logloss: 0.134472	valid_1's auc: 0.829546	valid_1's binary_logloss: 0.141596
    [10]	valid_0's auc: 0.862998	valid_0's binary_logloss: 0.133271	valid_1's auc: 0.828894	valid_1's binary_logloss: 0.140928
    [11]	valid_0's auc: 0.864761	valid_0's binary_logloss: 0.132122	valid_1's auc: 0.828717	valid_1's binary_logloss: 0.140351
    [12]	valid_0's auc: 0.865861	valid_0's binary_logloss: 0.131127	valid_1's auc: 0.827893	valid_1's binary_logloss: 0.139897
    [13]	valid_0's auc: 0.867377	valid_0's binary_logloss: 0.130201	valid_1's auc: 0.828411	valid_1's binary_logloss: 0.139452
    [14]	valid_0's auc: 0.868645	valid_0's binary_logloss: 0.129388	valid_1's auc: 0.828859	valid_1's binary_logloss: 0.139104
    [15]	valid_0's auc: 0.869811	valid_0's binary_logloss: 0.12864	valid_1's auc: 0.827872	valid_1's binary_logloss: 0.138919
    [16]	valid_0's auc: 0.870761	valid_0's binary_logloss: 0.127902	valid_1's auc: 0.827629	valid_1's binary_logloss: 0.138614
    [17]	valid_0's auc: 0.872392	valid_0's binary_logloss: 0.127247	valid_1's auc: 0.828475	valid_1's binary_logloss: 0.138424
    [18]	valid_0's auc: 0.873984	valid_0's binary_logloss: 0.126643	valid_1's auc: 0.828038	valid_1's binary_logloss: 0.13836
    [19]	valid_0's auc: 0.875277	valid_0's binary_logloss: 0.126018	valid_1's auc: 0.828307	valid_1's binary_logloss: 0.138229
    [20]	valid_0's auc: 0.876767	valid_0's binary_logloss: 0.125429	valid_1's auc: 0.828398	valid_1's binary_logloss: 0.138132
    [21]	valid_0's auc: 0.878127	valid_0's binary_logloss: 0.124871	valid_1's auc: 0.827975	valid_1's binary_logloss: 0.138032
    [22]	valid_0's auc: 0.879333	valid_0's binary_logloss: 0.12434	valid_1's auc: 0.828026	valid_1's binary_logloss: 0.137957
    [23]	valid_0's auc: 0.880379	valid_0's binary_logloss: 0.123879	valid_1's auc: 0.827567	valid_1's binary_logloss: 0.138018
    [24]	valid_0's auc: 0.881429	valid_0's binary_logloss: 0.123358	valid_1's auc: 0.827472	valid_1's binary_logloss: 0.137975
    [25]	valid_0's auc: 0.882335	valid_0's binary_logloss: 0.122902	valid_1's auc: 0.827456	valid_1's binary_logloss: 0.137997
    [26]	valid_0's auc: 0.883467	valid_0's binary_logloss: 0.122419	valid_1's auc: 0.827539	valid_1's binary_logloss: 0.137977
    [27]	valid_0's auc: 0.884573	valid_0's binary_logloss: 0.121954	valid_1's auc: 0.827608	valid_1's binary_logloss: 0.137962
    [28]	valid_0's auc: 0.885354	valid_0's binary_logloss: 0.12151	valid_1's auc: 0.827828	valid_1's binary_logloss: 0.137979
    [29]	valid_0's auc: 0.88633	valid_0's binary_logloss: 0.1211	valid_1's auc: 0.827743	valid_1's binary_logloss: 0.137916
    [30]	valid_0's auc: 0.887129	valid_0's binary_logloss: 0.120718	valid_1's auc: 0.82774	valid_1's binary_logloss: 0.13796
    [31]	valid_0's auc: 0.888235	valid_0's binary_logloss: 0.120283	valid_1's auc: 0.827163	valid_1's binary_logloss: 0.137963
    [32]	valid_0's auc: 0.889179	valid_0's binary_logloss: 0.119986	valid_1's auc: 0.826867	valid_1's binary_logloss: 0.13798
    [33]	valid_0's auc: 0.889826	valid_0's binary_logloss: 0.119626	valid_1's auc: 0.826594	valid_1's binary_logloss: 0.13808
    [34]	valid_0's auc: 0.890813	valid_0's binary_logloss: 0.119231	valid_1's auc: 0.826475	valid_1's binary_logloss: 0.13814
    [35]	valid_0's auc: 0.891526	valid_0's binary_logloss: 0.118839	valid_1's auc: 0.826365	valid_1's binary_logloss: 0.138135
    [36]	valid_0's auc: 0.892313	valid_0's binary_logloss: 0.1185	valid_1's auc: 0.825614	valid_1's binary_logloss: 0.138235
    [37]	valid_0's auc: 0.892917	valid_0's binary_logloss: 0.118143	valid_1's auc: 0.825345	valid_1's binary_logloss: 0.138329
    [38]	valid_0's auc: 0.89337	valid_0's binary_logloss: 0.117833	valid_1's auc: 0.824926	valid_1's binary_logloss: 0.138429
    [39]	valid_0's auc: 0.893935	valid_0's binary_logloss: 0.117548	valid_1's auc: 0.824655	valid_1's binary_logloss: 0.138523
    Early stopping, best iteration is:
    [9]	valid_0's auc: 0.861868	valid_0's binary_logloss: 0.134472	valid_1's auc: 0.829546	valid_1's binary_logloss: 0.141596
    [1]	valid_0's auc: 0.822135	valid_0's binary_logloss: 0.157275	valid_1's auc: 0.80763	valid_1's binary_logloss: 0.159959
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.826629	valid_0's binary_logloss: 0.15186	valid_1's auc: 0.811377	valid_1's binary_logloss: 0.155197
    [3]	valid_0's auc: 0.830813	valid_0's binary_logloss: 0.147949	valid_1's auc: 0.813917	valid_1's binary_logloss: 0.151946
    [4]	valid_0's auc: 0.833387	valid_0's binary_logloss: 0.144809	valid_1's auc: 0.814641	valid_1's binary_logloss: 0.149358
    [5]	valid_0's auc: 0.838116	valid_0's binary_logloss: 0.142332	valid_1's auc: 0.816965	valid_1's binary_logloss: 0.147361
    [6]	valid_0's auc: 0.841055	valid_0's binary_logloss: 0.140267	valid_1's auc: 0.81978	valid_1's binary_logloss: 0.145867
    [7]	valid_0's auc: 0.842021	valid_0's binary_logloss: 0.138495	valid_1's auc: 0.819689	valid_1's binary_logloss: 0.144575
    [8]	valid_0's auc: 0.843828	valid_0's binary_logloss: 0.137029	valid_1's auc: 0.822496	valid_1's binary_logloss: 0.143422
    [9]	valid_0's auc: 0.845522	valid_0's binary_logloss: 0.135783	valid_1's auc: 0.823288	valid_1's binary_logloss: 0.142428
    [10]	valid_0's auc: 0.847654	valid_0's binary_logloss: 0.134654	valid_1's auc: 0.824658	valid_1's binary_logloss: 0.141712
    [11]	valid_0's auc: 0.849302	valid_0's binary_logloss: 0.133679	valid_1's auc: 0.826788	valid_1's binary_logloss: 0.140983
    [12]	valid_0's auc: 0.850949	valid_0's binary_logloss: 0.132827	valid_1's auc: 0.828316	valid_1's binary_logloss: 0.140403
    [13]	valid_0's auc: 0.852852	valid_0's binary_logloss: 0.132062	valid_1's auc: 0.830745	valid_1's binary_logloss: 0.139849
    [14]	valid_0's auc: 0.854299	valid_0's binary_logloss: 0.131363	valid_1's auc: 0.832741	valid_1's binary_logloss: 0.139383
    [15]	valid_0's auc: 0.855376	valid_0's binary_logloss: 0.13071	valid_1's auc: 0.833869	valid_1's binary_logloss: 0.138963
    [16]	valid_0's auc: 0.856265	valid_0's binary_logloss: 0.130166	valid_1's auc: 0.833925	valid_1's binary_logloss: 0.138745
    [17]	valid_0's auc: 0.85735	valid_0's binary_logloss: 0.129615	valid_1's auc: 0.834227	valid_1's binary_logloss: 0.138474
    [18]	valid_0's auc: 0.859279	valid_0's binary_logloss: 0.129153	valid_1's auc: 0.834146	valid_1's binary_logloss: 0.138306
    [19]	valid_0's auc: 0.860099	valid_0's binary_logloss: 0.128684	valid_1's auc: 0.833391	valid_1's binary_logloss: 0.138142
    [20]	valid_0's auc: 0.861239	valid_0's binary_logloss: 0.128272	valid_1's auc: 0.833651	valid_1's binary_logloss: 0.13795
    [21]	valid_0's auc: 0.862153	valid_0's binary_logloss: 0.127883	valid_1's auc: 0.834662	valid_1's binary_logloss: 0.137829
    [22]	valid_0's auc: 0.863466	valid_0's binary_logloss: 0.127439	valid_1's auc: 0.834505	valid_1's binary_logloss: 0.137762
    [23]	valid_0's auc: 0.86379	valid_0's binary_logloss: 0.127135	valid_1's auc: 0.834377	valid_1's binary_logloss: 0.137657
    [24]	valid_0's auc: 0.864548	valid_0's binary_logloss: 0.126788	valid_1's auc: 0.834289	valid_1's binary_logloss: 0.137499
    [25]	valid_0's auc: 0.865681	valid_0's binary_logloss: 0.126451	valid_1's auc: 0.834899	valid_1's binary_logloss: 0.137329
    [26]	valid_0's auc: 0.866659	valid_0's binary_logloss: 0.126121	valid_1's auc: 0.835946	valid_1's binary_logloss: 0.137177
    [27]	valid_0's auc: 0.867505	valid_0's binary_logloss: 0.125853	valid_1's auc: 0.836616	valid_1's binary_logloss: 0.137016
    [28]	valid_0's auc: 0.868089	valid_0's binary_logloss: 0.125584	valid_1's auc: 0.836565	valid_1's binary_logloss: 0.137
    [29]	valid_0's auc: 0.868709	valid_0's binary_logloss: 0.125327	valid_1's auc: 0.837028	valid_1's binary_logloss: 0.136878
    [30]	valid_0's auc: 0.87006	valid_0's binary_logloss: 0.12501	valid_1's auc: 0.837512	valid_1's binary_logloss: 0.136792
    [31]	valid_0's auc: 0.870612	valid_0's binary_logloss: 0.124704	valid_1's auc: 0.837541	valid_1's binary_logloss: 0.136757
    [32]	valid_0's auc: 0.871234	valid_0's binary_logloss: 0.124431	valid_1's auc: 0.837141	valid_1's binary_logloss: 0.136755
    [33]	valid_0's auc: 0.872027	valid_0's binary_logloss: 0.124167	valid_1's auc: 0.837258	valid_1's binary_logloss: 0.136728
    [34]	valid_0's auc: 0.87244	valid_0's binary_logloss: 0.123956	valid_1's auc: 0.836829	valid_1's binary_logloss: 0.136799
    [35]	valid_0's auc: 0.873259	valid_0's binary_logloss: 0.123713	valid_1's auc: 0.836871	valid_1's binary_logloss: 0.136775
    [36]	valid_0's auc: 0.873695	valid_0's binary_logloss: 0.123497	valid_1's auc: 0.836512	valid_1's binary_logloss: 0.136859
    [37]	valid_0's auc: 0.874365	valid_0's binary_logloss: 0.123259	valid_1's auc: 0.836421	valid_1's binary_logloss: 0.136872
    [38]	valid_0's auc: 0.874998	valid_0's binary_logloss: 0.123014	valid_1's auc: 0.83628	valid_1's binary_logloss: 0.136939
    [39]	valid_0's auc: 0.875666	valid_0's binary_logloss: 0.122811	valid_1's auc: 0.836577	valid_1's binary_logloss: 0.13689
    [40]	valid_0's auc: 0.875931	valid_0's binary_logloss: 0.122638	valid_1's auc: 0.836148	valid_1's binary_logloss: 0.13697
    [41]	valid_0's auc: 0.876403	valid_0's binary_logloss: 0.122474	valid_1's auc: 0.836008	valid_1's binary_logloss: 0.137005
    [42]	valid_0's auc: 0.87697	valid_0's binary_logloss: 0.122306	valid_1's auc: 0.835814	valid_1's binary_logloss: 0.137031
    [43]	valid_0's auc: 0.877305	valid_0's binary_logloss: 0.122115	valid_1's auc: 0.835467	valid_1's binary_logloss: 0.137112
    [44]	valid_0's auc: 0.87782	valid_0's binary_logloss: 0.121932	valid_1's auc: 0.835339	valid_1's binary_logloss: 0.137217
    [45]	valid_0's auc: 0.878559	valid_0's binary_logloss: 0.121701	valid_1's auc: 0.83506	valid_1's binary_logloss: 0.137256
    [46]	valid_0's auc: 0.878895	valid_0's binary_logloss: 0.121564	valid_1's auc: 0.834955	valid_1's binary_logloss: 0.137255
    [47]	valid_0's auc: 0.879643	valid_0's binary_logloss: 0.121306	valid_1's auc: 0.834987	valid_1's binary_logloss: 0.137275
    [48]	valid_0's auc: 0.88002	valid_0's binary_logloss: 0.121119	valid_1's auc: 0.834864	valid_1's binary_logloss: 0.137329
    [49]	valid_0's auc: 0.880318	valid_0's binary_logloss: 0.120965	valid_1's auc: 0.834547	valid_1's binary_logloss: 0.137394
    [50]	valid_0's auc: 0.880967	valid_0's binary_logloss: 0.12082	valid_1's auc: 0.834588	valid_1's binary_logloss: 0.137388
    [51]	valid_0's auc: 0.8813	valid_0's binary_logloss: 0.120647	valid_1's auc: 0.834288	valid_1's binary_logloss: 0.137467
    [52]	valid_0's auc: 0.881706	valid_0's binary_logloss: 0.120482	valid_1's auc: 0.834305	valid_1's binary_logloss: 0.137491
    [53]	valid_0's auc: 0.882192	valid_0's binary_logloss: 0.1203	valid_1's auc: 0.834662	valid_1's binary_logloss: 0.137492
    [54]	valid_0's auc: 0.882307	valid_0's binary_logloss: 0.120188	valid_1's auc: 0.834241	valid_1's binary_logloss: 0.13763
    [55]	valid_0's auc: 0.883062	valid_0's binary_logloss: 0.119929	valid_1's auc: 0.834145	valid_1's binary_logloss: 0.137674
    [56]	valid_0's auc: 0.883272	valid_0's binary_logloss: 0.119823	valid_1's auc: 0.833884	valid_1's binary_logloss: 0.137762
    [57]	valid_0's auc: 0.883628	valid_0's binary_logloss: 0.11968	valid_1's auc: 0.833563	valid_1's binary_logloss: 0.13782
    [58]	valid_0's auc: 0.884097	valid_0's binary_logloss: 0.119546	valid_1's auc: 0.833647	valid_1's binary_logloss: 0.137781
    [59]	valid_0's auc: 0.884337	valid_0's binary_logloss: 0.119387	valid_1's auc: 0.833138	valid_1's binary_logloss: 0.137878
    [60]	valid_0's auc: 0.884777	valid_0's binary_logloss: 0.119188	valid_1's auc: 0.832909	valid_1's binary_logloss: 0.13796
    [61]	valid_0's auc: 0.885076	valid_0's binary_logloss: 0.119054	valid_1's auc: 0.833025	valid_1's binary_logloss: 0.137959
    Early stopping, best iteration is:
    [31]	valid_0's auc: 0.870612	valid_0's binary_logloss: 0.124704	valid_1's auc: 0.837541	valid_1's binary_logloss: 0.136757
    [1]	valid_0's auc: 0.825036	valid_0's binary_logloss: 0.157424	valid_1's auc: 0.807618	valid_1's binary_logloss: 0.159921
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.829804	valid_0's binary_logloss: 0.152132	valid_1's auc: 0.815884	valid_1's binary_logloss: 0.155124
    [3]	valid_0's auc: 0.835123	valid_0's binary_logloss: 0.148284	valid_1's auc: 0.81743	valid_1's binary_logloss: 0.151751
    [4]	valid_0's auc: 0.841579	valid_0's binary_logloss: 0.145192	valid_1's auc: 0.823237	valid_1's binary_logloss: 0.149159
    [5]	valid_0's auc: 0.843972	valid_0's binary_logloss: 0.142753	valid_1's auc: 0.82747	valid_1's binary_logloss: 0.147042
    [6]	valid_0's auc: 0.846408	valid_0's binary_logloss: 0.140687	valid_1's auc: 0.830402	valid_1's binary_logloss: 0.145398
    [7]	valid_0's auc: 0.848617	valid_0's binary_logloss: 0.138978	valid_1's auc: 0.830319	valid_1's binary_logloss: 0.144054
    [8]	valid_0's auc: 0.849973	valid_0's binary_logloss: 0.137527	valid_1's auc: 0.831414	valid_1's binary_logloss: 0.142955
    [9]	valid_0's auc: 0.852151	valid_0's binary_logloss: 0.136279	valid_1's auc: 0.834577	valid_1's binary_logloss: 0.141825
    [10]	valid_0's auc: 0.852962	valid_0's binary_logloss: 0.135231	valid_1's auc: 0.835301	valid_1's binary_logloss: 0.140947
    [11]	valid_0's auc: 0.853769	valid_0's binary_logloss: 0.134247	valid_1's auc: 0.836379	valid_1's binary_logloss: 0.140121
    [12]	valid_0's auc: 0.855739	valid_0's binary_logloss: 0.133341	valid_1's auc: 0.83615	valid_1's binary_logloss: 0.139629
    [13]	valid_0's auc: 0.857549	valid_0's binary_logloss: 0.132607	valid_1's auc: 0.836232	valid_1's binary_logloss: 0.139209
    [14]	valid_0's auc: 0.858955	valid_0's binary_logloss: 0.131867	valid_1's auc: 0.836387	valid_1's binary_logloss: 0.138827
    [15]	valid_0's auc: 0.860018	valid_0's binary_logloss: 0.13125	valid_1's auc: 0.837299	valid_1's binary_logloss: 0.138441
    [16]	valid_0's auc: 0.861059	valid_0's binary_logloss: 0.130679	valid_1's auc: 0.838197	valid_1's binary_logloss: 0.138074
    [17]	valid_0's auc: 0.862112	valid_0's binary_logloss: 0.130128	valid_1's auc: 0.838835	valid_1's binary_logloss: 0.137791
    [18]	valid_0's auc: 0.862994	valid_0's binary_logloss: 0.129646	valid_1's auc: 0.838904	valid_1's binary_logloss: 0.137577
    [19]	valid_0's auc: 0.863568	valid_0's binary_logloss: 0.129195	valid_1's auc: 0.839028	valid_1's binary_logloss: 0.137319
    [20]	valid_0's auc: 0.86472	valid_0's binary_logloss: 0.128729	valid_1's auc: 0.83984	valid_1's binary_logloss: 0.137089
    [21]	valid_0's auc: 0.866051	valid_0's binary_logloss: 0.128315	valid_1's auc: 0.840385	valid_1's binary_logloss: 0.136889
    [22]	valid_0's auc: 0.866997	valid_0's binary_logloss: 0.127908	valid_1's auc: 0.840544	valid_1's binary_logloss: 0.136724
    [23]	valid_0's auc: 0.867998	valid_0's binary_logloss: 0.127516	valid_1's auc: 0.840471	valid_1's binary_logloss: 0.136628
    [24]	valid_0's auc: 0.868835	valid_0's binary_logloss: 0.127186	valid_1's auc: 0.84011	valid_1's binary_logloss: 0.136573
    [25]	valid_0's auc: 0.869535	valid_0's binary_logloss: 0.126855	valid_1's auc: 0.83987	valid_1's binary_logloss: 0.136502
    [26]	valid_0's auc: 0.870439	valid_0's binary_logloss: 0.126515	valid_1's auc: 0.840538	valid_1's binary_logloss: 0.136387
    [27]	valid_0's auc: 0.871466	valid_0's binary_logloss: 0.126179	valid_1's auc: 0.840326	valid_1's binary_logloss: 0.136301
    [28]	valid_0's auc: 0.872022	valid_0's binary_logloss: 0.125897	valid_1's auc: 0.840415	valid_1's binary_logloss: 0.136253
    [29]	valid_0's auc: 0.873022	valid_0's binary_logloss: 0.125624	valid_1's auc: 0.840531	valid_1's binary_logloss: 0.136185
    [30]	valid_0's auc: 0.873626	valid_0's binary_logloss: 0.125329	valid_1's auc: 0.840407	valid_1's binary_logloss: 0.136188
    [31]	valid_0's auc: 0.874227	valid_0's binary_logloss: 0.12509	valid_1's auc: 0.841102	valid_1's binary_logloss: 0.136046
    [32]	valid_0's auc: 0.874828	valid_0's binary_logloss: 0.124856	valid_1's auc: 0.841125	valid_1's binary_logloss: 0.135986
    [33]	valid_0's auc: 0.875492	valid_0's binary_logloss: 0.124583	valid_1's auc: 0.841558	valid_1's binary_logloss: 0.135907
    [34]	valid_0's auc: 0.876123	valid_0's binary_logloss: 0.124343	valid_1's auc: 0.841695	valid_1's binary_logloss: 0.135865
    [35]	valid_0's auc: 0.876893	valid_0's binary_logloss: 0.124064	valid_1's auc: 0.841945	valid_1's binary_logloss: 0.135831
    [36]	valid_0's auc: 0.877421	valid_0's binary_logloss: 0.123823	valid_1's auc: 0.841949	valid_1's binary_logloss: 0.135864
    [37]	valid_0's auc: 0.877945	valid_0's binary_logloss: 0.123612	valid_1's auc: 0.842028	valid_1's binary_logloss: 0.135826
    [38]	valid_0's auc: 0.878391	valid_0's binary_logloss: 0.123437	valid_1's auc: 0.841969	valid_1's binary_logloss: 0.135815
    [39]	valid_0's auc: 0.878886	valid_0's binary_logloss: 0.123211	valid_1's auc: 0.84193	valid_1's binary_logloss: 0.135809
    [40]	valid_0's auc: 0.879487	valid_0's binary_logloss: 0.123003	valid_1's auc: 0.841846	valid_1's binary_logloss: 0.135816
    [41]	valid_0's auc: 0.880192	valid_0's binary_logloss: 0.12276	valid_1's auc: 0.841501	valid_1's binary_logloss: 0.135878
    [42]	valid_0's auc: 0.880817	valid_0's binary_logloss: 0.122555	valid_1's auc: 0.841224	valid_1's binary_logloss: 0.135894
    [43]	valid_0's auc: 0.881168	valid_0's binary_logloss: 0.122344	valid_1's auc: 0.841189	valid_1's binary_logloss: 0.135895
    [44]	valid_0's auc: 0.88156	valid_0's binary_logloss: 0.122151	valid_1's auc: 0.841136	valid_1's binary_logloss: 0.135914
    [45]	valid_0's auc: 0.882438	valid_0's binary_logloss: 0.121855	valid_1's auc: 0.840553	valid_1's binary_logloss: 0.136015
    [46]	valid_0's auc: 0.883226	valid_0's binary_logloss: 0.121617	valid_1's auc: 0.840057	valid_1's binary_logloss: 0.136084
    [47]	valid_0's auc: 0.883701	valid_0's binary_logloss: 0.121428	valid_1's auc: 0.839905	valid_1's binary_logloss: 0.13611
    [48]	valid_0's auc: 0.884405	valid_0's binary_logloss: 0.121174	valid_1's auc: 0.840003	valid_1's binary_logloss: 0.136077
    [49]	valid_0's auc: 0.8847	valid_0's binary_logloss: 0.12101	valid_1's auc: 0.839683	valid_1's binary_logloss: 0.136145
    [50]	valid_0's auc: 0.885111	valid_0's binary_logloss: 0.120814	valid_1's auc: 0.839749	valid_1's binary_logloss: 0.136099
    [51]	valid_0's auc: 0.885425	valid_0's binary_logloss: 0.12065	valid_1's auc: 0.839416	valid_1's binary_logloss: 0.13609
    [52]	valid_0's auc: 0.885826	valid_0's binary_logloss: 0.120466	valid_1's auc: 0.83916	valid_1's binary_logloss: 0.136158
    [53]	valid_0's auc: 0.88609	valid_0's binary_logloss: 0.120311	valid_1's auc: 0.839381	valid_1's binary_logloss: 0.136151
    [54]	valid_0's auc: 0.886458	valid_0's binary_logloss: 0.120122	valid_1's auc: 0.839222	valid_1's binary_logloss: 0.136197
    [55]	valid_0's auc: 0.887239	valid_0's binary_logloss: 0.11994	valid_1's auc: 0.839191	valid_1's binary_logloss: 0.136207
    [56]	valid_0's auc: 0.887885	valid_0's binary_logloss: 0.119726	valid_1's auc: 0.839099	valid_1's binary_logloss: 0.136256
    [57]	valid_0's auc: 0.888068	valid_0's binary_logloss: 0.119592	valid_1's auc: 0.83885	valid_1's binary_logloss: 0.136275
    [58]	valid_0's auc: 0.888634	valid_0's binary_logloss: 0.119455	valid_1's auc: 0.838851	valid_1's binary_logloss: 0.136269
    [59]	valid_0's auc: 0.888873	valid_0's binary_logloss: 0.119334	valid_1's auc: 0.838683	valid_1's binary_logloss: 0.13629
    [60]	valid_0's auc: 0.889172	valid_0's binary_logloss: 0.119194	valid_1's auc: 0.838562	valid_1's binary_logloss: 0.136301
    [61]	valid_0's auc: 0.889511	valid_0's binary_logloss: 0.119036	valid_1's auc: 0.838472	valid_1's binary_logloss: 0.136335
    [62]	valid_0's auc: 0.889884	valid_0's binary_logloss: 0.118877	valid_1's auc: 0.838355	valid_1's binary_logloss: 0.136343
    [63]	valid_0's auc: 0.890436	valid_0's binary_logloss: 0.118679	valid_1's auc: 0.838056	valid_1's binary_logloss: 0.136425
    [64]	valid_0's auc: 0.890662	valid_0's binary_logloss: 0.118521	valid_1's auc: 0.837425	valid_1's binary_logloss: 0.136547
    [65]	valid_0's auc: 0.890862	valid_0's binary_logloss: 0.118389	valid_1's auc: 0.837399	valid_1's binary_logloss: 0.13655
    [66]	valid_0's auc: 0.890931	valid_0's binary_logloss: 0.118288	valid_1's auc: 0.837035	valid_1's binary_logloss: 0.13665
    [67]	valid_0's auc: 0.891015	valid_0's binary_logloss: 0.118183	valid_1's auc: 0.836707	valid_1's binary_logloss: 0.136783
    Early stopping, best iteration is:
    [37]	valid_0's auc: 0.877945	valid_0's binary_logloss: 0.123612	valid_1's auc: 0.842028	valid_1's binary_logloss: 0.135826
    [1]	valid_0's auc: 0.823507	valid_0's binary_logloss: 0.157501	valid_1's auc: 0.810415	valid_1's binary_logloss: 0.160155
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.829581	valid_0's binary_logloss: 0.152083	valid_1's auc: 0.815091	valid_1's binary_logloss: 0.155417
    [3]	valid_0's auc: 0.834308	valid_0's binary_logloss: 0.148148	valid_1's auc: 0.818622	valid_1's binary_logloss: 0.152027
    [4]	valid_0's auc: 0.836859	valid_0's binary_logloss: 0.145159	valid_1's auc: 0.819436	valid_1's binary_logloss: 0.149485
    [5]	valid_0's auc: 0.841846	valid_0's binary_logloss: 0.142678	valid_1's auc: 0.823324	valid_1's binary_logloss: 0.147372
    [6]	valid_0's auc: 0.844625	valid_0's binary_logloss: 0.140636	valid_1's auc: 0.824092	valid_1's binary_logloss: 0.145715
    [7]	valid_0's auc: 0.848223	valid_0's binary_logloss: 0.138877	valid_1's auc: 0.826887	valid_1's binary_logloss: 0.144253
    [8]	valid_0's auc: 0.849635	valid_0's binary_logloss: 0.137406	valid_1's auc: 0.829164	valid_1's binary_logloss: 0.143039
    [9]	valid_0's auc: 0.850865	valid_0's binary_logloss: 0.136105	valid_1's auc: 0.828916	valid_1's binary_logloss: 0.142109
    [10]	valid_0's auc: 0.852877	valid_0's binary_logloss: 0.134962	valid_1's auc: 0.829565	valid_1's binary_logloss: 0.141319
    [11]	valid_0's auc: 0.853498	valid_0's binary_logloss: 0.134001	valid_1's auc: 0.829604	valid_1's binary_logloss: 0.140625
    [12]	valid_0's auc: 0.854677	valid_0's binary_logloss: 0.133137	valid_1's auc: 0.828978	valid_1's binary_logloss: 0.140133
    [13]	valid_0's auc: 0.856296	valid_0's binary_logloss: 0.132367	valid_1's auc: 0.830184	valid_1's binary_logloss: 0.139696
    [14]	valid_0's auc: 0.858364	valid_0's binary_logloss: 0.131592	valid_1's auc: 0.831576	valid_1's binary_logloss: 0.139209
    [15]	valid_0's auc: 0.859632	valid_0's binary_logloss: 0.130943	valid_1's auc: 0.833147	valid_1's binary_logloss: 0.138759
    [16]	valid_0's auc: 0.86077	valid_0's binary_logloss: 0.130306	valid_1's auc: 0.833392	valid_1's binary_logloss: 0.138459
    [17]	valid_0's auc: 0.861722	valid_0's binary_logloss: 0.129793	valid_1's auc: 0.834279	valid_1's binary_logloss: 0.138085
    [18]	valid_0's auc: 0.862253	valid_0's binary_logloss: 0.129343	valid_1's auc: 0.834323	valid_1's binary_logloss: 0.137866
    [19]	valid_0's auc: 0.863766	valid_0's binary_logloss: 0.128824	valid_1's auc: 0.834736	valid_1's binary_logloss: 0.13763
    [20]	valid_0's auc: 0.8643	valid_0's binary_logloss: 0.128423	valid_1's auc: 0.834528	valid_1's binary_logloss: 0.137554
    [21]	valid_0's auc: 0.865267	valid_0's binary_logloss: 0.127958	valid_1's auc: 0.834478	valid_1's binary_logloss: 0.137406
    [22]	valid_0's auc: 0.866608	valid_0's binary_logloss: 0.127506	valid_1's auc: 0.834905	valid_1's binary_logloss: 0.137271
    [23]	valid_0's auc: 0.867804	valid_0's binary_logloss: 0.127104	valid_1's auc: 0.834293	valid_1's binary_logloss: 0.137211
    [24]	valid_0's auc: 0.86846	valid_0's binary_logloss: 0.12677	valid_1's auc: 0.833842	valid_1's binary_logloss: 0.137176
    [25]	valid_0's auc: 0.869535	valid_0's binary_logloss: 0.126409	valid_1's auc: 0.834057	valid_1's binary_logloss: 0.13708
    [26]	valid_0's auc: 0.870946	valid_0's binary_logloss: 0.126026	valid_1's auc: 0.833797	valid_1's binary_logloss: 0.137021
    [27]	valid_0's auc: 0.871863	valid_0's binary_logloss: 0.12567	valid_1's auc: 0.83361	valid_1's binary_logloss: 0.13702
    [28]	valid_0's auc: 0.872503	valid_0's binary_logloss: 0.125332	valid_1's auc: 0.833415	valid_1's binary_logloss: 0.136948
    [29]	valid_0's auc: 0.873256	valid_0's binary_logloss: 0.125049	valid_1's auc: 0.83344	valid_1's binary_logloss: 0.136909
    [30]	valid_0's auc: 0.874265	valid_0's binary_logloss: 0.124767	valid_1's auc: 0.833129	valid_1's binary_logloss: 0.136924
    [31]	valid_0's auc: 0.875083	valid_0's binary_logloss: 0.124478	valid_1's auc: 0.833207	valid_1's binary_logloss: 0.136915
    [32]	valid_0's auc: 0.875789	valid_0's binary_logloss: 0.124224	valid_1's auc: 0.833079	valid_1's binary_logloss: 0.136916
    [33]	valid_0's auc: 0.876187	valid_0's binary_logloss: 0.123994	valid_1's auc: 0.833172	valid_1's binary_logloss: 0.136856
    [34]	valid_0's auc: 0.876733	valid_0's binary_logloss: 0.123732	valid_1's auc: 0.833412	valid_1's binary_logloss: 0.136808
    [35]	valid_0's auc: 0.877384	valid_0's binary_logloss: 0.123461	valid_1's auc: 0.833287	valid_1's binary_logloss: 0.136785
    [36]	valid_0's auc: 0.877802	valid_0's binary_logloss: 0.123212	valid_1's auc: 0.83302	valid_1's binary_logloss: 0.13685
    [37]	valid_0's auc: 0.878799	valid_0's binary_logloss: 0.122947	valid_1's auc: 0.832295	valid_1's binary_logloss: 0.136905
    [38]	valid_0's auc: 0.879503	valid_0's binary_logloss: 0.122719	valid_1's auc: 0.832134	valid_1's binary_logloss: 0.136919
    [39]	valid_0's auc: 0.880358	valid_0's binary_logloss: 0.122488	valid_1's auc: 0.832266	valid_1's binary_logloss: 0.136938
    [40]	valid_0's auc: 0.881013	valid_0's binary_logloss: 0.122287	valid_1's auc: 0.832133	valid_1's binary_logloss: 0.136932
    [41]	valid_0's auc: 0.881424	valid_0's binary_logloss: 0.122107	valid_1's auc: 0.832209	valid_1's binary_logloss: 0.136934
    [42]	valid_0's auc: 0.881984	valid_0's binary_logloss: 0.121921	valid_1's auc: 0.831997	valid_1's binary_logloss: 0.136976
    [43]	valid_0's auc: 0.882635	valid_0's binary_logloss: 0.121694	valid_1's auc: 0.831527	valid_1's binary_logloss: 0.137038
    [44]	valid_0's auc: 0.883243	valid_0's binary_logloss: 0.121523	valid_1's auc: 0.831593	valid_1's binary_logloss: 0.137055
    [45]	valid_0's auc: 0.883616	valid_0's binary_logloss: 0.121345	valid_1's auc: 0.831225	valid_1's binary_logloss: 0.137134
    [46]	valid_0's auc: 0.884106	valid_0's binary_logloss: 0.121155	valid_1's auc: 0.831678	valid_1's binary_logloss: 0.137064
    [47]	valid_0's auc: 0.884462	valid_0's binary_logloss: 0.120968	valid_1's auc: 0.832127	valid_1's binary_logloss: 0.137012
    [48]	valid_0's auc: 0.884804	valid_0's binary_logloss: 0.120754	valid_1's auc: 0.832474	valid_1's binary_logloss: 0.136991
    [49]	valid_0's auc: 0.885391	valid_0's binary_logloss: 0.120518	valid_1's auc: 0.832375	valid_1's binary_logloss: 0.137054
    [50]	valid_0's auc: 0.885988	valid_0's binary_logloss: 0.120287	valid_1's auc: 0.832272	valid_1's binary_logloss: 0.137065
    [51]	valid_0's auc: 0.886393	valid_0's binary_logloss: 0.120131	valid_1's auc: 0.831867	valid_1's binary_logloss: 0.137159
    [52]	valid_0's auc: 0.886879	valid_0's binary_logloss: 0.119924	valid_1's auc: 0.831971	valid_1's binary_logloss: 0.137133
    Early stopping, best iteration is:
    [22]	valid_0's auc: 0.866608	valid_0's binary_logloss: 0.127506	valid_1's auc: 0.834905	valid_1's binary_logloss: 0.137271
    [1]	valid_0's auc: 0.822135	valid_0's binary_logloss: 0.157275	valid_1's auc: 0.80763	valid_1's binary_logloss: 0.159959
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.826629	valid_0's binary_logloss: 0.15186	valid_1's auc: 0.811377	valid_1's binary_logloss: 0.155197
    [3]	valid_0's auc: 0.830813	valid_0's binary_logloss: 0.147949	valid_1's auc: 0.813917	valid_1's binary_logloss: 0.151946
    [4]	valid_0's auc: 0.833387	valid_0's binary_logloss: 0.144809	valid_1's auc: 0.814641	valid_1's binary_logloss: 0.149358
    [5]	valid_0's auc: 0.838116	valid_0's binary_logloss: 0.142332	valid_1's auc: 0.816965	valid_1's binary_logloss: 0.147361
    [6]	valid_0's auc: 0.841055	valid_0's binary_logloss: 0.140267	valid_1's auc: 0.81978	valid_1's binary_logloss: 0.145867
    [7]	valid_0's auc: 0.842021	valid_0's binary_logloss: 0.138495	valid_1's auc: 0.819689	valid_1's binary_logloss: 0.144575
    [8]	valid_0's auc: 0.843828	valid_0's binary_logloss: 0.137029	valid_1's auc: 0.822496	valid_1's binary_logloss: 0.143422
    [9]	valid_0's auc: 0.845522	valid_0's binary_logloss: 0.135783	valid_1's auc: 0.823288	valid_1's binary_logloss: 0.142428
    [10]	valid_0's auc: 0.847654	valid_0's binary_logloss: 0.134654	valid_1's auc: 0.824658	valid_1's binary_logloss: 0.141712
    [11]	valid_0's auc: 0.849302	valid_0's binary_logloss: 0.133679	valid_1's auc: 0.826788	valid_1's binary_logloss: 0.140983
    [12]	valid_0's auc: 0.850949	valid_0's binary_logloss: 0.132827	valid_1's auc: 0.828316	valid_1's binary_logloss: 0.140403
    [13]	valid_0's auc: 0.852852	valid_0's binary_logloss: 0.132062	valid_1's auc: 0.830745	valid_1's binary_logloss: 0.139849
    [14]	valid_0's auc: 0.854299	valid_0's binary_logloss: 0.131363	valid_1's auc: 0.832741	valid_1's binary_logloss: 0.139383
    [15]	valid_0's auc: 0.855376	valid_0's binary_logloss: 0.13071	valid_1's auc: 0.833869	valid_1's binary_logloss: 0.138963
    [16]	valid_0's auc: 0.856265	valid_0's binary_logloss: 0.130166	valid_1's auc: 0.833925	valid_1's binary_logloss: 0.138745
    [17]	valid_0's auc: 0.85735	valid_0's binary_logloss: 0.129615	valid_1's auc: 0.834227	valid_1's binary_logloss: 0.138474
    [18]	valid_0's auc: 0.859279	valid_0's binary_logloss: 0.129153	valid_1's auc: 0.834146	valid_1's binary_logloss: 0.138306
    [19]	valid_0's auc: 0.860099	valid_0's binary_logloss: 0.128684	valid_1's auc: 0.833391	valid_1's binary_logloss: 0.138142
    [20]	valid_0's auc: 0.861239	valid_0's binary_logloss: 0.128272	valid_1's auc: 0.833651	valid_1's binary_logloss: 0.13795
    [21]	valid_0's auc: 0.862153	valid_0's binary_logloss: 0.127883	valid_1's auc: 0.834662	valid_1's binary_logloss: 0.137829
    [22]	valid_0's auc: 0.863466	valid_0's binary_logloss: 0.127439	valid_1's auc: 0.834505	valid_1's binary_logloss: 0.137762
    [23]	valid_0's auc: 0.86379	valid_0's binary_logloss: 0.127135	valid_1's auc: 0.834377	valid_1's binary_logloss: 0.137657
    [24]	valid_0's auc: 0.864548	valid_0's binary_logloss: 0.126788	valid_1's auc: 0.834289	valid_1's binary_logloss: 0.137499
    [25]	valid_0's auc: 0.865681	valid_0's binary_logloss: 0.126451	valid_1's auc: 0.834899	valid_1's binary_logloss: 0.137329
    [26]	valid_0's auc: 0.866659	valid_0's binary_logloss: 0.126121	valid_1's auc: 0.835946	valid_1's binary_logloss: 0.137177
    [27]	valid_0's auc: 0.867505	valid_0's binary_logloss: 0.125853	valid_1's auc: 0.836616	valid_1's binary_logloss: 0.137016
    [28]	valid_0's auc: 0.868089	valid_0's binary_logloss: 0.125584	valid_1's auc: 0.836565	valid_1's binary_logloss: 0.137
    [29]	valid_0's auc: 0.868709	valid_0's binary_logloss: 0.125327	valid_1's auc: 0.837028	valid_1's binary_logloss: 0.136878
    [30]	valid_0's auc: 0.87006	valid_0's binary_logloss: 0.12501	valid_1's auc: 0.837512	valid_1's binary_logloss: 0.136792
    [31]	valid_0's auc: 0.870612	valid_0's binary_logloss: 0.124704	valid_1's auc: 0.837541	valid_1's binary_logloss: 0.136757
    [32]	valid_0's auc: 0.871234	valid_0's binary_logloss: 0.124431	valid_1's auc: 0.837141	valid_1's binary_logloss: 0.136755
    [33]	valid_0's auc: 0.872027	valid_0's binary_logloss: 0.124167	valid_1's auc: 0.837258	valid_1's binary_logloss: 0.136728
    [34]	valid_0's auc: 0.87244	valid_0's binary_logloss: 0.123956	valid_1's auc: 0.836829	valid_1's binary_logloss: 0.136799
    [35]	valid_0's auc: 0.873259	valid_0's binary_logloss: 0.123713	valid_1's auc: 0.836871	valid_1's binary_logloss: 0.136775
    [36]	valid_0's auc: 0.873695	valid_0's binary_logloss: 0.123497	valid_1's auc: 0.836512	valid_1's binary_logloss: 0.136859
    [37]	valid_0's auc: 0.874365	valid_0's binary_logloss: 0.123259	valid_1's auc: 0.836421	valid_1's binary_logloss: 0.136872
    [38]	valid_0's auc: 0.874998	valid_0's binary_logloss: 0.123014	valid_1's auc: 0.83628	valid_1's binary_logloss: 0.136939
    [39]	valid_0's auc: 0.875666	valid_0's binary_logloss: 0.122811	valid_1's auc: 0.836577	valid_1's binary_logloss: 0.13689
    [40]	valid_0's auc: 0.875931	valid_0's binary_logloss: 0.122638	valid_1's auc: 0.836148	valid_1's binary_logloss: 0.13697
    [41]	valid_0's auc: 0.876403	valid_0's binary_logloss: 0.122474	valid_1's auc: 0.836008	valid_1's binary_logloss: 0.137005
    [42]	valid_0's auc: 0.87697	valid_0's binary_logloss: 0.122306	valid_1's auc: 0.835814	valid_1's binary_logloss: 0.137031
    [43]	valid_0's auc: 0.877305	valid_0's binary_logloss: 0.122115	valid_1's auc: 0.835467	valid_1's binary_logloss: 0.137112
    [44]	valid_0's auc: 0.87782	valid_0's binary_logloss: 0.121932	valid_1's auc: 0.835339	valid_1's binary_logloss: 0.137217
    [45]	valid_0's auc: 0.878559	valid_0's binary_logloss: 0.121701	valid_1's auc: 0.83506	valid_1's binary_logloss: 0.137256
    [46]	valid_0's auc: 0.878895	valid_0's binary_logloss: 0.121564	valid_1's auc: 0.834955	valid_1's binary_logloss: 0.137255
    [47]	valid_0's auc: 0.879643	valid_0's binary_logloss: 0.121306	valid_1's auc: 0.834987	valid_1's binary_logloss: 0.137275
    [48]	valid_0's auc: 0.88002	valid_0's binary_logloss: 0.121119	valid_1's auc: 0.834864	valid_1's binary_logloss: 0.137329
    [49]	valid_0's auc: 0.880318	valid_0's binary_logloss: 0.120965	valid_1's auc: 0.834547	valid_1's binary_logloss: 0.137394
    [50]	valid_0's auc: 0.880967	valid_0's binary_logloss: 0.12082	valid_1's auc: 0.834588	valid_1's binary_logloss: 0.137388
    [51]	valid_0's auc: 0.8813	valid_0's binary_logloss: 0.120647	valid_1's auc: 0.834288	valid_1's binary_logloss: 0.137467
    [52]	valid_0's auc: 0.881706	valid_0's binary_logloss: 0.120482	valid_1's auc: 0.834305	valid_1's binary_logloss: 0.137491
    [53]	valid_0's auc: 0.882192	valid_0's binary_logloss: 0.1203	valid_1's auc: 0.834662	valid_1's binary_logloss: 0.137492
    [54]	valid_0's auc: 0.882307	valid_0's binary_logloss: 0.120188	valid_1's auc: 0.834241	valid_1's binary_logloss: 0.13763
    [55]	valid_0's auc: 0.883062	valid_0's binary_logloss: 0.119929	valid_1's auc: 0.834145	valid_1's binary_logloss: 0.137674
    [56]	valid_0's auc: 0.883272	valid_0's binary_logloss: 0.119823	valid_1's auc: 0.833884	valid_1's binary_logloss: 0.137762
    [57]	valid_0's auc: 0.883628	valid_0's binary_logloss: 0.11968	valid_1's auc: 0.833563	valid_1's binary_logloss: 0.13782
    [58]	valid_0's auc: 0.884097	valid_0's binary_logloss: 0.119546	valid_1's auc: 0.833647	valid_1's binary_logloss: 0.137781
    [59]	valid_0's auc: 0.884337	valid_0's binary_logloss: 0.119387	valid_1's auc: 0.833138	valid_1's binary_logloss: 0.137878
    [60]	valid_0's auc: 0.884777	valid_0's binary_logloss: 0.119188	valid_1's auc: 0.832909	valid_1's binary_logloss: 0.13796
    [61]	valid_0's auc: 0.885076	valid_0's binary_logloss: 0.119054	valid_1's auc: 0.833025	valid_1's binary_logloss: 0.137959
    Early stopping, best iteration is:
    [31]	valid_0's auc: 0.870612	valid_0's binary_logloss: 0.124704	valid_1's auc: 0.837541	valid_1's binary_logloss: 0.136757
    [1]	valid_0's auc: 0.825036	valid_0's binary_logloss: 0.157424	valid_1's auc: 0.807618	valid_1's binary_logloss: 0.159921
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.829804	valid_0's binary_logloss: 0.152132	valid_1's auc: 0.815884	valid_1's binary_logloss: 0.155124
    [3]	valid_0's auc: 0.835123	valid_0's binary_logloss: 0.148284	valid_1's auc: 0.81743	valid_1's binary_logloss: 0.151751
    [4]	valid_0's auc: 0.841579	valid_0's binary_logloss: 0.145192	valid_1's auc: 0.823237	valid_1's binary_logloss: 0.149159
    [5]	valid_0's auc: 0.843972	valid_0's binary_logloss: 0.142753	valid_1's auc: 0.82747	valid_1's binary_logloss: 0.147042
    [6]	valid_0's auc: 0.846408	valid_0's binary_logloss: 0.140687	valid_1's auc: 0.830402	valid_1's binary_logloss: 0.145398
    [7]	valid_0's auc: 0.848617	valid_0's binary_logloss: 0.138978	valid_1's auc: 0.830319	valid_1's binary_logloss: 0.144054
    [8]	valid_0's auc: 0.849973	valid_0's binary_logloss: 0.137527	valid_1's auc: 0.831414	valid_1's binary_logloss: 0.142955
    [9]	valid_0's auc: 0.852151	valid_0's binary_logloss: 0.136279	valid_1's auc: 0.834577	valid_1's binary_logloss: 0.141825
    [10]	valid_0's auc: 0.852962	valid_0's binary_logloss: 0.135231	valid_1's auc: 0.835301	valid_1's binary_logloss: 0.140947
    [11]	valid_0's auc: 0.853769	valid_0's binary_logloss: 0.134247	valid_1's auc: 0.836379	valid_1's binary_logloss: 0.140121
    [12]	valid_0's auc: 0.855739	valid_0's binary_logloss: 0.133341	valid_1's auc: 0.83615	valid_1's binary_logloss: 0.139629
    [13]	valid_0's auc: 0.857549	valid_0's binary_logloss: 0.132607	valid_1's auc: 0.836232	valid_1's binary_logloss: 0.139209
    [14]	valid_0's auc: 0.858955	valid_0's binary_logloss: 0.131867	valid_1's auc: 0.836387	valid_1's binary_logloss: 0.138827
    [15]	valid_0's auc: 0.860018	valid_0's binary_logloss: 0.13125	valid_1's auc: 0.837299	valid_1's binary_logloss: 0.138441
    [16]	valid_0's auc: 0.861059	valid_0's binary_logloss: 0.130679	valid_1's auc: 0.838197	valid_1's binary_logloss: 0.138074
    [17]	valid_0's auc: 0.862112	valid_0's binary_logloss: 0.130128	valid_1's auc: 0.838835	valid_1's binary_logloss: 0.137791
    [18]	valid_0's auc: 0.862994	valid_0's binary_logloss: 0.129646	valid_1's auc: 0.838904	valid_1's binary_logloss: 0.137577
    [19]	valid_0's auc: 0.863568	valid_0's binary_logloss: 0.129195	valid_1's auc: 0.839028	valid_1's binary_logloss: 0.137319
    [20]	valid_0's auc: 0.86472	valid_0's binary_logloss: 0.128729	valid_1's auc: 0.83984	valid_1's binary_logloss: 0.137089
    [21]	valid_0's auc: 0.866051	valid_0's binary_logloss: 0.128315	valid_1's auc: 0.840385	valid_1's binary_logloss: 0.136889
    [22]	valid_0's auc: 0.866997	valid_0's binary_logloss: 0.127908	valid_1's auc: 0.840544	valid_1's binary_logloss: 0.136724
    [23]	valid_0's auc: 0.867998	valid_0's binary_logloss: 0.127516	valid_1's auc: 0.840471	valid_1's binary_logloss: 0.136628
    [24]	valid_0's auc: 0.868835	valid_0's binary_logloss: 0.127186	valid_1's auc: 0.84011	valid_1's binary_logloss: 0.136573
    [25]	valid_0's auc: 0.869535	valid_0's binary_logloss: 0.126855	valid_1's auc: 0.83987	valid_1's binary_logloss: 0.136502
    [26]	valid_0's auc: 0.870439	valid_0's binary_logloss: 0.126515	valid_1's auc: 0.840538	valid_1's binary_logloss: 0.136387
    [27]	valid_0's auc: 0.871466	valid_0's binary_logloss: 0.126179	valid_1's auc: 0.840326	valid_1's binary_logloss: 0.136301
    [28]	valid_0's auc: 0.872022	valid_0's binary_logloss: 0.125897	valid_1's auc: 0.840415	valid_1's binary_logloss: 0.136253
    [29]	valid_0's auc: 0.873022	valid_0's binary_logloss: 0.125624	valid_1's auc: 0.840531	valid_1's binary_logloss: 0.136185
    [30]	valid_0's auc: 0.873626	valid_0's binary_logloss: 0.125329	valid_1's auc: 0.840407	valid_1's binary_logloss: 0.136188
    [31]	valid_0's auc: 0.874227	valid_0's binary_logloss: 0.12509	valid_1's auc: 0.841102	valid_1's binary_logloss: 0.136046
    [32]	valid_0's auc: 0.874828	valid_0's binary_logloss: 0.124856	valid_1's auc: 0.841125	valid_1's binary_logloss: 0.135986
    [33]	valid_0's auc: 0.875492	valid_0's binary_logloss: 0.124583	valid_1's auc: 0.841558	valid_1's binary_logloss: 0.135907
    [34]	valid_0's auc: 0.876123	valid_0's binary_logloss: 0.124343	valid_1's auc: 0.841695	valid_1's binary_logloss: 0.135865
    [35]	valid_0's auc: 0.876893	valid_0's binary_logloss: 0.124064	valid_1's auc: 0.841945	valid_1's binary_logloss: 0.135831
    [36]	valid_0's auc: 0.877421	valid_0's binary_logloss: 0.123823	valid_1's auc: 0.841949	valid_1's binary_logloss: 0.135864
    [37]	valid_0's auc: 0.877945	valid_0's binary_logloss: 0.123612	valid_1's auc: 0.842028	valid_1's binary_logloss: 0.135826
    [38]	valid_0's auc: 0.878391	valid_0's binary_logloss: 0.123437	valid_1's auc: 0.841969	valid_1's binary_logloss: 0.135815
    [39]	valid_0's auc: 0.878886	valid_0's binary_logloss: 0.123211	valid_1's auc: 0.84193	valid_1's binary_logloss: 0.135809
    [40]	valid_0's auc: 0.879487	valid_0's binary_logloss: 0.123003	valid_1's auc: 0.841846	valid_1's binary_logloss: 0.135816
    [41]	valid_0's auc: 0.880192	valid_0's binary_logloss: 0.12276	valid_1's auc: 0.841501	valid_1's binary_logloss: 0.135878
    [42]	valid_0's auc: 0.880817	valid_0's binary_logloss: 0.122555	valid_1's auc: 0.841224	valid_1's binary_logloss: 0.135894
    [43]	valid_0's auc: 0.881168	valid_0's binary_logloss: 0.122344	valid_1's auc: 0.841189	valid_1's binary_logloss: 0.135895
    [44]	valid_0's auc: 0.88156	valid_0's binary_logloss: 0.122151	valid_1's auc: 0.841136	valid_1's binary_logloss: 0.135914
    [45]	valid_0's auc: 0.882438	valid_0's binary_logloss: 0.121855	valid_1's auc: 0.840553	valid_1's binary_logloss: 0.136015
    [46]	valid_0's auc: 0.883226	valid_0's binary_logloss: 0.121617	valid_1's auc: 0.840057	valid_1's binary_logloss: 0.136084
    [47]	valid_0's auc: 0.883701	valid_0's binary_logloss: 0.121428	valid_1's auc: 0.839905	valid_1's binary_logloss: 0.13611
    [48]	valid_0's auc: 0.884405	valid_0's binary_logloss: 0.121174	valid_1's auc: 0.840003	valid_1's binary_logloss: 0.136077
    [49]	valid_0's auc: 0.8847	valid_0's binary_logloss: 0.12101	valid_1's auc: 0.839683	valid_1's binary_logloss: 0.136145
    [50]	valid_0's auc: 0.885111	valid_0's binary_logloss: 0.120814	valid_1's auc: 0.839749	valid_1's binary_logloss: 0.136099
    [51]	valid_0's auc: 0.885425	valid_0's binary_logloss: 0.12065	valid_1's auc: 0.839416	valid_1's binary_logloss: 0.13609
    [52]	valid_0's auc: 0.885826	valid_0's binary_logloss: 0.120466	valid_1's auc: 0.83916	valid_1's binary_logloss: 0.136158
    [53]	valid_0's auc: 0.88609	valid_0's binary_logloss: 0.120311	valid_1's auc: 0.839381	valid_1's binary_logloss: 0.136151
    [54]	valid_0's auc: 0.886458	valid_0's binary_logloss: 0.120122	valid_1's auc: 0.839222	valid_1's binary_logloss: 0.136197
    [55]	valid_0's auc: 0.887239	valid_0's binary_logloss: 0.11994	valid_1's auc: 0.839191	valid_1's binary_logloss: 0.136207
    [56]	valid_0's auc: 0.887885	valid_0's binary_logloss: 0.119726	valid_1's auc: 0.839099	valid_1's binary_logloss: 0.136256
    [57]	valid_0's auc: 0.888068	valid_0's binary_logloss: 0.119592	valid_1's auc: 0.83885	valid_1's binary_logloss: 0.136275
    [58]	valid_0's auc: 0.888634	valid_0's binary_logloss: 0.119455	valid_1's auc: 0.838851	valid_1's binary_logloss: 0.136269
    [59]	valid_0's auc: 0.888873	valid_0's binary_logloss: 0.119334	valid_1's auc: 0.838683	valid_1's binary_logloss: 0.13629
    [60]	valid_0's auc: 0.889172	valid_0's binary_logloss: 0.119194	valid_1's auc: 0.838562	valid_1's binary_logloss: 0.136301
    [61]	valid_0's auc: 0.889511	valid_0's binary_logloss: 0.119036	valid_1's auc: 0.838472	valid_1's binary_logloss: 0.136335
    [62]	valid_0's auc: 0.889884	valid_0's binary_logloss: 0.118877	valid_1's auc: 0.838355	valid_1's binary_logloss: 0.136343
    [63]	valid_0's auc: 0.890436	valid_0's binary_logloss: 0.118679	valid_1's auc: 0.838056	valid_1's binary_logloss: 0.136425
    [64]	valid_0's auc: 0.890662	valid_0's binary_logloss: 0.118521	valid_1's auc: 0.837425	valid_1's binary_logloss: 0.136547
    [65]	valid_0's auc: 0.890862	valid_0's binary_logloss: 0.118389	valid_1's auc: 0.837399	valid_1's binary_logloss: 0.13655
    [66]	valid_0's auc: 0.890931	valid_0's binary_logloss: 0.118288	valid_1's auc: 0.837035	valid_1's binary_logloss: 0.13665
    [67]	valid_0's auc: 0.891015	valid_0's binary_logloss: 0.118183	valid_1's auc: 0.836707	valid_1's binary_logloss: 0.136783
    Early stopping, best iteration is:
    [37]	valid_0's auc: 0.877945	valid_0's binary_logloss: 0.123612	valid_1's auc: 0.842028	valid_1's binary_logloss: 0.135826
    [1]	valid_0's auc: 0.823507	valid_0's binary_logloss: 0.157501	valid_1's auc: 0.810415	valid_1's binary_logloss: 0.160155
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.829581	valid_0's binary_logloss: 0.152083	valid_1's auc: 0.815091	valid_1's binary_logloss: 0.155417
    [3]	valid_0's auc: 0.834308	valid_0's binary_logloss: 0.148148	valid_1's auc: 0.818622	valid_1's binary_logloss: 0.152027
    [4]	valid_0's auc: 0.836859	valid_0's binary_logloss: 0.145159	valid_1's auc: 0.819436	valid_1's binary_logloss: 0.149485
    [5]	valid_0's auc: 0.841846	valid_0's binary_logloss: 0.142678	valid_1's auc: 0.823324	valid_1's binary_logloss: 0.147372
    [6]	valid_0's auc: 0.844625	valid_0's binary_logloss: 0.140636	valid_1's auc: 0.824092	valid_1's binary_logloss: 0.145715
    [7]	valid_0's auc: 0.848223	valid_0's binary_logloss: 0.138877	valid_1's auc: 0.826887	valid_1's binary_logloss: 0.144253
    [8]	valid_0's auc: 0.849635	valid_0's binary_logloss: 0.137406	valid_1's auc: 0.829164	valid_1's binary_logloss: 0.143039
    [9]	valid_0's auc: 0.850865	valid_0's binary_logloss: 0.136105	valid_1's auc: 0.828916	valid_1's binary_logloss: 0.142109
    [10]	valid_0's auc: 0.852877	valid_0's binary_logloss: 0.134962	valid_1's auc: 0.829565	valid_1's binary_logloss: 0.141319
    [11]	valid_0's auc: 0.853498	valid_0's binary_logloss: 0.134001	valid_1's auc: 0.829604	valid_1's binary_logloss: 0.140625
    [12]	valid_0's auc: 0.854677	valid_0's binary_logloss: 0.133137	valid_1's auc: 0.828978	valid_1's binary_logloss: 0.140133
    [13]	valid_0's auc: 0.856296	valid_0's binary_logloss: 0.132367	valid_1's auc: 0.830184	valid_1's binary_logloss: 0.139696
    [14]	valid_0's auc: 0.858364	valid_0's binary_logloss: 0.131592	valid_1's auc: 0.831576	valid_1's binary_logloss: 0.139209
    [15]	valid_0's auc: 0.859632	valid_0's binary_logloss: 0.130943	valid_1's auc: 0.833147	valid_1's binary_logloss: 0.138759
    [16]	valid_0's auc: 0.86077	valid_0's binary_logloss: 0.130306	valid_1's auc: 0.833392	valid_1's binary_logloss: 0.138459
    [17]	valid_0's auc: 0.861722	valid_0's binary_logloss: 0.129793	valid_1's auc: 0.834279	valid_1's binary_logloss: 0.138085
    [18]	valid_0's auc: 0.862253	valid_0's binary_logloss: 0.129343	valid_1's auc: 0.834323	valid_1's binary_logloss: 0.137866
    [19]	valid_0's auc: 0.863766	valid_0's binary_logloss: 0.128824	valid_1's auc: 0.834736	valid_1's binary_logloss: 0.13763
    [20]	valid_0's auc: 0.8643	valid_0's binary_logloss: 0.128423	valid_1's auc: 0.834528	valid_1's binary_logloss: 0.137554
    [21]	valid_0's auc: 0.865267	valid_0's binary_logloss: 0.127958	valid_1's auc: 0.834478	valid_1's binary_logloss: 0.137406
    [22]	valid_0's auc: 0.866608	valid_0's binary_logloss: 0.127506	valid_1's auc: 0.834905	valid_1's binary_logloss: 0.137271
    [23]	valid_0's auc: 0.867804	valid_0's binary_logloss: 0.127104	valid_1's auc: 0.834293	valid_1's binary_logloss: 0.137211
    [24]	valid_0's auc: 0.86846	valid_0's binary_logloss: 0.12677	valid_1's auc: 0.833842	valid_1's binary_logloss: 0.137176
    [25]	valid_0's auc: 0.869535	valid_0's binary_logloss: 0.126409	valid_1's auc: 0.834057	valid_1's binary_logloss: 0.13708
    [26]	valid_0's auc: 0.870946	valid_0's binary_logloss: 0.126026	valid_1's auc: 0.833797	valid_1's binary_logloss: 0.137021
    [27]	valid_0's auc: 0.871863	valid_0's binary_logloss: 0.12567	valid_1's auc: 0.83361	valid_1's binary_logloss: 0.13702
    [28]	valid_0's auc: 0.872503	valid_0's binary_logloss: 0.125332	valid_1's auc: 0.833415	valid_1's binary_logloss: 0.136948
    [29]	valid_0's auc: 0.873256	valid_0's binary_logloss: 0.125049	valid_1's auc: 0.83344	valid_1's binary_logloss: 0.136909
    [30]	valid_0's auc: 0.874265	valid_0's binary_logloss: 0.124767	valid_1's auc: 0.833129	valid_1's binary_logloss: 0.136924
    [31]	valid_0's auc: 0.875083	valid_0's binary_logloss: 0.124478	valid_1's auc: 0.833207	valid_1's binary_logloss: 0.136915
    [32]	valid_0's auc: 0.875789	valid_0's binary_logloss: 0.124224	valid_1's auc: 0.833079	valid_1's binary_logloss: 0.136916
    [33]	valid_0's auc: 0.876187	valid_0's binary_logloss: 0.123994	valid_1's auc: 0.833172	valid_1's binary_logloss: 0.136856
    [34]	valid_0's auc: 0.876733	valid_0's binary_logloss: 0.123732	valid_1's auc: 0.833412	valid_1's binary_logloss: 0.136808
    [35]	valid_0's auc: 0.877384	valid_0's binary_logloss: 0.123461	valid_1's auc: 0.833287	valid_1's binary_logloss: 0.136785
    [36]	valid_0's auc: 0.877802	valid_0's binary_logloss: 0.123212	valid_1's auc: 0.83302	valid_1's binary_logloss: 0.13685
    [37]	valid_0's auc: 0.878799	valid_0's binary_logloss: 0.122947	valid_1's auc: 0.832295	valid_1's binary_logloss: 0.136905
    [38]	valid_0's auc: 0.879503	valid_0's binary_logloss: 0.122719	valid_1's auc: 0.832134	valid_1's binary_logloss: 0.136919
    [39]	valid_0's auc: 0.880358	valid_0's binary_logloss: 0.122488	valid_1's auc: 0.832266	valid_1's binary_logloss: 0.136938
    [40]	valid_0's auc: 0.881013	valid_0's binary_logloss: 0.122287	valid_1's auc: 0.832133	valid_1's binary_logloss: 0.136932
    [41]	valid_0's auc: 0.881424	valid_0's binary_logloss: 0.122107	valid_1's auc: 0.832209	valid_1's binary_logloss: 0.136934
    [42]	valid_0's auc: 0.881984	valid_0's binary_logloss: 0.121921	valid_1's auc: 0.831997	valid_1's binary_logloss: 0.136976
    [43]	valid_0's auc: 0.882635	valid_0's binary_logloss: 0.121694	valid_1's auc: 0.831527	valid_1's binary_logloss: 0.137038
    [44]	valid_0's auc: 0.883243	valid_0's binary_logloss: 0.121523	valid_1's auc: 0.831593	valid_1's binary_logloss: 0.137055
    [45]	valid_0's auc: 0.883616	valid_0's binary_logloss: 0.121345	valid_1's auc: 0.831225	valid_1's binary_logloss: 0.137134
    [46]	valid_0's auc: 0.884106	valid_0's binary_logloss: 0.121155	valid_1's auc: 0.831678	valid_1's binary_logloss: 0.137064
    [47]	valid_0's auc: 0.884462	valid_0's binary_logloss: 0.120968	valid_1's auc: 0.832127	valid_1's binary_logloss: 0.137012
    [48]	valid_0's auc: 0.884804	valid_0's binary_logloss: 0.120754	valid_1's auc: 0.832474	valid_1's binary_logloss: 0.136991
    [49]	valid_0's auc: 0.885391	valid_0's binary_logloss: 0.120518	valid_1's auc: 0.832375	valid_1's binary_logloss: 0.137054
    [50]	valid_0's auc: 0.885988	valid_0's binary_logloss: 0.120287	valid_1's auc: 0.832272	valid_1's binary_logloss: 0.137065
    [51]	valid_0's auc: 0.886393	valid_0's binary_logloss: 0.120131	valid_1's auc: 0.831867	valid_1's binary_logloss: 0.137159
    [52]	valid_0's auc: 0.886879	valid_0's binary_logloss: 0.119924	valid_1's auc: 0.831971	valid_1's binary_logloss: 0.137133
    Early stopping, best iteration is:
    [22]	valid_0's auc: 0.866608	valid_0's binary_logloss: 0.127506	valid_1's auc: 0.834905	valid_1's binary_logloss: 0.137271
    [1]	valid_0's auc: 0.825869	valid_0's binary_logloss: 0.156436	valid_1's auc: 0.803397	valid_1's binary_logloss: 0.159993
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.834474	valid_0's binary_logloss: 0.150597	valid_1's auc: 0.809972	valid_1's binary_logloss: 0.155122
    [3]	valid_0's auc: 0.840511	valid_0's binary_logloss: 0.146374	valid_1's auc: 0.819137	valid_1's binary_logloss: 0.15148
    [4]	valid_0's auc: 0.844283	valid_0's binary_logloss: 0.143027	valid_1's auc: 0.820472	valid_1's binary_logloss: 0.14904
    [5]	valid_0's auc: 0.85046	valid_0's binary_logloss: 0.140321	valid_1's auc: 0.82606	valid_1's binary_logloss: 0.146982
    [6]	valid_0's auc: 0.852881	valid_0's binary_logloss: 0.13813	valid_1's auc: 0.823869	valid_1's binary_logloss: 0.145517
    [7]	valid_0's auc: 0.857268	valid_0's binary_logloss: 0.136149	valid_1's auc: 0.827586	valid_1's binary_logloss: 0.144156
    [8]	valid_0's auc: 0.859401	valid_0's binary_logloss: 0.13452	valid_1's auc: 0.829428	valid_1's binary_logloss: 0.14303
    [9]	valid_0's auc: 0.861435	valid_0's binary_logloss: 0.133011	valid_1's auc: 0.82971	valid_1's binary_logloss: 0.142113
    [10]	valid_0's auc: 0.862282	valid_0's binary_logloss: 0.131742	valid_1's auc: 0.830527	valid_1's binary_logloss: 0.141392
    [11]	valid_0's auc: 0.86448	valid_0's binary_logloss: 0.130536	valid_1's auc: 0.83065	valid_1's binary_logloss: 0.140854
    [12]	valid_0's auc: 0.865909	valid_0's binary_logloss: 0.129495	valid_1's auc: 0.830763	valid_1's binary_logloss: 0.140425
    [13]	valid_0's auc: 0.867083	valid_0's binary_logloss: 0.128537	valid_1's auc: 0.830097	valid_1's binary_logloss: 0.140034
    [14]	valid_0's auc: 0.869164	valid_0's binary_logloss: 0.127593	valid_1's auc: 0.831242	valid_1's binary_logloss: 0.139589
    [15]	valid_0's auc: 0.869779	valid_0's binary_logloss: 0.126753	valid_1's auc: 0.830054	valid_1's binary_logloss: 0.13935
    [16]	valid_0's auc: 0.870798	valid_0's binary_logloss: 0.125998	valid_1's auc: 0.829854	valid_1's binary_logloss: 0.139147
    [17]	valid_0's auc: 0.872413	valid_0's binary_logloss: 0.125321	valid_1's auc: 0.829939	valid_1's binary_logloss: 0.138859
    [18]	valid_0's auc: 0.873656	valid_0's binary_logloss: 0.124646	valid_1's auc: 0.83033	valid_1's binary_logloss: 0.138678
    [19]	valid_0's auc: 0.874787	valid_0's binary_logloss: 0.124094	valid_1's auc: 0.829747	valid_1's binary_logloss: 0.138635
    [20]	valid_0's auc: 0.875793	valid_0's binary_logloss: 0.123547	valid_1's auc: 0.829455	valid_1's binary_logloss: 0.138492
    [21]	valid_0's auc: 0.87694	valid_0's binary_logloss: 0.122969	valid_1's auc: 0.830945	valid_1's binary_logloss: 0.13825
    [22]	valid_0's auc: 0.878632	valid_0's binary_logloss: 0.122407	valid_1's auc: 0.831483	valid_1's binary_logloss: 0.138097
    [23]	valid_0's auc: 0.8796	valid_0's binary_logloss: 0.121869	valid_1's auc: 0.831408	valid_1's binary_logloss: 0.138071
    [24]	valid_0's auc: 0.881298	valid_0's binary_logloss: 0.121363	valid_1's auc: 0.831984	valid_1's binary_logloss: 0.137912
    [25]	valid_0's auc: 0.882699	valid_0's binary_logloss: 0.120863	valid_1's auc: 0.831941	valid_1's binary_logloss: 0.137824
    [26]	valid_0's auc: 0.883799	valid_0's binary_logloss: 0.120379	valid_1's auc: 0.832897	valid_1's binary_logloss: 0.137649
    [27]	valid_0's auc: 0.884893	valid_0's binary_logloss: 0.119944	valid_1's auc: 0.832841	valid_1's binary_logloss: 0.13763
    [28]	valid_0's auc: 0.88588	valid_0's binary_logloss: 0.119545	valid_1's auc: 0.833413	valid_1's binary_logloss: 0.137495
    [29]	valid_0's auc: 0.886461	valid_0's binary_logloss: 0.11916	valid_1's auc: 0.833437	valid_1's binary_logloss: 0.137511
    [30]	valid_0's auc: 0.887592	valid_0's binary_logloss: 0.118739	valid_1's auc: 0.833429	valid_1's binary_logloss: 0.137514
    [31]	valid_0's auc: 0.888552	valid_0's binary_logloss: 0.118399	valid_1's auc: 0.834036	valid_1's binary_logloss: 0.137459
    [32]	valid_0's auc: 0.889448	valid_0's binary_logloss: 0.118019	valid_1's auc: 0.833754	valid_1's binary_logloss: 0.137498
    [33]	valid_0's auc: 0.890203	valid_0's binary_logloss: 0.117711	valid_1's auc: 0.833853	valid_1's binary_logloss: 0.137491
    [34]	valid_0's auc: 0.890868	valid_0's binary_logloss: 0.117395	valid_1's auc: 0.833648	valid_1's binary_logloss: 0.137532
    [35]	valid_0's auc: 0.891378	valid_0's binary_logloss: 0.117106	valid_1's auc: 0.833479	valid_1's binary_logloss: 0.137577
    [36]	valid_0's auc: 0.892061	valid_0's binary_logloss: 0.116741	valid_1's auc: 0.832608	valid_1's binary_logloss: 0.137799
    [37]	valid_0's auc: 0.892715	valid_0's binary_logloss: 0.116403	valid_1's auc: 0.83208	valid_1's binary_logloss: 0.137938
    [38]	valid_0's auc: 0.893146	valid_0's binary_logloss: 0.116126	valid_1's auc: 0.83196	valid_1's binary_logloss: 0.137994
    [39]	valid_0's auc: 0.893963	valid_0's binary_logloss: 0.115783	valid_1's auc: 0.831584	valid_1's binary_logloss: 0.138076
    [40]	valid_0's auc: 0.894274	valid_0's binary_logloss: 0.115535	valid_1's auc: 0.831346	valid_1's binary_logloss: 0.13818
    [41]	valid_0's auc: 0.894809	valid_0's binary_logloss: 0.115247	valid_1's auc: 0.831197	valid_1's binary_logloss: 0.138289
    [42]	valid_0's auc: 0.895332	valid_0's binary_logloss: 0.114916	valid_1's auc: 0.830893	valid_1's binary_logloss: 0.138343
    [43]	valid_0's auc: 0.895864	valid_0's binary_logloss: 0.114701	valid_1's auc: 0.83129	valid_1's binary_logloss: 0.138307
    [44]	valid_0's auc: 0.89623	valid_0's binary_logloss: 0.114411	valid_1's auc: 0.831004	valid_1's binary_logloss: 0.138408
    [45]	valid_0's auc: 0.896616	valid_0's binary_logloss: 0.11413	valid_1's auc: 0.830369	valid_1's binary_logloss: 0.138547
    [46]	valid_0's auc: 0.897076	valid_0's binary_logloss: 0.113919	valid_1's auc: 0.830246	valid_1's binary_logloss: 0.138644
    [47]	valid_0's auc: 0.897393	valid_0's binary_logloss: 0.113691	valid_1's auc: 0.829999	valid_1's binary_logloss: 0.138769
    [48]	valid_0's auc: 0.897745	valid_0's binary_logloss: 0.113465	valid_1's auc: 0.829593	valid_1's binary_logloss: 0.138908
    [49]	valid_0's auc: 0.898203	valid_0's binary_logloss: 0.113214	valid_1's auc: 0.829209	valid_1's binary_logloss: 0.139045
    [50]	valid_0's auc: 0.898546	valid_0's binary_logloss: 0.112978	valid_1's auc: 0.829064	valid_1's binary_logloss: 0.139115
    [51]	valid_0's auc: 0.898958	valid_0's binary_logloss: 0.112723	valid_1's auc: 0.82913	valid_1's binary_logloss: 0.139163
    [52]	valid_0's auc: 0.899296	valid_0's binary_logloss: 0.112541	valid_1's auc: 0.828532	valid_1's binary_logloss: 0.139358
    [53]	valid_0's auc: 0.899966	valid_0's binary_logloss: 0.112208	valid_1's auc: 0.828277	valid_1's binary_logloss: 0.139431
    [54]	valid_0's auc: 0.900225	valid_0's binary_logloss: 0.112001	valid_1's auc: 0.827866	valid_1's binary_logloss: 0.139573
    [55]	valid_0's auc: 0.90036	valid_0's binary_logloss: 0.111812	valid_1's auc: 0.827887	valid_1's binary_logloss: 0.139655
    [56]	valid_0's auc: 0.900528	valid_0's binary_logloss: 0.111638	valid_1's auc: 0.827809	valid_1's binary_logloss: 0.139702
    [57]	valid_0's auc: 0.900692	valid_0's binary_logloss: 0.111452	valid_1's auc: 0.827613	valid_1's binary_logloss: 0.139758
    [58]	valid_0's auc: 0.901275	valid_0's binary_logloss: 0.111262	valid_1's auc: 0.827848	valid_1's binary_logloss: 0.139758
    [59]	valid_0's auc: 0.90149	valid_0's binary_logloss: 0.111086	valid_1's auc: 0.827643	valid_1's binary_logloss: 0.139835
    [60]	valid_0's auc: 0.901669	valid_0's binary_logloss: 0.110873	valid_1's auc: 0.827187	valid_1's binary_logloss: 0.13998
    [61]	valid_0's auc: 0.901715	valid_0's binary_logloss: 0.110743	valid_1's auc: 0.826643	valid_1's binary_logloss: 0.140136
    Early stopping, best iteration is:
    [31]	valid_0's auc: 0.888552	valid_0's binary_logloss: 0.118399	valid_1's auc: 0.834036	valid_1's binary_logloss: 0.137459
    [1]	valid_0's auc: 0.828293	valid_0's binary_logloss: 0.156591	valid_1's auc: 0.803729	valid_1's binary_logloss: 0.159809
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.836848	valid_0's binary_logloss: 0.150955	valid_1's auc: 0.813468	valid_1's binary_logloss: 0.155406
    [3]	valid_0's auc: 0.842786	valid_0's binary_logloss: 0.146796	valid_1's auc: 0.817407	valid_1's binary_logloss: 0.151952
    [4]	valid_0's auc: 0.851348	valid_0's binary_logloss: 0.143395	valid_1's auc: 0.823949	valid_1's binary_logloss: 0.14934
    [5]	valid_0's auc: 0.855149	valid_0's binary_logloss: 0.140626	valid_1's auc: 0.827122	valid_1's binary_logloss: 0.147185
    [6]	valid_0's auc: 0.857679	valid_0's binary_logloss: 0.13837	valid_1's auc: 0.826758	valid_1's binary_logloss: 0.145708
    [7]	valid_0's auc: 0.86155	valid_0's binary_logloss: 0.136445	valid_1's auc: 0.828472	valid_1's binary_logloss: 0.144294
    [8]	valid_0's auc: 0.862588	valid_0's binary_logloss: 0.134808	valid_1's auc: 0.828623	valid_1's binary_logloss: 0.143251
    [9]	valid_0's auc: 0.865115	valid_0's binary_logloss: 0.133359	valid_1's auc: 0.831471	valid_1's binary_logloss: 0.142276
    [10]	valid_0's auc: 0.866894	valid_0's binary_logloss: 0.132073	valid_1's auc: 0.831985	valid_1's binary_logloss: 0.141408
    [11]	valid_0's auc: 0.869042	valid_0's binary_logloss: 0.130873	valid_1's auc: 0.834107	valid_1's binary_logloss: 0.140661
    [12]	valid_0's auc: 0.870884	valid_0's binary_logloss: 0.129808	valid_1's auc: 0.833841	valid_1's binary_logloss: 0.140137
    [13]	valid_0's auc: 0.87235	valid_0's binary_logloss: 0.128859	valid_1's auc: 0.834392	valid_1's binary_logloss: 0.139657
    [14]	valid_0's auc: 0.873638	valid_0's binary_logloss: 0.128012	valid_1's auc: 0.834569	valid_1's binary_logloss: 0.13928
    [15]	valid_0's auc: 0.874792	valid_0's binary_logloss: 0.127177	valid_1's auc: 0.83417	valid_1's binary_logloss: 0.138989
    [16]	valid_0's auc: 0.876361	valid_0's binary_logloss: 0.126304	valid_1's auc: 0.835986	valid_1's binary_logloss: 0.138606
    [17]	valid_0's auc: 0.878328	valid_0's binary_logloss: 0.125589	valid_1's auc: 0.836721	valid_1's binary_logloss: 0.138362
    [18]	valid_0's auc: 0.879835	valid_0's binary_logloss: 0.124908	valid_1's auc: 0.836975	valid_1's binary_logloss: 0.138199
    [19]	valid_0's auc: 0.881056	valid_0's binary_logloss: 0.124303	valid_1's auc: 0.837163	valid_1's binary_logloss: 0.137993
    [20]	valid_0's auc: 0.881684	valid_0's binary_logloss: 0.123706	valid_1's auc: 0.836767	valid_1's binary_logloss: 0.137922
    [21]	valid_0's auc: 0.882852	valid_0's binary_logloss: 0.123138	valid_1's auc: 0.835726	valid_1's binary_logloss: 0.137937
    [22]	valid_0's auc: 0.883782	valid_0's binary_logloss: 0.122621	valid_1's auc: 0.836033	valid_1's binary_logloss: 0.137832
    [23]	valid_0's auc: 0.884894	valid_0's binary_logloss: 0.122101	valid_1's auc: 0.835965	valid_1's binary_logloss: 0.137761
    [24]	valid_0's auc: 0.886129	valid_0's binary_logloss: 0.121539	valid_1's auc: 0.836276	valid_1's binary_logloss: 0.137662
    [25]	valid_0's auc: 0.887299	valid_0's binary_logloss: 0.121003	valid_1's auc: 0.836494	valid_1's binary_logloss: 0.137562
    [26]	valid_0's auc: 0.888167	valid_0's binary_logloss: 0.120533	valid_1's auc: 0.836735	valid_1's binary_logloss: 0.137471
    [27]	valid_0's auc: 0.889209	valid_0's binary_logloss: 0.120033	valid_1's auc: 0.837077	valid_1's binary_logloss: 0.137335
    [28]	valid_0's auc: 0.890083	valid_0's binary_logloss: 0.119628	valid_1's auc: 0.836646	valid_1's binary_logloss: 0.137396
    [29]	valid_0's auc: 0.891268	valid_0's binary_logloss: 0.1192	valid_1's auc: 0.836893	valid_1's binary_logloss: 0.137334
    [30]	valid_0's auc: 0.892026	valid_0's binary_logloss: 0.118779	valid_1's auc: 0.836987	valid_1's binary_logloss: 0.13734
    [31]	valid_0's auc: 0.892612	valid_0's binary_logloss: 0.1184	valid_1's auc: 0.835995	valid_1's binary_logloss: 0.137457
    [32]	valid_0's auc: 0.893409	valid_0's binary_logloss: 0.118028	valid_1's auc: 0.836329	valid_1's binary_logloss: 0.137426
    [33]	valid_0's auc: 0.894715	valid_0's binary_logloss: 0.117634	valid_1's auc: 0.836237	valid_1's binary_logloss: 0.137428
    [34]	valid_0's auc: 0.896129	valid_0's binary_logloss: 0.117239	valid_1's auc: 0.837278	valid_1's binary_logloss: 0.137234
    [35]	valid_0's auc: 0.897111	valid_0's binary_logloss: 0.116881	valid_1's auc: 0.837313	valid_1's binary_logloss: 0.137208
    [36]	valid_0's auc: 0.89791	valid_0's binary_logloss: 0.116525	valid_1's auc: 0.837094	valid_1's binary_logloss: 0.137227
    [37]	valid_0's auc: 0.898327	valid_0's binary_logloss: 0.116184	valid_1's auc: 0.836808	valid_1's binary_logloss: 0.137267
    [38]	valid_0's auc: 0.899162	valid_0's binary_logloss: 0.115859	valid_1's auc: 0.836473	valid_1's binary_logloss: 0.137323
    [39]	valid_0's auc: 0.899593	valid_0's binary_logloss: 0.115562	valid_1's auc: 0.836433	valid_1's binary_logloss: 0.137356
    [40]	valid_0's auc: 0.899942	valid_0's binary_logloss: 0.115297	valid_1's auc: 0.836299	valid_1's binary_logloss: 0.137382
    [41]	valid_0's auc: 0.900439	valid_0's binary_logloss: 0.114998	valid_1's auc: 0.835877	valid_1's binary_logloss: 0.137474
    [42]	valid_0's auc: 0.90094	valid_0's binary_logloss: 0.1147	valid_1's auc: 0.835016	valid_1's binary_logloss: 0.137628
    [43]	valid_0's auc: 0.901223	valid_0's binary_logloss: 0.11447	valid_1's auc: 0.834849	valid_1's binary_logloss: 0.137661
    [44]	valid_0's auc: 0.901438	valid_0's binary_logloss: 0.11423	valid_1's auc: 0.834743	valid_1's binary_logloss: 0.137699
    [45]	valid_0's auc: 0.902058	valid_0's binary_logloss: 0.113982	valid_1's auc: 0.834615	valid_1's binary_logloss: 0.13774
    [46]	valid_0's auc: 0.902431	valid_0's binary_logloss: 0.113713	valid_1's auc: 0.834207	valid_1's binary_logloss: 0.13784
    [47]	valid_0's auc: 0.902812	valid_0's binary_logloss: 0.113487	valid_1's auc: 0.833758	valid_1's binary_logloss: 0.137985
    [48]	valid_0's auc: 0.90306	valid_0's binary_logloss: 0.113301	valid_1's auc: 0.833219	valid_1's binary_logloss: 0.138122
    [49]	valid_0's auc: 0.903267	valid_0's binary_logloss: 0.113063	valid_1's auc: 0.832937	valid_1's binary_logloss: 0.138197
    [50]	valid_0's auc: 0.903705	valid_0's binary_logloss: 0.112863	valid_1's auc: 0.832815	valid_1's binary_logloss: 0.13824
    [51]	valid_0's auc: 0.904223	valid_0's binary_logloss: 0.112624	valid_1's auc: 0.832789	valid_1's binary_logloss: 0.138311
    [52]	valid_0's auc: 0.904867	valid_0's binary_logloss: 0.112372	valid_1's auc: 0.832399	valid_1's binary_logloss: 0.138416
    [53]	valid_0's auc: 0.90521	valid_0's binary_logloss: 0.112133	valid_1's auc: 0.832296	valid_1's binary_logloss: 0.1385
    [54]	valid_0's auc: 0.905689	valid_0's binary_logloss: 0.111851	valid_1's auc: 0.831911	valid_1's binary_logloss: 0.138563
    [55]	valid_0's auc: 0.905941	valid_0's binary_logloss: 0.111672	valid_1's auc: 0.831691	valid_1's binary_logloss: 0.138659
    [56]	valid_0's auc: 0.906184	valid_0's binary_logloss: 0.111482	valid_1's auc: 0.83155	valid_1's binary_logloss: 0.138723
    [57]	valid_0's auc: 0.906352	valid_0's binary_logloss: 0.111271	valid_1's auc: 0.830993	valid_1's binary_logloss: 0.138832
    [58]	valid_0's auc: 0.906874	valid_0's binary_logloss: 0.110997	valid_1's auc: 0.831151	valid_1's binary_logloss: 0.138806
    [59]	valid_0's auc: 0.906995	valid_0's binary_logloss: 0.110797	valid_1's auc: 0.830642	valid_1's binary_logloss: 0.138926
    [60]	valid_0's auc: 0.907239	valid_0's binary_logloss: 0.110639	valid_1's auc: 0.830434	valid_1's binary_logloss: 0.139017
    [61]	valid_0's auc: 0.907442	valid_0's binary_logloss: 0.110465	valid_1's auc: 0.829813	valid_1's binary_logloss: 0.13916
    [62]	valid_0's auc: 0.908053	valid_0's binary_logloss: 0.110189	valid_1's auc: 0.829969	valid_1's binary_logloss: 0.139161
    [63]	valid_0's auc: 0.908139	valid_0's binary_logloss: 0.110026	valid_1's auc: 0.829479	valid_1's binary_logloss: 0.139308
    [64]	valid_0's auc: 0.908183	valid_0's binary_logloss: 0.109892	valid_1's auc: 0.828768	valid_1's binary_logloss: 0.139496
    [65]	valid_0's auc: 0.908915	valid_0's binary_logloss: 0.109601	valid_1's auc: 0.828359	valid_1's binary_logloss: 0.139582
    Early stopping, best iteration is:
    [35]	valid_0's auc: 0.897111	valid_0's binary_logloss: 0.116881	valid_1's auc: 0.837313	valid_1's binary_logloss: 0.137208
    [1]	valid_0's auc: 0.833154	valid_0's binary_logloss: 0.156832	valid_1's auc: 0.809041	valid_1's binary_logloss: 0.159834
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.842696	valid_0's binary_logloss: 0.15097	valid_1's auc: 0.816072	valid_1's binary_logloss: 0.155054
    [3]	valid_0's auc: 0.848628	valid_0's binary_logloss: 0.146679	valid_1's auc: 0.821987	valid_1's binary_logloss: 0.151615
    [4]	valid_0's auc: 0.852189	valid_0's binary_logloss: 0.143409	valid_1's auc: 0.823257	valid_1's binary_logloss: 0.148971
    [5]	valid_0's auc: 0.85445	valid_0's binary_logloss: 0.140691	valid_1's auc: 0.826693	valid_1's binary_logloss: 0.146922
    [6]	valid_0's auc: 0.856663	valid_0's binary_logloss: 0.138477	valid_1's auc: 0.827879	valid_1's binary_logloss: 0.145076
    [7]	valid_0's auc: 0.858141	valid_0's binary_logloss: 0.136527	valid_1's auc: 0.828071	valid_1's binary_logloss: 0.143771
    [8]	valid_0's auc: 0.860436	valid_0's binary_logloss: 0.134845	valid_1's auc: 0.830521	valid_1's binary_logloss: 0.142677
    [9]	valid_0's auc: 0.861929	valid_0's binary_logloss: 0.133357	valid_1's auc: 0.831252	valid_1's binary_logloss: 0.14175
    [10]	valid_0's auc: 0.865268	valid_0's binary_logloss: 0.132002	valid_1's auc: 0.831065	valid_1's binary_logloss: 0.141032
    [11]	valid_0's auc: 0.86773	valid_0's binary_logloss: 0.130767	valid_1's auc: 0.831969	valid_1's binary_logloss: 0.140477
    [12]	valid_0's auc: 0.869797	valid_0's binary_logloss: 0.129639	valid_1's auc: 0.831747	valid_1's binary_logloss: 0.139964
    [13]	valid_0's auc: 0.871962	valid_0's binary_logloss: 0.128704	valid_1's auc: 0.832299	valid_1's binary_logloss: 0.13953
    [14]	valid_0's auc: 0.873656	valid_0's binary_logloss: 0.12785	valid_1's auc: 0.831874	valid_1's binary_logloss: 0.139277
    [15]	valid_0's auc: 0.875313	valid_0's binary_logloss: 0.127041	valid_1's auc: 0.831922	valid_1's binary_logloss: 0.139006
    [16]	valid_0's auc: 0.875813	valid_0's binary_logloss: 0.126291	valid_1's auc: 0.832339	valid_1's binary_logloss: 0.138679
    [17]	valid_0's auc: 0.877574	valid_0's binary_logloss: 0.125606	valid_1's auc: 0.832459	valid_1's binary_logloss: 0.138444
    [18]	valid_0's auc: 0.879134	valid_0's binary_logloss: 0.12489	valid_1's auc: 0.832409	valid_1's binary_logloss: 0.138268
    [19]	valid_0's auc: 0.880176	valid_0's binary_logloss: 0.124247	valid_1's auc: 0.832557	valid_1's binary_logloss: 0.138164
    [20]	valid_0's auc: 0.881608	valid_0's binary_logloss: 0.123609	valid_1's auc: 0.832053	valid_1's binary_logloss: 0.138136
    [21]	valid_0's auc: 0.882632	valid_0's binary_logloss: 0.123013	valid_1's auc: 0.831852	valid_1's binary_logloss: 0.138081
    [22]	valid_0's auc: 0.883684	valid_0's binary_logloss: 0.12245	valid_1's auc: 0.832034	valid_1's binary_logloss: 0.137978
    [23]	valid_0's auc: 0.885039	valid_0's binary_logloss: 0.121827	valid_1's auc: 0.831422	valid_1's binary_logloss: 0.138011
    [24]	valid_0's auc: 0.886247	valid_0's binary_logloss: 0.121291	valid_1's auc: 0.831253	valid_1's binary_logloss: 0.137942
    [25]	valid_0's auc: 0.887301	valid_0's binary_logloss: 0.120809	valid_1's auc: 0.831276	valid_1's binary_logloss: 0.137924
    [26]	valid_0's auc: 0.888621	valid_0's binary_logloss: 0.120287	valid_1's auc: 0.830735	valid_1's binary_logloss: 0.137975
    [27]	valid_0's auc: 0.889227	valid_0's binary_logloss: 0.119878	valid_1's auc: 0.830569	valid_1's binary_logloss: 0.137976
    [28]	valid_0's auc: 0.890381	valid_0's binary_logloss: 0.119445	valid_1's auc: 0.830281	valid_1's binary_logloss: 0.137968
    [29]	valid_0's auc: 0.891139	valid_0's binary_logloss: 0.119016	valid_1's auc: 0.830916	valid_1's binary_logloss: 0.137839
    [30]	valid_0's auc: 0.892417	valid_0's binary_logloss: 0.118535	valid_1's auc: 0.830189	valid_1's binary_logloss: 0.137909
    [31]	valid_0's auc: 0.893102	valid_0's binary_logloss: 0.118157	valid_1's auc: 0.830117	valid_1's binary_logloss: 0.137933
    [32]	valid_0's auc: 0.893844	valid_0's binary_logloss: 0.117764	valid_1's auc: 0.829877	valid_1's binary_logloss: 0.137962
    [33]	valid_0's auc: 0.894508	valid_0's binary_logloss: 0.117419	valid_1's auc: 0.829552	valid_1's binary_logloss: 0.138044
    [34]	valid_0's auc: 0.895157	valid_0's binary_logloss: 0.117051	valid_1's auc: 0.830486	valid_1's binary_logloss: 0.137893
    [35]	valid_0's auc: 0.89588	valid_0's binary_logloss: 0.116679	valid_1's auc: 0.83041	valid_1's binary_logloss: 0.137896
    [36]	valid_0's auc: 0.896611	valid_0's binary_logloss: 0.116295	valid_1's auc: 0.829969	valid_1's binary_logloss: 0.138026
    [37]	valid_0's auc: 0.89762	valid_0's binary_logloss: 0.115979	valid_1's auc: 0.829802	valid_1's binary_logloss: 0.138089
    [38]	valid_0's auc: 0.898028	valid_0's binary_logloss: 0.115667	valid_1's auc: 0.829346	valid_1's binary_logloss: 0.138207
    [39]	valid_0's auc: 0.89839	valid_0's binary_logloss: 0.115401	valid_1's auc: 0.829208	valid_1's binary_logloss: 0.138191
    [40]	valid_0's auc: 0.898968	valid_0's binary_logloss: 0.115042	valid_1's auc: 0.828576	valid_1's binary_logloss: 0.138317
    [41]	valid_0's auc: 0.899373	valid_0's binary_logloss: 0.114766	valid_1's auc: 0.828382	valid_1's binary_logloss: 0.138408
    [42]	valid_0's auc: 0.900042	valid_0's binary_logloss: 0.114467	valid_1's auc: 0.828023	valid_1's binary_logloss: 0.138489
    [43]	valid_0's auc: 0.900488	valid_0's binary_logloss: 0.114194	valid_1's auc: 0.827815	valid_1's binary_logloss: 0.138591
    [44]	valid_0's auc: 0.901004	valid_0's binary_logloss: 0.113865	valid_1's auc: 0.827321	valid_1's binary_logloss: 0.138736
    [45]	valid_0's auc: 0.901401	valid_0's binary_logloss: 0.113616	valid_1's auc: 0.826875	valid_1's binary_logloss: 0.138874
    [46]	valid_0's auc: 0.901795	valid_0's binary_logloss: 0.113348	valid_1's auc: 0.82643	valid_1's binary_logloss: 0.138955
    [47]	valid_0's auc: 0.902244	valid_0's binary_logloss: 0.11309	valid_1's auc: 0.825825	valid_1's binary_logloss: 0.13908
    [48]	valid_0's auc: 0.902756	valid_0's binary_logloss: 0.112891	valid_1's auc: 0.825639	valid_1's binary_logloss: 0.139117
    [49]	valid_0's auc: 0.903206	valid_0's binary_logloss: 0.112607	valid_1's auc: 0.825607	valid_1's binary_logloss: 0.139184
    Early stopping, best iteration is:
    [19]	valid_0's auc: 0.880176	valid_0's binary_logloss: 0.124247	valid_1's auc: 0.832557	valid_1's binary_logloss: 0.138164
    [1]	valid_0's auc: 0.825869	valid_0's binary_logloss: 0.156436	valid_1's auc: 0.803397	valid_1's binary_logloss: 0.159993
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.834474	valid_0's binary_logloss: 0.150597	valid_1's auc: 0.809972	valid_1's binary_logloss: 0.155122
    [3]	valid_0's auc: 0.840511	valid_0's binary_logloss: 0.146374	valid_1's auc: 0.819137	valid_1's binary_logloss: 0.15148
    [4]	valid_0's auc: 0.844283	valid_0's binary_logloss: 0.143027	valid_1's auc: 0.820472	valid_1's binary_logloss: 0.14904
    [5]	valid_0's auc: 0.85046	valid_0's binary_logloss: 0.140321	valid_1's auc: 0.82606	valid_1's binary_logloss: 0.146982
    [6]	valid_0's auc: 0.852881	valid_0's binary_logloss: 0.13813	valid_1's auc: 0.823869	valid_1's binary_logloss: 0.145517
    [7]	valid_0's auc: 0.857268	valid_0's binary_logloss: 0.136149	valid_1's auc: 0.827586	valid_1's binary_logloss: 0.144156
    [8]	valid_0's auc: 0.859401	valid_0's binary_logloss: 0.13452	valid_1's auc: 0.829428	valid_1's binary_logloss: 0.14303
    [9]	valid_0's auc: 0.861435	valid_0's binary_logloss: 0.133011	valid_1's auc: 0.82971	valid_1's binary_logloss: 0.142113
    [10]	valid_0's auc: 0.862282	valid_0's binary_logloss: 0.131742	valid_1's auc: 0.830527	valid_1's binary_logloss: 0.141392
    [11]	valid_0's auc: 0.86448	valid_0's binary_logloss: 0.130536	valid_1's auc: 0.83065	valid_1's binary_logloss: 0.140854
    [12]	valid_0's auc: 0.865909	valid_0's binary_logloss: 0.129495	valid_1's auc: 0.830763	valid_1's binary_logloss: 0.140425
    [13]	valid_0's auc: 0.867083	valid_0's binary_logloss: 0.128537	valid_1's auc: 0.830097	valid_1's binary_logloss: 0.140034
    [14]	valid_0's auc: 0.869164	valid_0's binary_logloss: 0.127593	valid_1's auc: 0.831242	valid_1's binary_logloss: 0.139589
    [15]	valid_0's auc: 0.869779	valid_0's binary_logloss: 0.126753	valid_1's auc: 0.830054	valid_1's binary_logloss: 0.13935
    [16]	valid_0's auc: 0.870798	valid_0's binary_logloss: 0.125998	valid_1's auc: 0.829854	valid_1's binary_logloss: 0.139147
    [17]	valid_0's auc: 0.872413	valid_0's binary_logloss: 0.125321	valid_1's auc: 0.829939	valid_1's binary_logloss: 0.138859
    [18]	valid_0's auc: 0.873656	valid_0's binary_logloss: 0.124646	valid_1's auc: 0.83033	valid_1's binary_logloss: 0.138678
    [19]	valid_0's auc: 0.874787	valid_0's binary_logloss: 0.124094	valid_1's auc: 0.829747	valid_1's binary_logloss: 0.138635
    [20]	valid_0's auc: 0.875793	valid_0's binary_logloss: 0.123547	valid_1's auc: 0.829455	valid_1's binary_logloss: 0.138492
    [21]	valid_0's auc: 0.87694	valid_0's binary_logloss: 0.122969	valid_1's auc: 0.830945	valid_1's binary_logloss: 0.13825
    [22]	valid_0's auc: 0.878632	valid_0's binary_logloss: 0.122407	valid_1's auc: 0.831483	valid_1's binary_logloss: 0.138097
    [23]	valid_0's auc: 0.8796	valid_0's binary_logloss: 0.121869	valid_1's auc: 0.831408	valid_1's binary_logloss: 0.138071
    [24]	valid_0's auc: 0.881298	valid_0's binary_logloss: 0.121363	valid_1's auc: 0.831984	valid_1's binary_logloss: 0.137912
    [25]	valid_0's auc: 0.882699	valid_0's binary_logloss: 0.120863	valid_1's auc: 0.831941	valid_1's binary_logloss: 0.137824
    [26]	valid_0's auc: 0.883799	valid_0's binary_logloss: 0.120379	valid_1's auc: 0.832897	valid_1's binary_logloss: 0.137649
    [27]	valid_0's auc: 0.884893	valid_0's binary_logloss: 0.119944	valid_1's auc: 0.832841	valid_1's binary_logloss: 0.13763
    [28]	valid_0's auc: 0.88588	valid_0's binary_logloss: 0.119545	valid_1's auc: 0.833413	valid_1's binary_logloss: 0.137495
    [29]	valid_0's auc: 0.886461	valid_0's binary_logloss: 0.11916	valid_1's auc: 0.833437	valid_1's binary_logloss: 0.137511
    [30]	valid_0's auc: 0.887592	valid_0's binary_logloss: 0.118739	valid_1's auc: 0.833429	valid_1's binary_logloss: 0.137514
    [31]	valid_0's auc: 0.888552	valid_0's binary_logloss: 0.118399	valid_1's auc: 0.834036	valid_1's binary_logloss: 0.137459
    [32]	valid_0's auc: 0.889448	valid_0's binary_logloss: 0.118019	valid_1's auc: 0.833754	valid_1's binary_logloss: 0.137498
    [33]	valid_0's auc: 0.890203	valid_0's binary_logloss: 0.117711	valid_1's auc: 0.833853	valid_1's binary_logloss: 0.137491
    [34]	valid_0's auc: 0.890868	valid_0's binary_logloss: 0.117395	valid_1's auc: 0.833648	valid_1's binary_logloss: 0.137532
    [35]	valid_0's auc: 0.891378	valid_0's binary_logloss: 0.117106	valid_1's auc: 0.833479	valid_1's binary_logloss: 0.137577
    [36]	valid_0's auc: 0.892061	valid_0's binary_logloss: 0.116741	valid_1's auc: 0.832608	valid_1's binary_logloss: 0.137799
    [37]	valid_0's auc: 0.892715	valid_0's binary_logloss: 0.116403	valid_1's auc: 0.83208	valid_1's binary_logloss: 0.137938
    [38]	valid_0's auc: 0.893146	valid_0's binary_logloss: 0.116126	valid_1's auc: 0.83196	valid_1's binary_logloss: 0.137994
    [39]	valid_0's auc: 0.893963	valid_0's binary_logloss: 0.115783	valid_1's auc: 0.831584	valid_1's binary_logloss: 0.138076
    [40]	valid_0's auc: 0.894274	valid_0's binary_logloss: 0.115535	valid_1's auc: 0.831346	valid_1's binary_logloss: 0.13818
    [41]	valid_0's auc: 0.894809	valid_0's binary_logloss: 0.115247	valid_1's auc: 0.831197	valid_1's binary_logloss: 0.138289
    [42]	valid_0's auc: 0.895332	valid_0's binary_logloss: 0.114916	valid_1's auc: 0.830893	valid_1's binary_logloss: 0.138343
    [43]	valid_0's auc: 0.895864	valid_0's binary_logloss: 0.114701	valid_1's auc: 0.83129	valid_1's binary_logloss: 0.138307
    [44]	valid_0's auc: 0.89623	valid_0's binary_logloss: 0.114411	valid_1's auc: 0.831004	valid_1's binary_logloss: 0.138408
    [45]	valid_0's auc: 0.896616	valid_0's binary_logloss: 0.11413	valid_1's auc: 0.830369	valid_1's binary_logloss: 0.138547
    [46]	valid_0's auc: 0.897076	valid_0's binary_logloss: 0.113919	valid_1's auc: 0.830246	valid_1's binary_logloss: 0.138644
    [47]	valid_0's auc: 0.897393	valid_0's binary_logloss: 0.113691	valid_1's auc: 0.829999	valid_1's binary_logloss: 0.138769
    [48]	valid_0's auc: 0.897745	valid_0's binary_logloss: 0.113465	valid_1's auc: 0.829593	valid_1's binary_logloss: 0.138908
    [49]	valid_0's auc: 0.898203	valid_0's binary_logloss: 0.113214	valid_1's auc: 0.829209	valid_1's binary_logloss: 0.139045
    [50]	valid_0's auc: 0.898546	valid_0's binary_logloss: 0.112978	valid_1's auc: 0.829064	valid_1's binary_logloss: 0.139115
    [51]	valid_0's auc: 0.898958	valid_0's binary_logloss: 0.112723	valid_1's auc: 0.82913	valid_1's binary_logloss: 0.139163
    [52]	valid_0's auc: 0.899296	valid_0's binary_logloss: 0.112541	valid_1's auc: 0.828532	valid_1's binary_logloss: 0.139358
    [53]	valid_0's auc: 0.899966	valid_0's binary_logloss: 0.112208	valid_1's auc: 0.828277	valid_1's binary_logloss: 0.139431
    [54]	valid_0's auc: 0.900225	valid_0's binary_logloss: 0.112001	valid_1's auc: 0.827866	valid_1's binary_logloss: 0.139573
    [55]	valid_0's auc: 0.90036	valid_0's binary_logloss: 0.111812	valid_1's auc: 0.827887	valid_1's binary_logloss: 0.139655
    [56]	valid_0's auc: 0.900528	valid_0's binary_logloss: 0.111638	valid_1's auc: 0.827809	valid_1's binary_logloss: 0.139702
    [57]	valid_0's auc: 0.900692	valid_0's binary_logloss: 0.111452	valid_1's auc: 0.827613	valid_1's binary_logloss: 0.139758
    [58]	valid_0's auc: 0.901275	valid_0's binary_logloss: 0.111262	valid_1's auc: 0.827848	valid_1's binary_logloss: 0.139758
    [59]	valid_0's auc: 0.90149	valid_0's binary_logloss: 0.111086	valid_1's auc: 0.827643	valid_1's binary_logloss: 0.139835
    [60]	valid_0's auc: 0.901669	valid_0's binary_logloss: 0.110873	valid_1's auc: 0.827187	valid_1's binary_logloss: 0.13998
    [61]	valid_0's auc: 0.901715	valid_0's binary_logloss: 0.110743	valid_1's auc: 0.826643	valid_1's binary_logloss: 0.140136
    Early stopping, best iteration is:
    [31]	valid_0's auc: 0.888552	valid_0's binary_logloss: 0.118399	valid_1's auc: 0.834036	valid_1's binary_logloss: 0.137459
    [1]	valid_0's auc: 0.828293	valid_0's binary_logloss: 0.156591	valid_1's auc: 0.803729	valid_1's binary_logloss: 0.159809
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.836848	valid_0's binary_logloss: 0.150955	valid_1's auc: 0.813468	valid_1's binary_logloss: 0.155406
    [3]	valid_0's auc: 0.842786	valid_0's binary_logloss: 0.146796	valid_1's auc: 0.817407	valid_1's binary_logloss: 0.151952
    [4]	valid_0's auc: 0.851348	valid_0's binary_logloss: 0.143395	valid_1's auc: 0.823949	valid_1's binary_logloss: 0.14934
    [5]	valid_0's auc: 0.855149	valid_0's binary_logloss: 0.140626	valid_1's auc: 0.827122	valid_1's binary_logloss: 0.147185
    [6]	valid_0's auc: 0.857679	valid_0's binary_logloss: 0.13837	valid_1's auc: 0.826758	valid_1's binary_logloss: 0.145708
    [7]	valid_0's auc: 0.86155	valid_0's binary_logloss: 0.136445	valid_1's auc: 0.828472	valid_1's binary_logloss: 0.144294
    [8]	valid_0's auc: 0.862588	valid_0's binary_logloss: 0.134808	valid_1's auc: 0.828623	valid_1's binary_logloss: 0.143251
    [9]	valid_0's auc: 0.865115	valid_0's binary_logloss: 0.133359	valid_1's auc: 0.831471	valid_1's binary_logloss: 0.142276
    [10]	valid_0's auc: 0.866894	valid_0's binary_logloss: 0.132073	valid_1's auc: 0.831985	valid_1's binary_logloss: 0.141408
    [11]	valid_0's auc: 0.869042	valid_0's binary_logloss: 0.130873	valid_1's auc: 0.834107	valid_1's binary_logloss: 0.140661
    [12]	valid_0's auc: 0.870884	valid_0's binary_logloss: 0.129808	valid_1's auc: 0.833841	valid_1's binary_logloss: 0.140137
    [13]	valid_0's auc: 0.87235	valid_0's binary_logloss: 0.128859	valid_1's auc: 0.834392	valid_1's binary_logloss: 0.139657
    [14]	valid_0's auc: 0.873638	valid_0's binary_logloss: 0.128012	valid_1's auc: 0.834569	valid_1's binary_logloss: 0.13928
    [15]	valid_0's auc: 0.874792	valid_0's binary_logloss: 0.127177	valid_1's auc: 0.83417	valid_1's binary_logloss: 0.138989
    [16]	valid_0's auc: 0.876361	valid_0's binary_logloss: 0.126304	valid_1's auc: 0.835986	valid_1's binary_logloss: 0.138606
    [17]	valid_0's auc: 0.878328	valid_0's binary_logloss: 0.125589	valid_1's auc: 0.836721	valid_1's binary_logloss: 0.138362
    [18]	valid_0's auc: 0.879835	valid_0's binary_logloss: 0.124908	valid_1's auc: 0.836975	valid_1's binary_logloss: 0.138199
    [19]	valid_0's auc: 0.881056	valid_0's binary_logloss: 0.124303	valid_1's auc: 0.837163	valid_1's binary_logloss: 0.137993
    [20]	valid_0's auc: 0.881684	valid_0's binary_logloss: 0.123706	valid_1's auc: 0.836767	valid_1's binary_logloss: 0.137922
    [21]	valid_0's auc: 0.882852	valid_0's binary_logloss: 0.123138	valid_1's auc: 0.835726	valid_1's binary_logloss: 0.137937
    [22]	valid_0's auc: 0.883782	valid_0's binary_logloss: 0.122621	valid_1's auc: 0.836033	valid_1's binary_logloss: 0.137832
    [23]	valid_0's auc: 0.884894	valid_0's binary_logloss: 0.122101	valid_1's auc: 0.835965	valid_1's binary_logloss: 0.137761
    [24]	valid_0's auc: 0.886129	valid_0's binary_logloss: 0.121539	valid_1's auc: 0.836276	valid_1's binary_logloss: 0.137662
    [25]	valid_0's auc: 0.887299	valid_0's binary_logloss: 0.121003	valid_1's auc: 0.836494	valid_1's binary_logloss: 0.137562
    [26]	valid_0's auc: 0.888167	valid_0's binary_logloss: 0.120533	valid_1's auc: 0.836735	valid_1's binary_logloss: 0.137471
    [27]	valid_0's auc: 0.889209	valid_0's binary_logloss: 0.120033	valid_1's auc: 0.837077	valid_1's binary_logloss: 0.137335
    [28]	valid_0's auc: 0.890083	valid_0's binary_logloss: 0.119628	valid_1's auc: 0.836646	valid_1's binary_logloss: 0.137396
    [29]	valid_0's auc: 0.891268	valid_0's binary_logloss: 0.1192	valid_1's auc: 0.836893	valid_1's binary_logloss: 0.137334
    [30]	valid_0's auc: 0.892026	valid_0's binary_logloss: 0.118779	valid_1's auc: 0.836987	valid_1's binary_logloss: 0.13734
    [31]	valid_0's auc: 0.892612	valid_0's binary_logloss: 0.1184	valid_1's auc: 0.835995	valid_1's binary_logloss: 0.137457
    [32]	valid_0's auc: 0.893409	valid_0's binary_logloss: 0.118028	valid_1's auc: 0.836329	valid_1's binary_logloss: 0.137426
    [33]	valid_0's auc: 0.894715	valid_0's binary_logloss: 0.117634	valid_1's auc: 0.836237	valid_1's binary_logloss: 0.137428
    [34]	valid_0's auc: 0.896129	valid_0's binary_logloss: 0.117239	valid_1's auc: 0.837278	valid_1's binary_logloss: 0.137234
    [35]	valid_0's auc: 0.897111	valid_0's binary_logloss: 0.116881	valid_1's auc: 0.837313	valid_1's binary_logloss: 0.137208
    [36]	valid_0's auc: 0.89791	valid_0's binary_logloss: 0.116525	valid_1's auc: 0.837094	valid_1's binary_logloss: 0.137227
    [37]	valid_0's auc: 0.898327	valid_0's binary_logloss: 0.116184	valid_1's auc: 0.836808	valid_1's binary_logloss: 0.137267
    [38]	valid_0's auc: 0.899162	valid_0's binary_logloss: 0.115859	valid_1's auc: 0.836473	valid_1's binary_logloss: 0.137323
    [39]	valid_0's auc: 0.899593	valid_0's binary_logloss: 0.115562	valid_1's auc: 0.836433	valid_1's binary_logloss: 0.137356
    [40]	valid_0's auc: 0.899942	valid_0's binary_logloss: 0.115297	valid_1's auc: 0.836299	valid_1's binary_logloss: 0.137382
    [41]	valid_0's auc: 0.900439	valid_0's binary_logloss: 0.114998	valid_1's auc: 0.835877	valid_1's binary_logloss: 0.137474
    [42]	valid_0's auc: 0.90094	valid_0's binary_logloss: 0.1147	valid_1's auc: 0.835016	valid_1's binary_logloss: 0.137628
    [43]	valid_0's auc: 0.901223	valid_0's binary_logloss: 0.11447	valid_1's auc: 0.834849	valid_1's binary_logloss: 0.137661
    [44]	valid_0's auc: 0.901438	valid_0's binary_logloss: 0.11423	valid_1's auc: 0.834743	valid_1's binary_logloss: 0.137699
    [45]	valid_0's auc: 0.902058	valid_0's binary_logloss: 0.113982	valid_1's auc: 0.834615	valid_1's binary_logloss: 0.13774
    [46]	valid_0's auc: 0.902431	valid_0's binary_logloss: 0.113713	valid_1's auc: 0.834207	valid_1's binary_logloss: 0.13784
    [47]	valid_0's auc: 0.902812	valid_0's binary_logloss: 0.113487	valid_1's auc: 0.833758	valid_1's binary_logloss: 0.137985
    [48]	valid_0's auc: 0.90306	valid_0's binary_logloss: 0.113301	valid_1's auc: 0.833219	valid_1's binary_logloss: 0.138122
    [49]	valid_0's auc: 0.903267	valid_0's binary_logloss: 0.113063	valid_1's auc: 0.832937	valid_1's binary_logloss: 0.138197
    [50]	valid_0's auc: 0.903705	valid_0's binary_logloss: 0.112863	valid_1's auc: 0.832815	valid_1's binary_logloss: 0.13824
    [51]	valid_0's auc: 0.904223	valid_0's binary_logloss: 0.112624	valid_1's auc: 0.832789	valid_1's binary_logloss: 0.138311
    [52]	valid_0's auc: 0.904867	valid_0's binary_logloss: 0.112372	valid_1's auc: 0.832399	valid_1's binary_logloss: 0.138416
    [53]	valid_0's auc: 0.90521	valid_0's binary_logloss: 0.112133	valid_1's auc: 0.832296	valid_1's binary_logloss: 0.1385
    [54]	valid_0's auc: 0.905689	valid_0's binary_logloss: 0.111851	valid_1's auc: 0.831911	valid_1's binary_logloss: 0.138563
    [55]	valid_0's auc: 0.905941	valid_0's binary_logloss: 0.111672	valid_1's auc: 0.831691	valid_1's binary_logloss: 0.138659
    [56]	valid_0's auc: 0.906184	valid_0's binary_logloss: 0.111482	valid_1's auc: 0.83155	valid_1's binary_logloss: 0.138723
    [57]	valid_0's auc: 0.906352	valid_0's binary_logloss: 0.111271	valid_1's auc: 0.830993	valid_1's binary_logloss: 0.138832
    [58]	valid_0's auc: 0.906874	valid_0's binary_logloss: 0.110997	valid_1's auc: 0.831151	valid_1's binary_logloss: 0.138806
    [59]	valid_0's auc: 0.906995	valid_0's binary_logloss: 0.110797	valid_1's auc: 0.830642	valid_1's binary_logloss: 0.138926
    [60]	valid_0's auc: 0.907239	valid_0's binary_logloss: 0.110639	valid_1's auc: 0.830434	valid_1's binary_logloss: 0.139017
    [61]	valid_0's auc: 0.907442	valid_0's binary_logloss: 0.110465	valid_1's auc: 0.829813	valid_1's binary_logloss: 0.13916
    [62]	valid_0's auc: 0.908053	valid_0's binary_logloss: 0.110189	valid_1's auc: 0.829969	valid_1's binary_logloss: 0.139161
    [63]	valid_0's auc: 0.908139	valid_0's binary_logloss: 0.110026	valid_1's auc: 0.829479	valid_1's binary_logloss: 0.139308
    [64]	valid_0's auc: 0.908183	valid_0's binary_logloss: 0.109892	valid_1's auc: 0.828768	valid_1's binary_logloss: 0.139496
    [65]	valid_0's auc: 0.908915	valid_0's binary_logloss: 0.109601	valid_1's auc: 0.828359	valid_1's binary_logloss: 0.139582
    Early stopping, best iteration is:
    [35]	valid_0's auc: 0.897111	valid_0's binary_logloss: 0.116881	valid_1's auc: 0.837313	valid_1's binary_logloss: 0.137208
    [1]	valid_0's auc: 0.833154	valid_0's binary_logloss: 0.156832	valid_1's auc: 0.809041	valid_1's binary_logloss: 0.159834
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.842696	valid_0's binary_logloss: 0.15097	valid_1's auc: 0.816072	valid_1's binary_logloss: 0.155054
    [3]	valid_0's auc: 0.848628	valid_0's binary_logloss: 0.146679	valid_1's auc: 0.821987	valid_1's binary_logloss: 0.151615
    [4]	valid_0's auc: 0.852189	valid_0's binary_logloss: 0.143409	valid_1's auc: 0.823257	valid_1's binary_logloss: 0.148971
    [5]	valid_0's auc: 0.85445	valid_0's binary_logloss: 0.140691	valid_1's auc: 0.826693	valid_1's binary_logloss: 0.146922
    [6]	valid_0's auc: 0.856663	valid_0's binary_logloss: 0.138477	valid_1's auc: 0.827879	valid_1's binary_logloss: 0.145076
    [7]	valid_0's auc: 0.858141	valid_0's binary_logloss: 0.136527	valid_1's auc: 0.828071	valid_1's binary_logloss: 0.143771
    [8]	valid_0's auc: 0.860436	valid_0's binary_logloss: 0.134845	valid_1's auc: 0.830521	valid_1's binary_logloss: 0.142677
    [9]	valid_0's auc: 0.861929	valid_0's binary_logloss: 0.133357	valid_1's auc: 0.831252	valid_1's binary_logloss: 0.14175
    [10]	valid_0's auc: 0.865268	valid_0's binary_logloss: 0.132002	valid_1's auc: 0.831065	valid_1's binary_logloss: 0.141032
    [11]	valid_0's auc: 0.86773	valid_0's binary_logloss: 0.130767	valid_1's auc: 0.831969	valid_1's binary_logloss: 0.140477
    [12]	valid_0's auc: 0.869797	valid_0's binary_logloss: 0.129639	valid_1's auc: 0.831747	valid_1's binary_logloss: 0.139964
    [13]	valid_0's auc: 0.871962	valid_0's binary_logloss: 0.128704	valid_1's auc: 0.832299	valid_1's binary_logloss: 0.13953
    [14]	valid_0's auc: 0.873656	valid_0's binary_logloss: 0.12785	valid_1's auc: 0.831874	valid_1's binary_logloss: 0.139277
    [15]	valid_0's auc: 0.875313	valid_0's binary_logloss: 0.127041	valid_1's auc: 0.831922	valid_1's binary_logloss: 0.139006
    [16]	valid_0's auc: 0.875813	valid_0's binary_logloss: 0.126291	valid_1's auc: 0.832339	valid_1's binary_logloss: 0.138679
    [17]	valid_0's auc: 0.877574	valid_0's binary_logloss: 0.125606	valid_1's auc: 0.832459	valid_1's binary_logloss: 0.138444
    [18]	valid_0's auc: 0.879134	valid_0's binary_logloss: 0.12489	valid_1's auc: 0.832409	valid_1's binary_logloss: 0.138268
    [19]	valid_0's auc: 0.880176	valid_0's binary_logloss: 0.124247	valid_1's auc: 0.832557	valid_1's binary_logloss: 0.138164
    [20]	valid_0's auc: 0.881608	valid_0's binary_logloss: 0.123609	valid_1's auc: 0.832053	valid_1's binary_logloss: 0.138136
    [21]	valid_0's auc: 0.882632	valid_0's binary_logloss: 0.123013	valid_1's auc: 0.831852	valid_1's binary_logloss: 0.138081
    [22]	valid_0's auc: 0.883684	valid_0's binary_logloss: 0.12245	valid_1's auc: 0.832034	valid_1's binary_logloss: 0.137978
    [23]	valid_0's auc: 0.885039	valid_0's binary_logloss: 0.121827	valid_1's auc: 0.831422	valid_1's binary_logloss: 0.138011
    [24]	valid_0's auc: 0.886247	valid_0's binary_logloss: 0.121291	valid_1's auc: 0.831253	valid_1's binary_logloss: 0.137942
    [25]	valid_0's auc: 0.887301	valid_0's binary_logloss: 0.120809	valid_1's auc: 0.831276	valid_1's binary_logloss: 0.137924
    [26]	valid_0's auc: 0.888621	valid_0's binary_logloss: 0.120287	valid_1's auc: 0.830735	valid_1's binary_logloss: 0.137975
    [27]	valid_0's auc: 0.889227	valid_0's binary_logloss: 0.119878	valid_1's auc: 0.830569	valid_1's binary_logloss: 0.137976
    [28]	valid_0's auc: 0.890381	valid_0's binary_logloss: 0.119445	valid_1's auc: 0.830281	valid_1's binary_logloss: 0.137968
    [29]	valid_0's auc: 0.891139	valid_0's binary_logloss: 0.119016	valid_1's auc: 0.830916	valid_1's binary_logloss: 0.137839
    [30]	valid_0's auc: 0.892417	valid_0's binary_logloss: 0.118535	valid_1's auc: 0.830189	valid_1's binary_logloss: 0.137909
    [31]	valid_0's auc: 0.893102	valid_0's binary_logloss: 0.118157	valid_1's auc: 0.830117	valid_1's binary_logloss: 0.137933
    [32]	valid_0's auc: 0.893844	valid_0's binary_logloss: 0.117764	valid_1's auc: 0.829877	valid_1's binary_logloss: 0.137962
    [33]	valid_0's auc: 0.894508	valid_0's binary_logloss: 0.117419	valid_1's auc: 0.829552	valid_1's binary_logloss: 0.138044
    [34]	valid_0's auc: 0.895157	valid_0's binary_logloss: 0.117051	valid_1's auc: 0.830486	valid_1's binary_logloss: 0.137893
    [35]	valid_0's auc: 0.89588	valid_0's binary_logloss: 0.116679	valid_1's auc: 0.83041	valid_1's binary_logloss: 0.137896
    [36]	valid_0's auc: 0.896611	valid_0's binary_logloss: 0.116295	valid_1's auc: 0.829969	valid_1's binary_logloss: 0.138026
    [37]	valid_0's auc: 0.89762	valid_0's binary_logloss: 0.115979	valid_1's auc: 0.829802	valid_1's binary_logloss: 0.138089
    [38]	valid_0's auc: 0.898028	valid_0's binary_logloss: 0.115667	valid_1's auc: 0.829346	valid_1's binary_logloss: 0.138207
    [39]	valid_0's auc: 0.89839	valid_0's binary_logloss: 0.115401	valid_1's auc: 0.829208	valid_1's binary_logloss: 0.138191
    [40]	valid_0's auc: 0.898968	valid_0's binary_logloss: 0.115042	valid_1's auc: 0.828576	valid_1's binary_logloss: 0.138317
    [41]	valid_0's auc: 0.899373	valid_0's binary_logloss: 0.114766	valid_1's auc: 0.828382	valid_1's binary_logloss: 0.138408
    [42]	valid_0's auc: 0.900042	valid_0's binary_logloss: 0.114467	valid_1's auc: 0.828023	valid_1's binary_logloss: 0.138489
    [43]	valid_0's auc: 0.900488	valid_0's binary_logloss: 0.114194	valid_1's auc: 0.827815	valid_1's binary_logloss: 0.138591
    [44]	valid_0's auc: 0.901004	valid_0's binary_logloss: 0.113865	valid_1's auc: 0.827321	valid_1's binary_logloss: 0.138736
    [45]	valid_0's auc: 0.901401	valid_0's binary_logloss: 0.113616	valid_1's auc: 0.826875	valid_1's binary_logloss: 0.138874
    [46]	valid_0's auc: 0.901795	valid_0's binary_logloss: 0.113348	valid_1's auc: 0.82643	valid_1's binary_logloss: 0.138955
    [47]	valid_0's auc: 0.902244	valid_0's binary_logloss: 0.11309	valid_1's auc: 0.825825	valid_1's binary_logloss: 0.13908
    [48]	valid_0's auc: 0.902756	valid_0's binary_logloss: 0.112891	valid_1's auc: 0.825639	valid_1's binary_logloss: 0.139117
    [49]	valid_0's auc: 0.903206	valid_0's binary_logloss: 0.112607	valid_1's auc: 0.825607	valid_1's binary_logloss: 0.139184
    Early stopping, best iteration is:
    [19]	valid_0's auc: 0.880176	valid_0's binary_logloss: 0.124247	valid_1's auc: 0.832557	valid_1's binary_logloss: 0.138164
    [1]	valid_0's auc: 0.822874	valid_0's binary_logloss: 0.157449	valid_1's auc: 0.804936	valid_1's binary_logloss: 0.160126
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.827064	valid_0's binary_logloss: 0.152115	valid_1's auc: 0.810262	valid_1's binary_logloss: 0.155403
    [3]	valid_0's auc: 0.830773	valid_0's binary_logloss: 0.148229	valid_1's auc: 0.814099	valid_1's binary_logloss: 0.151956
    [4]	valid_0's auc: 0.834585	valid_0's binary_logloss: 0.145242	valid_1's auc: 0.815291	valid_1's binary_logloss: 0.149379
    [5]	valid_0's auc: 0.839767	valid_0's binary_logloss: 0.142798	valid_1's auc: 0.819992	valid_1's binary_logloss: 0.147278
    [6]	valid_0's auc: 0.845277	valid_0's binary_logloss: 0.140696	valid_1's auc: 0.824997	valid_1's binary_logloss: 0.145614
    [7]	valid_0's auc: 0.84706	valid_0's binary_logloss: 0.139021	valid_1's auc: 0.827429	valid_1's binary_logloss: 0.144095
    [8]	valid_0's auc: 0.847609	valid_0's binary_logloss: 0.137638	valid_1's auc: 0.827987	valid_1's binary_logloss: 0.143002
    [9]	valid_0's auc: 0.848913	valid_0's binary_logloss: 0.136364	valid_1's auc: 0.82986	valid_1's binary_logloss: 0.141984
    [10]	valid_0's auc: 0.850178	valid_0's binary_logloss: 0.135236	valid_1's auc: 0.829651	valid_1's binary_logloss: 0.141228
    [11]	valid_0's auc: 0.85127	valid_0's binary_logloss: 0.134273	valid_1's auc: 0.830496	valid_1's binary_logloss: 0.140509
    [12]	valid_0's auc: 0.852586	valid_0's binary_logloss: 0.133411	valid_1's auc: 0.830525	valid_1's binary_logloss: 0.13997
    [13]	valid_0's auc: 0.853732	valid_0's binary_logloss: 0.132666	valid_1's auc: 0.829319	valid_1's binary_logloss: 0.13957
    [14]	valid_0's auc: 0.854775	valid_0's binary_logloss: 0.13201	valid_1's auc: 0.832116	valid_1's binary_logloss: 0.139106
    [15]	valid_0's auc: 0.855714	valid_0's binary_logloss: 0.131409	valid_1's auc: 0.833454	valid_1's binary_logloss: 0.138684
    [16]	valid_0's auc: 0.856099	valid_0's binary_logloss: 0.130853	valid_1's auc: 0.832243	valid_1's binary_logloss: 0.138485
    [17]	valid_0's auc: 0.857475	valid_0's binary_logloss: 0.130295	valid_1's auc: 0.832577	valid_1's binary_logloss: 0.138179
    [18]	valid_0's auc: 0.857908	valid_0's binary_logloss: 0.129866	valid_1's auc: 0.832652	valid_1's binary_logloss: 0.138084
    [19]	valid_0's auc: 0.859226	valid_0's binary_logloss: 0.129384	valid_1's auc: 0.832887	valid_1's binary_logloss: 0.137828
    [20]	valid_0's auc: 0.860296	valid_0's binary_logloss: 0.129003	valid_1's auc: 0.834111	valid_1's binary_logloss: 0.137643
    [21]	valid_0's auc: 0.860934	valid_0's binary_logloss: 0.128619	valid_1's auc: 0.834602	valid_1's binary_logloss: 0.137471
    [22]	valid_0's auc: 0.861712	valid_0's binary_logloss: 0.12825	valid_1's auc: 0.834828	valid_1's binary_logloss: 0.137291
    [23]	valid_0's auc: 0.862939	valid_0's binary_logloss: 0.127869	valid_1's auc: 0.835767	valid_1's binary_logloss: 0.137067
    [24]	valid_0's auc: 0.863968	valid_0's binary_logloss: 0.127525	valid_1's auc: 0.835485	valid_1's binary_logloss: 0.13701
    [25]	valid_0's auc: 0.865029	valid_0's binary_logloss: 0.127234	valid_1's auc: 0.835507	valid_1's binary_logloss: 0.136911
    [26]	valid_0's auc: 0.865758	valid_0's binary_logloss: 0.126952	valid_1's auc: 0.835743	valid_1's binary_logloss: 0.136823
    [27]	valid_0's auc: 0.866244	valid_0's binary_logloss: 0.126688	valid_1's auc: 0.836049	valid_1's binary_logloss: 0.136736
    [28]	valid_0's auc: 0.866607	valid_0's binary_logloss: 0.126455	valid_1's auc: 0.836043	valid_1's binary_logloss: 0.136687
    [29]	valid_0's auc: 0.867322	valid_0's binary_logloss: 0.126255	valid_1's auc: 0.836375	valid_1's binary_logloss: 0.136596
    [30]	valid_0's auc: 0.867997	valid_0's binary_logloss: 0.126022	valid_1's auc: 0.836518	valid_1's binary_logloss: 0.136563
    [31]	valid_0's auc: 0.868888	valid_0's binary_logloss: 0.125781	valid_1's auc: 0.836949	valid_1's binary_logloss: 0.13648
    [32]	valid_0's auc: 0.869544	valid_0's binary_logloss: 0.125524	valid_1's auc: 0.836774	valid_1's binary_logloss: 0.136506
    [33]	valid_0's auc: 0.870314	valid_0's binary_logloss: 0.125324	valid_1's auc: 0.837039	valid_1's binary_logloss: 0.13642
    [34]	valid_0's auc: 0.871207	valid_0's binary_logloss: 0.125096	valid_1's auc: 0.836859	valid_1's binary_logloss: 0.136441
    [35]	valid_0's auc: 0.871788	valid_0's binary_logloss: 0.124883	valid_1's auc: 0.836641	valid_1's binary_logloss: 0.136467
    [36]	valid_0's auc: 0.872403	valid_0's binary_logloss: 0.124635	valid_1's auc: 0.836917	valid_1's binary_logloss: 0.13642
    [37]	valid_0's auc: 0.872914	valid_0's binary_logloss: 0.124439	valid_1's auc: 0.836892	valid_1's binary_logloss: 0.136489
    [38]	valid_0's auc: 0.873469	valid_0's binary_logloss: 0.124233	valid_1's auc: 0.836816	valid_1's binary_logloss: 0.136502
    [39]	valid_0's auc: 0.873971	valid_0's binary_logloss: 0.124017	valid_1's auc: 0.83659	valid_1's binary_logloss: 0.136543
    [40]	valid_0's auc: 0.875013	valid_0's binary_logloss: 0.123715	valid_1's auc: 0.836381	valid_1's binary_logloss: 0.136593
    [41]	valid_0's auc: 0.875341	valid_0's binary_logloss: 0.123538	valid_1's auc: 0.8363	valid_1's binary_logloss: 0.136624
    [42]	valid_0's auc: 0.875886	valid_0's binary_logloss: 0.123285	valid_1's auc: 0.836043	valid_1's binary_logloss: 0.136665
    [43]	valid_0's auc: 0.87633	valid_0's binary_logloss: 0.12311	valid_1's auc: 0.836018	valid_1's binary_logloss: 0.136647
    [44]	valid_0's auc: 0.876816	valid_0's binary_logloss: 0.122878	valid_1's auc: 0.836061	valid_1's binary_logloss: 0.136664
    [45]	valid_0's auc: 0.877116	valid_0's binary_logloss: 0.12271	valid_1's auc: 0.836517	valid_1's binary_logloss: 0.136608
    [46]	valid_0's auc: 0.877567	valid_0's binary_logloss: 0.122536	valid_1's auc: 0.836051	valid_1's binary_logloss: 0.13667
    [47]	valid_0's auc: 0.878192	valid_0's binary_logloss: 0.122332	valid_1's auc: 0.835827	valid_1's binary_logloss: 0.136694
    [48]	valid_0's auc: 0.878371	valid_0's binary_logloss: 0.122208	valid_1's auc: 0.835831	valid_1's binary_logloss: 0.136717
    [49]	valid_0's auc: 0.878669	valid_0's binary_logloss: 0.12204	valid_1's auc: 0.835913	valid_1's binary_logloss: 0.136679
    [50]	valid_0's auc: 0.879288	valid_0's binary_logloss: 0.121825	valid_1's auc: 0.835468	valid_1's binary_logloss: 0.136734
    [51]	valid_0's auc: 0.879552	valid_0's binary_logloss: 0.121647	valid_1's auc: 0.835095	valid_1's binary_logloss: 0.13684
    [52]	valid_0's auc: 0.880033	valid_0's binary_logloss: 0.121459	valid_1's auc: 0.83496	valid_1's binary_logloss: 0.136927
    [53]	valid_0's auc: 0.880447	valid_0's binary_logloss: 0.121297	valid_1's auc: 0.834955	valid_1's binary_logloss: 0.136953
    [54]	valid_0's auc: 0.88087	valid_0's binary_logloss: 0.121116	valid_1's auc: 0.834926	valid_1's binary_logloss: 0.136994
    [55]	valid_0's auc: 0.881105	valid_0's binary_logloss: 0.12097	valid_1's auc: 0.834861	valid_1's binary_logloss: 0.137043
    [56]	valid_0's auc: 0.88161	valid_0's binary_logloss: 0.120772	valid_1's auc: 0.834629	valid_1's binary_logloss: 0.137125
    [57]	valid_0's auc: 0.882134	valid_0's binary_logloss: 0.120595	valid_1's auc: 0.834576	valid_1's binary_logloss: 0.137157
    [58]	valid_0's auc: 0.882429	valid_0's binary_logloss: 0.120441	valid_1's auc: 0.834319	valid_1's binary_logloss: 0.137209
    [59]	valid_0's auc: 0.882799	valid_0's binary_logloss: 0.120268	valid_1's auc: 0.834028	valid_1's binary_logloss: 0.1373
    [60]	valid_0's auc: 0.883147	valid_0's binary_logloss: 0.120094	valid_1's auc: 0.833999	valid_1's binary_logloss: 0.137339
    [61]	valid_0's auc: 0.883628	valid_0's binary_logloss: 0.119914	valid_1's auc: 0.834234	valid_1's binary_logloss: 0.137314
    [62]	valid_0's auc: 0.884034	valid_0's binary_logloss: 0.119753	valid_1's auc: 0.834179	valid_1's binary_logloss: 0.137356
    [63]	valid_0's auc: 0.884169	valid_0's binary_logloss: 0.119623	valid_1's auc: 0.834035	valid_1's binary_logloss: 0.137378
    Early stopping, best iteration is:
    [33]	valid_0's auc: 0.870314	valid_0's binary_logloss: 0.125324	valid_1's auc: 0.837039	valid_1's binary_logloss: 0.13642
    [1]	valid_0's auc: 0.826066	valid_0's binary_logloss: 0.157509	valid_1's auc: 0.810763	valid_1's binary_logloss: 0.160177
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.830558	valid_0's binary_logloss: 0.152317	valid_1's auc: 0.814406	valid_1's binary_logloss: 0.15553
    [3]	valid_0's auc: 0.833526	valid_0's binary_logloss: 0.148529	valid_1's auc: 0.815841	valid_1's binary_logloss: 0.152134
    [4]	valid_0's auc: 0.839653	valid_0's binary_logloss: 0.145555	valid_1's auc: 0.821767	valid_1's binary_logloss: 0.149519
    [5]	valid_0's auc: 0.842943	valid_0's binary_logloss: 0.143073	valid_1's auc: 0.826334	valid_1's binary_logloss: 0.14736
    [6]	valid_0's auc: 0.845929	valid_0's binary_logloss: 0.141096	valid_1's auc: 0.828882	valid_1's binary_logloss: 0.145649
    [7]	valid_0's auc: 0.847231	valid_0's binary_logloss: 0.139394	valid_1's auc: 0.830621	valid_1's binary_logloss: 0.144201
    [8]	valid_0's auc: 0.849609	valid_0's binary_logloss: 0.137956	valid_1's auc: 0.831324	valid_1's binary_logloss: 0.143105
    [9]	valid_0's auc: 0.851333	valid_0's binary_logloss: 0.136733	valid_1's auc: 0.83377	valid_1's binary_logloss: 0.142018
    [10]	valid_0's auc: 0.852692	valid_0's binary_logloss: 0.135654	valid_1's auc: 0.834493	valid_1's binary_logloss: 0.141149
    [11]	valid_0's auc: 0.854022	valid_0's binary_logloss: 0.134688	valid_1's auc: 0.834616	valid_1's binary_logloss: 0.140437
    [12]	valid_0's auc: 0.854419	valid_0's binary_logloss: 0.13393	valid_1's auc: 0.83589	valid_1's binary_logloss: 0.139765
    [13]	valid_0's auc: 0.854853	valid_0's binary_logloss: 0.133229	valid_1's auc: 0.835999	valid_1's binary_logloss: 0.139284
    [14]	valid_0's auc: 0.856241	valid_0's binary_logloss: 0.132532	valid_1's auc: 0.837392	valid_1's binary_logloss: 0.138795
    [15]	valid_0's auc: 0.85803	valid_0's binary_logloss: 0.131862	valid_1's auc: 0.83818	valid_1's binary_logloss: 0.138366
    [16]	valid_0's auc: 0.859289	valid_0's binary_logloss: 0.131311	valid_1's auc: 0.838523	valid_1's binary_logloss: 0.138137
    [17]	valid_0's auc: 0.860115	valid_0's binary_logloss: 0.130812	valid_1's auc: 0.839359	valid_1's binary_logloss: 0.137816
    [18]	valid_0's auc: 0.860807	valid_0's binary_logloss: 0.130408	valid_1's auc: 0.840179	valid_1's binary_logloss: 0.13749
    [19]	valid_0's auc: 0.861568	valid_0's binary_logloss: 0.129954	valid_1's auc: 0.839855	valid_1's binary_logloss: 0.137387
    [20]	valid_0's auc: 0.863194	valid_0's binary_logloss: 0.129482	valid_1's auc: 0.839715	valid_1's binary_logloss: 0.137226
    [21]	valid_0's auc: 0.864273	valid_0's binary_logloss: 0.12907	valid_1's auc: 0.840516	valid_1's binary_logloss: 0.136991
    [22]	valid_0's auc: 0.865334	valid_0's binary_logloss: 0.128648	valid_1's auc: 0.841302	valid_1's binary_logloss: 0.136769
    [23]	valid_0's auc: 0.866353	valid_0's binary_logloss: 0.128302	valid_1's auc: 0.841085	valid_1's binary_logloss: 0.136653
    [24]	valid_0's auc: 0.86699	valid_0's binary_logloss: 0.127957	valid_1's auc: 0.841235	valid_1's binary_logloss: 0.136611
    [25]	valid_0's auc: 0.867587	valid_0's binary_logloss: 0.127634	valid_1's auc: 0.840948	valid_1's binary_logloss: 0.136644
    [26]	valid_0's auc: 0.868507	valid_0's binary_logloss: 0.127298	valid_1's auc: 0.841033	valid_1's binary_logloss: 0.136526
    [27]	valid_0's auc: 0.869362	valid_0's binary_logloss: 0.126976	valid_1's auc: 0.840939	valid_1's binary_logloss: 0.136445
    [28]	valid_0's auc: 0.870126	valid_0's binary_logloss: 0.126688	valid_1's auc: 0.840452	valid_1's binary_logloss: 0.13645
    [29]	valid_0's auc: 0.870568	valid_0's binary_logloss: 0.126442	valid_1's auc: 0.840809	valid_1's binary_logloss: 0.136345
    [30]	valid_0's auc: 0.871323	valid_0's binary_logloss: 0.126166	valid_1's auc: 0.840599	valid_1's binary_logloss: 0.136334
    [31]	valid_0's auc: 0.872181	valid_0's binary_logloss: 0.125929	valid_1's auc: 0.840663	valid_1's binary_logloss: 0.136267
    [32]	valid_0's auc: 0.873089	valid_0's binary_logloss: 0.125641	valid_1's auc: 0.841679	valid_1's binary_logloss: 0.136141
    [33]	valid_0's auc: 0.873443	valid_0's binary_logloss: 0.125415	valid_1's auc: 0.841597	valid_1's binary_logloss: 0.136138
    [34]	valid_0's auc: 0.874111	valid_0's binary_logloss: 0.125163	valid_1's auc: 0.841194	valid_1's binary_logloss: 0.13614
    [35]	valid_0's auc: 0.874804	valid_0's binary_logloss: 0.124891	valid_1's auc: 0.841136	valid_1's binary_logloss: 0.136214
    [36]	valid_0's auc: 0.875123	valid_0's binary_logloss: 0.124721	valid_1's auc: 0.840883	valid_1's binary_logloss: 0.136268
    [37]	valid_0's auc: 0.875964	valid_0's binary_logloss: 0.124479	valid_1's auc: 0.840327	valid_1's binary_logloss: 0.136363
    [38]	valid_0's auc: 0.876484	valid_0's binary_logloss: 0.12422	valid_1's auc: 0.840317	valid_1's binary_logloss: 0.136378
    [39]	valid_0's auc: 0.877529	valid_0's binary_logloss: 0.123975	valid_1's auc: 0.840224	valid_1's binary_logloss: 0.136372
    [40]	valid_0's auc: 0.877916	valid_0's binary_logloss: 0.123803	valid_1's auc: 0.840099	valid_1's binary_logloss: 0.136418
    [41]	valid_0's auc: 0.878404	valid_0's binary_logloss: 0.123606	valid_1's auc: 0.839447	valid_1's binary_logloss: 0.136498
    [42]	valid_0's auc: 0.878787	valid_0's binary_logloss: 0.123458	valid_1's auc: 0.839737	valid_1's binary_logloss: 0.136466
    [43]	valid_0's auc: 0.879296	valid_0's binary_logloss: 0.123263	valid_1's auc: 0.838848	valid_1's binary_logloss: 0.136611
    [44]	valid_0's auc: 0.879561	valid_0's binary_logloss: 0.123111	valid_1's auc: 0.83939	valid_1's binary_logloss: 0.136509
    [45]	valid_0's auc: 0.880167	valid_0's binary_logloss: 0.122912	valid_1's auc: 0.83916	valid_1's binary_logloss: 0.136564
    [46]	valid_0's auc: 0.880652	valid_0's binary_logloss: 0.12273	valid_1's auc: 0.839133	valid_1's binary_logloss: 0.136588
    [47]	valid_0's auc: 0.881351	valid_0's binary_logloss: 0.122478	valid_1's auc: 0.839108	valid_1's binary_logloss: 0.136599
    [48]	valid_0's auc: 0.881786	valid_0's binary_logloss: 0.122308	valid_1's auc: 0.838894	valid_1's binary_logloss: 0.136636
    [49]	valid_0's auc: 0.88242	valid_0's binary_logloss: 0.122054	valid_1's auc: 0.838476	valid_1's binary_logloss: 0.136727
    [50]	valid_0's auc: 0.88279	valid_0's binary_logloss: 0.121862	valid_1's auc: 0.837735	valid_1's binary_logloss: 0.136833
    [51]	valid_0's auc: 0.883422	valid_0's binary_logloss: 0.121646	valid_1's auc: 0.837691	valid_1's binary_logloss: 0.136833
    [52]	valid_0's auc: 0.884035	valid_0's binary_logloss: 0.121461	valid_1's auc: 0.837503	valid_1's binary_logloss: 0.136856
    [53]	valid_0's auc: 0.884636	valid_0's binary_logloss: 0.121237	valid_1's auc: 0.837159	valid_1's binary_logloss: 0.136933
    [54]	valid_0's auc: 0.885019	valid_0's binary_logloss: 0.121086	valid_1's auc: 0.836758	valid_1's binary_logloss: 0.136995
    [55]	valid_0's auc: 0.88546	valid_0's binary_logloss: 0.120908	valid_1's auc: 0.836692	valid_1's binary_logloss: 0.137006
    [56]	valid_0's auc: 0.885711	valid_0's binary_logloss: 0.120745	valid_1's auc: 0.836703	valid_1's binary_logloss: 0.136998
    [57]	valid_0's auc: 0.886081	valid_0's binary_logloss: 0.120544	valid_1's auc: 0.836462	valid_1's binary_logloss: 0.137073
    [58]	valid_0's auc: 0.886576	valid_0's binary_logloss: 0.120352	valid_1's auc: 0.836241	valid_1's binary_logloss: 0.137127
    [59]	valid_0's auc: 0.886878	valid_0's binary_logloss: 0.120191	valid_1's auc: 0.8358	valid_1's binary_logloss: 0.137207
    [60]	valid_0's auc: 0.887158	valid_0's binary_logloss: 0.120005	valid_1's auc: 0.835688	valid_1's binary_logloss: 0.137272
    [61]	valid_0's auc: 0.88755	valid_0's binary_logloss: 0.119855	valid_1's auc: 0.835652	valid_1's binary_logloss: 0.137262
    [62]	valid_0's auc: 0.887817	valid_0's binary_logloss: 0.11971	valid_1's auc: 0.835355	valid_1's binary_logloss: 0.13732
    Early stopping, best iteration is:
    [32]	valid_0's auc: 0.873089	valid_0's binary_logloss: 0.125641	valid_1's auc: 0.841679	valid_1's binary_logloss: 0.136141
    [1]	valid_0's auc: 0.822527	valid_0's binary_logloss: 0.157702	valid_1's auc: 0.811735	valid_1's binary_logloss: 0.160053
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.828897	valid_0's binary_logloss: 0.152381	valid_1's auc: 0.815279	valid_1's binary_logloss: 0.155308
    [3]	valid_0's auc: 0.833978	valid_0's binary_logloss: 0.148529	valid_1's auc: 0.819863	valid_1's binary_logloss: 0.151959
    [4]	valid_0's auc: 0.837641	valid_0's binary_logloss: 0.145558	valid_1's auc: 0.821032	valid_1's binary_logloss: 0.149309
    [5]	valid_0's auc: 0.841306	valid_0's binary_logloss: 0.143184	valid_1's auc: 0.824926	valid_1's binary_logloss: 0.147249
    [6]	valid_0's auc: 0.843853	valid_0's binary_logloss: 0.141094	valid_1's auc: 0.825446	valid_1's binary_logloss: 0.145583
    [7]	valid_0's auc: 0.846589	valid_0's binary_logloss: 0.13939	valid_1's auc: 0.829069	valid_1's binary_logloss: 0.144172
    [8]	valid_0's auc: 0.847805	valid_0's binary_logloss: 0.137988	valid_1's auc: 0.829457	valid_1's binary_logloss: 0.143004
    [9]	valid_0's auc: 0.850007	valid_0's binary_logloss: 0.136683	valid_1's auc: 0.831298	valid_1's binary_logloss: 0.141959
    [10]	valid_0's auc: 0.850514	valid_0's binary_logloss: 0.135597	valid_1's auc: 0.830745	valid_1's binary_logloss: 0.141234
    [11]	valid_0's auc: 0.851694	valid_0's binary_logloss: 0.134644	valid_1's auc: 0.831218	valid_1's binary_logloss: 0.140531
    [12]	valid_0's auc: 0.854647	valid_0's binary_logloss: 0.133779	valid_1's auc: 0.832445	valid_1's binary_logloss: 0.140011
    [13]	valid_0's auc: 0.8552	valid_0's binary_logloss: 0.13303	valid_1's auc: 0.832525	valid_1's binary_logloss: 0.139542
    [14]	valid_0's auc: 0.85611	valid_0's binary_logloss: 0.132355	valid_1's auc: 0.832559	valid_1's binary_logloss: 0.139125
    [15]	valid_0's auc: 0.856886	valid_0's binary_logloss: 0.131779	valid_1's auc: 0.833401	valid_1's binary_logloss: 0.138708
    [16]	valid_0's auc: 0.857397	valid_0's binary_logloss: 0.13126	valid_1's auc: 0.833709	valid_1's binary_logloss: 0.138376
    [17]	valid_0's auc: 0.859038	valid_0's binary_logloss: 0.130686	valid_1's auc: 0.83451	valid_1's binary_logloss: 0.138079
    [18]	valid_0's auc: 0.859754	valid_0's binary_logloss: 0.130225	valid_1's auc: 0.834886	valid_1's binary_logloss: 0.137839
    [19]	valid_0's auc: 0.861354	valid_0's binary_logloss: 0.129689	valid_1's auc: 0.835687	valid_1's binary_logloss: 0.137572
    [20]	valid_0's auc: 0.862114	valid_0's binary_logloss: 0.129263	valid_1's auc: 0.83587	valid_1's binary_logloss: 0.137393
    [21]	valid_0's auc: 0.863196	valid_0's binary_logloss: 0.128834	valid_1's auc: 0.835611	valid_1's binary_logloss: 0.137244
    [22]	valid_0's auc: 0.864467	valid_0's binary_logloss: 0.128421	valid_1's auc: 0.836013	valid_1's binary_logloss: 0.137127
    [23]	valid_0's auc: 0.865716	valid_0's binary_logloss: 0.12802	valid_1's auc: 0.835915	valid_1's binary_logloss: 0.13701
    [24]	valid_0's auc: 0.866358	valid_0's binary_logloss: 0.127649	valid_1's auc: 0.835345	valid_1's binary_logloss: 0.136983
    [25]	valid_0's auc: 0.867017	valid_0's binary_logloss: 0.127326	valid_1's auc: 0.835203	valid_1's binary_logloss: 0.13697
    [26]	valid_0's auc: 0.867793	valid_0's binary_logloss: 0.126998	valid_1's auc: 0.835068	valid_1's binary_logloss: 0.13692
    [27]	valid_0's auc: 0.86892	valid_0's binary_logloss: 0.126634	valid_1's auc: 0.835127	valid_1's binary_logloss: 0.136839
    [28]	valid_0's auc: 0.869611	valid_0's binary_logloss: 0.126332	valid_1's auc: 0.835011	valid_1's binary_logloss: 0.136765
    [29]	valid_0's auc: 0.870502	valid_0's binary_logloss: 0.126032	valid_1's auc: 0.835017	valid_1's binary_logloss: 0.136691
    [30]	valid_0's auc: 0.87109	valid_0's binary_logloss: 0.125752	valid_1's auc: 0.835346	valid_1's binary_logloss: 0.136657
    [31]	valid_0's auc: 0.871834	valid_0's binary_logloss: 0.125492	valid_1's auc: 0.835469	valid_1's binary_logloss: 0.136553
    [32]	valid_0's auc: 0.8727	valid_0's binary_logloss: 0.125224	valid_1's auc: 0.834809	valid_1's binary_logloss: 0.136626
    [33]	valid_0's auc: 0.873327	valid_0's binary_logloss: 0.12497	valid_1's auc: 0.834899	valid_1's binary_logloss: 0.136576
    [34]	valid_0's auc: 0.873759	valid_0's binary_logloss: 0.124747	valid_1's auc: 0.835299	valid_1's binary_logloss: 0.136478
    [35]	valid_0's auc: 0.874436	valid_0's binary_logloss: 0.124498	valid_1's auc: 0.834852	valid_1's binary_logloss: 0.136556
    [36]	valid_0's auc: 0.875262	valid_0's binary_logloss: 0.124237	valid_1's auc: 0.834479	valid_1's binary_logloss: 0.13659
    [37]	valid_0's auc: 0.875938	valid_0's binary_logloss: 0.123998	valid_1's auc: 0.834359	valid_1's binary_logloss: 0.136644
    [38]	valid_0's auc: 0.876343	valid_0's binary_logloss: 0.123795	valid_1's auc: 0.834106	valid_1's binary_logloss: 0.136687
    [39]	valid_0's auc: 0.877139	valid_0's binary_logloss: 0.123529	valid_1's auc: 0.83359	valid_1's binary_logloss: 0.136724
    [40]	valid_0's auc: 0.877911	valid_0's binary_logloss: 0.123326	valid_1's auc: 0.833512	valid_1's binary_logloss: 0.136737
    [41]	valid_0's auc: 0.87875	valid_0's binary_logloss: 0.123086	valid_1's auc: 0.833202	valid_1's binary_logloss: 0.136752
    [42]	valid_0's auc: 0.879342	valid_0's binary_logloss: 0.122885	valid_1's auc: 0.832649	valid_1's binary_logloss: 0.136842
    [43]	valid_0's auc: 0.879629	valid_0's binary_logloss: 0.122717	valid_1's auc: 0.832179	valid_1's binary_logloss: 0.136927
    [44]	valid_0's auc: 0.880215	valid_0's binary_logloss: 0.122487	valid_1's auc: 0.832172	valid_1's binary_logloss: 0.13695
    [45]	valid_0's auc: 0.880851	valid_0's binary_logloss: 0.122245	valid_1's auc: 0.831665	valid_1's binary_logloss: 0.137053
    [46]	valid_0's auc: 0.881492	valid_0's binary_logloss: 0.121994	valid_1's auc: 0.83185	valid_1's binary_logloss: 0.137071
    [47]	valid_0's auc: 0.881818	valid_0's binary_logloss: 0.121836	valid_1's auc: 0.831946	valid_1's binary_logloss: 0.137043
    [48]	valid_0's auc: 0.882136	valid_0's binary_logloss: 0.121683	valid_1's auc: 0.832078	valid_1's binary_logloss: 0.137058
    [49]	valid_0's auc: 0.882509	valid_0's binary_logloss: 0.121512	valid_1's auc: 0.832046	valid_1's binary_logloss: 0.13711
    [50]	valid_0's auc: 0.882872	valid_0's binary_logloss: 0.121296	valid_1's auc: 0.83209	valid_1's binary_logloss: 0.137124
    [51]	valid_0's auc: 0.883328	valid_0's binary_logloss: 0.121099	valid_1's auc: 0.831958	valid_1's binary_logloss: 0.137201
    [52]	valid_0's auc: 0.883722	valid_0's binary_logloss: 0.120897	valid_1's auc: 0.831752	valid_1's binary_logloss: 0.13722
    Early stopping, best iteration is:
    [22]	valid_0's auc: 0.864467	valid_0's binary_logloss: 0.128421	valid_1's auc: 0.836013	valid_1's binary_logloss: 0.137127
    [1]	valid_0's auc: 0.822874	valid_0's binary_logloss: 0.157449	valid_1's auc: 0.804936	valid_1's binary_logloss: 0.160126
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.827064	valid_0's binary_logloss: 0.152115	valid_1's auc: 0.810262	valid_1's binary_logloss: 0.155403
    [3]	valid_0's auc: 0.830773	valid_0's binary_logloss: 0.148229	valid_1's auc: 0.814099	valid_1's binary_logloss: 0.151956
    [4]	valid_0's auc: 0.834585	valid_0's binary_logloss: 0.145242	valid_1's auc: 0.815291	valid_1's binary_logloss: 0.149379
    [5]	valid_0's auc: 0.839767	valid_0's binary_logloss: 0.142798	valid_1's auc: 0.819992	valid_1's binary_logloss: 0.147278
    [6]	valid_0's auc: 0.845277	valid_0's binary_logloss: 0.140696	valid_1's auc: 0.824997	valid_1's binary_logloss: 0.145614
    [7]	valid_0's auc: 0.84706	valid_0's binary_logloss: 0.139021	valid_1's auc: 0.827429	valid_1's binary_logloss: 0.144095
    [8]	valid_0's auc: 0.847609	valid_0's binary_logloss: 0.137638	valid_1's auc: 0.827987	valid_1's binary_logloss: 0.143002
    [9]	valid_0's auc: 0.848913	valid_0's binary_logloss: 0.136364	valid_1's auc: 0.82986	valid_1's binary_logloss: 0.141984
    [10]	valid_0's auc: 0.850178	valid_0's binary_logloss: 0.135236	valid_1's auc: 0.829651	valid_1's binary_logloss: 0.141228
    [11]	valid_0's auc: 0.85127	valid_0's binary_logloss: 0.134273	valid_1's auc: 0.830496	valid_1's binary_logloss: 0.140509
    [12]	valid_0's auc: 0.852586	valid_0's binary_logloss: 0.133411	valid_1's auc: 0.830525	valid_1's binary_logloss: 0.13997
    [13]	valid_0's auc: 0.853732	valid_0's binary_logloss: 0.132666	valid_1's auc: 0.829319	valid_1's binary_logloss: 0.13957
    [14]	valid_0's auc: 0.854775	valid_0's binary_logloss: 0.13201	valid_1's auc: 0.832116	valid_1's binary_logloss: 0.139106
    [15]	valid_0's auc: 0.855714	valid_0's binary_logloss: 0.131409	valid_1's auc: 0.833454	valid_1's binary_logloss: 0.138684
    [16]	valid_0's auc: 0.856099	valid_0's binary_logloss: 0.130853	valid_1's auc: 0.832243	valid_1's binary_logloss: 0.138485
    [17]	valid_0's auc: 0.857475	valid_0's binary_logloss: 0.130295	valid_1's auc: 0.832577	valid_1's binary_logloss: 0.138179
    [18]	valid_0's auc: 0.857908	valid_0's binary_logloss: 0.129866	valid_1's auc: 0.832652	valid_1's binary_logloss: 0.138084
    [19]	valid_0's auc: 0.859226	valid_0's binary_logloss: 0.129384	valid_1's auc: 0.832887	valid_1's binary_logloss: 0.137828
    [20]	valid_0's auc: 0.860296	valid_0's binary_logloss: 0.129003	valid_1's auc: 0.834111	valid_1's binary_logloss: 0.137643
    [21]	valid_0's auc: 0.860934	valid_0's binary_logloss: 0.128619	valid_1's auc: 0.834602	valid_1's binary_logloss: 0.137471
    [22]	valid_0's auc: 0.861712	valid_0's binary_logloss: 0.12825	valid_1's auc: 0.834828	valid_1's binary_logloss: 0.137291
    [23]	valid_0's auc: 0.862939	valid_0's binary_logloss: 0.127869	valid_1's auc: 0.835767	valid_1's binary_logloss: 0.137067
    [24]	valid_0's auc: 0.863968	valid_0's binary_logloss: 0.127525	valid_1's auc: 0.835485	valid_1's binary_logloss: 0.13701
    [25]	valid_0's auc: 0.865029	valid_0's binary_logloss: 0.127234	valid_1's auc: 0.835507	valid_1's binary_logloss: 0.136911
    [26]	valid_0's auc: 0.865758	valid_0's binary_logloss: 0.126952	valid_1's auc: 0.835743	valid_1's binary_logloss: 0.136823
    [27]	valid_0's auc: 0.866244	valid_0's binary_logloss: 0.126688	valid_1's auc: 0.836049	valid_1's binary_logloss: 0.136736
    [28]	valid_0's auc: 0.866607	valid_0's binary_logloss: 0.126455	valid_1's auc: 0.836043	valid_1's binary_logloss: 0.136687
    [29]	valid_0's auc: 0.867322	valid_0's binary_logloss: 0.126255	valid_1's auc: 0.836375	valid_1's binary_logloss: 0.136596
    [30]	valid_0's auc: 0.867997	valid_0's binary_logloss: 0.126022	valid_1's auc: 0.836518	valid_1's binary_logloss: 0.136563
    [31]	valid_0's auc: 0.868888	valid_0's binary_logloss: 0.125781	valid_1's auc: 0.836949	valid_1's binary_logloss: 0.13648
    [32]	valid_0's auc: 0.869544	valid_0's binary_logloss: 0.125524	valid_1's auc: 0.836774	valid_1's binary_logloss: 0.136506
    [33]	valid_0's auc: 0.870314	valid_0's binary_logloss: 0.125324	valid_1's auc: 0.837039	valid_1's binary_logloss: 0.13642
    [34]	valid_0's auc: 0.871207	valid_0's binary_logloss: 0.125096	valid_1's auc: 0.836859	valid_1's binary_logloss: 0.136441
    [35]	valid_0's auc: 0.871788	valid_0's binary_logloss: 0.124883	valid_1's auc: 0.836641	valid_1's binary_logloss: 0.136467
    [36]	valid_0's auc: 0.872403	valid_0's binary_logloss: 0.124635	valid_1's auc: 0.836917	valid_1's binary_logloss: 0.13642
    [37]	valid_0's auc: 0.872914	valid_0's binary_logloss: 0.124439	valid_1's auc: 0.836892	valid_1's binary_logloss: 0.136489
    [38]	valid_0's auc: 0.873469	valid_0's binary_logloss: 0.124233	valid_1's auc: 0.836816	valid_1's binary_logloss: 0.136502
    [39]	valid_0's auc: 0.873971	valid_0's binary_logloss: 0.124017	valid_1's auc: 0.83659	valid_1's binary_logloss: 0.136543
    [40]	valid_0's auc: 0.875013	valid_0's binary_logloss: 0.123715	valid_1's auc: 0.836381	valid_1's binary_logloss: 0.136593
    [41]	valid_0's auc: 0.875341	valid_0's binary_logloss: 0.123538	valid_1's auc: 0.8363	valid_1's binary_logloss: 0.136624
    [42]	valid_0's auc: 0.875886	valid_0's binary_logloss: 0.123285	valid_1's auc: 0.836043	valid_1's binary_logloss: 0.136665
    [43]	valid_0's auc: 0.87633	valid_0's binary_logloss: 0.12311	valid_1's auc: 0.836018	valid_1's binary_logloss: 0.136647
    [44]	valid_0's auc: 0.876816	valid_0's binary_logloss: 0.122878	valid_1's auc: 0.836061	valid_1's binary_logloss: 0.136664
    [45]	valid_0's auc: 0.877116	valid_0's binary_logloss: 0.12271	valid_1's auc: 0.836517	valid_1's binary_logloss: 0.136608
    [46]	valid_0's auc: 0.877567	valid_0's binary_logloss: 0.122536	valid_1's auc: 0.836051	valid_1's binary_logloss: 0.13667
    [47]	valid_0's auc: 0.878192	valid_0's binary_logloss: 0.122332	valid_1's auc: 0.835827	valid_1's binary_logloss: 0.136694
    [48]	valid_0's auc: 0.878371	valid_0's binary_logloss: 0.122208	valid_1's auc: 0.835831	valid_1's binary_logloss: 0.136717
    [49]	valid_0's auc: 0.878669	valid_0's binary_logloss: 0.12204	valid_1's auc: 0.835913	valid_1's binary_logloss: 0.136679
    [50]	valid_0's auc: 0.879288	valid_0's binary_logloss: 0.121825	valid_1's auc: 0.835468	valid_1's binary_logloss: 0.136734
    [51]	valid_0's auc: 0.879552	valid_0's binary_logloss: 0.121647	valid_1's auc: 0.835095	valid_1's binary_logloss: 0.13684
    [52]	valid_0's auc: 0.880033	valid_0's binary_logloss: 0.121459	valid_1's auc: 0.83496	valid_1's binary_logloss: 0.136927
    [53]	valid_0's auc: 0.880447	valid_0's binary_logloss: 0.121297	valid_1's auc: 0.834955	valid_1's binary_logloss: 0.136953
    [54]	valid_0's auc: 0.88087	valid_0's binary_logloss: 0.121116	valid_1's auc: 0.834926	valid_1's binary_logloss: 0.136994
    [55]	valid_0's auc: 0.881105	valid_0's binary_logloss: 0.12097	valid_1's auc: 0.834861	valid_1's binary_logloss: 0.137043
    [56]	valid_0's auc: 0.88161	valid_0's binary_logloss: 0.120772	valid_1's auc: 0.834629	valid_1's binary_logloss: 0.137125
    [57]	valid_0's auc: 0.882134	valid_0's binary_logloss: 0.120595	valid_1's auc: 0.834576	valid_1's binary_logloss: 0.137157
    [58]	valid_0's auc: 0.882429	valid_0's binary_logloss: 0.120441	valid_1's auc: 0.834319	valid_1's binary_logloss: 0.137209
    [59]	valid_0's auc: 0.882799	valid_0's binary_logloss: 0.120268	valid_1's auc: 0.834028	valid_1's binary_logloss: 0.1373
    [60]	valid_0's auc: 0.883147	valid_0's binary_logloss: 0.120094	valid_1's auc: 0.833999	valid_1's binary_logloss: 0.137339
    [61]	valid_0's auc: 0.883628	valid_0's binary_logloss: 0.119914	valid_1's auc: 0.834234	valid_1's binary_logloss: 0.137314
    [62]	valid_0's auc: 0.884034	valid_0's binary_logloss: 0.119753	valid_1's auc: 0.834179	valid_1's binary_logloss: 0.137356
    [63]	valid_0's auc: 0.884169	valid_0's binary_logloss: 0.119623	valid_1's auc: 0.834035	valid_1's binary_logloss: 0.137378
    Early stopping, best iteration is:
    [33]	valid_0's auc: 0.870314	valid_0's binary_logloss: 0.125324	valid_1's auc: 0.837039	valid_1's binary_logloss: 0.13642
    [1]	valid_0's auc: 0.826066	valid_0's binary_logloss: 0.157509	valid_1's auc: 0.810763	valid_1's binary_logloss: 0.160177
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.830558	valid_0's binary_logloss: 0.152317	valid_1's auc: 0.814406	valid_1's binary_logloss: 0.15553
    [3]	valid_0's auc: 0.833526	valid_0's binary_logloss: 0.148529	valid_1's auc: 0.815841	valid_1's binary_logloss: 0.152134
    [4]	valid_0's auc: 0.839653	valid_0's binary_logloss: 0.145555	valid_1's auc: 0.821767	valid_1's binary_logloss: 0.149519
    [5]	valid_0's auc: 0.842943	valid_0's binary_logloss: 0.143073	valid_1's auc: 0.826334	valid_1's binary_logloss: 0.14736
    [6]	valid_0's auc: 0.845929	valid_0's binary_logloss: 0.141096	valid_1's auc: 0.828882	valid_1's binary_logloss: 0.145649
    [7]	valid_0's auc: 0.847231	valid_0's binary_logloss: 0.139394	valid_1's auc: 0.830621	valid_1's binary_logloss: 0.144201
    [8]	valid_0's auc: 0.849609	valid_0's binary_logloss: 0.137956	valid_1's auc: 0.831324	valid_1's binary_logloss: 0.143105
    [9]	valid_0's auc: 0.851333	valid_0's binary_logloss: 0.136733	valid_1's auc: 0.83377	valid_1's binary_logloss: 0.142018
    [10]	valid_0's auc: 0.852692	valid_0's binary_logloss: 0.135654	valid_1's auc: 0.834493	valid_1's binary_logloss: 0.141149
    [11]	valid_0's auc: 0.854022	valid_0's binary_logloss: 0.134688	valid_1's auc: 0.834616	valid_1's binary_logloss: 0.140437
    [12]	valid_0's auc: 0.854419	valid_0's binary_logloss: 0.13393	valid_1's auc: 0.83589	valid_1's binary_logloss: 0.139765
    [13]	valid_0's auc: 0.854853	valid_0's binary_logloss: 0.133229	valid_1's auc: 0.835999	valid_1's binary_logloss: 0.139284
    [14]	valid_0's auc: 0.856241	valid_0's binary_logloss: 0.132532	valid_1's auc: 0.837392	valid_1's binary_logloss: 0.138795
    [15]	valid_0's auc: 0.85803	valid_0's binary_logloss: 0.131862	valid_1's auc: 0.83818	valid_1's binary_logloss: 0.138366
    [16]	valid_0's auc: 0.859289	valid_0's binary_logloss: 0.131311	valid_1's auc: 0.838523	valid_1's binary_logloss: 0.138137
    [17]	valid_0's auc: 0.860115	valid_0's binary_logloss: 0.130812	valid_1's auc: 0.839359	valid_1's binary_logloss: 0.137816
    [18]	valid_0's auc: 0.860807	valid_0's binary_logloss: 0.130408	valid_1's auc: 0.840179	valid_1's binary_logloss: 0.13749
    [19]	valid_0's auc: 0.861568	valid_0's binary_logloss: 0.129954	valid_1's auc: 0.839855	valid_1's binary_logloss: 0.137387
    [20]	valid_0's auc: 0.863194	valid_0's binary_logloss: 0.129482	valid_1's auc: 0.839715	valid_1's binary_logloss: 0.137226
    [21]	valid_0's auc: 0.864273	valid_0's binary_logloss: 0.12907	valid_1's auc: 0.840516	valid_1's binary_logloss: 0.136991
    [22]	valid_0's auc: 0.865334	valid_0's binary_logloss: 0.128648	valid_1's auc: 0.841302	valid_1's binary_logloss: 0.136769
    [23]	valid_0's auc: 0.866353	valid_0's binary_logloss: 0.128302	valid_1's auc: 0.841085	valid_1's binary_logloss: 0.136653
    [24]	valid_0's auc: 0.86699	valid_0's binary_logloss: 0.127957	valid_1's auc: 0.841235	valid_1's binary_logloss: 0.136611
    [25]	valid_0's auc: 0.867587	valid_0's binary_logloss: 0.127634	valid_1's auc: 0.840948	valid_1's binary_logloss: 0.136644
    [26]	valid_0's auc: 0.868507	valid_0's binary_logloss: 0.127298	valid_1's auc: 0.841033	valid_1's binary_logloss: 0.136526
    [27]	valid_0's auc: 0.869362	valid_0's binary_logloss: 0.126976	valid_1's auc: 0.840939	valid_1's binary_logloss: 0.136445
    [28]	valid_0's auc: 0.870126	valid_0's binary_logloss: 0.126688	valid_1's auc: 0.840452	valid_1's binary_logloss: 0.13645
    [29]	valid_0's auc: 0.870568	valid_0's binary_logloss: 0.126442	valid_1's auc: 0.840809	valid_1's binary_logloss: 0.136345
    [30]	valid_0's auc: 0.871323	valid_0's binary_logloss: 0.126166	valid_1's auc: 0.840599	valid_1's binary_logloss: 0.136334
    [31]	valid_0's auc: 0.872181	valid_0's binary_logloss: 0.125929	valid_1's auc: 0.840663	valid_1's binary_logloss: 0.136267
    [32]	valid_0's auc: 0.873089	valid_0's binary_logloss: 0.125641	valid_1's auc: 0.841679	valid_1's binary_logloss: 0.136141
    [33]	valid_0's auc: 0.873443	valid_0's binary_logloss: 0.125415	valid_1's auc: 0.841597	valid_1's binary_logloss: 0.136138
    [34]	valid_0's auc: 0.874111	valid_0's binary_logloss: 0.125163	valid_1's auc: 0.841194	valid_1's binary_logloss: 0.13614
    [35]	valid_0's auc: 0.874804	valid_0's binary_logloss: 0.124891	valid_1's auc: 0.841136	valid_1's binary_logloss: 0.136214
    [36]	valid_0's auc: 0.875123	valid_0's binary_logloss: 0.124721	valid_1's auc: 0.840883	valid_1's binary_logloss: 0.136268
    [37]	valid_0's auc: 0.875964	valid_0's binary_logloss: 0.124479	valid_1's auc: 0.840327	valid_1's binary_logloss: 0.136363
    [38]	valid_0's auc: 0.876484	valid_0's binary_logloss: 0.12422	valid_1's auc: 0.840317	valid_1's binary_logloss: 0.136378
    [39]	valid_0's auc: 0.877529	valid_0's binary_logloss: 0.123975	valid_1's auc: 0.840224	valid_1's binary_logloss: 0.136372
    [40]	valid_0's auc: 0.877916	valid_0's binary_logloss: 0.123803	valid_1's auc: 0.840099	valid_1's binary_logloss: 0.136418
    [41]	valid_0's auc: 0.878404	valid_0's binary_logloss: 0.123606	valid_1's auc: 0.839447	valid_1's binary_logloss: 0.136498
    [42]	valid_0's auc: 0.878787	valid_0's binary_logloss: 0.123458	valid_1's auc: 0.839737	valid_1's binary_logloss: 0.136466
    [43]	valid_0's auc: 0.879296	valid_0's binary_logloss: 0.123263	valid_1's auc: 0.838848	valid_1's binary_logloss: 0.136611
    [44]	valid_0's auc: 0.879561	valid_0's binary_logloss: 0.123111	valid_1's auc: 0.83939	valid_1's binary_logloss: 0.136509
    [45]	valid_0's auc: 0.880167	valid_0's binary_logloss: 0.122912	valid_1's auc: 0.83916	valid_1's binary_logloss: 0.136564
    [46]	valid_0's auc: 0.880652	valid_0's binary_logloss: 0.12273	valid_1's auc: 0.839133	valid_1's binary_logloss: 0.136588
    [47]	valid_0's auc: 0.881351	valid_0's binary_logloss: 0.122478	valid_1's auc: 0.839108	valid_1's binary_logloss: 0.136599
    [48]	valid_0's auc: 0.881786	valid_0's binary_logloss: 0.122308	valid_1's auc: 0.838894	valid_1's binary_logloss: 0.136636
    [49]	valid_0's auc: 0.88242	valid_0's binary_logloss: 0.122054	valid_1's auc: 0.838476	valid_1's binary_logloss: 0.136727
    [50]	valid_0's auc: 0.88279	valid_0's binary_logloss: 0.121862	valid_1's auc: 0.837735	valid_1's binary_logloss: 0.136833
    [51]	valid_0's auc: 0.883422	valid_0's binary_logloss: 0.121646	valid_1's auc: 0.837691	valid_1's binary_logloss: 0.136833
    [52]	valid_0's auc: 0.884035	valid_0's binary_logloss: 0.121461	valid_1's auc: 0.837503	valid_1's binary_logloss: 0.136856
    [53]	valid_0's auc: 0.884636	valid_0's binary_logloss: 0.121237	valid_1's auc: 0.837159	valid_1's binary_logloss: 0.136933
    [54]	valid_0's auc: 0.885019	valid_0's binary_logloss: 0.121086	valid_1's auc: 0.836758	valid_1's binary_logloss: 0.136995
    [55]	valid_0's auc: 0.88546	valid_0's binary_logloss: 0.120908	valid_1's auc: 0.836692	valid_1's binary_logloss: 0.137006
    [56]	valid_0's auc: 0.885711	valid_0's binary_logloss: 0.120745	valid_1's auc: 0.836703	valid_1's binary_logloss: 0.136998
    [57]	valid_0's auc: 0.886081	valid_0's binary_logloss: 0.120544	valid_1's auc: 0.836462	valid_1's binary_logloss: 0.137073
    [58]	valid_0's auc: 0.886576	valid_0's binary_logloss: 0.120352	valid_1's auc: 0.836241	valid_1's binary_logloss: 0.137127
    [59]	valid_0's auc: 0.886878	valid_0's binary_logloss: 0.120191	valid_1's auc: 0.8358	valid_1's binary_logloss: 0.137207
    [60]	valid_0's auc: 0.887158	valid_0's binary_logloss: 0.120005	valid_1's auc: 0.835688	valid_1's binary_logloss: 0.137272
    [61]	valid_0's auc: 0.88755	valid_0's binary_logloss: 0.119855	valid_1's auc: 0.835652	valid_1's binary_logloss: 0.137262
    [62]	valid_0's auc: 0.887817	valid_0's binary_logloss: 0.11971	valid_1's auc: 0.835355	valid_1's binary_logloss: 0.13732
    Early stopping, best iteration is:
    [32]	valid_0's auc: 0.873089	valid_0's binary_logloss: 0.125641	valid_1's auc: 0.841679	valid_1's binary_logloss: 0.136141
    [1]	valid_0's auc: 0.822527	valid_0's binary_logloss: 0.157702	valid_1's auc: 0.811735	valid_1's binary_logloss: 0.160053
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.828897	valid_0's binary_logloss: 0.152381	valid_1's auc: 0.815279	valid_1's binary_logloss: 0.155308
    [3]	valid_0's auc: 0.833978	valid_0's binary_logloss: 0.148529	valid_1's auc: 0.819863	valid_1's binary_logloss: 0.151959
    [4]	valid_0's auc: 0.837641	valid_0's binary_logloss: 0.145558	valid_1's auc: 0.821032	valid_1's binary_logloss: 0.149309
    [5]	valid_0's auc: 0.841306	valid_0's binary_logloss: 0.143184	valid_1's auc: 0.824926	valid_1's binary_logloss: 0.147249
    [6]	valid_0's auc: 0.843853	valid_0's binary_logloss: 0.141094	valid_1's auc: 0.825446	valid_1's binary_logloss: 0.145583
    [7]	valid_0's auc: 0.846589	valid_0's binary_logloss: 0.13939	valid_1's auc: 0.829069	valid_1's binary_logloss: 0.144172
    [8]	valid_0's auc: 0.847805	valid_0's binary_logloss: 0.137988	valid_1's auc: 0.829457	valid_1's binary_logloss: 0.143004
    [9]	valid_0's auc: 0.850007	valid_0's binary_logloss: 0.136683	valid_1's auc: 0.831298	valid_1's binary_logloss: 0.141959
    [10]	valid_0's auc: 0.850514	valid_0's binary_logloss: 0.135597	valid_1's auc: 0.830745	valid_1's binary_logloss: 0.141234
    [11]	valid_0's auc: 0.851694	valid_0's binary_logloss: 0.134644	valid_1's auc: 0.831218	valid_1's binary_logloss: 0.140531
    [12]	valid_0's auc: 0.854647	valid_0's binary_logloss: 0.133779	valid_1's auc: 0.832445	valid_1's binary_logloss: 0.140011
    [13]	valid_0's auc: 0.8552	valid_0's binary_logloss: 0.13303	valid_1's auc: 0.832525	valid_1's binary_logloss: 0.139542
    [14]	valid_0's auc: 0.85611	valid_0's binary_logloss: 0.132355	valid_1's auc: 0.832559	valid_1's binary_logloss: 0.139125
    [15]	valid_0's auc: 0.856886	valid_0's binary_logloss: 0.131779	valid_1's auc: 0.833401	valid_1's binary_logloss: 0.138708
    [16]	valid_0's auc: 0.857397	valid_0's binary_logloss: 0.13126	valid_1's auc: 0.833709	valid_1's binary_logloss: 0.138376
    [17]	valid_0's auc: 0.859038	valid_0's binary_logloss: 0.130686	valid_1's auc: 0.83451	valid_1's binary_logloss: 0.138079
    [18]	valid_0's auc: 0.859754	valid_0's binary_logloss: 0.130225	valid_1's auc: 0.834886	valid_1's binary_logloss: 0.137839
    [19]	valid_0's auc: 0.861354	valid_0's binary_logloss: 0.129689	valid_1's auc: 0.835687	valid_1's binary_logloss: 0.137572
    [20]	valid_0's auc: 0.862114	valid_0's binary_logloss: 0.129263	valid_1's auc: 0.83587	valid_1's binary_logloss: 0.137393
    [21]	valid_0's auc: 0.863196	valid_0's binary_logloss: 0.128834	valid_1's auc: 0.835611	valid_1's binary_logloss: 0.137244
    [22]	valid_0's auc: 0.864467	valid_0's binary_logloss: 0.128421	valid_1's auc: 0.836013	valid_1's binary_logloss: 0.137127
    [23]	valid_0's auc: 0.865716	valid_0's binary_logloss: 0.12802	valid_1's auc: 0.835915	valid_1's binary_logloss: 0.13701
    [24]	valid_0's auc: 0.866358	valid_0's binary_logloss: 0.127649	valid_1's auc: 0.835345	valid_1's binary_logloss: 0.136983
    [25]	valid_0's auc: 0.867017	valid_0's binary_logloss: 0.127326	valid_1's auc: 0.835203	valid_1's binary_logloss: 0.13697
    [26]	valid_0's auc: 0.867793	valid_0's binary_logloss: 0.126998	valid_1's auc: 0.835068	valid_1's binary_logloss: 0.13692
    [27]	valid_0's auc: 0.86892	valid_0's binary_logloss: 0.126634	valid_1's auc: 0.835127	valid_1's binary_logloss: 0.136839
    [28]	valid_0's auc: 0.869611	valid_0's binary_logloss: 0.126332	valid_1's auc: 0.835011	valid_1's binary_logloss: 0.136765
    [29]	valid_0's auc: 0.870502	valid_0's binary_logloss: 0.126032	valid_1's auc: 0.835017	valid_1's binary_logloss: 0.136691
    [30]	valid_0's auc: 0.87109	valid_0's binary_logloss: 0.125752	valid_1's auc: 0.835346	valid_1's binary_logloss: 0.136657
    [31]	valid_0's auc: 0.871834	valid_0's binary_logloss: 0.125492	valid_1's auc: 0.835469	valid_1's binary_logloss: 0.136553
    [32]	valid_0's auc: 0.8727	valid_0's binary_logloss: 0.125224	valid_1's auc: 0.834809	valid_1's binary_logloss: 0.136626
    [33]	valid_0's auc: 0.873327	valid_0's binary_logloss: 0.12497	valid_1's auc: 0.834899	valid_1's binary_logloss: 0.136576
    [34]	valid_0's auc: 0.873759	valid_0's binary_logloss: 0.124747	valid_1's auc: 0.835299	valid_1's binary_logloss: 0.136478
    [35]	valid_0's auc: 0.874436	valid_0's binary_logloss: 0.124498	valid_1's auc: 0.834852	valid_1's binary_logloss: 0.136556
    [36]	valid_0's auc: 0.875262	valid_0's binary_logloss: 0.124237	valid_1's auc: 0.834479	valid_1's binary_logloss: 0.13659
    [37]	valid_0's auc: 0.875938	valid_0's binary_logloss: 0.123998	valid_1's auc: 0.834359	valid_1's binary_logloss: 0.136644
    [38]	valid_0's auc: 0.876343	valid_0's binary_logloss: 0.123795	valid_1's auc: 0.834106	valid_1's binary_logloss: 0.136687
    [39]	valid_0's auc: 0.877139	valid_0's binary_logloss: 0.123529	valid_1's auc: 0.83359	valid_1's binary_logloss: 0.136724
    [40]	valid_0's auc: 0.877911	valid_0's binary_logloss: 0.123326	valid_1's auc: 0.833512	valid_1's binary_logloss: 0.136737
    [41]	valid_0's auc: 0.87875	valid_0's binary_logloss: 0.123086	valid_1's auc: 0.833202	valid_1's binary_logloss: 0.136752
    [42]	valid_0's auc: 0.879342	valid_0's binary_logloss: 0.122885	valid_1's auc: 0.832649	valid_1's binary_logloss: 0.136842
    [43]	valid_0's auc: 0.879629	valid_0's binary_logloss: 0.122717	valid_1's auc: 0.832179	valid_1's binary_logloss: 0.136927
    [44]	valid_0's auc: 0.880215	valid_0's binary_logloss: 0.122487	valid_1's auc: 0.832172	valid_1's binary_logloss: 0.13695
    [45]	valid_0's auc: 0.880851	valid_0's binary_logloss: 0.122245	valid_1's auc: 0.831665	valid_1's binary_logloss: 0.137053
    [46]	valid_0's auc: 0.881492	valid_0's binary_logloss: 0.121994	valid_1's auc: 0.83185	valid_1's binary_logloss: 0.137071
    [47]	valid_0's auc: 0.881818	valid_0's binary_logloss: 0.121836	valid_1's auc: 0.831946	valid_1's binary_logloss: 0.137043
    [48]	valid_0's auc: 0.882136	valid_0's binary_logloss: 0.121683	valid_1's auc: 0.832078	valid_1's binary_logloss: 0.137058
    [49]	valid_0's auc: 0.882509	valid_0's binary_logloss: 0.121512	valid_1's auc: 0.832046	valid_1's binary_logloss: 0.13711
    [50]	valid_0's auc: 0.882872	valid_0's binary_logloss: 0.121296	valid_1's auc: 0.83209	valid_1's binary_logloss: 0.137124
    [51]	valid_0's auc: 0.883328	valid_0's binary_logloss: 0.121099	valid_1's auc: 0.831958	valid_1's binary_logloss: 0.137201
    [52]	valid_0's auc: 0.883722	valid_0's binary_logloss: 0.120897	valid_1's auc: 0.831752	valid_1's binary_logloss: 0.13722
    Early stopping, best iteration is:
    [22]	valid_0's auc: 0.864467	valid_0's binary_logloss: 0.128421	valid_1's auc: 0.836013	valid_1's binary_logloss: 0.137127
    [1]	valid_0's auc: 0.829442	valid_0's binary_logloss: 0.156848	valid_1's auc: 0.801853	valid_1's binary_logloss: 0.159917
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.835044	valid_0's binary_logloss: 0.151129	valid_1's auc: 0.810636	valid_1's binary_logloss: 0.155191
    [3]	valid_0's auc: 0.84106	valid_0's binary_logloss: 0.146986	valid_1's auc: 0.81428	valid_1's binary_logloss: 0.151786
    [4]	valid_0's auc: 0.849104	valid_0's binary_logloss: 0.143746	valid_1's auc: 0.822369	valid_1's binary_logloss: 0.149174
    [5]	valid_0's auc: 0.852054	valid_0's binary_logloss: 0.14111	valid_1's auc: 0.825422	valid_1's binary_logloss: 0.147108
    [6]	valid_0's auc: 0.854219	valid_0's binary_logloss: 0.138941	valid_1's auc: 0.828128	valid_1's binary_logloss: 0.145314
    [7]	valid_0's auc: 0.857758	valid_0's binary_logloss: 0.136956	valid_1's auc: 0.828178	valid_1's binary_logloss: 0.144053
    [8]	valid_0's auc: 0.859433	valid_0's binary_logloss: 0.135341	valid_1's auc: 0.828907	valid_1's binary_logloss: 0.142933
    [9]	valid_0's auc: 0.860562	valid_0's binary_logloss: 0.133965	valid_1's auc: 0.830266	valid_1's binary_logloss: 0.141913
    [10]	valid_0's auc: 0.862253	valid_0's binary_logloss: 0.132732	valid_1's auc: 0.830436	valid_1's binary_logloss: 0.141159
    [11]	valid_0's auc: 0.863863	valid_0's binary_logloss: 0.131616	valid_1's auc: 0.830888	valid_1's binary_logloss: 0.140465
    [12]	valid_0's auc: 0.86535	valid_0's binary_logloss: 0.130642	valid_1's auc: 0.828473	valid_1's binary_logloss: 0.140089
    [13]	valid_0's auc: 0.866315	valid_0's binary_logloss: 0.129707	valid_1's auc: 0.828442	valid_1's binary_logloss: 0.139673
    [14]	valid_0's auc: 0.868091	valid_0's binary_logloss: 0.128809	valid_1's auc: 0.828803	valid_1's binary_logloss: 0.13925
    [15]	valid_0's auc: 0.869154	valid_0's binary_logloss: 0.128055	valid_1's auc: 0.829147	valid_1's binary_logloss: 0.138981
    [16]	valid_0's auc: 0.870086	valid_0's binary_logloss: 0.127376	valid_1's auc: 0.829188	valid_1's binary_logloss: 0.13878
    [17]	valid_0's auc: 0.870857	valid_0's binary_logloss: 0.126705	valid_1's auc: 0.829277	valid_1's binary_logloss: 0.138553
    [18]	valid_0's auc: 0.87202	valid_0's binary_logloss: 0.126023	valid_1's auc: 0.829614	valid_1's binary_logloss: 0.138375
    [19]	valid_0's auc: 0.873171	valid_0's binary_logloss: 0.125459	valid_1's auc: 0.830335	valid_1's binary_logloss: 0.13824
    [20]	valid_0's auc: 0.874593	valid_0's binary_logloss: 0.124876	valid_1's auc: 0.831752	valid_1's binary_logloss: 0.13798
    [21]	valid_0's auc: 0.875741	valid_0's binary_logloss: 0.124281	valid_1's auc: 0.832761	valid_1's binary_logloss: 0.137687
    [22]	valid_0's auc: 0.877448	valid_0's binary_logloss: 0.123659	valid_1's auc: 0.833679	valid_1's binary_logloss: 0.137492
    [23]	valid_0's auc: 0.878493	valid_0's binary_logloss: 0.123115	valid_1's auc: 0.833912	valid_1's binary_logloss: 0.137434
    [24]	valid_0's auc: 0.879193	valid_0's binary_logloss: 0.122682	valid_1's auc: 0.833363	valid_1's binary_logloss: 0.137364
    [25]	valid_0's auc: 0.880227	valid_0's binary_logloss: 0.122229	valid_1's auc: 0.833173	valid_1's binary_logloss: 0.137358
    [26]	valid_0's auc: 0.881585	valid_0's binary_logloss: 0.121752	valid_1's auc: 0.832904	valid_1's binary_logloss: 0.13741
    [27]	valid_0's auc: 0.882473	valid_0's binary_logloss: 0.1213	valid_1's auc: 0.832408	valid_1's binary_logloss: 0.137589
    [28]	valid_0's auc: 0.883437	valid_0's binary_logloss: 0.120929	valid_1's auc: 0.832986	valid_1's binary_logloss: 0.137512
    [29]	valid_0's auc: 0.884502	valid_0's binary_logloss: 0.120484	valid_1's auc: 0.833732	valid_1's binary_logloss: 0.137414
    [30]	valid_0's auc: 0.88539	valid_0's binary_logloss: 0.120132	valid_1's auc: 0.83355	valid_1's binary_logloss: 0.137429
    [31]	valid_0's auc: 0.886163	valid_0's binary_logloss: 0.11975	valid_1's auc: 0.833288	valid_1's binary_logloss: 0.137459
    [32]	valid_0's auc: 0.886867	valid_0's binary_logloss: 0.119396	valid_1's auc: 0.833051	valid_1's binary_logloss: 0.137503
    [33]	valid_0's auc: 0.887733	valid_0's binary_logloss: 0.119013	valid_1's auc: 0.832946	valid_1's binary_logloss: 0.137467
    [34]	valid_0's auc: 0.888681	valid_0's binary_logloss: 0.118651	valid_1's auc: 0.833519	valid_1's binary_logloss: 0.137391
    [35]	valid_0's auc: 0.889275	valid_0's binary_logloss: 0.118341	valid_1's auc: 0.833734	valid_1's binary_logloss: 0.137371
    [36]	valid_0's auc: 0.890084	valid_0's binary_logloss: 0.117965	valid_1's auc: 0.833231	valid_1's binary_logloss: 0.137505
    [37]	valid_0's auc: 0.890726	valid_0's binary_logloss: 0.11765	valid_1's auc: 0.833136	valid_1's binary_logloss: 0.13761
    [38]	valid_0's auc: 0.891158	valid_0's binary_logloss: 0.11734	valid_1's auc: 0.832938	valid_1's binary_logloss: 0.137668
    [39]	valid_0's auc: 0.89196	valid_0's binary_logloss: 0.117006	valid_1's auc: 0.832307	valid_1's binary_logloss: 0.13782
    [40]	valid_0's auc: 0.892449	valid_0's binary_logloss: 0.116703	valid_1's auc: 0.832354	valid_1's binary_logloss: 0.137858
    [41]	valid_0's auc: 0.892978	valid_0's binary_logloss: 0.116414	valid_1's auc: 0.831731	valid_1's binary_logloss: 0.137998
    [42]	valid_0's auc: 0.893542	valid_0's binary_logloss: 0.116071	valid_1's auc: 0.830942	valid_1's binary_logloss: 0.138189
    [43]	valid_0's auc: 0.894078	valid_0's binary_logloss: 0.115805	valid_1's auc: 0.830997	valid_1's binary_logloss: 0.138182
    [44]	valid_0's auc: 0.894576	valid_0's binary_logloss: 0.115516	valid_1's auc: 0.830968	valid_1's binary_logloss: 0.138257
    [45]	valid_0's auc: 0.895099	valid_0's binary_logloss: 0.115236	valid_1's auc: 0.830506	valid_1's binary_logloss: 0.13839
    [46]	valid_0's auc: 0.895608	valid_0's binary_logloss: 0.114971	valid_1's auc: 0.829892	valid_1's binary_logloss: 0.138556
    [47]	valid_0's auc: 0.895999	valid_0's binary_logloss: 0.1147	valid_1's auc: 0.829591	valid_1's binary_logloss: 0.138649
    [48]	valid_0's auc: 0.89641	valid_0's binary_logloss: 0.11442	valid_1's auc: 0.829768	valid_1's binary_logloss: 0.138698
    [49]	valid_0's auc: 0.896742	valid_0's binary_logloss: 0.114182	valid_1's auc: 0.830072	valid_1's binary_logloss: 0.138692
    [50]	valid_0's auc: 0.896929	valid_0's binary_logloss: 0.113989	valid_1's auc: 0.829948	valid_1's binary_logloss: 0.138769
    [51]	valid_0's auc: 0.897717	valid_0's binary_logloss: 0.113718	valid_1's auc: 0.830297	valid_1's binary_logloss: 0.138742
    [52]	valid_0's auc: 0.898093	valid_0's binary_logloss: 0.113472	valid_1's auc: 0.830332	valid_1's binary_logloss: 0.138776
    [53]	valid_0's auc: 0.898387	valid_0's binary_logloss: 0.113271	valid_1's auc: 0.830278	valid_1's binary_logloss: 0.138833
    Early stopping, best iteration is:
    [23]	valid_0's auc: 0.878493	valid_0's binary_logloss: 0.123115	valid_1's auc: 0.833912	valid_1's binary_logloss: 0.137434
    [1]	valid_0's auc: 0.834008	valid_0's binary_logloss: 0.156932	valid_1's auc: 0.806689	valid_1's binary_logloss: 0.159986
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.841271	valid_0's binary_logloss: 0.151528	valid_1's auc: 0.816568	valid_1's binary_logloss: 0.155337
    [3]	valid_0's auc: 0.843827	valid_0's binary_logloss: 0.147378	valid_1's auc: 0.818842	valid_1's binary_logloss: 0.151989
    [4]	valid_0's auc: 0.850477	valid_0's binary_logloss: 0.144112	valid_1's auc: 0.824908	valid_1's binary_logloss: 0.149225
    [5]	valid_0's auc: 0.853808	valid_0's binary_logloss: 0.141422	valid_1's auc: 0.826744	valid_1's binary_logloss: 0.147175
    [6]	valid_0's auc: 0.856663	valid_0's binary_logloss: 0.139222	valid_1's auc: 0.828432	valid_1's binary_logloss: 0.145499
    [7]	valid_0's auc: 0.859664	valid_0's binary_logloss: 0.137318	valid_1's auc: 0.829523	valid_1's binary_logloss: 0.14402
    [8]	valid_0's auc: 0.86186	valid_0's binary_logloss: 0.135665	valid_1's auc: 0.8314	valid_1's binary_logloss: 0.142812
    [9]	valid_0's auc: 0.863104	valid_0's binary_logloss: 0.134357	valid_1's auc: 0.831976	valid_1's binary_logloss: 0.141915
    [10]	valid_0's auc: 0.865027	valid_0's binary_logloss: 0.133117	valid_1's auc: 0.83369	valid_1's binary_logloss: 0.141045
    [11]	valid_0's auc: 0.866437	valid_0's binary_logloss: 0.132026	valid_1's auc: 0.834219	valid_1's binary_logloss: 0.140347
    [12]	valid_0's auc: 0.868039	valid_0's binary_logloss: 0.131061	valid_1's auc: 0.834388	valid_1's binary_logloss: 0.139858
    [13]	valid_0's auc: 0.869537	valid_0's binary_logloss: 0.130159	valid_1's auc: 0.835402	valid_1's binary_logloss: 0.139334
    [14]	valid_0's auc: 0.870971	valid_0's binary_logloss: 0.12932	valid_1's auc: 0.835583	valid_1's binary_logloss: 0.138962
    [15]	valid_0's auc: 0.872381	valid_0's binary_logloss: 0.128543	valid_1's auc: 0.836121	valid_1's binary_logloss: 0.138579
    [16]	valid_0's auc: 0.873871	valid_0's binary_logloss: 0.127833	valid_1's auc: 0.836592	valid_1's binary_logloss: 0.13832
    [17]	valid_0's auc: 0.87491	valid_0's binary_logloss: 0.127212	valid_1's auc: 0.836031	valid_1's binary_logloss: 0.138138
    [18]	valid_0's auc: 0.875677	valid_0's binary_logloss: 0.126616	valid_1's auc: 0.835779	valid_1's binary_logloss: 0.137944
    [19]	valid_0's auc: 0.876894	valid_0's binary_logloss: 0.126018	valid_1's auc: 0.836666	valid_1's binary_logloss: 0.137657
    [20]	valid_0's auc: 0.877834	valid_0's binary_logloss: 0.125465	valid_1's auc: 0.836513	valid_1's binary_logloss: 0.137603
    [21]	valid_0's auc: 0.878861	valid_0's binary_logloss: 0.124918	valid_1's auc: 0.836696	valid_1's binary_logloss: 0.137441
    [22]	valid_0's auc: 0.880257	valid_0's binary_logloss: 0.124391	valid_1's auc: 0.836449	valid_1's binary_logloss: 0.137422
    [23]	valid_0's auc: 0.8818	valid_0's binary_logloss: 0.123782	valid_1's auc: 0.836709	valid_1's binary_logloss: 0.137292
    [24]	valid_0's auc: 0.882923	valid_0's binary_logloss: 0.123257	valid_1's auc: 0.836765	valid_1's binary_logloss: 0.137199
    [25]	valid_0's auc: 0.883924	valid_0's binary_logloss: 0.122788	valid_1's auc: 0.836654	valid_1's binary_logloss: 0.137164
    [26]	valid_0's auc: 0.885046	valid_0's binary_logloss: 0.12229	valid_1's auc: 0.836564	valid_1's binary_logloss: 0.137112
    [27]	valid_0's auc: 0.886069	valid_0's binary_logloss: 0.121841	valid_1's auc: 0.836535	valid_1's binary_logloss: 0.137076
    [28]	valid_0's auc: 0.886927	valid_0's binary_logloss: 0.121432	valid_1's auc: 0.836263	valid_1's binary_logloss: 0.137134
    [29]	valid_0's auc: 0.887853	valid_0's binary_logloss: 0.121034	valid_1's auc: 0.836202	valid_1's binary_logloss: 0.137072
    [30]	valid_0's auc: 0.888728	valid_0's binary_logloss: 0.120631	valid_1's auc: 0.836619	valid_1's binary_logloss: 0.136982
    [31]	valid_0's auc: 0.889701	valid_0's binary_logloss: 0.120172	valid_1's auc: 0.836517	valid_1's binary_logloss: 0.136934
    [32]	valid_0's auc: 0.890993	valid_0's binary_logloss: 0.119756	valid_1's auc: 0.836361	valid_1's binary_logloss: 0.136931
    [33]	valid_0's auc: 0.892053	valid_0's binary_logloss: 0.119407	valid_1's auc: 0.836583	valid_1's binary_logloss: 0.136846
    [34]	valid_0's auc: 0.892821	valid_0's binary_logloss: 0.119049	valid_1's auc: 0.836327	valid_1's binary_logloss: 0.136961
    [35]	valid_0's auc: 0.893663	valid_0's binary_logloss: 0.118651	valid_1's auc: 0.836579	valid_1's binary_logloss: 0.136912
    [36]	valid_0's auc: 0.894523	valid_0's binary_logloss: 0.11827	valid_1's auc: 0.836209	valid_1's binary_logloss: 0.136964
    [37]	valid_0's auc: 0.895475	valid_0's binary_logloss: 0.117947	valid_1's auc: 0.835968	valid_1's binary_logloss: 0.13697
    [38]	valid_0's auc: 0.895896	valid_0's binary_logloss: 0.117661	valid_1's auc: 0.836021	valid_1's binary_logloss: 0.136984
    [39]	valid_0's auc: 0.896908	valid_0's binary_logloss: 0.117297	valid_1's auc: 0.835969	valid_1's binary_logloss: 0.137061
    [40]	valid_0's auc: 0.897428	valid_0's binary_logloss: 0.116993	valid_1's auc: 0.835781	valid_1's binary_logloss: 0.137135
    [41]	valid_0's auc: 0.89802	valid_0's binary_logloss: 0.116616	valid_1's auc: 0.834706	valid_1's binary_logloss: 0.137377
    [42]	valid_0's auc: 0.898553	valid_0's binary_logloss: 0.116307	valid_1's auc: 0.834495	valid_1's binary_logloss: 0.137447
    [43]	valid_0's auc: 0.898979	valid_0's binary_logloss: 0.11602	valid_1's auc: 0.834078	valid_1's binary_logloss: 0.137573
    [44]	valid_0's auc: 0.899439	valid_0's binary_logloss: 0.115752	valid_1's auc: 0.833605	valid_1's binary_logloss: 0.137706
    [45]	valid_0's auc: 0.899915	valid_0's binary_logloss: 0.115497	valid_1's auc: 0.833114	valid_1's binary_logloss: 0.137835
    [46]	valid_0's auc: 0.90021	valid_0's binary_logloss: 0.115245	valid_1's auc: 0.833104	valid_1's binary_logloss: 0.137879
    [47]	valid_0's auc: 0.90064	valid_0's binary_logloss: 0.114971	valid_1's auc: 0.83304	valid_1's binary_logloss: 0.137916
    [48]	valid_0's auc: 0.901303	valid_0's binary_logloss: 0.114665	valid_1's auc: 0.833281	valid_1's binary_logloss: 0.137925
    [49]	valid_0's auc: 0.901644	valid_0's binary_logloss: 0.114372	valid_1's auc: 0.83306	valid_1's binary_logloss: 0.138017
    [50]	valid_0's auc: 0.902036	valid_0's binary_logloss: 0.114104	valid_1's auc: 0.832675	valid_1's binary_logloss: 0.13811
    [51]	valid_0's auc: 0.902469	valid_0's binary_logloss: 0.113883	valid_1's auc: 0.832339	valid_1's binary_logloss: 0.138195
    [52]	valid_0's auc: 0.902667	valid_0's binary_logloss: 0.113693	valid_1's auc: 0.831957	valid_1's binary_logloss: 0.138298
    [53]	valid_0's auc: 0.902822	valid_0's binary_logloss: 0.113485	valid_1's auc: 0.831693	valid_1's binary_logloss: 0.138383
    [54]	valid_0's auc: 0.902977	valid_0's binary_logloss: 0.113301	valid_1's auc: 0.830964	valid_1's binary_logloss: 0.138566
    Early stopping, best iteration is:
    [24]	valid_0's auc: 0.882923	valid_0's binary_logloss: 0.123257	valid_1's auc: 0.836765	valid_1's binary_logloss: 0.137199
    [1]	valid_0's auc: 0.832048	valid_0's binary_logloss: 0.157225	valid_1's auc: 0.811916	valid_1's binary_logloss: 0.159885
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.843711	valid_0's binary_logloss: 0.151535	valid_1's auc: 0.821654	valid_1's binary_logloss: 0.154778
    [3]	valid_0's auc: 0.846336	valid_0's binary_logloss: 0.14743	valid_1's auc: 0.823358	valid_1's binary_logloss: 0.151371
    [4]	valid_0's auc: 0.849091	valid_0's binary_logloss: 0.144223	valid_1's auc: 0.826879	valid_1's binary_logloss: 0.148584
    [5]	valid_0's auc: 0.851041	valid_0's binary_logloss: 0.141653	valid_1's auc: 0.825653	valid_1's binary_logloss: 0.14656
    [6]	valid_0's auc: 0.853453	valid_0's binary_logloss: 0.139394	valid_1's auc: 0.828113	valid_1's binary_logloss: 0.144822
    [7]	valid_0's auc: 0.857868	valid_0's binary_logloss: 0.137439	valid_1's auc: 0.828786	valid_1's binary_logloss: 0.143565
    [8]	valid_0's auc: 0.860559	valid_0's binary_logloss: 0.135866	valid_1's auc: 0.829347	valid_1's binary_logloss: 0.142504
    [9]	valid_0's auc: 0.861868	valid_0's binary_logloss: 0.134472	valid_1's auc: 0.829546	valid_1's binary_logloss: 0.141596
    [10]	valid_0's auc: 0.862998	valid_0's binary_logloss: 0.133271	valid_1's auc: 0.828894	valid_1's binary_logloss: 0.140928
    [11]	valid_0's auc: 0.864761	valid_0's binary_logloss: 0.132122	valid_1's auc: 0.828717	valid_1's binary_logloss: 0.140351
    [12]	valid_0's auc: 0.865861	valid_0's binary_logloss: 0.131127	valid_1's auc: 0.827893	valid_1's binary_logloss: 0.139897
    [13]	valid_0's auc: 0.867377	valid_0's binary_logloss: 0.130201	valid_1's auc: 0.828411	valid_1's binary_logloss: 0.139452
    [14]	valid_0's auc: 0.868645	valid_0's binary_logloss: 0.129388	valid_1's auc: 0.828859	valid_1's binary_logloss: 0.139104
    [15]	valid_0's auc: 0.869811	valid_0's binary_logloss: 0.12864	valid_1's auc: 0.827872	valid_1's binary_logloss: 0.138919
    [16]	valid_0's auc: 0.870761	valid_0's binary_logloss: 0.127902	valid_1's auc: 0.827629	valid_1's binary_logloss: 0.138614
    [17]	valid_0's auc: 0.872392	valid_0's binary_logloss: 0.127247	valid_1's auc: 0.828475	valid_1's binary_logloss: 0.138424
    [18]	valid_0's auc: 0.873984	valid_0's binary_logloss: 0.126643	valid_1's auc: 0.828038	valid_1's binary_logloss: 0.13836
    [19]	valid_0's auc: 0.875277	valid_0's binary_logloss: 0.126018	valid_1's auc: 0.828307	valid_1's binary_logloss: 0.138229
    [20]	valid_0's auc: 0.876767	valid_0's binary_logloss: 0.125429	valid_1's auc: 0.828398	valid_1's binary_logloss: 0.138132
    [21]	valid_0's auc: 0.878127	valid_0's binary_logloss: 0.124871	valid_1's auc: 0.827975	valid_1's binary_logloss: 0.138032
    [22]	valid_0's auc: 0.879333	valid_0's binary_logloss: 0.12434	valid_1's auc: 0.828026	valid_1's binary_logloss: 0.137957
    [23]	valid_0's auc: 0.880379	valid_0's binary_logloss: 0.123879	valid_1's auc: 0.827567	valid_1's binary_logloss: 0.138018
    [24]	valid_0's auc: 0.881429	valid_0's binary_logloss: 0.123358	valid_1's auc: 0.827472	valid_1's binary_logloss: 0.137975
    [25]	valid_0's auc: 0.882335	valid_0's binary_logloss: 0.122902	valid_1's auc: 0.827456	valid_1's binary_logloss: 0.137997
    [26]	valid_0's auc: 0.883467	valid_0's binary_logloss: 0.122419	valid_1's auc: 0.827539	valid_1's binary_logloss: 0.137977
    [27]	valid_0's auc: 0.884573	valid_0's binary_logloss: 0.121954	valid_1's auc: 0.827608	valid_1's binary_logloss: 0.137962
    [28]	valid_0's auc: 0.885354	valid_0's binary_logloss: 0.12151	valid_1's auc: 0.827828	valid_1's binary_logloss: 0.137979
    [29]	valid_0's auc: 0.88633	valid_0's binary_logloss: 0.1211	valid_1's auc: 0.827743	valid_1's binary_logloss: 0.137916
    [30]	valid_0's auc: 0.887129	valid_0's binary_logloss: 0.120718	valid_1's auc: 0.82774	valid_1's binary_logloss: 0.13796
    [31]	valid_0's auc: 0.888235	valid_0's binary_logloss: 0.120283	valid_1's auc: 0.827163	valid_1's binary_logloss: 0.137963
    [32]	valid_0's auc: 0.889179	valid_0's binary_logloss: 0.119986	valid_1's auc: 0.826867	valid_1's binary_logloss: 0.13798
    [33]	valid_0's auc: 0.889826	valid_0's binary_logloss: 0.119626	valid_1's auc: 0.826594	valid_1's binary_logloss: 0.13808
    [34]	valid_0's auc: 0.890813	valid_0's binary_logloss: 0.119231	valid_1's auc: 0.826475	valid_1's binary_logloss: 0.13814
    [35]	valid_0's auc: 0.891526	valid_0's binary_logloss: 0.118839	valid_1's auc: 0.826365	valid_1's binary_logloss: 0.138135
    [36]	valid_0's auc: 0.892313	valid_0's binary_logloss: 0.1185	valid_1's auc: 0.825614	valid_1's binary_logloss: 0.138235
    [37]	valid_0's auc: 0.892917	valid_0's binary_logloss: 0.118143	valid_1's auc: 0.825345	valid_1's binary_logloss: 0.138329
    [38]	valid_0's auc: 0.89337	valid_0's binary_logloss: 0.117833	valid_1's auc: 0.824926	valid_1's binary_logloss: 0.138429
    [39]	valid_0's auc: 0.893935	valid_0's binary_logloss: 0.117548	valid_1's auc: 0.824655	valid_1's binary_logloss: 0.138523
    Early stopping, best iteration is:
    [9]	valid_0's auc: 0.861868	valid_0's binary_logloss: 0.134472	valid_1's auc: 0.829546	valid_1's binary_logloss: 0.141596
    [1]	valid_0's auc: 0.829442	valid_0's binary_logloss: 0.156848	valid_1's auc: 0.801853	valid_1's binary_logloss: 0.159917
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.835044	valid_0's binary_logloss: 0.151129	valid_1's auc: 0.810636	valid_1's binary_logloss: 0.155191
    [3]	valid_0's auc: 0.84106	valid_0's binary_logloss: 0.146986	valid_1's auc: 0.81428	valid_1's binary_logloss: 0.151786
    [4]	valid_0's auc: 0.849104	valid_0's binary_logloss: 0.143746	valid_1's auc: 0.822369	valid_1's binary_logloss: 0.149174
    [5]	valid_0's auc: 0.852054	valid_0's binary_logloss: 0.14111	valid_1's auc: 0.825422	valid_1's binary_logloss: 0.147108
    [6]	valid_0's auc: 0.854219	valid_0's binary_logloss: 0.138941	valid_1's auc: 0.828128	valid_1's binary_logloss: 0.145314
    [7]	valid_0's auc: 0.857758	valid_0's binary_logloss: 0.136956	valid_1's auc: 0.828178	valid_1's binary_logloss: 0.144053
    [8]	valid_0's auc: 0.859433	valid_0's binary_logloss: 0.135341	valid_1's auc: 0.828907	valid_1's binary_logloss: 0.142933
    [9]	valid_0's auc: 0.860562	valid_0's binary_logloss: 0.133965	valid_1's auc: 0.830266	valid_1's binary_logloss: 0.141913
    [10]	valid_0's auc: 0.862253	valid_0's binary_logloss: 0.132732	valid_1's auc: 0.830436	valid_1's binary_logloss: 0.141159
    [11]	valid_0's auc: 0.863863	valid_0's binary_logloss: 0.131616	valid_1's auc: 0.830888	valid_1's binary_logloss: 0.140465
    [12]	valid_0's auc: 0.86535	valid_0's binary_logloss: 0.130642	valid_1's auc: 0.828473	valid_1's binary_logloss: 0.140089
    [13]	valid_0's auc: 0.866315	valid_0's binary_logloss: 0.129707	valid_1's auc: 0.828442	valid_1's binary_logloss: 0.139673
    [14]	valid_0's auc: 0.868091	valid_0's binary_logloss: 0.128809	valid_1's auc: 0.828803	valid_1's binary_logloss: 0.13925
    [15]	valid_0's auc: 0.869154	valid_0's binary_logloss: 0.128055	valid_1's auc: 0.829147	valid_1's binary_logloss: 0.138981
    [16]	valid_0's auc: 0.870086	valid_0's binary_logloss: 0.127376	valid_1's auc: 0.829188	valid_1's binary_logloss: 0.13878
    [17]	valid_0's auc: 0.870857	valid_0's binary_logloss: 0.126705	valid_1's auc: 0.829277	valid_1's binary_logloss: 0.138553
    [18]	valid_0's auc: 0.87202	valid_0's binary_logloss: 0.126023	valid_1's auc: 0.829614	valid_1's binary_logloss: 0.138375
    [19]	valid_0's auc: 0.873171	valid_0's binary_logloss: 0.125459	valid_1's auc: 0.830335	valid_1's binary_logloss: 0.13824
    [20]	valid_0's auc: 0.874593	valid_0's binary_logloss: 0.124876	valid_1's auc: 0.831752	valid_1's binary_logloss: 0.13798
    [21]	valid_0's auc: 0.875741	valid_0's binary_logloss: 0.124281	valid_1's auc: 0.832761	valid_1's binary_logloss: 0.137687
    [22]	valid_0's auc: 0.877448	valid_0's binary_logloss: 0.123659	valid_1's auc: 0.833679	valid_1's binary_logloss: 0.137492
    [23]	valid_0's auc: 0.878493	valid_0's binary_logloss: 0.123115	valid_1's auc: 0.833912	valid_1's binary_logloss: 0.137434
    [24]	valid_0's auc: 0.879193	valid_0's binary_logloss: 0.122682	valid_1's auc: 0.833363	valid_1's binary_logloss: 0.137364
    [25]	valid_0's auc: 0.880227	valid_0's binary_logloss: 0.122229	valid_1's auc: 0.833173	valid_1's binary_logloss: 0.137358
    [26]	valid_0's auc: 0.881585	valid_0's binary_logloss: 0.121752	valid_1's auc: 0.832904	valid_1's binary_logloss: 0.13741
    [27]	valid_0's auc: 0.882473	valid_0's binary_logloss: 0.1213	valid_1's auc: 0.832408	valid_1's binary_logloss: 0.137589
    [28]	valid_0's auc: 0.883437	valid_0's binary_logloss: 0.120929	valid_1's auc: 0.832986	valid_1's binary_logloss: 0.137512
    [29]	valid_0's auc: 0.884502	valid_0's binary_logloss: 0.120484	valid_1's auc: 0.833732	valid_1's binary_logloss: 0.137414
    [30]	valid_0's auc: 0.88539	valid_0's binary_logloss: 0.120132	valid_1's auc: 0.83355	valid_1's binary_logloss: 0.137429
    [31]	valid_0's auc: 0.886163	valid_0's binary_logloss: 0.11975	valid_1's auc: 0.833288	valid_1's binary_logloss: 0.137459
    [32]	valid_0's auc: 0.886867	valid_0's binary_logloss: 0.119396	valid_1's auc: 0.833051	valid_1's binary_logloss: 0.137503
    [33]	valid_0's auc: 0.887733	valid_0's binary_logloss: 0.119013	valid_1's auc: 0.832946	valid_1's binary_logloss: 0.137467
    [34]	valid_0's auc: 0.888681	valid_0's binary_logloss: 0.118651	valid_1's auc: 0.833519	valid_1's binary_logloss: 0.137391
    [35]	valid_0's auc: 0.889275	valid_0's binary_logloss: 0.118341	valid_1's auc: 0.833734	valid_1's binary_logloss: 0.137371
    [36]	valid_0's auc: 0.890084	valid_0's binary_logloss: 0.117965	valid_1's auc: 0.833231	valid_1's binary_logloss: 0.137505
    [37]	valid_0's auc: 0.890726	valid_0's binary_logloss: 0.11765	valid_1's auc: 0.833136	valid_1's binary_logloss: 0.13761
    [38]	valid_0's auc: 0.891158	valid_0's binary_logloss: 0.11734	valid_1's auc: 0.832938	valid_1's binary_logloss: 0.137668
    [39]	valid_0's auc: 0.89196	valid_0's binary_logloss: 0.117006	valid_1's auc: 0.832307	valid_1's binary_logloss: 0.13782
    [40]	valid_0's auc: 0.892449	valid_0's binary_logloss: 0.116703	valid_1's auc: 0.832354	valid_1's binary_logloss: 0.137858
    [41]	valid_0's auc: 0.892978	valid_0's binary_logloss: 0.116414	valid_1's auc: 0.831731	valid_1's binary_logloss: 0.137998
    [42]	valid_0's auc: 0.893542	valid_0's binary_logloss: 0.116071	valid_1's auc: 0.830942	valid_1's binary_logloss: 0.138189
    [43]	valid_0's auc: 0.894078	valid_0's binary_logloss: 0.115805	valid_1's auc: 0.830997	valid_1's binary_logloss: 0.138182
    [44]	valid_0's auc: 0.894576	valid_0's binary_logloss: 0.115516	valid_1's auc: 0.830968	valid_1's binary_logloss: 0.138257
    [45]	valid_0's auc: 0.895099	valid_0's binary_logloss: 0.115236	valid_1's auc: 0.830506	valid_1's binary_logloss: 0.13839
    [46]	valid_0's auc: 0.895608	valid_0's binary_logloss: 0.114971	valid_1's auc: 0.829892	valid_1's binary_logloss: 0.138556
    [47]	valid_0's auc: 0.895999	valid_0's binary_logloss: 0.1147	valid_1's auc: 0.829591	valid_1's binary_logloss: 0.138649
    [48]	valid_0's auc: 0.89641	valid_0's binary_logloss: 0.11442	valid_1's auc: 0.829768	valid_1's binary_logloss: 0.138698
    [49]	valid_0's auc: 0.896742	valid_0's binary_logloss: 0.114182	valid_1's auc: 0.830072	valid_1's binary_logloss: 0.138692
    [50]	valid_0's auc: 0.896929	valid_0's binary_logloss: 0.113989	valid_1's auc: 0.829948	valid_1's binary_logloss: 0.138769
    [51]	valid_0's auc: 0.897717	valid_0's binary_logloss: 0.113718	valid_1's auc: 0.830297	valid_1's binary_logloss: 0.138742
    [52]	valid_0's auc: 0.898093	valid_0's binary_logloss: 0.113472	valid_1's auc: 0.830332	valid_1's binary_logloss: 0.138776
    [53]	valid_0's auc: 0.898387	valid_0's binary_logloss: 0.113271	valid_1's auc: 0.830278	valid_1's binary_logloss: 0.138833
    Early stopping, best iteration is:
    [23]	valid_0's auc: 0.878493	valid_0's binary_logloss: 0.123115	valid_1's auc: 0.833912	valid_1's binary_logloss: 0.137434
    [1]	valid_0's auc: 0.834008	valid_0's binary_logloss: 0.156932	valid_1's auc: 0.806689	valid_1's binary_logloss: 0.159986
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.841271	valid_0's binary_logloss: 0.151528	valid_1's auc: 0.816568	valid_1's binary_logloss: 0.155337
    [3]	valid_0's auc: 0.843827	valid_0's binary_logloss: 0.147378	valid_1's auc: 0.818842	valid_1's binary_logloss: 0.151989
    [4]	valid_0's auc: 0.850477	valid_0's binary_logloss: 0.144112	valid_1's auc: 0.824908	valid_1's binary_logloss: 0.149225
    [5]	valid_0's auc: 0.853808	valid_0's binary_logloss: 0.141422	valid_1's auc: 0.826744	valid_1's binary_logloss: 0.147175
    [6]	valid_0's auc: 0.856663	valid_0's binary_logloss: 0.139222	valid_1's auc: 0.828432	valid_1's binary_logloss: 0.145499
    [7]	valid_0's auc: 0.859664	valid_0's binary_logloss: 0.137318	valid_1's auc: 0.829523	valid_1's binary_logloss: 0.14402
    [8]	valid_0's auc: 0.86186	valid_0's binary_logloss: 0.135665	valid_1's auc: 0.8314	valid_1's binary_logloss: 0.142812
    [9]	valid_0's auc: 0.863104	valid_0's binary_logloss: 0.134357	valid_1's auc: 0.831976	valid_1's binary_logloss: 0.141915
    [10]	valid_0's auc: 0.865027	valid_0's binary_logloss: 0.133117	valid_1's auc: 0.83369	valid_1's binary_logloss: 0.141045
    [11]	valid_0's auc: 0.866437	valid_0's binary_logloss: 0.132026	valid_1's auc: 0.834219	valid_1's binary_logloss: 0.140347
    [12]	valid_0's auc: 0.868039	valid_0's binary_logloss: 0.131061	valid_1's auc: 0.834388	valid_1's binary_logloss: 0.139858
    [13]	valid_0's auc: 0.869537	valid_0's binary_logloss: 0.130159	valid_1's auc: 0.835402	valid_1's binary_logloss: 0.139334
    [14]	valid_0's auc: 0.870971	valid_0's binary_logloss: 0.12932	valid_1's auc: 0.835583	valid_1's binary_logloss: 0.138962
    [15]	valid_0's auc: 0.872381	valid_0's binary_logloss: 0.128543	valid_1's auc: 0.836121	valid_1's binary_logloss: 0.138579
    [16]	valid_0's auc: 0.873871	valid_0's binary_logloss: 0.127833	valid_1's auc: 0.836592	valid_1's binary_logloss: 0.13832
    [17]	valid_0's auc: 0.87491	valid_0's binary_logloss: 0.127212	valid_1's auc: 0.836031	valid_1's binary_logloss: 0.138138
    [18]	valid_0's auc: 0.875677	valid_0's binary_logloss: 0.126616	valid_1's auc: 0.835779	valid_1's binary_logloss: 0.137944
    [19]	valid_0's auc: 0.876894	valid_0's binary_logloss: 0.126018	valid_1's auc: 0.836666	valid_1's binary_logloss: 0.137657
    [20]	valid_0's auc: 0.877834	valid_0's binary_logloss: 0.125465	valid_1's auc: 0.836513	valid_1's binary_logloss: 0.137603
    [21]	valid_0's auc: 0.878861	valid_0's binary_logloss: 0.124918	valid_1's auc: 0.836696	valid_1's binary_logloss: 0.137441
    [22]	valid_0's auc: 0.880257	valid_0's binary_logloss: 0.124391	valid_1's auc: 0.836449	valid_1's binary_logloss: 0.137422
    [23]	valid_0's auc: 0.8818	valid_0's binary_logloss: 0.123782	valid_1's auc: 0.836709	valid_1's binary_logloss: 0.137292
    [24]	valid_0's auc: 0.882923	valid_0's binary_logloss: 0.123257	valid_1's auc: 0.836765	valid_1's binary_logloss: 0.137199
    [25]	valid_0's auc: 0.883924	valid_0's binary_logloss: 0.122788	valid_1's auc: 0.836654	valid_1's binary_logloss: 0.137164
    [26]	valid_0's auc: 0.885046	valid_0's binary_logloss: 0.12229	valid_1's auc: 0.836564	valid_1's binary_logloss: 0.137112
    [27]	valid_0's auc: 0.886069	valid_0's binary_logloss: 0.121841	valid_1's auc: 0.836535	valid_1's binary_logloss: 0.137076
    [28]	valid_0's auc: 0.886927	valid_0's binary_logloss: 0.121432	valid_1's auc: 0.836263	valid_1's binary_logloss: 0.137134
    [29]	valid_0's auc: 0.887853	valid_0's binary_logloss: 0.121034	valid_1's auc: 0.836202	valid_1's binary_logloss: 0.137072
    [30]	valid_0's auc: 0.888728	valid_0's binary_logloss: 0.120631	valid_1's auc: 0.836619	valid_1's binary_logloss: 0.136982
    [31]	valid_0's auc: 0.889701	valid_0's binary_logloss: 0.120172	valid_1's auc: 0.836517	valid_1's binary_logloss: 0.136934
    [32]	valid_0's auc: 0.890993	valid_0's binary_logloss: 0.119756	valid_1's auc: 0.836361	valid_1's binary_logloss: 0.136931
    [33]	valid_0's auc: 0.892053	valid_0's binary_logloss: 0.119407	valid_1's auc: 0.836583	valid_1's binary_logloss: 0.136846
    [34]	valid_0's auc: 0.892821	valid_0's binary_logloss: 0.119049	valid_1's auc: 0.836327	valid_1's binary_logloss: 0.136961
    [35]	valid_0's auc: 0.893663	valid_0's binary_logloss: 0.118651	valid_1's auc: 0.836579	valid_1's binary_logloss: 0.136912
    [36]	valid_0's auc: 0.894523	valid_0's binary_logloss: 0.11827	valid_1's auc: 0.836209	valid_1's binary_logloss: 0.136964
    [37]	valid_0's auc: 0.895475	valid_0's binary_logloss: 0.117947	valid_1's auc: 0.835968	valid_1's binary_logloss: 0.13697
    [38]	valid_0's auc: 0.895896	valid_0's binary_logloss: 0.117661	valid_1's auc: 0.836021	valid_1's binary_logloss: 0.136984
    [39]	valid_0's auc: 0.896908	valid_0's binary_logloss: 0.117297	valid_1's auc: 0.835969	valid_1's binary_logloss: 0.137061
    [40]	valid_0's auc: 0.897428	valid_0's binary_logloss: 0.116993	valid_1's auc: 0.835781	valid_1's binary_logloss: 0.137135
    [41]	valid_0's auc: 0.89802	valid_0's binary_logloss: 0.116616	valid_1's auc: 0.834706	valid_1's binary_logloss: 0.137377
    [42]	valid_0's auc: 0.898553	valid_0's binary_logloss: 0.116307	valid_1's auc: 0.834495	valid_1's binary_logloss: 0.137447
    [43]	valid_0's auc: 0.898979	valid_0's binary_logloss: 0.11602	valid_1's auc: 0.834078	valid_1's binary_logloss: 0.137573
    [44]	valid_0's auc: 0.899439	valid_0's binary_logloss: 0.115752	valid_1's auc: 0.833605	valid_1's binary_logloss: 0.137706
    [45]	valid_0's auc: 0.899915	valid_0's binary_logloss: 0.115497	valid_1's auc: 0.833114	valid_1's binary_logloss: 0.137835
    [46]	valid_0's auc: 0.90021	valid_0's binary_logloss: 0.115245	valid_1's auc: 0.833104	valid_1's binary_logloss: 0.137879
    [47]	valid_0's auc: 0.90064	valid_0's binary_logloss: 0.114971	valid_1's auc: 0.83304	valid_1's binary_logloss: 0.137916
    [48]	valid_0's auc: 0.901303	valid_0's binary_logloss: 0.114665	valid_1's auc: 0.833281	valid_1's binary_logloss: 0.137925
    [49]	valid_0's auc: 0.901644	valid_0's binary_logloss: 0.114372	valid_1's auc: 0.83306	valid_1's binary_logloss: 0.138017
    [50]	valid_0's auc: 0.902036	valid_0's binary_logloss: 0.114104	valid_1's auc: 0.832675	valid_1's binary_logloss: 0.13811
    [51]	valid_0's auc: 0.902469	valid_0's binary_logloss: 0.113883	valid_1's auc: 0.832339	valid_1's binary_logloss: 0.138195
    [52]	valid_0's auc: 0.902667	valid_0's binary_logloss: 0.113693	valid_1's auc: 0.831957	valid_1's binary_logloss: 0.138298
    [53]	valid_0's auc: 0.902822	valid_0's binary_logloss: 0.113485	valid_1's auc: 0.831693	valid_1's binary_logloss: 0.138383
    [54]	valid_0's auc: 0.902977	valid_0's binary_logloss: 0.113301	valid_1's auc: 0.830964	valid_1's binary_logloss: 0.138566
    Early stopping, best iteration is:
    [24]	valid_0's auc: 0.882923	valid_0's binary_logloss: 0.123257	valid_1's auc: 0.836765	valid_1's binary_logloss: 0.137199
    [1]	valid_0's auc: 0.832048	valid_0's binary_logloss: 0.157225	valid_1's auc: 0.811916	valid_1's binary_logloss: 0.159885
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.843711	valid_0's binary_logloss: 0.151535	valid_1's auc: 0.821654	valid_1's binary_logloss: 0.154778
    [3]	valid_0's auc: 0.846336	valid_0's binary_logloss: 0.14743	valid_1's auc: 0.823358	valid_1's binary_logloss: 0.151371
    [4]	valid_0's auc: 0.849091	valid_0's binary_logloss: 0.144223	valid_1's auc: 0.826879	valid_1's binary_logloss: 0.148584
    [5]	valid_0's auc: 0.851041	valid_0's binary_logloss: 0.141653	valid_1's auc: 0.825653	valid_1's binary_logloss: 0.14656
    [6]	valid_0's auc: 0.853453	valid_0's binary_logloss: 0.139394	valid_1's auc: 0.828113	valid_1's binary_logloss: 0.144822
    [7]	valid_0's auc: 0.857868	valid_0's binary_logloss: 0.137439	valid_1's auc: 0.828786	valid_1's binary_logloss: 0.143565
    [8]	valid_0's auc: 0.860559	valid_0's binary_logloss: 0.135866	valid_1's auc: 0.829347	valid_1's binary_logloss: 0.142504
    [9]	valid_0's auc: 0.861868	valid_0's binary_logloss: 0.134472	valid_1's auc: 0.829546	valid_1's binary_logloss: 0.141596
    [10]	valid_0's auc: 0.862998	valid_0's binary_logloss: 0.133271	valid_1's auc: 0.828894	valid_1's binary_logloss: 0.140928
    [11]	valid_0's auc: 0.864761	valid_0's binary_logloss: 0.132122	valid_1's auc: 0.828717	valid_1's binary_logloss: 0.140351
    [12]	valid_0's auc: 0.865861	valid_0's binary_logloss: 0.131127	valid_1's auc: 0.827893	valid_1's binary_logloss: 0.139897
    [13]	valid_0's auc: 0.867377	valid_0's binary_logloss: 0.130201	valid_1's auc: 0.828411	valid_1's binary_logloss: 0.139452
    [14]	valid_0's auc: 0.868645	valid_0's binary_logloss: 0.129388	valid_1's auc: 0.828859	valid_1's binary_logloss: 0.139104
    [15]	valid_0's auc: 0.869811	valid_0's binary_logloss: 0.12864	valid_1's auc: 0.827872	valid_1's binary_logloss: 0.138919
    [16]	valid_0's auc: 0.870761	valid_0's binary_logloss: 0.127902	valid_1's auc: 0.827629	valid_1's binary_logloss: 0.138614
    [17]	valid_0's auc: 0.872392	valid_0's binary_logloss: 0.127247	valid_1's auc: 0.828475	valid_1's binary_logloss: 0.138424
    [18]	valid_0's auc: 0.873984	valid_0's binary_logloss: 0.126643	valid_1's auc: 0.828038	valid_1's binary_logloss: 0.13836
    [19]	valid_0's auc: 0.875277	valid_0's binary_logloss: 0.126018	valid_1's auc: 0.828307	valid_1's binary_logloss: 0.138229
    [20]	valid_0's auc: 0.876767	valid_0's binary_logloss: 0.125429	valid_1's auc: 0.828398	valid_1's binary_logloss: 0.138132
    [21]	valid_0's auc: 0.878127	valid_0's binary_logloss: 0.124871	valid_1's auc: 0.827975	valid_1's binary_logloss: 0.138032
    [22]	valid_0's auc: 0.879333	valid_0's binary_logloss: 0.12434	valid_1's auc: 0.828026	valid_1's binary_logloss: 0.137957
    [23]	valid_0's auc: 0.880379	valid_0's binary_logloss: 0.123879	valid_1's auc: 0.827567	valid_1's binary_logloss: 0.138018
    [24]	valid_0's auc: 0.881429	valid_0's binary_logloss: 0.123358	valid_1's auc: 0.827472	valid_1's binary_logloss: 0.137975
    [25]	valid_0's auc: 0.882335	valid_0's binary_logloss: 0.122902	valid_1's auc: 0.827456	valid_1's binary_logloss: 0.137997
    [26]	valid_0's auc: 0.883467	valid_0's binary_logloss: 0.122419	valid_1's auc: 0.827539	valid_1's binary_logloss: 0.137977
    [27]	valid_0's auc: 0.884573	valid_0's binary_logloss: 0.121954	valid_1's auc: 0.827608	valid_1's binary_logloss: 0.137962
    [28]	valid_0's auc: 0.885354	valid_0's binary_logloss: 0.12151	valid_1's auc: 0.827828	valid_1's binary_logloss: 0.137979
    [29]	valid_0's auc: 0.88633	valid_0's binary_logloss: 0.1211	valid_1's auc: 0.827743	valid_1's binary_logloss: 0.137916
    [30]	valid_0's auc: 0.887129	valid_0's binary_logloss: 0.120718	valid_1's auc: 0.82774	valid_1's binary_logloss: 0.13796
    [31]	valid_0's auc: 0.888235	valid_0's binary_logloss: 0.120283	valid_1's auc: 0.827163	valid_1's binary_logloss: 0.137963
    [32]	valid_0's auc: 0.889179	valid_0's binary_logloss: 0.119986	valid_1's auc: 0.826867	valid_1's binary_logloss: 0.13798
    [33]	valid_0's auc: 0.889826	valid_0's binary_logloss: 0.119626	valid_1's auc: 0.826594	valid_1's binary_logloss: 0.13808
    [34]	valid_0's auc: 0.890813	valid_0's binary_logloss: 0.119231	valid_1's auc: 0.826475	valid_1's binary_logloss: 0.13814
    [35]	valid_0's auc: 0.891526	valid_0's binary_logloss: 0.118839	valid_1's auc: 0.826365	valid_1's binary_logloss: 0.138135
    [36]	valid_0's auc: 0.892313	valid_0's binary_logloss: 0.1185	valid_1's auc: 0.825614	valid_1's binary_logloss: 0.138235
    [37]	valid_0's auc: 0.892917	valid_0's binary_logloss: 0.118143	valid_1's auc: 0.825345	valid_1's binary_logloss: 0.138329
    [38]	valid_0's auc: 0.89337	valid_0's binary_logloss: 0.117833	valid_1's auc: 0.824926	valid_1's binary_logloss: 0.138429
    [39]	valid_0's auc: 0.893935	valid_0's binary_logloss: 0.117548	valid_1's auc: 0.824655	valid_1's binary_logloss: 0.138523
    Early stopping, best iteration is:
    [9]	valid_0's auc: 0.861868	valid_0's binary_logloss: 0.134472	valid_1's auc: 0.829546	valid_1's binary_logloss: 0.141596
    [1]	valid_0's auc: 0.822135	valid_0's binary_logloss: 0.157275	valid_1's auc: 0.80763	valid_1's binary_logloss: 0.159959
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.826629	valid_0's binary_logloss: 0.15186	valid_1's auc: 0.811377	valid_1's binary_logloss: 0.155197
    [3]	valid_0's auc: 0.830813	valid_0's binary_logloss: 0.147949	valid_1's auc: 0.813917	valid_1's binary_logloss: 0.151946
    [4]	valid_0's auc: 0.833387	valid_0's binary_logloss: 0.144809	valid_1's auc: 0.814641	valid_1's binary_logloss: 0.149358
    [5]	valid_0's auc: 0.838116	valid_0's binary_logloss: 0.142332	valid_1's auc: 0.816965	valid_1's binary_logloss: 0.147361
    [6]	valid_0's auc: 0.841055	valid_0's binary_logloss: 0.140267	valid_1's auc: 0.81978	valid_1's binary_logloss: 0.145867
    [7]	valid_0's auc: 0.842021	valid_0's binary_logloss: 0.138495	valid_1's auc: 0.819689	valid_1's binary_logloss: 0.144575
    [8]	valid_0's auc: 0.843828	valid_0's binary_logloss: 0.137029	valid_1's auc: 0.822496	valid_1's binary_logloss: 0.143422
    [9]	valid_0's auc: 0.845522	valid_0's binary_logloss: 0.135783	valid_1's auc: 0.823288	valid_1's binary_logloss: 0.142428
    [10]	valid_0's auc: 0.847654	valid_0's binary_logloss: 0.134654	valid_1's auc: 0.824658	valid_1's binary_logloss: 0.141712
    [11]	valid_0's auc: 0.849302	valid_0's binary_logloss: 0.133679	valid_1's auc: 0.826788	valid_1's binary_logloss: 0.140983
    [12]	valid_0's auc: 0.850949	valid_0's binary_logloss: 0.132827	valid_1's auc: 0.828316	valid_1's binary_logloss: 0.140403
    [13]	valid_0's auc: 0.852852	valid_0's binary_logloss: 0.132062	valid_1's auc: 0.830745	valid_1's binary_logloss: 0.139849
    [14]	valid_0's auc: 0.854299	valid_0's binary_logloss: 0.131363	valid_1's auc: 0.832741	valid_1's binary_logloss: 0.139383
    [15]	valid_0's auc: 0.855376	valid_0's binary_logloss: 0.13071	valid_1's auc: 0.833869	valid_1's binary_logloss: 0.138963
    [16]	valid_0's auc: 0.856265	valid_0's binary_logloss: 0.130166	valid_1's auc: 0.833925	valid_1's binary_logloss: 0.138745
    [17]	valid_0's auc: 0.85735	valid_0's binary_logloss: 0.129615	valid_1's auc: 0.834227	valid_1's binary_logloss: 0.138474
    [18]	valid_0's auc: 0.859279	valid_0's binary_logloss: 0.129153	valid_1's auc: 0.834146	valid_1's binary_logloss: 0.138306
    [19]	valid_0's auc: 0.860099	valid_0's binary_logloss: 0.128684	valid_1's auc: 0.833391	valid_1's binary_logloss: 0.138142
    [20]	valid_0's auc: 0.861239	valid_0's binary_logloss: 0.128272	valid_1's auc: 0.833651	valid_1's binary_logloss: 0.13795
    [21]	valid_0's auc: 0.862153	valid_0's binary_logloss: 0.127883	valid_1's auc: 0.834662	valid_1's binary_logloss: 0.137829
    [22]	valid_0's auc: 0.863466	valid_0's binary_logloss: 0.127439	valid_1's auc: 0.834505	valid_1's binary_logloss: 0.137762
    [23]	valid_0's auc: 0.86379	valid_0's binary_logloss: 0.127135	valid_1's auc: 0.834377	valid_1's binary_logloss: 0.137657
    [24]	valid_0's auc: 0.864548	valid_0's binary_logloss: 0.126788	valid_1's auc: 0.834289	valid_1's binary_logloss: 0.137499
    [25]	valid_0's auc: 0.865681	valid_0's binary_logloss: 0.126451	valid_1's auc: 0.834899	valid_1's binary_logloss: 0.137329
    [26]	valid_0's auc: 0.866659	valid_0's binary_logloss: 0.126121	valid_1's auc: 0.835946	valid_1's binary_logloss: 0.137177
    [27]	valid_0's auc: 0.867505	valid_0's binary_logloss: 0.125853	valid_1's auc: 0.836616	valid_1's binary_logloss: 0.137016
    [28]	valid_0's auc: 0.868089	valid_0's binary_logloss: 0.125584	valid_1's auc: 0.836565	valid_1's binary_logloss: 0.137
    [29]	valid_0's auc: 0.868709	valid_0's binary_logloss: 0.125327	valid_1's auc: 0.837028	valid_1's binary_logloss: 0.136878
    [30]	valid_0's auc: 0.87006	valid_0's binary_logloss: 0.12501	valid_1's auc: 0.837512	valid_1's binary_logloss: 0.136792
    [31]	valid_0's auc: 0.870612	valid_0's binary_logloss: 0.124704	valid_1's auc: 0.837541	valid_1's binary_logloss: 0.136757
    [32]	valid_0's auc: 0.871234	valid_0's binary_logloss: 0.124431	valid_1's auc: 0.837141	valid_1's binary_logloss: 0.136755
    [33]	valid_0's auc: 0.872027	valid_0's binary_logloss: 0.124167	valid_1's auc: 0.837258	valid_1's binary_logloss: 0.136728
    [34]	valid_0's auc: 0.87244	valid_0's binary_logloss: 0.123956	valid_1's auc: 0.836829	valid_1's binary_logloss: 0.136799
    [35]	valid_0's auc: 0.873259	valid_0's binary_logloss: 0.123713	valid_1's auc: 0.836871	valid_1's binary_logloss: 0.136775
    [36]	valid_0's auc: 0.873695	valid_0's binary_logloss: 0.123497	valid_1's auc: 0.836512	valid_1's binary_logloss: 0.136859
    [37]	valid_0's auc: 0.874365	valid_0's binary_logloss: 0.123259	valid_1's auc: 0.836421	valid_1's binary_logloss: 0.136872
    [38]	valid_0's auc: 0.874998	valid_0's binary_logloss: 0.123014	valid_1's auc: 0.83628	valid_1's binary_logloss: 0.136939
    [39]	valid_0's auc: 0.875666	valid_0's binary_logloss: 0.122811	valid_1's auc: 0.836577	valid_1's binary_logloss: 0.13689
    [40]	valid_0's auc: 0.875931	valid_0's binary_logloss: 0.122638	valid_1's auc: 0.836148	valid_1's binary_logloss: 0.13697
    [41]	valid_0's auc: 0.876403	valid_0's binary_logloss: 0.122474	valid_1's auc: 0.836008	valid_1's binary_logloss: 0.137005
    [42]	valid_0's auc: 0.87697	valid_0's binary_logloss: 0.122306	valid_1's auc: 0.835814	valid_1's binary_logloss: 0.137031
    [43]	valid_0's auc: 0.877305	valid_0's binary_logloss: 0.122115	valid_1's auc: 0.835467	valid_1's binary_logloss: 0.137112
    [44]	valid_0's auc: 0.87782	valid_0's binary_logloss: 0.121932	valid_1's auc: 0.835339	valid_1's binary_logloss: 0.137217
    [45]	valid_0's auc: 0.878559	valid_0's binary_logloss: 0.121701	valid_1's auc: 0.83506	valid_1's binary_logloss: 0.137256
    [46]	valid_0's auc: 0.878895	valid_0's binary_logloss: 0.121564	valid_1's auc: 0.834955	valid_1's binary_logloss: 0.137255
    [47]	valid_0's auc: 0.879643	valid_0's binary_logloss: 0.121306	valid_1's auc: 0.834987	valid_1's binary_logloss: 0.137275
    [48]	valid_0's auc: 0.88002	valid_0's binary_logloss: 0.121119	valid_1's auc: 0.834864	valid_1's binary_logloss: 0.137329
    [49]	valid_0's auc: 0.880318	valid_0's binary_logloss: 0.120965	valid_1's auc: 0.834547	valid_1's binary_logloss: 0.137394
    [50]	valid_0's auc: 0.880967	valid_0's binary_logloss: 0.12082	valid_1's auc: 0.834588	valid_1's binary_logloss: 0.137388
    [51]	valid_0's auc: 0.8813	valid_0's binary_logloss: 0.120647	valid_1's auc: 0.834288	valid_1's binary_logloss: 0.137467
    [52]	valid_0's auc: 0.881706	valid_0's binary_logloss: 0.120482	valid_1's auc: 0.834305	valid_1's binary_logloss: 0.137491
    [53]	valid_0's auc: 0.882192	valid_0's binary_logloss: 0.1203	valid_1's auc: 0.834662	valid_1's binary_logloss: 0.137492
    [54]	valid_0's auc: 0.882307	valid_0's binary_logloss: 0.120188	valid_1's auc: 0.834241	valid_1's binary_logloss: 0.13763
    [55]	valid_0's auc: 0.883062	valid_0's binary_logloss: 0.119929	valid_1's auc: 0.834145	valid_1's binary_logloss: 0.137674
    [56]	valid_0's auc: 0.883272	valid_0's binary_logloss: 0.119823	valid_1's auc: 0.833884	valid_1's binary_logloss: 0.137762
    [57]	valid_0's auc: 0.883628	valid_0's binary_logloss: 0.11968	valid_1's auc: 0.833563	valid_1's binary_logloss: 0.13782
    [58]	valid_0's auc: 0.884097	valid_0's binary_logloss: 0.119546	valid_1's auc: 0.833647	valid_1's binary_logloss: 0.137781
    [59]	valid_0's auc: 0.884337	valid_0's binary_logloss: 0.119387	valid_1's auc: 0.833138	valid_1's binary_logloss: 0.137878
    [60]	valid_0's auc: 0.884777	valid_0's binary_logloss: 0.119188	valid_1's auc: 0.832909	valid_1's binary_logloss: 0.13796
    [61]	valid_0's auc: 0.885076	valid_0's binary_logloss: 0.119054	valid_1's auc: 0.833025	valid_1's binary_logloss: 0.137959
    Early stopping, best iteration is:
    [31]	valid_0's auc: 0.870612	valid_0's binary_logloss: 0.124704	valid_1's auc: 0.837541	valid_1's binary_logloss: 0.136757
    [1]	valid_0's auc: 0.825036	valid_0's binary_logloss: 0.157424	valid_1's auc: 0.807618	valid_1's binary_logloss: 0.159921
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.829804	valid_0's binary_logloss: 0.152132	valid_1's auc: 0.815884	valid_1's binary_logloss: 0.155124
    [3]	valid_0's auc: 0.835123	valid_0's binary_logloss: 0.148284	valid_1's auc: 0.81743	valid_1's binary_logloss: 0.151751
    [4]	valid_0's auc: 0.841579	valid_0's binary_logloss: 0.145192	valid_1's auc: 0.823237	valid_1's binary_logloss: 0.149159
    [5]	valid_0's auc: 0.843972	valid_0's binary_logloss: 0.142753	valid_1's auc: 0.82747	valid_1's binary_logloss: 0.147042
    [6]	valid_0's auc: 0.846408	valid_0's binary_logloss: 0.140687	valid_1's auc: 0.830402	valid_1's binary_logloss: 0.145398
    [7]	valid_0's auc: 0.848617	valid_0's binary_logloss: 0.138978	valid_1's auc: 0.830319	valid_1's binary_logloss: 0.144054
    [8]	valid_0's auc: 0.849973	valid_0's binary_logloss: 0.137527	valid_1's auc: 0.831414	valid_1's binary_logloss: 0.142955
    [9]	valid_0's auc: 0.852151	valid_0's binary_logloss: 0.136279	valid_1's auc: 0.834577	valid_1's binary_logloss: 0.141825
    [10]	valid_0's auc: 0.852962	valid_0's binary_logloss: 0.135231	valid_1's auc: 0.835301	valid_1's binary_logloss: 0.140947
    [11]	valid_0's auc: 0.853769	valid_0's binary_logloss: 0.134247	valid_1's auc: 0.836379	valid_1's binary_logloss: 0.140121
    [12]	valid_0's auc: 0.855739	valid_0's binary_logloss: 0.133341	valid_1's auc: 0.83615	valid_1's binary_logloss: 0.139629
    [13]	valid_0's auc: 0.857549	valid_0's binary_logloss: 0.132607	valid_1's auc: 0.836232	valid_1's binary_logloss: 0.139209
    [14]	valid_0's auc: 0.858955	valid_0's binary_logloss: 0.131867	valid_1's auc: 0.836387	valid_1's binary_logloss: 0.138827
    [15]	valid_0's auc: 0.860018	valid_0's binary_logloss: 0.13125	valid_1's auc: 0.837299	valid_1's binary_logloss: 0.138441
    [16]	valid_0's auc: 0.861059	valid_0's binary_logloss: 0.130679	valid_1's auc: 0.838197	valid_1's binary_logloss: 0.138074
    [17]	valid_0's auc: 0.862112	valid_0's binary_logloss: 0.130128	valid_1's auc: 0.838835	valid_1's binary_logloss: 0.137791
    [18]	valid_0's auc: 0.862994	valid_0's binary_logloss: 0.129646	valid_1's auc: 0.838904	valid_1's binary_logloss: 0.137577
    [19]	valid_0's auc: 0.863568	valid_0's binary_logloss: 0.129195	valid_1's auc: 0.839028	valid_1's binary_logloss: 0.137319
    [20]	valid_0's auc: 0.86472	valid_0's binary_logloss: 0.128729	valid_1's auc: 0.83984	valid_1's binary_logloss: 0.137089
    [21]	valid_0's auc: 0.866051	valid_0's binary_logloss: 0.128315	valid_1's auc: 0.840385	valid_1's binary_logloss: 0.136889
    [22]	valid_0's auc: 0.866997	valid_0's binary_logloss: 0.127908	valid_1's auc: 0.840544	valid_1's binary_logloss: 0.136724
    [23]	valid_0's auc: 0.867998	valid_0's binary_logloss: 0.127516	valid_1's auc: 0.840471	valid_1's binary_logloss: 0.136628
    [24]	valid_0's auc: 0.868835	valid_0's binary_logloss: 0.127186	valid_1's auc: 0.84011	valid_1's binary_logloss: 0.136573
    [25]	valid_0's auc: 0.869535	valid_0's binary_logloss: 0.126855	valid_1's auc: 0.83987	valid_1's binary_logloss: 0.136502
    [26]	valid_0's auc: 0.870439	valid_0's binary_logloss: 0.126515	valid_1's auc: 0.840538	valid_1's binary_logloss: 0.136387
    [27]	valid_0's auc: 0.871466	valid_0's binary_logloss: 0.126179	valid_1's auc: 0.840326	valid_1's binary_logloss: 0.136301
    [28]	valid_0's auc: 0.872022	valid_0's binary_logloss: 0.125897	valid_1's auc: 0.840415	valid_1's binary_logloss: 0.136253
    [29]	valid_0's auc: 0.873022	valid_0's binary_logloss: 0.125624	valid_1's auc: 0.840531	valid_1's binary_logloss: 0.136185
    [30]	valid_0's auc: 0.873626	valid_0's binary_logloss: 0.125329	valid_1's auc: 0.840407	valid_1's binary_logloss: 0.136188
    [31]	valid_0's auc: 0.874227	valid_0's binary_logloss: 0.12509	valid_1's auc: 0.841102	valid_1's binary_logloss: 0.136046
    [32]	valid_0's auc: 0.874828	valid_0's binary_logloss: 0.124856	valid_1's auc: 0.841125	valid_1's binary_logloss: 0.135986
    [33]	valid_0's auc: 0.875492	valid_0's binary_logloss: 0.124583	valid_1's auc: 0.841558	valid_1's binary_logloss: 0.135907
    [34]	valid_0's auc: 0.876123	valid_0's binary_logloss: 0.124343	valid_1's auc: 0.841695	valid_1's binary_logloss: 0.135865
    [35]	valid_0's auc: 0.876893	valid_0's binary_logloss: 0.124064	valid_1's auc: 0.841945	valid_1's binary_logloss: 0.135831
    [36]	valid_0's auc: 0.877421	valid_0's binary_logloss: 0.123823	valid_1's auc: 0.841949	valid_1's binary_logloss: 0.135864
    [37]	valid_0's auc: 0.877945	valid_0's binary_logloss: 0.123612	valid_1's auc: 0.842028	valid_1's binary_logloss: 0.135826
    [38]	valid_0's auc: 0.878391	valid_0's binary_logloss: 0.123437	valid_1's auc: 0.841969	valid_1's binary_logloss: 0.135815
    [39]	valid_0's auc: 0.878886	valid_0's binary_logloss: 0.123211	valid_1's auc: 0.84193	valid_1's binary_logloss: 0.135809
    [40]	valid_0's auc: 0.879487	valid_0's binary_logloss: 0.123003	valid_1's auc: 0.841846	valid_1's binary_logloss: 0.135816
    [41]	valid_0's auc: 0.880192	valid_0's binary_logloss: 0.12276	valid_1's auc: 0.841501	valid_1's binary_logloss: 0.135878
    [42]	valid_0's auc: 0.880817	valid_0's binary_logloss: 0.122555	valid_1's auc: 0.841224	valid_1's binary_logloss: 0.135894
    [43]	valid_0's auc: 0.881168	valid_0's binary_logloss: 0.122344	valid_1's auc: 0.841189	valid_1's binary_logloss: 0.135895
    [44]	valid_0's auc: 0.88156	valid_0's binary_logloss: 0.122151	valid_1's auc: 0.841136	valid_1's binary_logloss: 0.135914
    [45]	valid_0's auc: 0.882438	valid_0's binary_logloss: 0.121855	valid_1's auc: 0.840553	valid_1's binary_logloss: 0.136015
    [46]	valid_0's auc: 0.883226	valid_0's binary_logloss: 0.121617	valid_1's auc: 0.840057	valid_1's binary_logloss: 0.136084
    [47]	valid_0's auc: 0.883701	valid_0's binary_logloss: 0.121428	valid_1's auc: 0.839905	valid_1's binary_logloss: 0.13611
    [48]	valid_0's auc: 0.884405	valid_0's binary_logloss: 0.121174	valid_1's auc: 0.840003	valid_1's binary_logloss: 0.136077
    [49]	valid_0's auc: 0.8847	valid_0's binary_logloss: 0.12101	valid_1's auc: 0.839683	valid_1's binary_logloss: 0.136145
    [50]	valid_0's auc: 0.885111	valid_0's binary_logloss: 0.120814	valid_1's auc: 0.839749	valid_1's binary_logloss: 0.136099
    [51]	valid_0's auc: 0.885425	valid_0's binary_logloss: 0.12065	valid_1's auc: 0.839416	valid_1's binary_logloss: 0.13609
    [52]	valid_0's auc: 0.885826	valid_0's binary_logloss: 0.120466	valid_1's auc: 0.83916	valid_1's binary_logloss: 0.136158
    [53]	valid_0's auc: 0.88609	valid_0's binary_logloss: 0.120311	valid_1's auc: 0.839381	valid_1's binary_logloss: 0.136151
    [54]	valid_0's auc: 0.886458	valid_0's binary_logloss: 0.120122	valid_1's auc: 0.839222	valid_1's binary_logloss: 0.136197
    [55]	valid_0's auc: 0.887239	valid_0's binary_logloss: 0.11994	valid_1's auc: 0.839191	valid_1's binary_logloss: 0.136207
    [56]	valid_0's auc: 0.887885	valid_0's binary_logloss: 0.119726	valid_1's auc: 0.839099	valid_1's binary_logloss: 0.136256
    [57]	valid_0's auc: 0.888068	valid_0's binary_logloss: 0.119592	valid_1's auc: 0.83885	valid_1's binary_logloss: 0.136275
    [58]	valid_0's auc: 0.888634	valid_0's binary_logloss: 0.119455	valid_1's auc: 0.838851	valid_1's binary_logloss: 0.136269
    [59]	valid_0's auc: 0.888873	valid_0's binary_logloss: 0.119334	valid_1's auc: 0.838683	valid_1's binary_logloss: 0.13629
    [60]	valid_0's auc: 0.889172	valid_0's binary_logloss: 0.119194	valid_1's auc: 0.838562	valid_1's binary_logloss: 0.136301
    [61]	valid_0's auc: 0.889511	valid_0's binary_logloss: 0.119036	valid_1's auc: 0.838472	valid_1's binary_logloss: 0.136335
    [62]	valid_0's auc: 0.889884	valid_0's binary_logloss: 0.118877	valid_1's auc: 0.838355	valid_1's binary_logloss: 0.136343
    [63]	valid_0's auc: 0.890436	valid_0's binary_logloss: 0.118679	valid_1's auc: 0.838056	valid_1's binary_logloss: 0.136425
    [64]	valid_0's auc: 0.890662	valid_0's binary_logloss: 0.118521	valid_1's auc: 0.837425	valid_1's binary_logloss: 0.136547
    [65]	valid_0's auc: 0.890862	valid_0's binary_logloss: 0.118389	valid_1's auc: 0.837399	valid_1's binary_logloss: 0.13655
    [66]	valid_0's auc: 0.890931	valid_0's binary_logloss: 0.118288	valid_1's auc: 0.837035	valid_1's binary_logloss: 0.13665
    [67]	valid_0's auc: 0.891015	valid_0's binary_logloss: 0.118183	valid_1's auc: 0.836707	valid_1's binary_logloss: 0.136783
    Early stopping, best iteration is:
    [37]	valid_0's auc: 0.877945	valid_0's binary_logloss: 0.123612	valid_1's auc: 0.842028	valid_1's binary_logloss: 0.135826
    [1]	valid_0's auc: 0.823507	valid_0's binary_logloss: 0.157501	valid_1's auc: 0.810415	valid_1's binary_logloss: 0.160155
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.829581	valid_0's binary_logloss: 0.152083	valid_1's auc: 0.815091	valid_1's binary_logloss: 0.155417
    [3]	valid_0's auc: 0.834308	valid_0's binary_logloss: 0.148148	valid_1's auc: 0.818622	valid_1's binary_logloss: 0.152027
    [4]	valid_0's auc: 0.836859	valid_0's binary_logloss: 0.145159	valid_1's auc: 0.819436	valid_1's binary_logloss: 0.149485
    [5]	valid_0's auc: 0.841846	valid_0's binary_logloss: 0.142678	valid_1's auc: 0.823324	valid_1's binary_logloss: 0.147372
    [6]	valid_0's auc: 0.844625	valid_0's binary_logloss: 0.140636	valid_1's auc: 0.824092	valid_1's binary_logloss: 0.145715
    [7]	valid_0's auc: 0.848223	valid_0's binary_logloss: 0.138877	valid_1's auc: 0.826887	valid_1's binary_logloss: 0.144253
    [8]	valid_0's auc: 0.849635	valid_0's binary_logloss: 0.137406	valid_1's auc: 0.829164	valid_1's binary_logloss: 0.143039
    [9]	valid_0's auc: 0.850865	valid_0's binary_logloss: 0.136105	valid_1's auc: 0.828916	valid_1's binary_logloss: 0.142109
    [10]	valid_0's auc: 0.852877	valid_0's binary_logloss: 0.134962	valid_1's auc: 0.829565	valid_1's binary_logloss: 0.141319
    [11]	valid_0's auc: 0.853498	valid_0's binary_logloss: 0.134001	valid_1's auc: 0.829604	valid_1's binary_logloss: 0.140625
    [12]	valid_0's auc: 0.854677	valid_0's binary_logloss: 0.133137	valid_1's auc: 0.828978	valid_1's binary_logloss: 0.140133
    [13]	valid_0's auc: 0.856296	valid_0's binary_logloss: 0.132367	valid_1's auc: 0.830184	valid_1's binary_logloss: 0.139696
    [14]	valid_0's auc: 0.858364	valid_0's binary_logloss: 0.131592	valid_1's auc: 0.831576	valid_1's binary_logloss: 0.139209
    [15]	valid_0's auc: 0.859632	valid_0's binary_logloss: 0.130943	valid_1's auc: 0.833147	valid_1's binary_logloss: 0.138759
    [16]	valid_0's auc: 0.86077	valid_0's binary_logloss: 0.130306	valid_1's auc: 0.833392	valid_1's binary_logloss: 0.138459
    [17]	valid_0's auc: 0.861722	valid_0's binary_logloss: 0.129793	valid_1's auc: 0.834279	valid_1's binary_logloss: 0.138085
    [18]	valid_0's auc: 0.862253	valid_0's binary_logloss: 0.129343	valid_1's auc: 0.834323	valid_1's binary_logloss: 0.137866
    [19]	valid_0's auc: 0.863766	valid_0's binary_logloss: 0.128824	valid_1's auc: 0.834736	valid_1's binary_logloss: 0.13763
    [20]	valid_0's auc: 0.8643	valid_0's binary_logloss: 0.128423	valid_1's auc: 0.834528	valid_1's binary_logloss: 0.137554
    [21]	valid_0's auc: 0.865267	valid_0's binary_logloss: 0.127958	valid_1's auc: 0.834478	valid_1's binary_logloss: 0.137406
    [22]	valid_0's auc: 0.866608	valid_0's binary_logloss: 0.127506	valid_1's auc: 0.834905	valid_1's binary_logloss: 0.137271
    [23]	valid_0's auc: 0.867804	valid_0's binary_logloss: 0.127104	valid_1's auc: 0.834293	valid_1's binary_logloss: 0.137211
    [24]	valid_0's auc: 0.86846	valid_0's binary_logloss: 0.12677	valid_1's auc: 0.833842	valid_1's binary_logloss: 0.137176
    [25]	valid_0's auc: 0.869535	valid_0's binary_logloss: 0.126409	valid_1's auc: 0.834057	valid_1's binary_logloss: 0.13708
    [26]	valid_0's auc: 0.870946	valid_0's binary_logloss: 0.126026	valid_1's auc: 0.833797	valid_1's binary_logloss: 0.137021
    [27]	valid_0's auc: 0.871863	valid_0's binary_logloss: 0.12567	valid_1's auc: 0.83361	valid_1's binary_logloss: 0.13702
    [28]	valid_0's auc: 0.872503	valid_0's binary_logloss: 0.125332	valid_1's auc: 0.833415	valid_1's binary_logloss: 0.136948
    [29]	valid_0's auc: 0.873256	valid_0's binary_logloss: 0.125049	valid_1's auc: 0.83344	valid_1's binary_logloss: 0.136909
    [30]	valid_0's auc: 0.874265	valid_0's binary_logloss: 0.124767	valid_1's auc: 0.833129	valid_1's binary_logloss: 0.136924
    [31]	valid_0's auc: 0.875083	valid_0's binary_logloss: 0.124478	valid_1's auc: 0.833207	valid_1's binary_logloss: 0.136915
    [32]	valid_0's auc: 0.875789	valid_0's binary_logloss: 0.124224	valid_1's auc: 0.833079	valid_1's binary_logloss: 0.136916
    [33]	valid_0's auc: 0.876187	valid_0's binary_logloss: 0.123994	valid_1's auc: 0.833172	valid_1's binary_logloss: 0.136856
    [34]	valid_0's auc: 0.876733	valid_0's binary_logloss: 0.123732	valid_1's auc: 0.833412	valid_1's binary_logloss: 0.136808
    [35]	valid_0's auc: 0.877384	valid_0's binary_logloss: 0.123461	valid_1's auc: 0.833287	valid_1's binary_logloss: 0.136785
    [36]	valid_0's auc: 0.877802	valid_0's binary_logloss: 0.123212	valid_1's auc: 0.83302	valid_1's binary_logloss: 0.13685
    [37]	valid_0's auc: 0.878799	valid_0's binary_logloss: 0.122947	valid_1's auc: 0.832295	valid_1's binary_logloss: 0.136905
    [38]	valid_0's auc: 0.879503	valid_0's binary_logloss: 0.122719	valid_1's auc: 0.832134	valid_1's binary_logloss: 0.136919
    [39]	valid_0's auc: 0.880358	valid_0's binary_logloss: 0.122488	valid_1's auc: 0.832266	valid_1's binary_logloss: 0.136938
    [40]	valid_0's auc: 0.881013	valid_0's binary_logloss: 0.122287	valid_1's auc: 0.832133	valid_1's binary_logloss: 0.136932
    [41]	valid_0's auc: 0.881424	valid_0's binary_logloss: 0.122107	valid_1's auc: 0.832209	valid_1's binary_logloss: 0.136934
    [42]	valid_0's auc: 0.881984	valid_0's binary_logloss: 0.121921	valid_1's auc: 0.831997	valid_1's binary_logloss: 0.136976
    [43]	valid_0's auc: 0.882635	valid_0's binary_logloss: 0.121694	valid_1's auc: 0.831527	valid_1's binary_logloss: 0.137038
    [44]	valid_0's auc: 0.883243	valid_0's binary_logloss: 0.121523	valid_1's auc: 0.831593	valid_1's binary_logloss: 0.137055
    [45]	valid_0's auc: 0.883616	valid_0's binary_logloss: 0.121345	valid_1's auc: 0.831225	valid_1's binary_logloss: 0.137134
    [46]	valid_0's auc: 0.884106	valid_0's binary_logloss: 0.121155	valid_1's auc: 0.831678	valid_1's binary_logloss: 0.137064
    [47]	valid_0's auc: 0.884462	valid_0's binary_logloss: 0.120968	valid_1's auc: 0.832127	valid_1's binary_logloss: 0.137012
    [48]	valid_0's auc: 0.884804	valid_0's binary_logloss: 0.120754	valid_1's auc: 0.832474	valid_1's binary_logloss: 0.136991
    [49]	valid_0's auc: 0.885391	valid_0's binary_logloss: 0.120518	valid_1's auc: 0.832375	valid_1's binary_logloss: 0.137054
    [50]	valid_0's auc: 0.885988	valid_0's binary_logloss: 0.120287	valid_1's auc: 0.832272	valid_1's binary_logloss: 0.137065
    [51]	valid_0's auc: 0.886393	valid_0's binary_logloss: 0.120131	valid_1's auc: 0.831867	valid_1's binary_logloss: 0.137159
    [52]	valid_0's auc: 0.886879	valid_0's binary_logloss: 0.119924	valid_1's auc: 0.831971	valid_1's binary_logloss: 0.137133
    Early stopping, best iteration is:
    [22]	valid_0's auc: 0.866608	valid_0's binary_logloss: 0.127506	valid_1's auc: 0.834905	valid_1's binary_logloss: 0.137271
    [1]	valid_0's auc: 0.822135	valid_0's binary_logloss: 0.157275	valid_1's auc: 0.80763	valid_1's binary_logloss: 0.159959
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.826629	valid_0's binary_logloss: 0.15186	valid_1's auc: 0.811377	valid_1's binary_logloss: 0.155197
    [3]	valid_0's auc: 0.830813	valid_0's binary_logloss: 0.147949	valid_1's auc: 0.813917	valid_1's binary_logloss: 0.151946
    [4]	valid_0's auc: 0.833387	valid_0's binary_logloss: 0.144809	valid_1's auc: 0.814641	valid_1's binary_logloss: 0.149358
    [5]	valid_0's auc: 0.838116	valid_0's binary_logloss: 0.142332	valid_1's auc: 0.816965	valid_1's binary_logloss: 0.147361
    [6]	valid_0's auc: 0.841055	valid_0's binary_logloss: 0.140267	valid_1's auc: 0.81978	valid_1's binary_logloss: 0.145867
    [7]	valid_0's auc: 0.842021	valid_0's binary_logloss: 0.138495	valid_1's auc: 0.819689	valid_1's binary_logloss: 0.144575
    [8]	valid_0's auc: 0.843828	valid_0's binary_logloss: 0.137029	valid_1's auc: 0.822496	valid_1's binary_logloss: 0.143422
    [9]	valid_0's auc: 0.845522	valid_0's binary_logloss: 0.135783	valid_1's auc: 0.823288	valid_1's binary_logloss: 0.142428
    [10]	valid_0's auc: 0.847654	valid_0's binary_logloss: 0.134654	valid_1's auc: 0.824658	valid_1's binary_logloss: 0.141712
    [11]	valid_0's auc: 0.849302	valid_0's binary_logloss: 0.133679	valid_1's auc: 0.826788	valid_1's binary_logloss: 0.140983
    [12]	valid_0's auc: 0.850949	valid_0's binary_logloss: 0.132827	valid_1's auc: 0.828316	valid_1's binary_logloss: 0.140403
    [13]	valid_0's auc: 0.852852	valid_0's binary_logloss: 0.132062	valid_1's auc: 0.830745	valid_1's binary_logloss: 0.139849
    [14]	valid_0's auc: 0.854299	valid_0's binary_logloss: 0.131363	valid_1's auc: 0.832741	valid_1's binary_logloss: 0.139383
    [15]	valid_0's auc: 0.855376	valid_0's binary_logloss: 0.13071	valid_1's auc: 0.833869	valid_1's binary_logloss: 0.138963
    [16]	valid_0's auc: 0.856265	valid_0's binary_logloss: 0.130166	valid_1's auc: 0.833925	valid_1's binary_logloss: 0.138745
    [17]	valid_0's auc: 0.85735	valid_0's binary_logloss: 0.129615	valid_1's auc: 0.834227	valid_1's binary_logloss: 0.138474
    [18]	valid_0's auc: 0.859279	valid_0's binary_logloss: 0.129153	valid_1's auc: 0.834146	valid_1's binary_logloss: 0.138306
    [19]	valid_0's auc: 0.860099	valid_0's binary_logloss: 0.128684	valid_1's auc: 0.833391	valid_1's binary_logloss: 0.138142
    [20]	valid_0's auc: 0.861239	valid_0's binary_logloss: 0.128272	valid_1's auc: 0.833651	valid_1's binary_logloss: 0.13795
    [21]	valid_0's auc: 0.862153	valid_0's binary_logloss: 0.127883	valid_1's auc: 0.834662	valid_1's binary_logloss: 0.137829
    [22]	valid_0's auc: 0.863466	valid_0's binary_logloss: 0.127439	valid_1's auc: 0.834505	valid_1's binary_logloss: 0.137762
    [23]	valid_0's auc: 0.86379	valid_0's binary_logloss: 0.127135	valid_1's auc: 0.834377	valid_1's binary_logloss: 0.137657
    [24]	valid_0's auc: 0.864548	valid_0's binary_logloss: 0.126788	valid_1's auc: 0.834289	valid_1's binary_logloss: 0.137499
    [25]	valid_0's auc: 0.865681	valid_0's binary_logloss: 0.126451	valid_1's auc: 0.834899	valid_1's binary_logloss: 0.137329
    [26]	valid_0's auc: 0.866659	valid_0's binary_logloss: 0.126121	valid_1's auc: 0.835946	valid_1's binary_logloss: 0.137177
    [27]	valid_0's auc: 0.867505	valid_0's binary_logloss: 0.125853	valid_1's auc: 0.836616	valid_1's binary_logloss: 0.137016
    [28]	valid_0's auc: 0.868089	valid_0's binary_logloss: 0.125584	valid_1's auc: 0.836565	valid_1's binary_logloss: 0.137
    [29]	valid_0's auc: 0.868709	valid_0's binary_logloss: 0.125327	valid_1's auc: 0.837028	valid_1's binary_logloss: 0.136878
    [30]	valid_0's auc: 0.87006	valid_0's binary_logloss: 0.12501	valid_1's auc: 0.837512	valid_1's binary_logloss: 0.136792
    [31]	valid_0's auc: 0.870612	valid_0's binary_logloss: 0.124704	valid_1's auc: 0.837541	valid_1's binary_logloss: 0.136757
    [32]	valid_0's auc: 0.871234	valid_0's binary_logloss: 0.124431	valid_1's auc: 0.837141	valid_1's binary_logloss: 0.136755
    [33]	valid_0's auc: 0.872027	valid_0's binary_logloss: 0.124167	valid_1's auc: 0.837258	valid_1's binary_logloss: 0.136728
    [34]	valid_0's auc: 0.87244	valid_0's binary_logloss: 0.123956	valid_1's auc: 0.836829	valid_1's binary_logloss: 0.136799
    [35]	valid_0's auc: 0.873259	valid_0's binary_logloss: 0.123713	valid_1's auc: 0.836871	valid_1's binary_logloss: 0.136775
    [36]	valid_0's auc: 0.873695	valid_0's binary_logloss: 0.123497	valid_1's auc: 0.836512	valid_1's binary_logloss: 0.136859
    [37]	valid_0's auc: 0.874365	valid_0's binary_logloss: 0.123259	valid_1's auc: 0.836421	valid_1's binary_logloss: 0.136872
    [38]	valid_0's auc: 0.874998	valid_0's binary_logloss: 0.123014	valid_1's auc: 0.83628	valid_1's binary_logloss: 0.136939
    [39]	valid_0's auc: 0.875666	valid_0's binary_logloss: 0.122811	valid_1's auc: 0.836577	valid_1's binary_logloss: 0.13689
    [40]	valid_0's auc: 0.875931	valid_0's binary_logloss: 0.122638	valid_1's auc: 0.836148	valid_1's binary_logloss: 0.13697
    [41]	valid_0's auc: 0.876403	valid_0's binary_logloss: 0.122474	valid_1's auc: 0.836008	valid_1's binary_logloss: 0.137005
    [42]	valid_0's auc: 0.87697	valid_0's binary_logloss: 0.122306	valid_1's auc: 0.835814	valid_1's binary_logloss: 0.137031
    [43]	valid_0's auc: 0.877305	valid_0's binary_logloss: 0.122115	valid_1's auc: 0.835467	valid_1's binary_logloss: 0.137112
    [44]	valid_0's auc: 0.87782	valid_0's binary_logloss: 0.121932	valid_1's auc: 0.835339	valid_1's binary_logloss: 0.137217
    [45]	valid_0's auc: 0.878559	valid_0's binary_logloss: 0.121701	valid_1's auc: 0.83506	valid_1's binary_logloss: 0.137256
    [46]	valid_0's auc: 0.878895	valid_0's binary_logloss: 0.121564	valid_1's auc: 0.834955	valid_1's binary_logloss: 0.137255
    [47]	valid_0's auc: 0.879643	valid_0's binary_logloss: 0.121306	valid_1's auc: 0.834987	valid_1's binary_logloss: 0.137275
    [48]	valid_0's auc: 0.88002	valid_0's binary_logloss: 0.121119	valid_1's auc: 0.834864	valid_1's binary_logloss: 0.137329
    [49]	valid_0's auc: 0.880318	valid_0's binary_logloss: 0.120965	valid_1's auc: 0.834547	valid_1's binary_logloss: 0.137394
    [50]	valid_0's auc: 0.880967	valid_0's binary_logloss: 0.12082	valid_1's auc: 0.834588	valid_1's binary_logloss: 0.137388
    [51]	valid_0's auc: 0.8813	valid_0's binary_logloss: 0.120647	valid_1's auc: 0.834288	valid_1's binary_logloss: 0.137467
    [52]	valid_0's auc: 0.881706	valid_0's binary_logloss: 0.120482	valid_1's auc: 0.834305	valid_1's binary_logloss: 0.137491
    [53]	valid_0's auc: 0.882192	valid_0's binary_logloss: 0.1203	valid_1's auc: 0.834662	valid_1's binary_logloss: 0.137492
    [54]	valid_0's auc: 0.882307	valid_0's binary_logloss: 0.120188	valid_1's auc: 0.834241	valid_1's binary_logloss: 0.13763
    [55]	valid_0's auc: 0.883062	valid_0's binary_logloss: 0.119929	valid_1's auc: 0.834145	valid_1's binary_logloss: 0.137674
    [56]	valid_0's auc: 0.883272	valid_0's binary_logloss: 0.119823	valid_1's auc: 0.833884	valid_1's binary_logloss: 0.137762
    [57]	valid_0's auc: 0.883628	valid_0's binary_logloss: 0.11968	valid_1's auc: 0.833563	valid_1's binary_logloss: 0.13782
    [58]	valid_0's auc: 0.884097	valid_0's binary_logloss: 0.119546	valid_1's auc: 0.833647	valid_1's binary_logloss: 0.137781
    [59]	valid_0's auc: 0.884337	valid_0's binary_logloss: 0.119387	valid_1's auc: 0.833138	valid_1's binary_logloss: 0.137878
    [60]	valid_0's auc: 0.884777	valid_0's binary_logloss: 0.119188	valid_1's auc: 0.832909	valid_1's binary_logloss: 0.13796
    [61]	valid_0's auc: 0.885076	valid_0's binary_logloss: 0.119054	valid_1's auc: 0.833025	valid_1's binary_logloss: 0.137959
    Early stopping, best iteration is:
    [31]	valid_0's auc: 0.870612	valid_0's binary_logloss: 0.124704	valid_1's auc: 0.837541	valid_1's binary_logloss: 0.136757
    [1]	valid_0's auc: 0.825036	valid_0's binary_logloss: 0.157424	valid_1's auc: 0.807618	valid_1's binary_logloss: 0.159921
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.829804	valid_0's binary_logloss: 0.152132	valid_1's auc: 0.815884	valid_1's binary_logloss: 0.155124
    [3]	valid_0's auc: 0.835123	valid_0's binary_logloss: 0.148284	valid_1's auc: 0.81743	valid_1's binary_logloss: 0.151751
    [4]	valid_0's auc: 0.841579	valid_0's binary_logloss: 0.145192	valid_1's auc: 0.823237	valid_1's binary_logloss: 0.149159
    [5]	valid_0's auc: 0.843972	valid_0's binary_logloss: 0.142753	valid_1's auc: 0.82747	valid_1's binary_logloss: 0.147042
    [6]	valid_0's auc: 0.846408	valid_0's binary_logloss: 0.140687	valid_1's auc: 0.830402	valid_1's binary_logloss: 0.145398
    [7]	valid_0's auc: 0.848617	valid_0's binary_logloss: 0.138978	valid_1's auc: 0.830319	valid_1's binary_logloss: 0.144054
    [8]	valid_0's auc: 0.849973	valid_0's binary_logloss: 0.137527	valid_1's auc: 0.831414	valid_1's binary_logloss: 0.142955
    [9]	valid_0's auc: 0.852151	valid_0's binary_logloss: 0.136279	valid_1's auc: 0.834577	valid_1's binary_logloss: 0.141825
    [10]	valid_0's auc: 0.852962	valid_0's binary_logloss: 0.135231	valid_1's auc: 0.835301	valid_1's binary_logloss: 0.140947
    [11]	valid_0's auc: 0.853769	valid_0's binary_logloss: 0.134247	valid_1's auc: 0.836379	valid_1's binary_logloss: 0.140121
    [12]	valid_0's auc: 0.855739	valid_0's binary_logloss: 0.133341	valid_1's auc: 0.83615	valid_1's binary_logloss: 0.139629
    [13]	valid_0's auc: 0.857549	valid_0's binary_logloss: 0.132607	valid_1's auc: 0.836232	valid_1's binary_logloss: 0.139209
    [14]	valid_0's auc: 0.858955	valid_0's binary_logloss: 0.131867	valid_1's auc: 0.836387	valid_1's binary_logloss: 0.138827
    [15]	valid_0's auc: 0.860018	valid_0's binary_logloss: 0.13125	valid_1's auc: 0.837299	valid_1's binary_logloss: 0.138441
    [16]	valid_0's auc: 0.861059	valid_0's binary_logloss: 0.130679	valid_1's auc: 0.838197	valid_1's binary_logloss: 0.138074
    [17]	valid_0's auc: 0.862112	valid_0's binary_logloss: 0.130128	valid_1's auc: 0.838835	valid_1's binary_logloss: 0.137791
    [18]	valid_0's auc: 0.862994	valid_0's binary_logloss: 0.129646	valid_1's auc: 0.838904	valid_1's binary_logloss: 0.137577
    [19]	valid_0's auc: 0.863568	valid_0's binary_logloss: 0.129195	valid_1's auc: 0.839028	valid_1's binary_logloss: 0.137319
    [20]	valid_0's auc: 0.86472	valid_0's binary_logloss: 0.128729	valid_1's auc: 0.83984	valid_1's binary_logloss: 0.137089
    [21]	valid_0's auc: 0.866051	valid_0's binary_logloss: 0.128315	valid_1's auc: 0.840385	valid_1's binary_logloss: 0.136889
    [22]	valid_0's auc: 0.866997	valid_0's binary_logloss: 0.127908	valid_1's auc: 0.840544	valid_1's binary_logloss: 0.136724
    [23]	valid_0's auc: 0.867998	valid_0's binary_logloss: 0.127516	valid_1's auc: 0.840471	valid_1's binary_logloss: 0.136628
    [24]	valid_0's auc: 0.868835	valid_0's binary_logloss: 0.127186	valid_1's auc: 0.84011	valid_1's binary_logloss: 0.136573
    [25]	valid_0's auc: 0.869535	valid_0's binary_logloss: 0.126855	valid_1's auc: 0.83987	valid_1's binary_logloss: 0.136502
    [26]	valid_0's auc: 0.870439	valid_0's binary_logloss: 0.126515	valid_1's auc: 0.840538	valid_1's binary_logloss: 0.136387
    [27]	valid_0's auc: 0.871466	valid_0's binary_logloss: 0.126179	valid_1's auc: 0.840326	valid_1's binary_logloss: 0.136301
    [28]	valid_0's auc: 0.872022	valid_0's binary_logloss: 0.125897	valid_1's auc: 0.840415	valid_1's binary_logloss: 0.136253
    [29]	valid_0's auc: 0.873022	valid_0's binary_logloss: 0.125624	valid_1's auc: 0.840531	valid_1's binary_logloss: 0.136185
    [30]	valid_0's auc: 0.873626	valid_0's binary_logloss: 0.125329	valid_1's auc: 0.840407	valid_1's binary_logloss: 0.136188
    [31]	valid_0's auc: 0.874227	valid_0's binary_logloss: 0.12509	valid_1's auc: 0.841102	valid_1's binary_logloss: 0.136046
    [32]	valid_0's auc: 0.874828	valid_0's binary_logloss: 0.124856	valid_1's auc: 0.841125	valid_1's binary_logloss: 0.135986
    [33]	valid_0's auc: 0.875492	valid_0's binary_logloss: 0.124583	valid_1's auc: 0.841558	valid_1's binary_logloss: 0.135907
    [34]	valid_0's auc: 0.876123	valid_0's binary_logloss: 0.124343	valid_1's auc: 0.841695	valid_1's binary_logloss: 0.135865
    [35]	valid_0's auc: 0.876893	valid_0's binary_logloss: 0.124064	valid_1's auc: 0.841945	valid_1's binary_logloss: 0.135831
    [36]	valid_0's auc: 0.877421	valid_0's binary_logloss: 0.123823	valid_1's auc: 0.841949	valid_1's binary_logloss: 0.135864
    [37]	valid_0's auc: 0.877945	valid_0's binary_logloss: 0.123612	valid_1's auc: 0.842028	valid_1's binary_logloss: 0.135826
    [38]	valid_0's auc: 0.878391	valid_0's binary_logloss: 0.123437	valid_1's auc: 0.841969	valid_1's binary_logloss: 0.135815
    [39]	valid_0's auc: 0.878886	valid_0's binary_logloss: 0.123211	valid_1's auc: 0.84193	valid_1's binary_logloss: 0.135809
    [40]	valid_0's auc: 0.879487	valid_0's binary_logloss: 0.123003	valid_1's auc: 0.841846	valid_1's binary_logloss: 0.135816
    [41]	valid_0's auc: 0.880192	valid_0's binary_logloss: 0.12276	valid_1's auc: 0.841501	valid_1's binary_logloss: 0.135878
    [42]	valid_0's auc: 0.880817	valid_0's binary_logloss: 0.122555	valid_1's auc: 0.841224	valid_1's binary_logloss: 0.135894
    [43]	valid_0's auc: 0.881168	valid_0's binary_logloss: 0.122344	valid_1's auc: 0.841189	valid_1's binary_logloss: 0.135895
    [44]	valid_0's auc: 0.88156	valid_0's binary_logloss: 0.122151	valid_1's auc: 0.841136	valid_1's binary_logloss: 0.135914
    [45]	valid_0's auc: 0.882438	valid_0's binary_logloss: 0.121855	valid_1's auc: 0.840553	valid_1's binary_logloss: 0.136015
    [46]	valid_0's auc: 0.883226	valid_0's binary_logloss: 0.121617	valid_1's auc: 0.840057	valid_1's binary_logloss: 0.136084
    [47]	valid_0's auc: 0.883701	valid_0's binary_logloss: 0.121428	valid_1's auc: 0.839905	valid_1's binary_logloss: 0.13611
    [48]	valid_0's auc: 0.884405	valid_0's binary_logloss: 0.121174	valid_1's auc: 0.840003	valid_1's binary_logloss: 0.136077
    [49]	valid_0's auc: 0.8847	valid_0's binary_logloss: 0.12101	valid_1's auc: 0.839683	valid_1's binary_logloss: 0.136145
    [50]	valid_0's auc: 0.885111	valid_0's binary_logloss: 0.120814	valid_1's auc: 0.839749	valid_1's binary_logloss: 0.136099
    [51]	valid_0's auc: 0.885425	valid_0's binary_logloss: 0.12065	valid_1's auc: 0.839416	valid_1's binary_logloss: 0.13609
    [52]	valid_0's auc: 0.885826	valid_0's binary_logloss: 0.120466	valid_1's auc: 0.83916	valid_1's binary_logloss: 0.136158
    [53]	valid_0's auc: 0.88609	valid_0's binary_logloss: 0.120311	valid_1's auc: 0.839381	valid_1's binary_logloss: 0.136151
    [54]	valid_0's auc: 0.886458	valid_0's binary_logloss: 0.120122	valid_1's auc: 0.839222	valid_1's binary_logloss: 0.136197
    [55]	valid_0's auc: 0.887239	valid_0's binary_logloss: 0.11994	valid_1's auc: 0.839191	valid_1's binary_logloss: 0.136207
    [56]	valid_0's auc: 0.887885	valid_0's binary_logloss: 0.119726	valid_1's auc: 0.839099	valid_1's binary_logloss: 0.136256
    [57]	valid_0's auc: 0.888068	valid_0's binary_logloss: 0.119592	valid_1's auc: 0.83885	valid_1's binary_logloss: 0.136275
    [58]	valid_0's auc: 0.888634	valid_0's binary_logloss: 0.119455	valid_1's auc: 0.838851	valid_1's binary_logloss: 0.136269
    [59]	valid_0's auc: 0.888873	valid_0's binary_logloss: 0.119334	valid_1's auc: 0.838683	valid_1's binary_logloss: 0.13629
    [60]	valid_0's auc: 0.889172	valid_0's binary_logloss: 0.119194	valid_1's auc: 0.838562	valid_1's binary_logloss: 0.136301
    [61]	valid_0's auc: 0.889511	valid_0's binary_logloss: 0.119036	valid_1's auc: 0.838472	valid_1's binary_logloss: 0.136335
    [62]	valid_0's auc: 0.889884	valid_0's binary_logloss: 0.118877	valid_1's auc: 0.838355	valid_1's binary_logloss: 0.136343
    [63]	valid_0's auc: 0.890436	valid_0's binary_logloss: 0.118679	valid_1's auc: 0.838056	valid_1's binary_logloss: 0.136425
    [64]	valid_0's auc: 0.890662	valid_0's binary_logloss: 0.118521	valid_1's auc: 0.837425	valid_1's binary_logloss: 0.136547
    [65]	valid_0's auc: 0.890862	valid_0's binary_logloss: 0.118389	valid_1's auc: 0.837399	valid_1's binary_logloss: 0.13655
    [66]	valid_0's auc: 0.890931	valid_0's binary_logloss: 0.118288	valid_1's auc: 0.837035	valid_1's binary_logloss: 0.13665
    [67]	valid_0's auc: 0.891015	valid_0's binary_logloss: 0.118183	valid_1's auc: 0.836707	valid_1's binary_logloss: 0.136783
    Early stopping, best iteration is:
    [37]	valid_0's auc: 0.877945	valid_0's binary_logloss: 0.123612	valid_1's auc: 0.842028	valid_1's binary_logloss: 0.135826
    [1]	valid_0's auc: 0.823507	valid_0's binary_logloss: 0.157501	valid_1's auc: 0.810415	valid_1's binary_logloss: 0.160155
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.829581	valid_0's binary_logloss: 0.152083	valid_1's auc: 0.815091	valid_1's binary_logloss: 0.155417
    [3]	valid_0's auc: 0.834308	valid_0's binary_logloss: 0.148148	valid_1's auc: 0.818622	valid_1's binary_logloss: 0.152027
    [4]	valid_0's auc: 0.836859	valid_0's binary_logloss: 0.145159	valid_1's auc: 0.819436	valid_1's binary_logloss: 0.149485
    [5]	valid_0's auc: 0.841846	valid_0's binary_logloss: 0.142678	valid_1's auc: 0.823324	valid_1's binary_logloss: 0.147372
    [6]	valid_0's auc: 0.844625	valid_0's binary_logloss: 0.140636	valid_1's auc: 0.824092	valid_1's binary_logloss: 0.145715
    [7]	valid_0's auc: 0.848223	valid_0's binary_logloss: 0.138877	valid_1's auc: 0.826887	valid_1's binary_logloss: 0.144253
    [8]	valid_0's auc: 0.849635	valid_0's binary_logloss: 0.137406	valid_1's auc: 0.829164	valid_1's binary_logloss: 0.143039
    [9]	valid_0's auc: 0.850865	valid_0's binary_logloss: 0.136105	valid_1's auc: 0.828916	valid_1's binary_logloss: 0.142109
    [10]	valid_0's auc: 0.852877	valid_0's binary_logloss: 0.134962	valid_1's auc: 0.829565	valid_1's binary_logloss: 0.141319
    [11]	valid_0's auc: 0.853498	valid_0's binary_logloss: 0.134001	valid_1's auc: 0.829604	valid_1's binary_logloss: 0.140625
    [12]	valid_0's auc: 0.854677	valid_0's binary_logloss: 0.133137	valid_1's auc: 0.828978	valid_1's binary_logloss: 0.140133
    [13]	valid_0's auc: 0.856296	valid_0's binary_logloss: 0.132367	valid_1's auc: 0.830184	valid_1's binary_logloss: 0.139696
    [14]	valid_0's auc: 0.858364	valid_0's binary_logloss: 0.131592	valid_1's auc: 0.831576	valid_1's binary_logloss: 0.139209
    [15]	valid_0's auc: 0.859632	valid_0's binary_logloss: 0.130943	valid_1's auc: 0.833147	valid_1's binary_logloss: 0.138759
    [16]	valid_0's auc: 0.86077	valid_0's binary_logloss: 0.130306	valid_1's auc: 0.833392	valid_1's binary_logloss: 0.138459
    [17]	valid_0's auc: 0.861722	valid_0's binary_logloss: 0.129793	valid_1's auc: 0.834279	valid_1's binary_logloss: 0.138085
    [18]	valid_0's auc: 0.862253	valid_0's binary_logloss: 0.129343	valid_1's auc: 0.834323	valid_1's binary_logloss: 0.137866
    [19]	valid_0's auc: 0.863766	valid_0's binary_logloss: 0.128824	valid_1's auc: 0.834736	valid_1's binary_logloss: 0.13763
    [20]	valid_0's auc: 0.8643	valid_0's binary_logloss: 0.128423	valid_1's auc: 0.834528	valid_1's binary_logloss: 0.137554
    [21]	valid_0's auc: 0.865267	valid_0's binary_logloss: 0.127958	valid_1's auc: 0.834478	valid_1's binary_logloss: 0.137406
    [22]	valid_0's auc: 0.866608	valid_0's binary_logloss: 0.127506	valid_1's auc: 0.834905	valid_1's binary_logloss: 0.137271
    [23]	valid_0's auc: 0.867804	valid_0's binary_logloss: 0.127104	valid_1's auc: 0.834293	valid_1's binary_logloss: 0.137211
    [24]	valid_0's auc: 0.86846	valid_0's binary_logloss: 0.12677	valid_1's auc: 0.833842	valid_1's binary_logloss: 0.137176
    [25]	valid_0's auc: 0.869535	valid_0's binary_logloss: 0.126409	valid_1's auc: 0.834057	valid_1's binary_logloss: 0.13708
    [26]	valid_0's auc: 0.870946	valid_0's binary_logloss: 0.126026	valid_1's auc: 0.833797	valid_1's binary_logloss: 0.137021
    [27]	valid_0's auc: 0.871863	valid_0's binary_logloss: 0.12567	valid_1's auc: 0.83361	valid_1's binary_logloss: 0.13702
    [28]	valid_0's auc: 0.872503	valid_0's binary_logloss: 0.125332	valid_1's auc: 0.833415	valid_1's binary_logloss: 0.136948
    [29]	valid_0's auc: 0.873256	valid_0's binary_logloss: 0.125049	valid_1's auc: 0.83344	valid_1's binary_logloss: 0.136909
    [30]	valid_0's auc: 0.874265	valid_0's binary_logloss: 0.124767	valid_1's auc: 0.833129	valid_1's binary_logloss: 0.136924
    [31]	valid_0's auc: 0.875083	valid_0's binary_logloss: 0.124478	valid_1's auc: 0.833207	valid_1's binary_logloss: 0.136915
    [32]	valid_0's auc: 0.875789	valid_0's binary_logloss: 0.124224	valid_1's auc: 0.833079	valid_1's binary_logloss: 0.136916
    [33]	valid_0's auc: 0.876187	valid_0's binary_logloss: 0.123994	valid_1's auc: 0.833172	valid_1's binary_logloss: 0.136856
    [34]	valid_0's auc: 0.876733	valid_0's binary_logloss: 0.123732	valid_1's auc: 0.833412	valid_1's binary_logloss: 0.136808
    [35]	valid_0's auc: 0.877384	valid_0's binary_logloss: 0.123461	valid_1's auc: 0.833287	valid_1's binary_logloss: 0.136785
    [36]	valid_0's auc: 0.877802	valid_0's binary_logloss: 0.123212	valid_1's auc: 0.83302	valid_1's binary_logloss: 0.13685
    [37]	valid_0's auc: 0.878799	valid_0's binary_logloss: 0.122947	valid_1's auc: 0.832295	valid_1's binary_logloss: 0.136905
    [38]	valid_0's auc: 0.879503	valid_0's binary_logloss: 0.122719	valid_1's auc: 0.832134	valid_1's binary_logloss: 0.136919
    [39]	valid_0's auc: 0.880358	valid_0's binary_logloss: 0.122488	valid_1's auc: 0.832266	valid_1's binary_logloss: 0.136938
    [40]	valid_0's auc: 0.881013	valid_0's binary_logloss: 0.122287	valid_1's auc: 0.832133	valid_1's binary_logloss: 0.136932
    [41]	valid_0's auc: 0.881424	valid_0's binary_logloss: 0.122107	valid_1's auc: 0.832209	valid_1's binary_logloss: 0.136934
    [42]	valid_0's auc: 0.881984	valid_0's binary_logloss: 0.121921	valid_1's auc: 0.831997	valid_1's binary_logloss: 0.136976
    [43]	valid_0's auc: 0.882635	valid_0's binary_logloss: 0.121694	valid_1's auc: 0.831527	valid_1's binary_logloss: 0.137038
    [44]	valid_0's auc: 0.883243	valid_0's binary_logloss: 0.121523	valid_1's auc: 0.831593	valid_1's binary_logloss: 0.137055
    [45]	valid_0's auc: 0.883616	valid_0's binary_logloss: 0.121345	valid_1's auc: 0.831225	valid_1's binary_logloss: 0.137134
    [46]	valid_0's auc: 0.884106	valid_0's binary_logloss: 0.121155	valid_1's auc: 0.831678	valid_1's binary_logloss: 0.137064
    [47]	valid_0's auc: 0.884462	valid_0's binary_logloss: 0.120968	valid_1's auc: 0.832127	valid_1's binary_logloss: 0.137012
    [48]	valid_0's auc: 0.884804	valid_0's binary_logloss: 0.120754	valid_1's auc: 0.832474	valid_1's binary_logloss: 0.136991
    [49]	valid_0's auc: 0.885391	valid_0's binary_logloss: 0.120518	valid_1's auc: 0.832375	valid_1's binary_logloss: 0.137054
    [50]	valid_0's auc: 0.885988	valid_0's binary_logloss: 0.120287	valid_1's auc: 0.832272	valid_1's binary_logloss: 0.137065
    [51]	valid_0's auc: 0.886393	valid_0's binary_logloss: 0.120131	valid_1's auc: 0.831867	valid_1's binary_logloss: 0.137159
    [52]	valid_0's auc: 0.886879	valid_0's binary_logloss: 0.119924	valid_1's auc: 0.831971	valid_1's binary_logloss: 0.137133
    Early stopping, best iteration is:
    [22]	valid_0's auc: 0.866608	valid_0's binary_logloss: 0.127506	valid_1's auc: 0.834905	valid_1's binary_logloss: 0.137271
    [1]	valid_0's auc: 0.825869	valid_0's binary_logloss: 0.156436	valid_1's auc: 0.803397	valid_1's binary_logloss: 0.159993
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.834474	valid_0's binary_logloss: 0.150597	valid_1's auc: 0.809972	valid_1's binary_logloss: 0.155122
    [3]	valid_0's auc: 0.840511	valid_0's binary_logloss: 0.146374	valid_1's auc: 0.819137	valid_1's binary_logloss: 0.15148
    [4]	valid_0's auc: 0.844283	valid_0's binary_logloss: 0.143027	valid_1's auc: 0.820472	valid_1's binary_logloss: 0.14904
    [5]	valid_0's auc: 0.85046	valid_0's binary_logloss: 0.140321	valid_1's auc: 0.82606	valid_1's binary_logloss: 0.146982
    [6]	valid_0's auc: 0.852881	valid_0's binary_logloss: 0.13813	valid_1's auc: 0.823869	valid_1's binary_logloss: 0.145517
    [7]	valid_0's auc: 0.857268	valid_0's binary_logloss: 0.136149	valid_1's auc: 0.827586	valid_1's binary_logloss: 0.144156
    [8]	valid_0's auc: 0.859401	valid_0's binary_logloss: 0.13452	valid_1's auc: 0.829428	valid_1's binary_logloss: 0.14303
    [9]	valid_0's auc: 0.861435	valid_0's binary_logloss: 0.133011	valid_1's auc: 0.82971	valid_1's binary_logloss: 0.142113
    [10]	valid_0's auc: 0.862282	valid_0's binary_logloss: 0.131742	valid_1's auc: 0.830527	valid_1's binary_logloss: 0.141392
    [11]	valid_0's auc: 0.86448	valid_0's binary_logloss: 0.130536	valid_1's auc: 0.83065	valid_1's binary_logloss: 0.140854
    [12]	valid_0's auc: 0.865909	valid_0's binary_logloss: 0.129495	valid_1's auc: 0.830763	valid_1's binary_logloss: 0.140425
    [13]	valid_0's auc: 0.867083	valid_0's binary_logloss: 0.128537	valid_1's auc: 0.830097	valid_1's binary_logloss: 0.140034
    [14]	valid_0's auc: 0.869164	valid_0's binary_logloss: 0.127593	valid_1's auc: 0.831242	valid_1's binary_logloss: 0.139589
    [15]	valid_0's auc: 0.869779	valid_0's binary_logloss: 0.126753	valid_1's auc: 0.830054	valid_1's binary_logloss: 0.13935
    [16]	valid_0's auc: 0.870798	valid_0's binary_logloss: 0.125998	valid_1's auc: 0.829854	valid_1's binary_logloss: 0.139147
    [17]	valid_0's auc: 0.872413	valid_0's binary_logloss: 0.125321	valid_1's auc: 0.829939	valid_1's binary_logloss: 0.138859
    [18]	valid_0's auc: 0.873656	valid_0's binary_logloss: 0.124646	valid_1's auc: 0.83033	valid_1's binary_logloss: 0.138678
    [19]	valid_0's auc: 0.874787	valid_0's binary_logloss: 0.124094	valid_1's auc: 0.829747	valid_1's binary_logloss: 0.138635
    [20]	valid_0's auc: 0.875793	valid_0's binary_logloss: 0.123547	valid_1's auc: 0.829455	valid_1's binary_logloss: 0.138492
    [21]	valid_0's auc: 0.87694	valid_0's binary_logloss: 0.122969	valid_1's auc: 0.830945	valid_1's binary_logloss: 0.13825
    [22]	valid_0's auc: 0.878632	valid_0's binary_logloss: 0.122407	valid_1's auc: 0.831483	valid_1's binary_logloss: 0.138097
    [23]	valid_0's auc: 0.8796	valid_0's binary_logloss: 0.121869	valid_1's auc: 0.831408	valid_1's binary_logloss: 0.138071
    [24]	valid_0's auc: 0.881298	valid_0's binary_logloss: 0.121363	valid_1's auc: 0.831984	valid_1's binary_logloss: 0.137912
    [25]	valid_0's auc: 0.882699	valid_0's binary_logloss: 0.120863	valid_1's auc: 0.831941	valid_1's binary_logloss: 0.137824
    [26]	valid_0's auc: 0.883799	valid_0's binary_logloss: 0.120379	valid_1's auc: 0.832897	valid_1's binary_logloss: 0.137649
    [27]	valid_0's auc: 0.884893	valid_0's binary_logloss: 0.119944	valid_1's auc: 0.832841	valid_1's binary_logloss: 0.13763
    [28]	valid_0's auc: 0.88588	valid_0's binary_logloss: 0.119545	valid_1's auc: 0.833413	valid_1's binary_logloss: 0.137495
    [29]	valid_0's auc: 0.886461	valid_0's binary_logloss: 0.11916	valid_1's auc: 0.833437	valid_1's binary_logloss: 0.137511
    [30]	valid_0's auc: 0.887592	valid_0's binary_logloss: 0.118739	valid_1's auc: 0.833429	valid_1's binary_logloss: 0.137514
    [31]	valid_0's auc: 0.888552	valid_0's binary_logloss: 0.118399	valid_1's auc: 0.834036	valid_1's binary_logloss: 0.137459
    [32]	valid_0's auc: 0.889448	valid_0's binary_logloss: 0.118019	valid_1's auc: 0.833754	valid_1's binary_logloss: 0.137498
    [33]	valid_0's auc: 0.890203	valid_0's binary_logloss: 0.117711	valid_1's auc: 0.833853	valid_1's binary_logloss: 0.137491
    [34]	valid_0's auc: 0.890868	valid_0's binary_logloss: 0.117395	valid_1's auc: 0.833648	valid_1's binary_logloss: 0.137532
    [35]	valid_0's auc: 0.891378	valid_0's binary_logloss: 0.117106	valid_1's auc: 0.833479	valid_1's binary_logloss: 0.137577
    [36]	valid_0's auc: 0.892061	valid_0's binary_logloss: 0.116741	valid_1's auc: 0.832608	valid_1's binary_logloss: 0.137799
    [37]	valid_0's auc: 0.892715	valid_0's binary_logloss: 0.116403	valid_1's auc: 0.83208	valid_1's binary_logloss: 0.137938
    [38]	valid_0's auc: 0.893146	valid_0's binary_logloss: 0.116126	valid_1's auc: 0.83196	valid_1's binary_logloss: 0.137994
    [39]	valid_0's auc: 0.893963	valid_0's binary_logloss: 0.115783	valid_1's auc: 0.831584	valid_1's binary_logloss: 0.138076
    [40]	valid_0's auc: 0.894274	valid_0's binary_logloss: 0.115535	valid_1's auc: 0.831346	valid_1's binary_logloss: 0.13818
    [41]	valid_0's auc: 0.894809	valid_0's binary_logloss: 0.115247	valid_1's auc: 0.831197	valid_1's binary_logloss: 0.138289
    [42]	valid_0's auc: 0.895332	valid_0's binary_logloss: 0.114916	valid_1's auc: 0.830893	valid_1's binary_logloss: 0.138343
    [43]	valid_0's auc: 0.895864	valid_0's binary_logloss: 0.114701	valid_1's auc: 0.83129	valid_1's binary_logloss: 0.138307
    [44]	valid_0's auc: 0.89623	valid_0's binary_logloss: 0.114411	valid_1's auc: 0.831004	valid_1's binary_logloss: 0.138408
    [45]	valid_0's auc: 0.896616	valid_0's binary_logloss: 0.11413	valid_1's auc: 0.830369	valid_1's binary_logloss: 0.138547
    [46]	valid_0's auc: 0.897076	valid_0's binary_logloss: 0.113919	valid_1's auc: 0.830246	valid_1's binary_logloss: 0.138644
    [47]	valid_0's auc: 0.897393	valid_0's binary_logloss: 0.113691	valid_1's auc: 0.829999	valid_1's binary_logloss: 0.138769
    [48]	valid_0's auc: 0.897745	valid_0's binary_logloss: 0.113465	valid_1's auc: 0.829593	valid_1's binary_logloss: 0.138908
    [49]	valid_0's auc: 0.898203	valid_0's binary_logloss: 0.113214	valid_1's auc: 0.829209	valid_1's binary_logloss: 0.139045
    [50]	valid_0's auc: 0.898546	valid_0's binary_logloss: 0.112978	valid_1's auc: 0.829064	valid_1's binary_logloss: 0.139115
    [51]	valid_0's auc: 0.898958	valid_0's binary_logloss: 0.112723	valid_1's auc: 0.82913	valid_1's binary_logloss: 0.139163
    [52]	valid_0's auc: 0.899296	valid_0's binary_logloss: 0.112541	valid_1's auc: 0.828532	valid_1's binary_logloss: 0.139358
    [53]	valid_0's auc: 0.899966	valid_0's binary_logloss: 0.112208	valid_1's auc: 0.828277	valid_1's binary_logloss: 0.139431
    [54]	valid_0's auc: 0.900225	valid_0's binary_logloss: 0.112001	valid_1's auc: 0.827866	valid_1's binary_logloss: 0.139573
    [55]	valid_0's auc: 0.90036	valid_0's binary_logloss: 0.111812	valid_1's auc: 0.827887	valid_1's binary_logloss: 0.139655
    [56]	valid_0's auc: 0.900528	valid_0's binary_logloss: 0.111638	valid_1's auc: 0.827809	valid_1's binary_logloss: 0.139702
    [57]	valid_0's auc: 0.900692	valid_0's binary_logloss: 0.111452	valid_1's auc: 0.827613	valid_1's binary_logloss: 0.139758
    [58]	valid_0's auc: 0.901275	valid_0's binary_logloss: 0.111262	valid_1's auc: 0.827848	valid_1's binary_logloss: 0.139758
    [59]	valid_0's auc: 0.90149	valid_0's binary_logloss: 0.111086	valid_1's auc: 0.827643	valid_1's binary_logloss: 0.139835
    [60]	valid_0's auc: 0.901669	valid_0's binary_logloss: 0.110873	valid_1's auc: 0.827187	valid_1's binary_logloss: 0.13998
    [61]	valid_0's auc: 0.901715	valid_0's binary_logloss: 0.110743	valid_1's auc: 0.826643	valid_1's binary_logloss: 0.140136
    Early stopping, best iteration is:
    [31]	valid_0's auc: 0.888552	valid_0's binary_logloss: 0.118399	valid_1's auc: 0.834036	valid_1's binary_logloss: 0.137459
    [1]	valid_0's auc: 0.828293	valid_0's binary_logloss: 0.156591	valid_1's auc: 0.803729	valid_1's binary_logloss: 0.159809
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.836848	valid_0's binary_logloss: 0.150955	valid_1's auc: 0.813468	valid_1's binary_logloss: 0.155406
    [3]	valid_0's auc: 0.842786	valid_0's binary_logloss: 0.146796	valid_1's auc: 0.817407	valid_1's binary_logloss: 0.151952
    [4]	valid_0's auc: 0.851348	valid_0's binary_logloss: 0.143395	valid_1's auc: 0.823949	valid_1's binary_logloss: 0.14934
    [5]	valid_0's auc: 0.855149	valid_0's binary_logloss: 0.140626	valid_1's auc: 0.827122	valid_1's binary_logloss: 0.147185
    [6]	valid_0's auc: 0.857679	valid_0's binary_logloss: 0.13837	valid_1's auc: 0.826758	valid_1's binary_logloss: 0.145708
    [7]	valid_0's auc: 0.86155	valid_0's binary_logloss: 0.136445	valid_1's auc: 0.828472	valid_1's binary_logloss: 0.144294
    [8]	valid_0's auc: 0.862588	valid_0's binary_logloss: 0.134808	valid_1's auc: 0.828623	valid_1's binary_logloss: 0.143251
    [9]	valid_0's auc: 0.865115	valid_0's binary_logloss: 0.133359	valid_1's auc: 0.831471	valid_1's binary_logloss: 0.142276
    [10]	valid_0's auc: 0.866894	valid_0's binary_logloss: 0.132073	valid_1's auc: 0.831985	valid_1's binary_logloss: 0.141408
    [11]	valid_0's auc: 0.869042	valid_0's binary_logloss: 0.130873	valid_1's auc: 0.834107	valid_1's binary_logloss: 0.140661
    [12]	valid_0's auc: 0.870884	valid_0's binary_logloss: 0.129808	valid_1's auc: 0.833841	valid_1's binary_logloss: 0.140137
    [13]	valid_0's auc: 0.87235	valid_0's binary_logloss: 0.128859	valid_1's auc: 0.834392	valid_1's binary_logloss: 0.139657
    [14]	valid_0's auc: 0.873638	valid_0's binary_logloss: 0.128012	valid_1's auc: 0.834569	valid_1's binary_logloss: 0.13928
    [15]	valid_0's auc: 0.874792	valid_0's binary_logloss: 0.127177	valid_1's auc: 0.83417	valid_1's binary_logloss: 0.138989
    [16]	valid_0's auc: 0.876361	valid_0's binary_logloss: 0.126304	valid_1's auc: 0.835986	valid_1's binary_logloss: 0.138606
    [17]	valid_0's auc: 0.878328	valid_0's binary_logloss: 0.125589	valid_1's auc: 0.836721	valid_1's binary_logloss: 0.138362
    [18]	valid_0's auc: 0.879835	valid_0's binary_logloss: 0.124908	valid_1's auc: 0.836975	valid_1's binary_logloss: 0.138199
    [19]	valid_0's auc: 0.881056	valid_0's binary_logloss: 0.124303	valid_1's auc: 0.837163	valid_1's binary_logloss: 0.137993
    [20]	valid_0's auc: 0.881684	valid_0's binary_logloss: 0.123706	valid_1's auc: 0.836767	valid_1's binary_logloss: 0.137922
    [21]	valid_0's auc: 0.882852	valid_0's binary_logloss: 0.123138	valid_1's auc: 0.835726	valid_1's binary_logloss: 0.137937
    [22]	valid_0's auc: 0.883782	valid_0's binary_logloss: 0.122621	valid_1's auc: 0.836033	valid_1's binary_logloss: 0.137832
    [23]	valid_0's auc: 0.884894	valid_0's binary_logloss: 0.122101	valid_1's auc: 0.835965	valid_1's binary_logloss: 0.137761
    [24]	valid_0's auc: 0.886129	valid_0's binary_logloss: 0.121539	valid_1's auc: 0.836276	valid_1's binary_logloss: 0.137662
    [25]	valid_0's auc: 0.887299	valid_0's binary_logloss: 0.121003	valid_1's auc: 0.836494	valid_1's binary_logloss: 0.137562
    [26]	valid_0's auc: 0.888167	valid_0's binary_logloss: 0.120533	valid_1's auc: 0.836735	valid_1's binary_logloss: 0.137471
    [27]	valid_0's auc: 0.889209	valid_0's binary_logloss: 0.120033	valid_1's auc: 0.837077	valid_1's binary_logloss: 0.137335
    [28]	valid_0's auc: 0.890083	valid_0's binary_logloss: 0.119628	valid_1's auc: 0.836646	valid_1's binary_logloss: 0.137396
    [29]	valid_0's auc: 0.891268	valid_0's binary_logloss: 0.1192	valid_1's auc: 0.836893	valid_1's binary_logloss: 0.137334
    [30]	valid_0's auc: 0.892026	valid_0's binary_logloss: 0.118779	valid_1's auc: 0.836987	valid_1's binary_logloss: 0.13734
    [31]	valid_0's auc: 0.892612	valid_0's binary_logloss: 0.1184	valid_1's auc: 0.835995	valid_1's binary_logloss: 0.137457
    [32]	valid_0's auc: 0.893409	valid_0's binary_logloss: 0.118028	valid_1's auc: 0.836329	valid_1's binary_logloss: 0.137426
    [33]	valid_0's auc: 0.894715	valid_0's binary_logloss: 0.117634	valid_1's auc: 0.836237	valid_1's binary_logloss: 0.137428
    [34]	valid_0's auc: 0.896129	valid_0's binary_logloss: 0.117239	valid_1's auc: 0.837278	valid_1's binary_logloss: 0.137234
    [35]	valid_0's auc: 0.897111	valid_0's binary_logloss: 0.116881	valid_1's auc: 0.837313	valid_1's binary_logloss: 0.137208
    [36]	valid_0's auc: 0.89791	valid_0's binary_logloss: 0.116525	valid_1's auc: 0.837094	valid_1's binary_logloss: 0.137227
    [37]	valid_0's auc: 0.898327	valid_0's binary_logloss: 0.116184	valid_1's auc: 0.836808	valid_1's binary_logloss: 0.137267
    [38]	valid_0's auc: 0.899162	valid_0's binary_logloss: 0.115859	valid_1's auc: 0.836473	valid_1's binary_logloss: 0.137323
    [39]	valid_0's auc: 0.899593	valid_0's binary_logloss: 0.115562	valid_1's auc: 0.836433	valid_1's binary_logloss: 0.137356
    [40]	valid_0's auc: 0.899942	valid_0's binary_logloss: 0.115297	valid_1's auc: 0.836299	valid_1's binary_logloss: 0.137382
    [41]	valid_0's auc: 0.900439	valid_0's binary_logloss: 0.114998	valid_1's auc: 0.835877	valid_1's binary_logloss: 0.137474
    [42]	valid_0's auc: 0.90094	valid_0's binary_logloss: 0.1147	valid_1's auc: 0.835016	valid_1's binary_logloss: 0.137628
    [43]	valid_0's auc: 0.901223	valid_0's binary_logloss: 0.11447	valid_1's auc: 0.834849	valid_1's binary_logloss: 0.137661
    [44]	valid_0's auc: 0.901438	valid_0's binary_logloss: 0.11423	valid_1's auc: 0.834743	valid_1's binary_logloss: 0.137699
    [45]	valid_0's auc: 0.902058	valid_0's binary_logloss: 0.113982	valid_1's auc: 0.834615	valid_1's binary_logloss: 0.13774
    [46]	valid_0's auc: 0.902431	valid_0's binary_logloss: 0.113713	valid_1's auc: 0.834207	valid_1's binary_logloss: 0.13784
    [47]	valid_0's auc: 0.902812	valid_0's binary_logloss: 0.113487	valid_1's auc: 0.833758	valid_1's binary_logloss: 0.137985
    [48]	valid_0's auc: 0.90306	valid_0's binary_logloss: 0.113301	valid_1's auc: 0.833219	valid_1's binary_logloss: 0.138122
    [49]	valid_0's auc: 0.903267	valid_0's binary_logloss: 0.113063	valid_1's auc: 0.832937	valid_1's binary_logloss: 0.138197
    [50]	valid_0's auc: 0.903705	valid_0's binary_logloss: 0.112863	valid_1's auc: 0.832815	valid_1's binary_logloss: 0.13824
    [51]	valid_0's auc: 0.904223	valid_0's binary_logloss: 0.112624	valid_1's auc: 0.832789	valid_1's binary_logloss: 0.138311
    [52]	valid_0's auc: 0.904867	valid_0's binary_logloss: 0.112372	valid_1's auc: 0.832399	valid_1's binary_logloss: 0.138416
    [53]	valid_0's auc: 0.90521	valid_0's binary_logloss: 0.112133	valid_1's auc: 0.832296	valid_1's binary_logloss: 0.1385
    [54]	valid_0's auc: 0.905689	valid_0's binary_logloss: 0.111851	valid_1's auc: 0.831911	valid_1's binary_logloss: 0.138563
    [55]	valid_0's auc: 0.905941	valid_0's binary_logloss: 0.111672	valid_1's auc: 0.831691	valid_1's binary_logloss: 0.138659
    [56]	valid_0's auc: 0.906184	valid_0's binary_logloss: 0.111482	valid_1's auc: 0.83155	valid_1's binary_logloss: 0.138723
    [57]	valid_0's auc: 0.906352	valid_0's binary_logloss: 0.111271	valid_1's auc: 0.830993	valid_1's binary_logloss: 0.138832
    [58]	valid_0's auc: 0.906874	valid_0's binary_logloss: 0.110997	valid_1's auc: 0.831151	valid_1's binary_logloss: 0.138806
    [59]	valid_0's auc: 0.906995	valid_0's binary_logloss: 0.110797	valid_1's auc: 0.830642	valid_1's binary_logloss: 0.138926
    [60]	valid_0's auc: 0.907239	valid_0's binary_logloss: 0.110639	valid_1's auc: 0.830434	valid_1's binary_logloss: 0.139017
    [61]	valid_0's auc: 0.907442	valid_0's binary_logloss: 0.110465	valid_1's auc: 0.829813	valid_1's binary_logloss: 0.13916
    [62]	valid_0's auc: 0.908053	valid_0's binary_logloss: 0.110189	valid_1's auc: 0.829969	valid_1's binary_logloss: 0.139161
    [63]	valid_0's auc: 0.908139	valid_0's binary_logloss: 0.110026	valid_1's auc: 0.829479	valid_1's binary_logloss: 0.139308
    [64]	valid_0's auc: 0.908183	valid_0's binary_logloss: 0.109892	valid_1's auc: 0.828768	valid_1's binary_logloss: 0.139496
    [65]	valid_0's auc: 0.908915	valid_0's binary_logloss: 0.109601	valid_1's auc: 0.828359	valid_1's binary_logloss: 0.139582
    Early stopping, best iteration is:
    [35]	valid_0's auc: 0.897111	valid_0's binary_logloss: 0.116881	valid_1's auc: 0.837313	valid_1's binary_logloss: 0.137208
    [1]	valid_0's auc: 0.833154	valid_0's binary_logloss: 0.156832	valid_1's auc: 0.809041	valid_1's binary_logloss: 0.159834
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.842696	valid_0's binary_logloss: 0.15097	valid_1's auc: 0.816072	valid_1's binary_logloss: 0.155054
    [3]	valid_0's auc: 0.848628	valid_0's binary_logloss: 0.146679	valid_1's auc: 0.821987	valid_1's binary_logloss: 0.151615
    [4]	valid_0's auc: 0.852189	valid_0's binary_logloss: 0.143409	valid_1's auc: 0.823257	valid_1's binary_logloss: 0.148971
    [5]	valid_0's auc: 0.85445	valid_0's binary_logloss: 0.140691	valid_1's auc: 0.826693	valid_1's binary_logloss: 0.146922
    [6]	valid_0's auc: 0.856663	valid_0's binary_logloss: 0.138477	valid_1's auc: 0.827879	valid_1's binary_logloss: 0.145076
    [7]	valid_0's auc: 0.858141	valid_0's binary_logloss: 0.136527	valid_1's auc: 0.828071	valid_1's binary_logloss: 0.143771
    [8]	valid_0's auc: 0.860436	valid_0's binary_logloss: 0.134845	valid_1's auc: 0.830521	valid_1's binary_logloss: 0.142677
    [9]	valid_0's auc: 0.861929	valid_0's binary_logloss: 0.133357	valid_1's auc: 0.831252	valid_1's binary_logloss: 0.14175
    [10]	valid_0's auc: 0.865268	valid_0's binary_logloss: 0.132002	valid_1's auc: 0.831065	valid_1's binary_logloss: 0.141032
    [11]	valid_0's auc: 0.86773	valid_0's binary_logloss: 0.130767	valid_1's auc: 0.831969	valid_1's binary_logloss: 0.140477
    [12]	valid_0's auc: 0.869797	valid_0's binary_logloss: 0.129639	valid_1's auc: 0.831747	valid_1's binary_logloss: 0.139964
    [13]	valid_0's auc: 0.871962	valid_0's binary_logloss: 0.128704	valid_1's auc: 0.832299	valid_1's binary_logloss: 0.13953
    [14]	valid_0's auc: 0.873656	valid_0's binary_logloss: 0.12785	valid_1's auc: 0.831874	valid_1's binary_logloss: 0.139277
    [15]	valid_0's auc: 0.875313	valid_0's binary_logloss: 0.127041	valid_1's auc: 0.831922	valid_1's binary_logloss: 0.139006
    [16]	valid_0's auc: 0.875813	valid_0's binary_logloss: 0.126291	valid_1's auc: 0.832339	valid_1's binary_logloss: 0.138679
    [17]	valid_0's auc: 0.877574	valid_0's binary_logloss: 0.125606	valid_1's auc: 0.832459	valid_1's binary_logloss: 0.138444
    [18]	valid_0's auc: 0.879134	valid_0's binary_logloss: 0.12489	valid_1's auc: 0.832409	valid_1's binary_logloss: 0.138268
    [19]	valid_0's auc: 0.880176	valid_0's binary_logloss: 0.124247	valid_1's auc: 0.832557	valid_1's binary_logloss: 0.138164
    [20]	valid_0's auc: 0.881608	valid_0's binary_logloss: 0.123609	valid_1's auc: 0.832053	valid_1's binary_logloss: 0.138136
    [21]	valid_0's auc: 0.882632	valid_0's binary_logloss: 0.123013	valid_1's auc: 0.831852	valid_1's binary_logloss: 0.138081
    [22]	valid_0's auc: 0.883684	valid_0's binary_logloss: 0.12245	valid_1's auc: 0.832034	valid_1's binary_logloss: 0.137978
    [23]	valid_0's auc: 0.885039	valid_0's binary_logloss: 0.121827	valid_1's auc: 0.831422	valid_1's binary_logloss: 0.138011
    [24]	valid_0's auc: 0.886247	valid_0's binary_logloss: 0.121291	valid_1's auc: 0.831253	valid_1's binary_logloss: 0.137942
    [25]	valid_0's auc: 0.887301	valid_0's binary_logloss: 0.120809	valid_1's auc: 0.831276	valid_1's binary_logloss: 0.137924
    [26]	valid_0's auc: 0.888621	valid_0's binary_logloss: 0.120287	valid_1's auc: 0.830735	valid_1's binary_logloss: 0.137975
    [27]	valid_0's auc: 0.889227	valid_0's binary_logloss: 0.119878	valid_1's auc: 0.830569	valid_1's binary_logloss: 0.137976
    [28]	valid_0's auc: 0.890381	valid_0's binary_logloss: 0.119445	valid_1's auc: 0.830281	valid_1's binary_logloss: 0.137968
    [29]	valid_0's auc: 0.891139	valid_0's binary_logloss: 0.119016	valid_1's auc: 0.830916	valid_1's binary_logloss: 0.137839
    [30]	valid_0's auc: 0.892417	valid_0's binary_logloss: 0.118535	valid_1's auc: 0.830189	valid_1's binary_logloss: 0.137909
    [31]	valid_0's auc: 0.893102	valid_0's binary_logloss: 0.118157	valid_1's auc: 0.830117	valid_1's binary_logloss: 0.137933
    [32]	valid_0's auc: 0.893844	valid_0's binary_logloss: 0.117764	valid_1's auc: 0.829877	valid_1's binary_logloss: 0.137962
    [33]	valid_0's auc: 0.894508	valid_0's binary_logloss: 0.117419	valid_1's auc: 0.829552	valid_1's binary_logloss: 0.138044
    [34]	valid_0's auc: 0.895157	valid_0's binary_logloss: 0.117051	valid_1's auc: 0.830486	valid_1's binary_logloss: 0.137893
    [35]	valid_0's auc: 0.89588	valid_0's binary_logloss: 0.116679	valid_1's auc: 0.83041	valid_1's binary_logloss: 0.137896
    [36]	valid_0's auc: 0.896611	valid_0's binary_logloss: 0.116295	valid_1's auc: 0.829969	valid_1's binary_logloss: 0.138026
    [37]	valid_0's auc: 0.89762	valid_0's binary_logloss: 0.115979	valid_1's auc: 0.829802	valid_1's binary_logloss: 0.138089
    [38]	valid_0's auc: 0.898028	valid_0's binary_logloss: 0.115667	valid_1's auc: 0.829346	valid_1's binary_logloss: 0.138207
    [39]	valid_0's auc: 0.89839	valid_0's binary_logloss: 0.115401	valid_1's auc: 0.829208	valid_1's binary_logloss: 0.138191
    [40]	valid_0's auc: 0.898968	valid_0's binary_logloss: 0.115042	valid_1's auc: 0.828576	valid_1's binary_logloss: 0.138317
    [41]	valid_0's auc: 0.899373	valid_0's binary_logloss: 0.114766	valid_1's auc: 0.828382	valid_1's binary_logloss: 0.138408
    [42]	valid_0's auc: 0.900042	valid_0's binary_logloss: 0.114467	valid_1's auc: 0.828023	valid_1's binary_logloss: 0.138489
    [43]	valid_0's auc: 0.900488	valid_0's binary_logloss: 0.114194	valid_1's auc: 0.827815	valid_1's binary_logloss: 0.138591
    [44]	valid_0's auc: 0.901004	valid_0's binary_logloss: 0.113865	valid_1's auc: 0.827321	valid_1's binary_logloss: 0.138736
    [45]	valid_0's auc: 0.901401	valid_0's binary_logloss: 0.113616	valid_1's auc: 0.826875	valid_1's binary_logloss: 0.138874
    [46]	valid_0's auc: 0.901795	valid_0's binary_logloss: 0.113348	valid_1's auc: 0.82643	valid_1's binary_logloss: 0.138955
    [47]	valid_0's auc: 0.902244	valid_0's binary_logloss: 0.11309	valid_1's auc: 0.825825	valid_1's binary_logloss: 0.13908
    [48]	valid_0's auc: 0.902756	valid_0's binary_logloss: 0.112891	valid_1's auc: 0.825639	valid_1's binary_logloss: 0.139117
    [49]	valid_0's auc: 0.903206	valid_0's binary_logloss: 0.112607	valid_1's auc: 0.825607	valid_1's binary_logloss: 0.139184
    Early stopping, best iteration is:
    [19]	valid_0's auc: 0.880176	valid_0's binary_logloss: 0.124247	valid_1's auc: 0.832557	valid_1's binary_logloss: 0.138164
    [1]	valid_0's auc: 0.825869	valid_0's binary_logloss: 0.156436	valid_1's auc: 0.803397	valid_1's binary_logloss: 0.159993
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.834474	valid_0's binary_logloss: 0.150597	valid_1's auc: 0.809972	valid_1's binary_logloss: 0.155122
    [3]	valid_0's auc: 0.840511	valid_0's binary_logloss: 0.146374	valid_1's auc: 0.819137	valid_1's binary_logloss: 0.15148
    [4]	valid_0's auc: 0.844283	valid_0's binary_logloss: 0.143027	valid_1's auc: 0.820472	valid_1's binary_logloss: 0.14904
    [5]	valid_0's auc: 0.85046	valid_0's binary_logloss: 0.140321	valid_1's auc: 0.82606	valid_1's binary_logloss: 0.146982
    [6]	valid_0's auc: 0.852881	valid_0's binary_logloss: 0.13813	valid_1's auc: 0.823869	valid_1's binary_logloss: 0.145517
    [7]	valid_0's auc: 0.857268	valid_0's binary_logloss: 0.136149	valid_1's auc: 0.827586	valid_1's binary_logloss: 0.144156
    [8]	valid_0's auc: 0.859401	valid_0's binary_logloss: 0.13452	valid_1's auc: 0.829428	valid_1's binary_logloss: 0.14303
    [9]	valid_0's auc: 0.861435	valid_0's binary_logloss: 0.133011	valid_1's auc: 0.82971	valid_1's binary_logloss: 0.142113
    [10]	valid_0's auc: 0.862282	valid_0's binary_logloss: 0.131742	valid_1's auc: 0.830527	valid_1's binary_logloss: 0.141392
    [11]	valid_0's auc: 0.86448	valid_0's binary_logloss: 0.130536	valid_1's auc: 0.83065	valid_1's binary_logloss: 0.140854
    [12]	valid_0's auc: 0.865909	valid_0's binary_logloss: 0.129495	valid_1's auc: 0.830763	valid_1's binary_logloss: 0.140425
    [13]	valid_0's auc: 0.867083	valid_0's binary_logloss: 0.128537	valid_1's auc: 0.830097	valid_1's binary_logloss: 0.140034
    [14]	valid_0's auc: 0.869164	valid_0's binary_logloss: 0.127593	valid_1's auc: 0.831242	valid_1's binary_logloss: 0.139589
    [15]	valid_0's auc: 0.869779	valid_0's binary_logloss: 0.126753	valid_1's auc: 0.830054	valid_1's binary_logloss: 0.13935
    [16]	valid_0's auc: 0.870798	valid_0's binary_logloss: 0.125998	valid_1's auc: 0.829854	valid_1's binary_logloss: 0.139147
    [17]	valid_0's auc: 0.872413	valid_0's binary_logloss: 0.125321	valid_1's auc: 0.829939	valid_1's binary_logloss: 0.138859
    [18]	valid_0's auc: 0.873656	valid_0's binary_logloss: 0.124646	valid_1's auc: 0.83033	valid_1's binary_logloss: 0.138678
    [19]	valid_0's auc: 0.874787	valid_0's binary_logloss: 0.124094	valid_1's auc: 0.829747	valid_1's binary_logloss: 0.138635
    [20]	valid_0's auc: 0.875793	valid_0's binary_logloss: 0.123547	valid_1's auc: 0.829455	valid_1's binary_logloss: 0.138492
    [21]	valid_0's auc: 0.87694	valid_0's binary_logloss: 0.122969	valid_1's auc: 0.830945	valid_1's binary_logloss: 0.13825
    [22]	valid_0's auc: 0.878632	valid_0's binary_logloss: 0.122407	valid_1's auc: 0.831483	valid_1's binary_logloss: 0.138097
    [23]	valid_0's auc: 0.8796	valid_0's binary_logloss: 0.121869	valid_1's auc: 0.831408	valid_1's binary_logloss: 0.138071
    [24]	valid_0's auc: 0.881298	valid_0's binary_logloss: 0.121363	valid_1's auc: 0.831984	valid_1's binary_logloss: 0.137912
    [25]	valid_0's auc: 0.882699	valid_0's binary_logloss: 0.120863	valid_1's auc: 0.831941	valid_1's binary_logloss: 0.137824
    [26]	valid_0's auc: 0.883799	valid_0's binary_logloss: 0.120379	valid_1's auc: 0.832897	valid_1's binary_logloss: 0.137649
    [27]	valid_0's auc: 0.884893	valid_0's binary_logloss: 0.119944	valid_1's auc: 0.832841	valid_1's binary_logloss: 0.13763
    [28]	valid_0's auc: 0.88588	valid_0's binary_logloss: 0.119545	valid_1's auc: 0.833413	valid_1's binary_logloss: 0.137495
    [29]	valid_0's auc: 0.886461	valid_0's binary_logloss: 0.11916	valid_1's auc: 0.833437	valid_1's binary_logloss: 0.137511
    [30]	valid_0's auc: 0.887592	valid_0's binary_logloss: 0.118739	valid_1's auc: 0.833429	valid_1's binary_logloss: 0.137514
    [31]	valid_0's auc: 0.888552	valid_0's binary_logloss: 0.118399	valid_1's auc: 0.834036	valid_1's binary_logloss: 0.137459
    [32]	valid_0's auc: 0.889448	valid_0's binary_logloss: 0.118019	valid_1's auc: 0.833754	valid_1's binary_logloss: 0.137498
    [33]	valid_0's auc: 0.890203	valid_0's binary_logloss: 0.117711	valid_1's auc: 0.833853	valid_1's binary_logloss: 0.137491
    [34]	valid_0's auc: 0.890868	valid_0's binary_logloss: 0.117395	valid_1's auc: 0.833648	valid_1's binary_logloss: 0.137532
    [35]	valid_0's auc: 0.891378	valid_0's binary_logloss: 0.117106	valid_1's auc: 0.833479	valid_1's binary_logloss: 0.137577
    [36]	valid_0's auc: 0.892061	valid_0's binary_logloss: 0.116741	valid_1's auc: 0.832608	valid_1's binary_logloss: 0.137799
    [37]	valid_0's auc: 0.892715	valid_0's binary_logloss: 0.116403	valid_1's auc: 0.83208	valid_1's binary_logloss: 0.137938
    [38]	valid_0's auc: 0.893146	valid_0's binary_logloss: 0.116126	valid_1's auc: 0.83196	valid_1's binary_logloss: 0.137994
    [39]	valid_0's auc: 0.893963	valid_0's binary_logloss: 0.115783	valid_1's auc: 0.831584	valid_1's binary_logloss: 0.138076
    [40]	valid_0's auc: 0.894274	valid_0's binary_logloss: 0.115535	valid_1's auc: 0.831346	valid_1's binary_logloss: 0.13818
    [41]	valid_0's auc: 0.894809	valid_0's binary_logloss: 0.115247	valid_1's auc: 0.831197	valid_1's binary_logloss: 0.138289
    [42]	valid_0's auc: 0.895332	valid_0's binary_logloss: 0.114916	valid_1's auc: 0.830893	valid_1's binary_logloss: 0.138343
    [43]	valid_0's auc: 0.895864	valid_0's binary_logloss: 0.114701	valid_1's auc: 0.83129	valid_1's binary_logloss: 0.138307
    [44]	valid_0's auc: 0.89623	valid_0's binary_logloss: 0.114411	valid_1's auc: 0.831004	valid_1's binary_logloss: 0.138408
    [45]	valid_0's auc: 0.896616	valid_0's binary_logloss: 0.11413	valid_1's auc: 0.830369	valid_1's binary_logloss: 0.138547
    [46]	valid_0's auc: 0.897076	valid_0's binary_logloss: 0.113919	valid_1's auc: 0.830246	valid_1's binary_logloss: 0.138644
    [47]	valid_0's auc: 0.897393	valid_0's binary_logloss: 0.113691	valid_1's auc: 0.829999	valid_1's binary_logloss: 0.138769
    [48]	valid_0's auc: 0.897745	valid_0's binary_logloss: 0.113465	valid_1's auc: 0.829593	valid_1's binary_logloss: 0.138908
    [49]	valid_0's auc: 0.898203	valid_0's binary_logloss: 0.113214	valid_1's auc: 0.829209	valid_1's binary_logloss: 0.139045
    [50]	valid_0's auc: 0.898546	valid_0's binary_logloss: 0.112978	valid_1's auc: 0.829064	valid_1's binary_logloss: 0.139115
    [51]	valid_0's auc: 0.898958	valid_0's binary_logloss: 0.112723	valid_1's auc: 0.82913	valid_1's binary_logloss: 0.139163
    [52]	valid_0's auc: 0.899296	valid_0's binary_logloss: 0.112541	valid_1's auc: 0.828532	valid_1's binary_logloss: 0.139358
    [53]	valid_0's auc: 0.899966	valid_0's binary_logloss: 0.112208	valid_1's auc: 0.828277	valid_1's binary_logloss: 0.139431
    [54]	valid_0's auc: 0.900225	valid_0's binary_logloss: 0.112001	valid_1's auc: 0.827866	valid_1's binary_logloss: 0.139573
    [55]	valid_0's auc: 0.90036	valid_0's binary_logloss: 0.111812	valid_1's auc: 0.827887	valid_1's binary_logloss: 0.139655
    [56]	valid_0's auc: 0.900528	valid_0's binary_logloss: 0.111638	valid_1's auc: 0.827809	valid_1's binary_logloss: 0.139702
    [57]	valid_0's auc: 0.900692	valid_0's binary_logloss: 0.111452	valid_1's auc: 0.827613	valid_1's binary_logloss: 0.139758
    [58]	valid_0's auc: 0.901275	valid_0's binary_logloss: 0.111262	valid_1's auc: 0.827848	valid_1's binary_logloss: 0.139758
    [59]	valid_0's auc: 0.90149	valid_0's binary_logloss: 0.111086	valid_1's auc: 0.827643	valid_1's binary_logloss: 0.139835
    [60]	valid_0's auc: 0.901669	valid_0's binary_logloss: 0.110873	valid_1's auc: 0.827187	valid_1's binary_logloss: 0.13998
    [61]	valid_0's auc: 0.901715	valid_0's binary_logloss: 0.110743	valid_1's auc: 0.826643	valid_1's binary_logloss: 0.140136
    Early stopping, best iteration is:
    [31]	valid_0's auc: 0.888552	valid_0's binary_logloss: 0.118399	valid_1's auc: 0.834036	valid_1's binary_logloss: 0.137459
    [1]	valid_0's auc: 0.828293	valid_0's binary_logloss: 0.156591	valid_1's auc: 0.803729	valid_1's binary_logloss: 0.159809
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.836848	valid_0's binary_logloss: 0.150955	valid_1's auc: 0.813468	valid_1's binary_logloss: 0.155406
    [3]	valid_0's auc: 0.842786	valid_0's binary_logloss: 0.146796	valid_1's auc: 0.817407	valid_1's binary_logloss: 0.151952
    [4]	valid_0's auc: 0.851348	valid_0's binary_logloss: 0.143395	valid_1's auc: 0.823949	valid_1's binary_logloss: 0.14934
    [5]	valid_0's auc: 0.855149	valid_0's binary_logloss: 0.140626	valid_1's auc: 0.827122	valid_1's binary_logloss: 0.147185
    [6]	valid_0's auc: 0.857679	valid_0's binary_logloss: 0.13837	valid_1's auc: 0.826758	valid_1's binary_logloss: 0.145708
    [7]	valid_0's auc: 0.86155	valid_0's binary_logloss: 0.136445	valid_1's auc: 0.828472	valid_1's binary_logloss: 0.144294
    [8]	valid_0's auc: 0.862588	valid_0's binary_logloss: 0.134808	valid_1's auc: 0.828623	valid_1's binary_logloss: 0.143251
    [9]	valid_0's auc: 0.865115	valid_0's binary_logloss: 0.133359	valid_1's auc: 0.831471	valid_1's binary_logloss: 0.142276
    [10]	valid_0's auc: 0.866894	valid_0's binary_logloss: 0.132073	valid_1's auc: 0.831985	valid_1's binary_logloss: 0.141408
    [11]	valid_0's auc: 0.869042	valid_0's binary_logloss: 0.130873	valid_1's auc: 0.834107	valid_1's binary_logloss: 0.140661
    [12]	valid_0's auc: 0.870884	valid_0's binary_logloss: 0.129808	valid_1's auc: 0.833841	valid_1's binary_logloss: 0.140137
    [13]	valid_0's auc: 0.87235	valid_0's binary_logloss: 0.128859	valid_1's auc: 0.834392	valid_1's binary_logloss: 0.139657
    [14]	valid_0's auc: 0.873638	valid_0's binary_logloss: 0.128012	valid_1's auc: 0.834569	valid_1's binary_logloss: 0.13928
    [15]	valid_0's auc: 0.874792	valid_0's binary_logloss: 0.127177	valid_1's auc: 0.83417	valid_1's binary_logloss: 0.138989
    [16]	valid_0's auc: 0.876361	valid_0's binary_logloss: 0.126304	valid_1's auc: 0.835986	valid_1's binary_logloss: 0.138606
    [17]	valid_0's auc: 0.878328	valid_0's binary_logloss: 0.125589	valid_1's auc: 0.836721	valid_1's binary_logloss: 0.138362
    [18]	valid_0's auc: 0.879835	valid_0's binary_logloss: 0.124908	valid_1's auc: 0.836975	valid_1's binary_logloss: 0.138199
    [19]	valid_0's auc: 0.881056	valid_0's binary_logloss: 0.124303	valid_1's auc: 0.837163	valid_1's binary_logloss: 0.137993
    [20]	valid_0's auc: 0.881684	valid_0's binary_logloss: 0.123706	valid_1's auc: 0.836767	valid_1's binary_logloss: 0.137922
    [21]	valid_0's auc: 0.882852	valid_0's binary_logloss: 0.123138	valid_1's auc: 0.835726	valid_1's binary_logloss: 0.137937
    [22]	valid_0's auc: 0.883782	valid_0's binary_logloss: 0.122621	valid_1's auc: 0.836033	valid_1's binary_logloss: 0.137832
    [23]	valid_0's auc: 0.884894	valid_0's binary_logloss: 0.122101	valid_1's auc: 0.835965	valid_1's binary_logloss: 0.137761
    [24]	valid_0's auc: 0.886129	valid_0's binary_logloss: 0.121539	valid_1's auc: 0.836276	valid_1's binary_logloss: 0.137662
    [25]	valid_0's auc: 0.887299	valid_0's binary_logloss: 0.121003	valid_1's auc: 0.836494	valid_1's binary_logloss: 0.137562
    [26]	valid_0's auc: 0.888167	valid_0's binary_logloss: 0.120533	valid_1's auc: 0.836735	valid_1's binary_logloss: 0.137471
    [27]	valid_0's auc: 0.889209	valid_0's binary_logloss: 0.120033	valid_1's auc: 0.837077	valid_1's binary_logloss: 0.137335
    [28]	valid_0's auc: 0.890083	valid_0's binary_logloss: 0.119628	valid_1's auc: 0.836646	valid_1's binary_logloss: 0.137396
    [29]	valid_0's auc: 0.891268	valid_0's binary_logloss: 0.1192	valid_1's auc: 0.836893	valid_1's binary_logloss: 0.137334
    [30]	valid_0's auc: 0.892026	valid_0's binary_logloss: 0.118779	valid_1's auc: 0.836987	valid_1's binary_logloss: 0.13734
    [31]	valid_0's auc: 0.892612	valid_0's binary_logloss: 0.1184	valid_1's auc: 0.835995	valid_1's binary_logloss: 0.137457
    [32]	valid_0's auc: 0.893409	valid_0's binary_logloss: 0.118028	valid_1's auc: 0.836329	valid_1's binary_logloss: 0.137426
    [33]	valid_0's auc: 0.894715	valid_0's binary_logloss: 0.117634	valid_1's auc: 0.836237	valid_1's binary_logloss: 0.137428
    [34]	valid_0's auc: 0.896129	valid_0's binary_logloss: 0.117239	valid_1's auc: 0.837278	valid_1's binary_logloss: 0.137234
    [35]	valid_0's auc: 0.897111	valid_0's binary_logloss: 0.116881	valid_1's auc: 0.837313	valid_1's binary_logloss: 0.137208
    [36]	valid_0's auc: 0.89791	valid_0's binary_logloss: 0.116525	valid_1's auc: 0.837094	valid_1's binary_logloss: 0.137227
    [37]	valid_0's auc: 0.898327	valid_0's binary_logloss: 0.116184	valid_1's auc: 0.836808	valid_1's binary_logloss: 0.137267
    [38]	valid_0's auc: 0.899162	valid_0's binary_logloss: 0.115859	valid_1's auc: 0.836473	valid_1's binary_logloss: 0.137323
    [39]	valid_0's auc: 0.899593	valid_0's binary_logloss: 0.115562	valid_1's auc: 0.836433	valid_1's binary_logloss: 0.137356
    [40]	valid_0's auc: 0.899942	valid_0's binary_logloss: 0.115297	valid_1's auc: 0.836299	valid_1's binary_logloss: 0.137382
    [41]	valid_0's auc: 0.900439	valid_0's binary_logloss: 0.114998	valid_1's auc: 0.835877	valid_1's binary_logloss: 0.137474
    [42]	valid_0's auc: 0.90094	valid_0's binary_logloss: 0.1147	valid_1's auc: 0.835016	valid_1's binary_logloss: 0.137628
    [43]	valid_0's auc: 0.901223	valid_0's binary_logloss: 0.11447	valid_1's auc: 0.834849	valid_1's binary_logloss: 0.137661
    [44]	valid_0's auc: 0.901438	valid_0's binary_logloss: 0.11423	valid_1's auc: 0.834743	valid_1's binary_logloss: 0.137699
    [45]	valid_0's auc: 0.902058	valid_0's binary_logloss: 0.113982	valid_1's auc: 0.834615	valid_1's binary_logloss: 0.13774
    [46]	valid_0's auc: 0.902431	valid_0's binary_logloss: 0.113713	valid_1's auc: 0.834207	valid_1's binary_logloss: 0.13784
    [47]	valid_0's auc: 0.902812	valid_0's binary_logloss: 0.113487	valid_1's auc: 0.833758	valid_1's binary_logloss: 0.137985
    [48]	valid_0's auc: 0.90306	valid_0's binary_logloss: 0.113301	valid_1's auc: 0.833219	valid_1's binary_logloss: 0.138122
    [49]	valid_0's auc: 0.903267	valid_0's binary_logloss: 0.113063	valid_1's auc: 0.832937	valid_1's binary_logloss: 0.138197
    [50]	valid_0's auc: 0.903705	valid_0's binary_logloss: 0.112863	valid_1's auc: 0.832815	valid_1's binary_logloss: 0.13824
    [51]	valid_0's auc: 0.904223	valid_0's binary_logloss: 0.112624	valid_1's auc: 0.832789	valid_1's binary_logloss: 0.138311
    [52]	valid_0's auc: 0.904867	valid_0's binary_logloss: 0.112372	valid_1's auc: 0.832399	valid_1's binary_logloss: 0.138416
    [53]	valid_0's auc: 0.90521	valid_0's binary_logloss: 0.112133	valid_1's auc: 0.832296	valid_1's binary_logloss: 0.1385
    [54]	valid_0's auc: 0.905689	valid_0's binary_logloss: 0.111851	valid_1's auc: 0.831911	valid_1's binary_logloss: 0.138563
    [55]	valid_0's auc: 0.905941	valid_0's binary_logloss: 0.111672	valid_1's auc: 0.831691	valid_1's binary_logloss: 0.138659
    [56]	valid_0's auc: 0.906184	valid_0's binary_logloss: 0.111482	valid_1's auc: 0.83155	valid_1's binary_logloss: 0.138723
    [57]	valid_0's auc: 0.906352	valid_0's binary_logloss: 0.111271	valid_1's auc: 0.830993	valid_1's binary_logloss: 0.138832
    [58]	valid_0's auc: 0.906874	valid_0's binary_logloss: 0.110997	valid_1's auc: 0.831151	valid_1's binary_logloss: 0.138806
    [59]	valid_0's auc: 0.906995	valid_0's binary_logloss: 0.110797	valid_1's auc: 0.830642	valid_1's binary_logloss: 0.138926
    [60]	valid_0's auc: 0.907239	valid_0's binary_logloss: 0.110639	valid_1's auc: 0.830434	valid_1's binary_logloss: 0.139017
    [61]	valid_0's auc: 0.907442	valid_0's binary_logloss: 0.110465	valid_1's auc: 0.829813	valid_1's binary_logloss: 0.13916
    [62]	valid_0's auc: 0.908053	valid_0's binary_logloss: 0.110189	valid_1's auc: 0.829969	valid_1's binary_logloss: 0.139161
    [63]	valid_0's auc: 0.908139	valid_0's binary_logloss: 0.110026	valid_1's auc: 0.829479	valid_1's binary_logloss: 0.139308
    [64]	valid_0's auc: 0.908183	valid_0's binary_logloss: 0.109892	valid_1's auc: 0.828768	valid_1's binary_logloss: 0.139496
    [65]	valid_0's auc: 0.908915	valid_0's binary_logloss: 0.109601	valid_1's auc: 0.828359	valid_1's binary_logloss: 0.139582
    Early stopping, best iteration is:
    [35]	valid_0's auc: 0.897111	valid_0's binary_logloss: 0.116881	valid_1's auc: 0.837313	valid_1's binary_logloss: 0.137208
    [1]	valid_0's auc: 0.833154	valid_0's binary_logloss: 0.156832	valid_1's auc: 0.809041	valid_1's binary_logloss: 0.159834
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.842696	valid_0's binary_logloss: 0.15097	valid_1's auc: 0.816072	valid_1's binary_logloss: 0.155054
    [3]	valid_0's auc: 0.848628	valid_0's binary_logloss: 0.146679	valid_1's auc: 0.821987	valid_1's binary_logloss: 0.151615
    [4]	valid_0's auc: 0.852189	valid_0's binary_logloss: 0.143409	valid_1's auc: 0.823257	valid_1's binary_logloss: 0.148971
    [5]	valid_0's auc: 0.85445	valid_0's binary_logloss: 0.140691	valid_1's auc: 0.826693	valid_1's binary_logloss: 0.146922
    [6]	valid_0's auc: 0.856663	valid_0's binary_logloss: 0.138477	valid_1's auc: 0.827879	valid_1's binary_logloss: 0.145076
    [7]	valid_0's auc: 0.858141	valid_0's binary_logloss: 0.136527	valid_1's auc: 0.828071	valid_1's binary_logloss: 0.143771
    [8]	valid_0's auc: 0.860436	valid_0's binary_logloss: 0.134845	valid_1's auc: 0.830521	valid_1's binary_logloss: 0.142677
    [9]	valid_0's auc: 0.861929	valid_0's binary_logloss: 0.133357	valid_1's auc: 0.831252	valid_1's binary_logloss: 0.14175
    [10]	valid_0's auc: 0.865268	valid_0's binary_logloss: 0.132002	valid_1's auc: 0.831065	valid_1's binary_logloss: 0.141032
    [11]	valid_0's auc: 0.86773	valid_0's binary_logloss: 0.130767	valid_1's auc: 0.831969	valid_1's binary_logloss: 0.140477
    [12]	valid_0's auc: 0.869797	valid_0's binary_logloss: 0.129639	valid_1's auc: 0.831747	valid_1's binary_logloss: 0.139964
    [13]	valid_0's auc: 0.871962	valid_0's binary_logloss: 0.128704	valid_1's auc: 0.832299	valid_1's binary_logloss: 0.13953
    [14]	valid_0's auc: 0.873656	valid_0's binary_logloss: 0.12785	valid_1's auc: 0.831874	valid_1's binary_logloss: 0.139277
    [15]	valid_0's auc: 0.875313	valid_0's binary_logloss: 0.127041	valid_1's auc: 0.831922	valid_1's binary_logloss: 0.139006
    [16]	valid_0's auc: 0.875813	valid_0's binary_logloss: 0.126291	valid_1's auc: 0.832339	valid_1's binary_logloss: 0.138679
    [17]	valid_0's auc: 0.877574	valid_0's binary_logloss: 0.125606	valid_1's auc: 0.832459	valid_1's binary_logloss: 0.138444
    [18]	valid_0's auc: 0.879134	valid_0's binary_logloss: 0.12489	valid_1's auc: 0.832409	valid_1's binary_logloss: 0.138268
    [19]	valid_0's auc: 0.880176	valid_0's binary_logloss: 0.124247	valid_1's auc: 0.832557	valid_1's binary_logloss: 0.138164
    [20]	valid_0's auc: 0.881608	valid_0's binary_logloss: 0.123609	valid_1's auc: 0.832053	valid_1's binary_logloss: 0.138136
    [21]	valid_0's auc: 0.882632	valid_0's binary_logloss: 0.123013	valid_1's auc: 0.831852	valid_1's binary_logloss: 0.138081
    [22]	valid_0's auc: 0.883684	valid_0's binary_logloss: 0.12245	valid_1's auc: 0.832034	valid_1's binary_logloss: 0.137978
    [23]	valid_0's auc: 0.885039	valid_0's binary_logloss: 0.121827	valid_1's auc: 0.831422	valid_1's binary_logloss: 0.138011
    [24]	valid_0's auc: 0.886247	valid_0's binary_logloss: 0.121291	valid_1's auc: 0.831253	valid_1's binary_logloss: 0.137942
    [25]	valid_0's auc: 0.887301	valid_0's binary_logloss: 0.120809	valid_1's auc: 0.831276	valid_1's binary_logloss: 0.137924
    [26]	valid_0's auc: 0.888621	valid_0's binary_logloss: 0.120287	valid_1's auc: 0.830735	valid_1's binary_logloss: 0.137975
    [27]	valid_0's auc: 0.889227	valid_0's binary_logloss: 0.119878	valid_1's auc: 0.830569	valid_1's binary_logloss: 0.137976
    [28]	valid_0's auc: 0.890381	valid_0's binary_logloss: 0.119445	valid_1's auc: 0.830281	valid_1's binary_logloss: 0.137968
    [29]	valid_0's auc: 0.891139	valid_0's binary_logloss: 0.119016	valid_1's auc: 0.830916	valid_1's binary_logloss: 0.137839
    [30]	valid_0's auc: 0.892417	valid_0's binary_logloss: 0.118535	valid_1's auc: 0.830189	valid_1's binary_logloss: 0.137909
    [31]	valid_0's auc: 0.893102	valid_0's binary_logloss: 0.118157	valid_1's auc: 0.830117	valid_1's binary_logloss: 0.137933
    [32]	valid_0's auc: 0.893844	valid_0's binary_logloss: 0.117764	valid_1's auc: 0.829877	valid_1's binary_logloss: 0.137962
    [33]	valid_0's auc: 0.894508	valid_0's binary_logloss: 0.117419	valid_1's auc: 0.829552	valid_1's binary_logloss: 0.138044
    [34]	valid_0's auc: 0.895157	valid_0's binary_logloss: 0.117051	valid_1's auc: 0.830486	valid_1's binary_logloss: 0.137893
    [35]	valid_0's auc: 0.89588	valid_0's binary_logloss: 0.116679	valid_1's auc: 0.83041	valid_1's binary_logloss: 0.137896
    [36]	valid_0's auc: 0.896611	valid_0's binary_logloss: 0.116295	valid_1's auc: 0.829969	valid_1's binary_logloss: 0.138026
    [37]	valid_0's auc: 0.89762	valid_0's binary_logloss: 0.115979	valid_1's auc: 0.829802	valid_1's binary_logloss: 0.138089
    [38]	valid_0's auc: 0.898028	valid_0's binary_logloss: 0.115667	valid_1's auc: 0.829346	valid_1's binary_logloss: 0.138207
    [39]	valid_0's auc: 0.89839	valid_0's binary_logloss: 0.115401	valid_1's auc: 0.829208	valid_1's binary_logloss: 0.138191
    [40]	valid_0's auc: 0.898968	valid_0's binary_logloss: 0.115042	valid_1's auc: 0.828576	valid_1's binary_logloss: 0.138317
    [41]	valid_0's auc: 0.899373	valid_0's binary_logloss: 0.114766	valid_1's auc: 0.828382	valid_1's binary_logloss: 0.138408
    [42]	valid_0's auc: 0.900042	valid_0's binary_logloss: 0.114467	valid_1's auc: 0.828023	valid_1's binary_logloss: 0.138489
    [43]	valid_0's auc: 0.900488	valid_0's binary_logloss: 0.114194	valid_1's auc: 0.827815	valid_1's binary_logloss: 0.138591
    [44]	valid_0's auc: 0.901004	valid_0's binary_logloss: 0.113865	valid_1's auc: 0.827321	valid_1's binary_logloss: 0.138736
    [45]	valid_0's auc: 0.901401	valid_0's binary_logloss: 0.113616	valid_1's auc: 0.826875	valid_1's binary_logloss: 0.138874
    [46]	valid_0's auc: 0.901795	valid_0's binary_logloss: 0.113348	valid_1's auc: 0.82643	valid_1's binary_logloss: 0.138955
    [47]	valid_0's auc: 0.902244	valid_0's binary_logloss: 0.11309	valid_1's auc: 0.825825	valid_1's binary_logloss: 0.13908
    [48]	valid_0's auc: 0.902756	valid_0's binary_logloss: 0.112891	valid_1's auc: 0.825639	valid_1's binary_logloss: 0.139117
    [49]	valid_0's auc: 0.903206	valid_0's binary_logloss: 0.112607	valid_1's auc: 0.825607	valid_1's binary_logloss: 0.139184
    Early stopping, best iteration is:
    [19]	valid_0's auc: 0.880176	valid_0's binary_logloss: 0.124247	valid_1's auc: 0.832557	valid_1's binary_logloss: 0.138164
    [1]	valid_0's auc: 0.822874	valid_0's binary_logloss: 0.157449	valid_1's auc: 0.804936	valid_1's binary_logloss: 0.160126
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.827064	valid_0's binary_logloss: 0.152115	valid_1's auc: 0.810262	valid_1's binary_logloss: 0.155403
    [3]	valid_0's auc: 0.830773	valid_0's binary_logloss: 0.148229	valid_1's auc: 0.814099	valid_1's binary_logloss: 0.151956
    [4]	valid_0's auc: 0.834585	valid_0's binary_logloss: 0.145242	valid_1's auc: 0.815291	valid_1's binary_logloss: 0.149379
    [5]	valid_0's auc: 0.839767	valid_0's binary_logloss: 0.142798	valid_1's auc: 0.819992	valid_1's binary_logloss: 0.147278
    [6]	valid_0's auc: 0.845277	valid_0's binary_logloss: 0.140696	valid_1's auc: 0.824997	valid_1's binary_logloss: 0.145614
    [7]	valid_0's auc: 0.84706	valid_0's binary_logloss: 0.139021	valid_1's auc: 0.827429	valid_1's binary_logloss: 0.144095
    [8]	valid_0's auc: 0.847609	valid_0's binary_logloss: 0.137638	valid_1's auc: 0.827987	valid_1's binary_logloss: 0.143002
    [9]	valid_0's auc: 0.848913	valid_0's binary_logloss: 0.136364	valid_1's auc: 0.82986	valid_1's binary_logloss: 0.141984
    [10]	valid_0's auc: 0.850178	valid_0's binary_logloss: 0.135236	valid_1's auc: 0.829651	valid_1's binary_logloss: 0.141228
    [11]	valid_0's auc: 0.85127	valid_0's binary_logloss: 0.134273	valid_1's auc: 0.830496	valid_1's binary_logloss: 0.140509
    [12]	valid_0's auc: 0.852586	valid_0's binary_logloss: 0.133411	valid_1's auc: 0.830525	valid_1's binary_logloss: 0.13997
    [13]	valid_0's auc: 0.853732	valid_0's binary_logloss: 0.132666	valid_1's auc: 0.829319	valid_1's binary_logloss: 0.13957
    [14]	valid_0's auc: 0.854775	valid_0's binary_logloss: 0.13201	valid_1's auc: 0.832116	valid_1's binary_logloss: 0.139106
    [15]	valid_0's auc: 0.855714	valid_0's binary_logloss: 0.131409	valid_1's auc: 0.833454	valid_1's binary_logloss: 0.138684
    [16]	valid_0's auc: 0.856099	valid_0's binary_logloss: 0.130853	valid_1's auc: 0.832243	valid_1's binary_logloss: 0.138485
    [17]	valid_0's auc: 0.857475	valid_0's binary_logloss: 0.130295	valid_1's auc: 0.832577	valid_1's binary_logloss: 0.138179
    [18]	valid_0's auc: 0.857908	valid_0's binary_logloss: 0.129866	valid_1's auc: 0.832652	valid_1's binary_logloss: 0.138084
    [19]	valid_0's auc: 0.859226	valid_0's binary_logloss: 0.129384	valid_1's auc: 0.832887	valid_1's binary_logloss: 0.137828
    [20]	valid_0's auc: 0.860296	valid_0's binary_logloss: 0.129003	valid_1's auc: 0.834111	valid_1's binary_logloss: 0.137643
    [21]	valid_0's auc: 0.860934	valid_0's binary_logloss: 0.128619	valid_1's auc: 0.834602	valid_1's binary_logloss: 0.137471
    [22]	valid_0's auc: 0.861712	valid_0's binary_logloss: 0.12825	valid_1's auc: 0.834828	valid_1's binary_logloss: 0.137291
    [23]	valid_0's auc: 0.862939	valid_0's binary_logloss: 0.127869	valid_1's auc: 0.835767	valid_1's binary_logloss: 0.137067
    [24]	valid_0's auc: 0.863968	valid_0's binary_logloss: 0.127525	valid_1's auc: 0.835485	valid_1's binary_logloss: 0.13701
    [25]	valid_0's auc: 0.865029	valid_0's binary_logloss: 0.127234	valid_1's auc: 0.835507	valid_1's binary_logloss: 0.136911
    [26]	valid_0's auc: 0.865758	valid_0's binary_logloss: 0.126952	valid_1's auc: 0.835743	valid_1's binary_logloss: 0.136823
    [27]	valid_0's auc: 0.866244	valid_0's binary_logloss: 0.126688	valid_1's auc: 0.836049	valid_1's binary_logloss: 0.136736
    [28]	valid_0's auc: 0.866607	valid_0's binary_logloss: 0.126455	valid_1's auc: 0.836043	valid_1's binary_logloss: 0.136687
    [29]	valid_0's auc: 0.867322	valid_0's binary_logloss: 0.126255	valid_1's auc: 0.836375	valid_1's binary_logloss: 0.136596
    [30]	valid_0's auc: 0.867997	valid_0's binary_logloss: 0.126022	valid_1's auc: 0.836518	valid_1's binary_logloss: 0.136563
    [31]	valid_0's auc: 0.868888	valid_0's binary_logloss: 0.125781	valid_1's auc: 0.836949	valid_1's binary_logloss: 0.13648
    [32]	valid_0's auc: 0.869544	valid_0's binary_logloss: 0.125524	valid_1's auc: 0.836774	valid_1's binary_logloss: 0.136506
    [33]	valid_0's auc: 0.870314	valid_0's binary_logloss: 0.125324	valid_1's auc: 0.837039	valid_1's binary_logloss: 0.13642
    [34]	valid_0's auc: 0.871207	valid_0's binary_logloss: 0.125096	valid_1's auc: 0.836859	valid_1's binary_logloss: 0.136441
    [35]	valid_0's auc: 0.871788	valid_0's binary_logloss: 0.124883	valid_1's auc: 0.836641	valid_1's binary_logloss: 0.136467
    [36]	valid_0's auc: 0.872403	valid_0's binary_logloss: 0.124635	valid_1's auc: 0.836917	valid_1's binary_logloss: 0.13642
    [37]	valid_0's auc: 0.872914	valid_0's binary_logloss: 0.124439	valid_1's auc: 0.836892	valid_1's binary_logloss: 0.136489
    [38]	valid_0's auc: 0.873469	valid_0's binary_logloss: 0.124233	valid_1's auc: 0.836816	valid_1's binary_logloss: 0.136502
    [39]	valid_0's auc: 0.873971	valid_0's binary_logloss: 0.124017	valid_1's auc: 0.83659	valid_1's binary_logloss: 0.136543
    [40]	valid_0's auc: 0.875013	valid_0's binary_logloss: 0.123715	valid_1's auc: 0.836381	valid_1's binary_logloss: 0.136593
    [41]	valid_0's auc: 0.875341	valid_0's binary_logloss: 0.123538	valid_1's auc: 0.8363	valid_1's binary_logloss: 0.136624
    [42]	valid_0's auc: 0.875886	valid_0's binary_logloss: 0.123285	valid_1's auc: 0.836043	valid_1's binary_logloss: 0.136665
    [43]	valid_0's auc: 0.87633	valid_0's binary_logloss: 0.12311	valid_1's auc: 0.836018	valid_1's binary_logloss: 0.136647
    [44]	valid_0's auc: 0.876816	valid_0's binary_logloss: 0.122878	valid_1's auc: 0.836061	valid_1's binary_logloss: 0.136664
    [45]	valid_0's auc: 0.877116	valid_0's binary_logloss: 0.12271	valid_1's auc: 0.836517	valid_1's binary_logloss: 0.136608
    [46]	valid_0's auc: 0.877567	valid_0's binary_logloss: 0.122536	valid_1's auc: 0.836051	valid_1's binary_logloss: 0.13667
    [47]	valid_0's auc: 0.878192	valid_0's binary_logloss: 0.122332	valid_1's auc: 0.835827	valid_1's binary_logloss: 0.136694
    [48]	valid_0's auc: 0.878371	valid_0's binary_logloss: 0.122208	valid_1's auc: 0.835831	valid_1's binary_logloss: 0.136717
    [49]	valid_0's auc: 0.878669	valid_0's binary_logloss: 0.12204	valid_1's auc: 0.835913	valid_1's binary_logloss: 0.136679
    [50]	valid_0's auc: 0.879288	valid_0's binary_logloss: 0.121825	valid_1's auc: 0.835468	valid_1's binary_logloss: 0.136734
    [51]	valid_0's auc: 0.879552	valid_0's binary_logloss: 0.121647	valid_1's auc: 0.835095	valid_1's binary_logloss: 0.13684
    [52]	valid_0's auc: 0.880033	valid_0's binary_logloss: 0.121459	valid_1's auc: 0.83496	valid_1's binary_logloss: 0.136927
    [53]	valid_0's auc: 0.880447	valid_0's binary_logloss: 0.121297	valid_1's auc: 0.834955	valid_1's binary_logloss: 0.136953
    [54]	valid_0's auc: 0.88087	valid_0's binary_logloss: 0.121116	valid_1's auc: 0.834926	valid_1's binary_logloss: 0.136994
    [55]	valid_0's auc: 0.881105	valid_0's binary_logloss: 0.12097	valid_1's auc: 0.834861	valid_1's binary_logloss: 0.137043
    [56]	valid_0's auc: 0.88161	valid_0's binary_logloss: 0.120772	valid_1's auc: 0.834629	valid_1's binary_logloss: 0.137125
    [57]	valid_0's auc: 0.882134	valid_0's binary_logloss: 0.120595	valid_1's auc: 0.834576	valid_1's binary_logloss: 0.137157
    [58]	valid_0's auc: 0.882429	valid_0's binary_logloss: 0.120441	valid_1's auc: 0.834319	valid_1's binary_logloss: 0.137209
    [59]	valid_0's auc: 0.882799	valid_0's binary_logloss: 0.120268	valid_1's auc: 0.834028	valid_1's binary_logloss: 0.1373
    [60]	valid_0's auc: 0.883147	valid_0's binary_logloss: 0.120094	valid_1's auc: 0.833999	valid_1's binary_logloss: 0.137339
    [61]	valid_0's auc: 0.883628	valid_0's binary_logloss: 0.119914	valid_1's auc: 0.834234	valid_1's binary_logloss: 0.137314
    [62]	valid_0's auc: 0.884034	valid_0's binary_logloss: 0.119753	valid_1's auc: 0.834179	valid_1's binary_logloss: 0.137356
    [63]	valid_0's auc: 0.884169	valid_0's binary_logloss: 0.119623	valid_1's auc: 0.834035	valid_1's binary_logloss: 0.137378
    Early stopping, best iteration is:
    [33]	valid_0's auc: 0.870314	valid_0's binary_logloss: 0.125324	valid_1's auc: 0.837039	valid_1's binary_logloss: 0.13642
    [1]	valid_0's auc: 0.826066	valid_0's binary_logloss: 0.157509	valid_1's auc: 0.810763	valid_1's binary_logloss: 0.160177
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.830558	valid_0's binary_logloss: 0.152317	valid_1's auc: 0.814406	valid_1's binary_logloss: 0.15553
    [3]	valid_0's auc: 0.833526	valid_0's binary_logloss: 0.148529	valid_1's auc: 0.815841	valid_1's binary_logloss: 0.152134
    [4]	valid_0's auc: 0.839653	valid_0's binary_logloss: 0.145555	valid_1's auc: 0.821767	valid_1's binary_logloss: 0.149519
    [5]	valid_0's auc: 0.842943	valid_0's binary_logloss: 0.143073	valid_1's auc: 0.826334	valid_1's binary_logloss: 0.14736
    [6]	valid_0's auc: 0.845929	valid_0's binary_logloss: 0.141096	valid_1's auc: 0.828882	valid_1's binary_logloss: 0.145649
    [7]	valid_0's auc: 0.847231	valid_0's binary_logloss: 0.139394	valid_1's auc: 0.830621	valid_1's binary_logloss: 0.144201
    [8]	valid_0's auc: 0.849609	valid_0's binary_logloss: 0.137956	valid_1's auc: 0.831324	valid_1's binary_logloss: 0.143105
    [9]	valid_0's auc: 0.851333	valid_0's binary_logloss: 0.136733	valid_1's auc: 0.83377	valid_1's binary_logloss: 0.142018
    [10]	valid_0's auc: 0.852692	valid_0's binary_logloss: 0.135654	valid_1's auc: 0.834493	valid_1's binary_logloss: 0.141149
    [11]	valid_0's auc: 0.854022	valid_0's binary_logloss: 0.134688	valid_1's auc: 0.834616	valid_1's binary_logloss: 0.140437
    [12]	valid_0's auc: 0.854419	valid_0's binary_logloss: 0.13393	valid_1's auc: 0.83589	valid_1's binary_logloss: 0.139765
    [13]	valid_0's auc: 0.854853	valid_0's binary_logloss: 0.133229	valid_1's auc: 0.835999	valid_1's binary_logloss: 0.139284
    [14]	valid_0's auc: 0.856241	valid_0's binary_logloss: 0.132532	valid_1's auc: 0.837392	valid_1's binary_logloss: 0.138795
    [15]	valid_0's auc: 0.85803	valid_0's binary_logloss: 0.131862	valid_1's auc: 0.83818	valid_1's binary_logloss: 0.138366
    [16]	valid_0's auc: 0.859289	valid_0's binary_logloss: 0.131311	valid_1's auc: 0.838523	valid_1's binary_logloss: 0.138137
    [17]	valid_0's auc: 0.860115	valid_0's binary_logloss: 0.130812	valid_1's auc: 0.839359	valid_1's binary_logloss: 0.137816
    [18]	valid_0's auc: 0.860807	valid_0's binary_logloss: 0.130408	valid_1's auc: 0.840179	valid_1's binary_logloss: 0.13749
    [19]	valid_0's auc: 0.861568	valid_0's binary_logloss: 0.129954	valid_1's auc: 0.839855	valid_1's binary_logloss: 0.137387
    [20]	valid_0's auc: 0.863194	valid_0's binary_logloss: 0.129482	valid_1's auc: 0.839715	valid_1's binary_logloss: 0.137226
    [21]	valid_0's auc: 0.864273	valid_0's binary_logloss: 0.12907	valid_1's auc: 0.840516	valid_1's binary_logloss: 0.136991
    [22]	valid_0's auc: 0.865334	valid_0's binary_logloss: 0.128648	valid_1's auc: 0.841302	valid_1's binary_logloss: 0.136769
    [23]	valid_0's auc: 0.866353	valid_0's binary_logloss: 0.128302	valid_1's auc: 0.841085	valid_1's binary_logloss: 0.136653
    [24]	valid_0's auc: 0.86699	valid_0's binary_logloss: 0.127957	valid_1's auc: 0.841235	valid_1's binary_logloss: 0.136611
    [25]	valid_0's auc: 0.867587	valid_0's binary_logloss: 0.127634	valid_1's auc: 0.840948	valid_1's binary_logloss: 0.136644
    [26]	valid_0's auc: 0.868507	valid_0's binary_logloss: 0.127298	valid_1's auc: 0.841033	valid_1's binary_logloss: 0.136526
    [27]	valid_0's auc: 0.869362	valid_0's binary_logloss: 0.126976	valid_1's auc: 0.840939	valid_1's binary_logloss: 0.136445
    [28]	valid_0's auc: 0.870126	valid_0's binary_logloss: 0.126688	valid_1's auc: 0.840452	valid_1's binary_logloss: 0.13645
    [29]	valid_0's auc: 0.870568	valid_0's binary_logloss: 0.126442	valid_1's auc: 0.840809	valid_1's binary_logloss: 0.136345
    [30]	valid_0's auc: 0.871323	valid_0's binary_logloss: 0.126166	valid_1's auc: 0.840599	valid_1's binary_logloss: 0.136334
    [31]	valid_0's auc: 0.872181	valid_0's binary_logloss: 0.125929	valid_1's auc: 0.840663	valid_1's binary_logloss: 0.136267
    [32]	valid_0's auc: 0.873089	valid_0's binary_logloss: 0.125641	valid_1's auc: 0.841679	valid_1's binary_logloss: 0.136141
    [33]	valid_0's auc: 0.873443	valid_0's binary_logloss: 0.125415	valid_1's auc: 0.841597	valid_1's binary_logloss: 0.136138
    [34]	valid_0's auc: 0.874111	valid_0's binary_logloss: 0.125163	valid_1's auc: 0.841194	valid_1's binary_logloss: 0.13614
    [35]	valid_0's auc: 0.874804	valid_0's binary_logloss: 0.124891	valid_1's auc: 0.841136	valid_1's binary_logloss: 0.136214
    [36]	valid_0's auc: 0.875123	valid_0's binary_logloss: 0.124721	valid_1's auc: 0.840883	valid_1's binary_logloss: 0.136268
    [37]	valid_0's auc: 0.875964	valid_0's binary_logloss: 0.124479	valid_1's auc: 0.840327	valid_1's binary_logloss: 0.136363
    [38]	valid_0's auc: 0.876484	valid_0's binary_logloss: 0.12422	valid_1's auc: 0.840317	valid_1's binary_logloss: 0.136378
    [39]	valid_0's auc: 0.877529	valid_0's binary_logloss: 0.123975	valid_1's auc: 0.840224	valid_1's binary_logloss: 0.136372
    [40]	valid_0's auc: 0.877916	valid_0's binary_logloss: 0.123803	valid_1's auc: 0.840099	valid_1's binary_logloss: 0.136418
    [41]	valid_0's auc: 0.878404	valid_0's binary_logloss: 0.123606	valid_1's auc: 0.839447	valid_1's binary_logloss: 0.136498
    [42]	valid_0's auc: 0.878787	valid_0's binary_logloss: 0.123458	valid_1's auc: 0.839737	valid_1's binary_logloss: 0.136466
    [43]	valid_0's auc: 0.879296	valid_0's binary_logloss: 0.123263	valid_1's auc: 0.838848	valid_1's binary_logloss: 0.136611
    [44]	valid_0's auc: 0.879561	valid_0's binary_logloss: 0.123111	valid_1's auc: 0.83939	valid_1's binary_logloss: 0.136509
    [45]	valid_0's auc: 0.880167	valid_0's binary_logloss: 0.122912	valid_1's auc: 0.83916	valid_1's binary_logloss: 0.136564
    [46]	valid_0's auc: 0.880652	valid_0's binary_logloss: 0.12273	valid_1's auc: 0.839133	valid_1's binary_logloss: 0.136588
    [47]	valid_0's auc: 0.881351	valid_0's binary_logloss: 0.122478	valid_1's auc: 0.839108	valid_1's binary_logloss: 0.136599
    [48]	valid_0's auc: 0.881786	valid_0's binary_logloss: 0.122308	valid_1's auc: 0.838894	valid_1's binary_logloss: 0.136636
    [49]	valid_0's auc: 0.88242	valid_0's binary_logloss: 0.122054	valid_1's auc: 0.838476	valid_1's binary_logloss: 0.136727
    [50]	valid_0's auc: 0.88279	valid_0's binary_logloss: 0.121862	valid_1's auc: 0.837735	valid_1's binary_logloss: 0.136833
    [51]	valid_0's auc: 0.883422	valid_0's binary_logloss: 0.121646	valid_1's auc: 0.837691	valid_1's binary_logloss: 0.136833
    [52]	valid_0's auc: 0.884035	valid_0's binary_logloss: 0.121461	valid_1's auc: 0.837503	valid_1's binary_logloss: 0.136856
    [53]	valid_0's auc: 0.884636	valid_0's binary_logloss: 0.121237	valid_1's auc: 0.837159	valid_1's binary_logloss: 0.136933
    [54]	valid_0's auc: 0.885019	valid_0's binary_logloss: 0.121086	valid_1's auc: 0.836758	valid_1's binary_logloss: 0.136995
    [55]	valid_0's auc: 0.88546	valid_0's binary_logloss: 0.120908	valid_1's auc: 0.836692	valid_1's binary_logloss: 0.137006
    [56]	valid_0's auc: 0.885711	valid_0's binary_logloss: 0.120745	valid_1's auc: 0.836703	valid_1's binary_logloss: 0.136998
    [57]	valid_0's auc: 0.886081	valid_0's binary_logloss: 0.120544	valid_1's auc: 0.836462	valid_1's binary_logloss: 0.137073
    [58]	valid_0's auc: 0.886576	valid_0's binary_logloss: 0.120352	valid_1's auc: 0.836241	valid_1's binary_logloss: 0.137127
    [59]	valid_0's auc: 0.886878	valid_0's binary_logloss: 0.120191	valid_1's auc: 0.8358	valid_1's binary_logloss: 0.137207
    [60]	valid_0's auc: 0.887158	valid_0's binary_logloss: 0.120005	valid_1's auc: 0.835688	valid_1's binary_logloss: 0.137272
    [61]	valid_0's auc: 0.88755	valid_0's binary_logloss: 0.119855	valid_1's auc: 0.835652	valid_1's binary_logloss: 0.137262
    [62]	valid_0's auc: 0.887817	valid_0's binary_logloss: 0.11971	valid_1's auc: 0.835355	valid_1's binary_logloss: 0.13732
    Early stopping, best iteration is:
    [32]	valid_0's auc: 0.873089	valid_0's binary_logloss: 0.125641	valid_1's auc: 0.841679	valid_1's binary_logloss: 0.136141
    [1]	valid_0's auc: 0.822527	valid_0's binary_logloss: 0.157702	valid_1's auc: 0.811735	valid_1's binary_logloss: 0.160053
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.828897	valid_0's binary_logloss: 0.152381	valid_1's auc: 0.815279	valid_1's binary_logloss: 0.155308
    [3]	valid_0's auc: 0.833978	valid_0's binary_logloss: 0.148529	valid_1's auc: 0.819863	valid_1's binary_logloss: 0.151959
    [4]	valid_0's auc: 0.837641	valid_0's binary_logloss: 0.145558	valid_1's auc: 0.821032	valid_1's binary_logloss: 0.149309
    [5]	valid_0's auc: 0.841306	valid_0's binary_logloss: 0.143184	valid_1's auc: 0.824926	valid_1's binary_logloss: 0.147249
    [6]	valid_0's auc: 0.843853	valid_0's binary_logloss: 0.141094	valid_1's auc: 0.825446	valid_1's binary_logloss: 0.145583
    [7]	valid_0's auc: 0.846589	valid_0's binary_logloss: 0.13939	valid_1's auc: 0.829069	valid_1's binary_logloss: 0.144172
    [8]	valid_0's auc: 0.847805	valid_0's binary_logloss: 0.137988	valid_1's auc: 0.829457	valid_1's binary_logloss: 0.143004
    [9]	valid_0's auc: 0.850007	valid_0's binary_logloss: 0.136683	valid_1's auc: 0.831298	valid_1's binary_logloss: 0.141959
    [10]	valid_0's auc: 0.850514	valid_0's binary_logloss: 0.135597	valid_1's auc: 0.830745	valid_1's binary_logloss: 0.141234
    [11]	valid_0's auc: 0.851694	valid_0's binary_logloss: 0.134644	valid_1's auc: 0.831218	valid_1's binary_logloss: 0.140531
    [12]	valid_0's auc: 0.854647	valid_0's binary_logloss: 0.133779	valid_1's auc: 0.832445	valid_1's binary_logloss: 0.140011
    [13]	valid_0's auc: 0.8552	valid_0's binary_logloss: 0.13303	valid_1's auc: 0.832525	valid_1's binary_logloss: 0.139542
    [14]	valid_0's auc: 0.85611	valid_0's binary_logloss: 0.132355	valid_1's auc: 0.832559	valid_1's binary_logloss: 0.139125
    [15]	valid_0's auc: 0.856886	valid_0's binary_logloss: 0.131779	valid_1's auc: 0.833401	valid_1's binary_logloss: 0.138708
    [16]	valid_0's auc: 0.857397	valid_0's binary_logloss: 0.13126	valid_1's auc: 0.833709	valid_1's binary_logloss: 0.138376
    [17]	valid_0's auc: 0.859038	valid_0's binary_logloss: 0.130686	valid_1's auc: 0.83451	valid_1's binary_logloss: 0.138079
    [18]	valid_0's auc: 0.859754	valid_0's binary_logloss: 0.130225	valid_1's auc: 0.834886	valid_1's binary_logloss: 0.137839
    [19]	valid_0's auc: 0.861354	valid_0's binary_logloss: 0.129689	valid_1's auc: 0.835687	valid_1's binary_logloss: 0.137572
    [20]	valid_0's auc: 0.862114	valid_0's binary_logloss: 0.129263	valid_1's auc: 0.83587	valid_1's binary_logloss: 0.137393
    [21]	valid_0's auc: 0.863196	valid_0's binary_logloss: 0.128834	valid_1's auc: 0.835611	valid_1's binary_logloss: 0.137244
    [22]	valid_0's auc: 0.864467	valid_0's binary_logloss: 0.128421	valid_1's auc: 0.836013	valid_1's binary_logloss: 0.137127
    [23]	valid_0's auc: 0.865716	valid_0's binary_logloss: 0.12802	valid_1's auc: 0.835915	valid_1's binary_logloss: 0.13701
    [24]	valid_0's auc: 0.866358	valid_0's binary_logloss: 0.127649	valid_1's auc: 0.835345	valid_1's binary_logloss: 0.136983
    [25]	valid_0's auc: 0.867017	valid_0's binary_logloss: 0.127326	valid_1's auc: 0.835203	valid_1's binary_logloss: 0.13697
    [26]	valid_0's auc: 0.867793	valid_0's binary_logloss: 0.126998	valid_1's auc: 0.835068	valid_1's binary_logloss: 0.13692
    [27]	valid_0's auc: 0.86892	valid_0's binary_logloss: 0.126634	valid_1's auc: 0.835127	valid_1's binary_logloss: 0.136839
    [28]	valid_0's auc: 0.869611	valid_0's binary_logloss: 0.126332	valid_1's auc: 0.835011	valid_1's binary_logloss: 0.136765
    [29]	valid_0's auc: 0.870502	valid_0's binary_logloss: 0.126032	valid_1's auc: 0.835017	valid_1's binary_logloss: 0.136691
    [30]	valid_0's auc: 0.87109	valid_0's binary_logloss: 0.125752	valid_1's auc: 0.835346	valid_1's binary_logloss: 0.136657
    [31]	valid_0's auc: 0.871834	valid_0's binary_logloss: 0.125492	valid_1's auc: 0.835469	valid_1's binary_logloss: 0.136553
    [32]	valid_0's auc: 0.8727	valid_0's binary_logloss: 0.125224	valid_1's auc: 0.834809	valid_1's binary_logloss: 0.136626
    [33]	valid_0's auc: 0.873327	valid_0's binary_logloss: 0.12497	valid_1's auc: 0.834899	valid_1's binary_logloss: 0.136576
    [34]	valid_0's auc: 0.873759	valid_0's binary_logloss: 0.124747	valid_1's auc: 0.835299	valid_1's binary_logloss: 0.136478
    [35]	valid_0's auc: 0.874436	valid_0's binary_logloss: 0.124498	valid_1's auc: 0.834852	valid_1's binary_logloss: 0.136556
    [36]	valid_0's auc: 0.875262	valid_0's binary_logloss: 0.124237	valid_1's auc: 0.834479	valid_1's binary_logloss: 0.13659
    [37]	valid_0's auc: 0.875938	valid_0's binary_logloss: 0.123998	valid_1's auc: 0.834359	valid_1's binary_logloss: 0.136644
    [38]	valid_0's auc: 0.876343	valid_0's binary_logloss: 0.123795	valid_1's auc: 0.834106	valid_1's binary_logloss: 0.136687
    [39]	valid_0's auc: 0.877139	valid_0's binary_logloss: 0.123529	valid_1's auc: 0.83359	valid_1's binary_logloss: 0.136724
    [40]	valid_0's auc: 0.877911	valid_0's binary_logloss: 0.123326	valid_1's auc: 0.833512	valid_1's binary_logloss: 0.136737
    [41]	valid_0's auc: 0.87875	valid_0's binary_logloss: 0.123086	valid_1's auc: 0.833202	valid_1's binary_logloss: 0.136752
    [42]	valid_0's auc: 0.879342	valid_0's binary_logloss: 0.122885	valid_1's auc: 0.832649	valid_1's binary_logloss: 0.136842
    [43]	valid_0's auc: 0.879629	valid_0's binary_logloss: 0.122717	valid_1's auc: 0.832179	valid_1's binary_logloss: 0.136927
    [44]	valid_0's auc: 0.880215	valid_0's binary_logloss: 0.122487	valid_1's auc: 0.832172	valid_1's binary_logloss: 0.13695
    [45]	valid_0's auc: 0.880851	valid_0's binary_logloss: 0.122245	valid_1's auc: 0.831665	valid_1's binary_logloss: 0.137053
    [46]	valid_0's auc: 0.881492	valid_0's binary_logloss: 0.121994	valid_1's auc: 0.83185	valid_1's binary_logloss: 0.137071
    [47]	valid_0's auc: 0.881818	valid_0's binary_logloss: 0.121836	valid_1's auc: 0.831946	valid_1's binary_logloss: 0.137043
    [48]	valid_0's auc: 0.882136	valid_0's binary_logloss: 0.121683	valid_1's auc: 0.832078	valid_1's binary_logloss: 0.137058
    [49]	valid_0's auc: 0.882509	valid_0's binary_logloss: 0.121512	valid_1's auc: 0.832046	valid_1's binary_logloss: 0.13711
    [50]	valid_0's auc: 0.882872	valid_0's binary_logloss: 0.121296	valid_1's auc: 0.83209	valid_1's binary_logloss: 0.137124
    [51]	valid_0's auc: 0.883328	valid_0's binary_logloss: 0.121099	valid_1's auc: 0.831958	valid_1's binary_logloss: 0.137201
    [52]	valid_0's auc: 0.883722	valid_0's binary_logloss: 0.120897	valid_1's auc: 0.831752	valid_1's binary_logloss: 0.13722
    Early stopping, best iteration is:
    [22]	valid_0's auc: 0.864467	valid_0's binary_logloss: 0.128421	valid_1's auc: 0.836013	valid_1's binary_logloss: 0.137127
    [1]	valid_0's auc: 0.822874	valid_0's binary_logloss: 0.157449	valid_1's auc: 0.804936	valid_1's binary_logloss: 0.160126
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.827064	valid_0's binary_logloss: 0.152115	valid_1's auc: 0.810262	valid_1's binary_logloss: 0.155403
    [3]	valid_0's auc: 0.830773	valid_0's binary_logloss: 0.148229	valid_1's auc: 0.814099	valid_1's binary_logloss: 0.151956
    [4]	valid_0's auc: 0.834585	valid_0's binary_logloss: 0.145242	valid_1's auc: 0.815291	valid_1's binary_logloss: 0.149379
    [5]	valid_0's auc: 0.839767	valid_0's binary_logloss: 0.142798	valid_1's auc: 0.819992	valid_1's binary_logloss: 0.147278
    [6]	valid_0's auc: 0.845277	valid_0's binary_logloss: 0.140696	valid_1's auc: 0.824997	valid_1's binary_logloss: 0.145614
    [7]	valid_0's auc: 0.84706	valid_0's binary_logloss: 0.139021	valid_1's auc: 0.827429	valid_1's binary_logloss: 0.144095
    [8]	valid_0's auc: 0.847609	valid_0's binary_logloss: 0.137638	valid_1's auc: 0.827987	valid_1's binary_logloss: 0.143002
    [9]	valid_0's auc: 0.848913	valid_0's binary_logloss: 0.136364	valid_1's auc: 0.82986	valid_1's binary_logloss: 0.141984
    [10]	valid_0's auc: 0.850178	valid_0's binary_logloss: 0.135236	valid_1's auc: 0.829651	valid_1's binary_logloss: 0.141228
    [11]	valid_0's auc: 0.85127	valid_0's binary_logloss: 0.134273	valid_1's auc: 0.830496	valid_1's binary_logloss: 0.140509
    [12]	valid_0's auc: 0.852586	valid_0's binary_logloss: 0.133411	valid_1's auc: 0.830525	valid_1's binary_logloss: 0.13997
    [13]	valid_0's auc: 0.853732	valid_0's binary_logloss: 0.132666	valid_1's auc: 0.829319	valid_1's binary_logloss: 0.13957
    [14]	valid_0's auc: 0.854775	valid_0's binary_logloss: 0.13201	valid_1's auc: 0.832116	valid_1's binary_logloss: 0.139106
    [15]	valid_0's auc: 0.855714	valid_0's binary_logloss: 0.131409	valid_1's auc: 0.833454	valid_1's binary_logloss: 0.138684
    [16]	valid_0's auc: 0.856099	valid_0's binary_logloss: 0.130853	valid_1's auc: 0.832243	valid_1's binary_logloss: 0.138485
    [17]	valid_0's auc: 0.857475	valid_0's binary_logloss: 0.130295	valid_1's auc: 0.832577	valid_1's binary_logloss: 0.138179
    [18]	valid_0's auc: 0.857908	valid_0's binary_logloss: 0.129866	valid_1's auc: 0.832652	valid_1's binary_logloss: 0.138084
    [19]	valid_0's auc: 0.859226	valid_0's binary_logloss: 0.129384	valid_1's auc: 0.832887	valid_1's binary_logloss: 0.137828
    [20]	valid_0's auc: 0.860296	valid_0's binary_logloss: 0.129003	valid_1's auc: 0.834111	valid_1's binary_logloss: 0.137643
    [21]	valid_0's auc: 0.860934	valid_0's binary_logloss: 0.128619	valid_1's auc: 0.834602	valid_1's binary_logloss: 0.137471
    [22]	valid_0's auc: 0.861712	valid_0's binary_logloss: 0.12825	valid_1's auc: 0.834828	valid_1's binary_logloss: 0.137291
    [23]	valid_0's auc: 0.862939	valid_0's binary_logloss: 0.127869	valid_1's auc: 0.835767	valid_1's binary_logloss: 0.137067
    [24]	valid_0's auc: 0.863968	valid_0's binary_logloss: 0.127525	valid_1's auc: 0.835485	valid_1's binary_logloss: 0.13701
    [25]	valid_0's auc: 0.865029	valid_0's binary_logloss: 0.127234	valid_1's auc: 0.835507	valid_1's binary_logloss: 0.136911
    [26]	valid_0's auc: 0.865758	valid_0's binary_logloss: 0.126952	valid_1's auc: 0.835743	valid_1's binary_logloss: 0.136823
    [27]	valid_0's auc: 0.866244	valid_0's binary_logloss: 0.126688	valid_1's auc: 0.836049	valid_1's binary_logloss: 0.136736
    [28]	valid_0's auc: 0.866607	valid_0's binary_logloss: 0.126455	valid_1's auc: 0.836043	valid_1's binary_logloss: 0.136687
    [29]	valid_0's auc: 0.867322	valid_0's binary_logloss: 0.126255	valid_1's auc: 0.836375	valid_1's binary_logloss: 0.136596
    [30]	valid_0's auc: 0.867997	valid_0's binary_logloss: 0.126022	valid_1's auc: 0.836518	valid_1's binary_logloss: 0.136563
    [31]	valid_0's auc: 0.868888	valid_0's binary_logloss: 0.125781	valid_1's auc: 0.836949	valid_1's binary_logloss: 0.13648
    [32]	valid_0's auc: 0.869544	valid_0's binary_logloss: 0.125524	valid_1's auc: 0.836774	valid_1's binary_logloss: 0.136506
    [33]	valid_0's auc: 0.870314	valid_0's binary_logloss: 0.125324	valid_1's auc: 0.837039	valid_1's binary_logloss: 0.13642
    [34]	valid_0's auc: 0.871207	valid_0's binary_logloss: 0.125096	valid_1's auc: 0.836859	valid_1's binary_logloss: 0.136441
    [35]	valid_0's auc: 0.871788	valid_0's binary_logloss: 0.124883	valid_1's auc: 0.836641	valid_1's binary_logloss: 0.136467
    [36]	valid_0's auc: 0.872403	valid_0's binary_logloss: 0.124635	valid_1's auc: 0.836917	valid_1's binary_logloss: 0.13642
    [37]	valid_0's auc: 0.872914	valid_0's binary_logloss: 0.124439	valid_1's auc: 0.836892	valid_1's binary_logloss: 0.136489
    [38]	valid_0's auc: 0.873469	valid_0's binary_logloss: 0.124233	valid_1's auc: 0.836816	valid_1's binary_logloss: 0.136502
    [39]	valid_0's auc: 0.873971	valid_0's binary_logloss: 0.124017	valid_1's auc: 0.83659	valid_1's binary_logloss: 0.136543
    [40]	valid_0's auc: 0.875013	valid_0's binary_logloss: 0.123715	valid_1's auc: 0.836381	valid_1's binary_logloss: 0.136593
    [41]	valid_0's auc: 0.875341	valid_0's binary_logloss: 0.123538	valid_1's auc: 0.8363	valid_1's binary_logloss: 0.136624
    [42]	valid_0's auc: 0.875886	valid_0's binary_logloss: 0.123285	valid_1's auc: 0.836043	valid_1's binary_logloss: 0.136665
    [43]	valid_0's auc: 0.87633	valid_0's binary_logloss: 0.12311	valid_1's auc: 0.836018	valid_1's binary_logloss: 0.136647
    [44]	valid_0's auc: 0.876816	valid_0's binary_logloss: 0.122878	valid_1's auc: 0.836061	valid_1's binary_logloss: 0.136664
    [45]	valid_0's auc: 0.877116	valid_0's binary_logloss: 0.12271	valid_1's auc: 0.836517	valid_1's binary_logloss: 0.136608
    [46]	valid_0's auc: 0.877567	valid_0's binary_logloss: 0.122536	valid_1's auc: 0.836051	valid_1's binary_logloss: 0.13667
    [47]	valid_0's auc: 0.878192	valid_0's binary_logloss: 0.122332	valid_1's auc: 0.835827	valid_1's binary_logloss: 0.136694
    [48]	valid_0's auc: 0.878371	valid_0's binary_logloss: 0.122208	valid_1's auc: 0.835831	valid_1's binary_logloss: 0.136717
    [49]	valid_0's auc: 0.878669	valid_0's binary_logloss: 0.12204	valid_1's auc: 0.835913	valid_1's binary_logloss: 0.136679
    [50]	valid_0's auc: 0.879288	valid_0's binary_logloss: 0.121825	valid_1's auc: 0.835468	valid_1's binary_logloss: 0.136734
    [51]	valid_0's auc: 0.879552	valid_0's binary_logloss: 0.121647	valid_1's auc: 0.835095	valid_1's binary_logloss: 0.13684
    [52]	valid_0's auc: 0.880033	valid_0's binary_logloss: 0.121459	valid_1's auc: 0.83496	valid_1's binary_logloss: 0.136927
    [53]	valid_0's auc: 0.880447	valid_0's binary_logloss: 0.121297	valid_1's auc: 0.834955	valid_1's binary_logloss: 0.136953
    [54]	valid_0's auc: 0.88087	valid_0's binary_logloss: 0.121116	valid_1's auc: 0.834926	valid_1's binary_logloss: 0.136994
    [55]	valid_0's auc: 0.881105	valid_0's binary_logloss: 0.12097	valid_1's auc: 0.834861	valid_1's binary_logloss: 0.137043
    [56]	valid_0's auc: 0.88161	valid_0's binary_logloss: 0.120772	valid_1's auc: 0.834629	valid_1's binary_logloss: 0.137125
    [57]	valid_0's auc: 0.882134	valid_0's binary_logloss: 0.120595	valid_1's auc: 0.834576	valid_1's binary_logloss: 0.137157
    [58]	valid_0's auc: 0.882429	valid_0's binary_logloss: 0.120441	valid_1's auc: 0.834319	valid_1's binary_logloss: 0.137209
    [59]	valid_0's auc: 0.882799	valid_0's binary_logloss: 0.120268	valid_1's auc: 0.834028	valid_1's binary_logloss: 0.1373
    [60]	valid_0's auc: 0.883147	valid_0's binary_logloss: 0.120094	valid_1's auc: 0.833999	valid_1's binary_logloss: 0.137339
    [61]	valid_0's auc: 0.883628	valid_0's binary_logloss: 0.119914	valid_1's auc: 0.834234	valid_1's binary_logloss: 0.137314
    [62]	valid_0's auc: 0.884034	valid_0's binary_logloss: 0.119753	valid_1's auc: 0.834179	valid_1's binary_logloss: 0.137356
    [63]	valid_0's auc: 0.884169	valid_0's binary_logloss: 0.119623	valid_1's auc: 0.834035	valid_1's binary_logloss: 0.137378
    Early stopping, best iteration is:
    [33]	valid_0's auc: 0.870314	valid_0's binary_logloss: 0.125324	valid_1's auc: 0.837039	valid_1's binary_logloss: 0.13642
    [1]	valid_0's auc: 0.826066	valid_0's binary_logloss: 0.157509	valid_1's auc: 0.810763	valid_1's binary_logloss: 0.160177
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.830558	valid_0's binary_logloss: 0.152317	valid_1's auc: 0.814406	valid_1's binary_logloss: 0.15553
    [3]	valid_0's auc: 0.833526	valid_0's binary_logloss: 0.148529	valid_1's auc: 0.815841	valid_1's binary_logloss: 0.152134
    [4]	valid_0's auc: 0.839653	valid_0's binary_logloss: 0.145555	valid_1's auc: 0.821767	valid_1's binary_logloss: 0.149519
    [5]	valid_0's auc: 0.842943	valid_0's binary_logloss: 0.143073	valid_1's auc: 0.826334	valid_1's binary_logloss: 0.14736
    [6]	valid_0's auc: 0.845929	valid_0's binary_logloss: 0.141096	valid_1's auc: 0.828882	valid_1's binary_logloss: 0.145649
    [7]	valid_0's auc: 0.847231	valid_0's binary_logloss: 0.139394	valid_1's auc: 0.830621	valid_1's binary_logloss: 0.144201
    [8]	valid_0's auc: 0.849609	valid_0's binary_logloss: 0.137956	valid_1's auc: 0.831324	valid_1's binary_logloss: 0.143105
    [9]	valid_0's auc: 0.851333	valid_0's binary_logloss: 0.136733	valid_1's auc: 0.83377	valid_1's binary_logloss: 0.142018
    [10]	valid_0's auc: 0.852692	valid_0's binary_logloss: 0.135654	valid_1's auc: 0.834493	valid_1's binary_logloss: 0.141149
    [11]	valid_0's auc: 0.854022	valid_0's binary_logloss: 0.134688	valid_1's auc: 0.834616	valid_1's binary_logloss: 0.140437
    [12]	valid_0's auc: 0.854419	valid_0's binary_logloss: 0.13393	valid_1's auc: 0.83589	valid_1's binary_logloss: 0.139765
    [13]	valid_0's auc: 0.854853	valid_0's binary_logloss: 0.133229	valid_1's auc: 0.835999	valid_1's binary_logloss: 0.139284
    [14]	valid_0's auc: 0.856241	valid_0's binary_logloss: 0.132532	valid_1's auc: 0.837392	valid_1's binary_logloss: 0.138795
    [15]	valid_0's auc: 0.85803	valid_0's binary_logloss: 0.131862	valid_1's auc: 0.83818	valid_1's binary_logloss: 0.138366
    [16]	valid_0's auc: 0.859289	valid_0's binary_logloss: 0.131311	valid_1's auc: 0.838523	valid_1's binary_logloss: 0.138137
    [17]	valid_0's auc: 0.860115	valid_0's binary_logloss: 0.130812	valid_1's auc: 0.839359	valid_1's binary_logloss: 0.137816
    [18]	valid_0's auc: 0.860807	valid_0's binary_logloss: 0.130408	valid_1's auc: 0.840179	valid_1's binary_logloss: 0.13749
    [19]	valid_0's auc: 0.861568	valid_0's binary_logloss: 0.129954	valid_1's auc: 0.839855	valid_1's binary_logloss: 0.137387
    [20]	valid_0's auc: 0.863194	valid_0's binary_logloss: 0.129482	valid_1's auc: 0.839715	valid_1's binary_logloss: 0.137226
    [21]	valid_0's auc: 0.864273	valid_0's binary_logloss: 0.12907	valid_1's auc: 0.840516	valid_1's binary_logloss: 0.136991
    [22]	valid_0's auc: 0.865334	valid_0's binary_logloss: 0.128648	valid_1's auc: 0.841302	valid_1's binary_logloss: 0.136769
    [23]	valid_0's auc: 0.866353	valid_0's binary_logloss: 0.128302	valid_1's auc: 0.841085	valid_1's binary_logloss: 0.136653
    [24]	valid_0's auc: 0.86699	valid_0's binary_logloss: 0.127957	valid_1's auc: 0.841235	valid_1's binary_logloss: 0.136611
    [25]	valid_0's auc: 0.867587	valid_0's binary_logloss: 0.127634	valid_1's auc: 0.840948	valid_1's binary_logloss: 0.136644
    [26]	valid_0's auc: 0.868507	valid_0's binary_logloss: 0.127298	valid_1's auc: 0.841033	valid_1's binary_logloss: 0.136526
    [27]	valid_0's auc: 0.869362	valid_0's binary_logloss: 0.126976	valid_1's auc: 0.840939	valid_1's binary_logloss: 0.136445
    [28]	valid_0's auc: 0.870126	valid_0's binary_logloss: 0.126688	valid_1's auc: 0.840452	valid_1's binary_logloss: 0.13645
    [29]	valid_0's auc: 0.870568	valid_0's binary_logloss: 0.126442	valid_1's auc: 0.840809	valid_1's binary_logloss: 0.136345
    [30]	valid_0's auc: 0.871323	valid_0's binary_logloss: 0.126166	valid_1's auc: 0.840599	valid_1's binary_logloss: 0.136334
    [31]	valid_0's auc: 0.872181	valid_0's binary_logloss: 0.125929	valid_1's auc: 0.840663	valid_1's binary_logloss: 0.136267
    [32]	valid_0's auc: 0.873089	valid_0's binary_logloss: 0.125641	valid_1's auc: 0.841679	valid_1's binary_logloss: 0.136141
    [33]	valid_0's auc: 0.873443	valid_0's binary_logloss: 0.125415	valid_1's auc: 0.841597	valid_1's binary_logloss: 0.136138
    [34]	valid_0's auc: 0.874111	valid_0's binary_logloss: 0.125163	valid_1's auc: 0.841194	valid_1's binary_logloss: 0.13614
    [35]	valid_0's auc: 0.874804	valid_0's binary_logloss: 0.124891	valid_1's auc: 0.841136	valid_1's binary_logloss: 0.136214
    [36]	valid_0's auc: 0.875123	valid_0's binary_logloss: 0.124721	valid_1's auc: 0.840883	valid_1's binary_logloss: 0.136268
    [37]	valid_0's auc: 0.875964	valid_0's binary_logloss: 0.124479	valid_1's auc: 0.840327	valid_1's binary_logloss: 0.136363
    [38]	valid_0's auc: 0.876484	valid_0's binary_logloss: 0.12422	valid_1's auc: 0.840317	valid_1's binary_logloss: 0.136378
    [39]	valid_0's auc: 0.877529	valid_0's binary_logloss: 0.123975	valid_1's auc: 0.840224	valid_1's binary_logloss: 0.136372
    [40]	valid_0's auc: 0.877916	valid_0's binary_logloss: 0.123803	valid_1's auc: 0.840099	valid_1's binary_logloss: 0.136418
    [41]	valid_0's auc: 0.878404	valid_0's binary_logloss: 0.123606	valid_1's auc: 0.839447	valid_1's binary_logloss: 0.136498
    [42]	valid_0's auc: 0.878787	valid_0's binary_logloss: 0.123458	valid_1's auc: 0.839737	valid_1's binary_logloss: 0.136466
    [43]	valid_0's auc: 0.879296	valid_0's binary_logloss: 0.123263	valid_1's auc: 0.838848	valid_1's binary_logloss: 0.136611
    [44]	valid_0's auc: 0.879561	valid_0's binary_logloss: 0.123111	valid_1's auc: 0.83939	valid_1's binary_logloss: 0.136509
    [45]	valid_0's auc: 0.880167	valid_0's binary_logloss: 0.122912	valid_1's auc: 0.83916	valid_1's binary_logloss: 0.136564
    [46]	valid_0's auc: 0.880652	valid_0's binary_logloss: 0.12273	valid_1's auc: 0.839133	valid_1's binary_logloss: 0.136588
    [47]	valid_0's auc: 0.881351	valid_0's binary_logloss: 0.122478	valid_1's auc: 0.839108	valid_1's binary_logloss: 0.136599
    [48]	valid_0's auc: 0.881786	valid_0's binary_logloss: 0.122308	valid_1's auc: 0.838894	valid_1's binary_logloss: 0.136636
    [49]	valid_0's auc: 0.88242	valid_0's binary_logloss: 0.122054	valid_1's auc: 0.838476	valid_1's binary_logloss: 0.136727
    [50]	valid_0's auc: 0.88279	valid_0's binary_logloss: 0.121862	valid_1's auc: 0.837735	valid_1's binary_logloss: 0.136833
    [51]	valid_0's auc: 0.883422	valid_0's binary_logloss: 0.121646	valid_1's auc: 0.837691	valid_1's binary_logloss: 0.136833
    [52]	valid_0's auc: 0.884035	valid_0's binary_logloss: 0.121461	valid_1's auc: 0.837503	valid_1's binary_logloss: 0.136856
    [53]	valid_0's auc: 0.884636	valid_0's binary_logloss: 0.121237	valid_1's auc: 0.837159	valid_1's binary_logloss: 0.136933
    [54]	valid_0's auc: 0.885019	valid_0's binary_logloss: 0.121086	valid_1's auc: 0.836758	valid_1's binary_logloss: 0.136995
    [55]	valid_0's auc: 0.88546	valid_0's binary_logloss: 0.120908	valid_1's auc: 0.836692	valid_1's binary_logloss: 0.137006
    [56]	valid_0's auc: 0.885711	valid_0's binary_logloss: 0.120745	valid_1's auc: 0.836703	valid_1's binary_logloss: 0.136998
    [57]	valid_0's auc: 0.886081	valid_0's binary_logloss: 0.120544	valid_1's auc: 0.836462	valid_1's binary_logloss: 0.137073
    [58]	valid_0's auc: 0.886576	valid_0's binary_logloss: 0.120352	valid_1's auc: 0.836241	valid_1's binary_logloss: 0.137127
    [59]	valid_0's auc: 0.886878	valid_0's binary_logloss: 0.120191	valid_1's auc: 0.8358	valid_1's binary_logloss: 0.137207
    [60]	valid_0's auc: 0.887158	valid_0's binary_logloss: 0.120005	valid_1's auc: 0.835688	valid_1's binary_logloss: 0.137272
    [61]	valid_0's auc: 0.88755	valid_0's binary_logloss: 0.119855	valid_1's auc: 0.835652	valid_1's binary_logloss: 0.137262
    [62]	valid_0's auc: 0.887817	valid_0's binary_logloss: 0.11971	valid_1's auc: 0.835355	valid_1's binary_logloss: 0.13732
    Early stopping, best iteration is:
    [32]	valid_0's auc: 0.873089	valid_0's binary_logloss: 0.125641	valid_1's auc: 0.841679	valid_1's binary_logloss: 0.136141
    [1]	valid_0's auc: 0.822527	valid_0's binary_logloss: 0.157702	valid_1's auc: 0.811735	valid_1's binary_logloss: 0.160053
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.828897	valid_0's binary_logloss: 0.152381	valid_1's auc: 0.815279	valid_1's binary_logloss: 0.155308
    [3]	valid_0's auc: 0.833978	valid_0's binary_logloss: 0.148529	valid_1's auc: 0.819863	valid_1's binary_logloss: 0.151959
    [4]	valid_0's auc: 0.837641	valid_0's binary_logloss: 0.145558	valid_1's auc: 0.821032	valid_1's binary_logloss: 0.149309
    [5]	valid_0's auc: 0.841306	valid_0's binary_logloss: 0.143184	valid_1's auc: 0.824926	valid_1's binary_logloss: 0.147249
    [6]	valid_0's auc: 0.843853	valid_0's binary_logloss: 0.141094	valid_1's auc: 0.825446	valid_1's binary_logloss: 0.145583
    [7]	valid_0's auc: 0.846589	valid_0's binary_logloss: 0.13939	valid_1's auc: 0.829069	valid_1's binary_logloss: 0.144172
    [8]	valid_0's auc: 0.847805	valid_0's binary_logloss: 0.137988	valid_1's auc: 0.829457	valid_1's binary_logloss: 0.143004
    [9]	valid_0's auc: 0.850007	valid_0's binary_logloss: 0.136683	valid_1's auc: 0.831298	valid_1's binary_logloss: 0.141959
    [10]	valid_0's auc: 0.850514	valid_0's binary_logloss: 0.135597	valid_1's auc: 0.830745	valid_1's binary_logloss: 0.141234
    [11]	valid_0's auc: 0.851694	valid_0's binary_logloss: 0.134644	valid_1's auc: 0.831218	valid_1's binary_logloss: 0.140531
    [12]	valid_0's auc: 0.854647	valid_0's binary_logloss: 0.133779	valid_1's auc: 0.832445	valid_1's binary_logloss: 0.140011
    [13]	valid_0's auc: 0.8552	valid_0's binary_logloss: 0.13303	valid_1's auc: 0.832525	valid_1's binary_logloss: 0.139542
    [14]	valid_0's auc: 0.85611	valid_0's binary_logloss: 0.132355	valid_1's auc: 0.832559	valid_1's binary_logloss: 0.139125
    [15]	valid_0's auc: 0.856886	valid_0's binary_logloss: 0.131779	valid_1's auc: 0.833401	valid_1's binary_logloss: 0.138708
    [16]	valid_0's auc: 0.857397	valid_0's binary_logloss: 0.13126	valid_1's auc: 0.833709	valid_1's binary_logloss: 0.138376
    [17]	valid_0's auc: 0.859038	valid_0's binary_logloss: 0.130686	valid_1's auc: 0.83451	valid_1's binary_logloss: 0.138079
    [18]	valid_0's auc: 0.859754	valid_0's binary_logloss: 0.130225	valid_1's auc: 0.834886	valid_1's binary_logloss: 0.137839
    [19]	valid_0's auc: 0.861354	valid_0's binary_logloss: 0.129689	valid_1's auc: 0.835687	valid_1's binary_logloss: 0.137572
    [20]	valid_0's auc: 0.862114	valid_0's binary_logloss: 0.129263	valid_1's auc: 0.83587	valid_1's binary_logloss: 0.137393
    [21]	valid_0's auc: 0.863196	valid_0's binary_logloss: 0.128834	valid_1's auc: 0.835611	valid_1's binary_logloss: 0.137244
    [22]	valid_0's auc: 0.864467	valid_0's binary_logloss: 0.128421	valid_1's auc: 0.836013	valid_1's binary_logloss: 0.137127
    [23]	valid_0's auc: 0.865716	valid_0's binary_logloss: 0.12802	valid_1's auc: 0.835915	valid_1's binary_logloss: 0.13701
    [24]	valid_0's auc: 0.866358	valid_0's binary_logloss: 0.127649	valid_1's auc: 0.835345	valid_1's binary_logloss: 0.136983
    [25]	valid_0's auc: 0.867017	valid_0's binary_logloss: 0.127326	valid_1's auc: 0.835203	valid_1's binary_logloss: 0.13697
    [26]	valid_0's auc: 0.867793	valid_0's binary_logloss: 0.126998	valid_1's auc: 0.835068	valid_1's binary_logloss: 0.13692
    [27]	valid_0's auc: 0.86892	valid_0's binary_logloss: 0.126634	valid_1's auc: 0.835127	valid_1's binary_logloss: 0.136839
    [28]	valid_0's auc: 0.869611	valid_0's binary_logloss: 0.126332	valid_1's auc: 0.835011	valid_1's binary_logloss: 0.136765
    [29]	valid_0's auc: 0.870502	valid_0's binary_logloss: 0.126032	valid_1's auc: 0.835017	valid_1's binary_logloss: 0.136691
    [30]	valid_0's auc: 0.87109	valid_0's binary_logloss: 0.125752	valid_1's auc: 0.835346	valid_1's binary_logloss: 0.136657
    [31]	valid_0's auc: 0.871834	valid_0's binary_logloss: 0.125492	valid_1's auc: 0.835469	valid_1's binary_logloss: 0.136553
    [32]	valid_0's auc: 0.8727	valid_0's binary_logloss: 0.125224	valid_1's auc: 0.834809	valid_1's binary_logloss: 0.136626
    [33]	valid_0's auc: 0.873327	valid_0's binary_logloss: 0.12497	valid_1's auc: 0.834899	valid_1's binary_logloss: 0.136576
    [34]	valid_0's auc: 0.873759	valid_0's binary_logloss: 0.124747	valid_1's auc: 0.835299	valid_1's binary_logloss: 0.136478
    [35]	valid_0's auc: 0.874436	valid_0's binary_logloss: 0.124498	valid_1's auc: 0.834852	valid_1's binary_logloss: 0.136556
    [36]	valid_0's auc: 0.875262	valid_0's binary_logloss: 0.124237	valid_1's auc: 0.834479	valid_1's binary_logloss: 0.13659
    [37]	valid_0's auc: 0.875938	valid_0's binary_logloss: 0.123998	valid_1's auc: 0.834359	valid_1's binary_logloss: 0.136644
    [38]	valid_0's auc: 0.876343	valid_0's binary_logloss: 0.123795	valid_1's auc: 0.834106	valid_1's binary_logloss: 0.136687
    [39]	valid_0's auc: 0.877139	valid_0's binary_logloss: 0.123529	valid_1's auc: 0.83359	valid_1's binary_logloss: 0.136724
    [40]	valid_0's auc: 0.877911	valid_0's binary_logloss: 0.123326	valid_1's auc: 0.833512	valid_1's binary_logloss: 0.136737
    [41]	valid_0's auc: 0.87875	valid_0's binary_logloss: 0.123086	valid_1's auc: 0.833202	valid_1's binary_logloss: 0.136752
    [42]	valid_0's auc: 0.879342	valid_0's binary_logloss: 0.122885	valid_1's auc: 0.832649	valid_1's binary_logloss: 0.136842
    [43]	valid_0's auc: 0.879629	valid_0's binary_logloss: 0.122717	valid_1's auc: 0.832179	valid_1's binary_logloss: 0.136927
    [44]	valid_0's auc: 0.880215	valid_0's binary_logloss: 0.122487	valid_1's auc: 0.832172	valid_1's binary_logloss: 0.13695
    [45]	valid_0's auc: 0.880851	valid_0's binary_logloss: 0.122245	valid_1's auc: 0.831665	valid_1's binary_logloss: 0.137053
    [46]	valid_0's auc: 0.881492	valid_0's binary_logloss: 0.121994	valid_1's auc: 0.83185	valid_1's binary_logloss: 0.137071
    [47]	valid_0's auc: 0.881818	valid_0's binary_logloss: 0.121836	valid_1's auc: 0.831946	valid_1's binary_logloss: 0.137043
    [48]	valid_0's auc: 0.882136	valid_0's binary_logloss: 0.121683	valid_1's auc: 0.832078	valid_1's binary_logloss: 0.137058
    [49]	valid_0's auc: 0.882509	valid_0's binary_logloss: 0.121512	valid_1's auc: 0.832046	valid_1's binary_logloss: 0.13711
    [50]	valid_0's auc: 0.882872	valid_0's binary_logloss: 0.121296	valid_1's auc: 0.83209	valid_1's binary_logloss: 0.137124
    [51]	valid_0's auc: 0.883328	valid_0's binary_logloss: 0.121099	valid_1's auc: 0.831958	valid_1's binary_logloss: 0.137201
    [52]	valid_0's auc: 0.883722	valid_0's binary_logloss: 0.120897	valid_1's auc: 0.831752	valid_1's binary_logloss: 0.13722
    Early stopping, best iteration is:
    [22]	valid_0's auc: 0.864467	valid_0's binary_logloss: 0.128421	valid_1's auc: 0.836013	valid_1's binary_logloss: 0.137127
    [1]	valid_0's auc: 0.829442	valid_0's binary_logloss: 0.156848	valid_1's auc: 0.801853	valid_1's binary_logloss: 0.159917
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.835044	valid_0's binary_logloss: 0.151129	valid_1's auc: 0.810636	valid_1's binary_logloss: 0.155191
    [3]	valid_0's auc: 0.84106	valid_0's binary_logloss: 0.146986	valid_1's auc: 0.81428	valid_1's binary_logloss: 0.151786
    [4]	valid_0's auc: 0.849104	valid_0's binary_logloss: 0.143746	valid_1's auc: 0.822369	valid_1's binary_logloss: 0.149174
    [5]	valid_0's auc: 0.852054	valid_0's binary_logloss: 0.14111	valid_1's auc: 0.825422	valid_1's binary_logloss: 0.147108
    [6]	valid_0's auc: 0.854219	valid_0's binary_logloss: 0.138941	valid_1's auc: 0.828128	valid_1's binary_logloss: 0.145314
    [7]	valid_0's auc: 0.857758	valid_0's binary_logloss: 0.136956	valid_1's auc: 0.828178	valid_1's binary_logloss: 0.144053
    [8]	valid_0's auc: 0.859433	valid_0's binary_logloss: 0.135341	valid_1's auc: 0.828907	valid_1's binary_logloss: 0.142933
    [9]	valid_0's auc: 0.860562	valid_0's binary_logloss: 0.133965	valid_1's auc: 0.830266	valid_1's binary_logloss: 0.141913
    [10]	valid_0's auc: 0.862253	valid_0's binary_logloss: 0.132732	valid_1's auc: 0.830436	valid_1's binary_logloss: 0.141159
    [11]	valid_0's auc: 0.863863	valid_0's binary_logloss: 0.131616	valid_1's auc: 0.830888	valid_1's binary_logloss: 0.140465
    [12]	valid_0's auc: 0.86535	valid_0's binary_logloss: 0.130642	valid_1's auc: 0.828473	valid_1's binary_logloss: 0.140089
    [13]	valid_0's auc: 0.866315	valid_0's binary_logloss: 0.129707	valid_1's auc: 0.828442	valid_1's binary_logloss: 0.139673
    [14]	valid_0's auc: 0.868091	valid_0's binary_logloss: 0.128809	valid_1's auc: 0.828803	valid_1's binary_logloss: 0.13925
    [15]	valid_0's auc: 0.869154	valid_0's binary_logloss: 0.128055	valid_1's auc: 0.829147	valid_1's binary_logloss: 0.138981
    [16]	valid_0's auc: 0.870086	valid_0's binary_logloss: 0.127376	valid_1's auc: 0.829188	valid_1's binary_logloss: 0.13878
    [17]	valid_0's auc: 0.870857	valid_0's binary_logloss: 0.126705	valid_1's auc: 0.829277	valid_1's binary_logloss: 0.138553
    [18]	valid_0's auc: 0.87202	valid_0's binary_logloss: 0.126023	valid_1's auc: 0.829614	valid_1's binary_logloss: 0.138375
    [19]	valid_0's auc: 0.873171	valid_0's binary_logloss: 0.125459	valid_1's auc: 0.830335	valid_1's binary_logloss: 0.13824
    [20]	valid_0's auc: 0.874593	valid_0's binary_logloss: 0.124876	valid_1's auc: 0.831752	valid_1's binary_logloss: 0.13798
    [21]	valid_0's auc: 0.875741	valid_0's binary_logloss: 0.124281	valid_1's auc: 0.832761	valid_1's binary_logloss: 0.137687
    [22]	valid_0's auc: 0.877448	valid_0's binary_logloss: 0.123659	valid_1's auc: 0.833679	valid_1's binary_logloss: 0.137492
    [23]	valid_0's auc: 0.878493	valid_0's binary_logloss: 0.123115	valid_1's auc: 0.833912	valid_1's binary_logloss: 0.137434
    [24]	valid_0's auc: 0.879193	valid_0's binary_logloss: 0.122682	valid_1's auc: 0.833363	valid_1's binary_logloss: 0.137364
    [25]	valid_0's auc: 0.880227	valid_0's binary_logloss: 0.122229	valid_1's auc: 0.833173	valid_1's binary_logloss: 0.137358
    [26]	valid_0's auc: 0.881585	valid_0's binary_logloss: 0.121752	valid_1's auc: 0.832904	valid_1's binary_logloss: 0.13741
    [27]	valid_0's auc: 0.882473	valid_0's binary_logloss: 0.1213	valid_1's auc: 0.832408	valid_1's binary_logloss: 0.137589
    [28]	valid_0's auc: 0.883437	valid_0's binary_logloss: 0.120929	valid_1's auc: 0.832986	valid_1's binary_logloss: 0.137512
    [29]	valid_0's auc: 0.884502	valid_0's binary_logloss: 0.120484	valid_1's auc: 0.833732	valid_1's binary_logloss: 0.137414
    [30]	valid_0's auc: 0.88539	valid_0's binary_logloss: 0.120132	valid_1's auc: 0.83355	valid_1's binary_logloss: 0.137429
    [31]	valid_0's auc: 0.886163	valid_0's binary_logloss: 0.11975	valid_1's auc: 0.833288	valid_1's binary_logloss: 0.137459
    [32]	valid_0's auc: 0.886867	valid_0's binary_logloss: 0.119396	valid_1's auc: 0.833051	valid_1's binary_logloss: 0.137503
    [33]	valid_0's auc: 0.887733	valid_0's binary_logloss: 0.119013	valid_1's auc: 0.832946	valid_1's binary_logloss: 0.137467
    [34]	valid_0's auc: 0.888681	valid_0's binary_logloss: 0.118651	valid_1's auc: 0.833519	valid_1's binary_logloss: 0.137391
    [35]	valid_0's auc: 0.889275	valid_0's binary_logloss: 0.118341	valid_1's auc: 0.833734	valid_1's binary_logloss: 0.137371
    [36]	valid_0's auc: 0.890084	valid_0's binary_logloss: 0.117965	valid_1's auc: 0.833231	valid_1's binary_logloss: 0.137505
    [37]	valid_0's auc: 0.890726	valid_0's binary_logloss: 0.11765	valid_1's auc: 0.833136	valid_1's binary_logloss: 0.13761
    [38]	valid_0's auc: 0.891158	valid_0's binary_logloss: 0.11734	valid_1's auc: 0.832938	valid_1's binary_logloss: 0.137668
    [39]	valid_0's auc: 0.89196	valid_0's binary_logloss: 0.117006	valid_1's auc: 0.832307	valid_1's binary_logloss: 0.13782
    [40]	valid_0's auc: 0.892449	valid_0's binary_logloss: 0.116703	valid_1's auc: 0.832354	valid_1's binary_logloss: 0.137858
    [41]	valid_0's auc: 0.892978	valid_0's binary_logloss: 0.116414	valid_1's auc: 0.831731	valid_1's binary_logloss: 0.137998
    [42]	valid_0's auc: 0.893542	valid_0's binary_logloss: 0.116071	valid_1's auc: 0.830942	valid_1's binary_logloss: 0.138189
    [43]	valid_0's auc: 0.894078	valid_0's binary_logloss: 0.115805	valid_1's auc: 0.830997	valid_1's binary_logloss: 0.138182
    [44]	valid_0's auc: 0.894576	valid_0's binary_logloss: 0.115516	valid_1's auc: 0.830968	valid_1's binary_logloss: 0.138257
    [45]	valid_0's auc: 0.895099	valid_0's binary_logloss: 0.115236	valid_1's auc: 0.830506	valid_1's binary_logloss: 0.13839
    [46]	valid_0's auc: 0.895608	valid_0's binary_logloss: 0.114971	valid_1's auc: 0.829892	valid_1's binary_logloss: 0.138556
    [47]	valid_0's auc: 0.895999	valid_0's binary_logloss: 0.1147	valid_1's auc: 0.829591	valid_1's binary_logloss: 0.138649
    [48]	valid_0's auc: 0.89641	valid_0's binary_logloss: 0.11442	valid_1's auc: 0.829768	valid_1's binary_logloss: 0.138698
    [49]	valid_0's auc: 0.896742	valid_0's binary_logloss: 0.114182	valid_1's auc: 0.830072	valid_1's binary_logloss: 0.138692
    [50]	valid_0's auc: 0.896929	valid_0's binary_logloss: 0.113989	valid_1's auc: 0.829948	valid_1's binary_logloss: 0.138769
    [51]	valid_0's auc: 0.897717	valid_0's binary_logloss: 0.113718	valid_1's auc: 0.830297	valid_1's binary_logloss: 0.138742
    [52]	valid_0's auc: 0.898093	valid_0's binary_logloss: 0.113472	valid_1's auc: 0.830332	valid_1's binary_logloss: 0.138776
    [53]	valid_0's auc: 0.898387	valid_0's binary_logloss: 0.113271	valid_1's auc: 0.830278	valid_1's binary_logloss: 0.138833
    Early stopping, best iteration is:
    [23]	valid_0's auc: 0.878493	valid_0's binary_logloss: 0.123115	valid_1's auc: 0.833912	valid_1's binary_logloss: 0.137434
    [1]	valid_0's auc: 0.834008	valid_0's binary_logloss: 0.156932	valid_1's auc: 0.806689	valid_1's binary_logloss: 0.159986
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.841271	valid_0's binary_logloss: 0.151528	valid_1's auc: 0.816568	valid_1's binary_logloss: 0.155337
    [3]	valid_0's auc: 0.843827	valid_0's binary_logloss: 0.147378	valid_1's auc: 0.818842	valid_1's binary_logloss: 0.151989
    [4]	valid_0's auc: 0.850477	valid_0's binary_logloss: 0.144112	valid_1's auc: 0.824908	valid_1's binary_logloss: 0.149225
    [5]	valid_0's auc: 0.853808	valid_0's binary_logloss: 0.141422	valid_1's auc: 0.826744	valid_1's binary_logloss: 0.147175
    [6]	valid_0's auc: 0.856663	valid_0's binary_logloss: 0.139222	valid_1's auc: 0.828432	valid_1's binary_logloss: 0.145499
    [7]	valid_0's auc: 0.859664	valid_0's binary_logloss: 0.137318	valid_1's auc: 0.829523	valid_1's binary_logloss: 0.14402
    [8]	valid_0's auc: 0.86186	valid_0's binary_logloss: 0.135665	valid_1's auc: 0.8314	valid_1's binary_logloss: 0.142812
    [9]	valid_0's auc: 0.863104	valid_0's binary_logloss: 0.134357	valid_1's auc: 0.831976	valid_1's binary_logloss: 0.141915
    [10]	valid_0's auc: 0.865027	valid_0's binary_logloss: 0.133117	valid_1's auc: 0.83369	valid_1's binary_logloss: 0.141045
    [11]	valid_0's auc: 0.866437	valid_0's binary_logloss: 0.132026	valid_1's auc: 0.834219	valid_1's binary_logloss: 0.140347
    [12]	valid_0's auc: 0.868039	valid_0's binary_logloss: 0.131061	valid_1's auc: 0.834388	valid_1's binary_logloss: 0.139858
    [13]	valid_0's auc: 0.869537	valid_0's binary_logloss: 0.130159	valid_1's auc: 0.835402	valid_1's binary_logloss: 0.139334
    [14]	valid_0's auc: 0.870971	valid_0's binary_logloss: 0.12932	valid_1's auc: 0.835583	valid_1's binary_logloss: 0.138962
    [15]	valid_0's auc: 0.872381	valid_0's binary_logloss: 0.128543	valid_1's auc: 0.836121	valid_1's binary_logloss: 0.138579
    [16]	valid_0's auc: 0.873871	valid_0's binary_logloss: 0.127833	valid_1's auc: 0.836592	valid_1's binary_logloss: 0.13832
    [17]	valid_0's auc: 0.87491	valid_0's binary_logloss: 0.127212	valid_1's auc: 0.836031	valid_1's binary_logloss: 0.138138
    [18]	valid_0's auc: 0.875677	valid_0's binary_logloss: 0.126616	valid_1's auc: 0.835779	valid_1's binary_logloss: 0.137944
    [19]	valid_0's auc: 0.876894	valid_0's binary_logloss: 0.126018	valid_1's auc: 0.836666	valid_1's binary_logloss: 0.137657
    [20]	valid_0's auc: 0.877834	valid_0's binary_logloss: 0.125465	valid_1's auc: 0.836513	valid_1's binary_logloss: 0.137603
    [21]	valid_0's auc: 0.878861	valid_0's binary_logloss: 0.124918	valid_1's auc: 0.836696	valid_1's binary_logloss: 0.137441
    [22]	valid_0's auc: 0.880257	valid_0's binary_logloss: 0.124391	valid_1's auc: 0.836449	valid_1's binary_logloss: 0.137422
    [23]	valid_0's auc: 0.8818	valid_0's binary_logloss: 0.123782	valid_1's auc: 0.836709	valid_1's binary_logloss: 0.137292
    [24]	valid_0's auc: 0.882923	valid_0's binary_logloss: 0.123257	valid_1's auc: 0.836765	valid_1's binary_logloss: 0.137199
    [25]	valid_0's auc: 0.883924	valid_0's binary_logloss: 0.122788	valid_1's auc: 0.836654	valid_1's binary_logloss: 0.137164
    [26]	valid_0's auc: 0.885046	valid_0's binary_logloss: 0.12229	valid_1's auc: 0.836564	valid_1's binary_logloss: 0.137112
    [27]	valid_0's auc: 0.886069	valid_0's binary_logloss: 0.121841	valid_1's auc: 0.836535	valid_1's binary_logloss: 0.137076
    [28]	valid_0's auc: 0.886927	valid_0's binary_logloss: 0.121432	valid_1's auc: 0.836263	valid_1's binary_logloss: 0.137134
    [29]	valid_0's auc: 0.887853	valid_0's binary_logloss: 0.121034	valid_1's auc: 0.836202	valid_1's binary_logloss: 0.137072
    [30]	valid_0's auc: 0.888728	valid_0's binary_logloss: 0.120631	valid_1's auc: 0.836619	valid_1's binary_logloss: 0.136982
    [31]	valid_0's auc: 0.889701	valid_0's binary_logloss: 0.120172	valid_1's auc: 0.836517	valid_1's binary_logloss: 0.136934
    [32]	valid_0's auc: 0.890993	valid_0's binary_logloss: 0.119756	valid_1's auc: 0.836361	valid_1's binary_logloss: 0.136931
    [33]	valid_0's auc: 0.892053	valid_0's binary_logloss: 0.119407	valid_1's auc: 0.836583	valid_1's binary_logloss: 0.136846
    [34]	valid_0's auc: 0.892821	valid_0's binary_logloss: 0.119049	valid_1's auc: 0.836327	valid_1's binary_logloss: 0.136961
    [35]	valid_0's auc: 0.893663	valid_0's binary_logloss: 0.118651	valid_1's auc: 0.836579	valid_1's binary_logloss: 0.136912
    [36]	valid_0's auc: 0.894523	valid_0's binary_logloss: 0.11827	valid_1's auc: 0.836209	valid_1's binary_logloss: 0.136964
    [37]	valid_0's auc: 0.895475	valid_0's binary_logloss: 0.117947	valid_1's auc: 0.835968	valid_1's binary_logloss: 0.13697
    [38]	valid_0's auc: 0.895896	valid_0's binary_logloss: 0.117661	valid_1's auc: 0.836021	valid_1's binary_logloss: 0.136984
    [39]	valid_0's auc: 0.896908	valid_0's binary_logloss: 0.117297	valid_1's auc: 0.835969	valid_1's binary_logloss: 0.137061
    [40]	valid_0's auc: 0.897428	valid_0's binary_logloss: 0.116993	valid_1's auc: 0.835781	valid_1's binary_logloss: 0.137135
    [41]	valid_0's auc: 0.89802	valid_0's binary_logloss: 0.116616	valid_1's auc: 0.834706	valid_1's binary_logloss: 0.137377
    [42]	valid_0's auc: 0.898553	valid_0's binary_logloss: 0.116307	valid_1's auc: 0.834495	valid_1's binary_logloss: 0.137447
    [43]	valid_0's auc: 0.898979	valid_0's binary_logloss: 0.11602	valid_1's auc: 0.834078	valid_1's binary_logloss: 0.137573
    [44]	valid_0's auc: 0.899439	valid_0's binary_logloss: 0.115752	valid_1's auc: 0.833605	valid_1's binary_logloss: 0.137706
    [45]	valid_0's auc: 0.899915	valid_0's binary_logloss: 0.115497	valid_1's auc: 0.833114	valid_1's binary_logloss: 0.137835
    [46]	valid_0's auc: 0.90021	valid_0's binary_logloss: 0.115245	valid_1's auc: 0.833104	valid_1's binary_logloss: 0.137879
    [47]	valid_0's auc: 0.90064	valid_0's binary_logloss: 0.114971	valid_1's auc: 0.83304	valid_1's binary_logloss: 0.137916
    [48]	valid_0's auc: 0.901303	valid_0's binary_logloss: 0.114665	valid_1's auc: 0.833281	valid_1's binary_logloss: 0.137925
    [49]	valid_0's auc: 0.901644	valid_0's binary_logloss: 0.114372	valid_1's auc: 0.83306	valid_1's binary_logloss: 0.138017
    [50]	valid_0's auc: 0.902036	valid_0's binary_logloss: 0.114104	valid_1's auc: 0.832675	valid_1's binary_logloss: 0.13811
    [51]	valid_0's auc: 0.902469	valid_0's binary_logloss: 0.113883	valid_1's auc: 0.832339	valid_1's binary_logloss: 0.138195
    [52]	valid_0's auc: 0.902667	valid_0's binary_logloss: 0.113693	valid_1's auc: 0.831957	valid_1's binary_logloss: 0.138298
    [53]	valid_0's auc: 0.902822	valid_0's binary_logloss: 0.113485	valid_1's auc: 0.831693	valid_1's binary_logloss: 0.138383
    [54]	valid_0's auc: 0.902977	valid_0's binary_logloss: 0.113301	valid_1's auc: 0.830964	valid_1's binary_logloss: 0.138566
    Early stopping, best iteration is:
    [24]	valid_0's auc: 0.882923	valid_0's binary_logloss: 0.123257	valid_1's auc: 0.836765	valid_1's binary_logloss: 0.137199
    [1]	valid_0's auc: 0.832048	valid_0's binary_logloss: 0.157225	valid_1's auc: 0.811916	valid_1's binary_logloss: 0.159885
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.843711	valid_0's binary_logloss: 0.151535	valid_1's auc: 0.821654	valid_1's binary_logloss: 0.154778
    [3]	valid_0's auc: 0.846336	valid_0's binary_logloss: 0.14743	valid_1's auc: 0.823358	valid_1's binary_logloss: 0.151371
    [4]	valid_0's auc: 0.849091	valid_0's binary_logloss: 0.144223	valid_1's auc: 0.826879	valid_1's binary_logloss: 0.148584
    [5]	valid_0's auc: 0.851041	valid_0's binary_logloss: 0.141653	valid_1's auc: 0.825653	valid_1's binary_logloss: 0.14656
    [6]	valid_0's auc: 0.853453	valid_0's binary_logloss: 0.139394	valid_1's auc: 0.828113	valid_1's binary_logloss: 0.144822
    [7]	valid_0's auc: 0.857868	valid_0's binary_logloss: 0.137439	valid_1's auc: 0.828786	valid_1's binary_logloss: 0.143565
    [8]	valid_0's auc: 0.860559	valid_0's binary_logloss: 0.135866	valid_1's auc: 0.829347	valid_1's binary_logloss: 0.142504
    [9]	valid_0's auc: 0.861868	valid_0's binary_logloss: 0.134472	valid_1's auc: 0.829546	valid_1's binary_logloss: 0.141596
    [10]	valid_0's auc: 0.862998	valid_0's binary_logloss: 0.133271	valid_1's auc: 0.828894	valid_1's binary_logloss: 0.140928
    [11]	valid_0's auc: 0.864761	valid_0's binary_logloss: 0.132122	valid_1's auc: 0.828717	valid_1's binary_logloss: 0.140351
    [12]	valid_0's auc: 0.865861	valid_0's binary_logloss: 0.131127	valid_1's auc: 0.827893	valid_1's binary_logloss: 0.139897
    [13]	valid_0's auc: 0.867377	valid_0's binary_logloss: 0.130201	valid_1's auc: 0.828411	valid_1's binary_logloss: 0.139452
    [14]	valid_0's auc: 0.868645	valid_0's binary_logloss: 0.129388	valid_1's auc: 0.828859	valid_1's binary_logloss: 0.139104
    [15]	valid_0's auc: 0.869811	valid_0's binary_logloss: 0.12864	valid_1's auc: 0.827872	valid_1's binary_logloss: 0.138919
    [16]	valid_0's auc: 0.870761	valid_0's binary_logloss: 0.127902	valid_1's auc: 0.827629	valid_1's binary_logloss: 0.138614
    [17]	valid_0's auc: 0.872392	valid_0's binary_logloss: 0.127247	valid_1's auc: 0.828475	valid_1's binary_logloss: 0.138424
    [18]	valid_0's auc: 0.873984	valid_0's binary_logloss: 0.126643	valid_1's auc: 0.828038	valid_1's binary_logloss: 0.13836
    [19]	valid_0's auc: 0.875277	valid_0's binary_logloss: 0.126018	valid_1's auc: 0.828307	valid_1's binary_logloss: 0.138229
    [20]	valid_0's auc: 0.876767	valid_0's binary_logloss: 0.125429	valid_1's auc: 0.828398	valid_1's binary_logloss: 0.138132
    [21]	valid_0's auc: 0.878127	valid_0's binary_logloss: 0.124871	valid_1's auc: 0.827975	valid_1's binary_logloss: 0.138032
    [22]	valid_0's auc: 0.879333	valid_0's binary_logloss: 0.12434	valid_1's auc: 0.828026	valid_1's binary_logloss: 0.137957
    [23]	valid_0's auc: 0.880379	valid_0's binary_logloss: 0.123879	valid_1's auc: 0.827567	valid_1's binary_logloss: 0.138018
    [24]	valid_0's auc: 0.881429	valid_0's binary_logloss: 0.123358	valid_1's auc: 0.827472	valid_1's binary_logloss: 0.137975
    [25]	valid_0's auc: 0.882335	valid_0's binary_logloss: 0.122902	valid_1's auc: 0.827456	valid_1's binary_logloss: 0.137997
    [26]	valid_0's auc: 0.883467	valid_0's binary_logloss: 0.122419	valid_1's auc: 0.827539	valid_1's binary_logloss: 0.137977
    [27]	valid_0's auc: 0.884573	valid_0's binary_logloss: 0.121954	valid_1's auc: 0.827608	valid_1's binary_logloss: 0.137962
    [28]	valid_0's auc: 0.885354	valid_0's binary_logloss: 0.12151	valid_1's auc: 0.827828	valid_1's binary_logloss: 0.137979
    [29]	valid_0's auc: 0.88633	valid_0's binary_logloss: 0.1211	valid_1's auc: 0.827743	valid_1's binary_logloss: 0.137916
    [30]	valid_0's auc: 0.887129	valid_0's binary_logloss: 0.120718	valid_1's auc: 0.82774	valid_1's binary_logloss: 0.13796
    [31]	valid_0's auc: 0.888235	valid_0's binary_logloss: 0.120283	valid_1's auc: 0.827163	valid_1's binary_logloss: 0.137963
    [32]	valid_0's auc: 0.889179	valid_0's binary_logloss: 0.119986	valid_1's auc: 0.826867	valid_1's binary_logloss: 0.13798
    [33]	valid_0's auc: 0.889826	valid_0's binary_logloss: 0.119626	valid_1's auc: 0.826594	valid_1's binary_logloss: 0.13808
    [34]	valid_0's auc: 0.890813	valid_0's binary_logloss: 0.119231	valid_1's auc: 0.826475	valid_1's binary_logloss: 0.13814
    [35]	valid_0's auc: 0.891526	valid_0's binary_logloss: 0.118839	valid_1's auc: 0.826365	valid_1's binary_logloss: 0.138135
    [36]	valid_0's auc: 0.892313	valid_0's binary_logloss: 0.1185	valid_1's auc: 0.825614	valid_1's binary_logloss: 0.138235
    [37]	valid_0's auc: 0.892917	valid_0's binary_logloss: 0.118143	valid_1's auc: 0.825345	valid_1's binary_logloss: 0.138329
    [38]	valid_0's auc: 0.89337	valid_0's binary_logloss: 0.117833	valid_1's auc: 0.824926	valid_1's binary_logloss: 0.138429
    [39]	valid_0's auc: 0.893935	valid_0's binary_logloss: 0.117548	valid_1's auc: 0.824655	valid_1's binary_logloss: 0.138523
    Early stopping, best iteration is:
    [9]	valid_0's auc: 0.861868	valid_0's binary_logloss: 0.134472	valid_1's auc: 0.829546	valid_1's binary_logloss: 0.141596
    [1]	valid_0's auc: 0.829442	valid_0's binary_logloss: 0.156848	valid_1's auc: 0.801853	valid_1's binary_logloss: 0.159917
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.835044	valid_0's binary_logloss: 0.151129	valid_1's auc: 0.810636	valid_1's binary_logloss: 0.155191
    [3]	valid_0's auc: 0.84106	valid_0's binary_logloss: 0.146986	valid_1's auc: 0.81428	valid_1's binary_logloss: 0.151786
    [4]	valid_0's auc: 0.849104	valid_0's binary_logloss: 0.143746	valid_1's auc: 0.822369	valid_1's binary_logloss: 0.149174
    [5]	valid_0's auc: 0.852054	valid_0's binary_logloss: 0.14111	valid_1's auc: 0.825422	valid_1's binary_logloss: 0.147108
    [6]	valid_0's auc: 0.854219	valid_0's binary_logloss: 0.138941	valid_1's auc: 0.828128	valid_1's binary_logloss: 0.145314
    [7]	valid_0's auc: 0.857758	valid_0's binary_logloss: 0.136956	valid_1's auc: 0.828178	valid_1's binary_logloss: 0.144053
    [8]	valid_0's auc: 0.859433	valid_0's binary_logloss: 0.135341	valid_1's auc: 0.828907	valid_1's binary_logloss: 0.142933
    [9]	valid_0's auc: 0.860562	valid_0's binary_logloss: 0.133965	valid_1's auc: 0.830266	valid_1's binary_logloss: 0.141913
    [10]	valid_0's auc: 0.862253	valid_0's binary_logloss: 0.132732	valid_1's auc: 0.830436	valid_1's binary_logloss: 0.141159
    [11]	valid_0's auc: 0.863863	valid_0's binary_logloss: 0.131616	valid_1's auc: 0.830888	valid_1's binary_logloss: 0.140465
    [12]	valid_0's auc: 0.86535	valid_0's binary_logloss: 0.130642	valid_1's auc: 0.828473	valid_1's binary_logloss: 0.140089
    [13]	valid_0's auc: 0.866315	valid_0's binary_logloss: 0.129707	valid_1's auc: 0.828442	valid_1's binary_logloss: 0.139673
    [14]	valid_0's auc: 0.868091	valid_0's binary_logloss: 0.128809	valid_1's auc: 0.828803	valid_1's binary_logloss: 0.13925
    [15]	valid_0's auc: 0.869154	valid_0's binary_logloss: 0.128055	valid_1's auc: 0.829147	valid_1's binary_logloss: 0.138981
    [16]	valid_0's auc: 0.870086	valid_0's binary_logloss: 0.127376	valid_1's auc: 0.829188	valid_1's binary_logloss: 0.13878
    [17]	valid_0's auc: 0.870857	valid_0's binary_logloss: 0.126705	valid_1's auc: 0.829277	valid_1's binary_logloss: 0.138553
    [18]	valid_0's auc: 0.87202	valid_0's binary_logloss: 0.126023	valid_1's auc: 0.829614	valid_1's binary_logloss: 0.138375
    [19]	valid_0's auc: 0.873171	valid_0's binary_logloss: 0.125459	valid_1's auc: 0.830335	valid_1's binary_logloss: 0.13824
    [20]	valid_0's auc: 0.874593	valid_0's binary_logloss: 0.124876	valid_1's auc: 0.831752	valid_1's binary_logloss: 0.13798
    [21]	valid_0's auc: 0.875741	valid_0's binary_logloss: 0.124281	valid_1's auc: 0.832761	valid_1's binary_logloss: 0.137687
    [22]	valid_0's auc: 0.877448	valid_0's binary_logloss: 0.123659	valid_1's auc: 0.833679	valid_1's binary_logloss: 0.137492
    [23]	valid_0's auc: 0.878493	valid_0's binary_logloss: 0.123115	valid_1's auc: 0.833912	valid_1's binary_logloss: 0.137434
    [24]	valid_0's auc: 0.879193	valid_0's binary_logloss: 0.122682	valid_1's auc: 0.833363	valid_1's binary_logloss: 0.137364
    [25]	valid_0's auc: 0.880227	valid_0's binary_logloss: 0.122229	valid_1's auc: 0.833173	valid_1's binary_logloss: 0.137358
    [26]	valid_0's auc: 0.881585	valid_0's binary_logloss: 0.121752	valid_1's auc: 0.832904	valid_1's binary_logloss: 0.13741
    [27]	valid_0's auc: 0.882473	valid_0's binary_logloss: 0.1213	valid_1's auc: 0.832408	valid_1's binary_logloss: 0.137589
    [28]	valid_0's auc: 0.883437	valid_0's binary_logloss: 0.120929	valid_1's auc: 0.832986	valid_1's binary_logloss: 0.137512
    [29]	valid_0's auc: 0.884502	valid_0's binary_logloss: 0.120484	valid_1's auc: 0.833732	valid_1's binary_logloss: 0.137414
    [30]	valid_0's auc: 0.88539	valid_0's binary_logloss: 0.120132	valid_1's auc: 0.83355	valid_1's binary_logloss: 0.137429
    [31]	valid_0's auc: 0.886163	valid_0's binary_logloss: 0.11975	valid_1's auc: 0.833288	valid_1's binary_logloss: 0.137459
    [32]	valid_0's auc: 0.886867	valid_0's binary_logloss: 0.119396	valid_1's auc: 0.833051	valid_1's binary_logloss: 0.137503
    [33]	valid_0's auc: 0.887733	valid_0's binary_logloss: 0.119013	valid_1's auc: 0.832946	valid_1's binary_logloss: 0.137467
    [34]	valid_0's auc: 0.888681	valid_0's binary_logloss: 0.118651	valid_1's auc: 0.833519	valid_1's binary_logloss: 0.137391
    [35]	valid_0's auc: 0.889275	valid_0's binary_logloss: 0.118341	valid_1's auc: 0.833734	valid_1's binary_logloss: 0.137371
    [36]	valid_0's auc: 0.890084	valid_0's binary_logloss: 0.117965	valid_1's auc: 0.833231	valid_1's binary_logloss: 0.137505
    [37]	valid_0's auc: 0.890726	valid_0's binary_logloss: 0.11765	valid_1's auc: 0.833136	valid_1's binary_logloss: 0.13761
    [38]	valid_0's auc: 0.891158	valid_0's binary_logloss: 0.11734	valid_1's auc: 0.832938	valid_1's binary_logloss: 0.137668
    [39]	valid_0's auc: 0.89196	valid_0's binary_logloss: 0.117006	valid_1's auc: 0.832307	valid_1's binary_logloss: 0.13782
    [40]	valid_0's auc: 0.892449	valid_0's binary_logloss: 0.116703	valid_1's auc: 0.832354	valid_1's binary_logloss: 0.137858
    [41]	valid_0's auc: 0.892978	valid_0's binary_logloss: 0.116414	valid_1's auc: 0.831731	valid_1's binary_logloss: 0.137998
    [42]	valid_0's auc: 0.893542	valid_0's binary_logloss: 0.116071	valid_1's auc: 0.830942	valid_1's binary_logloss: 0.138189
    [43]	valid_0's auc: 0.894078	valid_0's binary_logloss: 0.115805	valid_1's auc: 0.830997	valid_1's binary_logloss: 0.138182
    [44]	valid_0's auc: 0.894576	valid_0's binary_logloss: 0.115516	valid_1's auc: 0.830968	valid_1's binary_logloss: 0.138257
    [45]	valid_0's auc: 0.895099	valid_0's binary_logloss: 0.115236	valid_1's auc: 0.830506	valid_1's binary_logloss: 0.13839
    [46]	valid_0's auc: 0.895608	valid_0's binary_logloss: 0.114971	valid_1's auc: 0.829892	valid_1's binary_logloss: 0.138556
    [47]	valid_0's auc: 0.895999	valid_0's binary_logloss: 0.1147	valid_1's auc: 0.829591	valid_1's binary_logloss: 0.138649
    [48]	valid_0's auc: 0.89641	valid_0's binary_logloss: 0.11442	valid_1's auc: 0.829768	valid_1's binary_logloss: 0.138698
    [49]	valid_0's auc: 0.896742	valid_0's binary_logloss: 0.114182	valid_1's auc: 0.830072	valid_1's binary_logloss: 0.138692
    [50]	valid_0's auc: 0.896929	valid_0's binary_logloss: 0.113989	valid_1's auc: 0.829948	valid_1's binary_logloss: 0.138769
    [51]	valid_0's auc: 0.897717	valid_0's binary_logloss: 0.113718	valid_1's auc: 0.830297	valid_1's binary_logloss: 0.138742
    [52]	valid_0's auc: 0.898093	valid_0's binary_logloss: 0.113472	valid_1's auc: 0.830332	valid_1's binary_logloss: 0.138776
    [53]	valid_0's auc: 0.898387	valid_0's binary_logloss: 0.113271	valid_1's auc: 0.830278	valid_1's binary_logloss: 0.138833
    Early stopping, best iteration is:
    [23]	valid_0's auc: 0.878493	valid_0's binary_logloss: 0.123115	valid_1's auc: 0.833912	valid_1's binary_logloss: 0.137434
    [1]	valid_0's auc: 0.834008	valid_0's binary_logloss: 0.156932	valid_1's auc: 0.806689	valid_1's binary_logloss: 0.159986
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.841271	valid_0's binary_logloss: 0.151528	valid_1's auc: 0.816568	valid_1's binary_logloss: 0.155337
    [3]	valid_0's auc: 0.843827	valid_0's binary_logloss: 0.147378	valid_1's auc: 0.818842	valid_1's binary_logloss: 0.151989
    [4]	valid_0's auc: 0.850477	valid_0's binary_logloss: 0.144112	valid_1's auc: 0.824908	valid_1's binary_logloss: 0.149225
    [5]	valid_0's auc: 0.853808	valid_0's binary_logloss: 0.141422	valid_1's auc: 0.826744	valid_1's binary_logloss: 0.147175
    [6]	valid_0's auc: 0.856663	valid_0's binary_logloss: 0.139222	valid_1's auc: 0.828432	valid_1's binary_logloss: 0.145499
    [7]	valid_0's auc: 0.859664	valid_0's binary_logloss: 0.137318	valid_1's auc: 0.829523	valid_1's binary_logloss: 0.14402
    [8]	valid_0's auc: 0.86186	valid_0's binary_logloss: 0.135665	valid_1's auc: 0.8314	valid_1's binary_logloss: 0.142812
    [9]	valid_0's auc: 0.863104	valid_0's binary_logloss: 0.134357	valid_1's auc: 0.831976	valid_1's binary_logloss: 0.141915
    [10]	valid_0's auc: 0.865027	valid_0's binary_logloss: 0.133117	valid_1's auc: 0.83369	valid_1's binary_logloss: 0.141045
    [11]	valid_0's auc: 0.866437	valid_0's binary_logloss: 0.132026	valid_1's auc: 0.834219	valid_1's binary_logloss: 0.140347
    [12]	valid_0's auc: 0.868039	valid_0's binary_logloss: 0.131061	valid_1's auc: 0.834388	valid_1's binary_logloss: 0.139858
    [13]	valid_0's auc: 0.869537	valid_0's binary_logloss: 0.130159	valid_1's auc: 0.835402	valid_1's binary_logloss: 0.139334
    [14]	valid_0's auc: 0.870971	valid_0's binary_logloss: 0.12932	valid_1's auc: 0.835583	valid_1's binary_logloss: 0.138962
    [15]	valid_0's auc: 0.872381	valid_0's binary_logloss: 0.128543	valid_1's auc: 0.836121	valid_1's binary_logloss: 0.138579
    [16]	valid_0's auc: 0.873871	valid_0's binary_logloss: 0.127833	valid_1's auc: 0.836592	valid_1's binary_logloss: 0.13832
    [17]	valid_0's auc: 0.87491	valid_0's binary_logloss: 0.127212	valid_1's auc: 0.836031	valid_1's binary_logloss: 0.138138
    [18]	valid_0's auc: 0.875677	valid_0's binary_logloss: 0.126616	valid_1's auc: 0.835779	valid_1's binary_logloss: 0.137944
    [19]	valid_0's auc: 0.876894	valid_0's binary_logloss: 0.126018	valid_1's auc: 0.836666	valid_1's binary_logloss: 0.137657
    [20]	valid_0's auc: 0.877834	valid_0's binary_logloss: 0.125465	valid_1's auc: 0.836513	valid_1's binary_logloss: 0.137603
    [21]	valid_0's auc: 0.878861	valid_0's binary_logloss: 0.124918	valid_1's auc: 0.836696	valid_1's binary_logloss: 0.137441
    [22]	valid_0's auc: 0.880257	valid_0's binary_logloss: 0.124391	valid_1's auc: 0.836449	valid_1's binary_logloss: 0.137422
    [23]	valid_0's auc: 0.8818	valid_0's binary_logloss: 0.123782	valid_1's auc: 0.836709	valid_1's binary_logloss: 0.137292
    [24]	valid_0's auc: 0.882923	valid_0's binary_logloss: 0.123257	valid_1's auc: 0.836765	valid_1's binary_logloss: 0.137199
    [25]	valid_0's auc: 0.883924	valid_0's binary_logloss: 0.122788	valid_1's auc: 0.836654	valid_1's binary_logloss: 0.137164
    [26]	valid_0's auc: 0.885046	valid_0's binary_logloss: 0.12229	valid_1's auc: 0.836564	valid_1's binary_logloss: 0.137112
    [27]	valid_0's auc: 0.886069	valid_0's binary_logloss: 0.121841	valid_1's auc: 0.836535	valid_1's binary_logloss: 0.137076
    [28]	valid_0's auc: 0.886927	valid_0's binary_logloss: 0.121432	valid_1's auc: 0.836263	valid_1's binary_logloss: 0.137134
    [29]	valid_0's auc: 0.887853	valid_0's binary_logloss: 0.121034	valid_1's auc: 0.836202	valid_1's binary_logloss: 0.137072
    [30]	valid_0's auc: 0.888728	valid_0's binary_logloss: 0.120631	valid_1's auc: 0.836619	valid_1's binary_logloss: 0.136982
    [31]	valid_0's auc: 0.889701	valid_0's binary_logloss: 0.120172	valid_1's auc: 0.836517	valid_1's binary_logloss: 0.136934
    [32]	valid_0's auc: 0.890993	valid_0's binary_logloss: 0.119756	valid_1's auc: 0.836361	valid_1's binary_logloss: 0.136931
    [33]	valid_0's auc: 0.892053	valid_0's binary_logloss: 0.119407	valid_1's auc: 0.836583	valid_1's binary_logloss: 0.136846
    [34]	valid_0's auc: 0.892821	valid_0's binary_logloss: 0.119049	valid_1's auc: 0.836327	valid_1's binary_logloss: 0.136961
    [35]	valid_0's auc: 0.893663	valid_0's binary_logloss: 0.118651	valid_1's auc: 0.836579	valid_1's binary_logloss: 0.136912
    [36]	valid_0's auc: 0.894523	valid_0's binary_logloss: 0.11827	valid_1's auc: 0.836209	valid_1's binary_logloss: 0.136964
    [37]	valid_0's auc: 0.895475	valid_0's binary_logloss: 0.117947	valid_1's auc: 0.835968	valid_1's binary_logloss: 0.13697
    [38]	valid_0's auc: 0.895896	valid_0's binary_logloss: 0.117661	valid_1's auc: 0.836021	valid_1's binary_logloss: 0.136984
    [39]	valid_0's auc: 0.896908	valid_0's binary_logloss: 0.117297	valid_1's auc: 0.835969	valid_1's binary_logloss: 0.137061
    [40]	valid_0's auc: 0.897428	valid_0's binary_logloss: 0.116993	valid_1's auc: 0.835781	valid_1's binary_logloss: 0.137135
    [41]	valid_0's auc: 0.89802	valid_0's binary_logloss: 0.116616	valid_1's auc: 0.834706	valid_1's binary_logloss: 0.137377
    [42]	valid_0's auc: 0.898553	valid_0's binary_logloss: 0.116307	valid_1's auc: 0.834495	valid_1's binary_logloss: 0.137447
    [43]	valid_0's auc: 0.898979	valid_0's binary_logloss: 0.11602	valid_1's auc: 0.834078	valid_1's binary_logloss: 0.137573
    [44]	valid_0's auc: 0.899439	valid_0's binary_logloss: 0.115752	valid_1's auc: 0.833605	valid_1's binary_logloss: 0.137706
    [45]	valid_0's auc: 0.899915	valid_0's binary_logloss: 0.115497	valid_1's auc: 0.833114	valid_1's binary_logloss: 0.137835
    [46]	valid_0's auc: 0.90021	valid_0's binary_logloss: 0.115245	valid_1's auc: 0.833104	valid_1's binary_logloss: 0.137879
    [47]	valid_0's auc: 0.90064	valid_0's binary_logloss: 0.114971	valid_1's auc: 0.83304	valid_1's binary_logloss: 0.137916
    [48]	valid_0's auc: 0.901303	valid_0's binary_logloss: 0.114665	valid_1's auc: 0.833281	valid_1's binary_logloss: 0.137925
    [49]	valid_0's auc: 0.901644	valid_0's binary_logloss: 0.114372	valid_1's auc: 0.83306	valid_1's binary_logloss: 0.138017
    [50]	valid_0's auc: 0.902036	valid_0's binary_logloss: 0.114104	valid_1's auc: 0.832675	valid_1's binary_logloss: 0.13811
    [51]	valid_0's auc: 0.902469	valid_0's binary_logloss: 0.113883	valid_1's auc: 0.832339	valid_1's binary_logloss: 0.138195
    [52]	valid_0's auc: 0.902667	valid_0's binary_logloss: 0.113693	valid_1's auc: 0.831957	valid_1's binary_logloss: 0.138298
    [53]	valid_0's auc: 0.902822	valid_0's binary_logloss: 0.113485	valid_1's auc: 0.831693	valid_1's binary_logloss: 0.138383
    [54]	valid_0's auc: 0.902977	valid_0's binary_logloss: 0.113301	valid_1's auc: 0.830964	valid_1's binary_logloss: 0.138566
    Early stopping, best iteration is:
    [24]	valid_0's auc: 0.882923	valid_0's binary_logloss: 0.123257	valid_1's auc: 0.836765	valid_1's binary_logloss: 0.137199
    [1]	valid_0's auc: 0.832048	valid_0's binary_logloss: 0.157225	valid_1's auc: 0.811916	valid_1's binary_logloss: 0.159885
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.843711	valid_0's binary_logloss: 0.151535	valid_1's auc: 0.821654	valid_1's binary_logloss: 0.154778
    [3]	valid_0's auc: 0.846336	valid_0's binary_logloss: 0.14743	valid_1's auc: 0.823358	valid_1's binary_logloss: 0.151371
    [4]	valid_0's auc: 0.849091	valid_0's binary_logloss: 0.144223	valid_1's auc: 0.826879	valid_1's binary_logloss: 0.148584
    [5]	valid_0's auc: 0.851041	valid_0's binary_logloss: 0.141653	valid_1's auc: 0.825653	valid_1's binary_logloss: 0.14656
    [6]	valid_0's auc: 0.853453	valid_0's binary_logloss: 0.139394	valid_1's auc: 0.828113	valid_1's binary_logloss: 0.144822
    [7]	valid_0's auc: 0.857868	valid_0's binary_logloss: 0.137439	valid_1's auc: 0.828786	valid_1's binary_logloss: 0.143565
    [8]	valid_0's auc: 0.860559	valid_0's binary_logloss: 0.135866	valid_1's auc: 0.829347	valid_1's binary_logloss: 0.142504
    [9]	valid_0's auc: 0.861868	valid_0's binary_logloss: 0.134472	valid_1's auc: 0.829546	valid_1's binary_logloss: 0.141596
    [10]	valid_0's auc: 0.862998	valid_0's binary_logloss: 0.133271	valid_1's auc: 0.828894	valid_1's binary_logloss: 0.140928
    [11]	valid_0's auc: 0.864761	valid_0's binary_logloss: 0.132122	valid_1's auc: 0.828717	valid_1's binary_logloss: 0.140351
    [12]	valid_0's auc: 0.865861	valid_0's binary_logloss: 0.131127	valid_1's auc: 0.827893	valid_1's binary_logloss: 0.139897
    [13]	valid_0's auc: 0.867377	valid_0's binary_logloss: 0.130201	valid_1's auc: 0.828411	valid_1's binary_logloss: 0.139452
    [14]	valid_0's auc: 0.868645	valid_0's binary_logloss: 0.129388	valid_1's auc: 0.828859	valid_1's binary_logloss: 0.139104
    [15]	valid_0's auc: 0.869811	valid_0's binary_logloss: 0.12864	valid_1's auc: 0.827872	valid_1's binary_logloss: 0.138919
    [16]	valid_0's auc: 0.870761	valid_0's binary_logloss: 0.127902	valid_1's auc: 0.827629	valid_1's binary_logloss: 0.138614
    [17]	valid_0's auc: 0.872392	valid_0's binary_logloss: 0.127247	valid_1's auc: 0.828475	valid_1's binary_logloss: 0.138424
    [18]	valid_0's auc: 0.873984	valid_0's binary_logloss: 0.126643	valid_1's auc: 0.828038	valid_1's binary_logloss: 0.13836
    [19]	valid_0's auc: 0.875277	valid_0's binary_logloss: 0.126018	valid_1's auc: 0.828307	valid_1's binary_logloss: 0.138229
    [20]	valid_0's auc: 0.876767	valid_0's binary_logloss: 0.125429	valid_1's auc: 0.828398	valid_1's binary_logloss: 0.138132
    [21]	valid_0's auc: 0.878127	valid_0's binary_logloss: 0.124871	valid_1's auc: 0.827975	valid_1's binary_logloss: 0.138032
    [22]	valid_0's auc: 0.879333	valid_0's binary_logloss: 0.12434	valid_1's auc: 0.828026	valid_1's binary_logloss: 0.137957
    [23]	valid_0's auc: 0.880379	valid_0's binary_logloss: 0.123879	valid_1's auc: 0.827567	valid_1's binary_logloss: 0.138018
    [24]	valid_0's auc: 0.881429	valid_0's binary_logloss: 0.123358	valid_1's auc: 0.827472	valid_1's binary_logloss: 0.137975
    [25]	valid_0's auc: 0.882335	valid_0's binary_logloss: 0.122902	valid_1's auc: 0.827456	valid_1's binary_logloss: 0.137997
    [26]	valid_0's auc: 0.883467	valid_0's binary_logloss: 0.122419	valid_1's auc: 0.827539	valid_1's binary_logloss: 0.137977
    [27]	valid_0's auc: 0.884573	valid_0's binary_logloss: 0.121954	valid_1's auc: 0.827608	valid_1's binary_logloss: 0.137962
    [28]	valid_0's auc: 0.885354	valid_0's binary_logloss: 0.12151	valid_1's auc: 0.827828	valid_1's binary_logloss: 0.137979
    [29]	valid_0's auc: 0.88633	valid_0's binary_logloss: 0.1211	valid_1's auc: 0.827743	valid_1's binary_logloss: 0.137916
    [30]	valid_0's auc: 0.887129	valid_0's binary_logloss: 0.120718	valid_1's auc: 0.82774	valid_1's binary_logloss: 0.13796
    [31]	valid_0's auc: 0.888235	valid_0's binary_logloss: 0.120283	valid_1's auc: 0.827163	valid_1's binary_logloss: 0.137963
    [32]	valid_0's auc: 0.889179	valid_0's binary_logloss: 0.119986	valid_1's auc: 0.826867	valid_1's binary_logloss: 0.13798
    [33]	valid_0's auc: 0.889826	valid_0's binary_logloss: 0.119626	valid_1's auc: 0.826594	valid_1's binary_logloss: 0.13808
    [34]	valid_0's auc: 0.890813	valid_0's binary_logloss: 0.119231	valid_1's auc: 0.826475	valid_1's binary_logloss: 0.13814
    [35]	valid_0's auc: 0.891526	valid_0's binary_logloss: 0.118839	valid_1's auc: 0.826365	valid_1's binary_logloss: 0.138135
    [36]	valid_0's auc: 0.892313	valid_0's binary_logloss: 0.1185	valid_1's auc: 0.825614	valid_1's binary_logloss: 0.138235
    [37]	valid_0's auc: 0.892917	valid_0's binary_logloss: 0.118143	valid_1's auc: 0.825345	valid_1's binary_logloss: 0.138329
    [38]	valid_0's auc: 0.89337	valid_0's binary_logloss: 0.117833	valid_1's auc: 0.824926	valid_1's binary_logloss: 0.138429
    [39]	valid_0's auc: 0.893935	valid_0's binary_logloss: 0.117548	valid_1's auc: 0.824655	valid_1's binary_logloss: 0.138523
    Early stopping, best iteration is:
    [9]	valid_0's auc: 0.861868	valid_0's binary_logloss: 0.134472	valid_1's auc: 0.829546	valid_1's binary_logloss: 0.141596
    [1]	valid_0's auc: 0.822135	valid_0's binary_logloss: 0.157275	valid_1's auc: 0.80763	valid_1's binary_logloss: 0.159959
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.826629	valid_0's binary_logloss: 0.15186	valid_1's auc: 0.811377	valid_1's binary_logloss: 0.155197
    [3]	valid_0's auc: 0.830813	valid_0's binary_logloss: 0.147949	valid_1's auc: 0.813917	valid_1's binary_logloss: 0.151946
    [4]	valid_0's auc: 0.833387	valid_0's binary_logloss: 0.144809	valid_1's auc: 0.814641	valid_1's binary_logloss: 0.149358
    [5]	valid_0's auc: 0.838116	valid_0's binary_logloss: 0.142332	valid_1's auc: 0.816965	valid_1's binary_logloss: 0.147361
    [6]	valid_0's auc: 0.841055	valid_0's binary_logloss: 0.140267	valid_1's auc: 0.81978	valid_1's binary_logloss: 0.145867
    [7]	valid_0's auc: 0.842021	valid_0's binary_logloss: 0.138495	valid_1's auc: 0.819689	valid_1's binary_logloss: 0.144575
    [8]	valid_0's auc: 0.843828	valid_0's binary_logloss: 0.137029	valid_1's auc: 0.822496	valid_1's binary_logloss: 0.143422
    [9]	valid_0's auc: 0.845522	valid_0's binary_logloss: 0.135783	valid_1's auc: 0.823288	valid_1's binary_logloss: 0.142428
    [10]	valid_0's auc: 0.847654	valid_0's binary_logloss: 0.134654	valid_1's auc: 0.824658	valid_1's binary_logloss: 0.141712
    [11]	valid_0's auc: 0.849302	valid_0's binary_logloss: 0.133679	valid_1's auc: 0.826788	valid_1's binary_logloss: 0.140983
    [12]	valid_0's auc: 0.850949	valid_0's binary_logloss: 0.132827	valid_1's auc: 0.828316	valid_1's binary_logloss: 0.140403
    [13]	valid_0's auc: 0.852852	valid_0's binary_logloss: 0.132062	valid_1's auc: 0.830745	valid_1's binary_logloss: 0.139849
    [14]	valid_0's auc: 0.854299	valid_0's binary_logloss: 0.131363	valid_1's auc: 0.832741	valid_1's binary_logloss: 0.139383
    [15]	valid_0's auc: 0.855376	valid_0's binary_logloss: 0.13071	valid_1's auc: 0.833869	valid_1's binary_logloss: 0.138963
    [16]	valid_0's auc: 0.856265	valid_0's binary_logloss: 0.130166	valid_1's auc: 0.833925	valid_1's binary_logloss: 0.138745
    [17]	valid_0's auc: 0.85735	valid_0's binary_logloss: 0.129615	valid_1's auc: 0.834227	valid_1's binary_logloss: 0.138474
    [18]	valid_0's auc: 0.859279	valid_0's binary_logloss: 0.129153	valid_1's auc: 0.834146	valid_1's binary_logloss: 0.138306
    [19]	valid_0's auc: 0.860099	valid_0's binary_logloss: 0.128684	valid_1's auc: 0.833391	valid_1's binary_logloss: 0.138142
    [20]	valid_0's auc: 0.861239	valid_0's binary_logloss: 0.128272	valid_1's auc: 0.833651	valid_1's binary_logloss: 0.13795
    [21]	valid_0's auc: 0.862153	valid_0's binary_logloss: 0.127883	valid_1's auc: 0.834662	valid_1's binary_logloss: 0.137829
    [22]	valid_0's auc: 0.863466	valid_0's binary_logloss: 0.127439	valid_1's auc: 0.834505	valid_1's binary_logloss: 0.137762
    [23]	valid_0's auc: 0.86379	valid_0's binary_logloss: 0.127135	valid_1's auc: 0.834377	valid_1's binary_logloss: 0.137657
    [24]	valid_0's auc: 0.864548	valid_0's binary_logloss: 0.126788	valid_1's auc: 0.834289	valid_1's binary_logloss: 0.137499
    [25]	valid_0's auc: 0.865681	valid_0's binary_logloss: 0.126451	valid_1's auc: 0.834899	valid_1's binary_logloss: 0.137329
    [26]	valid_0's auc: 0.866659	valid_0's binary_logloss: 0.126121	valid_1's auc: 0.835946	valid_1's binary_logloss: 0.137177
    [27]	valid_0's auc: 0.867505	valid_0's binary_logloss: 0.125853	valid_1's auc: 0.836616	valid_1's binary_logloss: 0.137016
    [28]	valid_0's auc: 0.868089	valid_0's binary_logloss: 0.125584	valid_1's auc: 0.836565	valid_1's binary_logloss: 0.137
    [29]	valid_0's auc: 0.868709	valid_0's binary_logloss: 0.125327	valid_1's auc: 0.837028	valid_1's binary_logloss: 0.136878
    [30]	valid_0's auc: 0.87006	valid_0's binary_logloss: 0.12501	valid_1's auc: 0.837512	valid_1's binary_logloss: 0.136792
    [31]	valid_0's auc: 0.870612	valid_0's binary_logloss: 0.124704	valid_1's auc: 0.837541	valid_1's binary_logloss: 0.136757
    [32]	valid_0's auc: 0.871234	valid_0's binary_logloss: 0.124431	valid_1's auc: 0.837141	valid_1's binary_logloss: 0.136755
    [33]	valid_0's auc: 0.872027	valid_0's binary_logloss: 0.124167	valid_1's auc: 0.837258	valid_1's binary_logloss: 0.136728
    [34]	valid_0's auc: 0.87244	valid_0's binary_logloss: 0.123956	valid_1's auc: 0.836829	valid_1's binary_logloss: 0.136799
    [35]	valid_0's auc: 0.873259	valid_0's binary_logloss: 0.123713	valid_1's auc: 0.836871	valid_1's binary_logloss: 0.136775
    [36]	valid_0's auc: 0.873695	valid_0's binary_logloss: 0.123497	valid_1's auc: 0.836512	valid_1's binary_logloss: 0.136859
    [37]	valid_0's auc: 0.874365	valid_0's binary_logloss: 0.123259	valid_1's auc: 0.836421	valid_1's binary_logloss: 0.136872
    [38]	valid_0's auc: 0.874998	valid_0's binary_logloss: 0.123014	valid_1's auc: 0.83628	valid_1's binary_logloss: 0.136939
    [39]	valid_0's auc: 0.875666	valid_0's binary_logloss: 0.122811	valid_1's auc: 0.836577	valid_1's binary_logloss: 0.13689
    [40]	valid_0's auc: 0.875931	valid_0's binary_logloss: 0.122638	valid_1's auc: 0.836148	valid_1's binary_logloss: 0.13697
    [41]	valid_0's auc: 0.876403	valid_0's binary_logloss: 0.122474	valid_1's auc: 0.836008	valid_1's binary_logloss: 0.137005
    [42]	valid_0's auc: 0.87697	valid_0's binary_logloss: 0.122306	valid_1's auc: 0.835814	valid_1's binary_logloss: 0.137031
    [43]	valid_0's auc: 0.877305	valid_0's binary_logloss: 0.122115	valid_1's auc: 0.835467	valid_1's binary_logloss: 0.137112
    [44]	valid_0's auc: 0.87782	valid_0's binary_logloss: 0.121932	valid_1's auc: 0.835339	valid_1's binary_logloss: 0.137217
    [45]	valid_0's auc: 0.878559	valid_0's binary_logloss: 0.121701	valid_1's auc: 0.83506	valid_1's binary_logloss: 0.137256
    [46]	valid_0's auc: 0.878895	valid_0's binary_logloss: 0.121564	valid_1's auc: 0.834955	valid_1's binary_logloss: 0.137255
    [47]	valid_0's auc: 0.879643	valid_0's binary_logloss: 0.121306	valid_1's auc: 0.834987	valid_1's binary_logloss: 0.137275
    [48]	valid_0's auc: 0.88002	valid_0's binary_logloss: 0.121119	valid_1's auc: 0.834864	valid_1's binary_logloss: 0.137329
    [49]	valid_0's auc: 0.880318	valid_0's binary_logloss: 0.120965	valid_1's auc: 0.834547	valid_1's binary_logloss: 0.137394
    [50]	valid_0's auc: 0.880967	valid_0's binary_logloss: 0.12082	valid_1's auc: 0.834588	valid_1's binary_logloss: 0.137388
    [51]	valid_0's auc: 0.8813	valid_0's binary_logloss: 0.120647	valid_1's auc: 0.834288	valid_1's binary_logloss: 0.137467
    [52]	valid_0's auc: 0.881706	valid_0's binary_logloss: 0.120482	valid_1's auc: 0.834305	valid_1's binary_logloss: 0.137491
    [53]	valid_0's auc: 0.882192	valid_0's binary_logloss: 0.1203	valid_1's auc: 0.834662	valid_1's binary_logloss: 0.137492
    [54]	valid_0's auc: 0.882307	valid_0's binary_logloss: 0.120188	valid_1's auc: 0.834241	valid_1's binary_logloss: 0.13763
    [55]	valid_0's auc: 0.883062	valid_0's binary_logloss: 0.119929	valid_1's auc: 0.834145	valid_1's binary_logloss: 0.137674
    [56]	valid_0's auc: 0.883272	valid_0's binary_logloss: 0.119823	valid_1's auc: 0.833884	valid_1's binary_logloss: 0.137762
    [57]	valid_0's auc: 0.883628	valid_0's binary_logloss: 0.11968	valid_1's auc: 0.833563	valid_1's binary_logloss: 0.13782
    [58]	valid_0's auc: 0.884097	valid_0's binary_logloss: 0.119546	valid_1's auc: 0.833647	valid_1's binary_logloss: 0.137781
    [59]	valid_0's auc: 0.884337	valid_0's binary_logloss: 0.119387	valid_1's auc: 0.833138	valid_1's binary_logloss: 0.137878
    [60]	valid_0's auc: 0.884777	valid_0's binary_logloss: 0.119188	valid_1's auc: 0.832909	valid_1's binary_logloss: 0.13796
    [61]	valid_0's auc: 0.885076	valid_0's binary_logloss: 0.119054	valid_1's auc: 0.833025	valid_1's binary_logloss: 0.137959
    Early stopping, best iteration is:
    [31]	valid_0's auc: 0.870612	valid_0's binary_logloss: 0.124704	valid_1's auc: 0.837541	valid_1's binary_logloss: 0.136757
    [1]	valid_0's auc: 0.825036	valid_0's binary_logloss: 0.157424	valid_1's auc: 0.807618	valid_1's binary_logloss: 0.159921
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.829804	valid_0's binary_logloss: 0.152132	valid_1's auc: 0.815884	valid_1's binary_logloss: 0.155124
    [3]	valid_0's auc: 0.835123	valid_0's binary_logloss: 0.148284	valid_1's auc: 0.81743	valid_1's binary_logloss: 0.151751
    [4]	valid_0's auc: 0.841579	valid_0's binary_logloss: 0.145192	valid_1's auc: 0.823237	valid_1's binary_logloss: 0.149159
    [5]	valid_0's auc: 0.843972	valid_0's binary_logloss: 0.142753	valid_1's auc: 0.82747	valid_1's binary_logloss: 0.147042
    [6]	valid_0's auc: 0.846408	valid_0's binary_logloss: 0.140687	valid_1's auc: 0.830402	valid_1's binary_logloss: 0.145398
    [7]	valid_0's auc: 0.848617	valid_0's binary_logloss: 0.138978	valid_1's auc: 0.830319	valid_1's binary_logloss: 0.144054
    [8]	valid_0's auc: 0.849973	valid_0's binary_logloss: 0.137527	valid_1's auc: 0.831414	valid_1's binary_logloss: 0.142955
    [9]	valid_0's auc: 0.852151	valid_0's binary_logloss: 0.136279	valid_1's auc: 0.834577	valid_1's binary_logloss: 0.141825
    [10]	valid_0's auc: 0.852962	valid_0's binary_logloss: 0.135231	valid_1's auc: 0.835301	valid_1's binary_logloss: 0.140947
    [11]	valid_0's auc: 0.853769	valid_0's binary_logloss: 0.134247	valid_1's auc: 0.836379	valid_1's binary_logloss: 0.140121
    [12]	valid_0's auc: 0.855739	valid_0's binary_logloss: 0.133341	valid_1's auc: 0.83615	valid_1's binary_logloss: 0.139629
    [13]	valid_0's auc: 0.857549	valid_0's binary_logloss: 0.132607	valid_1's auc: 0.836232	valid_1's binary_logloss: 0.139209
    [14]	valid_0's auc: 0.858955	valid_0's binary_logloss: 0.131867	valid_1's auc: 0.836387	valid_1's binary_logloss: 0.138827
    [15]	valid_0's auc: 0.860018	valid_0's binary_logloss: 0.13125	valid_1's auc: 0.837299	valid_1's binary_logloss: 0.138441
    [16]	valid_0's auc: 0.861059	valid_0's binary_logloss: 0.130679	valid_1's auc: 0.838197	valid_1's binary_logloss: 0.138074
    [17]	valid_0's auc: 0.862112	valid_0's binary_logloss: 0.130128	valid_1's auc: 0.838835	valid_1's binary_logloss: 0.137791
    [18]	valid_0's auc: 0.862994	valid_0's binary_logloss: 0.129646	valid_1's auc: 0.838904	valid_1's binary_logloss: 0.137577
    [19]	valid_0's auc: 0.863568	valid_0's binary_logloss: 0.129195	valid_1's auc: 0.839028	valid_1's binary_logloss: 0.137319
    [20]	valid_0's auc: 0.86472	valid_0's binary_logloss: 0.128729	valid_1's auc: 0.83984	valid_1's binary_logloss: 0.137089
    [21]	valid_0's auc: 0.866051	valid_0's binary_logloss: 0.128315	valid_1's auc: 0.840385	valid_1's binary_logloss: 0.136889
    [22]	valid_0's auc: 0.866997	valid_0's binary_logloss: 0.127908	valid_1's auc: 0.840544	valid_1's binary_logloss: 0.136724
    [23]	valid_0's auc: 0.867998	valid_0's binary_logloss: 0.127516	valid_1's auc: 0.840471	valid_1's binary_logloss: 0.136628
    [24]	valid_0's auc: 0.868835	valid_0's binary_logloss: 0.127186	valid_1's auc: 0.84011	valid_1's binary_logloss: 0.136573
    [25]	valid_0's auc: 0.869535	valid_0's binary_logloss: 0.126855	valid_1's auc: 0.83987	valid_1's binary_logloss: 0.136502
    [26]	valid_0's auc: 0.870439	valid_0's binary_logloss: 0.126515	valid_1's auc: 0.840538	valid_1's binary_logloss: 0.136387
    [27]	valid_0's auc: 0.871466	valid_0's binary_logloss: 0.126179	valid_1's auc: 0.840326	valid_1's binary_logloss: 0.136301
    [28]	valid_0's auc: 0.872022	valid_0's binary_logloss: 0.125897	valid_1's auc: 0.840415	valid_1's binary_logloss: 0.136253
    [29]	valid_0's auc: 0.873022	valid_0's binary_logloss: 0.125624	valid_1's auc: 0.840531	valid_1's binary_logloss: 0.136185
    [30]	valid_0's auc: 0.873626	valid_0's binary_logloss: 0.125329	valid_1's auc: 0.840407	valid_1's binary_logloss: 0.136188
    [31]	valid_0's auc: 0.874227	valid_0's binary_logloss: 0.12509	valid_1's auc: 0.841102	valid_1's binary_logloss: 0.136046
    [32]	valid_0's auc: 0.874828	valid_0's binary_logloss: 0.124856	valid_1's auc: 0.841125	valid_1's binary_logloss: 0.135986
    [33]	valid_0's auc: 0.875492	valid_0's binary_logloss: 0.124583	valid_1's auc: 0.841558	valid_1's binary_logloss: 0.135907
    [34]	valid_0's auc: 0.876123	valid_0's binary_logloss: 0.124343	valid_1's auc: 0.841695	valid_1's binary_logloss: 0.135865
    [35]	valid_0's auc: 0.876893	valid_0's binary_logloss: 0.124064	valid_1's auc: 0.841945	valid_1's binary_logloss: 0.135831
    [36]	valid_0's auc: 0.877421	valid_0's binary_logloss: 0.123823	valid_1's auc: 0.841949	valid_1's binary_logloss: 0.135864
    [37]	valid_0's auc: 0.877945	valid_0's binary_logloss: 0.123612	valid_1's auc: 0.842028	valid_1's binary_logloss: 0.135826
    [38]	valid_0's auc: 0.878391	valid_0's binary_logloss: 0.123437	valid_1's auc: 0.841969	valid_1's binary_logloss: 0.135815
    [39]	valid_0's auc: 0.878886	valid_0's binary_logloss: 0.123211	valid_1's auc: 0.84193	valid_1's binary_logloss: 0.135809
    [40]	valid_0's auc: 0.879487	valid_0's binary_logloss: 0.123003	valid_1's auc: 0.841846	valid_1's binary_logloss: 0.135816
    [41]	valid_0's auc: 0.880192	valid_0's binary_logloss: 0.12276	valid_1's auc: 0.841501	valid_1's binary_logloss: 0.135878
    [42]	valid_0's auc: 0.880817	valid_0's binary_logloss: 0.122555	valid_1's auc: 0.841224	valid_1's binary_logloss: 0.135894
    [43]	valid_0's auc: 0.881168	valid_0's binary_logloss: 0.122344	valid_1's auc: 0.841189	valid_1's binary_logloss: 0.135895
    [44]	valid_0's auc: 0.88156	valid_0's binary_logloss: 0.122151	valid_1's auc: 0.841136	valid_1's binary_logloss: 0.135914
    [45]	valid_0's auc: 0.882438	valid_0's binary_logloss: 0.121855	valid_1's auc: 0.840553	valid_1's binary_logloss: 0.136015
    [46]	valid_0's auc: 0.883226	valid_0's binary_logloss: 0.121617	valid_1's auc: 0.840057	valid_1's binary_logloss: 0.136084
    [47]	valid_0's auc: 0.883701	valid_0's binary_logloss: 0.121428	valid_1's auc: 0.839905	valid_1's binary_logloss: 0.13611
    [48]	valid_0's auc: 0.884405	valid_0's binary_logloss: 0.121174	valid_1's auc: 0.840003	valid_1's binary_logloss: 0.136077
    [49]	valid_0's auc: 0.8847	valid_0's binary_logloss: 0.12101	valid_1's auc: 0.839683	valid_1's binary_logloss: 0.136145
    [50]	valid_0's auc: 0.885111	valid_0's binary_logloss: 0.120814	valid_1's auc: 0.839749	valid_1's binary_logloss: 0.136099
    [51]	valid_0's auc: 0.885425	valid_0's binary_logloss: 0.12065	valid_1's auc: 0.839416	valid_1's binary_logloss: 0.13609
    [52]	valid_0's auc: 0.885826	valid_0's binary_logloss: 0.120466	valid_1's auc: 0.83916	valid_1's binary_logloss: 0.136158
    [53]	valid_0's auc: 0.88609	valid_0's binary_logloss: 0.120311	valid_1's auc: 0.839381	valid_1's binary_logloss: 0.136151
    [54]	valid_0's auc: 0.886458	valid_0's binary_logloss: 0.120122	valid_1's auc: 0.839222	valid_1's binary_logloss: 0.136197
    [55]	valid_0's auc: 0.887239	valid_0's binary_logloss: 0.11994	valid_1's auc: 0.839191	valid_1's binary_logloss: 0.136207
    [56]	valid_0's auc: 0.887885	valid_0's binary_logloss: 0.119726	valid_1's auc: 0.839099	valid_1's binary_logloss: 0.136256
    [57]	valid_0's auc: 0.888068	valid_0's binary_logloss: 0.119592	valid_1's auc: 0.83885	valid_1's binary_logloss: 0.136275
    [58]	valid_0's auc: 0.888634	valid_0's binary_logloss: 0.119455	valid_1's auc: 0.838851	valid_1's binary_logloss: 0.136269
    [59]	valid_0's auc: 0.888873	valid_0's binary_logloss: 0.119334	valid_1's auc: 0.838683	valid_1's binary_logloss: 0.13629
    [60]	valid_0's auc: 0.889172	valid_0's binary_logloss: 0.119194	valid_1's auc: 0.838562	valid_1's binary_logloss: 0.136301
    [61]	valid_0's auc: 0.889511	valid_0's binary_logloss: 0.119036	valid_1's auc: 0.838472	valid_1's binary_logloss: 0.136335
    [62]	valid_0's auc: 0.889884	valid_0's binary_logloss: 0.118877	valid_1's auc: 0.838355	valid_1's binary_logloss: 0.136343
    [63]	valid_0's auc: 0.890436	valid_0's binary_logloss: 0.118679	valid_1's auc: 0.838056	valid_1's binary_logloss: 0.136425
    [64]	valid_0's auc: 0.890662	valid_0's binary_logloss: 0.118521	valid_1's auc: 0.837425	valid_1's binary_logloss: 0.136547
    [65]	valid_0's auc: 0.890862	valid_0's binary_logloss: 0.118389	valid_1's auc: 0.837399	valid_1's binary_logloss: 0.13655
    [66]	valid_0's auc: 0.890931	valid_0's binary_logloss: 0.118288	valid_1's auc: 0.837035	valid_1's binary_logloss: 0.13665
    [67]	valid_0's auc: 0.891015	valid_0's binary_logloss: 0.118183	valid_1's auc: 0.836707	valid_1's binary_logloss: 0.136783
    Early stopping, best iteration is:
    [37]	valid_0's auc: 0.877945	valid_0's binary_logloss: 0.123612	valid_1's auc: 0.842028	valid_1's binary_logloss: 0.135826
    [1]	valid_0's auc: 0.823507	valid_0's binary_logloss: 0.157501	valid_1's auc: 0.810415	valid_1's binary_logloss: 0.160155
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.829581	valid_0's binary_logloss: 0.152083	valid_1's auc: 0.815091	valid_1's binary_logloss: 0.155417
    [3]	valid_0's auc: 0.834308	valid_0's binary_logloss: 0.148148	valid_1's auc: 0.818622	valid_1's binary_logloss: 0.152027
    [4]	valid_0's auc: 0.836859	valid_0's binary_logloss: 0.145159	valid_1's auc: 0.819436	valid_1's binary_logloss: 0.149485
    [5]	valid_0's auc: 0.841846	valid_0's binary_logloss: 0.142678	valid_1's auc: 0.823324	valid_1's binary_logloss: 0.147372
    [6]	valid_0's auc: 0.844625	valid_0's binary_logloss: 0.140636	valid_1's auc: 0.824092	valid_1's binary_logloss: 0.145715
    [7]	valid_0's auc: 0.848223	valid_0's binary_logloss: 0.138877	valid_1's auc: 0.826887	valid_1's binary_logloss: 0.144253
    [8]	valid_0's auc: 0.849635	valid_0's binary_logloss: 0.137406	valid_1's auc: 0.829164	valid_1's binary_logloss: 0.143039
    [9]	valid_0's auc: 0.850865	valid_0's binary_logloss: 0.136105	valid_1's auc: 0.828916	valid_1's binary_logloss: 0.142109
    [10]	valid_0's auc: 0.852877	valid_0's binary_logloss: 0.134962	valid_1's auc: 0.829565	valid_1's binary_logloss: 0.141319
    [11]	valid_0's auc: 0.853498	valid_0's binary_logloss: 0.134001	valid_1's auc: 0.829604	valid_1's binary_logloss: 0.140625
    [12]	valid_0's auc: 0.854677	valid_0's binary_logloss: 0.133137	valid_1's auc: 0.828978	valid_1's binary_logloss: 0.140133
    [13]	valid_0's auc: 0.856296	valid_0's binary_logloss: 0.132367	valid_1's auc: 0.830184	valid_1's binary_logloss: 0.139696
    [14]	valid_0's auc: 0.858364	valid_0's binary_logloss: 0.131592	valid_1's auc: 0.831576	valid_1's binary_logloss: 0.139209
    [15]	valid_0's auc: 0.859632	valid_0's binary_logloss: 0.130943	valid_1's auc: 0.833147	valid_1's binary_logloss: 0.138759
    [16]	valid_0's auc: 0.86077	valid_0's binary_logloss: 0.130306	valid_1's auc: 0.833392	valid_1's binary_logloss: 0.138459
    [17]	valid_0's auc: 0.861722	valid_0's binary_logloss: 0.129793	valid_1's auc: 0.834279	valid_1's binary_logloss: 0.138085
    [18]	valid_0's auc: 0.862253	valid_0's binary_logloss: 0.129343	valid_1's auc: 0.834323	valid_1's binary_logloss: 0.137866
    [19]	valid_0's auc: 0.863766	valid_0's binary_logloss: 0.128824	valid_1's auc: 0.834736	valid_1's binary_logloss: 0.13763
    [20]	valid_0's auc: 0.8643	valid_0's binary_logloss: 0.128423	valid_1's auc: 0.834528	valid_1's binary_logloss: 0.137554
    [21]	valid_0's auc: 0.865267	valid_0's binary_logloss: 0.127958	valid_1's auc: 0.834478	valid_1's binary_logloss: 0.137406
    [22]	valid_0's auc: 0.866608	valid_0's binary_logloss: 0.127506	valid_1's auc: 0.834905	valid_1's binary_logloss: 0.137271
    [23]	valid_0's auc: 0.867804	valid_0's binary_logloss: 0.127104	valid_1's auc: 0.834293	valid_1's binary_logloss: 0.137211
    [24]	valid_0's auc: 0.86846	valid_0's binary_logloss: 0.12677	valid_1's auc: 0.833842	valid_1's binary_logloss: 0.137176
    [25]	valid_0's auc: 0.869535	valid_0's binary_logloss: 0.126409	valid_1's auc: 0.834057	valid_1's binary_logloss: 0.13708
    [26]	valid_0's auc: 0.870946	valid_0's binary_logloss: 0.126026	valid_1's auc: 0.833797	valid_1's binary_logloss: 0.137021
    [27]	valid_0's auc: 0.871863	valid_0's binary_logloss: 0.12567	valid_1's auc: 0.83361	valid_1's binary_logloss: 0.13702
    [28]	valid_0's auc: 0.872503	valid_0's binary_logloss: 0.125332	valid_1's auc: 0.833415	valid_1's binary_logloss: 0.136948
    [29]	valid_0's auc: 0.873256	valid_0's binary_logloss: 0.125049	valid_1's auc: 0.83344	valid_1's binary_logloss: 0.136909
    [30]	valid_0's auc: 0.874265	valid_0's binary_logloss: 0.124767	valid_1's auc: 0.833129	valid_1's binary_logloss: 0.136924
    [31]	valid_0's auc: 0.875083	valid_0's binary_logloss: 0.124478	valid_1's auc: 0.833207	valid_1's binary_logloss: 0.136915
    [32]	valid_0's auc: 0.875789	valid_0's binary_logloss: 0.124224	valid_1's auc: 0.833079	valid_1's binary_logloss: 0.136916
    [33]	valid_0's auc: 0.876187	valid_0's binary_logloss: 0.123994	valid_1's auc: 0.833172	valid_1's binary_logloss: 0.136856
    [34]	valid_0's auc: 0.876733	valid_0's binary_logloss: 0.123732	valid_1's auc: 0.833412	valid_1's binary_logloss: 0.136808
    [35]	valid_0's auc: 0.877384	valid_0's binary_logloss: 0.123461	valid_1's auc: 0.833287	valid_1's binary_logloss: 0.136785
    [36]	valid_0's auc: 0.877802	valid_0's binary_logloss: 0.123212	valid_1's auc: 0.83302	valid_1's binary_logloss: 0.13685
    [37]	valid_0's auc: 0.878799	valid_0's binary_logloss: 0.122947	valid_1's auc: 0.832295	valid_1's binary_logloss: 0.136905
    [38]	valid_0's auc: 0.879503	valid_0's binary_logloss: 0.122719	valid_1's auc: 0.832134	valid_1's binary_logloss: 0.136919
    [39]	valid_0's auc: 0.880358	valid_0's binary_logloss: 0.122488	valid_1's auc: 0.832266	valid_1's binary_logloss: 0.136938
    [40]	valid_0's auc: 0.881013	valid_0's binary_logloss: 0.122287	valid_1's auc: 0.832133	valid_1's binary_logloss: 0.136932
    [41]	valid_0's auc: 0.881424	valid_0's binary_logloss: 0.122107	valid_1's auc: 0.832209	valid_1's binary_logloss: 0.136934
    [42]	valid_0's auc: 0.881984	valid_0's binary_logloss: 0.121921	valid_1's auc: 0.831997	valid_1's binary_logloss: 0.136976
    [43]	valid_0's auc: 0.882635	valid_0's binary_logloss: 0.121694	valid_1's auc: 0.831527	valid_1's binary_logloss: 0.137038
    [44]	valid_0's auc: 0.883243	valid_0's binary_logloss: 0.121523	valid_1's auc: 0.831593	valid_1's binary_logloss: 0.137055
    [45]	valid_0's auc: 0.883616	valid_0's binary_logloss: 0.121345	valid_1's auc: 0.831225	valid_1's binary_logloss: 0.137134
    [46]	valid_0's auc: 0.884106	valid_0's binary_logloss: 0.121155	valid_1's auc: 0.831678	valid_1's binary_logloss: 0.137064
    [47]	valid_0's auc: 0.884462	valid_0's binary_logloss: 0.120968	valid_1's auc: 0.832127	valid_1's binary_logloss: 0.137012
    [48]	valid_0's auc: 0.884804	valid_0's binary_logloss: 0.120754	valid_1's auc: 0.832474	valid_1's binary_logloss: 0.136991
    [49]	valid_0's auc: 0.885391	valid_0's binary_logloss: 0.120518	valid_1's auc: 0.832375	valid_1's binary_logloss: 0.137054
    [50]	valid_0's auc: 0.885988	valid_0's binary_logloss: 0.120287	valid_1's auc: 0.832272	valid_1's binary_logloss: 0.137065
    [51]	valid_0's auc: 0.886393	valid_0's binary_logloss: 0.120131	valid_1's auc: 0.831867	valid_1's binary_logloss: 0.137159
    [52]	valid_0's auc: 0.886879	valid_0's binary_logloss: 0.119924	valid_1's auc: 0.831971	valid_1's binary_logloss: 0.137133
    Early stopping, best iteration is:
    [22]	valid_0's auc: 0.866608	valid_0's binary_logloss: 0.127506	valid_1's auc: 0.834905	valid_1's binary_logloss: 0.137271
    [1]	valid_0's auc: 0.822135	valid_0's binary_logloss: 0.157275	valid_1's auc: 0.80763	valid_1's binary_logloss: 0.159959
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.826629	valid_0's binary_logloss: 0.15186	valid_1's auc: 0.811377	valid_1's binary_logloss: 0.155197
    [3]	valid_0's auc: 0.830813	valid_0's binary_logloss: 0.147949	valid_1's auc: 0.813917	valid_1's binary_logloss: 0.151946
    [4]	valid_0's auc: 0.833387	valid_0's binary_logloss: 0.144809	valid_1's auc: 0.814641	valid_1's binary_logloss: 0.149358
    [5]	valid_0's auc: 0.838116	valid_0's binary_logloss: 0.142332	valid_1's auc: 0.816965	valid_1's binary_logloss: 0.147361
    [6]	valid_0's auc: 0.841055	valid_0's binary_logloss: 0.140267	valid_1's auc: 0.81978	valid_1's binary_logloss: 0.145867
    [7]	valid_0's auc: 0.842021	valid_0's binary_logloss: 0.138495	valid_1's auc: 0.819689	valid_1's binary_logloss: 0.144575
    [8]	valid_0's auc: 0.843828	valid_0's binary_logloss: 0.137029	valid_1's auc: 0.822496	valid_1's binary_logloss: 0.143422
    [9]	valid_0's auc: 0.845522	valid_0's binary_logloss: 0.135783	valid_1's auc: 0.823288	valid_1's binary_logloss: 0.142428
    [10]	valid_0's auc: 0.847654	valid_0's binary_logloss: 0.134654	valid_1's auc: 0.824658	valid_1's binary_logloss: 0.141712
    [11]	valid_0's auc: 0.849302	valid_0's binary_logloss: 0.133679	valid_1's auc: 0.826788	valid_1's binary_logloss: 0.140983
    [12]	valid_0's auc: 0.850949	valid_0's binary_logloss: 0.132827	valid_1's auc: 0.828316	valid_1's binary_logloss: 0.140403
    [13]	valid_0's auc: 0.852852	valid_0's binary_logloss: 0.132062	valid_1's auc: 0.830745	valid_1's binary_logloss: 0.139849
    [14]	valid_0's auc: 0.854299	valid_0's binary_logloss: 0.131363	valid_1's auc: 0.832741	valid_1's binary_logloss: 0.139383
    [15]	valid_0's auc: 0.855376	valid_0's binary_logloss: 0.13071	valid_1's auc: 0.833869	valid_1's binary_logloss: 0.138963
    [16]	valid_0's auc: 0.856265	valid_0's binary_logloss: 0.130166	valid_1's auc: 0.833925	valid_1's binary_logloss: 0.138745
    [17]	valid_0's auc: 0.85735	valid_0's binary_logloss: 0.129615	valid_1's auc: 0.834227	valid_1's binary_logloss: 0.138474
    [18]	valid_0's auc: 0.859279	valid_0's binary_logloss: 0.129153	valid_1's auc: 0.834146	valid_1's binary_logloss: 0.138306
    [19]	valid_0's auc: 0.860099	valid_0's binary_logloss: 0.128684	valid_1's auc: 0.833391	valid_1's binary_logloss: 0.138142
    [20]	valid_0's auc: 0.861239	valid_0's binary_logloss: 0.128272	valid_1's auc: 0.833651	valid_1's binary_logloss: 0.13795
    [21]	valid_0's auc: 0.862153	valid_0's binary_logloss: 0.127883	valid_1's auc: 0.834662	valid_1's binary_logloss: 0.137829
    [22]	valid_0's auc: 0.863466	valid_0's binary_logloss: 0.127439	valid_1's auc: 0.834505	valid_1's binary_logloss: 0.137762
    [23]	valid_0's auc: 0.86379	valid_0's binary_logloss: 0.127135	valid_1's auc: 0.834377	valid_1's binary_logloss: 0.137657
    [24]	valid_0's auc: 0.864548	valid_0's binary_logloss: 0.126788	valid_1's auc: 0.834289	valid_1's binary_logloss: 0.137499
    [25]	valid_0's auc: 0.865681	valid_0's binary_logloss: 0.126451	valid_1's auc: 0.834899	valid_1's binary_logloss: 0.137329
    [26]	valid_0's auc: 0.866659	valid_0's binary_logloss: 0.126121	valid_1's auc: 0.835946	valid_1's binary_logloss: 0.137177
    [27]	valid_0's auc: 0.867505	valid_0's binary_logloss: 0.125853	valid_1's auc: 0.836616	valid_1's binary_logloss: 0.137016
    [28]	valid_0's auc: 0.868089	valid_0's binary_logloss: 0.125584	valid_1's auc: 0.836565	valid_1's binary_logloss: 0.137
    [29]	valid_0's auc: 0.868709	valid_0's binary_logloss: 0.125327	valid_1's auc: 0.837028	valid_1's binary_logloss: 0.136878
    [30]	valid_0's auc: 0.87006	valid_0's binary_logloss: 0.12501	valid_1's auc: 0.837512	valid_1's binary_logloss: 0.136792
    [31]	valid_0's auc: 0.870612	valid_0's binary_logloss: 0.124704	valid_1's auc: 0.837541	valid_1's binary_logloss: 0.136757
    [32]	valid_0's auc: 0.871234	valid_0's binary_logloss: 0.124431	valid_1's auc: 0.837141	valid_1's binary_logloss: 0.136755
    [33]	valid_0's auc: 0.872027	valid_0's binary_logloss: 0.124167	valid_1's auc: 0.837258	valid_1's binary_logloss: 0.136728
    [34]	valid_0's auc: 0.87244	valid_0's binary_logloss: 0.123956	valid_1's auc: 0.836829	valid_1's binary_logloss: 0.136799
    [35]	valid_0's auc: 0.873259	valid_0's binary_logloss: 0.123713	valid_1's auc: 0.836871	valid_1's binary_logloss: 0.136775
    [36]	valid_0's auc: 0.873695	valid_0's binary_logloss: 0.123497	valid_1's auc: 0.836512	valid_1's binary_logloss: 0.136859
    [37]	valid_0's auc: 0.874365	valid_0's binary_logloss: 0.123259	valid_1's auc: 0.836421	valid_1's binary_logloss: 0.136872
    [38]	valid_0's auc: 0.874998	valid_0's binary_logloss: 0.123014	valid_1's auc: 0.83628	valid_1's binary_logloss: 0.136939
    [39]	valid_0's auc: 0.875666	valid_0's binary_logloss: 0.122811	valid_1's auc: 0.836577	valid_1's binary_logloss: 0.13689
    [40]	valid_0's auc: 0.875931	valid_0's binary_logloss: 0.122638	valid_1's auc: 0.836148	valid_1's binary_logloss: 0.13697
    [41]	valid_0's auc: 0.876403	valid_0's binary_logloss: 0.122474	valid_1's auc: 0.836008	valid_1's binary_logloss: 0.137005
    [42]	valid_0's auc: 0.87697	valid_0's binary_logloss: 0.122306	valid_1's auc: 0.835814	valid_1's binary_logloss: 0.137031
    [43]	valid_0's auc: 0.877305	valid_0's binary_logloss: 0.122115	valid_1's auc: 0.835467	valid_1's binary_logloss: 0.137112
    [44]	valid_0's auc: 0.87782	valid_0's binary_logloss: 0.121932	valid_1's auc: 0.835339	valid_1's binary_logloss: 0.137217
    [45]	valid_0's auc: 0.878559	valid_0's binary_logloss: 0.121701	valid_1's auc: 0.83506	valid_1's binary_logloss: 0.137256
    [46]	valid_0's auc: 0.878895	valid_0's binary_logloss: 0.121564	valid_1's auc: 0.834955	valid_1's binary_logloss: 0.137255
    [47]	valid_0's auc: 0.879643	valid_0's binary_logloss: 0.121306	valid_1's auc: 0.834987	valid_1's binary_logloss: 0.137275
    [48]	valid_0's auc: 0.88002	valid_0's binary_logloss: 0.121119	valid_1's auc: 0.834864	valid_1's binary_logloss: 0.137329
    [49]	valid_0's auc: 0.880318	valid_0's binary_logloss: 0.120965	valid_1's auc: 0.834547	valid_1's binary_logloss: 0.137394
    [50]	valid_0's auc: 0.880967	valid_0's binary_logloss: 0.12082	valid_1's auc: 0.834588	valid_1's binary_logloss: 0.137388
    [51]	valid_0's auc: 0.8813	valid_0's binary_logloss: 0.120647	valid_1's auc: 0.834288	valid_1's binary_logloss: 0.137467
    [52]	valid_0's auc: 0.881706	valid_0's binary_logloss: 0.120482	valid_1's auc: 0.834305	valid_1's binary_logloss: 0.137491
    [53]	valid_0's auc: 0.882192	valid_0's binary_logloss: 0.1203	valid_1's auc: 0.834662	valid_1's binary_logloss: 0.137492
    [54]	valid_0's auc: 0.882307	valid_0's binary_logloss: 0.120188	valid_1's auc: 0.834241	valid_1's binary_logloss: 0.13763
    [55]	valid_0's auc: 0.883062	valid_0's binary_logloss: 0.119929	valid_1's auc: 0.834145	valid_1's binary_logloss: 0.137674
    [56]	valid_0's auc: 0.883272	valid_0's binary_logloss: 0.119823	valid_1's auc: 0.833884	valid_1's binary_logloss: 0.137762
    [57]	valid_0's auc: 0.883628	valid_0's binary_logloss: 0.11968	valid_1's auc: 0.833563	valid_1's binary_logloss: 0.13782
    [58]	valid_0's auc: 0.884097	valid_0's binary_logloss: 0.119546	valid_1's auc: 0.833647	valid_1's binary_logloss: 0.137781
    [59]	valid_0's auc: 0.884337	valid_0's binary_logloss: 0.119387	valid_1's auc: 0.833138	valid_1's binary_logloss: 0.137878
    [60]	valid_0's auc: 0.884777	valid_0's binary_logloss: 0.119188	valid_1's auc: 0.832909	valid_1's binary_logloss: 0.13796
    [61]	valid_0's auc: 0.885076	valid_0's binary_logloss: 0.119054	valid_1's auc: 0.833025	valid_1's binary_logloss: 0.137959
    Early stopping, best iteration is:
    [31]	valid_0's auc: 0.870612	valid_0's binary_logloss: 0.124704	valid_1's auc: 0.837541	valid_1's binary_logloss: 0.136757
    [1]	valid_0's auc: 0.825036	valid_0's binary_logloss: 0.157424	valid_1's auc: 0.807618	valid_1's binary_logloss: 0.159921
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.829804	valid_0's binary_logloss: 0.152132	valid_1's auc: 0.815884	valid_1's binary_logloss: 0.155124
    [3]	valid_0's auc: 0.835123	valid_0's binary_logloss: 0.148284	valid_1's auc: 0.81743	valid_1's binary_logloss: 0.151751
    [4]	valid_0's auc: 0.841579	valid_0's binary_logloss: 0.145192	valid_1's auc: 0.823237	valid_1's binary_logloss: 0.149159
    [5]	valid_0's auc: 0.843972	valid_0's binary_logloss: 0.142753	valid_1's auc: 0.82747	valid_1's binary_logloss: 0.147042
    [6]	valid_0's auc: 0.846408	valid_0's binary_logloss: 0.140687	valid_1's auc: 0.830402	valid_1's binary_logloss: 0.145398
    [7]	valid_0's auc: 0.848617	valid_0's binary_logloss: 0.138978	valid_1's auc: 0.830319	valid_1's binary_logloss: 0.144054
    [8]	valid_0's auc: 0.849973	valid_0's binary_logloss: 0.137527	valid_1's auc: 0.831414	valid_1's binary_logloss: 0.142955
    [9]	valid_0's auc: 0.852151	valid_0's binary_logloss: 0.136279	valid_1's auc: 0.834577	valid_1's binary_logloss: 0.141825
    [10]	valid_0's auc: 0.852962	valid_0's binary_logloss: 0.135231	valid_1's auc: 0.835301	valid_1's binary_logloss: 0.140947
    [11]	valid_0's auc: 0.853769	valid_0's binary_logloss: 0.134247	valid_1's auc: 0.836379	valid_1's binary_logloss: 0.140121
    [12]	valid_0's auc: 0.855739	valid_0's binary_logloss: 0.133341	valid_1's auc: 0.83615	valid_1's binary_logloss: 0.139629
    [13]	valid_0's auc: 0.857549	valid_0's binary_logloss: 0.132607	valid_1's auc: 0.836232	valid_1's binary_logloss: 0.139209
    [14]	valid_0's auc: 0.858955	valid_0's binary_logloss: 0.131867	valid_1's auc: 0.836387	valid_1's binary_logloss: 0.138827
    [15]	valid_0's auc: 0.860018	valid_0's binary_logloss: 0.13125	valid_1's auc: 0.837299	valid_1's binary_logloss: 0.138441
    [16]	valid_0's auc: 0.861059	valid_0's binary_logloss: 0.130679	valid_1's auc: 0.838197	valid_1's binary_logloss: 0.138074
    [17]	valid_0's auc: 0.862112	valid_0's binary_logloss: 0.130128	valid_1's auc: 0.838835	valid_1's binary_logloss: 0.137791
    [18]	valid_0's auc: 0.862994	valid_0's binary_logloss: 0.129646	valid_1's auc: 0.838904	valid_1's binary_logloss: 0.137577
    [19]	valid_0's auc: 0.863568	valid_0's binary_logloss: 0.129195	valid_1's auc: 0.839028	valid_1's binary_logloss: 0.137319
    [20]	valid_0's auc: 0.86472	valid_0's binary_logloss: 0.128729	valid_1's auc: 0.83984	valid_1's binary_logloss: 0.137089
    [21]	valid_0's auc: 0.866051	valid_0's binary_logloss: 0.128315	valid_1's auc: 0.840385	valid_1's binary_logloss: 0.136889
    [22]	valid_0's auc: 0.866997	valid_0's binary_logloss: 0.127908	valid_1's auc: 0.840544	valid_1's binary_logloss: 0.136724
    [23]	valid_0's auc: 0.867998	valid_0's binary_logloss: 0.127516	valid_1's auc: 0.840471	valid_1's binary_logloss: 0.136628
    [24]	valid_0's auc: 0.868835	valid_0's binary_logloss: 0.127186	valid_1's auc: 0.84011	valid_1's binary_logloss: 0.136573
    [25]	valid_0's auc: 0.869535	valid_0's binary_logloss: 0.126855	valid_1's auc: 0.83987	valid_1's binary_logloss: 0.136502
    [26]	valid_0's auc: 0.870439	valid_0's binary_logloss: 0.126515	valid_1's auc: 0.840538	valid_1's binary_logloss: 0.136387
    [27]	valid_0's auc: 0.871466	valid_0's binary_logloss: 0.126179	valid_1's auc: 0.840326	valid_1's binary_logloss: 0.136301
    [28]	valid_0's auc: 0.872022	valid_0's binary_logloss: 0.125897	valid_1's auc: 0.840415	valid_1's binary_logloss: 0.136253
    [29]	valid_0's auc: 0.873022	valid_0's binary_logloss: 0.125624	valid_1's auc: 0.840531	valid_1's binary_logloss: 0.136185
    [30]	valid_0's auc: 0.873626	valid_0's binary_logloss: 0.125329	valid_1's auc: 0.840407	valid_1's binary_logloss: 0.136188
    [31]	valid_0's auc: 0.874227	valid_0's binary_logloss: 0.12509	valid_1's auc: 0.841102	valid_1's binary_logloss: 0.136046
    [32]	valid_0's auc: 0.874828	valid_0's binary_logloss: 0.124856	valid_1's auc: 0.841125	valid_1's binary_logloss: 0.135986
    [33]	valid_0's auc: 0.875492	valid_0's binary_logloss: 0.124583	valid_1's auc: 0.841558	valid_1's binary_logloss: 0.135907
    [34]	valid_0's auc: 0.876123	valid_0's binary_logloss: 0.124343	valid_1's auc: 0.841695	valid_1's binary_logloss: 0.135865
    [35]	valid_0's auc: 0.876893	valid_0's binary_logloss: 0.124064	valid_1's auc: 0.841945	valid_1's binary_logloss: 0.135831
    [36]	valid_0's auc: 0.877421	valid_0's binary_logloss: 0.123823	valid_1's auc: 0.841949	valid_1's binary_logloss: 0.135864
    [37]	valid_0's auc: 0.877945	valid_0's binary_logloss: 0.123612	valid_1's auc: 0.842028	valid_1's binary_logloss: 0.135826
    [38]	valid_0's auc: 0.878391	valid_0's binary_logloss: 0.123437	valid_1's auc: 0.841969	valid_1's binary_logloss: 0.135815
    [39]	valid_0's auc: 0.878886	valid_0's binary_logloss: 0.123211	valid_1's auc: 0.84193	valid_1's binary_logloss: 0.135809
    [40]	valid_0's auc: 0.879487	valid_0's binary_logloss: 0.123003	valid_1's auc: 0.841846	valid_1's binary_logloss: 0.135816
    [41]	valid_0's auc: 0.880192	valid_0's binary_logloss: 0.12276	valid_1's auc: 0.841501	valid_1's binary_logloss: 0.135878
    [42]	valid_0's auc: 0.880817	valid_0's binary_logloss: 0.122555	valid_1's auc: 0.841224	valid_1's binary_logloss: 0.135894
    [43]	valid_0's auc: 0.881168	valid_0's binary_logloss: 0.122344	valid_1's auc: 0.841189	valid_1's binary_logloss: 0.135895
    [44]	valid_0's auc: 0.88156	valid_0's binary_logloss: 0.122151	valid_1's auc: 0.841136	valid_1's binary_logloss: 0.135914
    [45]	valid_0's auc: 0.882438	valid_0's binary_logloss: 0.121855	valid_1's auc: 0.840553	valid_1's binary_logloss: 0.136015
    [46]	valid_0's auc: 0.883226	valid_0's binary_logloss: 0.121617	valid_1's auc: 0.840057	valid_1's binary_logloss: 0.136084
    [47]	valid_0's auc: 0.883701	valid_0's binary_logloss: 0.121428	valid_1's auc: 0.839905	valid_1's binary_logloss: 0.13611
    [48]	valid_0's auc: 0.884405	valid_0's binary_logloss: 0.121174	valid_1's auc: 0.840003	valid_1's binary_logloss: 0.136077
    [49]	valid_0's auc: 0.8847	valid_0's binary_logloss: 0.12101	valid_1's auc: 0.839683	valid_1's binary_logloss: 0.136145
    [50]	valid_0's auc: 0.885111	valid_0's binary_logloss: 0.120814	valid_1's auc: 0.839749	valid_1's binary_logloss: 0.136099
    [51]	valid_0's auc: 0.885425	valid_0's binary_logloss: 0.12065	valid_1's auc: 0.839416	valid_1's binary_logloss: 0.13609
    [52]	valid_0's auc: 0.885826	valid_0's binary_logloss: 0.120466	valid_1's auc: 0.83916	valid_1's binary_logloss: 0.136158
    [53]	valid_0's auc: 0.88609	valid_0's binary_logloss: 0.120311	valid_1's auc: 0.839381	valid_1's binary_logloss: 0.136151
    [54]	valid_0's auc: 0.886458	valid_0's binary_logloss: 0.120122	valid_1's auc: 0.839222	valid_1's binary_logloss: 0.136197
    [55]	valid_0's auc: 0.887239	valid_0's binary_logloss: 0.11994	valid_1's auc: 0.839191	valid_1's binary_logloss: 0.136207
    [56]	valid_0's auc: 0.887885	valid_0's binary_logloss: 0.119726	valid_1's auc: 0.839099	valid_1's binary_logloss: 0.136256
    [57]	valid_0's auc: 0.888068	valid_0's binary_logloss: 0.119592	valid_1's auc: 0.83885	valid_1's binary_logloss: 0.136275
    [58]	valid_0's auc: 0.888634	valid_0's binary_logloss: 0.119455	valid_1's auc: 0.838851	valid_1's binary_logloss: 0.136269
    [59]	valid_0's auc: 0.888873	valid_0's binary_logloss: 0.119334	valid_1's auc: 0.838683	valid_1's binary_logloss: 0.13629
    [60]	valid_0's auc: 0.889172	valid_0's binary_logloss: 0.119194	valid_1's auc: 0.838562	valid_1's binary_logloss: 0.136301
    [61]	valid_0's auc: 0.889511	valid_0's binary_logloss: 0.119036	valid_1's auc: 0.838472	valid_1's binary_logloss: 0.136335
    [62]	valid_0's auc: 0.889884	valid_0's binary_logloss: 0.118877	valid_1's auc: 0.838355	valid_1's binary_logloss: 0.136343
    [63]	valid_0's auc: 0.890436	valid_0's binary_logloss: 0.118679	valid_1's auc: 0.838056	valid_1's binary_logloss: 0.136425
    [64]	valid_0's auc: 0.890662	valid_0's binary_logloss: 0.118521	valid_1's auc: 0.837425	valid_1's binary_logloss: 0.136547
    [65]	valid_0's auc: 0.890862	valid_0's binary_logloss: 0.118389	valid_1's auc: 0.837399	valid_1's binary_logloss: 0.13655
    [66]	valid_0's auc: 0.890931	valid_0's binary_logloss: 0.118288	valid_1's auc: 0.837035	valid_1's binary_logloss: 0.13665
    [67]	valid_0's auc: 0.891015	valid_0's binary_logloss: 0.118183	valid_1's auc: 0.836707	valid_1's binary_logloss: 0.136783
    Early stopping, best iteration is:
    [37]	valid_0's auc: 0.877945	valid_0's binary_logloss: 0.123612	valid_1's auc: 0.842028	valid_1's binary_logloss: 0.135826
    [1]	valid_0's auc: 0.823507	valid_0's binary_logloss: 0.157501	valid_1's auc: 0.810415	valid_1's binary_logloss: 0.160155
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.829581	valid_0's binary_logloss: 0.152083	valid_1's auc: 0.815091	valid_1's binary_logloss: 0.155417
    [3]	valid_0's auc: 0.834308	valid_0's binary_logloss: 0.148148	valid_1's auc: 0.818622	valid_1's binary_logloss: 0.152027
    [4]	valid_0's auc: 0.836859	valid_0's binary_logloss: 0.145159	valid_1's auc: 0.819436	valid_1's binary_logloss: 0.149485
    [5]	valid_0's auc: 0.841846	valid_0's binary_logloss: 0.142678	valid_1's auc: 0.823324	valid_1's binary_logloss: 0.147372
    [6]	valid_0's auc: 0.844625	valid_0's binary_logloss: 0.140636	valid_1's auc: 0.824092	valid_1's binary_logloss: 0.145715
    [7]	valid_0's auc: 0.848223	valid_0's binary_logloss: 0.138877	valid_1's auc: 0.826887	valid_1's binary_logloss: 0.144253
    [8]	valid_0's auc: 0.849635	valid_0's binary_logloss: 0.137406	valid_1's auc: 0.829164	valid_1's binary_logloss: 0.143039
    [9]	valid_0's auc: 0.850865	valid_0's binary_logloss: 0.136105	valid_1's auc: 0.828916	valid_1's binary_logloss: 0.142109
    [10]	valid_0's auc: 0.852877	valid_0's binary_logloss: 0.134962	valid_1's auc: 0.829565	valid_1's binary_logloss: 0.141319
    [11]	valid_0's auc: 0.853498	valid_0's binary_logloss: 0.134001	valid_1's auc: 0.829604	valid_1's binary_logloss: 0.140625
    [12]	valid_0's auc: 0.854677	valid_0's binary_logloss: 0.133137	valid_1's auc: 0.828978	valid_1's binary_logloss: 0.140133
    [13]	valid_0's auc: 0.856296	valid_0's binary_logloss: 0.132367	valid_1's auc: 0.830184	valid_1's binary_logloss: 0.139696
    [14]	valid_0's auc: 0.858364	valid_0's binary_logloss: 0.131592	valid_1's auc: 0.831576	valid_1's binary_logloss: 0.139209
    [15]	valid_0's auc: 0.859632	valid_0's binary_logloss: 0.130943	valid_1's auc: 0.833147	valid_1's binary_logloss: 0.138759
    [16]	valid_0's auc: 0.86077	valid_0's binary_logloss: 0.130306	valid_1's auc: 0.833392	valid_1's binary_logloss: 0.138459
    [17]	valid_0's auc: 0.861722	valid_0's binary_logloss: 0.129793	valid_1's auc: 0.834279	valid_1's binary_logloss: 0.138085
    [18]	valid_0's auc: 0.862253	valid_0's binary_logloss: 0.129343	valid_1's auc: 0.834323	valid_1's binary_logloss: 0.137866
    [19]	valid_0's auc: 0.863766	valid_0's binary_logloss: 0.128824	valid_1's auc: 0.834736	valid_1's binary_logloss: 0.13763
    [20]	valid_0's auc: 0.8643	valid_0's binary_logloss: 0.128423	valid_1's auc: 0.834528	valid_1's binary_logloss: 0.137554
    [21]	valid_0's auc: 0.865267	valid_0's binary_logloss: 0.127958	valid_1's auc: 0.834478	valid_1's binary_logloss: 0.137406
    [22]	valid_0's auc: 0.866608	valid_0's binary_logloss: 0.127506	valid_1's auc: 0.834905	valid_1's binary_logloss: 0.137271
    [23]	valid_0's auc: 0.867804	valid_0's binary_logloss: 0.127104	valid_1's auc: 0.834293	valid_1's binary_logloss: 0.137211
    [24]	valid_0's auc: 0.86846	valid_0's binary_logloss: 0.12677	valid_1's auc: 0.833842	valid_1's binary_logloss: 0.137176
    [25]	valid_0's auc: 0.869535	valid_0's binary_logloss: 0.126409	valid_1's auc: 0.834057	valid_1's binary_logloss: 0.13708
    [26]	valid_0's auc: 0.870946	valid_0's binary_logloss: 0.126026	valid_1's auc: 0.833797	valid_1's binary_logloss: 0.137021
    [27]	valid_0's auc: 0.871863	valid_0's binary_logloss: 0.12567	valid_1's auc: 0.83361	valid_1's binary_logloss: 0.13702
    [28]	valid_0's auc: 0.872503	valid_0's binary_logloss: 0.125332	valid_1's auc: 0.833415	valid_1's binary_logloss: 0.136948
    [29]	valid_0's auc: 0.873256	valid_0's binary_logloss: 0.125049	valid_1's auc: 0.83344	valid_1's binary_logloss: 0.136909
    [30]	valid_0's auc: 0.874265	valid_0's binary_logloss: 0.124767	valid_1's auc: 0.833129	valid_1's binary_logloss: 0.136924
    [31]	valid_0's auc: 0.875083	valid_0's binary_logloss: 0.124478	valid_1's auc: 0.833207	valid_1's binary_logloss: 0.136915
    [32]	valid_0's auc: 0.875789	valid_0's binary_logloss: 0.124224	valid_1's auc: 0.833079	valid_1's binary_logloss: 0.136916
    [33]	valid_0's auc: 0.876187	valid_0's binary_logloss: 0.123994	valid_1's auc: 0.833172	valid_1's binary_logloss: 0.136856
    [34]	valid_0's auc: 0.876733	valid_0's binary_logloss: 0.123732	valid_1's auc: 0.833412	valid_1's binary_logloss: 0.136808
    [35]	valid_0's auc: 0.877384	valid_0's binary_logloss: 0.123461	valid_1's auc: 0.833287	valid_1's binary_logloss: 0.136785
    [36]	valid_0's auc: 0.877802	valid_0's binary_logloss: 0.123212	valid_1's auc: 0.83302	valid_1's binary_logloss: 0.13685
    [37]	valid_0's auc: 0.878799	valid_0's binary_logloss: 0.122947	valid_1's auc: 0.832295	valid_1's binary_logloss: 0.136905
    [38]	valid_0's auc: 0.879503	valid_0's binary_logloss: 0.122719	valid_1's auc: 0.832134	valid_1's binary_logloss: 0.136919
    [39]	valid_0's auc: 0.880358	valid_0's binary_logloss: 0.122488	valid_1's auc: 0.832266	valid_1's binary_logloss: 0.136938
    [40]	valid_0's auc: 0.881013	valid_0's binary_logloss: 0.122287	valid_1's auc: 0.832133	valid_1's binary_logloss: 0.136932
    [41]	valid_0's auc: 0.881424	valid_0's binary_logloss: 0.122107	valid_1's auc: 0.832209	valid_1's binary_logloss: 0.136934
    [42]	valid_0's auc: 0.881984	valid_0's binary_logloss: 0.121921	valid_1's auc: 0.831997	valid_1's binary_logloss: 0.136976
    [43]	valid_0's auc: 0.882635	valid_0's binary_logloss: 0.121694	valid_1's auc: 0.831527	valid_1's binary_logloss: 0.137038
    [44]	valid_0's auc: 0.883243	valid_0's binary_logloss: 0.121523	valid_1's auc: 0.831593	valid_1's binary_logloss: 0.137055
    [45]	valid_0's auc: 0.883616	valid_0's binary_logloss: 0.121345	valid_1's auc: 0.831225	valid_1's binary_logloss: 0.137134
    [46]	valid_0's auc: 0.884106	valid_0's binary_logloss: 0.121155	valid_1's auc: 0.831678	valid_1's binary_logloss: 0.137064
    [47]	valid_0's auc: 0.884462	valid_0's binary_logloss: 0.120968	valid_1's auc: 0.832127	valid_1's binary_logloss: 0.137012
    [48]	valid_0's auc: 0.884804	valid_0's binary_logloss: 0.120754	valid_1's auc: 0.832474	valid_1's binary_logloss: 0.136991
    [49]	valid_0's auc: 0.885391	valid_0's binary_logloss: 0.120518	valid_1's auc: 0.832375	valid_1's binary_logloss: 0.137054
    [50]	valid_0's auc: 0.885988	valid_0's binary_logloss: 0.120287	valid_1's auc: 0.832272	valid_1's binary_logloss: 0.137065
    [51]	valid_0's auc: 0.886393	valid_0's binary_logloss: 0.120131	valid_1's auc: 0.831867	valid_1's binary_logloss: 0.137159
    [52]	valid_0's auc: 0.886879	valid_0's binary_logloss: 0.119924	valid_1's auc: 0.831971	valid_1's binary_logloss: 0.137133
    Early stopping, best iteration is:
    [22]	valid_0's auc: 0.866608	valid_0's binary_logloss: 0.127506	valid_1's auc: 0.834905	valid_1's binary_logloss: 0.137271
    [1]	valid_0's auc: 0.825869	valid_0's binary_logloss: 0.156436	valid_1's auc: 0.803397	valid_1's binary_logloss: 0.159993
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.834474	valid_0's binary_logloss: 0.150597	valid_1's auc: 0.809972	valid_1's binary_logloss: 0.155122
    [3]	valid_0's auc: 0.840511	valid_0's binary_logloss: 0.146374	valid_1's auc: 0.819137	valid_1's binary_logloss: 0.15148
    [4]	valid_0's auc: 0.844283	valid_0's binary_logloss: 0.143027	valid_1's auc: 0.820472	valid_1's binary_logloss: 0.14904
    [5]	valid_0's auc: 0.85046	valid_0's binary_logloss: 0.140321	valid_1's auc: 0.82606	valid_1's binary_logloss: 0.146982
    [6]	valid_0's auc: 0.852881	valid_0's binary_logloss: 0.13813	valid_1's auc: 0.823869	valid_1's binary_logloss: 0.145517
    [7]	valid_0's auc: 0.857268	valid_0's binary_logloss: 0.136149	valid_1's auc: 0.827586	valid_1's binary_logloss: 0.144156
    [8]	valid_0's auc: 0.859401	valid_0's binary_logloss: 0.13452	valid_1's auc: 0.829428	valid_1's binary_logloss: 0.14303
    [9]	valid_0's auc: 0.861435	valid_0's binary_logloss: 0.133011	valid_1's auc: 0.82971	valid_1's binary_logloss: 0.142113
    [10]	valid_0's auc: 0.862282	valid_0's binary_logloss: 0.131742	valid_1's auc: 0.830527	valid_1's binary_logloss: 0.141392
    [11]	valid_0's auc: 0.86448	valid_0's binary_logloss: 0.130536	valid_1's auc: 0.83065	valid_1's binary_logloss: 0.140854
    [12]	valid_0's auc: 0.865909	valid_0's binary_logloss: 0.129495	valid_1's auc: 0.830763	valid_1's binary_logloss: 0.140425
    [13]	valid_0's auc: 0.867083	valid_0's binary_logloss: 0.128537	valid_1's auc: 0.830097	valid_1's binary_logloss: 0.140034
    [14]	valid_0's auc: 0.869164	valid_0's binary_logloss: 0.127593	valid_1's auc: 0.831242	valid_1's binary_logloss: 0.139589
    [15]	valid_0's auc: 0.869779	valid_0's binary_logloss: 0.126753	valid_1's auc: 0.830054	valid_1's binary_logloss: 0.13935
    [16]	valid_0's auc: 0.870798	valid_0's binary_logloss: 0.125998	valid_1's auc: 0.829854	valid_1's binary_logloss: 0.139147
    [17]	valid_0's auc: 0.872413	valid_0's binary_logloss: 0.125321	valid_1's auc: 0.829939	valid_1's binary_logloss: 0.138859
    [18]	valid_0's auc: 0.873656	valid_0's binary_logloss: 0.124646	valid_1's auc: 0.83033	valid_1's binary_logloss: 0.138678
    [19]	valid_0's auc: 0.874787	valid_0's binary_logloss: 0.124094	valid_1's auc: 0.829747	valid_1's binary_logloss: 0.138635
    [20]	valid_0's auc: 0.875793	valid_0's binary_logloss: 0.123547	valid_1's auc: 0.829455	valid_1's binary_logloss: 0.138492
    [21]	valid_0's auc: 0.87694	valid_0's binary_logloss: 0.122969	valid_1's auc: 0.830945	valid_1's binary_logloss: 0.13825
    [22]	valid_0's auc: 0.878632	valid_0's binary_logloss: 0.122407	valid_1's auc: 0.831483	valid_1's binary_logloss: 0.138097
    [23]	valid_0's auc: 0.8796	valid_0's binary_logloss: 0.121869	valid_1's auc: 0.831408	valid_1's binary_logloss: 0.138071
    [24]	valid_0's auc: 0.881298	valid_0's binary_logloss: 0.121363	valid_1's auc: 0.831984	valid_1's binary_logloss: 0.137912
    [25]	valid_0's auc: 0.882699	valid_0's binary_logloss: 0.120863	valid_1's auc: 0.831941	valid_1's binary_logloss: 0.137824
    [26]	valid_0's auc: 0.883799	valid_0's binary_logloss: 0.120379	valid_1's auc: 0.832897	valid_1's binary_logloss: 0.137649
    [27]	valid_0's auc: 0.884893	valid_0's binary_logloss: 0.119944	valid_1's auc: 0.832841	valid_1's binary_logloss: 0.13763
    [28]	valid_0's auc: 0.88588	valid_0's binary_logloss: 0.119545	valid_1's auc: 0.833413	valid_1's binary_logloss: 0.137495
    [29]	valid_0's auc: 0.886461	valid_0's binary_logloss: 0.11916	valid_1's auc: 0.833437	valid_1's binary_logloss: 0.137511
    [30]	valid_0's auc: 0.887592	valid_0's binary_logloss: 0.118739	valid_1's auc: 0.833429	valid_1's binary_logloss: 0.137514
    [31]	valid_0's auc: 0.888552	valid_0's binary_logloss: 0.118399	valid_1's auc: 0.834036	valid_1's binary_logloss: 0.137459
    [32]	valid_0's auc: 0.889448	valid_0's binary_logloss: 0.118019	valid_1's auc: 0.833754	valid_1's binary_logloss: 0.137498
    [33]	valid_0's auc: 0.890203	valid_0's binary_logloss: 0.117711	valid_1's auc: 0.833853	valid_1's binary_logloss: 0.137491
    [34]	valid_0's auc: 0.890868	valid_0's binary_logloss: 0.117395	valid_1's auc: 0.833648	valid_1's binary_logloss: 0.137532
    [35]	valid_0's auc: 0.891378	valid_0's binary_logloss: 0.117106	valid_1's auc: 0.833479	valid_1's binary_logloss: 0.137577
    [36]	valid_0's auc: 0.892061	valid_0's binary_logloss: 0.116741	valid_1's auc: 0.832608	valid_1's binary_logloss: 0.137799
    [37]	valid_0's auc: 0.892715	valid_0's binary_logloss: 0.116403	valid_1's auc: 0.83208	valid_1's binary_logloss: 0.137938
    [38]	valid_0's auc: 0.893146	valid_0's binary_logloss: 0.116126	valid_1's auc: 0.83196	valid_1's binary_logloss: 0.137994
    [39]	valid_0's auc: 0.893963	valid_0's binary_logloss: 0.115783	valid_1's auc: 0.831584	valid_1's binary_logloss: 0.138076
    [40]	valid_0's auc: 0.894274	valid_0's binary_logloss: 0.115535	valid_1's auc: 0.831346	valid_1's binary_logloss: 0.13818
    [41]	valid_0's auc: 0.894809	valid_0's binary_logloss: 0.115247	valid_1's auc: 0.831197	valid_1's binary_logloss: 0.138289
    [42]	valid_0's auc: 0.895332	valid_0's binary_logloss: 0.114916	valid_1's auc: 0.830893	valid_1's binary_logloss: 0.138343
    [43]	valid_0's auc: 0.895864	valid_0's binary_logloss: 0.114701	valid_1's auc: 0.83129	valid_1's binary_logloss: 0.138307
    [44]	valid_0's auc: 0.89623	valid_0's binary_logloss: 0.114411	valid_1's auc: 0.831004	valid_1's binary_logloss: 0.138408
    [45]	valid_0's auc: 0.896616	valid_0's binary_logloss: 0.11413	valid_1's auc: 0.830369	valid_1's binary_logloss: 0.138547
    [46]	valid_0's auc: 0.897076	valid_0's binary_logloss: 0.113919	valid_1's auc: 0.830246	valid_1's binary_logloss: 0.138644
    [47]	valid_0's auc: 0.897393	valid_0's binary_logloss: 0.113691	valid_1's auc: 0.829999	valid_1's binary_logloss: 0.138769
    [48]	valid_0's auc: 0.897745	valid_0's binary_logloss: 0.113465	valid_1's auc: 0.829593	valid_1's binary_logloss: 0.138908
    [49]	valid_0's auc: 0.898203	valid_0's binary_logloss: 0.113214	valid_1's auc: 0.829209	valid_1's binary_logloss: 0.139045
    [50]	valid_0's auc: 0.898546	valid_0's binary_logloss: 0.112978	valid_1's auc: 0.829064	valid_1's binary_logloss: 0.139115
    [51]	valid_0's auc: 0.898958	valid_0's binary_logloss: 0.112723	valid_1's auc: 0.82913	valid_1's binary_logloss: 0.139163
    [52]	valid_0's auc: 0.899296	valid_0's binary_logloss: 0.112541	valid_1's auc: 0.828532	valid_1's binary_logloss: 0.139358
    [53]	valid_0's auc: 0.899966	valid_0's binary_logloss: 0.112208	valid_1's auc: 0.828277	valid_1's binary_logloss: 0.139431
    [54]	valid_0's auc: 0.900225	valid_0's binary_logloss: 0.112001	valid_1's auc: 0.827866	valid_1's binary_logloss: 0.139573
    [55]	valid_0's auc: 0.90036	valid_0's binary_logloss: 0.111812	valid_1's auc: 0.827887	valid_1's binary_logloss: 0.139655
    [56]	valid_0's auc: 0.900528	valid_0's binary_logloss: 0.111638	valid_1's auc: 0.827809	valid_1's binary_logloss: 0.139702
    [57]	valid_0's auc: 0.900692	valid_0's binary_logloss: 0.111452	valid_1's auc: 0.827613	valid_1's binary_logloss: 0.139758
    [58]	valid_0's auc: 0.901275	valid_0's binary_logloss: 0.111262	valid_1's auc: 0.827848	valid_1's binary_logloss: 0.139758
    [59]	valid_0's auc: 0.90149	valid_0's binary_logloss: 0.111086	valid_1's auc: 0.827643	valid_1's binary_logloss: 0.139835
    [60]	valid_0's auc: 0.901669	valid_0's binary_logloss: 0.110873	valid_1's auc: 0.827187	valid_1's binary_logloss: 0.13998
    [61]	valid_0's auc: 0.901715	valid_0's binary_logloss: 0.110743	valid_1's auc: 0.826643	valid_1's binary_logloss: 0.140136
    Early stopping, best iteration is:
    [31]	valid_0's auc: 0.888552	valid_0's binary_logloss: 0.118399	valid_1's auc: 0.834036	valid_1's binary_logloss: 0.137459
    [1]	valid_0's auc: 0.828293	valid_0's binary_logloss: 0.156591	valid_1's auc: 0.803729	valid_1's binary_logloss: 0.159809
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.836848	valid_0's binary_logloss: 0.150955	valid_1's auc: 0.813468	valid_1's binary_logloss: 0.155406
    [3]	valid_0's auc: 0.842786	valid_0's binary_logloss: 0.146796	valid_1's auc: 0.817407	valid_1's binary_logloss: 0.151952
    [4]	valid_0's auc: 0.851348	valid_0's binary_logloss: 0.143395	valid_1's auc: 0.823949	valid_1's binary_logloss: 0.14934
    [5]	valid_0's auc: 0.855149	valid_0's binary_logloss: 0.140626	valid_1's auc: 0.827122	valid_1's binary_logloss: 0.147185
    [6]	valid_0's auc: 0.857679	valid_0's binary_logloss: 0.13837	valid_1's auc: 0.826758	valid_1's binary_logloss: 0.145708
    [7]	valid_0's auc: 0.86155	valid_0's binary_logloss: 0.136445	valid_1's auc: 0.828472	valid_1's binary_logloss: 0.144294
    [8]	valid_0's auc: 0.862588	valid_0's binary_logloss: 0.134808	valid_1's auc: 0.828623	valid_1's binary_logloss: 0.143251
    [9]	valid_0's auc: 0.865115	valid_0's binary_logloss: 0.133359	valid_1's auc: 0.831471	valid_1's binary_logloss: 0.142276
    [10]	valid_0's auc: 0.866894	valid_0's binary_logloss: 0.132073	valid_1's auc: 0.831985	valid_1's binary_logloss: 0.141408
    [11]	valid_0's auc: 0.869042	valid_0's binary_logloss: 0.130873	valid_1's auc: 0.834107	valid_1's binary_logloss: 0.140661
    [12]	valid_0's auc: 0.870884	valid_0's binary_logloss: 0.129808	valid_1's auc: 0.833841	valid_1's binary_logloss: 0.140137
    [13]	valid_0's auc: 0.87235	valid_0's binary_logloss: 0.128859	valid_1's auc: 0.834392	valid_1's binary_logloss: 0.139657
    [14]	valid_0's auc: 0.873638	valid_0's binary_logloss: 0.128012	valid_1's auc: 0.834569	valid_1's binary_logloss: 0.13928
    [15]	valid_0's auc: 0.874792	valid_0's binary_logloss: 0.127177	valid_1's auc: 0.83417	valid_1's binary_logloss: 0.138989
    [16]	valid_0's auc: 0.876361	valid_0's binary_logloss: 0.126304	valid_1's auc: 0.835986	valid_1's binary_logloss: 0.138606
    [17]	valid_0's auc: 0.878328	valid_0's binary_logloss: 0.125589	valid_1's auc: 0.836721	valid_1's binary_logloss: 0.138362
    [18]	valid_0's auc: 0.879835	valid_0's binary_logloss: 0.124908	valid_1's auc: 0.836975	valid_1's binary_logloss: 0.138199
    [19]	valid_0's auc: 0.881056	valid_0's binary_logloss: 0.124303	valid_1's auc: 0.837163	valid_1's binary_logloss: 0.137993
    [20]	valid_0's auc: 0.881684	valid_0's binary_logloss: 0.123706	valid_1's auc: 0.836767	valid_1's binary_logloss: 0.137922
    [21]	valid_0's auc: 0.882852	valid_0's binary_logloss: 0.123138	valid_1's auc: 0.835726	valid_1's binary_logloss: 0.137937
    [22]	valid_0's auc: 0.883782	valid_0's binary_logloss: 0.122621	valid_1's auc: 0.836033	valid_1's binary_logloss: 0.137832
    [23]	valid_0's auc: 0.884894	valid_0's binary_logloss: 0.122101	valid_1's auc: 0.835965	valid_1's binary_logloss: 0.137761
    [24]	valid_0's auc: 0.886129	valid_0's binary_logloss: 0.121539	valid_1's auc: 0.836276	valid_1's binary_logloss: 0.137662
    [25]	valid_0's auc: 0.887299	valid_0's binary_logloss: 0.121003	valid_1's auc: 0.836494	valid_1's binary_logloss: 0.137562
    [26]	valid_0's auc: 0.888167	valid_0's binary_logloss: 0.120533	valid_1's auc: 0.836735	valid_1's binary_logloss: 0.137471
    [27]	valid_0's auc: 0.889209	valid_0's binary_logloss: 0.120033	valid_1's auc: 0.837077	valid_1's binary_logloss: 0.137335
    [28]	valid_0's auc: 0.890083	valid_0's binary_logloss: 0.119628	valid_1's auc: 0.836646	valid_1's binary_logloss: 0.137396
    [29]	valid_0's auc: 0.891268	valid_0's binary_logloss: 0.1192	valid_1's auc: 0.836893	valid_1's binary_logloss: 0.137334
    [30]	valid_0's auc: 0.892026	valid_0's binary_logloss: 0.118779	valid_1's auc: 0.836987	valid_1's binary_logloss: 0.13734
    [31]	valid_0's auc: 0.892612	valid_0's binary_logloss: 0.1184	valid_1's auc: 0.835995	valid_1's binary_logloss: 0.137457
    [32]	valid_0's auc: 0.893409	valid_0's binary_logloss: 0.118028	valid_1's auc: 0.836329	valid_1's binary_logloss: 0.137426
    [33]	valid_0's auc: 0.894715	valid_0's binary_logloss: 0.117634	valid_1's auc: 0.836237	valid_1's binary_logloss: 0.137428
    [34]	valid_0's auc: 0.896129	valid_0's binary_logloss: 0.117239	valid_1's auc: 0.837278	valid_1's binary_logloss: 0.137234
    [35]	valid_0's auc: 0.897111	valid_0's binary_logloss: 0.116881	valid_1's auc: 0.837313	valid_1's binary_logloss: 0.137208
    [36]	valid_0's auc: 0.89791	valid_0's binary_logloss: 0.116525	valid_1's auc: 0.837094	valid_1's binary_logloss: 0.137227
    [37]	valid_0's auc: 0.898327	valid_0's binary_logloss: 0.116184	valid_1's auc: 0.836808	valid_1's binary_logloss: 0.137267
    [38]	valid_0's auc: 0.899162	valid_0's binary_logloss: 0.115859	valid_1's auc: 0.836473	valid_1's binary_logloss: 0.137323
    [39]	valid_0's auc: 0.899593	valid_0's binary_logloss: 0.115562	valid_1's auc: 0.836433	valid_1's binary_logloss: 0.137356
    [40]	valid_0's auc: 0.899942	valid_0's binary_logloss: 0.115297	valid_1's auc: 0.836299	valid_1's binary_logloss: 0.137382
    [41]	valid_0's auc: 0.900439	valid_0's binary_logloss: 0.114998	valid_1's auc: 0.835877	valid_1's binary_logloss: 0.137474
    [42]	valid_0's auc: 0.90094	valid_0's binary_logloss: 0.1147	valid_1's auc: 0.835016	valid_1's binary_logloss: 0.137628
    [43]	valid_0's auc: 0.901223	valid_0's binary_logloss: 0.11447	valid_1's auc: 0.834849	valid_1's binary_logloss: 0.137661
    [44]	valid_0's auc: 0.901438	valid_0's binary_logloss: 0.11423	valid_1's auc: 0.834743	valid_1's binary_logloss: 0.137699
    [45]	valid_0's auc: 0.902058	valid_0's binary_logloss: 0.113982	valid_1's auc: 0.834615	valid_1's binary_logloss: 0.13774
    [46]	valid_0's auc: 0.902431	valid_0's binary_logloss: 0.113713	valid_1's auc: 0.834207	valid_1's binary_logloss: 0.13784
    [47]	valid_0's auc: 0.902812	valid_0's binary_logloss: 0.113487	valid_1's auc: 0.833758	valid_1's binary_logloss: 0.137985
    [48]	valid_0's auc: 0.90306	valid_0's binary_logloss: 0.113301	valid_1's auc: 0.833219	valid_1's binary_logloss: 0.138122
    [49]	valid_0's auc: 0.903267	valid_0's binary_logloss: 0.113063	valid_1's auc: 0.832937	valid_1's binary_logloss: 0.138197
    [50]	valid_0's auc: 0.903705	valid_0's binary_logloss: 0.112863	valid_1's auc: 0.832815	valid_1's binary_logloss: 0.13824
    [51]	valid_0's auc: 0.904223	valid_0's binary_logloss: 0.112624	valid_1's auc: 0.832789	valid_1's binary_logloss: 0.138311
    [52]	valid_0's auc: 0.904867	valid_0's binary_logloss: 0.112372	valid_1's auc: 0.832399	valid_1's binary_logloss: 0.138416
    [53]	valid_0's auc: 0.90521	valid_0's binary_logloss: 0.112133	valid_1's auc: 0.832296	valid_1's binary_logloss: 0.1385
    [54]	valid_0's auc: 0.905689	valid_0's binary_logloss: 0.111851	valid_1's auc: 0.831911	valid_1's binary_logloss: 0.138563
    [55]	valid_0's auc: 0.905941	valid_0's binary_logloss: 0.111672	valid_1's auc: 0.831691	valid_1's binary_logloss: 0.138659
    [56]	valid_0's auc: 0.906184	valid_0's binary_logloss: 0.111482	valid_1's auc: 0.83155	valid_1's binary_logloss: 0.138723
    [57]	valid_0's auc: 0.906352	valid_0's binary_logloss: 0.111271	valid_1's auc: 0.830993	valid_1's binary_logloss: 0.138832
    [58]	valid_0's auc: 0.906874	valid_0's binary_logloss: 0.110997	valid_1's auc: 0.831151	valid_1's binary_logloss: 0.138806
    [59]	valid_0's auc: 0.906995	valid_0's binary_logloss: 0.110797	valid_1's auc: 0.830642	valid_1's binary_logloss: 0.138926
    [60]	valid_0's auc: 0.907239	valid_0's binary_logloss: 0.110639	valid_1's auc: 0.830434	valid_1's binary_logloss: 0.139017
    [61]	valid_0's auc: 0.907442	valid_0's binary_logloss: 0.110465	valid_1's auc: 0.829813	valid_1's binary_logloss: 0.13916
    [62]	valid_0's auc: 0.908053	valid_0's binary_logloss: 0.110189	valid_1's auc: 0.829969	valid_1's binary_logloss: 0.139161
    [63]	valid_0's auc: 0.908139	valid_0's binary_logloss: 0.110026	valid_1's auc: 0.829479	valid_1's binary_logloss: 0.139308
    [64]	valid_0's auc: 0.908183	valid_0's binary_logloss: 0.109892	valid_1's auc: 0.828768	valid_1's binary_logloss: 0.139496
    [65]	valid_0's auc: 0.908915	valid_0's binary_logloss: 0.109601	valid_1's auc: 0.828359	valid_1's binary_logloss: 0.139582
    Early stopping, best iteration is:
    [35]	valid_0's auc: 0.897111	valid_0's binary_logloss: 0.116881	valid_1's auc: 0.837313	valid_1's binary_logloss: 0.137208
    [1]	valid_0's auc: 0.833154	valid_0's binary_logloss: 0.156832	valid_1's auc: 0.809041	valid_1's binary_logloss: 0.159834
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.842696	valid_0's binary_logloss: 0.15097	valid_1's auc: 0.816072	valid_1's binary_logloss: 0.155054
    [3]	valid_0's auc: 0.848628	valid_0's binary_logloss: 0.146679	valid_1's auc: 0.821987	valid_1's binary_logloss: 0.151615
    [4]	valid_0's auc: 0.852189	valid_0's binary_logloss: 0.143409	valid_1's auc: 0.823257	valid_1's binary_logloss: 0.148971
    [5]	valid_0's auc: 0.85445	valid_0's binary_logloss: 0.140691	valid_1's auc: 0.826693	valid_1's binary_logloss: 0.146922
    [6]	valid_0's auc: 0.856663	valid_0's binary_logloss: 0.138477	valid_1's auc: 0.827879	valid_1's binary_logloss: 0.145076
    [7]	valid_0's auc: 0.858141	valid_0's binary_logloss: 0.136527	valid_1's auc: 0.828071	valid_1's binary_logloss: 0.143771
    [8]	valid_0's auc: 0.860436	valid_0's binary_logloss: 0.134845	valid_1's auc: 0.830521	valid_1's binary_logloss: 0.142677
    [9]	valid_0's auc: 0.861929	valid_0's binary_logloss: 0.133357	valid_1's auc: 0.831252	valid_1's binary_logloss: 0.14175
    [10]	valid_0's auc: 0.865268	valid_0's binary_logloss: 0.132002	valid_1's auc: 0.831065	valid_1's binary_logloss: 0.141032
    [11]	valid_0's auc: 0.86773	valid_0's binary_logloss: 0.130767	valid_1's auc: 0.831969	valid_1's binary_logloss: 0.140477
    [12]	valid_0's auc: 0.869797	valid_0's binary_logloss: 0.129639	valid_1's auc: 0.831747	valid_1's binary_logloss: 0.139964
    [13]	valid_0's auc: 0.871962	valid_0's binary_logloss: 0.128704	valid_1's auc: 0.832299	valid_1's binary_logloss: 0.13953
    [14]	valid_0's auc: 0.873656	valid_0's binary_logloss: 0.12785	valid_1's auc: 0.831874	valid_1's binary_logloss: 0.139277
    [15]	valid_0's auc: 0.875313	valid_0's binary_logloss: 0.127041	valid_1's auc: 0.831922	valid_1's binary_logloss: 0.139006
    [16]	valid_0's auc: 0.875813	valid_0's binary_logloss: 0.126291	valid_1's auc: 0.832339	valid_1's binary_logloss: 0.138679
    [17]	valid_0's auc: 0.877574	valid_0's binary_logloss: 0.125606	valid_1's auc: 0.832459	valid_1's binary_logloss: 0.138444
    [18]	valid_0's auc: 0.879134	valid_0's binary_logloss: 0.12489	valid_1's auc: 0.832409	valid_1's binary_logloss: 0.138268
    [19]	valid_0's auc: 0.880176	valid_0's binary_logloss: 0.124247	valid_1's auc: 0.832557	valid_1's binary_logloss: 0.138164
    [20]	valid_0's auc: 0.881608	valid_0's binary_logloss: 0.123609	valid_1's auc: 0.832053	valid_1's binary_logloss: 0.138136
    [21]	valid_0's auc: 0.882632	valid_0's binary_logloss: 0.123013	valid_1's auc: 0.831852	valid_1's binary_logloss: 0.138081
    [22]	valid_0's auc: 0.883684	valid_0's binary_logloss: 0.12245	valid_1's auc: 0.832034	valid_1's binary_logloss: 0.137978
    [23]	valid_0's auc: 0.885039	valid_0's binary_logloss: 0.121827	valid_1's auc: 0.831422	valid_1's binary_logloss: 0.138011
    [24]	valid_0's auc: 0.886247	valid_0's binary_logloss: 0.121291	valid_1's auc: 0.831253	valid_1's binary_logloss: 0.137942
    [25]	valid_0's auc: 0.887301	valid_0's binary_logloss: 0.120809	valid_1's auc: 0.831276	valid_1's binary_logloss: 0.137924
    [26]	valid_0's auc: 0.888621	valid_0's binary_logloss: 0.120287	valid_1's auc: 0.830735	valid_1's binary_logloss: 0.137975
    [27]	valid_0's auc: 0.889227	valid_0's binary_logloss: 0.119878	valid_1's auc: 0.830569	valid_1's binary_logloss: 0.137976
    [28]	valid_0's auc: 0.890381	valid_0's binary_logloss: 0.119445	valid_1's auc: 0.830281	valid_1's binary_logloss: 0.137968
    [29]	valid_0's auc: 0.891139	valid_0's binary_logloss: 0.119016	valid_1's auc: 0.830916	valid_1's binary_logloss: 0.137839
    [30]	valid_0's auc: 0.892417	valid_0's binary_logloss: 0.118535	valid_1's auc: 0.830189	valid_1's binary_logloss: 0.137909
    [31]	valid_0's auc: 0.893102	valid_0's binary_logloss: 0.118157	valid_1's auc: 0.830117	valid_1's binary_logloss: 0.137933
    [32]	valid_0's auc: 0.893844	valid_0's binary_logloss: 0.117764	valid_1's auc: 0.829877	valid_1's binary_logloss: 0.137962
    [33]	valid_0's auc: 0.894508	valid_0's binary_logloss: 0.117419	valid_1's auc: 0.829552	valid_1's binary_logloss: 0.138044
    [34]	valid_0's auc: 0.895157	valid_0's binary_logloss: 0.117051	valid_1's auc: 0.830486	valid_1's binary_logloss: 0.137893
    [35]	valid_0's auc: 0.89588	valid_0's binary_logloss: 0.116679	valid_1's auc: 0.83041	valid_1's binary_logloss: 0.137896
    [36]	valid_0's auc: 0.896611	valid_0's binary_logloss: 0.116295	valid_1's auc: 0.829969	valid_1's binary_logloss: 0.138026
    [37]	valid_0's auc: 0.89762	valid_0's binary_logloss: 0.115979	valid_1's auc: 0.829802	valid_1's binary_logloss: 0.138089
    [38]	valid_0's auc: 0.898028	valid_0's binary_logloss: 0.115667	valid_1's auc: 0.829346	valid_1's binary_logloss: 0.138207
    [39]	valid_0's auc: 0.89839	valid_0's binary_logloss: 0.115401	valid_1's auc: 0.829208	valid_1's binary_logloss: 0.138191
    [40]	valid_0's auc: 0.898968	valid_0's binary_logloss: 0.115042	valid_1's auc: 0.828576	valid_1's binary_logloss: 0.138317
    [41]	valid_0's auc: 0.899373	valid_0's binary_logloss: 0.114766	valid_1's auc: 0.828382	valid_1's binary_logloss: 0.138408
    [42]	valid_0's auc: 0.900042	valid_0's binary_logloss: 0.114467	valid_1's auc: 0.828023	valid_1's binary_logloss: 0.138489
    [43]	valid_0's auc: 0.900488	valid_0's binary_logloss: 0.114194	valid_1's auc: 0.827815	valid_1's binary_logloss: 0.138591
    [44]	valid_0's auc: 0.901004	valid_0's binary_logloss: 0.113865	valid_1's auc: 0.827321	valid_1's binary_logloss: 0.138736
    [45]	valid_0's auc: 0.901401	valid_0's binary_logloss: 0.113616	valid_1's auc: 0.826875	valid_1's binary_logloss: 0.138874
    [46]	valid_0's auc: 0.901795	valid_0's binary_logloss: 0.113348	valid_1's auc: 0.82643	valid_1's binary_logloss: 0.138955
    [47]	valid_0's auc: 0.902244	valid_0's binary_logloss: 0.11309	valid_1's auc: 0.825825	valid_1's binary_logloss: 0.13908
    [48]	valid_0's auc: 0.902756	valid_0's binary_logloss: 0.112891	valid_1's auc: 0.825639	valid_1's binary_logloss: 0.139117
    [49]	valid_0's auc: 0.903206	valid_0's binary_logloss: 0.112607	valid_1's auc: 0.825607	valid_1's binary_logloss: 0.139184
    Early stopping, best iteration is:
    [19]	valid_0's auc: 0.880176	valid_0's binary_logloss: 0.124247	valid_1's auc: 0.832557	valid_1's binary_logloss: 0.138164
    [1]	valid_0's auc: 0.825869	valid_0's binary_logloss: 0.156436	valid_1's auc: 0.803397	valid_1's binary_logloss: 0.159993
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.834474	valid_0's binary_logloss: 0.150597	valid_1's auc: 0.809972	valid_1's binary_logloss: 0.155122
    [3]	valid_0's auc: 0.840511	valid_0's binary_logloss: 0.146374	valid_1's auc: 0.819137	valid_1's binary_logloss: 0.15148
    [4]	valid_0's auc: 0.844283	valid_0's binary_logloss: 0.143027	valid_1's auc: 0.820472	valid_1's binary_logloss: 0.14904
    [5]	valid_0's auc: 0.85046	valid_0's binary_logloss: 0.140321	valid_1's auc: 0.82606	valid_1's binary_logloss: 0.146982
    [6]	valid_0's auc: 0.852881	valid_0's binary_logloss: 0.13813	valid_1's auc: 0.823869	valid_1's binary_logloss: 0.145517
    [7]	valid_0's auc: 0.857268	valid_0's binary_logloss: 0.136149	valid_1's auc: 0.827586	valid_1's binary_logloss: 0.144156
    [8]	valid_0's auc: 0.859401	valid_0's binary_logloss: 0.13452	valid_1's auc: 0.829428	valid_1's binary_logloss: 0.14303
    [9]	valid_0's auc: 0.861435	valid_0's binary_logloss: 0.133011	valid_1's auc: 0.82971	valid_1's binary_logloss: 0.142113
    [10]	valid_0's auc: 0.862282	valid_0's binary_logloss: 0.131742	valid_1's auc: 0.830527	valid_1's binary_logloss: 0.141392
    [11]	valid_0's auc: 0.86448	valid_0's binary_logloss: 0.130536	valid_1's auc: 0.83065	valid_1's binary_logloss: 0.140854
    [12]	valid_0's auc: 0.865909	valid_0's binary_logloss: 0.129495	valid_1's auc: 0.830763	valid_1's binary_logloss: 0.140425
    [13]	valid_0's auc: 0.867083	valid_0's binary_logloss: 0.128537	valid_1's auc: 0.830097	valid_1's binary_logloss: 0.140034
    [14]	valid_0's auc: 0.869164	valid_0's binary_logloss: 0.127593	valid_1's auc: 0.831242	valid_1's binary_logloss: 0.139589
    [15]	valid_0's auc: 0.869779	valid_0's binary_logloss: 0.126753	valid_1's auc: 0.830054	valid_1's binary_logloss: 0.13935
    [16]	valid_0's auc: 0.870798	valid_0's binary_logloss: 0.125998	valid_1's auc: 0.829854	valid_1's binary_logloss: 0.139147
    [17]	valid_0's auc: 0.872413	valid_0's binary_logloss: 0.125321	valid_1's auc: 0.829939	valid_1's binary_logloss: 0.138859
    [18]	valid_0's auc: 0.873656	valid_0's binary_logloss: 0.124646	valid_1's auc: 0.83033	valid_1's binary_logloss: 0.138678
    [19]	valid_0's auc: 0.874787	valid_0's binary_logloss: 0.124094	valid_1's auc: 0.829747	valid_1's binary_logloss: 0.138635
    [20]	valid_0's auc: 0.875793	valid_0's binary_logloss: 0.123547	valid_1's auc: 0.829455	valid_1's binary_logloss: 0.138492
    [21]	valid_0's auc: 0.87694	valid_0's binary_logloss: 0.122969	valid_1's auc: 0.830945	valid_1's binary_logloss: 0.13825
    [22]	valid_0's auc: 0.878632	valid_0's binary_logloss: 0.122407	valid_1's auc: 0.831483	valid_1's binary_logloss: 0.138097
    [23]	valid_0's auc: 0.8796	valid_0's binary_logloss: 0.121869	valid_1's auc: 0.831408	valid_1's binary_logloss: 0.138071
    [24]	valid_0's auc: 0.881298	valid_0's binary_logloss: 0.121363	valid_1's auc: 0.831984	valid_1's binary_logloss: 0.137912
    [25]	valid_0's auc: 0.882699	valid_0's binary_logloss: 0.120863	valid_1's auc: 0.831941	valid_1's binary_logloss: 0.137824
    [26]	valid_0's auc: 0.883799	valid_0's binary_logloss: 0.120379	valid_1's auc: 0.832897	valid_1's binary_logloss: 0.137649
    [27]	valid_0's auc: 0.884893	valid_0's binary_logloss: 0.119944	valid_1's auc: 0.832841	valid_1's binary_logloss: 0.13763
    [28]	valid_0's auc: 0.88588	valid_0's binary_logloss: 0.119545	valid_1's auc: 0.833413	valid_1's binary_logloss: 0.137495
    [29]	valid_0's auc: 0.886461	valid_0's binary_logloss: 0.11916	valid_1's auc: 0.833437	valid_1's binary_logloss: 0.137511
    [30]	valid_0's auc: 0.887592	valid_0's binary_logloss: 0.118739	valid_1's auc: 0.833429	valid_1's binary_logloss: 0.137514
    [31]	valid_0's auc: 0.888552	valid_0's binary_logloss: 0.118399	valid_1's auc: 0.834036	valid_1's binary_logloss: 0.137459
    [32]	valid_0's auc: 0.889448	valid_0's binary_logloss: 0.118019	valid_1's auc: 0.833754	valid_1's binary_logloss: 0.137498
    [33]	valid_0's auc: 0.890203	valid_0's binary_logloss: 0.117711	valid_1's auc: 0.833853	valid_1's binary_logloss: 0.137491
    [34]	valid_0's auc: 0.890868	valid_0's binary_logloss: 0.117395	valid_1's auc: 0.833648	valid_1's binary_logloss: 0.137532
    [35]	valid_0's auc: 0.891378	valid_0's binary_logloss: 0.117106	valid_1's auc: 0.833479	valid_1's binary_logloss: 0.137577
    [36]	valid_0's auc: 0.892061	valid_0's binary_logloss: 0.116741	valid_1's auc: 0.832608	valid_1's binary_logloss: 0.137799
    [37]	valid_0's auc: 0.892715	valid_0's binary_logloss: 0.116403	valid_1's auc: 0.83208	valid_1's binary_logloss: 0.137938
    [38]	valid_0's auc: 0.893146	valid_0's binary_logloss: 0.116126	valid_1's auc: 0.83196	valid_1's binary_logloss: 0.137994
    [39]	valid_0's auc: 0.893963	valid_0's binary_logloss: 0.115783	valid_1's auc: 0.831584	valid_1's binary_logloss: 0.138076
    [40]	valid_0's auc: 0.894274	valid_0's binary_logloss: 0.115535	valid_1's auc: 0.831346	valid_1's binary_logloss: 0.13818
    [41]	valid_0's auc: 0.894809	valid_0's binary_logloss: 0.115247	valid_1's auc: 0.831197	valid_1's binary_logloss: 0.138289
    [42]	valid_0's auc: 0.895332	valid_0's binary_logloss: 0.114916	valid_1's auc: 0.830893	valid_1's binary_logloss: 0.138343
    [43]	valid_0's auc: 0.895864	valid_0's binary_logloss: 0.114701	valid_1's auc: 0.83129	valid_1's binary_logloss: 0.138307
    [44]	valid_0's auc: 0.89623	valid_0's binary_logloss: 0.114411	valid_1's auc: 0.831004	valid_1's binary_logloss: 0.138408
    [45]	valid_0's auc: 0.896616	valid_0's binary_logloss: 0.11413	valid_1's auc: 0.830369	valid_1's binary_logloss: 0.138547
    [46]	valid_0's auc: 0.897076	valid_0's binary_logloss: 0.113919	valid_1's auc: 0.830246	valid_1's binary_logloss: 0.138644
    [47]	valid_0's auc: 0.897393	valid_0's binary_logloss: 0.113691	valid_1's auc: 0.829999	valid_1's binary_logloss: 0.138769
    [48]	valid_0's auc: 0.897745	valid_0's binary_logloss: 0.113465	valid_1's auc: 0.829593	valid_1's binary_logloss: 0.138908
    [49]	valid_0's auc: 0.898203	valid_0's binary_logloss: 0.113214	valid_1's auc: 0.829209	valid_1's binary_logloss: 0.139045
    [50]	valid_0's auc: 0.898546	valid_0's binary_logloss: 0.112978	valid_1's auc: 0.829064	valid_1's binary_logloss: 0.139115
    [51]	valid_0's auc: 0.898958	valid_0's binary_logloss: 0.112723	valid_1's auc: 0.82913	valid_1's binary_logloss: 0.139163
    [52]	valid_0's auc: 0.899296	valid_0's binary_logloss: 0.112541	valid_1's auc: 0.828532	valid_1's binary_logloss: 0.139358
    [53]	valid_0's auc: 0.899966	valid_0's binary_logloss: 0.112208	valid_1's auc: 0.828277	valid_1's binary_logloss: 0.139431
    [54]	valid_0's auc: 0.900225	valid_0's binary_logloss: 0.112001	valid_1's auc: 0.827866	valid_1's binary_logloss: 0.139573
    [55]	valid_0's auc: 0.90036	valid_0's binary_logloss: 0.111812	valid_1's auc: 0.827887	valid_1's binary_logloss: 0.139655
    [56]	valid_0's auc: 0.900528	valid_0's binary_logloss: 0.111638	valid_1's auc: 0.827809	valid_1's binary_logloss: 0.139702
    [57]	valid_0's auc: 0.900692	valid_0's binary_logloss: 0.111452	valid_1's auc: 0.827613	valid_1's binary_logloss: 0.139758
    [58]	valid_0's auc: 0.901275	valid_0's binary_logloss: 0.111262	valid_1's auc: 0.827848	valid_1's binary_logloss: 0.139758
    [59]	valid_0's auc: 0.90149	valid_0's binary_logloss: 0.111086	valid_1's auc: 0.827643	valid_1's binary_logloss: 0.139835
    [60]	valid_0's auc: 0.901669	valid_0's binary_logloss: 0.110873	valid_1's auc: 0.827187	valid_1's binary_logloss: 0.13998
    [61]	valid_0's auc: 0.901715	valid_0's binary_logloss: 0.110743	valid_1's auc: 0.826643	valid_1's binary_logloss: 0.140136
    Early stopping, best iteration is:
    [31]	valid_0's auc: 0.888552	valid_0's binary_logloss: 0.118399	valid_1's auc: 0.834036	valid_1's binary_logloss: 0.137459
    [1]	valid_0's auc: 0.828293	valid_0's binary_logloss: 0.156591	valid_1's auc: 0.803729	valid_1's binary_logloss: 0.159809
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.836848	valid_0's binary_logloss: 0.150955	valid_1's auc: 0.813468	valid_1's binary_logloss: 0.155406
    [3]	valid_0's auc: 0.842786	valid_0's binary_logloss: 0.146796	valid_1's auc: 0.817407	valid_1's binary_logloss: 0.151952
    [4]	valid_0's auc: 0.851348	valid_0's binary_logloss: 0.143395	valid_1's auc: 0.823949	valid_1's binary_logloss: 0.14934
    [5]	valid_0's auc: 0.855149	valid_0's binary_logloss: 0.140626	valid_1's auc: 0.827122	valid_1's binary_logloss: 0.147185
    [6]	valid_0's auc: 0.857679	valid_0's binary_logloss: 0.13837	valid_1's auc: 0.826758	valid_1's binary_logloss: 0.145708
    [7]	valid_0's auc: 0.86155	valid_0's binary_logloss: 0.136445	valid_1's auc: 0.828472	valid_1's binary_logloss: 0.144294
    [8]	valid_0's auc: 0.862588	valid_0's binary_logloss: 0.134808	valid_1's auc: 0.828623	valid_1's binary_logloss: 0.143251
    [9]	valid_0's auc: 0.865115	valid_0's binary_logloss: 0.133359	valid_1's auc: 0.831471	valid_1's binary_logloss: 0.142276
    [10]	valid_0's auc: 0.866894	valid_0's binary_logloss: 0.132073	valid_1's auc: 0.831985	valid_1's binary_logloss: 0.141408
    [11]	valid_0's auc: 0.869042	valid_0's binary_logloss: 0.130873	valid_1's auc: 0.834107	valid_1's binary_logloss: 0.140661
    [12]	valid_0's auc: 0.870884	valid_0's binary_logloss: 0.129808	valid_1's auc: 0.833841	valid_1's binary_logloss: 0.140137
    [13]	valid_0's auc: 0.87235	valid_0's binary_logloss: 0.128859	valid_1's auc: 0.834392	valid_1's binary_logloss: 0.139657
    [14]	valid_0's auc: 0.873638	valid_0's binary_logloss: 0.128012	valid_1's auc: 0.834569	valid_1's binary_logloss: 0.13928
    [15]	valid_0's auc: 0.874792	valid_0's binary_logloss: 0.127177	valid_1's auc: 0.83417	valid_1's binary_logloss: 0.138989
    [16]	valid_0's auc: 0.876361	valid_0's binary_logloss: 0.126304	valid_1's auc: 0.835986	valid_1's binary_logloss: 0.138606
    [17]	valid_0's auc: 0.878328	valid_0's binary_logloss: 0.125589	valid_1's auc: 0.836721	valid_1's binary_logloss: 0.138362
    [18]	valid_0's auc: 0.879835	valid_0's binary_logloss: 0.124908	valid_1's auc: 0.836975	valid_1's binary_logloss: 0.138199
    [19]	valid_0's auc: 0.881056	valid_0's binary_logloss: 0.124303	valid_1's auc: 0.837163	valid_1's binary_logloss: 0.137993
    [20]	valid_0's auc: 0.881684	valid_0's binary_logloss: 0.123706	valid_1's auc: 0.836767	valid_1's binary_logloss: 0.137922
    [21]	valid_0's auc: 0.882852	valid_0's binary_logloss: 0.123138	valid_1's auc: 0.835726	valid_1's binary_logloss: 0.137937
    [22]	valid_0's auc: 0.883782	valid_0's binary_logloss: 0.122621	valid_1's auc: 0.836033	valid_1's binary_logloss: 0.137832
    [23]	valid_0's auc: 0.884894	valid_0's binary_logloss: 0.122101	valid_1's auc: 0.835965	valid_1's binary_logloss: 0.137761
    [24]	valid_0's auc: 0.886129	valid_0's binary_logloss: 0.121539	valid_1's auc: 0.836276	valid_1's binary_logloss: 0.137662
    [25]	valid_0's auc: 0.887299	valid_0's binary_logloss: 0.121003	valid_1's auc: 0.836494	valid_1's binary_logloss: 0.137562
    [26]	valid_0's auc: 0.888167	valid_0's binary_logloss: 0.120533	valid_1's auc: 0.836735	valid_1's binary_logloss: 0.137471
    [27]	valid_0's auc: 0.889209	valid_0's binary_logloss: 0.120033	valid_1's auc: 0.837077	valid_1's binary_logloss: 0.137335
    [28]	valid_0's auc: 0.890083	valid_0's binary_logloss: 0.119628	valid_1's auc: 0.836646	valid_1's binary_logloss: 0.137396
    [29]	valid_0's auc: 0.891268	valid_0's binary_logloss: 0.1192	valid_1's auc: 0.836893	valid_1's binary_logloss: 0.137334
    [30]	valid_0's auc: 0.892026	valid_0's binary_logloss: 0.118779	valid_1's auc: 0.836987	valid_1's binary_logloss: 0.13734
    [31]	valid_0's auc: 0.892612	valid_0's binary_logloss: 0.1184	valid_1's auc: 0.835995	valid_1's binary_logloss: 0.137457
    [32]	valid_0's auc: 0.893409	valid_0's binary_logloss: 0.118028	valid_1's auc: 0.836329	valid_1's binary_logloss: 0.137426
    [33]	valid_0's auc: 0.894715	valid_0's binary_logloss: 0.117634	valid_1's auc: 0.836237	valid_1's binary_logloss: 0.137428
    [34]	valid_0's auc: 0.896129	valid_0's binary_logloss: 0.117239	valid_1's auc: 0.837278	valid_1's binary_logloss: 0.137234
    [35]	valid_0's auc: 0.897111	valid_0's binary_logloss: 0.116881	valid_1's auc: 0.837313	valid_1's binary_logloss: 0.137208
    [36]	valid_0's auc: 0.89791	valid_0's binary_logloss: 0.116525	valid_1's auc: 0.837094	valid_1's binary_logloss: 0.137227
    [37]	valid_0's auc: 0.898327	valid_0's binary_logloss: 0.116184	valid_1's auc: 0.836808	valid_1's binary_logloss: 0.137267
    [38]	valid_0's auc: 0.899162	valid_0's binary_logloss: 0.115859	valid_1's auc: 0.836473	valid_1's binary_logloss: 0.137323
    [39]	valid_0's auc: 0.899593	valid_0's binary_logloss: 0.115562	valid_1's auc: 0.836433	valid_1's binary_logloss: 0.137356
    [40]	valid_0's auc: 0.899942	valid_0's binary_logloss: 0.115297	valid_1's auc: 0.836299	valid_1's binary_logloss: 0.137382
    [41]	valid_0's auc: 0.900439	valid_0's binary_logloss: 0.114998	valid_1's auc: 0.835877	valid_1's binary_logloss: 0.137474
    [42]	valid_0's auc: 0.90094	valid_0's binary_logloss: 0.1147	valid_1's auc: 0.835016	valid_1's binary_logloss: 0.137628
    [43]	valid_0's auc: 0.901223	valid_0's binary_logloss: 0.11447	valid_1's auc: 0.834849	valid_1's binary_logloss: 0.137661
    [44]	valid_0's auc: 0.901438	valid_0's binary_logloss: 0.11423	valid_1's auc: 0.834743	valid_1's binary_logloss: 0.137699
    [45]	valid_0's auc: 0.902058	valid_0's binary_logloss: 0.113982	valid_1's auc: 0.834615	valid_1's binary_logloss: 0.13774
    [46]	valid_0's auc: 0.902431	valid_0's binary_logloss: 0.113713	valid_1's auc: 0.834207	valid_1's binary_logloss: 0.13784
    [47]	valid_0's auc: 0.902812	valid_0's binary_logloss: 0.113487	valid_1's auc: 0.833758	valid_1's binary_logloss: 0.137985
    [48]	valid_0's auc: 0.90306	valid_0's binary_logloss: 0.113301	valid_1's auc: 0.833219	valid_1's binary_logloss: 0.138122
    [49]	valid_0's auc: 0.903267	valid_0's binary_logloss: 0.113063	valid_1's auc: 0.832937	valid_1's binary_logloss: 0.138197
    [50]	valid_0's auc: 0.903705	valid_0's binary_logloss: 0.112863	valid_1's auc: 0.832815	valid_1's binary_logloss: 0.13824
    [51]	valid_0's auc: 0.904223	valid_0's binary_logloss: 0.112624	valid_1's auc: 0.832789	valid_1's binary_logloss: 0.138311
    [52]	valid_0's auc: 0.904867	valid_0's binary_logloss: 0.112372	valid_1's auc: 0.832399	valid_1's binary_logloss: 0.138416
    [53]	valid_0's auc: 0.90521	valid_0's binary_logloss: 0.112133	valid_1's auc: 0.832296	valid_1's binary_logloss: 0.1385
    [54]	valid_0's auc: 0.905689	valid_0's binary_logloss: 0.111851	valid_1's auc: 0.831911	valid_1's binary_logloss: 0.138563
    [55]	valid_0's auc: 0.905941	valid_0's binary_logloss: 0.111672	valid_1's auc: 0.831691	valid_1's binary_logloss: 0.138659
    [56]	valid_0's auc: 0.906184	valid_0's binary_logloss: 0.111482	valid_1's auc: 0.83155	valid_1's binary_logloss: 0.138723
    [57]	valid_0's auc: 0.906352	valid_0's binary_logloss: 0.111271	valid_1's auc: 0.830993	valid_1's binary_logloss: 0.138832
    [58]	valid_0's auc: 0.906874	valid_0's binary_logloss: 0.110997	valid_1's auc: 0.831151	valid_1's binary_logloss: 0.138806
    [59]	valid_0's auc: 0.906995	valid_0's binary_logloss: 0.110797	valid_1's auc: 0.830642	valid_1's binary_logloss: 0.138926
    [60]	valid_0's auc: 0.907239	valid_0's binary_logloss: 0.110639	valid_1's auc: 0.830434	valid_1's binary_logloss: 0.139017
    [61]	valid_0's auc: 0.907442	valid_0's binary_logloss: 0.110465	valid_1's auc: 0.829813	valid_1's binary_logloss: 0.13916
    [62]	valid_0's auc: 0.908053	valid_0's binary_logloss: 0.110189	valid_1's auc: 0.829969	valid_1's binary_logloss: 0.139161
    [63]	valid_0's auc: 0.908139	valid_0's binary_logloss: 0.110026	valid_1's auc: 0.829479	valid_1's binary_logloss: 0.139308
    [64]	valid_0's auc: 0.908183	valid_0's binary_logloss: 0.109892	valid_1's auc: 0.828768	valid_1's binary_logloss: 0.139496
    [65]	valid_0's auc: 0.908915	valid_0's binary_logloss: 0.109601	valid_1's auc: 0.828359	valid_1's binary_logloss: 0.139582
    Early stopping, best iteration is:
    [35]	valid_0's auc: 0.897111	valid_0's binary_logloss: 0.116881	valid_1's auc: 0.837313	valid_1's binary_logloss: 0.137208
    [1]	valid_0's auc: 0.833154	valid_0's binary_logloss: 0.156832	valid_1's auc: 0.809041	valid_1's binary_logloss: 0.159834
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.842696	valid_0's binary_logloss: 0.15097	valid_1's auc: 0.816072	valid_1's binary_logloss: 0.155054
    [3]	valid_0's auc: 0.848628	valid_0's binary_logloss: 0.146679	valid_1's auc: 0.821987	valid_1's binary_logloss: 0.151615
    [4]	valid_0's auc: 0.852189	valid_0's binary_logloss: 0.143409	valid_1's auc: 0.823257	valid_1's binary_logloss: 0.148971
    [5]	valid_0's auc: 0.85445	valid_0's binary_logloss: 0.140691	valid_1's auc: 0.826693	valid_1's binary_logloss: 0.146922
    [6]	valid_0's auc: 0.856663	valid_0's binary_logloss: 0.138477	valid_1's auc: 0.827879	valid_1's binary_logloss: 0.145076
    [7]	valid_0's auc: 0.858141	valid_0's binary_logloss: 0.136527	valid_1's auc: 0.828071	valid_1's binary_logloss: 0.143771
    [8]	valid_0's auc: 0.860436	valid_0's binary_logloss: 0.134845	valid_1's auc: 0.830521	valid_1's binary_logloss: 0.142677
    [9]	valid_0's auc: 0.861929	valid_0's binary_logloss: 0.133357	valid_1's auc: 0.831252	valid_1's binary_logloss: 0.14175
    [10]	valid_0's auc: 0.865268	valid_0's binary_logloss: 0.132002	valid_1's auc: 0.831065	valid_1's binary_logloss: 0.141032
    [11]	valid_0's auc: 0.86773	valid_0's binary_logloss: 0.130767	valid_1's auc: 0.831969	valid_1's binary_logloss: 0.140477
    [12]	valid_0's auc: 0.869797	valid_0's binary_logloss: 0.129639	valid_1's auc: 0.831747	valid_1's binary_logloss: 0.139964
    [13]	valid_0's auc: 0.871962	valid_0's binary_logloss: 0.128704	valid_1's auc: 0.832299	valid_1's binary_logloss: 0.13953
    [14]	valid_0's auc: 0.873656	valid_0's binary_logloss: 0.12785	valid_1's auc: 0.831874	valid_1's binary_logloss: 0.139277
    [15]	valid_0's auc: 0.875313	valid_0's binary_logloss: 0.127041	valid_1's auc: 0.831922	valid_1's binary_logloss: 0.139006
    [16]	valid_0's auc: 0.875813	valid_0's binary_logloss: 0.126291	valid_1's auc: 0.832339	valid_1's binary_logloss: 0.138679
    [17]	valid_0's auc: 0.877574	valid_0's binary_logloss: 0.125606	valid_1's auc: 0.832459	valid_1's binary_logloss: 0.138444
    [18]	valid_0's auc: 0.879134	valid_0's binary_logloss: 0.12489	valid_1's auc: 0.832409	valid_1's binary_logloss: 0.138268
    [19]	valid_0's auc: 0.880176	valid_0's binary_logloss: 0.124247	valid_1's auc: 0.832557	valid_1's binary_logloss: 0.138164
    [20]	valid_0's auc: 0.881608	valid_0's binary_logloss: 0.123609	valid_1's auc: 0.832053	valid_1's binary_logloss: 0.138136
    [21]	valid_0's auc: 0.882632	valid_0's binary_logloss: 0.123013	valid_1's auc: 0.831852	valid_1's binary_logloss: 0.138081
    [22]	valid_0's auc: 0.883684	valid_0's binary_logloss: 0.12245	valid_1's auc: 0.832034	valid_1's binary_logloss: 0.137978
    [23]	valid_0's auc: 0.885039	valid_0's binary_logloss: 0.121827	valid_1's auc: 0.831422	valid_1's binary_logloss: 0.138011
    [24]	valid_0's auc: 0.886247	valid_0's binary_logloss: 0.121291	valid_1's auc: 0.831253	valid_1's binary_logloss: 0.137942
    [25]	valid_0's auc: 0.887301	valid_0's binary_logloss: 0.120809	valid_1's auc: 0.831276	valid_1's binary_logloss: 0.137924
    [26]	valid_0's auc: 0.888621	valid_0's binary_logloss: 0.120287	valid_1's auc: 0.830735	valid_1's binary_logloss: 0.137975
    [27]	valid_0's auc: 0.889227	valid_0's binary_logloss: 0.119878	valid_1's auc: 0.830569	valid_1's binary_logloss: 0.137976
    [28]	valid_0's auc: 0.890381	valid_0's binary_logloss: 0.119445	valid_1's auc: 0.830281	valid_1's binary_logloss: 0.137968
    [29]	valid_0's auc: 0.891139	valid_0's binary_logloss: 0.119016	valid_1's auc: 0.830916	valid_1's binary_logloss: 0.137839
    [30]	valid_0's auc: 0.892417	valid_0's binary_logloss: 0.118535	valid_1's auc: 0.830189	valid_1's binary_logloss: 0.137909
    [31]	valid_0's auc: 0.893102	valid_0's binary_logloss: 0.118157	valid_1's auc: 0.830117	valid_1's binary_logloss: 0.137933
    [32]	valid_0's auc: 0.893844	valid_0's binary_logloss: 0.117764	valid_1's auc: 0.829877	valid_1's binary_logloss: 0.137962
    [33]	valid_0's auc: 0.894508	valid_0's binary_logloss: 0.117419	valid_1's auc: 0.829552	valid_1's binary_logloss: 0.138044
    [34]	valid_0's auc: 0.895157	valid_0's binary_logloss: 0.117051	valid_1's auc: 0.830486	valid_1's binary_logloss: 0.137893
    [35]	valid_0's auc: 0.89588	valid_0's binary_logloss: 0.116679	valid_1's auc: 0.83041	valid_1's binary_logloss: 0.137896
    [36]	valid_0's auc: 0.896611	valid_0's binary_logloss: 0.116295	valid_1's auc: 0.829969	valid_1's binary_logloss: 0.138026
    [37]	valid_0's auc: 0.89762	valid_0's binary_logloss: 0.115979	valid_1's auc: 0.829802	valid_1's binary_logloss: 0.138089
    [38]	valid_0's auc: 0.898028	valid_0's binary_logloss: 0.115667	valid_1's auc: 0.829346	valid_1's binary_logloss: 0.138207
    [39]	valid_0's auc: 0.89839	valid_0's binary_logloss: 0.115401	valid_1's auc: 0.829208	valid_1's binary_logloss: 0.138191
    [40]	valid_0's auc: 0.898968	valid_0's binary_logloss: 0.115042	valid_1's auc: 0.828576	valid_1's binary_logloss: 0.138317
    [41]	valid_0's auc: 0.899373	valid_0's binary_logloss: 0.114766	valid_1's auc: 0.828382	valid_1's binary_logloss: 0.138408
    [42]	valid_0's auc: 0.900042	valid_0's binary_logloss: 0.114467	valid_1's auc: 0.828023	valid_1's binary_logloss: 0.138489
    [43]	valid_0's auc: 0.900488	valid_0's binary_logloss: 0.114194	valid_1's auc: 0.827815	valid_1's binary_logloss: 0.138591
    [44]	valid_0's auc: 0.901004	valid_0's binary_logloss: 0.113865	valid_1's auc: 0.827321	valid_1's binary_logloss: 0.138736
    [45]	valid_0's auc: 0.901401	valid_0's binary_logloss: 0.113616	valid_1's auc: 0.826875	valid_1's binary_logloss: 0.138874
    [46]	valid_0's auc: 0.901795	valid_0's binary_logloss: 0.113348	valid_1's auc: 0.82643	valid_1's binary_logloss: 0.138955
    [47]	valid_0's auc: 0.902244	valid_0's binary_logloss: 0.11309	valid_1's auc: 0.825825	valid_1's binary_logloss: 0.13908
    [48]	valid_0's auc: 0.902756	valid_0's binary_logloss: 0.112891	valid_1's auc: 0.825639	valid_1's binary_logloss: 0.139117
    [49]	valid_0's auc: 0.903206	valid_0's binary_logloss: 0.112607	valid_1's auc: 0.825607	valid_1's binary_logloss: 0.139184
    Early stopping, best iteration is:
    [19]	valid_0's auc: 0.880176	valid_0's binary_logloss: 0.124247	valid_1's auc: 0.832557	valid_1's binary_logloss: 0.138164
    [1]	valid_0's auc: 0.822874	valid_0's binary_logloss: 0.157449	valid_1's auc: 0.804936	valid_1's binary_logloss: 0.160126
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.827064	valid_0's binary_logloss: 0.152115	valid_1's auc: 0.810262	valid_1's binary_logloss: 0.155403
    [3]	valid_0's auc: 0.830773	valid_0's binary_logloss: 0.148229	valid_1's auc: 0.814099	valid_1's binary_logloss: 0.151956
    [4]	valid_0's auc: 0.834585	valid_0's binary_logloss: 0.145242	valid_1's auc: 0.815291	valid_1's binary_logloss: 0.149379
    [5]	valid_0's auc: 0.839767	valid_0's binary_logloss: 0.142798	valid_1's auc: 0.819992	valid_1's binary_logloss: 0.147278
    [6]	valid_0's auc: 0.845277	valid_0's binary_logloss: 0.140696	valid_1's auc: 0.824997	valid_1's binary_logloss: 0.145614
    [7]	valid_0's auc: 0.84706	valid_0's binary_logloss: 0.139021	valid_1's auc: 0.827429	valid_1's binary_logloss: 0.144095
    [8]	valid_0's auc: 0.847609	valid_0's binary_logloss: 0.137638	valid_1's auc: 0.827987	valid_1's binary_logloss: 0.143002
    [9]	valid_0's auc: 0.848913	valid_0's binary_logloss: 0.136364	valid_1's auc: 0.82986	valid_1's binary_logloss: 0.141984
    [10]	valid_0's auc: 0.850178	valid_0's binary_logloss: 0.135236	valid_1's auc: 0.829651	valid_1's binary_logloss: 0.141228
    [11]	valid_0's auc: 0.85127	valid_0's binary_logloss: 0.134273	valid_1's auc: 0.830496	valid_1's binary_logloss: 0.140509
    [12]	valid_0's auc: 0.852586	valid_0's binary_logloss: 0.133411	valid_1's auc: 0.830525	valid_1's binary_logloss: 0.13997
    [13]	valid_0's auc: 0.853732	valid_0's binary_logloss: 0.132666	valid_1's auc: 0.829319	valid_1's binary_logloss: 0.13957
    [14]	valid_0's auc: 0.854775	valid_0's binary_logloss: 0.13201	valid_1's auc: 0.832116	valid_1's binary_logloss: 0.139106
    [15]	valid_0's auc: 0.855714	valid_0's binary_logloss: 0.131409	valid_1's auc: 0.833454	valid_1's binary_logloss: 0.138684
    [16]	valid_0's auc: 0.856099	valid_0's binary_logloss: 0.130853	valid_1's auc: 0.832243	valid_1's binary_logloss: 0.138485
    [17]	valid_0's auc: 0.857475	valid_0's binary_logloss: 0.130295	valid_1's auc: 0.832577	valid_1's binary_logloss: 0.138179
    [18]	valid_0's auc: 0.857908	valid_0's binary_logloss: 0.129866	valid_1's auc: 0.832652	valid_1's binary_logloss: 0.138084
    [19]	valid_0's auc: 0.859226	valid_0's binary_logloss: 0.129384	valid_1's auc: 0.832887	valid_1's binary_logloss: 0.137828
    [20]	valid_0's auc: 0.860296	valid_0's binary_logloss: 0.129003	valid_1's auc: 0.834111	valid_1's binary_logloss: 0.137643
    [21]	valid_0's auc: 0.860934	valid_0's binary_logloss: 0.128619	valid_1's auc: 0.834602	valid_1's binary_logloss: 0.137471
    [22]	valid_0's auc: 0.861712	valid_0's binary_logloss: 0.12825	valid_1's auc: 0.834828	valid_1's binary_logloss: 0.137291
    [23]	valid_0's auc: 0.862939	valid_0's binary_logloss: 0.127869	valid_1's auc: 0.835767	valid_1's binary_logloss: 0.137067
    [24]	valid_0's auc: 0.863968	valid_0's binary_logloss: 0.127525	valid_1's auc: 0.835485	valid_1's binary_logloss: 0.13701
    [25]	valid_0's auc: 0.865029	valid_0's binary_logloss: 0.127234	valid_1's auc: 0.835507	valid_1's binary_logloss: 0.136911
    [26]	valid_0's auc: 0.865758	valid_0's binary_logloss: 0.126952	valid_1's auc: 0.835743	valid_1's binary_logloss: 0.136823
    [27]	valid_0's auc: 0.866244	valid_0's binary_logloss: 0.126688	valid_1's auc: 0.836049	valid_1's binary_logloss: 0.136736
    [28]	valid_0's auc: 0.866607	valid_0's binary_logloss: 0.126455	valid_1's auc: 0.836043	valid_1's binary_logloss: 0.136687
    [29]	valid_0's auc: 0.867322	valid_0's binary_logloss: 0.126255	valid_1's auc: 0.836375	valid_1's binary_logloss: 0.136596
    [30]	valid_0's auc: 0.867997	valid_0's binary_logloss: 0.126022	valid_1's auc: 0.836518	valid_1's binary_logloss: 0.136563
    [31]	valid_0's auc: 0.868888	valid_0's binary_logloss: 0.125781	valid_1's auc: 0.836949	valid_1's binary_logloss: 0.13648
    [32]	valid_0's auc: 0.869544	valid_0's binary_logloss: 0.125524	valid_1's auc: 0.836774	valid_1's binary_logloss: 0.136506
    [33]	valid_0's auc: 0.870314	valid_0's binary_logloss: 0.125324	valid_1's auc: 0.837039	valid_1's binary_logloss: 0.13642
    [34]	valid_0's auc: 0.871207	valid_0's binary_logloss: 0.125096	valid_1's auc: 0.836859	valid_1's binary_logloss: 0.136441
    [35]	valid_0's auc: 0.871788	valid_0's binary_logloss: 0.124883	valid_1's auc: 0.836641	valid_1's binary_logloss: 0.136467
    [36]	valid_0's auc: 0.872403	valid_0's binary_logloss: 0.124635	valid_1's auc: 0.836917	valid_1's binary_logloss: 0.13642
    [37]	valid_0's auc: 0.872914	valid_0's binary_logloss: 0.124439	valid_1's auc: 0.836892	valid_1's binary_logloss: 0.136489
    [38]	valid_0's auc: 0.873469	valid_0's binary_logloss: 0.124233	valid_1's auc: 0.836816	valid_1's binary_logloss: 0.136502
    [39]	valid_0's auc: 0.873971	valid_0's binary_logloss: 0.124017	valid_1's auc: 0.83659	valid_1's binary_logloss: 0.136543
    [40]	valid_0's auc: 0.875013	valid_0's binary_logloss: 0.123715	valid_1's auc: 0.836381	valid_1's binary_logloss: 0.136593
    [41]	valid_0's auc: 0.875341	valid_0's binary_logloss: 0.123538	valid_1's auc: 0.8363	valid_1's binary_logloss: 0.136624
    [42]	valid_0's auc: 0.875886	valid_0's binary_logloss: 0.123285	valid_1's auc: 0.836043	valid_1's binary_logloss: 0.136665
    [43]	valid_0's auc: 0.87633	valid_0's binary_logloss: 0.12311	valid_1's auc: 0.836018	valid_1's binary_logloss: 0.136647
    [44]	valid_0's auc: 0.876816	valid_0's binary_logloss: 0.122878	valid_1's auc: 0.836061	valid_1's binary_logloss: 0.136664
    [45]	valid_0's auc: 0.877116	valid_0's binary_logloss: 0.12271	valid_1's auc: 0.836517	valid_1's binary_logloss: 0.136608
    [46]	valid_0's auc: 0.877567	valid_0's binary_logloss: 0.122536	valid_1's auc: 0.836051	valid_1's binary_logloss: 0.13667
    [47]	valid_0's auc: 0.878192	valid_0's binary_logloss: 0.122332	valid_1's auc: 0.835827	valid_1's binary_logloss: 0.136694
    [48]	valid_0's auc: 0.878371	valid_0's binary_logloss: 0.122208	valid_1's auc: 0.835831	valid_1's binary_logloss: 0.136717
    [49]	valid_0's auc: 0.878669	valid_0's binary_logloss: 0.12204	valid_1's auc: 0.835913	valid_1's binary_logloss: 0.136679
    [50]	valid_0's auc: 0.879288	valid_0's binary_logloss: 0.121825	valid_1's auc: 0.835468	valid_1's binary_logloss: 0.136734
    [51]	valid_0's auc: 0.879552	valid_0's binary_logloss: 0.121647	valid_1's auc: 0.835095	valid_1's binary_logloss: 0.13684
    [52]	valid_0's auc: 0.880033	valid_0's binary_logloss: 0.121459	valid_1's auc: 0.83496	valid_1's binary_logloss: 0.136927
    [53]	valid_0's auc: 0.880447	valid_0's binary_logloss: 0.121297	valid_1's auc: 0.834955	valid_1's binary_logloss: 0.136953
    [54]	valid_0's auc: 0.88087	valid_0's binary_logloss: 0.121116	valid_1's auc: 0.834926	valid_1's binary_logloss: 0.136994
    [55]	valid_0's auc: 0.881105	valid_0's binary_logloss: 0.12097	valid_1's auc: 0.834861	valid_1's binary_logloss: 0.137043
    [56]	valid_0's auc: 0.88161	valid_0's binary_logloss: 0.120772	valid_1's auc: 0.834629	valid_1's binary_logloss: 0.137125
    [57]	valid_0's auc: 0.882134	valid_0's binary_logloss: 0.120595	valid_1's auc: 0.834576	valid_1's binary_logloss: 0.137157
    [58]	valid_0's auc: 0.882429	valid_0's binary_logloss: 0.120441	valid_1's auc: 0.834319	valid_1's binary_logloss: 0.137209
    [59]	valid_0's auc: 0.882799	valid_0's binary_logloss: 0.120268	valid_1's auc: 0.834028	valid_1's binary_logloss: 0.1373
    [60]	valid_0's auc: 0.883147	valid_0's binary_logloss: 0.120094	valid_1's auc: 0.833999	valid_1's binary_logloss: 0.137339
    [61]	valid_0's auc: 0.883628	valid_0's binary_logloss: 0.119914	valid_1's auc: 0.834234	valid_1's binary_logloss: 0.137314
    [62]	valid_0's auc: 0.884034	valid_0's binary_logloss: 0.119753	valid_1's auc: 0.834179	valid_1's binary_logloss: 0.137356
    [63]	valid_0's auc: 0.884169	valid_0's binary_logloss: 0.119623	valid_1's auc: 0.834035	valid_1's binary_logloss: 0.137378
    Early stopping, best iteration is:
    [33]	valid_0's auc: 0.870314	valid_0's binary_logloss: 0.125324	valid_1's auc: 0.837039	valid_1's binary_logloss: 0.13642
    [1]	valid_0's auc: 0.826066	valid_0's binary_logloss: 0.157509	valid_1's auc: 0.810763	valid_1's binary_logloss: 0.160177
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.830558	valid_0's binary_logloss: 0.152317	valid_1's auc: 0.814406	valid_1's binary_logloss: 0.15553
    [3]	valid_0's auc: 0.833526	valid_0's binary_logloss: 0.148529	valid_1's auc: 0.815841	valid_1's binary_logloss: 0.152134
    [4]	valid_0's auc: 0.839653	valid_0's binary_logloss: 0.145555	valid_1's auc: 0.821767	valid_1's binary_logloss: 0.149519
    [5]	valid_0's auc: 0.842943	valid_0's binary_logloss: 0.143073	valid_1's auc: 0.826334	valid_1's binary_logloss: 0.14736
    [6]	valid_0's auc: 0.845929	valid_0's binary_logloss: 0.141096	valid_1's auc: 0.828882	valid_1's binary_logloss: 0.145649
    [7]	valid_0's auc: 0.847231	valid_0's binary_logloss: 0.139394	valid_1's auc: 0.830621	valid_1's binary_logloss: 0.144201
    [8]	valid_0's auc: 0.849609	valid_0's binary_logloss: 0.137956	valid_1's auc: 0.831324	valid_1's binary_logloss: 0.143105
    [9]	valid_0's auc: 0.851333	valid_0's binary_logloss: 0.136733	valid_1's auc: 0.83377	valid_1's binary_logloss: 0.142018
    [10]	valid_0's auc: 0.852692	valid_0's binary_logloss: 0.135654	valid_1's auc: 0.834493	valid_1's binary_logloss: 0.141149
    [11]	valid_0's auc: 0.854022	valid_0's binary_logloss: 0.134688	valid_1's auc: 0.834616	valid_1's binary_logloss: 0.140437
    [12]	valid_0's auc: 0.854419	valid_0's binary_logloss: 0.13393	valid_1's auc: 0.83589	valid_1's binary_logloss: 0.139765
    [13]	valid_0's auc: 0.854853	valid_0's binary_logloss: 0.133229	valid_1's auc: 0.835999	valid_1's binary_logloss: 0.139284
    [14]	valid_0's auc: 0.856241	valid_0's binary_logloss: 0.132532	valid_1's auc: 0.837392	valid_1's binary_logloss: 0.138795
    [15]	valid_0's auc: 0.85803	valid_0's binary_logloss: 0.131862	valid_1's auc: 0.83818	valid_1's binary_logloss: 0.138366
    [16]	valid_0's auc: 0.859289	valid_0's binary_logloss: 0.131311	valid_1's auc: 0.838523	valid_1's binary_logloss: 0.138137
    [17]	valid_0's auc: 0.860115	valid_0's binary_logloss: 0.130812	valid_1's auc: 0.839359	valid_1's binary_logloss: 0.137816
    [18]	valid_0's auc: 0.860807	valid_0's binary_logloss: 0.130408	valid_1's auc: 0.840179	valid_1's binary_logloss: 0.13749
    [19]	valid_0's auc: 0.861568	valid_0's binary_logloss: 0.129954	valid_1's auc: 0.839855	valid_1's binary_logloss: 0.137387
    [20]	valid_0's auc: 0.863194	valid_0's binary_logloss: 0.129482	valid_1's auc: 0.839715	valid_1's binary_logloss: 0.137226
    [21]	valid_0's auc: 0.864273	valid_0's binary_logloss: 0.12907	valid_1's auc: 0.840516	valid_1's binary_logloss: 0.136991
    [22]	valid_0's auc: 0.865334	valid_0's binary_logloss: 0.128648	valid_1's auc: 0.841302	valid_1's binary_logloss: 0.136769
    [23]	valid_0's auc: 0.866353	valid_0's binary_logloss: 0.128302	valid_1's auc: 0.841085	valid_1's binary_logloss: 0.136653
    [24]	valid_0's auc: 0.86699	valid_0's binary_logloss: 0.127957	valid_1's auc: 0.841235	valid_1's binary_logloss: 0.136611
    [25]	valid_0's auc: 0.867587	valid_0's binary_logloss: 0.127634	valid_1's auc: 0.840948	valid_1's binary_logloss: 0.136644
    [26]	valid_0's auc: 0.868507	valid_0's binary_logloss: 0.127298	valid_1's auc: 0.841033	valid_1's binary_logloss: 0.136526
    [27]	valid_0's auc: 0.869362	valid_0's binary_logloss: 0.126976	valid_1's auc: 0.840939	valid_1's binary_logloss: 0.136445
    [28]	valid_0's auc: 0.870126	valid_0's binary_logloss: 0.126688	valid_1's auc: 0.840452	valid_1's binary_logloss: 0.13645
    [29]	valid_0's auc: 0.870568	valid_0's binary_logloss: 0.126442	valid_1's auc: 0.840809	valid_1's binary_logloss: 0.136345
    [30]	valid_0's auc: 0.871323	valid_0's binary_logloss: 0.126166	valid_1's auc: 0.840599	valid_1's binary_logloss: 0.136334
    [31]	valid_0's auc: 0.872181	valid_0's binary_logloss: 0.125929	valid_1's auc: 0.840663	valid_1's binary_logloss: 0.136267
    [32]	valid_0's auc: 0.873089	valid_0's binary_logloss: 0.125641	valid_1's auc: 0.841679	valid_1's binary_logloss: 0.136141
    [33]	valid_0's auc: 0.873443	valid_0's binary_logloss: 0.125415	valid_1's auc: 0.841597	valid_1's binary_logloss: 0.136138
    [34]	valid_0's auc: 0.874111	valid_0's binary_logloss: 0.125163	valid_1's auc: 0.841194	valid_1's binary_logloss: 0.13614
    [35]	valid_0's auc: 0.874804	valid_0's binary_logloss: 0.124891	valid_1's auc: 0.841136	valid_1's binary_logloss: 0.136214
    [36]	valid_0's auc: 0.875123	valid_0's binary_logloss: 0.124721	valid_1's auc: 0.840883	valid_1's binary_logloss: 0.136268
    [37]	valid_0's auc: 0.875964	valid_0's binary_logloss: 0.124479	valid_1's auc: 0.840327	valid_1's binary_logloss: 0.136363
    [38]	valid_0's auc: 0.876484	valid_0's binary_logloss: 0.12422	valid_1's auc: 0.840317	valid_1's binary_logloss: 0.136378
    [39]	valid_0's auc: 0.877529	valid_0's binary_logloss: 0.123975	valid_1's auc: 0.840224	valid_1's binary_logloss: 0.136372
    [40]	valid_0's auc: 0.877916	valid_0's binary_logloss: 0.123803	valid_1's auc: 0.840099	valid_1's binary_logloss: 0.136418
    [41]	valid_0's auc: 0.878404	valid_0's binary_logloss: 0.123606	valid_1's auc: 0.839447	valid_1's binary_logloss: 0.136498
    [42]	valid_0's auc: 0.878787	valid_0's binary_logloss: 0.123458	valid_1's auc: 0.839737	valid_1's binary_logloss: 0.136466
    [43]	valid_0's auc: 0.879296	valid_0's binary_logloss: 0.123263	valid_1's auc: 0.838848	valid_1's binary_logloss: 0.136611
    [44]	valid_0's auc: 0.879561	valid_0's binary_logloss: 0.123111	valid_1's auc: 0.83939	valid_1's binary_logloss: 0.136509
    [45]	valid_0's auc: 0.880167	valid_0's binary_logloss: 0.122912	valid_1's auc: 0.83916	valid_1's binary_logloss: 0.136564
    [46]	valid_0's auc: 0.880652	valid_0's binary_logloss: 0.12273	valid_1's auc: 0.839133	valid_1's binary_logloss: 0.136588
    [47]	valid_0's auc: 0.881351	valid_0's binary_logloss: 0.122478	valid_1's auc: 0.839108	valid_1's binary_logloss: 0.136599
    [48]	valid_0's auc: 0.881786	valid_0's binary_logloss: 0.122308	valid_1's auc: 0.838894	valid_1's binary_logloss: 0.136636
    [49]	valid_0's auc: 0.88242	valid_0's binary_logloss: 0.122054	valid_1's auc: 0.838476	valid_1's binary_logloss: 0.136727
    [50]	valid_0's auc: 0.88279	valid_0's binary_logloss: 0.121862	valid_1's auc: 0.837735	valid_1's binary_logloss: 0.136833
    [51]	valid_0's auc: 0.883422	valid_0's binary_logloss: 0.121646	valid_1's auc: 0.837691	valid_1's binary_logloss: 0.136833
    [52]	valid_0's auc: 0.884035	valid_0's binary_logloss: 0.121461	valid_1's auc: 0.837503	valid_1's binary_logloss: 0.136856
    [53]	valid_0's auc: 0.884636	valid_0's binary_logloss: 0.121237	valid_1's auc: 0.837159	valid_1's binary_logloss: 0.136933
    [54]	valid_0's auc: 0.885019	valid_0's binary_logloss: 0.121086	valid_1's auc: 0.836758	valid_1's binary_logloss: 0.136995
    [55]	valid_0's auc: 0.88546	valid_0's binary_logloss: 0.120908	valid_1's auc: 0.836692	valid_1's binary_logloss: 0.137006
    [56]	valid_0's auc: 0.885711	valid_0's binary_logloss: 0.120745	valid_1's auc: 0.836703	valid_1's binary_logloss: 0.136998
    [57]	valid_0's auc: 0.886081	valid_0's binary_logloss: 0.120544	valid_1's auc: 0.836462	valid_1's binary_logloss: 0.137073
    [58]	valid_0's auc: 0.886576	valid_0's binary_logloss: 0.120352	valid_1's auc: 0.836241	valid_1's binary_logloss: 0.137127
    [59]	valid_0's auc: 0.886878	valid_0's binary_logloss: 0.120191	valid_1's auc: 0.8358	valid_1's binary_logloss: 0.137207
    [60]	valid_0's auc: 0.887158	valid_0's binary_logloss: 0.120005	valid_1's auc: 0.835688	valid_1's binary_logloss: 0.137272
    [61]	valid_0's auc: 0.88755	valid_0's binary_logloss: 0.119855	valid_1's auc: 0.835652	valid_1's binary_logloss: 0.137262
    [62]	valid_0's auc: 0.887817	valid_0's binary_logloss: 0.11971	valid_1's auc: 0.835355	valid_1's binary_logloss: 0.13732
    Early stopping, best iteration is:
    [32]	valid_0's auc: 0.873089	valid_0's binary_logloss: 0.125641	valid_1's auc: 0.841679	valid_1's binary_logloss: 0.136141
    [1]	valid_0's auc: 0.822527	valid_0's binary_logloss: 0.157702	valid_1's auc: 0.811735	valid_1's binary_logloss: 0.160053
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.828897	valid_0's binary_logloss: 0.152381	valid_1's auc: 0.815279	valid_1's binary_logloss: 0.155308
    [3]	valid_0's auc: 0.833978	valid_0's binary_logloss: 0.148529	valid_1's auc: 0.819863	valid_1's binary_logloss: 0.151959
    [4]	valid_0's auc: 0.837641	valid_0's binary_logloss: 0.145558	valid_1's auc: 0.821032	valid_1's binary_logloss: 0.149309
    [5]	valid_0's auc: 0.841306	valid_0's binary_logloss: 0.143184	valid_1's auc: 0.824926	valid_1's binary_logloss: 0.147249
    [6]	valid_0's auc: 0.843853	valid_0's binary_logloss: 0.141094	valid_1's auc: 0.825446	valid_1's binary_logloss: 0.145583
    [7]	valid_0's auc: 0.846589	valid_0's binary_logloss: 0.13939	valid_1's auc: 0.829069	valid_1's binary_logloss: 0.144172
    [8]	valid_0's auc: 0.847805	valid_0's binary_logloss: 0.137988	valid_1's auc: 0.829457	valid_1's binary_logloss: 0.143004
    [9]	valid_0's auc: 0.850007	valid_0's binary_logloss: 0.136683	valid_1's auc: 0.831298	valid_1's binary_logloss: 0.141959
    [10]	valid_0's auc: 0.850514	valid_0's binary_logloss: 0.135597	valid_1's auc: 0.830745	valid_1's binary_logloss: 0.141234
    [11]	valid_0's auc: 0.851694	valid_0's binary_logloss: 0.134644	valid_1's auc: 0.831218	valid_1's binary_logloss: 0.140531
    [12]	valid_0's auc: 0.854647	valid_0's binary_logloss: 0.133779	valid_1's auc: 0.832445	valid_1's binary_logloss: 0.140011
    [13]	valid_0's auc: 0.8552	valid_0's binary_logloss: 0.13303	valid_1's auc: 0.832525	valid_1's binary_logloss: 0.139542
    [14]	valid_0's auc: 0.85611	valid_0's binary_logloss: 0.132355	valid_1's auc: 0.832559	valid_1's binary_logloss: 0.139125
    [15]	valid_0's auc: 0.856886	valid_0's binary_logloss: 0.131779	valid_1's auc: 0.833401	valid_1's binary_logloss: 0.138708
    [16]	valid_0's auc: 0.857397	valid_0's binary_logloss: 0.13126	valid_1's auc: 0.833709	valid_1's binary_logloss: 0.138376
    [17]	valid_0's auc: 0.859038	valid_0's binary_logloss: 0.130686	valid_1's auc: 0.83451	valid_1's binary_logloss: 0.138079
    [18]	valid_0's auc: 0.859754	valid_0's binary_logloss: 0.130225	valid_1's auc: 0.834886	valid_1's binary_logloss: 0.137839
    [19]	valid_0's auc: 0.861354	valid_0's binary_logloss: 0.129689	valid_1's auc: 0.835687	valid_1's binary_logloss: 0.137572
    [20]	valid_0's auc: 0.862114	valid_0's binary_logloss: 0.129263	valid_1's auc: 0.83587	valid_1's binary_logloss: 0.137393
    [21]	valid_0's auc: 0.863196	valid_0's binary_logloss: 0.128834	valid_1's auc: 0.835611	valid_1's binary_logloss: 0.137244
    [22]	valid_0's auc: 0.864467	valid_0's binary_logloss: 0.128421	valid_1's auc: 0.836013	valid_1's binary_logloss: 0.137127
    [23]	valid_0's auc: 0.865716	valid_0's binary_logloss: 0.12802	valid_1's auc: 0.835915	valid_1's binary_logloss: 0.13701
    [24]	valid_0's auc: 0.866358	valid_0's binary_logloss: 0.127649	valid_1's auc: 0.835345	valid_1's binary_logloss: 0.136983
    [25]	valid_0's auc: 0.867017	valid_0's binary_logloss: 0.127326	valid_1's auc: 0.835203	valid_1's binary_logloss: 0.13697
    [26]	valid_0's auc: 0.867793	valid_0's binary_logloss: 0.126998	valid_1's auc: 0.835068	valid_1's binary_logloss: 0.13692
    [27]	valid_0's auc: 0.86892	valid_0's binary_logloss: 0.126634	valid_1's auc: 0.835127	valid_1's binary_logloss: 0.136839
    [28]	valid_0's auc: 0.869611	valid_0's binary_logloss: 0.126332	valid_1's auc: 0.835011	valid_1's binary_logloss: 0.136765
    [29]	valid_0's auc: 0.870502	valid_0's binary_logloss: 0.126032	valid_1's auc: 0.835017	valid_1's binary_logloss: 0.136691
    [30]	valid_0's auc: 0.87109	valid_0's binary_logloss: 0.125752	valid_1's auc: 0.835346	valid_1's binary_logloss: 0.136657
    [31]	valid_0's auc: 0.871834	valid_0's binary_logloss: 0.125492	valid_1's auc: 0.835469	valid_1's binary_logloss: 0.136553
    [32]	valid_0's auc: 0.8727	valid_0's binary_logloss: 0.125224	valid_1's auc: 0.834809	valid_1's binary_logloss: 0.136626
    [33]	valid_0's auc: 0.873327	valid_0's binary_logloss: 0.12497	valid_1's auc: 0.834899	valid_1's binary_logloss: 0.136576
    [34]	valid_0's auc: 0.873759	valid_0's binary_logloss: 0.124747	valid_1's auc: 0.835299	valid_1's binary_logloss: 0.136478
    [35]	valid_0's auc: 0.874436	valid_0's binary_logloss: 0.124498	valid_1's auc: 0.834852	valid_1's binary_logloss: 0.136556
    [36]	valid_0's auc: 0.875262	valid_0's binary_logloss: 0.124237	valid_1's auc: 0.834479	valid_1's binary_logloss: 0.13659
    [37]	valid_0's auc: 0.875938	valid_0's binary_logloss: 0.123998	valid_1's auc: 0.834359	valid_1's binary_logloss: 0.136644
    [38]	valid_0's auc: 0.876343	valid_0's binary_logloss: 0.123795	valid_1's auc: 0.834106	valid_1's binary_logloss: 0.136687
    [39]	valid_0's auc: 0.877139	valid_0's binary_logloss: 0.123529	valid_1's auc: 0.83359	valid_1's binary_logloss: 0.136724
    [40]	valid_0's auc: 0.877911	valid_0's binary_logloss: 0.123326	valid_1's auc: 0.833512	valid_1's binary_logloss: 0.136737
    [41]	valid_0's auc: 0.87875	valid_0's binary_logloss: 0.123086	valid_1's auc: 0.833202	valid_1's binary_logloss: 0.136752
    [42]	valid_0's auc: 0.879342	valid_0's binary_logloss: 0.122885	valid_1's auc: 0.832649	valid_1's binary_logloss: 0.136842
    [43]	valid_0's auc: 0.879629	valid_0's binary_logloss: 0.122717	valid_1's auc: 0.832179	valid_1's binary_logloss: 0.136927
    [44]	valid_0's auc: 0.880215	valid_0's binary_logloss: 0.122487	valid_1's auc: 0.832172	valid_1's binary_logloss: 0.13695
    [45]	valid_0's auc: 0.880851	valid_0's binary_logloss: 0.122245	valid_1's auc: 0.831665	valid_1's binary_logloss: 0.137053
    [46]	valid_0's auc: 0.881492	valid_0's binary_logloss: 0.121994	valid_1's auc: 0.83185	valid_1's binary_logloss: 0.137071
    [47]	valid_0's auc: 0.881818	valid_0's binary_logloss: 0.121836	valid_1's auc: 0.831946	valid_1's binary_logloss: 0.137043
    [48]	valid_0's auc: 0.882136	valid_0's binary_logloss: 0.121683	valid_1's auc: 0.832078	valid_1's binary_logloss: 0.137058
    [49]	valid_0's auc: 0.882509	valid_0's binary_logloss: 0.121512	valid_1's auc: 0.832046	valid_1's binary_logloss: 0.13711
    [50]	valid_0's auc: 0.882872	valid_0's binary_logloss: 0.121296	valid_1's auc: 0.83209	valid_1's binary_logloss: 0.137124
    [51]	valid_0's auc: 0.883328	valid_0's binary_logloss: 0.121099	valid_1's auc: 0.831958	valid_1's binary_logloss: 0.137201
    [52]	valid_0's auc: 0.883722	valid_0's binary_logloss: 0.120897	valid_1's auc: 0.831752	valid_1's binary_logloss: 0.13722
    Early stopping, best iteration is:
    [22]	valid_0's auc: 0.864467	valid_0's binary_logloss: 0.128421	valid_1's auc: 0.836013	valid_1's binary_logloss: 0.137127
    [1]	valid_0's auc: 0.822874	valid_0's binary_logloss: 0.157449	valid_1's auc: 0.804936	valid_1's binary_logloss: 0.160126
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.827064	valid_0's binary_logloss: 0.152115	valid_1's auc: 0.810262	valid_1's binary_logloss: 0.155403
    [3]	valid_0's auc: 0.830773	valid_0's binary_logloss: 0.148229	valid_1's auc: 0.814099	valid_1's binary_logloss: 0.151956
    [4]	valid_0's auc: 0.834585	valid_0's binary_logloss: 0.145242	valid_1's auc: 0.815291	valid_1's binary_logloss: 0.149379
    [5]	valid_0's auc: 0.839767	valid_0's binary_logloss: 0.142798	valid_1's auc: 0.819992	valid_1's binary_logloss: 0.147278
    [6]	valid_0's auc: 0.845277	valid_0's binary_logloss: 0.140696	valid_1's auc: 0.824997	valid_1's binary_logloss: 0.145614
    [7]	valid_0's auc: 0.84706	valid_0's binary_logloss: 0.139021	valid_1's auc: 0.827429	valid_1's binary_logloss: 0.144095
    [8]	valid_0's auc: 0.847609	valid_0's binary_logloss: 0.137638	valid_1's auc: 0.827987	valid_1's binary_logloss: 0.143002
    [9]	valid_0's auc: 0.848913	valid_0's binary_logloss: 0.136364	valid_1's auc: 0.82986	valid_1's binary_logloss: 0.141984
    [10]	valid_0's auc: 0.850178	valid_0's binary_logloss: 0.135236	valid_1's auc: 0.829651	valid_1's binary_logloss: 0.141228
    [11]	valid_0's auc: 0.85127	valid_0's binary_logloss: 0.134273	valid_1's auc: 0.830496	valid_1's binary_logloss: 0.140509
    [12]	valid_0's auc: 0.852586	valid_0's binary_logloss: 0.133411	valid_1's auc: 0.830525	valid_1's binary_logloss: 0.13997
    [13]	valid_0's auc: 0.853732	valid_0's binary_logloss: 0.132666	valid_1's auc: 0.829319	valid_1's binary_logloss: 0.13957
    [14]	valid_0's auc: 0.854775	valid_0's binary_logloss: 0.13201	valid_1's auc: 0.832116	valid_1's binary_logloss: 0.139106
    [15]	valid_0's auc: 0.855714	valid_0's binary_logloss: 0.131409	valid_1's auc: 0.833454	valid_1's binary_logloss: 0.138684
    [16]	valid_0's auc: 0.856099	valid_0's binary_logloss: 0.130853	valid_1's auc: 0.832243	valid_1's binary_logloss: 0.138485
    [17]	valid_0's auc: 0.857475	valid_0's binary_logloss: 0.130295	valid_1's auc: 0.832577	valid_1's binary_logloss: 0.138179
    [18]	valid_0's auc: 0.857908	valid_0's binary_logloss: 0.129866	valid_1's auc: 0.832652	valid_1's binary_logloss: 0.138084
    [19]	valid_0's auc: 0.859226	valid_0's binary_logloss: 0.129384	valid_1's auc: 0.832887	valid_1's binary_logloss: 0.137828
    [20]	valid_0's auc: 0.860296	valid_0's binary_logloss: 0.129003	valid_1's auc: 0.834111	valid_1's binary_logloss: 0.137643
    [21]	valid_0's auc: 0.860934	valid_0's binary_logloss: 0.128619	valid_1's auc: 0.834602	valid_1's binary_logloss: 0.137471
    [22]	valid_0's auc: 0.861712	valid_0's binary_logloss: 0.12825	valid_1's auc: 0.834828	valid_1's binary_logloss: 0.137291
    [23]	valid_0's auc: 0.862939	valid_0's binary_logloss: 0.127869	valid_1's auc: 0.835767	valid_1's binary_logloss: 0.137067
    [24]	valid_0's auc: 0.863968	valid_0's binary_logloss: 0.127525	valid_1's auc: 0.835485	valid_1's binary_logloss: 0.13701
    [25]	valid_0's auc: 0.865029	valid_0's binary_logloss: 0.127234	valid_1's auc: 0.835507	valid_1's binary_logloss: 0.136911
    [26]	valid_0's auc: 0.865758	valid_0's binary_logloss: 0.126952	valid_1's auc: 0.835743	valid_1's binary_logloss: 0.136823
    [27]	valid_0's auc: 0.866244	valid_0's binary_logloss: 0.126688	valid_1's auc: 0.836049	valid_1's binary_logloss: 0.136736
    [28]	valid_0's auc: 0.866607	valid_0's binary_logloss: 0.126455	valid_1's auc: 0.836043	valid_1's binary_logloss: 0.136687
    [29]	valid_0's auc: 0.867322	valid_0's binary_logloss: 0.126255	valid_1's auc: 0.836375	valid_1's binary_logloss: 0.136596
    [30]	valid_0's auc: 0.867997	valid_0's binary_logloss: 0.126022	valid_1's auc: 0.836518	valid_1's binary_logloss: 0.136563
    [31]	valid_0's auc: 0.868888	valid_0's binary_logloss: 0.125781	valid_1's auc: 0.836949	valid_1's binary_logloss: 0.13648
    [32]	valid_0's auc: 0.869544	valid_0's binary_logloss: 0.125524	valid_1's auc: 0.836774	valid_1's binary_logloss: 0.136506
    [33]	valid_0's auc: 0.870314	valid_0's binary_logloss: 0.125324	valid_1's auc: 0.837039	valid_1's binary_logloss: 0.13642
    [34]	valid_0's auc: 0.871207	valid_0's binary_logloss: 0.125096	valid_1's auc: 0.836859	valid_1's binary_logloss: 0.136441
    [35]	valid_0's auc: 0.871788	valid_0's binary_logloss: 0.124883	valid_1's auc: 0.836641	valid_1's binary_logloss: 0.136467
    [36]	valid_0's auc: 0.872403	valid_0's binary_logloss: 0.124635	valid_1's auc: 0.836917	valid_1's binary_logloss: 0.13642
    [37]	valid_0's auc: 0.872914	valid_0's binary_logloss: 0.124439	valid_1's auc: 0.836892	valid_1's binary_logloss: 0.136489
    [38]	valid_0's auc: 0.873469	valid_0's binary_logloss: 0.124233	valid_1's auc: 0.836816	valid_1's binary_logloss: 0.136502
    [39]	valid_0's auc: 0.873971	valid_0's binary_logloss: 0.124017	valid_1's auc: 0.83659	valid_1's binary_logloss: 0.136543
    [40]	valid_0's auc: 0.875013	valid_0's binary_logloss: 0.123715	valid_1's auc: 0.836381	valid_1's binary_logloss: 0.136593
    [41]	valid_0's auc: 0.875341	valid_0's binary_logloss: 0.123538	valid_1's auc: 0.8363	valid_1's binary_logloss: 0.136624
    [42]	valid_0's auc: 0.875886	valid_0's binary_logloss: 0.123285	valid_1's auc: 0.836043	valid_1's binary_logloss: 0.136665
    [43]	valid_0's auc: 0.87633	valid_0's binary_logloss: 0.12311	valid_1's auc: 0.836018	valid_1's binary_logloss: 0.136647
    [44]	valid_0's auc: 0.876816	valid_0's binary_logloss: 0.122878	valid_1's auc: 0.836061	valid_1's binary_logloss: 0.136664
    [45]	valid_0's auc: 0.877116	valid_0's binary_logloss: 0.12271	valid_1's auc: 0.836517	valid_1's binary_logloss: 0.136608
    [46]	valid_0's auc: 0.877567	valid_0's binary_logloss: 0.122536	valid_1's auc: 0.836051	valid_1's binary_logloss: 0.13667
    [47]	valid_0's auc: 0.878192	valid_0's binary_logloss: 0.122332	valid_1's auc: 0.835827	valid_1's binary_logloss: 0.136694
    [48]	valid_0's auc: 0.878371	valid_0's binary_logloss: 0.122208	valid_1's auc: 0.835831	valid_1's binary_logloss: 0.136717
    [49]	valid_0's auc: 0.878669	valid_0's binary_logloss: 0.12204	valid_1's auc: 0.835913	valid_1's binary_logloss: 0.136679
    [50]	valid_0's auc: 0.879288	valid_0's binary_logloss: 0.121825	valid_1's auc: 0.835468	valid_1's binary_logloss: 0.136734
    [51]	valid_0's auc: 0.879552	valid_0's binary_logloss: 0.121647	valid_1's auc: 0.835095	valid_1's binary_logloss: 0.13684
    [52]	valid_0's auc: 0.880033	valid_0's binary_logloss: 0.121459	valid_1's auc: 0.83496	valid_1's binary_logloss: 0.136927
    [53]	valid_0's auc: 0.880447	valid_0's binary_logloss: 0.121297	valid_1's auc: 0.834955	valid_1's binary_logloss: 0.136953
    [54]	valid_0's auc: 0.88087	valid_0's binary_logloss: 0.121116	valid_1's auc: 0.834926	valid_1's binary_logloss: 0.136994
    [55]	valid_0's auc: 0.881105	valid_0's binary_logloss: 0.12097	valid_1's auc: 0.834861	valid_1's binary_logloss: 0.137043
    [56]	valid_0's auc: 0.88161	valid_0's binary_logloss: 0.120772	valid_1's auc: 0.834629	valid_1's binary_logloss: 0.137125
    [57]	valid_0's auc: 0.882134	valid_0's binary_logloss: 0.120595	valid_1's auc: 0.834576	valid_1's binary_logloss: 0.137157
    [58]	valid_0's auc: 0.882429	valid_0's binary_logloss: 0.120441	valid_1's auc: 0.834319	valid_1's binary_logloss: 0.137209
    [59]	valid_0's auc: 0.882799	valid_0's binary_logloss: 0.120268	valid_1's auc: 0.834028	valid_1's binary_logloss: 0.1373
    [60]	valid_0's auc: 0.883147	valid_0's binary_logloss: 0.120094	valid_1's auc: 0.833999	valid_1's binary_logloss: 0.137339
    [61]	valid_0's auc: 0.883628	valid_0's binary_logloss: 0.119914	valid_1's auc: 0.834234	valid_1's binary_logloss: 0.137314
    [62]	valid_0's auc: 0.884034	valid_0's binary_logloss: 0.119753	valid_1's auc: 0.834179	valid_1's binary_logloss: 0.137356
    [63]	valid_0's auc: 0.884169	valid_0's binary_logloss: 0.119623	valid_1's auc: 0.834035	valid_1's binary_logloss: 0.137378
    Early stopping, best iteration is:
    [33]	valid_0's auc: 0.870314	valid_0's binary_logloss: 0.125324	valid_1's auc: 0.837039	valid_1's binary_logloss: 0.13642
    [1]	valid_0's auc: 0.826066	valid_0's binary_logloss: 0.157509	valid_1's auc: 0.810763	valid_1's binary_logloss: 0.160177
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.830558	valid_0's binary_logloss: 0.152317	valid_1's auc: 0.814406	valid_1's binary_logloss: 0.15553
    [3]	valid_0's auc: 0.833526	valid_0's binary_logloss: 0.148529	valid_1's auc: 0.815841	valid_1's binary_logloss: 0.152134
    [4]	valid_0's auc: 0.839653	valid_0's binary_logloss: 0.145555	valid_1's auc: 0.821767	valid_1's binary_logloss: 0.149519
    [5]	valid_0's auc: 0.842943	valid_0's binary_logloss: 0.143073	valid_1's auc: 0.826334	valid_1's binary_logloss: 0.14736
    [6]	valid_0's auc: 0.845929	valid_0's binary_logloss: 0.141096	valid_1's auc: 0.828882	valid_1's binary_logloss: 0.145649
    [7]	valid_0's auc: 0.847231	valid_0's binary_logloss: 0.139394	valid_1's auc: 0.830621	valid_1's binary_logloss: 0.144201
    [8]	valid_0's auc: 0.849609	valid_0's binary_logloss: 0.137956	valid_1's auc: 0.831324	valid_1's binary_logloss: 0.143105
    [9]	valid_0's auc: 0.851333	valid_0's binary_logloss: 0.136733	valid_1's auc: 0.83377	valid_1's binary_logloss: 0.142018
    [10]	valid_0's auc: 0.852692	valid_0's binary_logloss: 0.135654	valid_1's auc: 0.834493	valid_1's binary_logloss: 0.141149
    [11]	valid_0's auc: 0.854022	valid_0's binary_logloss: 0.134688	valid_1's auc: 0.834616	valid_1's binary_logloss: 0.140437
    [12]	valid_0's auc: 0.854419	valid_0's binary_logloss: 0.13393	valid_1's auc: 0.83589	valid_1's binary_logloss: 0.139765
    [13]	valid_0's auc: 0.854853	valid_0's binary_logloss: 0.133229	valid_1's auc: 0.835999	valid_1's binary_logloss: 0.139284
    [14]	valid_0's auc: 0.856241	valid_0's binary_logloss: 0.132532	valid_1's auc: 0.837392	valid_1's binary_logloss: 0.138795
    [15]	valid_0's auc: 0.85803	valid_0's binary_logloss: 0.131862	valid_1's auc: 0.83818	valid_1's binary_logloss: 0.138366
    [16]	valid_0's auc: 0.859289	valid_0's binary_logloss: 0.131311	valid_1's auc: 0.838523	valid_1's binary_logloss: 0.138137
    [17]	valid_0's auc: 0.860115	valid_0's binary_logloss: 0.130812	valid_1's auc: 0.839359	valid_1's binary_logloss: 0.137816
    [18]	valid_0's auc: 0.860807	valid_0's binary_logloss: 0.130408	valid_1's auc: 0.840179	valid_1's binary_logloss: 0.13749
    [19]	valid_0's auc: 0.861568	valid_0's binary_logloss: 0.129954	valid_1's auc: 0.839855	valid_1's binary_logloss: 0.137387
    [20]	valid_0's auc: 0.863194	valid_0's binary_logloss: 0.129482	valid_1's auc: 0.839715	valid_1's binary_logloss: 0.137226
    [21]	valid_0's auc: 0.864273	valid_0's binary_logloss: 0.12907	valid_1's auc: 0.840516	valid_1's binary_logloss: 0.136991
    [22]	valid_0's auc: 0.865334	valid_0's binary_logloss: 0.128648	valid_1's auc: 0.841302	valid_1's binary_logloss: 0.136769
    [23]	valid_0's auc: 0.866353	valid_0's binary_logloss: 0.128302	valid_1's auc: 0.841085	valid_1's binary_logloss: 0.136653
    [24]	valid_0's auc: 0.86699	valid_0's binary_logloss: 0.127957	valid_1's auc: 0.841235	valid_1's binary_logloss: 0.136611
    [25]	valid_0's auc: 0.867587	valid_0's binary_logloss: 0.127634	valid_1's auc: 0.840948	valid_1's binary_logloss: 0.136644
    [26]	valid_0's auc: 0.868507	valid_0's binary_logloss: 0.127298	valid_1's auc: 0.841033	valid_1's binary_logloss: 0.136526
    [27]	valid_0's auc: 0.869362	valid_0's binary_logloss: 0.126976	valid_1's auc: 0.840939	valid_1's binary_logloss: 0.136445
    [28]	valid_0's auc: 0.870126	valid_0's binary_logloss: 0.126688	valid_1's auc: 0.840452	valid_1's binary_logloss: 0.13645
    [29]	valid_0's auc: 0.870568	valid_0's binary_logloss: 0.126442	valid_1's auc: 0.840809	valid_1's binary_logloss: 0.136345
    [30]	valid_0's auc: 0.871323	valid_0's binary_logloss: 0.126166	valid_1's auc: 0.840599	valid_1's binary_logloss: 0.136334
    [31]	valid_0's auc: 0.872181	valid_0's binary_logloss: 0.125929	valid_1's auc: 0.840663	valid_1's binary_logloss: 0.136267
    [32]	valid_0's auc: 0.873089	valid_0's binary_logloss: 0.125641	valid_1's auc: 0.841679	valid_1's binary_logloss: 0.136141
    [33]	valid_0's auc: 0.873443	valid_0's binary_logloss: 0.125415	valid_1's auc: 0.841597	valid_1's binary_logloss: 0.136138
    [34]	valid_0's auc: 0.874111	valid_0's binary_logloss: 0.125163	valid_1's auc: 0.841194	valid_1's binary_logloss: 0.13614
    [35]	valid_0's auc: 0.874804	valid_0's binary_logloss: 0.124891	valid_1's auc: 0.841136	valid_1's binary_logloss: 0.136214
    [36]	valid_0's auc: 0.875123	valid_0's binary_logloss: 0.124721	valid_1's auc: 0.840883	valid_1's binary_logloss: 0.136268
    [37]	valid_0's auc: 0.875964	valid_0's binary_logloss: 0.124479	valid_1's auc: 0.840327	valid_1's binary_logloss: 0.136363
    [38]	valid_0's auc: 0.876484	valid_0's binary_logloss: 0.12422	valid_1's auc: 0.840317	valid_1's binary_logloss: 0.136378
    [39]	valid_0's auc: 0.877529	valid_0's binary_logloss: 0.123975	valid_1's auc: 0.840224	valid_1's binary_logloss: 0.136372
    [40]	valid_0's auc: 0.877916	valid_0's binary_logloss: 0.123803	valid_1's auc: 0.840099	valid_1's binary_logloss: 0.136418
    [41]	valid_0's auc: 0.878404	valid_0's binary_logloss: 0.123606	valid_1's auc: 0.839447	valid_1's binary_logloss: 0.136498
    [42]	valid_0's auc: 0.878787	valid_0's binary_logloss: 0.123458	valid_1's auc: 0.839737	valid_1's binary_logloss: 0.136466
    [43]	valid_0's auc: 0.879296	valid_0's binary_logloss: 0.123263	valid_1's auc: 0.838848	valid_1's binary_logloss: 0.136611
    [44]	valid_0's auc: 0.879561	valid_0's binary_logloss: 0.123111	valid_1's auc: 0.83939	valid_1's binary_logloss: 0.136509
    [45]	valid_0's auc: 0.880167	valid_0's binary_logloss: 0.122912	valid_1's auc: 0.83916	valid_1's binary_logloss: 0.136564
    [46]	valid_0's auc: 0.880652	valid_0's binary_logloss: 0.12273	valid_1's auc: 0.839133	valid_1's binary_logloss: 0.136588
    [47]	valid_0's auc: 0.881351	valid_0's binary_logloss: 0.122478	valid_1's auc: 0.839108	valid_1's binary_logloss: 0.136599
    [48]	valid_0's auc: 0.881786	valid_0's binary_logloss: 0.122308	valid_1's auc: 0.838894	valid_1's binary_logloss: 0.136636
    [49]	valid_0's auc: 0.88242	valid_0's binary_logloss: 0.122054	valid_1's auc: 0.838476	valid_1's binary_logloss: 0.136727
    [50]	valid_0's auc: 0.88279	valid_0's binary_logloss: 0.121862	valid_1's auc: 0.837735	valid_1's binary_logloss: 0.136833
    [51]	valid_0's auc: 0.883422	valid_0's binary_logloss: 0.121646	valid_1's auc: 0.837691	valid_1's binary_logloss: 0.136833
    [52]	valid_0's auc: 0.884035	valid_0's binary_logloss: 0.121461	valid_1's auc: 0.837503	valid_1's binary_logloss: 0.136856
    [53]	valid_0's auc: 0.884636	valid_0's binary_logloss: 0.121237	valid_1's auc: 0.837159	valid_1's binary_logloss: 0.136933
    [54]	valid_0's auc: 0.885019	valid_0's binary_logloss: 0.121086	valid_1's auc: 0.836758	valid_1's binary_logloss: 0.136995
    [55]	valid_0's auc: 0.88546	valid_0's binary_logloss: 0.120908	valid_1's auc: 0.836692	valid_1's binary_logloss: 0.137006
    [56]	valid_0's auc: 0.885711	valid_0's binary_logloss: 0.120745	valid_1's auc: 0.836703	valid_1's binary_logloss: 0.136998
    [57]	valid_0's auc: 0.886081	valid_0's binary_logloss: 0.120544	valid_1's auc: 0.836462	valid_1's binary_logloss: 0.137073
    [58]	valid_0's auc: 0.886576	valid_0's binary_logloss: 0.120352	valid_1's auc: 0.836241	valid_1's binary_logloss: 0.137127
    [59]	valid_0's auc: 0.886878	valid_0's binary_logloss: 0.120191	valid_1's auc: 0.8358	valid_1's binary_logloss: 0.137207
    [60]	valid_0's auc: 0.887158	valid_0's binary_logloss: 0.120005	valid_1's auc: 0.835688	valid_1's binary_logloss: 0.137272
    [61]	valid_0's auc: 0.88755	valid_0's binary_logloss: 0.119855	valid_1's auc: 0.835652	valid_1's binary_logloss: 0.137262
    [62]	valid_0's auc: 0.887817	valid_0's binary_logloss: 0.11971	valid_1's auc: 0.835355	valid_1's binary_logloss: 0.13732
    Early stopping, best iteration is:
    [32]	valid_0's auc: 0.873089	valid_0's binary_logloss: 0.125641	valid_1's auc: 0.841679	valid_1's binary_logloss: 0.136141
    [1]	valid_0's auc: 0.822527	valid_0's binary_logloss: 0.157702	valid_1's auc: 0.811735	valid_1's binary_logloss: 0.160053
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.828897	valid_0's binary_logloss: 0.152381	valid_1's auc: 0.815279	valid_1's binary_logloss: 0.155308
    [3]	valid_0's auc: 0.833978	valid_0's binary_logloss: 0.148529	valid_1's auc: 0.819863	valid_1's binary_logloss: 0.151959
    [4]	valid_0's auc: 0.837641	valid_0's binary_logloss: 0.145558	valid_1's auc: 0.821032	valid_1's binary_logloss: 0.149309
    [5]	valid_0's auc: 0.841306	valid_0's binary_logloss: 0.143184	valid_1's auc: 0.824926	valid_1's binary_logloss: 0.147249
    [6]	valid_0's auc: 0.843853	valid_0's binary_logloss: 0.141094	valid_1's auc: 0.825446	valid_1's binary_logloss: 0.145583
    [7]	valid_0's auc: 0.846589	valid_0's binary_logloss: 0.13939	valid_1's auc: 0.829069	valid_1's binary_logloss: 0.144172
    [8]	valid_0's auc: 0.847805	valid_0's binary_logloss: 0.137988	valid_1's auc: 0.829457	valid_1's binary_logloss: 0.143004
    [9]	valid_0's auc: 0.850007	valid_0's binary_logloss: 0.136683	valid_1's auc: 0.831298	valid_1's binary_logloss: 0.141959
    [10]	valid_0's auc: 0.850514	valid_0's binary_logloss: 0.135597	valid_1's auc: 0.830745	valid_1's binary_logloss: 0.141234
    [11]	valid_0's auc: 0.851694	valid_0's binary_logloss: 0.134644	valid_1's auc: 0.831218	valid_1's binary_logloss: 0.140531
    [12]	valid_0's auc: 0.854647	valid_0's binary_logloss: 0.133779	valid_1's auc: 0.832445	valid_1's binary_logloss: 0.140011
    [13]	valid_0's auc: 0.8552	valid_0's binary_logloss: 0.13303	valid_1's auc: 0.832525	valid_1's binary_logloss: 0.139542
    [14]	valid_0's auc: 0.85611	valid_0's binary_logloss: 0.132355	valid_1's auc: 0.832559	valid_1's binary_logloss: 0.139125
    [15]	valid_0's auc: 0.856886	valid_0's binary_logloss: 0.131779	valid_1's auc: 0.833401	valid_1's binary_logloss: 0.138708
    [16]	valid_0's auc: 0.857397	valid_0's binary_logloss: 0.13126	valid_1's auc: 0.833709	valid_1's binary_logloss: 0.138376
    [17]	valid_0's auc: 0.859038	valid_0's binary_logloss: 0.130686	valid_1's auc: 0.83451	valid_1's binary_logloss: 0.138079
    [18]	valid_0's auc: 0.859754	valid_0's binary_logloss: 0.130225	valid_1's auc: 0.834886	valid_1's binary_logloss: 0.137839
    [19]	valid_0's auc: 0.861354	valid_0's binary_logloss: 0.129689	valid_1's auc: 0.835687	valid_1's binary_logloss: 0.137572
    [20]	valid_0's auc: 0.862114	valid_0's binary_logloss: 0.129263	valid_1's auc: 0.83587	valid_1's binary_logloss: 0.137393
    [21]	valid_0's auc: 0.863196	valid_0's binary_logloss: 0.128834	valid_1's auc: 0.835611	valid_1's binary_logloss: 0.137244
    [22]	valid_0's auc: 0.864467	valid_0's binary_logloss: 0.128421	valid_1's auc: 0.836013	valid_1's binary_logloss: 0.137127
    [23]	valid_0's auc: 0.865716	valid_0's binary_logloss: 0.12802	valid_1's auc: 0.835915	valid_1's binary_logloss: 0.13701
    [24]	valid_0's auc: 0.866358	valid_0's binary_logloss: 0.127649	valid_1's auc: 0.835345	valid_1's binary_logloss: 0.136983
    [25]	valid_0's auc: 0.867017	valid_0's binary_logloss: 0.127326	valid_1's auc: 0.835203	valid_1's binary_logloss: 0.13697
    [26]	valid_0's auc: 0.867793	valid_0's binary_logloss: 0.126998	valid_1's auc: 0.835068	valid_1's binary_logloss: 0.13692
    [27]	valid_0's auc: 0.86892	valid_0's binary_logloss: 0.126634	valid_1's auc: 0.835127	valid_1's binary_logloss: 0.136839
    [28]	valid_0's auc: 0.869611	valid_0's binary_logloss: 0.126332	valid_1's auc: 0.835011	valid_1's binary_logloss: 0.136765
    [29]	valid_0's auc: 0.870502	valid_0's binary_logloss: 0.126032	valid_1's auc: 0.835017	valid_1's binary_logloss: 0.136691
    [30]	valid_0's auc: 0.87109	valid_0's binary_logloss: 0.125752	valid_1's auc: 0.835346	valid_1's binary_logloss: 0.136657
    [31]	valid_0's auc: 0.871834	valid_0's binary_logloss: 0.125492	valid_1's auc: 0.835469	valid_1's binary_logloss: 0.136553
    [32]	valid_0's auc: 0.8727	valid_0's binary_logloss: 0.125224	valid_1's auc: 0.834809	valid_1's binary_logloss: 0.136626
    [33]	valid_0's auc: 0.873327	valid_0's binary_logloss: 0.12497	valid_1's auc: 0.834899	valid_1's binary_logloss: 0.136576
    [34]	valid_0's auc: 0.873759	valid_0's binary_logloss: 0.124747	valid_1's auc: 0.835299	valid_1's binary_logloss: 0.136478
    [35]	valid_0's auc: 0.874436	valid_0's binary_logloss: 0.124498	valid_1's auc: 0.834852	valid_1's binary_logloss: 0.136556
    [36]	valid_0's auc: 0.875262	valid_0's binary_logloss: 0.124237	valid_1's auc: 0.834479	valid_1's binary_logloss: 0.13659
    [37]	valid_0's auc: 0.875938	valid_0's binary_logloss: 0.123998	valid_1's auc: 0.834359	valid_1's binary_logloss: 0.136644
    [38]	valid_0's auc: 0.876343	valid_0's binary_logloss: 0.123795	valid_1's auc: 0.834106	valid_1's binary_logloss: 0.136687
    [39]	valid_0's auc: 0.877139	valid_0's binary_logloss: 0.123529	valid_1's auc: 0.83359	valid_1's binary_logloss: 0.136724
    [40]	valid_0's auc: 0.877911	valid_0's binary_logloss: 0.123326	valid_1's auc: 0.833512	valid_1's binary_logloss: 0.136737
    [41]	valid_0's auc: 0.87875	valid_0's binary_logloss: 0.123086	valid_1's auc: 0.833202	valid_1's binary_logloss: 0.136752
    [42]	valid_0's auc: 0.879342	valid_0's binary_logloss: 0.122885	valid_1's auc: 0.832649	valid_1's binary_logloss: 0.136842
    [43]	valid_0's auc: 0.879629	valid_0's binary_logloss: 0.122717	valid_1's auc: 0.832179	valid_1's binary_logloss: 0.136927
    [44]	valid_0's auc: 0.880215	valid_0's binary_logloss: 0.122487	valid_1's auc: 0.832172	valid_1's binary_logloss: 0.13695
    [45]	valid_0's auc: 0.880851	valid_0's binary_logloss: 0.122245	valid_1's auc: 0.831665	valid_1's binary_logloss: 0.137053
    [46]	valid_0's auc: 0.881492	valid_0's binary_logloss: 0.121994	valid_1's auc: 0.83185	valid_1's binary_logloss: 0.137071
    [47]	valid_0's auc: 0.881818	valid_0's binary_logloss: 0.121836	valid_1's auc: 0.831946	valid_1's binary_logloss: 0.137043
    [48]	valid_0's auc: 0.882136	valid_0's binary_logloss: 0.121683	valid_1's auc: 0.832078	valid_1's binary_logloss: 0.137058
    [49]	valid_0's auc: 0.882509	valid_0's binary_logloss: 0.121512	valid_1's auc: 0.832046	valid_1's binary_logloss: 0.13711
    [50]	valid_0's auc: 0.882872	valid_0's binary_logloss: 0.121296	valid_1's auc: 0.83209	valid_1's binary_logloss: 0.137124
    [51]	valid_0's auc: 0.883328	valid_0's binary_logloss: 0.121099	valid_1's auc: 0.831958	valid_1's binary_logloss: 0.137201
    [52]	valid_0's auc: 0.883722	valid_0's binary_logloss: 0.120897	valid_1's auc: 0.831752	valid_1's binary_logloss: 0.13722
    Early stopping, best iteration is:
    [22]	valid_0's auc: 0.864467	valid_0's binary_logloss: 0.128421	valid_1's auc: 0.836013	valid_1's binary_logloss: 0.137127
    [1]	valid_0's auc: 0.829442	valid_0's binary_logloss: 0.156848	valid_1's auc: 0.801853	valid_1's binary_logloss: 0.159917
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.835044	valid_0's binary_logloss: 0.151129	valid_1's auc: 0.810636	valid_1's binary_logloss: 0.155191
    [3]	valid_0's auc: 0.84106	valid_0's binary_logloss: 0.146986	valid_1's auc: 0.81428	valid_1's binary_logloss: 0.151786
    [4]	valid_0's auc: 0.849104	valid_0's binary_logloss: 0.143746	valid_1's auc: 0.822369	valid_1's binary_logloss: 0.149174
    [5]	valid_0's auc: 0.852054	valid_0's binary_logloss: 0.14111	valid_1's auc: 0.825422	valid_1's binary_logloss: 0.147108
    [6]	valid_0's auc: 0.854219	valid_0's binary_logloss: 0.138941	valid_1's auc: 0.828128	valid_1's binary_logloss: 0.145314
    [7]	valid_0's auc: 0.857758	valid_0's binary_logloss: 0.136956	valid_1's auc: 0.828178	valid_1's binary_logloss: 0.144053
    [8]	valid_0's auc: 0.859433	valid_0's binary_logloss: 0.135341	valid_1's auc: 0.828907	valid_1's binary_logloss: 0.142933
    [9]	valid_0's auc: 0.860562	valid_0's binary_logloss: 0.133965	valid_1's auc: 0.830266	valid_1's binary_logloss: 0.141913
    [10]	valid_0's auc: 0.862253	valid_0's binary_logloss: 0.132732	valid_1's auc: 0.830436	valid_1's binary_logloss: 0.141159
    [11]	valid_0's auc: 0.863863	valid_0's binary_logloss: 0.131616	valid_1's auc: 0.830888	valid_1's binary_logloss: 0.140465
    [12]	valid_0's auc: 0.86535	valid_0's binary_logloss: 0.130642	valid_1's auc: 0.828473	valid_1's binary_logloss: 0.140089
    [13]	valid_0's auc: 0.866315	valid_0's binary_logloss: 0.129707	valid_1's auc: 0.828442	valid_1's binary_logloss: 0.139673
    [14]	valid_0's auc: 0.868091	valid_0's binary_logloss: 0.128809	valid_1's auc: 0.828803	valid_1's binary_logloss: 0.13925
    [15]	valid_0's auc: 0.869154	valid_0's binary_logloss: 0.128055	valid_1's auc: 0.829147	valid_1's binary_logloss: 0.138981
    [16]	valid_0's auc: 0.870086	valid_0's binary_logloss: 0.127376	valid_1's auc: 0.829188	valid_1's binary_logloss: 0.13878
    [17]	valid_0's auc: 0.870857	valid_0's binary_logloss: 0.126705	valid_1's auc: 0.829277	valid_1's binary_logloss: 0.138553
    [18]	valid_0's auc: 0.87202	valid_0's binary_logloss: 0.126023	valid_1's auc: 0.829614	valid_1's binary_logloss: 0.138375
    [19]	valid_0's auc: 0.873171	valid_0's binary_logloss: 0.125459	valid_1's auc: 0.830335	valid_1's binary_logloss: 0.13824
    [20]	valid_0's auc: 0.874593	valid_0's binary_logloss: 0.124876	valid_1's auc: 0.831752	valid_1's binary_logloss: 0.13798
    [21]	valid_0's auc: 0.875741	valid_0's binary_logloss: 0.124281	valid_1's auc: 0.832761	valid_1's binary_logloss: 0.137687
    [22]	valid_0's auc: 0.877448	valid_0's binary_logloss: 0.123659	valid_1's auc: 0.833679	valid_1's binary_logloss: 0.137492
    [23]	valid_0's auc: 0.878493	valid_0's binary_logloss: 0.123115	valid_1's auc: 0.833912	valid_1's binary_logloss: 0.137434
    [24]	valid_0's auc: 0.879193	valid_0's binary_logloss: 0.122682	valid_1's auc: 0.833363	valid_1's binary_logloss: 0.137364
    [25]	valid_0's auc: 0.880227	valid_0's binary_logloss: 0.122229	valid_1's auc: 0.833173	valid_1's binary_logloss: 0.137358
    [26]	valid_0's auc: 0.881585	valid_0's binary_logloss: 0.121752	valid_1's auc: 0.832904	valid_1's binary_logloss: 0.13741
    [27]	valid_0's auc: 0.882473	valid_0's binary_logloss: 0.1213	valid_1's auc: 0.832408	valid_1's binary_logloss: 0.137589
    [28]	valid_0's auc: 0.883437	valid_0's binary_logloss: 0.120929	valid_1's auc: 0.832986	valid_1's binary_logloss: 0.137512
    [29]	valid_0's auc: 0.884502	valid_0's binary_logloss: 0.120484	valid_1's auc: 0.833732	valid_1's binary_logloss: 0.137414
    [30]	valid_0's auc: 0.88539	valid_0's binary_logloss: 0.120132	valid_1's auc: 0.83355	valid_1's binary_logloss: 0.137429
    [31]	valid_0's auc: 0.886163	valid_0's binary_logloss: 0.11975	valid_1's auc: 0.833288	valid_1's binary_logloss: 0.137459
    [32]	valid_0's auc: 0.886867	valid_0's binary_logloss: 0.119396	valid_1's auc: 0.833051	valid_1's binary_logloss: 0.137503
    [33]	valid_0's auc: 0.887733	valid_0's binary_logloss: 0.119013	valid_1's auc: 0.832946	valid_1's binary_logloss: 0.137467
    [34]	valid_0's auc: 0.888681	valid_0's binary_logloss: 0.118651	valid_1's auc: 0.833519	valid_1's binary_logloss: 0.137391
    [35]	valid_0's auc: 0.889275	valid_0's binary_logloss: 0.118341	valid_1's auc: 0.833734	valid_1's binary_logloss: 0.137371
    [36]	valid_0's auc: 0.890084	valid_0's binary_logloss: 0.117965	valid_1's auc: 0.833231	valid_1's binary_logloss: 0.137505
    [37]	valid_0's auc: 0.890726	valid_0's binary_logloss: 0.11765	valid_1's auc: 0.833136	valid_1's binary_logloss: 0.13761
    [38]	valid_0's auc: 0.891158	valid_0's binary_logloss: 0.11734	valid_1's auc: 0.832938	valid_1's binary_logloss: 0.137668
    [39]	valid_0's auc: 0.89196	valid_0's binary_logloss: 0.117006	valid_1's auc: 0.832307	valid_1's binary_logloss: 0.13782
    [40]	valid_0's auc: 0.892449	valid_0's binary_logloss: 0.116703	valid_1's auc: 0.832354	valid_1's binary_logloss: 0.137858
    [41]	valid_0's auc: 0.892978	valid_0's binary_logloss: 0.116414	valid_1's auc: 0.831731	valid_1's binary_logloss: 0.137998
    [42]	valid_0's auc: 0.893542	valid_0's binary_logloss: 0.116071	valid_1's auc: 0.830942	valid_1's binary_logloss: 0.138189
    [43]	valid_0's auc: 0.894078	valid_0's binary_logloss: 0.115805	valid_1's auc: 0.830997	valid_1's binary_logloss: 0.138182
    [44]	valid_0's auc: 0.894576	valid_0's binary_logloss: 0.115516	valid_1's auc: 0.830968	valid_1's binary_logloss: 0.138257
    [45]	valid_0's auc: 0.895099	valid_0's binary_logloss: 0.115236	valid_1's auc: 0.830506	valid_1's binary_logloss: 0.13839
    [46]	valid_0's auc: 0.895608	valid_0's binary_logloss: 0.114971	valid_1's auc: 0.829892	valid_1's binary_logloss: 0.138556
    [47]	valid_0's auc: 0.895999	valid_0's binary_logloss: 0.1147	valid_1's auc: 0.829591	valid_1's binary_logloss: 0.138649
    [48]	valid_0's auc: 0.89641	valid_0's binary_logloss: 0.11442	valid_1's auc: 0.829768	valid_1's binary_logloss: 0.138698
    [49]	valid_0's auc: 0.896742	valid_0's binary_logloss: 0.114182	valid_1's auc: 0.830072	valid_1's binary_logloss: 0.138692
    [50]	valid_0's auc: 0.896929	valid_0's binary_logloss: 0.113989	valid_1's auc: 0.829948	valid_1's binary_logloss: 0.138769
    [51]	valid_0's auc: 0.897717	valid_0's binary_logloss: 0.113718	valid_1's auc: 0.830297	valid_1's binary_logloss: 0.138742
    [52]	valid_0's auc: 0.898093	valid_0's binary_logloss: 0.113472	valid_1's auc: 0.830332	valid_1's binary_logloss: 0.138776
    [53]	valid_0's auc: 0.898387	valid_0's binary_logloss: 0.113271	valid_1's auc: 0.830278	valid_1's binary_logloss: 0.138833
    Early stopping, best iteration is:
    [23]	valid_0's auc: 0.878493	valid_0's binary_logloss: 0.123115	valid_1's auc: 0.833912	valid_1's binary_logloss: 0.137434
    [1]	valid_0's auc: 0.834008	valid_0's binary_logloss: 0.156932	valid_1's auc: 0.806689	valid_1's binary_logloss: 0.159986
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.841271	valid_0's binary_logloss: 0.151528	valid_1's auc: 0.816568	valid_1's binary_logloss: 0.155337
    [3]	valid_0's auc: 0.843827	valid_0's binary_logloss: 0.147378	valid_1's auc: 0.818842	valid_1's binary_logloss: 0.151989
    [4]	valid_0's auc: 0.850477	valid_0's binary_logloss: 0.144112	valid_1's auc: 0.824908	valid_1's binary_logloss: 0.149225
    [5]	valid_0's auc: 0.853808	valid_0's binary_logloss: 0.141422	valid_1's auc: 0.826744	valid_1's binary_logloss: 0.147175
    [6]	valid_0's auc: 0.856663	valid_0's binary_logloss: 0.139222	valid_1's auc: 0.828432	valid_1's binary_logloss: 0.145499
    [7]	valid_0's auc: 0.859664	valid_0's binary_logloss: 0.137318	valid_1's auc: 0.829523	valid_1's binary_logloss: 0.14402
    [8]	valid_0's auc: 0.86186	valid_0's binary_logloss: 0.135665	valid_1's auc: 0.8314	valid_1's binary_logloss: 0.142812
    [9]	valid_0's auc: 0.863104	valid_0's binary_logloss: 0.134357	valid_1's auc: 0.831976	valid_1's binary_logloss: 0.141915
    [10]	valid_0's auc: 0.865027	valid_0's binary_logloss: 0.133117	valid_1's auc: 0.83369	valid_1's binary_logloss: 0.141045
    [11]	valid_0's auc: 0.866437	valid_0's binary_logloss: 0.132026	valid_1's auc: 0.834219	valid_1's binary_logloss: 0.140347
    [12]	valid_0's auc: 0.868039	valid_0's binary_logloss: 0.131061	valid_1's auc: 0.834388	valid_1's binary_logloss: 0.139858
    [13]	valid_0's auc: 0.869537	valid_0's binary_logloss: 0.130159	valid_1's auc: 0.835402	valid_1's binary_logloss: 0.139334
    [14]	valid_0's auc: 0.870971	valid_0's binary_logloss: 0.12932	valid_1's auc: 0.835583	valid_1's binary_logloss: 0.138962
    [15]	valid_0's auc: 0.872381	valid_0's binary_logloss: 0.128543	valid_1's auc: 0.836121	valid_1's binary_logloss: 0.138579
    [16]	valid_0's auc: 0.873871	valid_0's binary_logloss: 0.127833	valid_1's auc: 0.836592	valid_1's binary_logloss: 0.13832
    [17]	valid_0's auc: 0.87491	valid_0's binary_logloss: 0.127212	valid_1's auc: 0.836031	valid_1's binary_logloss: 0.138138
    [18]	valid_0's auc: 0.875677	valid_0's binary_logloss: 0.126616	valid_1's auc: 0.835779	valid_1's binary_logloss: 0.137944
    [19]	valid_0's auc: 0.876894	valid_0's binary_logloss: 0.126018	valid_1's auc: 0.836666	valid_1's binary_logloss: 0.137657
    [20]	valid_0's auc: 0.877834	valid_0's binary_logloss: 0.125465	valid_1's auc: 0.836513	valid_1's binary_logloss: 0.137603
    [21]	valid_0's auc: 0.878861	valid_0's binary_logloss: 0.124918	valid_1's auc: 0.836696	valid_1's binary_logloss: 0.137441
    [22]	valid_0's auc: 0.880257	valid_0's binary_logloss: 0.124391	valid_1's auc: 0.836449	valid_1's binary_logloss: 0.137422
    [23]	valid_0's auc: 0.8818	valid_0's binary_logloss: 0.123782	valid_1's auc: 0.836709	valid_1's binary_logloss: 0.137292
    [24]	valid_0's auc: 0.882923	valid_0's binary_logloss: 0.123257	valid_1's auc: 0.836765	valid_1's binary_logloss: 0.137199
    [25]	valid_0's auc: 0.883924	valid_0's binary_logloss: 0.122788	valid_1's auc: 0.836654	valid_1's binary_logloss: 0.137164
    [26]	valid_0's auc: 0.885046	valid_0's binary_logloss: 0.12229	valid_1's auc: 0.836564	valid_1's binary_logloss: 0.137112
    [27]	valid_0's auc: 0.886069	valid_0's binary_logloss: 0.121841	valid_1's auc: 0.836535	valid_1's binary_logloss: 0.137076
    [28]	valid_0's auc: 0.886927	valid_0's binary_logloss: 0.121432	valid_1's auc: 0.836263	valid_1's binary_logloss: 0.137134
    [29]	valid_0's auc: 0.887853	valid_0's binary_logloss: 0.121034	valid_1's auc: 0.836202	valid_1's binary_logloss: 0.137072
    [30]	valid_0's auc: 0.888728	valid_0's binary_logloss: 0.120631	valid_1's auc: 0.836619	valid_1's binary_logloss: 0.136982
    [31]	valid_0's auc: 0.889701	valid_0's binary_logloss: 0.120172	valid_1's auc: 0.836517	valid_1's binary_logloss: 0.136934
    [32]	valid_0's auc: 0.890993	valid_0's binary_logloss: 0.119756	valid_1's auc: 0.836361	valid_1's binary_logloss: 0.136931
    [33]	valid_0's auc: 0.892053	valid_0's binary_logloss: 0.119407	valid_1's auc: 0.836583	valid_1's binary_logloss: 0.136846
    [34]	valid_0's auc: 0.892821	valid_0's binary_logloss: 0.119049	valid_1's auc: 0.836327	valid_1's binary_logloss: 0.136961
    [35]	valid_0's auc: 0.893663	valid_0's binary_logloss: 0.118651	valid_1's auc: 0.836579	valid_1's binary_logloss: 0.136912
    [36]	valid_0's auc: 0.894523	valid_0's binary_logloss: 0.11827	valid_1's auc: 0.836209	valid_1's binary_logloss: 0.136964
    [37]	valid_0's auc: 0.895475	valid_0's binary_logloss: 0.117947	valid_1's auc: 0.835968	valid_1's binary_logloss: 0.13697
    [38]	valid_0's auc: 0.895896	valid_0's binary_logloss: 0.117661	valid_1's auc: 0.836021	valid_1's binary_logloss: 0.136984
    [39]	valid_0's auc: 0.896908	valid_0's binary_logloss: 0.117297	valid_1's auc: 0.835969	valid_1's binary_logloss: 0.137061
    [40]	valid_0's auc: 0.897428	valid_0's binary_logloss: 0.116993	valid_1's auc: 0.835781	valid_1's binary_logloss: 0.137135
    [41]	valid_0's auc: 0.89802	valid_0's binary_logloss: 0.116616	valid_1's auc: 0.834706	valid_1's binary_logloss: 0.137377
    [42]	valid_0's auc: 0.898553	valid_0's binary_logloss: 0.116307	valid_1's auc: 0.834495	valid_1's binary_logloss: 0.137447
    [43]	valid_0's auc: 0.898979	valid_0's binary_logloss: 0.11602	valid_1's auc: 0.834078	valid_1's binary_logloss: 0.137573
    [44]	valid_0's auc: 0.899439	valid_0's binary_logloss: 0.115752	valid_1's auc: 0.833605	valid_1's binary_logloss: 0.137706
    [45]	valid_0's auc: 0.899915	valid_0's binary_logloss: 0.115497	valid_1's auc: 0.833114	valid_1's binary_logloss: 0.137835
    [46]	valid_0's auc: 0.90021	valid_0's binary_logloss: 0.115245	valid_1's auc: 0.833104	valid_1's binary_logloss: 0.137879
    [47]	valid_0's auc: 0.90064	valid_0's binary_logloss: 0.114971	valid_1's auc: 0.83304	valid_1's binary_logloss: 0.137916
    [48]	valid_0's auc: 0.901303	valid_0's binary_logloss: 0.114665	valid_1's auc: 0.833281	valid_1's binary_logloss: 0.137925
    [49]	valid_0's auc: 0.901644	valid_0's binary_logloss: 0.114372	valid_1's auc: 0.83306	valid_1's binary_logloss: 0.138017
    [50]	valid_0's auc: 0.902036	valid_0's binary_logloss: 0.114104	valid_1's auc: 0.832675	valid_1's binary_logloss: 0.13811
    [51]	valid_0's auc: 0.902469	valid_0's binary_logloss: 0.113883	valid_1's auc: 0.832339	valid_1's binary_logloss: 0.138195
    [52]	valid_0's auc: 0.902667	valid_0's binary_logloss: 0.113693	valid_1's auc: 0.831957	valid_1's binary_logloss: 0.138298
    [53]	valid_0's auc: 0.902822	valid_0's binary_logloss: 0.113485	valid_1's auc: 0.831693	valid_1's binary_logloss: 0.138383
    [54]	valid_0's auc: 0.902977	valid_0's binary_logloss: 0.113301	valid_1's auc: 0.830964	valid_1's binary_logloss: 0.138566
    Early stopping, best iteration is:
    [24]	valid_0's auc: 0.882923	valid_0's binary_logloss: 0.123257	valid_1's auc: 0.836765	valid_1's binary_logloss: 0.137199
    [1]	valid_0's auc: 0.832048	valid_0's binary_logloss: 0.157225	valid_1's auc: 0.811916	valid_1's binary_logloss: 0.159885
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.843711	valid_0's binary_logloss: 0.151535	valid_1's auc: 0.821654	valid_1's binary_logloss: 0.154778
    [3]	valid_0's auc: 0.846336	valid_0's binary_logloss: 0.14743	valid_1's auc: 0.823358	valid_1's binary_logloss: 0.151371
    [4]	valid_0's auc: 0.849091	valid_0's binary_logloss: 0.144223	valid_1's auc: 0.826879	valid_1's binary_logloss: 0.148584
    [5]	valid_0's auc: 0.851041	valid_0's binary_logloss: 0.141653	valid_1's auc: 0.825653	valid_1's binary_logloss: 0.14656
    [6]	valid_0's auc: 0.853453	valid_0's binary_logloss: 0.139394	valid_1's auc: 0.828113	valid_1's binary_logloss: 0.144822
    [7]	valid_0's auc: 0.857868	valid_0's binary_logloss: 0.137439	valid_1's auc: 0.828786	valid_1's binary_logloss: 0.143565
    [8]	valid_0's auc: 0.860559	valid_0's binary_logloss: 0.135866	valid_1's auc: 0.829347	valid_1's binary_logloss: 0.142504
    [9]	valid_0's auc: 0.861868	valid_0's binary_logloss: 0.134472	valid_1's auc: 0.829546	valid_1's binary_logloss: 0.141596
    [10]	valid_0's auc: 0.862998	valid_0's binary_logloss: 0.133271	valid_1's auc: 0.828894	valid_1's binary_logloss: 0.140928
    [11]	valid_0's auc: 0.864761	valid_0's binary_logloss: 0.132122	valid_1's auc: 0.828717	valid_1's binary_logloss: 0.140351
    [12]	valid_0's auc: 0.865861	valid_0's binary_logloss: 0.131127	valid_1's auc: 0.827893	valid_1's binary_logloss: 0.139897
    [13]	valid_0's auc: 0.867377	valid_0's binary_logloss: 0.130201	valid_1's auc: 0.828411	valid_1's binary_logloss: 0.139452
    [14]	valid_0's auc: 0.868645	valid_0's binary_logloss: 0.129388	valid_1's auc: 0.828859	valid_1's binary_logloss: 0.139104
    [15]	valid_0's auc: 0.869811	valid_0's binary_logloss: 0.12864	valid_1's auc: 0.827872	valid_1's binary_logloss: 0.138919
    [16]	valid_0's auc: 0.870761	valid_0's binary_logloss: 0.127902	valid_1's auc: 0.827629	valid_1's binary_logloss: 0.138614
    [17]	valid_0's auc: 0.872392	valid_0's binary_logloss: 0.127247	valid_1's auc: 0.828475	valid_1's binary_logloss: 0.138424
    [18]	valid_0's auc: 0.873984	valid_0's binary_logloss: 0.126643	valid_1's auc: 0.828038	valid_1's binary_logloss: 0.13836
    [19]	valid_0's auc: 0.875277	valid_0's binary_logloss: 0.126018	valid_1's auc: 0.828307	valid_1's binary_logloss: 0.138229
    [20]	valid_0's auc: 0.876767	valid_0's binary_logloss: 0.125429	valid_1's auc: 0.828398	valid_1's binary_logloss: 0.138132
    [21]	valid_0's auc: 0.878127	valid_0's binary_logloss: 0.124871	valid_1's auc: 0.827975	valid_1's binary_logloss: 0.138032
    [22]	valid_0's auc: 0.879333	valid_0's binary_logloss: 0.12434	valid_1's auc: 0.828026	valid_1's binary_logloss: 0.137957
    [23]	valid_0's auc: 0.880379	valid_0's binary_logloss: 0.123879	valid_1's auc: 0.827567	valid_1's binary_logloss: 0.138018
    [24]	valid_0's auc: 0.881429	valid_0's binary_logloss: 0.123358	valid_1's auc: 0.827472	valid_1's binary_logloss: 0.137975
    [25]	valid_0's auc: 0.882335	valid_0's binary_logloss: 0.122902	valid_1's auc: 0.827456	valid_1's binary_logloss: 0.137997
    [26]	valid_0's auc: 0.883467	valid_0's binary_logloss: 0.122419	valid_1's auc: 0.827539	valid_1's binary_logloss: 0.137977
    [27]	valid_0's auc: 0.884573	valid_0's binary_logloss: 0.121954	valid_1's auc: 0.827608	valid_1's binary_logloss: 0.137962
    [28]	valid_0's auc: 0.885354	valid_0's binary_logloss: 0.12151	valid_1's auc: 0.827828	valid_1's binary_logloss: 0.137979
    [29]	valid_0's auc: 0.88633	valid_0's binary_logloss: 0.1211	valid_1's auc: 0.827743	valid_1's binary_logloss: 0.137916
    [30]	valid_0's auc: 0.887129	valid_0's binary_logloss: 0.120718	valid_1's auc: 0.82774	valid_1's binary_logloss: 0.13796
    [31]	valid_0's auc: 0.888235	valid_0's binary_logloss: 0.120283	valid_1's auc: 0.827163	valid_1's binary_logloss: 0.137963
    [32]	valid_0's auc: 0.889179	valid_0's binary_logloss: 0.119986	valid_1's auc: 0.826867	valid_1's binary_logloss: 0.13798
    [33]	valid_0's auc: 0.889826	valid_0's binary_logloss: 0.119626	valid_1's auc: 0.826594	valid_1's binary_logloss: 0.13808
    [34]	valid_0's auc: 0.890813	valid_0's binary_logloss: 0.119231	valid_1's auc: 0.826475	valid_1's binary_logloss: 0.13814
    [35]	valid_0's auc: 0.891526	valid_0's binary_logloss: 0.118839	valid_1's auc: 0.826365	valid_1's binary_logloss: 0.138135
    [36]	valid_0's auc: 0.892313	valid_0's binary_logloss: 0.1185	valid_1's auc: 0.825614	valid_1's binary_logloss: 0.138235
    [37]	valid_0's auc: 0.892917	valid_0's binary_logloss: 0.118143	valid_1's auc: 0.825345	valid_1's binary_logloss: 0.138329
    [38]	valid_0's auc: 0.89337	valid_0's binary_logloss: 0.117833	valid_1's auc: 0.824926	valid_1's binary_logloss: 0.138429
    [39]	valid_0's auc: 0.893935	valid_0's binary_logloss: 0.117548	valid_1's auc: 0.824655	valid_1's binary_logloss: 0.138523
    Early stopping, best iteration is:
    [9]	valid_0's auc: 0.861868	valid_0's binary_logloss: 0.134472	valid_1's auc: 0.829546	valid_1's binary_logloss: 0.141596
    [1]	valid_0's auc: 0.829442	valid_0's binary_logloss: 0.156848	valid_1's auc: 0.801853	valid_1's binary_logloss: 0.159917
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.835044	valid_0's binary_logloss: 0.151129	valid_1's auc: 0.810636	valid_1's binary_logloss: 0.155191
    [3]	valid_0's auc: 0.84106	valid_0's binary_logloss: 0.146986	valid_1's auc: 0.81428	valid_1's binary_logloss: 0.151786
    [4]	valid_0's auc: 0.849104	valid_0's binary_logloss: 0.143746	valid_1's auc: 0.822369	valid_1's binary_logloss: 0.149174
    [5]	valid_0's auc: 0.852054	valid_0's binary_logloss: 0.14111	valid_1's auc: 0.825422	valid_1's binary_logloss: 0.147108
    [6]	valid_0's auc: 0.854219	valid_0's binary_logloss: 0.138941	valid_1's auc: 0.828128	valid_1's binary_logloss: 0.145314
    [7]	valid_0's auc: 0.857758	valid_0's binary_logloss: 0.136956	valid_1's auc: 0.828178	valid_1's binary_logloss: 0.144053
    [8]	valid_0's auc: 0.859433	valid_0's binary_logloss: 0.135341	valid_1's auc: 0.828907	valid_1's binary_logloss: 0.142933
    [9]	valid_0's auc: 0.860562	valid_0's binary_logloss: 0.133965	valid_1's auc: 0.830266	valid_1's binary_logloss: 0.141913
    [10]	valid_0's auc: 0.862253	valid_0's binary_logloss: 0.132732	valid_1's auc: 0.830436	valid_1's binary_logloss: 0.141159
    [11]	valid_0's auc: 0.863863	valid_0's binary_logloss: 0.131616	valid_1's auc: 0.830888	valid_1's binary_logloss: 0.140465
    [12]	valid_0's auc: 0.86535	valid_0's binary_logloss: 0.130642	valid_1's auc: 0.828473	valid_1's binary_logloss: 0.140089
    [13]	valid_0's auc: 0.866315	valid_0's binary_logloss: 0.129707	valid_1's auc: 0.828442	valid_1's binary_logloss: 0.139673
    [14]	valid_0's auc: 0.868091	valid_0's binary_logloss: 0.128809	valid_1's auc: 0.828803	valid_1's binary_logloss: 0.13925
    [15]	valid_0's auc: 0.869154	valid_0's binary_logloss: 0.128055	valid_1's auc: 0.829147	valid_1's binary_logloss: 0.138981
    [16]	valid_0's auc: 0.870086	valid_0's binary_logloss: 0.127376	valid_1's auc: 0.829188	valid_1's binary_logloss: 0.13878
    [17]	valid_0's auc: 0.870857	valid_0's binary_logloss: 0.126705	valid_1's auc: 0.829277	valid_1's binary_logloss: 0.138553
    [18]	valid_0's auc: 0.87202	valid_0's binary_logloss: 0.126023	valid_1's auc: 0.829614	valid_1's binary_logloss: 0.138375
    [19]	valid_0's auc: 0.873171	valid_0's binary_logloss: 0.125459	valid_1's auc: 0.830335	valid_1's binary_logloss: 0.13824
    [20]	valid_0's auc: 0.874593	valid_0's binary_logloss: 0.124876	valid_1's auc: 0.831752	valid_1's binary_logloss: 0.13798
    [21]	valid_0's auc: 0.875741	valid_0's binary_logloss: 0.124281	valid_1's auc: 0.832761	valid_1's binary_logloss: 0.137687
    [22]	valid_0's auc: 0.877448	valid_0's binary_logloss: 0.123659	valid_1's auc: 0.833679	valid_1's binary_logloss: 0.137492
    [23]	valid_0's auc: 0.878493	valid_0's binary_logloss: 0.123115	valid_1's auc: 0.833912	valid_1's binary_logloss: 0.137434
    [24]	valid_0's auc: 0.879193	valid_0's binary_logloss: 0.122682	valid_1's auc: 0.833363	valid_1's binary_logloss: 0.137364
    [25]	valid_0's auc: 0.880227	valid_0's binary_logloss: 0.122229	valid_1's auc: 0.833173	valid_1's binary_logloss: 0.137358
    [26]	valid_0's auc: 0.881585	valid_0's binary_logloss: 0.121752	valid_1's auc: 0.832904	valid_1's binary_logloss: 0.13741
    [27]	valid_0's auc: 0.882473	valid_0's binary_logloss: 0.1213	valid_1's auc: 0.832408	valid_1's binary_logloss: 0.137589
    [28]	valid_0's auc: 0.883437	valid_0's binary_logloss: 0.120929	valid_1's auc: 0.832986	valid_1's binary_logloss: 0.137512
    [29]	valid_0's auc: 0.884502	valid_0's binary_logloss: 0.120484	valid_1's auc: 0.833732	valid_1's binary_logloss: 0.137414
    [30]	valid_0's auc: 0.88539	valid_0's binary_logloss: 0.120132	valid_1's auc: 0.83355	valid_1's binary_logloss: 0.137429
    [31]	valid_0's auc: 0.886163	valid_0's binary_logloss: 0.11975	valid_1's auc: 0.833288	valid_1's binary_logloss: 0.137459
    [32]	valid_0's auc: 0.886867	valid_0's binary_logloss: 0.119396	valid_1's auc: 0.833051	valid_1's binary_logloss: 0.137503
    [33]	valid_0's auc: 0.887733	valid_0's binary_logloss: 0.119013	valid_1's auc: 0.832946	valid_1's binary_logloss: 0.137467
    [34]	valid_0's auc: 0.888681	valid_0's binary_logloss: 0.118651	valid_1's auc: 0.833519	valid_1's binary_logloss: 0.137391
    [35]	valid_0's auc: 0.889275	valid_0's binary_logloss: 0.118341	valid_1's auc: 0.833734	valid_1's binary_logloss: 0.137371
    [36]	valid_0's auc: 0.890084	valid_0's binary_logloss: 0.117965	valid_1's auc: 0.833231	valid_1's binary_logloss: 0.137505
    [37]	valid_0's auc: 0.890726	valid_0's binary_logloss: 0.11765	valid_1's auc: 0.833136	valid_1's binary_logloss: 0.13761
    [38]	valid_0's auc: 0.891158	valid_0's binary_logloss: 0.11734	valid_1's auc: 0.832938	valid_1's binary_logloss: 0.137668
    [39]	valid_0's auc: 0.89196	valid_0's binary_logloss: 0.117006	valid_1's auc: 0.832307	valid_1's binary_logloss: 0.13782
    [40]	valid_0's auc: 0.892449	valid_0's binary_logloss: 0.116703	valid_1's auc: 0.832354	valid_1's binary_logloss: 0.137858
    [41]	valid_0's auc: 0.892978	valid_0's binary_logloss: 0.116414	valid_1's auc: 0.831731	valid_1's binary_logloss: 0.137998
    [42]	valid_0's auc: 0.893542	valid_0's binary_logloss: 0.116071	valid_1's auc: 0.830942	valid_1's binary_logloss: 0.138189
    [43]	valid_0's auc: 0.894078	valid_0's binary_logloss: 0.115805	valid_1's auc: 0.830997	valid_1's binary_logloss: 0.138182
    [44]	valid_0's auc: 0.894576	valid_0's binary_logloss: 0.115516	valid_1's auc: 0.830968	valid_1's binary_logloss: 0.138257
    [45]	valid_0's auc: 0.895099	valid_0's binary_logloss: 0.115236	valid_1's auc: 0.830506	valid_1's binary_logloss: 0.13839
    [46]	valid_0's auc: 0.895608	valid_0's binary_logloss: 0.114971	valid_1's auc: 0.829892	valid_1's binary_logloss: 0.138556
    [47]	valid_0's auc: 0.895999	valid_0's binary_logloss: 0.1147	valid_1's auc: 0.829591	valid_1's binary_logloss: 0.138649
    [48]	valid_0's auc: 0.89641	valid_0's binary_logloss: 0.11442	valid_1's auc: 0.829768	valid_1's binary_logloss: 0.138698
    [49]	valid_0's auc: 0.896742	valid_0's binary_logloss: 0.114182	valid_1's auc: 0.830072	valid_1's binary_logloss: 0.138692
    [50]	valid_0's auc: 0.896929	valid_0's binary_logloss: 0.113989	valid_1's auc: 0.829948	valid_1's binary_logloss: 0.138769
    [51]	valid_0's auc: 0.897717	valid_0's binary_logloss: 0.113718	valid_1's auc: 0.830297	valid_1's binary_logloss: 0.138742
    [52]	valid_0's auc: 0.898093	valid_0's binary_logloss: 0.113472	valid_1's auc: 0.830332	valid_1's binary_logloss: 0.138776
    [53]	valid_0's auc: 0.898387	valid_0's binary_logloss: 0.113271	valid_1's auc: 0.830278	valid_1's binary_logloss: 0.138833
    Early stopping, best iteration is:
    [23]	valid_0's auc: 0.878493	valid_0's binary_logloss: 0.123115	valid_1's auc: 0.833912	valid_1's binary_logloss: 0.137434
    [1]	valid_0's auc: 0.834008	valid_0's binary_logloss: 0.156932	valid_1's auc: 0.806689	valid_1's binary_logloss: 0.159986
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.841271	valid_0's binary_logloss: 0.151528	valid_1's auc: 0.816568	valid_1's binary_logloss: 0.155337
    [3]	valid_0's auc: 0.843827	valid_0's binary_logloss: 0.147378	valid_1's auc: 0.818842	valid_1's binary_logloss: 0.151989
    [4]	valid_0's auc: 0.850477	valid_0's binary_logloss: 0.144112	valid_1's auc: 0.824908	valid_1's binary_logloss: 0.149225
    [5]	valid_0's auc: 0.853808	valid_0's binary_logloss: 0.141422	valid_1's auc: 0.826744	valid_1's binary_logloss: 0.147175
    [6]	valid_0's auc: 0.856663	valid_0's binary_logloss: 0.139222	valid_1's auc: 0.828432	valid_1's binary_logloss: 0.145499
    [7]	valid_0's auc: 0.859664	valid_0's binary_logloss: 0.137318	valid_1's auc: 0.829523	valid_1's binary_logloss: 0.14402
    [8]	valid_0's auc: 0.86186	valid_0's binary_logloss: 0.135665	valid_1's auc: 0.8314	valid_1's binary_logloss: 0.142812
    [9]	valid_0's auc: 0.863104	valid_0's binary_logloss: 0.134357	valid_1's auc: 0.831976	valid_1's binary_logloss: 0.141915
    [10]	valid_0's auc: 0.865027	valid_0's binary_logloss: 0.133117	valid_1's auc: 0.83369	valid_1's binary_logloss: 0.141045
    [11]	valid_0's auc: 0.866437	valid_0's binary_logloss: 0.132026	valid_1's auc: 0.834219	valid_1's binary_logloss: 0.140347
    [12]	valid_0's auc: 0.868039	valid_0's binary_logloss: 0.131061	valid_1's auc: 0.834388	valid_1's binary_logloss: 0.139858
    [13]	valid_0's auc: 0.869537	valid_0's binary_logloss: 0.130159	valid_1's auc: 0.835402	valid_1's binary_logloss: 0.139334
    [14]	valid_0's auc: 0.870971	valid_0's binary_logloss: 0.12932	valid_1's auc: 0.835583	valid_1's binary_logloss: 0.138962
    [15]	valid_0's auc: 0.872381	valid_0's binary_logloss: 0.128543	valid_1's auc: 0.836121	valid_1's binary_logloss: 0.138579
    [16]	valid_0's auc: 0.873871	valid_0's binary_logloss: 0.127833	valid_1's auc: 0.836592	valid_1's binary_logloss: 0.13832
    [17]	valid_0's auc: 0.87491	valid_0's binary_logloss: 0.127212	valid_1's auc: 0.836031	valid_1's binary_logloss: 0.138138
    [18]	valid_0's auc: 0.875677	valid_0's binary_logloss: 0.126616	valid_1's auc: 0.835779	valid_1's binary_logloss: 0.137944
    [19]	valid_0's auc: 0.876894	valid_0's binary_logloss: 0.126018	valid_1's auc: 0.836666	valid_1's binary_logloss: 0.137657
    [20]	valid_0's auc: 0.877834	valid_0's binary_logloss: 0.125465	valid_1's auc: 0.836513	valid_1's binary_logloss: 0.137603
    [21]	valid_0's auc: 0.878861	valid_0's binary_logloss: 0.124918	valid_1's auc: 0.836696	valid_1's binary_logloss: 0.137441
    [22]	valid_0's auc: 0.880257	valid_0's binary_logloss: 0.124391	valid_1's auc: 0.836449	valid_1's binary_logloss: 0.137422
    [23]	valid_0's auc: 0.8818	valid_0's binary_logloss: 0.123782	valid_1's auc: 0.836709	valid_1's binary_logloss: 0.137292
    [24]	valid_0's auc: 0.882923	valid_0's binary_logloss: 0.123257	valid_1's auc: 0.836765	valid_1's binary_logloss: 0.137199
    [25]	valid_0's auc: 0.883924	valid_0's binary_logloss: 0.122788	valid_1's auc: 0.836654	valid_1's binary_logloss: 0.137164
    [26]	valid_0's auc: 0.885046	valid_0's binary_logloss: 0.12229	valid_1's auc: 0.836564	valid_1's binary_logloss: 0.137112
    [27]	valid_0's auc: 0.886069	valid_0's binary_logloss: 0.121841	valid_1's auc: 0.836535	valid_1's binary_logloss: 0.137076
    [28]	valid_0's auc: 0.886927	valid_0's binary_logloss: 0.121432	valid_1's auc: 0.836263	valid_1's binary_logloss: 0.137134
    [29]	valid_0's auc: 0.887853	valid_0's binary_logloss: 0.121034	valid_1's auc: 0.836202	valid_1's binary_logloss: 0.137072
    [30]	valid_0's auc: 0.888728	valid_0's binary_logloss: 0.120631	valid_1's auc: 0.836619	valid_1's binary_logloss: 0.136982
    [31]	valid_0's auc: 0.889701	valid_0's binary_logloss: 0.120172	valid_1's auc: 0.836517	valid_1's binary_logloss: 0.136934
    [32]	valid_0's auc: 0.890993	valid_0's binary_logloss: 0.119756	valid_1's auc: 0.836361	valid_1's binary_logloss: 0.136931
    [33]	valid_0's auc: 0.892053	valid_0's binary_logloss: 0.119407	valid_1's auc: 0.836583	valid_1's binary_logloss: 0.136846
    [34]	valid_0's auc: 0.892821	valid_0's binary_logloss: 0.119049	valid_1's auc: 0.836327	valid_1's binary_logloss: 0.136961
    [35]	valid_0's auc: 0.893663	valid_0's binary_logloss: 0.118651	valid_1's auc: 0.836579	valid_1's binary_logloss: 0.136912
    [36]	valid_0's auc: 0.894523	valid_0's binary_logloss: 0.11827	valid_1's auc: 0.836209	valid_1's binary_logloss: 0.136964
    [37]	valid_0's auc: 0.895475	valid_0's binary_logloss: 0.117947	valid_1's auc: 0.835968	valid_1's binary_logloss: 0.13697
    [38]	valid_0's auc: 0.895896	valid_0's binary_logloss: 0.117661	valid_1's auc: 0.836021	valid_1's binary_logloss: 0.136984
    [39]	valid_0's auc: 0.896908	valid_0's binary_logloss: 0.117297	valid_1's auc: 0.835969	valid_1's binary_logloss: 0.137061
    [40]	valid_0's auc: 0.897428	valid_0's binary_logloss: 0.116993	valid_1's auc: 0.835781	valid_1's binary_logloss: 0.137135
    [41]	valid_0's auc: 0.89802	valid_0's binary_logloss: 0.116616	valid_1's auc: 0.834706	valid_1's binary_logloss: 0.137377
    [42]	valid_0's auc: 0.898553	valid_0's binary_logloss: 0.116307	valid_1's auc: 0.834495	valid_1's binary_logloss: 0.137447
    [43]	valid_0's auc: 0.898979	valid_0's binary_logloss: 0.11602	valid_1's auc: 0.834078	valid_1's binary_logloss: 0.137573
    [44]	valid_0's auc: 0.899439	valid_0's binary_logloss: 0.115752	valid_1's auc: 0.833605	valid_1's binary_logloss: 0.137706
    [45]	valid_0's auc: 0.899915	valid_0's binary_logloss: 0.115497	valid_1's auc: 0.833114	valid_1's binary_logloss: 0.137835
    [46]	valid_0's auc: 0.90021	valid_0's binary_logloss: 0.115245	valid_1's auc: 0.833104	valid_1's binary_logloss: 0.137879
    [47]	valid_0's auc: 0.90064	valid_0's binary_logloss: 0.114971	valid_1's auc: 0.83304	valid_1's binary_logloss: 0.137916
    [48]	valid_0's auc: 0.901303	valid_0's binary_logloss: 0.114665	valid_1's auc: 0.833281	valid_1's binary_logloss: 0.137925
    [49]	valid_0's auc: 0.901644	valid_0's binary_logloss: 0.114372	valid_1's auc: 0.83306	valid_1's binary_logloss: 0.138017
    [50]	valid_0's auc: 0.902036	valid_0's binary_logloss: 0.114104	valid_1's auc: 0.832675	valid_1's binary_logloss: 0.13811
    [51]	valid_0's auc: 0.902469	valid_0's binary_logloss: 0.113883	valid_1's auc: 0.832339	valid_1's binary_logloss: 0.138195
    [52]	valid_0's auc: 0.902667	valid_0's binary_logloss: 0.113693	valid_1's auc: 0.831957	valid_1's binary_logloss: 0.138298
    [53]	valid_0's auc: 0.902822	valid_0's binary_logloss: 0.113485	valid_1's auc: 0.831693	valid_1's binary_logloss: 0.138383
    [54]	valid_0's auc: 0.902977	valid_0's binary_logloss: 0.113301	valid_1's auc: 0.830964	valid_1's binary_logloss: 0.138566
    Early stopping, best iteration is:
    [24]	valid_0's auc: 0.882923	valid_0's binary_logloss: 0.123257	valid_1's auc: 0.836765	valid_1's binary_logloss: 0.137199
    [1]	valid_0's auc: 0.832048	valid_0's binary_logloss: 0.157225	valid_1's auc: 0.811916	valid_1's binary_logloss: 0.159885
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.843711	valid_0's binary_logloss: 0.151535	valid_1's auc: 0.821654	valid_1's binary_logloss: 0.154778
    [3]	valid_0's auc: 0.846336	valid_0's binary_logloss: 0.14743	valid_1's auc: 0.823358	valid_1's binary_logloss: 0.151371
    [4]	valid_0's auc: 0.849091	valid_0's binary_logloss: 0.144223	valid_1's auc: 0.826879	valid_1's binary_logloss: 0.148584
    [5]	valid_0's auc: 0.851041	valid_0's binary_logloss: 0.141653	valid_1's auc: 0.825653	valid_1's binary_logloss: 0.14656
    [6]	valid_0's auc: 0.853453	valid_0's binary_logloss: 0.139394	valid_1's auc: 0.828113	valid_1's binary_logloss: 0.144822
    [7]	valid_0's auc: 0.857868	valid_0's binary_logloss: 0.137439	valid_1's auc: 0.828786	valid_1's binary_logloss: 0.143565
    [8]	valid_0's auc: 0.860559	valid_0's binary_logloss: 0.135866	valid_1's auc: 0.829347	valid_1's binary_logloss: 0.142504
    [9]	valid_0's auc: 0.861868	valid_0's binary_logloss: 0.134472	valid_1's auc: 0.829546	valid_1's binary_logloss: 0.141596
    [10]	valid_0's auc: 0.862998	valid_0's binary_logloss: 0.133271	valid_1's auc: 0.828894	valid_1's binary_logloss: 0.140928
    [11]	valid_0's auc: 0.864761	valid_0's binary_logloss: 0.132122	valid_1's auc: 0.828717	valid_1's binary_logloss: 0.140351
    [12]	valid_0's auc: 0.865861	valid_0's binary_logloss: 0.131127	valid_1's auc: 0.827893	valid_1's binary_logloss: 0.139897
    [13]	valid_0's auc: 0.867377	valid_0's binary_logloss: 0.130201	valid_1's auc: 0.828411	valid_1's binary_logloss: 0.139452
    [14]	valid_0's auc: 0.868645	valid_0's binary_logloss: 0.129388	valid_1's auc: 0.828859	valid_1's binary_logloss: 0.139104
    [15]	valid_0's auc: 0.869811	valid_0's binary_logloss: 0.12864	valid_1's auc: 0.827872	valid_1's binary_logloss: 0.138919
    [16]	valid_0's auc: 0.870761	valid_0's binary_logloss: 0.127902	valid_1's auc: 0.827629	valid_1's binary_logloss: 0.138614
    [17]	valid_0's auc: 0.872392	valid_0's binary_logloss: 0.127247	valid_1's auc: 0.828475	valid_1's binary_logloss: 0.138424
    [18]	valid_0's auc: 0.873984	valid_0's binary_logloss: 0.126643	valid_1's auc: 0.828038	valid_1's binary_logloss: 0.13836
    [19]	valid_0's auc: 0.875277	valid_0's binary_logloss: 0.126018	valid_1's auc: 0.828307	valid_1's binary_logloss: 0.138229
    [20]	valid_0's auc: 0.876767	valid_0's binary_logloss: 0.125429	valid_1's auc: 0.828398	valid_1's binary_logloss: 0.138132
    [21]	valid_0's auc: 0.878127	valid_0's binary_logloss: 0.124871	valid_1's auc: 0.827975	valid_1's binary_logloss: 0.138032
    [22]	valid_0's auc: 0.879333	valid_0's binary_logloss: 0.12434	valid_1's auc: 0.828026	valid_1's binary_logloss: 0.137957
    [23]	valid_0's auc: 0.880379	valid_0's binary_logloss: 0.123879	valid_1's auc: 0.827567	valid_1's binary_logloss: 0.138018
    [24]	valid_0's auc: 0.881429	valid_0's binary_logloss: 0.123358	valid_1's auc: 0.827472	valid_1's binary_logloss: 0.137975
    [25]	valid_0's auc: 0.882335	valid_0's binary_logloss: 0.122902	valid_1's auc: 0.827456	valid_1's binary_logloss: 0.137997
    [26]	valid_0's auc: 0.883467	valid_0's binary_logloss: 0.122419	valid_1's auc: 0.827539	valid_1's binary_logloss: 0.137977
    [27]	valid_0's auc: 0.884573	valid_0's binary_logloss: 0.121954	valid_1's auc: 0.827608	valid_1's binary_logloss: 0.137962
    [28]	valid_0's auc: 0.885354	valid_0's binary_logloss: 0.12151	valid_1's auc: 0.827828	valid_1's binary_logloss: 0.137979
    [29]	valid_0's auc: 0.88633	valid_0's binary_logloss: 0.1211	valid_1's auc: 0.827743	valid_1's binary_logloss: 0.137916
    [30]	valid_0's auc: 0.887129	valid_0's binary_logloss: 0.120718	valid_1's auc: 0.82774	valid_1's binary_logloss: 0.13796
    [31]	valid_0's auc: 0.888235	valid_0's binary_logloss: 0.120283	valid_1's auc: 0.827163	valid_1's binary_logloss: 0.137963
    [32]	valid_0's auc: 0.889179	valid_0's binary_logloss: 0.119986	valid_1's auc: 0.826867	valid_1's binary_logloss: 0.13798
    [33]	valid_0's auc: 0.889826	valid_0's binary_logloss: 0.119626	valid_1's auc: 0.826594	valid_1's binary_logloss: 0.13808
    [34]	valid_0's auc: 0.890813	valid_0's binary_logloss: 0.119231	valid_1's auc: 0.826475	valid_1's binary_logloss: 0.13814
    [35]	valid_0's auc: 0.891526	valid_0's binary_logloss: 0.118839	valid_1's auc: 0.826365	valid_1's binary_logloss: 0.138135
    [36]	valid_0's auc: 0.892313	valid_0's binary_logloss: 0.1185	valid_1's auc: 0.825614	valid_1's binary_logloss: 0.138235
    [37]	valid_0's auc: 0.892917	valid_0's binary_logloss: 0.118143	valid_1's auc: 0.825345	valid_1's binary_logloss: 0.138329
    [38]	valid_0's auc: 0.89337	valid_0's binary_logloss: 0.117833	valid_1's auc: 0.824926	valid_1's binary_logloss: 0.138429
    [39]	valid_0's auc: 0.893935	valid_0's binary_logloss: 0.117548	valid_1's auc: 0.824655	valid_1's binary_logloss: 0.138523
    Early stopping, best iteration is:
    [9]	valid_0's auc: 0.861868	valid_0's binary_logloss: 0.134472	valid_1's auc: 0.829546	valid_1's binary_logloss: 0.141596
    [1]	training's auc: 0.839703	training's binary_logloss: 0.156809	valid_1's auc: 0.817921	valid_1's binary_logloss: 0.159606
    Training until validation scores don't improve for 30 rounds
    [2]	training's auc: 0.846955	training's binary_logloss: 0.15093	valid_1's auc: 0.823809	valid_1's binary_logloss: 0.154661
    [3]	training's auc: 0.848991	training's binary_logloss: 0.146677	valid_1's auc: 0.826612	valid_1's binary_logloss: 0.151027
    [4]	training's auc: 0.853853	training's binary_logloss: 0.143312	valid_1's auc: 0.831015	valid_1's binary_logloss: 0.14835
    [5]	training's auc: 0.857858	training's binary_logloss: 0.140499	valid_1's auc: 0.833098	valid_1's binary_logloss: 0.146243
    [6]	training's auc: 0.862974	training's binary_logloss: 0.13811	valid_1's auc: 0.835525	valid_1's binary_logloss: 0.144516
    [7]	training's auc: 0.864891	training's binary_logloss: 0.13613	valid_1's auc: 0.836754	valid_1's binary_logloss: 0.142962
    [8]	training's auc: 0.866894	training's binary_logloss: 0.134447	valid_1's auc: 0.837807	valid_1's binary_logloss: 0.141759
    [9]	training's auc: 0.868891	training's binary_logloss: 0.132961	valid_1's auc: 0.837235	valid_1's binary_logloss: 0.140753
    [10]	training's auc: 0.870855	training's binary_logloss: 0.13166	valid_1's auc: 0.836434	valid_1's binary_logloss: 0.140125
    [11]	training's auc: 0.87319	training's binary_logloss: 0.130496	valid_1's auc: 0.836197	valid_1's binary_logloss: 0.139432
    [12]	training's auc: 0.874708	training's binary_logloss: 0.129386	valid_1's auc: 0.836683	valid_1's binary_logloss: 0.138855
    [13]	training's auc: 0.876663	training's binary_logloss: 0.12835	valid_1's auc: 0.837402	valid_1's binary_logloss: 0.138418
    [14]	training's auc: 0.878001	training's binary_logloss: 0.127449	valid_1's auc: 0.838371	valid_1's binary_logloss: 0.137929
    [15]	training's auc: 0.879473	training's binary_logloss: 0.126619	valid_1's auc: 0.838336	valid_1's binary_logloss: 0.137589
    [16]	training's auc: 0.881106	training's binary_logloss: 0.125816	valid_1's auc: 0.838195	valid_1's binary_logloss: 0.137288
    [17]	training's auc: 0.883126	training's binary_logloss: 0.12499	valid_1's auc: 0.838418	valid_1's binary_logloss: 0.137055
    [18]	training's auc: 0.884356	training's binary_logloss: 0.124249	valid_1's auc: 0.838926	valid_1's binary_logloss: 0.136776
    [19]	training's auc: 0.88594	training's binary_logloss: 0.123496	valid_1's auc: 0.839233	valid_1's binary_logloss: 0.136559
    [20]	training's auc: 0.888056	training's binary_logloss: 0.122735	valid_1's auc: 0.839083	valid_1's binary_logloss: 0.136463
    [21]	training's auc: 0.889477	training's binary_logloss: 0.122129	valid_1's auc: 0.839342	valid_1's binary_logloss: 0.136344
    [22]	training's auc: 0.891031	training's binary_logloss: 0.121459	valid_1's auc: 0.839125	valid_1's binary_logloss: 0.136224
    [23]	training's auc: 0.892302	training's binary_logloss: 0.120801	valid_1's auc: 0.839178	valid_1's binary_logloss: 0.136064
    [24]	training's auc: 0.894064	training's binary_logloss: 0.120189	valid_1's auc: 0.838919	valid_1's binary_logloss: 0.135984
    [25]	training's auc: 0.89516	training's binary_logloss: 0.119684	valid_1's auc: 0.839098	valid_1's binary_logloss: 0.135884
    [26]	training's auc: 0.896622	training's binary_logloss: 0.119147	valid_1's auc: 0.838926	valid_1's binary_logloss: 0.135856
    [27]	training's auc: 0.897956	training's binary_logloss: 0.118643	valid_1's auc: 0.838849	valid_1's binary_logloss: 0.135807
    [28]	training's auc: 0.899396	training's binary_logloss: 0.118125	valid_1's auc: 0.839089	valid_1's binary_logloss: 0.135729
    [29]	training's auc: 0.900904	training's binary_logloss: 0.117554	valid_1's auc: 0.839291	valid_1's binary_logloss: 0.135654
    [30]	training's auc: 0.902057	training's binary_logloss: 0.117051	valid_1's auc: 0.8404	valid_1's binary_logloss: 0.135429
    [31]	training's auc: 0.903392	training's binary_logloss: 0.116595	valid_1's auc: 0.840918	valid_1's binary_logloss: 0.13536
    [32]	training's auc: 0.904918	training's binary_logloss: 0.116121	valid_1's auc: 0.840607	valid_1's binary_logloss: 0.13539
    [33]	training's auc: 0.905902	training's binary_logloss: 0.115648	valid_1's auc: 0.840391	valid_1's binary_logloss: 0.135425
    [34]	training's auc: 0.907036	training's binary_logloss: 0.115179	valid_1's auc: 0.840591	valid_1's binary_logloss: 0.135366
    [35]	training's auc: 0.908252	training's binary_logloss: 0.114794	valid_1's auc: 0.840881	valid_1's binary_logloss: 0.135296
    [36]	training's auc: 0.909447	training's binary_logloss: 0.114355	valid_1's auc: 0.84061	valid_1's binary_logloss: 0.135338
    [37]	training's auc: 0.910387	training's binary_logloss: 0.11396	valid_1's auc: 0.84028	valid_1's binary_logloss: 0.135405
    [38]	training's auc: 0.911352	training's binary_logloss: 0.113516	valid_1's auc: 0.840001	valid_1's binary_logloss: 0.13547
    [39]	training's auc: 0.91229	training's binary_logloss: 0.113122	valid_1's auc: 0.840069	valid_1's binary_logloss: 0.135492
    [40]	training's auc: 0.913171	training's binary_logloss: 0.112682	valid_1's auc: 0.839742	valid_1's binary_logloss: 0.135543
    [41]	training's auc: 0.914131	training's binary_logloss: 0.112242	valid_1's auc: 0.839955	valid_1's binary_logloss: 0.135512
    [42]	training's auc: 0.915024	training's binary_logloss: 0.111823	valid_1's auc: 0.839755	valid_1's binary_logloss: 0.13558
    [43]	training's auc: 0.91603	training's binary_logloss: 0.111419	valid_1's auc: 0.839072	valid_1's binary_logloss: 0.135686
    [44]	training's auc: 0.916876	training's binary_logloss: 0.111071	valid_1's auc: 0.839192	valid_1's binary_logloss: 0.135681
    [45]	training's auc: 0.917856	training's binary_logloss: 0.11069	valid_1's auc: 0.839109	valid_1's binary_logloss: 0.135728
    [46]	training's auc: 0.918632	training's binary_logloss: 0.110354	valid_1's auc: 0.838949	valid_1's binary_logloss: 0.135714
    [47]	training's auc: 0.919253	training's binary_logloss: 0.110018	valid_1's auc: 0.839085	valid_1's binary_logloss: 0.135698
    [48]	training's auc: 0.919948	training's binary_logloss: 0.109683	valid_1's auc: 0.838662	valid_1's binary_logloss: 0.135791
    [49]	training's auc: 0.920717	training's binary_logloss: 0.109383	valid_1's auc: 0.838076	valid_1's binary_logloss: 0.135947
    [50]	training's auc: 0.921298	training's binary_logloss: 0.109059	valid_1's auc: 0.837589	valid_1's binary_logloss: 0.136096
    [51]	training's auc: 0.921956	training's binary_logloss: 0.108704	valid_1's auc: 0.837788	valid_1's binary_logloss: 0.136063
    [52]	training's auc: 0.922533	training's binary_logloss: 0.108399	valid_1's auc: 0.837838	valid_1's binary_logloss: 0.136062
    [53]	training's auc: 0.923051	training's binary_logloss: 0.108087	valid_1's auc: 0.837959	valid_1's binary_logloss: 0.136037
    [54]	training's auc: 0.923634	training's binary_logloss: 0.107759	valid_1's auc: 0.837827	valid_1's binary_logloss: 0.136078
    [55]	training's auc: 0.924126	training's binary_logloss: 0.107471	valid_1's auc: 0.837786	valid_1's binary_logloss: 0.13611
    [56]	training's auc: 0.924951	training's binary_logloss: 0.107143	valid_1's auc: 0.837684	valid_1's binary_logloss: 0.136161
    [57]	training's auc: 0.925535	training's binary_logloss: 0.106823	valid_1's auc: 0.837707	valid_1's binary_logloss: 0.136192
    [58]	training's auc: 0.926062	training's binary_logloss: 0.106475	valid_1's auc: 0.837522	valid_1's binary_logloss: 0.136255
    [59]	training's auc: 0.926679	training's binary_logloss: 0.106216	valid_1's auc: 0.837701	valid_1's binary_logloss: 0.136228
    [60]	training's auc: 0.926961	training's binary_logloss: 0.105991	valid_1's auc: 0.837402	valid_1's binary_logloss: 0.136291
    [61]	training's auc: 0.927506	training's binary_logloss: 0.105667	valid_1's auc: 0.836949	valid_1's binary_logloss: 0.136425
    Early stopping, best iteration is:
    [31]	training's auc: 0.903392	training's binary_logloss: 0.116595	valid_1's auc: 0.840918	valid_1's binary_logloss: 0.13536
    최적의 파라미터: {'max_depth': 130, 'min_child_samples': 100, 'num_leaves': 64, 'subsample': 0.8}
    roc_auc_score: 0.8409177530667706



```python
#알고리즘 생성
lgbm_clf = LGBMClassifier(n_estimators=1000, max_depth=130, 
                         min_child_samples=100, num_leaves=64, subsampl=0.8)

#평가할 데이터 - 평가할 데이터에는 훈련 데이터를 포함시키지 않아도 됩니다.
evals = [(X_test, y_test)]

lgbm_clf.fit(X_train, y_train, early_stopping_rounds=200, eval_metric='auc', 
            eval_set=evals, verbose=True)

lgbm_roc_auc_score = roc_auc_score(y_test, lgbm_clf.predict_proba(X_test)[:,1], 
                                 average='micro')
print('roc_auc_score:', lgbm_roc_auc_score)
```

    [LightGBM] [Warning] Unknown parameter: subsampl
    [1]	valid_0's auc: 0.817921	valid_0's binary_logloss: 0.159606
    Training until validation scores don't improve for 200 rounds
    [2]	valid_0's auc: 0.823809	valid_0's binary_logloss: 0.154661
    [3]	valid_0's auc: 0.826612	valid_0's binary_logloss: 0.151027
    [4]	valid_0's auc: 0.831015	valid_0's binary_logloss: 0.14835
    [5]	valid_0's auc: 0.833098	valid_0's binary_logloss: 0.146243
    [6]	valid_0's auc: 0.835525	valid_0's binary_logloss: 0.144516
    [7]	valid_0's auc: 0.836754	valid_0's binary_logloss: 0.142962
    [8]	valid_0's auc: 0.837807	valid_0's binary_logloss: 0.141759
    [9]	valid_0's auc: 0.837235	valid_0's binary_logloss: 0.140753
    [10]	valid_0's auc: 0.836434	valid_0's binary_logloss: 0.140125
    [11]	valid_0's auc: 0.836197	valid_0's binary_logloss: 0.139432
    [12]	valid_0's auc: 0.836683	valid_0's binary_logloss: 0.138855
    [13]	valid_0's auc: 0.837402	valid_0's binary_logloss: 0.138418
    [14]	valid_0's auc: 0.838371	valid_0's binary_logloss: 0.137929
    [15]	valid_0's auc: 0.838336	valid_0's binary_logloss: 0.137589
    [16]	valid_0's auc: 0.838195	valid_0's binary_logloss: 0.137288
    [17]	valid_0's auc: 0.838418	valid_0's binary_logloss: 0.137055
    [18]	valid_0's auc: 0.838926	valid_0's binary_logloss: 0.136776
    [19]	valid_0's auc: 0.839233	valid_0's binary_logloss: 0.136559
    [20]	valid_0's auc: 0.839083	valid_0's binary_logloss: 0.136463
    [21]	valid_0's auc: 0.839342	valid_0's binary_logloss: 0.136344
    [22]	valid_0's auc: 0.839125	valid_0's binary_logloss: 0.136224
    [23]	valid_0's auc: 0.839178	valid_0's binary_logloss: 0.136064
    [24]	valid_0's auc: 0.838919	valid_0's binary_logloss: 0.135984
    [25]	valid_0's auc: 0.839098	valid_0's binary_logloss: 0.135884
    [26]	valid_0's auc: 0.838926	valid_0's binary_logloss: 0.135856
    [27]	valid_0's auc: 0.838849	valid_0's binary_logloss: 0.135807
    [28]	valid_0's auc: 0.839089	valid_0's binary_logloss: 0.135729
    [29]	valid_0's auc: 0.839291	valid_0's binary_logloss: 0.135654
    [30]	valid_0's auc: 0.8404	valid_0's binary_logloss: 0.135429
    [31]	valid_0's auc: 0.840918	valid_0's binary_logloss: 0.13536
    [32]	valid_0's auc: 0.840607	valid_0's binary_logloss: 0.13539
    [33]	valid_0's auc: 0.840391	valid_0's binary_logloss: 0.135425
    [34]	valid_0's auc: 0.840591	valid_0's binary_logloss: 0.135366
    [35]	valid_0's auc: 0.840881	valid_0's binary_logloss: 0.135296
    [36]	valid_0's auc: 0.84061	valid_0's binary_logloss: 0.135338
    [37]	valid_0's auc: 0.84028	valid_0's binary_logloss: 0.135405
    [38]	valid_0's auc: 0.840001	valid_0's binary_logloss: 0.13547
    [39]	valid_0's auc: 0.840069	valid_0's binary_logloss: 0.135492
    [40]	valid_0's auc: 0.839742	valid_0's binary_logloss: 0.135543
    [41]	valid_0's auc: 0.839955	valid_0's binary_logloss: 0.135512
    [42]	valid_0's auc: 0.839755	valid_0's binary_logloss: 0.13558
    [43]	valid_0's auc: 0.839072	valid_0's binary_logloss: 0.135686
    [44]	valid_0's auc: 0.839192	valid_0's binary_logloss: 0.135681
    [45]	valid_0's auc: 0.839109	valid_0's binary_logloss: 0.135728
    [46]	valid_0's auc: 0.838949	valid_0's binary_logloss: 0.135714
    [47]	valid_0's auc: 0.839085	valid_0's binary_logloss: 0.135698
    [48]	valid_0's auc: 0.838662	valid_0's binary_logloss: 0.135791
    [49]	valid_0's auc: 0.838076	valid_0's binary_logloss: 0.135947
    [50]	valid_0's auc: 0.837589	valid_0's binary_logloss: 0.136096
    [51]	valid_0's auc: 0.837788	valid_0's binary_logloss: 0.136063
    [52]	valid_0's auc: 0.837838	valid_0's binary_logloss: 0.136062
    [53]	valid_0's auc: 0.837959	valid_0's binary_logloss: 0.136037
    [54]	valid_0's auc: 0.837827	valid_0's binary_logloss: 0.136078
    [55]	valid_0's auc: 0.837786	valid_0's binary_logloss: 0.13611
    [56]	valid_0's auc: 0.837684	valid_0's binary_logloss: 0.136161
    [57]	valid_0's auc: 0.837707	valid_0's binary_logloss: 0.136192
    [58]	valid_0's auc: 0.837522	valid_0's binary_logloss: 0.136255
    [59]	valid_0's auc: 0.837701	valid_0's binary_logloss: 0.136228
    [60]	valid_0's auc: 0.837402	valid_0's binary_logloss: 0.136291
    [61]	valid_0's auc: 0.836949	valid_0's binary_logloss: 0.136425
    [62]	valid_0's auc: 0.837234	valid_0's binary_logloss: 0.136377
    [63]	valid_0's auc: 0.837471	valid_0's binary_logloss: 0.136327
    [64]	valid_0's auc: 0.837117	valid_0's binary_logloss: 0.136429
    [65]	valid_0's auc: 0.83707	valid_0's binary_logloss: 0.13647
    [66]	valid_0's auc: 0.836756	valid_0's binary_logloss: 0.136587
    [67]	valid_0's auc: 0.836678	valid_0's binary_logloss: 0.136632
    [68]	valid_0's auc: 0.83646	valid_0's binary_logloss: 0.136703
    [69]	valid_0's auc: 0.836459	valid_0's binary_logloss: 0.136675
    [70]	valid_0's auc: 0.836579	valid_0's binary_logloss: 0.136687
    [71]	valid_0's auc: 0.836648	valid_0's binary_logloss: 0.136685
    [72]	valid_0's auc: 0.836554	valid_0's binary_logloss: 0.136712
    [73]	valid_0's auc: 0.836303	valid_0's binary_logloss: 0.136787
    [74]	valid_0's auc: 0.836362	valid_0's binary_logloss: 0.136818
    [75]	valid_0's auc: 0.836445	valid_0's binary_logloss: 0.136827
    [76]	valid_0's auc: 0.836104	valid_0's binary_logloss: 0.136919
    [77]	valid_0's auc: 0.836244	valid_0's binary_logloss: 0.136907
    [78]	valid_0's auc: 0.835922	valid_0's binary_logloss: 0.136997
    [79]	valid_0's auc: 0.836012	valid_0's binary_logloss: 0.136991
    [80]	valid_0's auc: 0.836437	valid_0's binary_logloss: 0.136939
    [81]	valid_0's auc: 0.836016	valid_0's binary_logloss: 0.137079
    [82]	valid_0's auc: 0.836305	valid_0's binary_logloss: 0.137037
    [83]	valid_0's auc: 0.836328	valid_0's binary_logloss: 0.137084
    [84]	valid_0's auc: 0.836217	valid_0's binary_logloss: 0.13716
    [85]	valid_0's auc: 0.836107	valid_0's binary_logloss: 0.137218
    [86]	valid_0's auc: 0.836034	valid_0's binary_logloss: 0.137301
    [87]	valid_0's auc: 0.836057	valid_0's binary_logloss: 0.137319
    [88]	valid_0's auc: 0.83605	valid_0's binary_logloss: 0.137339
    [89]	valid_0's auc: 0.835925	valid_0's binary_logloss: 0.137387
    [90]	valid_0's auc: 0.835666	valid_0's binary_logloss: 0.137467
    [91]	valid_0's auc: 0.83529	valid_0's binary_logloss: 0.137592
    [92]	valid_0's auc: 0.835008	valid_0's binary_logloss: 0.137718
    [93]	valid_0's auc: 0.834794	valid_0's binary_logloss: 0.137802
    [94]	valid_0's auc: 0.834637	valid_0's binary_logloss: 0.137864
    [95]	valid_0's auc: 0.834336	valid_0's binary_logloss: 0.137967
    [96]	valid_0's auc: 0.833828	valid_0's binary_logloss: 0.138136
    [97]	valid_0's auc: 0.833834	valid_0's binary_logloss: 0.138183
    [98]	valid_0's auc: 0.833718	valid_0's binary_logloss: 0.138216
    [99]	valid_0's auc: 0.833994	valid_0's binary_logloss: 0.138196
    [100]	valid_0's auc: 0.833936	valid_0's binary_logloss: 0.13821
    [101]	valid_0's auc: 0.833975	valid_0's binary_logloss: 0.138227
    [102]	valid_0's auc: 0.83377	valid_0's binary_logloss: 0.138331
    [103]	valid_0's auc: 0.833742	valid_0's binary_logloss: 0.138386
    [104]	valid_0's auc: 0.833589	valid_0's binary_logloss: 0.138444
    [105]	valid_0's auc: 0.833418	valid_0's binary_logloss: 0.138508
    [106]	valid_0's auc: 0.833103	valid_0's binary_logloss: 0.138598
    [107]	valid_0's auc: 0.83285	valid_0's binary_logloss: 0.138677
    [108]	valid_0's auc: 0.83278	valid_0's binary_logloss: 0.138751
    [109]	valid_0's auc: 0.832387	valid_0's binary_logloss: 0.138873
    [110]	valid_0's auc: 0.832142	valid_0's binary_logloss: 0.138934
    [111]	valid_0's auc: 0.831822	valid_0's binary_logloss: 0.139001
    [112]	valid_0's auc: 0.831683	valid_0's binary_logloss: 0.139037
    [113]	valid_0's auc: 0.831683	valid_0's binary_logloss: 0.13904
    [114]	valid_0's auc: 0.83133	valid_0's binary_logloss: 0.139155
    [115]	valid_0's auc: 0.830962	valid_0's binary_logloss: 0.139254
    [116]	valid_0's auc: 0.830517	valid_0's binary_logloss: 0.139417
    [117]	valid_0's auc: 0.830655	valid_0's binary_logloss: 0.139444
    [118]	valid_0's auc: 0.830202	valid_0's binary_logloss: 0.139587
    [119]	valid_0's auc: 0.830022	valid_0's binary_logloss: 0.139675
    [120]	valid_0's auc: 0.829869	valid_0's binary_logloss: 0.139738
    [121]	valid_0's auc: 0.829483	valid_0's binary_logloss: 0.139854
    [122]	valid_0's auc: 0.829326	valid_0's binary_logloss: 0.139918
    [123]	valid_0's auc: 0.829338	valid_0's binary_logloss: 0.139938
    [124]	valid_0's auc: 0.829358	valid_0's binary_logloss: 0.139936
    [125]	valid_0's auc: 0.829229	valid_0's binary_logloss: 0.140006
    [126]	valid_0's auc: 0.828803	valid_0's binary_logloss: 0.140108
    [127]	valid_0's auc: 0.828701	valid_0's binary_logloss: 0.140137
    [128]	valid_0's auc: 0.828391	valid_0's binary_logloss: 0.140261
    [129]	valid_0's auc: 0.828176	valid_0's binary_logloss: 0.140333
    [130]	valid_0's auc: 0.828027	valid_0's binary_logloss: 0.140389
    [131]	valid_0's auc: 0.828166	valid_0's binary_logloss: 0.140408
    [132]	valid_0's auc: 0.828292	valid_0's binary_logloss: 0.14041
    [133]	valid_0's auc: 0.828371	valid_0's binary_logloss: 0.140432
    [134]	valid_0's auc: 0.827975	valid_0's binary_logloss: 0.140528
    [135]	valid_0's auc: 0.828155	valid_0's binary_logloss: 0.140537
    [136]	valid_0's auc: 0.827886	valid_0's binary_logloss: 0.140659
    [137]	valid_0's auc: 0.827672	valid_0's binary_logloss: 0.140791
    [138]	valid_0's auc: 0.827867	valid_0's binary_logloss: 0.14077
    [139]	valid_0's auc: 0.827913	valid_0's binary_logloss: 0.140777
    [140]	valid_0's auc: 0.827724	valid_0's binary_logloss: 0.140837
    [141]	valid_0's auc: 0.827451	valid_0's binary_logloss: 0.14096
    [142]	valid_0's auc: 0.827285	valid_0's binary_logloss: 0.141036
    [143]	valid_0's auc: 0.827098	valid_0's binary_logloss: 0.141097
    [144]	valid_0's auc: 0.826906	valid_0's binary_logloss: 0.14119
    [145]	valid_0's auc: 0.826837	valid_0's binary_logloss: 0.141253
    [146]	valid_0's auc: 0.826342	valid_0's binary_logloss: 0.141438
    [147]	valid_0's auc: 0.82606	valid_0's binary_logloss: 0.141555
    [148]	valid_0's auc: 0.825928	valid_0's binary_logloss: 0.141636
    [149]	valid_0's auc: 0.825839	valid_0's binary_logloss: 0.141675
    [150]	valid_0's auc: 0.825928	valid_0's binary_logloss: 0.141697
    [151]	valid_0's auc: 0.826123	valid_0's binary_logloss: 0.141698
    [152]	valid_0's auc: 0.826079	valid_0's binary_logloss: 0.141723
    [153]	valid_0's auc: 0.825793	valid_0's binary_logloss: 0.14184
    [154]	valid_0's auc: 0.825616	valid_0's binary_logloss: 0.141925
    [155]	valid_0's auc: 0.825444	valid_0's binary_logloss: 0.142025
    [156]	valid_0's auc: 0.82541	valid_0's binary_logloss: 0.142081
    [157]	valid_0's auc: 0.825145	valid_0's binary_logloss: 0.142217
    [158]	valid_0's auc: 0.825047	valid_0's binary_logloss: 0.142255
    [159]	valid_0's auc: 0.824938	valid_0's binary_logloss: 0.142283
    [160]	valid_0's auc: 0.824664	valid_0's binary_logloss: 0.142376
    [161]	valid_0's auc: 0.824774	valid_0's binary_logloss: 0.142402
    [162]	valid_0's auc: 0.824864	valid_0's binary_logloss: 0.142424
    [163]	valid_0's auc: 0.82447	valid_0's binary_logloss: 0.14254
    [164]	valid_0's auc: 0.824485	valid_0's binary_logloss: 0.142564
    [165]	valid_0's auc: 0.824606	valid_0's binary_logloss: 0.142574
    [166]	valid_0's auc: 0.824459	valid_0's binary_logloss: 0.142623
    [167]	valid_0's auc: 0.824291	valid_0's binary_logloss: 0.142732
    [168]	valid_0's auc: 0.824107	valid_0's binary_logloss: 0.142846
    [169]	valid_0's auc: 0.82414	valid_0's binary_logloss: 0.142906
    [170]	valid_0's auc: 0.824018	valid_0's binary_logloss: 0.142958
    [171]	valid_0's auc: 0.823953	valid_0's binary_logloss: 0.142998
    [172]	valid_0's auc: 0.823604	valid_0's binary_logloss: 0.143141
    [173]	valid_0's auc: 0.823538	valid_0's binary_logloss: 0.143208
    [174]	valid_0's auc: 0.822946	valid_0's binary_logloss: 0.143393
    [175]	valid_0's auc: 0.823047	valid_0's binary_logloss: 0.143413
    [176]	valid_0's auc: 0.823011	valid_0's binary_logloss: 0.143441
    [177]	valid_0's auc: 0.822773	valid_0's binary_logloss: 0.143517
    [178]	valid_0's auc: 0.822624	valid_0's binary_logloss: 0.143597
    [179]	valid_0's auc: 0.822223	valid_0's binary_logloss: 0.143731
    [180]	valid_0's auc: 0.82228	valid_0's binary_logloss: 0.143782
    [181]	valid_0's auc: 0.822121	valid_0's binary_logloss: 0.14383
    [182]	valid_0's auc: 0.822126	valid_0's binary_logloss: 0.143921
    [183]	valid_0's auc: 0.822068	valid_0's binary_logloss: 0.143969
    [184]	valid_0's auc: 0.821807	valid_0's binary_logloss: 0.144071
    [185]	valid_0's auc: 0.821667	valid_0's binary_logloss: 0.144112
    [186]	valid_0's auc: 0.821846	valid_0's binary_logloss: 0.144094
    [187]	valid_0's auc: 0.821696	valid_0's binary_logloss: 0.144195
    [188]	valid_0's auc: 0.821491	valid_0's binary_logloss: 0.144319
    [189]	valid_0's auc: 0.821297	valid_0's binary_logloss: 0.14441
    [190]	valid_0's auc: 0.82114	valid_0's binary_logloss: 0.14451
    [191]	valid_0's auc: 0.820972	valid_0's binary_logloss: 0.144598
    [192]	valid_0's auc: 0.82112	valid_0's binary_logloss: 0.144602
    [193]	valid_0's auc: 0.820828	valid_0's binary_logloss: 0.144699
    [194]	valid_0's auc: 0.820437	valid_0's binary_logloss: 0.144802
    [195]	valid_0's auc: 0.820397	valid_0's binary_logloss: 0.144834
    [196]	valid_0's auc: 0.820365	valid_0's binary_logloss: 0.144898
    [197]	valid_0's auc: 0.820644	valid_0's binary_logloss: 0.144908
    [198]	valid_0's auc: 0.820368	valid_0's binary_logloss: 0.144999
    [199]	valid_0's auc: 0.820367	valid_0's binary_logloss: 0.145052
    [200]	valid_0's auc: 0.820459	valid_0's binary_logloss: 0.1451
    [201]	valid_0's auc: 0.82021	valid_0's binary_logloss: 0.145228
    [202]	valid_0's auc: 0.820361	valid_0's binary_logloss: 0.145256
    [203]	valid_0's auc: 0.820252	valid_0's binary_logloss: 0.145312
    [204]	valid_0's auc: 0.820069	valid_0's binary_logloss: 0.145385
    [205]	valid_0's auc: 0.820309	valid_0's binary_logloss: 0.145381
    [206]	valid_0's auc: 0.820228	valid_0's binary_logloss: 0.145453
    [207]	valid_0's auc: 0.820068	valid_0's binary_logloss: 0.145517
    [208]	valid_0's auc: 0.819857	valid_0's binary_logloss: 0.145596
    [209]	valid_0's auc: 0.819641	valid_0's binary_logloss: 0.145752
    [210]	valid_0's auc: 0.819285	valid_0's binary_logloss: 0.145887
    [211]	valid_0's auc: 0.819025	valid_0's binary_logloss: 0.146004
    [212]	valid_0's auc: 0.819122	valid_0's binary_logloss: 0.146004
    [213]	valid_0's auc: 0.818782	valid_0's binary_logloss: 0.146135
    [214]	valid_0's auc: 0.818692	valid_0's binary_logloss: 0.146187
    [215]	valid_0's auc: 0.81879	valid_0's binary_logloss: 0.146241
    [216]	valid_0's auc: 0.818664	valid_0's binary_logloss: 0.146326
    [217]	valid_0's auc: 0.818524	valid_0's binary_logloss: 0.146432
    [218]	valid_0's auc: 0.818123	valid_0's binary_logloss: 0.146583
    [219]	valid_0's auc: 0.818211	valid_0's binary_logloss: 0.1466
    [220]	valid_0's auc: 0.817789	valid_0's binary_logloss: 0.146766
    [221]	valid_0's auc: 0.817337	valid_0's binary_logloss: 0.146951
    [222]	valid_0's auc: 0.817499	valid_0's binary_logloss: 0.146952
    [223]	valid_0's auc: 0.81741	valid_0's binary_logloss: 0.147008
    [224]	valid_0's auc: 0.817474	valid_0's binary_logloss: 0.147019
    [225]	valid_0's auc: 0.817455	valid_0's binary_logloss: 0.14704
    [226]	valid_0's auc: 0.817271	valid_0's binary_logloss: 0.147101
    [227]	valid_0's auc: 0.817203	valid_0's binary_logloss: 0.147187
    [228]	valid_0's auc: 0.817232	valid_0's binary_logloss: 0.147189
    [229]	valid_0's auc: 0.817236	valid_0's binary_logloss: 0.147231
    [230]	valid_0's auc: 0.817212	valid_0's binary_logloss: 0.147263
    [231]	valid_0's auc: 0.817021	valid_0's binary_logloss: 0.147343
    Early stopping, best iteration is:
    [31]	valid_0's auc: 0.840918	valid_0's binary_logloss: 0.13536
    roc_auc_score: 0.8409177530667706


# 신용카드 연체 예측


```python
import numpy as np
import pandas as pd
import matplotlib as plt

## 데이터 가져오기
card_df = pd.read_csv('./data/creditcard.csv')
print(card_df.head())
```

       Time        V1        V2        V3        V4        V5        V6        V7  \
    0   0.0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599   
    1   0.0  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803   
    2   1.0 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461   
    3   1.0 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609   
    4   2.0 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941   
    
             V8        V9  ...       V21       V22       V23       V24       V25  \
    0  0.098698  0.363787  ... -0.018307  0.277838 -0.110474  0.066928  0.128539   
    1  0.085102 -0.255425  ... -0.225775 -0.638672  0.101288 -0.339846  0.167170   
    2  0.247676 -1.514654  ...  0.247998  0.771679  0.909412 -0.689281 -0.327642   
    3  0.377436 -1.387024  ... -0.108300  0.005274 -0.190321 -1.175575  0.647376   
    4 -0.270533  0.817739  ... -0.009431  0.798278 -0.137458  0.141267 -0.206010   
    
            V26       V27       V28  Amount  Class  
    0 -0.189115  0.133558 -0.021053  149.62      0  
    1  0.125895 -0.008983  0.014724    2.69      0  
    2 -0.139097 -0.055353 -0.059752  378.66      0  
    3 -0.221929  0.062723  0.061458  123.50      0  
    4  0.502292  0.219422  0.215153   69.99      0  
    
    [5 rows x 31 columns]



```python
card_df.info()
#Class 가 Target - Label : 0인 경우가 정상 이고 1이 사기
#Amount 가 사용 금액
#Time 은 구분하기 위한 인덱스 용도의 컬럼


#Class 만 자료형이 정수이고 나머지는 모두 실수
#데이터의 결측치는 없음
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 284807 entries, 0 to 284806
    Data columns (total 31 columns):
     #   Column  Non-Null Count   Dtype  
    ---  ------  --------------   -----  
     0   Time    284807 non-null  float64
     1   V1      284807 non-null  float64
     2   V2      284807 non-null  float64
     3   V3      284807 non-null  float64
     4   V4      284807 non-null  float64
     5   V5      284807 non-null  float64
     6   V6      284807 non-null  float64
     7   V7      284807 non-null  float64
     8   V8      284807 non-null  float64
     9   V9      284807 non-null  float64
     10  V10     284807 non-null  float64
     11  V11     284807 non-null  float64
     12  V12     284807 non-null  float64
     13  V13     284807 non-null  float64
     14  V14     284807 non-null  float64
     15  V15     284807 non-null  float64
     16  V16     284807 non-null  float64
     17  V17     284807 non-null  float64
     18  V18     284807 non-null  float64
     19  V19     284807 non-null  float64
     20  V20     284807 non-null  float64
     21  V21     284807 non-null  float64
     22  V22     284807 non-null  float64
     23  V23     284807 non-null  float64
     24  V24     284807 non-null  float64
     25  V25     284807 non-null  float64
     26  V26     284807 non-null  float64
     27  V27     284807 non-null  float64
     28  V28     284807 non-null  float64
     29  Amount  284807 non-null  float64
     30  Class   284807 non-null  int64  
    dtypes: float64(30), int64(1)
    memory usage: 67.4 MB



```python
## 훈련 데이터 와 테스트 데이터 만들기
from sklearn.model_selection import train_test_split

#대입된 DataFrame 을 복제한 후 Time 열을 제거하고 리턴하는 함수
def get_preprocessed_df(df=None):
    #파이썬에서 데이터 복제
    df_copy = df.copy()
    df_copy.drop('Time', axis=1, inplace=True)
    return df_copy

#훈련 세트 와 테스트 세트를 분할해서 리턴하는 함수
def get_train_test_dataset(df=None):
    #Time 열을 제거
    df_copy = get_preprocessed_df(df)
    
    #캐글의 분류 문제는 마지막 열이 label 입니다.
    X_features = df_copy.iloc[:, :-1]
    y_target = df_copy.iloc[:, -1]
    
    #데이터 분할
    #데이터가 많아서 test_size를 0.3 으로 설정
    #stratify 은 층화표본추출 옵션으로 y_target 의 분포를 가지고 데이터를
    #분할
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, 
                             test_size=0.3, random_state=42, stratify=y_target)
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_train_test_dataset(card_df)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
```

    (199364, 29)
    (85443, 29)
    (199364,)
    (85443,)



```python
#훈련 데이터의 레이블 비율 확인
print(y_train.value_counts()/y_train.count() * 100)
#테스트 데이터의 레이블 비율 확인
print(y_test.value_counts()/y_test.count() * 100)
```

    0    99.827451
    1     0.172549
    Name: Class, dtype: float64
    0    99.826785
    1     0.173215
    Name: Class, dtype: float64



```python
## 분류에서의 평가지표를 출력해주는 함수
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score

def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba)
    
    print('오차 행렬')
    print(confusion)
    print('정확도:', accuracy)
    print('정밀도:', precision)
    print('재현율:', recall)
    print('F1:', f1)
    print('AUC:', roc_auc)
```


```python
## Logistic Regression 을 이용해서 분류를 해보고 평가지표를 출력
from sklearn.linear_model import LogisticRegression

## 분류 모델을 만들기
lr_clf = LogisticRegression()

## 훈련 데이터를 이용해서 훈련
lr_clf.fit(X_train, y_train)
## 테스트 데이터를 이용해서 예측
lr_pred = lr_clf.predict(X_test) #클래스를 예측
lr_pred_proba = lr_clf.predict_proba(X_test)[:, 1]#확률을 예측

#평가지표를 확인 - SVM 은 생성할 때 파라미터를 조절하지 않으면 AUC는 출력 못함
get_clf_eval(y_test, lr_pred, lr_pred_proba)
```

    오차 행렬
    [[85277    18]
     [   56    92]]
    정확도: 0.999133925541004
    정밀도: 0.8363636363636363
    재현율: 0.6216216216216216
    F1: 0.7131782945736433
    AUC: 0.9195280924866481


    C:\Users\admin\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:763: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(



```python
## 훈련 과 예측 과 평가지표를 출력해주는 함수
def get_model_train_eval(model, ftr_train=None, ftr_test=None, 
                         tgt_train=None, tgt_test=None):
    model.fit(ftr_train, tgt_train)
    pred = model.predict(ftr_test)
    pred_proba = model.predict_proba(ftr_test)[:,1]
    get_clf_eval(tgt_test, pred, pred_proba)
```


```python
lr_clf = LogisticRegression()
get_model_train_eval(lr_clf, X_train, X_test, y_train, y_test)
```

    오차 행렬
    [[85277    18]
     [   56    92]]
    정확도: 0.999133925541004
    정밀도: 0.8363636363636363
    재현율: 0.6216216216216216
    F1: 0.7131782945736433
    AUC: 0.9195280924866481


    C:\Users\admin\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:763: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(



```python
## LightGBM 수행
from lightgbm import LGBMClassifier

## 타겟의 레이블이 불균형 분포를 이룰 때는 boost_from_average를 False 로 설정
lgbm_clf = LGBMClassifier(n_estimators=1000, num_leaves=64,
                          boost_from_average=False)
get_model_train_eval(lgbm_clf, X_train, X_test, y_train, y_test)

#이 데이터에서는 로지스틱 회귀 보다는 LGBM 이 더 좋은 성능을 나타낼 가능성이 높음
#평가지표들이 한쪽이 일방적으로 좋은게 아닌 경우에는 내가 하고자 하는 업무와
#평가지표를 고려해서 선택을 해야 합니다.
```

    오차 행렬
    [[85289     6]
     [   34   114]]
    정확도: 0.9995318516437859
    정밀도: 0.95
    재현율: 0.7702702702702703
    F1: 0.8507462686567164
    AUC: 0.9696482636572912



```python
## 피처들의 데이터 분포 확인
card_df.describe()
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
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>284807.000000</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>...</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>284807.000000</td>
      <td>284807.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>94813.859575</td>
      <td>3.918649e-15</td>
      <td>5.682686e-16</td>
      <td>-8.761736e-15</td>
      <td>2.811118e-15</td>
      <td>-1.552103e-15</td>
      <td>2.040130e-15</td>
      <td>-1.698953e-15</td>
      <td>-1.893285e-16</td>
      <td>-3.147640e-15</td>
      <td>...</td>
      <td>1.473120e-16</td>
      <td>8.042109e-16</td>
      <td>5.282512e-16</td>
      <td>4.456271e-15</td>
      <td>1.426896e-15</td>
      <td>1.701640e-15</td>
      <td>-3.662252e-16</td>
      <td>-1.217809e-16</td>
      <td>88.349619</td>
      <td>0.001727</td>
    </tr>
    <tr>
      <th>std</th>
      <td>47488.145955</td>
      <td>1.958696e+00</td>
      <td>1.651309e+00</td>
      <td>1.516255e+00</td>
      <td>1.415869e+00</td>
      <td>1.380247e+00</td>
      <td>1.332271e+00</td>
      <td>1.237094e+00</td>
      <td>1.194353e+00</td>
      <td>1.098632e+00</td>
      <td>...</td>
      <td>7.345240e-01</td>
      <td>7.257016e-01</td>
      <td>6.244603e-01</td>
      <td>6.056471e-01</td>
      <td>5.212781e-01</td>
      <td>4.822270e-01</td>
      <td>4.036325e-01</td>
      <td>3.300833e-01</td>
      <td>250.120109</td>
      <td>0.041527</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>-5.640751e+01</td>
      <td>-7.271573e+01</td>
      <td>-4.832559e+01</td>
      <td>-5.683171e+00</td>
      <td>-1.137433e+02</td>
      <td>-2.616051e+01</td>
      <td>-4.355724e+01</td>
      <td>-7.321672e+01</td>
      <td>-1.343407e+01</td>
      <td>...</td>
      <td>-3.483038e+01</td>
      <td>-1.093314e+01</td>
      <td>-4.480774e+01</td>
      <td>-2.836627e+00</td>
      <td>-1.029540e+01</td>
      <td>-2.604551e+00</td>
      <td>-2.256568e+01</td>
      <td>-1.543008e+01</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>54201.500000</td>
      <td>-9.203734e-01</td>
      <td>-5.985499e-01</td>
      <td>-8.903648e-01</td>
      <td>-8.486401e-01</td>
      <td>-6.915971e-01</td>
      <td>-7.682956e-01</td>
      <td>-5.540759e-01</td>
      <td>-2.086297e-01</td>
      <td>-6.430976e-01</td>
      <td>...</td>
      <td>-2.283949e-01</td>
      <td>-5.423504e-01</td>
      <td>-1.618463e-01</td>
      <td>-3.545861e-01</td>
      <td>-3.171451e-01</td>
      <td>-3.269839e-01</td>
      <td>-7.083953e-02</td>
      <td>-5.295979e-02</td>
      <td>5.600000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>84692.000000</td>
      <td>1.810880e-02</td>
      <td>6.548556e-02</td>
      <td>1.798463e-01</td>
      <td>-1.984653e-02</td>
      <td>-5.433583e-02</td>
      <td>-2.741871e-01</td>
      <td>4.010308e-02</td>
      <td>2.235804e-02</td>
      <td>-5.142873e-02</td>
      <td>...</td>
      <td>-2.945017e-02</td>
      <td>6.781943e-03</td>
      <td>-1.119293e-02</td>
      <td>4.097606e-02</td>
      <td>1.659350e-02</td>
      <td>-5.213911e-02</td>
      <td>1.342146e-03</td>
      <td>1.124383e-02</td>
      <td>22.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>139320.500000</td>
      <td>1.315642e+00</td>
      <td>8.037239e-01</td>
      <td>1.027196e+00</td>
      <td>7.433413e-01</td>
      <td>6.119264e-01</td>
      <td>3.985649e-01</td>
      <td>5.704361e-01</td>
      <td>3.273459e-01</td>
      <td>5.971390e-01</td>
      <td>...</td>
      <td>1.863772e-01</td>
      <td>5.285536e-01</td>
      <td>1.476421e-01</td>
      <td>4.395266e-01</td>
      <td>3.507156e-01</td>
      <td>2.409522e-01</td>
      <td>9.104512e-02</td>
      <td>7.827995e-02</td>
      <td>77.165000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>172792.000000</td>
      <td>2.454930e+00</td>
      <td>2.205773e+01</td>
      <td>9.382558e+00</td>
      <td>1.687534e+01</td>
      <td>3.480167e+01</td>
      <td>7.330163e+01</td>
      <td>1.205895e+02</td>
      <td>2.000721e+01</td>
      <td>1.559499e+01</td>
      <td>...</td>
      <td>2.720284e+01</td>
      <td>1.050309e+01</td>
      <td>2.252841e+01</td>
      <td>4.584549e+00</td>
      <td>7.519589e+00</td>
      <td>3.517346e+00</td>
      <td>3.161220e+01</td>
      <td>3.384781e+01</td>
      <td>25691.160000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 31 columns</p>
</div>




```python
## Amount 데이터의 범위를 확인
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
plt.xticks(range(0,30000, 1000), rotation=90)
sns.distplot(card_df['Amount'])

## 데이터의 범위가 다른 데이터 들과 많이 다르고 대부분은 1000 이하이고 그 이상 
## 데이터는 드뭄 - 정규화를 고민
```

    C:\Users\admin\anaconda3\lib\site-packages\seaborn\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)





    <AxesSubplot:xlabel='Amount', ylabel='Density'>




    
![png](output_44_2.png)
    



```python
## 정규화를 수행
from sklearn.preprocessing import StandardScaler

## 이전 함수 수정 - 정규화
def get_preprocessed_df(df=None):
    #파이썬에서 데이터 복제
    df_copy = df.copy()
    
    #정규화 객체를 생성
    scaler = StandardScaler()
    
    #Amount 열에 정규화 수행
    amount_n = scaler.fit_transform(df_copy['Amount'].values.reshape(-1, 1))
    
    #정규화된 결과를 데이터프레임의 맨 앞에 추가
    df_copy.insert(0, 'Amount_Scaled', amount_n)
    #df_copy['Amount_Scaled'] = amount_n
    
    df_copy.drop('Time', axis=1, inplace=True)
    return df_copy

```


```python
## 정규화를 수행한 후 평가지표를 확인
X_train, X_test, y_train, y_test = get_train_test_dataset(card_df)

lr_clf = LogisticRegression()
get_model_train_eval(lr_clf, X_train, X_test, y_train, y_test)
## 오차 행렬을 확인해보면 이전보다 개선 된 것 처럼 보임

## 타겟의 레이블이 불균형 분포를 이룰 때는 boost_from_average를 False 로 설정
lgbm_clf = LGBMClassifier(n_estimators=1000, num_leaves=64,
                          boost_from_average=False)
get_model_train_eval(lgbm_clf, X_train, X_test, y_train, y_test)
## 오차 행렬을 확인해보면 이전보다 약간 않좋아짐
```

    C:\Users\admin\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:763: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(


    오차 행렬
    [[85280    15]
     [   54    94]]
    정확도: 0.9991924440855307
    정밀도: 0.8623853211009175
    재현율: 0.6351351351351351
    F1: 0.7315175097276264
    AUC: 0.9318271404648097
    오차 행렬
    [[85288     7]
     [   35   113]]
    정확도: 0.9995084442259752
    정밀도: 0.9416666666666667
    재현율: 0.7635135135135135
    F1: 0.8432835820895522
    AUC: 0.9667335780589781



```python
## 이전 함수 수정 - 로그 변환
def get_preprocessed_df(df=None):
    #파이썬에서 데이터 복제
    df_copy = df.copy()
    
    #Amount 열에 로그 변환 수행
    #log1p 는 log을 변환을 한 후 1을 더해주는 함수입니다.
    #회귀분석에서 레이블의 값을 로그 변환을 하면 훨씬 좋은 성능을 발휘하는 경우가 많음
    amount_n = np.log1p(df_copy['Amount'])
    
    #정규화된 결과를 데이터프레임의 맨 앞에 추가
    df_copy.insert(0, 'Amount_Scaled', amount_n)
    #df_copy['Amount_Scaled'] = amount_n
    
    df_copy.drop('Time', axis=1, inplace=True)
    return df_copy


## 로그변환를 수행한 후 평가지표를 확인
X_train, X_test, y_train, y_test = get_train_test_dataset(card_df)

lr_clf = LogisticRegression()
get_model_train_eval(lr_clf, X_train, X_test, y_train, y_test)
## 오차 행렬을 확인해보면 이전보다 개선 된 것 처럼 보임

## 타겟의 레이블이 불균형 분포를 이룰 때는 boost_from_average를 False 로 설정
lgbm_clf = LGBMClassifier(n_estimators=1000, num_leaves=64,
                          boost_from_average=False)
get_model_train_eval(lgbm_clf, X_train, X_test, y_train, y_test)
## 오차 행렬을 확인해보면 이전보다 약간 좋아짐
```

    C:\Users\admin\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:763: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(


    오차 행렬
    [[85267    28]
     [   48   100]]
    정확도: 0.9991105181231933
    정밀도: 0.78125
    재현율: 0.6756756756756757
    F1: 0.7246376811594203
    AUC: 0.9073559490670693
    오차 행렬
    [[85289     6]
     [   35   113]]
    정확도: 0.9995201479348805
    정밀도: 0.9495798319327731
    재현율: 0.7635135135135135
    F1: 0.846441947565543
    AUC: 0.9693003455416258



```python
# 피처들의 상관관계 확인
corr = card_df.corr()
print(corr)
```

                Time            V1            V2            V3            V4  \
    Time    1.000000  1.173963e-01 -1.059333e-02 -4.196182e-01 -1.052602e-01   
    V1      0.117396  1.000000e+00  4.135835e-16 -1.227819e-15 -9.215150e-16   
    V2     -0.010593  4.135835e-16  1.000000e+00  3.243764e-16 -1.121065e-15   
    V3     -0.419618 -1.227819e-15  3.243764e-16  1.000000e+00  4.711293e-16   
    V4     -0.105260 -9.215150e-16 -1.121065e-15  4.711293e-16  1.000000e+00   
    V5      0.173072  1.812612e-17  5.157519e-16 -6.539009e-17 -1.719944e-15   
    V6     -0.063016 -6.506567e-16  2.787346e-16  1.627627e-15 -7.491959e-16   
    V7      0.084714 -1.005191e-15  2.055934e-16  4.895305e-16 -4.104503e-16   
    V8     -0.036949 -2.433822e-16 -5.377041e-17 -1.268779e-15  5.697192e-16   
    V9     -0.008660 -1.513678e-16  1.978488e-17  5.568367e-16  6.923247e-16   
    V10     0.030617  7.388135e-17 -3.991394e-16  1.156587e-15  2.232685e-16   
    V11    -0.247689  2.125498e-16  1.975426e-16  1.576830e-15  3.459380e-16   
    V12     0.124348  2.053457e-16 -9.568710e-17  6.310231e-16 -5.625518e-16   
    V13    -0.065902 -2.425603e-17  6.295388e-16  2.807652e-16  1.303306e-16   
    V14    -0.098757 -5.020280e-16 -1.730566e-16  4.739859e-16  2.282280e-16   
    V15    -0.183453  3.547782e-16 -4.995814e-17  9.068793e-16  1.377649e-16   
    V16     0.011903  7.212815e-17  1.177316e-17  8.299445e-16 -9.614528e-16   
    V17    -0.073297 -3.879840e-16 -2.685296e-16  7.614712e-16 -2.699612e-16   
    V18     0.090438  3.230206e-17  3.284605e-16  1.509897e-16 -5.103644e-16   
    V19     0.028975  1.502024e-16 -7.118719e-18  3.463522e-16 -3.980557e-16   
    V20    -0.050866  4.654551e-16  2.506675e-16 -9.316409e-16 -1.857247e-16   
    V21     0.044736 -2.457409e-16 -8.480447e-17  5.706192e-17 -1.949553e-16   
    V22     0.144059 -4.290944e-16  1.526333e-16 -1.133902e-15 -6.276051e-17   
    V23     0.051142  6.168652e-16  1.634231e-16 -4.983035e-16  9.164206e-17   
    V24    -0.016182 -4.425156e-17  1.247925e-17  2.686834e-19  1.584638e-16   
    V25    -0.233083 -9.605737e-16 -4.478846e-16 -1.104734e-15  6.070716e-16   
    V26    -0.041407 -1.581290e-17  2.057310e-16 -1.238062e-16 -4.247268e-16   
    V27    -0.005135  1.198124e-16 -4.966953e-16  1.045747e-15  3.977061e-17   
    V28    -0.009413  2.083082e-15 -5.093836e-16  9.775546e-16 -2.761403e-18   
    Amount -0.010596 -2.277087e-01 -5.314089e-01 -2.108805e-01  9.873167e-02   
    Class  -0.012323 -1.013473e-01  9.128865e-02 -1.929608e-01  1.334475e-01   
    
                      V5            V6            V7            V8            V9  \
    Time    1.730721e-01 -6.301647e-02  8.471437e-02 -3.694943e-02 -8.660434e-03   
    V1      1.812612e-17 -6.506567e-16 -1.005191e-15 -2.433822e-16 -1.513678e-16   
    V2      5.157519e-16  2.787346e-16  2.055934e-16 -5.377041e-17  1.978488e-17   
    V3     -6.539009e-17  1.627627e-15  4.895305e-16 -1.268779e-15  5.568367e-16   
    V4     -1.719944e-15 -7.491959e-16 -4.104503e-16  5.697192e-16  6.923247e-16   
    V5      1.000000e+00  2.408382e-16  2.715541e-16  7.437229e-16  7.391702e-16   
    V6      2.408382e-16  1.000000e+00  1.191668e-16 -1.104219e-16  4.131207e-16   
    V7      2.715541e-16  1.191668e-16  1.000000e+00  3.344412e-16  1.122501e-15   
    V8      7.437229e-16 -1.104219e-16  3.344412e-16  1.000000e+00  4.356078e-16   
    V9      7.391702e-16  4.131207e-16  1.122501e-15  4.356078e-16  1.000000e+00   
    V10    -5.202306e-16  5.932243e-17 -7.492834e-17 -2.801370e-16 -4.642274e-16   
    V11     7.203963e-16  1.980503e-15  1.425248e-16  2.487043e-16  1.354680e-16   
    V12     7.412552e-16  2.375468e-16 -3.536655e-18  1.839891e-16 -1.079314e-15   
    V13     5.886991e-16 -1.211182e-16  1.266462e-17 -2.921856e-16  2.251072e-15   
    V14     6.565143e-16  2.621312e-16  2.607772e-16 -8.599156e-16  3.784757e-15   
    V15    -8.720275e-16 -1.531188e-15 -1.690540e-16  4.127777e-16 -1.051167e-15   
    V16     2.246261e-15  2.623672e-18  5.869302e-17 -5.254741e-16 -1.214086e-15   
    V17     1.281914e-16  2.015618e-16  2.177192e-16 -2.269549e-16  1.113695e-15   
    V18     5.308590e-16  1.223814e-16  7.604126e-17 -3.667974e-16  4.993240e-16   
    V19    -1.450421e-16 -1.865597e-16 -1.881008e-16 -3.875186e-16 -1.376135e-16   
    V20    -3.554057e-16 -1.858755e-16  9.379684e-16  2.033737e-16 -2.343720e-16   
    V21    -3.920976e-16  5.833316e-17 -2.027779e-16  3.892798e-16  1.936953e-16   
    V22     1.253751e-16 -4.705235e-19 -8.898922e-16  2.026927e-16 -7.071869e-16   
    V23    -8.428683e-18  1.046712e-16 -4.387401e-16  6.377260e-17 -5.214137e-16   
    V24    -1.149255e-15 -1.071589e-15  7.434913e-18 -1.047097e-16 -1.430343e-16   
    V25     4.808532e-16  4.562861e-16 -3.094082e-16 -4.653279e-16  6.757763e-16   
    V26     4.319541e-16 -1.357067e-16 -9.657637e-16 -1.727276e-16 -7.888853e-16   
    V27     6.590482e-16 -4.452461e-16 -1.782106e-15  1.299943e-16 -6.709655e-17   
    V28    -5.613951e-18  2.594754e-16 -2.776530e-16 -6.200930e-16  1.110541e-15   
    Amount -3.863563e-01  2.159812e-01  3.973113e-01 -1.030791e-01 -4.424560e-02   
    Class  -9.497430e-02 -4.364316e-02 -1.872566e-01  1.987512e-02 -9.773269e-02   
    
            ...           V21           V22           V23           V24  \
    Time    ...  4.473573e-02  1.440591e-01  5.114236e-02 -1.618187e-02   
    V1      ... -2.457409e-16 -4.290944e-16  6.168652e-16 -4.425156e-17   
    V2      ... -8.480447e-17  1.526333e-16  1.634231e-16  1.247925e-17   
    V3      ...  5.706192e-17 -1.133902e-15 -4.983035e-16  2.686834e-19   
    V4      ... -1.949553e-16 -6.276051e-17  9.164206e-17  1.584638e-16   
    V5      ... -3.920976e-16  1.253751e-16 -8.428683e-18 -1.149255e-15   
    V6      ...  5.833316e-17 -4.705235e-19  1.046712e-16 -1.071589e-15   
    V7      ... -2.027779e-16 -8.898922e-16 -4.387401e-16  7.434913e-18   
    V8      ...  3.892798e-16  2.026927e-16  6.377260e-17 -1.047097e-16   
    V9      ...  1.936953e-16 -7.071869e-16 -5.214137e-16 -1.430343e-16   
    V10     ...  1.177547e-15 -6.418202e-16  3.214491e-16 -1.355885e-16   
    V11     ... -5.658364e-16  7.772895e-16 -4.505332e-16  1.933267e-15   
    V12     ...  7.300527e-16  1.644699e-16  1.800885e-16  4.436512e-16   
    V13     ...  1.008461e-16  6.747721e-17 -7.132064e-16 -1.397470e-16   
    V14     ... -3.356561e-16  3.740383e-16  3.883204e-16  2.003482e-16   
    V15     ...  6.605263e-17 -4.208921e-16 -3.912243e-16 -4.478263e-16   
    V16     ... -4.715090e-16 -7.923387e-17  5.020770e-16 -3.005985e-16   
    V17     ... -8.230527e-16 -8.743398e-16  3.706214e-16 -2.403828e-16   
    V18     ... -9.408680e-16 -4.819365e-16 -1.912006e-16 -8.986916e-17   
    V19     ...  5.115885e-16 -1.163768e-15  7.032035e-16  2.587708e-17   
    V20     ... -7.614597e-16  1.009285e-15  2.712885e-16  1.277215e-16   
    V21     ...  1.000000e+00  3.649908e-15  8.119580e-16  1.761054e-16   
    V22     ...  3.649908e-15  1.000000e+00 -7.303916e-17  9.970809e-17   
    V23     ...  8.119580e-16 -7.303916e-17  1.000000e+00  2.130519e-17   
    V24     ...  1.761054e-16  9.970809e-17  2.130519e-17  1.000000e+00   
    V25     ... -1.686082e-16 -5.018575e-16 -8.232727e-17  1.015391e-15   
    V26     ... -5.557329e-16 -2.503187e-17  1.114524e-15  1.343722e-16   
    V27     ... -1.211281e-15  8.461337e-17  2.839721e-16 -2.274142e-16   
    V28     ...  5.278775e-16 -6.627203e-16  1.481903e-15 -2.819805e-16   
    Amount  ...  1.059989e-01 -6.480065e-02 -1.126326e-01  5.146217e-03   
    Class   ...  4.041338e-02  8.053175e-04 -2.685156e-03 -7.220907e-03   
    
                     V25           V26           V27           V28    Amount  \
    Time   -2.330828e-01 -4.140710e-02 -5.134591e-03 -9.412688e-03 -0.010596   
    V1     -9.605737e-16 -1.581290e-17  1.198124e-16  2.083082e-15 -0.227709   
    V2     -4.478846e-16  2.057310e-16 -4.966953e-16 -5.093836e-16 -0.531409   
    V3     -1.104734e-15 -1.238062e-16  1.045747e-15  9.775546e-16 -0.210880   
    V4      6.070716e-16 -4.247268e-16  3.977061e-17 -2.761403e-18  0.098732   
    V5      4.808532e-16  4.319541e-16  6.590482e-16 -5.613951e-18 -0.386356   
    V6      4.562861e-16 -1.357067e-16 -4.452461e-16  2.594754e-16  0.215981   
    V7     -3.094082e-16 -9.657637e-16 -1.782106e-15 -2.776530e-16  0.397311   
    V8     -4.653279e-16 -1.727276e-16  1.299943e-16 -6.200930e-16 -0.103079   
    V9      6.757763e-16 -7.888853e-16 -6.709655e-17  1.110541e-15 -0.044246   
    V10    -2.846052e-16 -3.028119e-16 -2.197977e-16  4.864782e-17 -0.101502   
    V11    -5.600475e-16 -1.003221e-16 -2.640281e-16 -3.792314e-16  0.000104   
    V12    -5.712973e-16 -2.359969e-16 -4.672391e-16  6.415167e-16 -0.009542   
    V13    -5.497612e-16 -1.769255e-16 -4.720898e-16  1.144372e-15  0.005293   
    V14    -8.547932e-16 -1.660327e-16  1.044274e-16  2.289427e-15  0.033751   
    V15     3.206423e-16  2.817791e-16 -1.143519e-15 -1.194130e-15 -0.002986   
    V16    -1.345418e-15 -7.290010e-16  6.789513e-16  7.588849e-16 -0.003910   
    V17     2.666806e-16  6.932833e-16  6.148525e-16 -5.534540e-17  0.007309   
    V18    -6.629212e-17  2.990167e-16  2.242791e-16  7.976796e-16  0.035650   
    V19     9.577163e-16  5.898033e-16 -2.959370e-16 -1.405379e-15 -0.056151   
    V20     1.410054e-16 -2.803504e-16 -1.138829e-15 -2.436795e-16  0.339403   
    V21    -1.686082e-16 -5.557329e-16 -1.211281e-15  5.278775e-16  0.105999   
    V22    -5.018575e-16 -2.503187e-17  8.461337e-17 -6.627203e-16 -0.064801   
    V23    -8.232727e-17  1.114524e-15  2.839721e-16  1.481903e-15 -0.112633   
    V24     1.015391e-15  1.343722e-16 -2.274142e-16 -2.819805e-16  0.005146   
    V25     1.000000e+00  2.646517e-15 -6.406679e-16 -7.008939e-16 -0.047837   
    V26     2.646517e-15  1.000000e+00 -3.667715e-16 -2.782204e-16 -0.003208   
    V27    -6.406679e-16 -3.667715e-16  1.000000e+00 -3.061287e-16  0.028825   
    V28    -7.008939e-16 -2.782204e-16 -3.061287e-16  1.000000e+00  0.010258   
    Amount -4.783686e-02 -3.208037e-03  2.882546e-02  1.025822e-02  1.000000   
    Class   3.307706e-03  4.455398e-03  1.757973e-02  9.536041e-03  0.005632   
    
               Class  
    Time   -0.012323  
    V1     -0.101347  
    V2      0.091289  
    V3     -0.192961  
    V4      0.133447  
    V5     -0.094974  
    V6     -0.043643  
    V7     -0.187257  
    V8      0.019875  
    V9     -0.097733  
    V10    -0.216883  
    V11     0.154876  
    V12    -0.260593  
    V13    -0.004570  
    V14    -0.302544  
    V15    -0.004223  
    V16    -0.196539  
    V17    -0.326481  
    V18    -0.111485  
    V19     0.034783  
    V20     0.020090  
    V21     0.040413  
    V22     0.000805  
    V23    -0.002685  
    V24    -0.007221  
    V25     0.003308  
    V26     0.004455  
    V27     0.017580  
    V28     0.009536  
    Amount  0.005632  
    Class   1.000000  
    
    [31 rows x 31 columns]



```python
## 피처가 많은 경우에는 heatmap 을 이용해서 상관관계를 시각화
import seaborn as sns

plt.figure(figsize=(9,9))
sns.heatmap(corr, cmap='RdBu')
```




    <AxesSubplot:>




    
![png](output_49_1.png)
    



```python
##Class 와의 상관관계 확인
print(corr['Class'])

##피처가 많을 때는 레이블과 상관계수의 절대값이 적은 피처를 제거하고 해보는 것도좋고
##피처들끼리 상관계수의 절대값이 높은 경우에는 피처를 제거하거나 
##하나로 모아서 처리하는 것이 좋은 성능을 나타낼 가능성이 높습니다.
##이러한 작업을 차원 축소 라고 합니다.
```

    Time     -0.012323
    V1       -0.101347
    V2        0.091289
    V3       -0.192961
    V4        0.133447
    V5       -0.094974
    V6       -0.043643
    V7       -0.187257
    V8        0.019875
    V9       -0.097733
    V10      -0.216883
    V11       0.154876
    V12      -0.260593
    V13      -0.004570
    V14      -0.302544
    V15      -0.004223
    V16      -0.196539
    V17      -0.326481
    V18      -0.111485
    V19       0.034783
    V20       0.020090
    V21       0.040413
    V22       0.000805
    V23      -0.002685
    V24      -0.007221
    V25       0.003308
    V26       0.004455
    V27       0.017580
    V28       0.009536
    Amount    0.005632
    Class     1.000000
    Name: Class, dtype: float64



```python
## 이상치(Outlier) 처리
## 가장 많이 사용하는 방법은 IQR(3사분위수 - 1사분위수) 을 이용하는 것인데
## IQR 에 1.5 정도를 곱해서 이 범위 바깥의 데이터를 이상치로 간주

## 데이터가 정규분포(가우시안 분포)를 따른다고 하면 
##평균에 표준편차에 1를 곱한 범위 내에 약 68.26% 정도의 데이터가 위치하고
##평균에 표준편차에 2를 곱한 범위 내에 약 95.44% 정도의 데이터가 위치하고
##평균에 표준편차에 1.96를 곱한 범위 내에 약 95% 정도의 데이터가 위치하고
##평균에 표준편차에 3를 곱한 범위 내에 약 99.74% 정도의 데이터가 위치합니다.


##IQR 을 이용해서 이상치를 조회해주는 함수
def get_outlier(df=None, column=None, weight=1.5):
    #1사분위 수와 3사분위 수 구하기
    fraud = df[df['Class'] == 1][column]
    quantile_25 = np.percentile(fraud.values, 25)
    quantile_75 = np.percentile(fraud.values, 75)
    
    #IQR 구하기
    iqr = quantile_75 - quantile_25
    #IQR 에 1.5를 곱하기
    iqr_weight = iqr * weight

    #하한과 상한을 만들기
    lowest_val = quantile_25 - iqr_weight
    highest_val = quantile_75 + iqr_weight

    #하한보다 작거나 상한보다 큰 데이터의 인덱스 가져오기
    outlier_index = fraud[(fraud < lowest_val) | (fraud > highest_val)].index
    return outlier_index
```


```python
## 이상치를 조회 - Class 와 상관관계가 가장 깊은 feature 만 수행
outlier_index = get_outlier(df = card_df, column='V17', weight=1.5)
print('V17 에서의 이상치:',outlier_index )

outlier_index = get_outlier(df = card_df, column='V14', weight=1.5)
print('V14 에서의 이상치:',outlier_index )
```

    V17 에서의 이상치: Int64Index([], dtype='int64')
    V14 에서의 이상치: Int64Index([8296, 8615, 9035, 9252], dtype='int64')



```python
## 이전 함수 수정 - 로그 변환
def get_preprocessed_df(df=None):
    #파이썬에서 데이터 복제
    df_copy = df.copy()
    
    #Amount 열에 로그 변환 수행
    #log1p 는 log을 변환을 한 후 1을 더해주는 함수입니다.
    #회귀분석에서 레이블의 값을 로그 변환을 하면 훨씬 좋은 성능을 발휘하는 경우가 많음
    amount_n = np.log1p(df_copy['Amount'])
    
    #정규화된 결과를 데이터프레임의 맨 앞에 추가
    df_copy.insert(0, 'Amount_Scaled', amount_n)
    #df_copy['Amount_Scaled'] = amount_n
    
    df_copy.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    #V14 열에서 이상치를 제거
    outlier_index = get_outlier(df=df_copy, column='V14', weight=1.5)
    df_copy.drop(outlier_index, axis=0, inplace=True)
    return df_copy


## 로그변환를 수행한 후 평가지표를 확인
X_train, X_test, y_train, y_test = get_train_test_dataset(card_df)

lr_clf = LogisticRegression()
get_model_train_eval(lr_clf, X_train, X_test, y_train, y_test)
## 오차 행렬을 확인해보면 이전보다 개선 된 것 처럼 보임

## 타겟의 레이블이 불균형 분포를 이룰 때는 boost_from_average를 False 로 설정
lgbm_clf = LGBMClassifier(n_estimators=1000, num_leaves=64,
                          boost_from_average=False)
get_model_train_eval(lgbm_clf, X_train, X_test, y_train, y_test)
## 오차 행렬을 확인해보면 이전보다 약간 좋아짐
```

    오차 행렬
    [[85281    14]
     [   52    94]]
    정확도: 0.9992275371308856
    정밀도: 0.8703703703703703
    재현율: 0.6438356164383562
    F1: 0.7401574803149606
    AUC: 0.9733605448295078
    오차 행렬
    [[85289     6]
     [   27   119]]
    정확도: 0.9996137685654428
    정밀도: 0.952
    재현율: 0.815068493150685
    F1: 0.8782287822878229
    AUC: 0.9741089546593731


## 오버 샘플링 
### 오버 샘플링을 위한 패키지 - imbalanced-learn 패키지
!pip install imbalanced-learn

오버 샘플링은 훈련 데이터에만 합니다.
테스트 데이터에는 오버 샘플링을 하지 않습니다.
레이블의 클래스 분포가 불균형하면 학습 모델이 제대로 학습을 못하게 되는 것입니다.



```python
#오버샘플링 하기
from imblearn.over_sampling import SMOTE

#오버샘플링 객체를 생성
smoth = SMOTE(random_state = 42)

X_train_over, y_train_over = smoth.fit_resample(X_train, y_train)
```


```python
#원래 데이터와 오버샘플링된 데이터 개수 확인
print(X_train.shape)
print(X_train_over.shape)
```

    (199362, 29)
    (398040, 29)



```python
#비율 확인
print(pd.Series(y_train).value_counts())
print(pd.Series(y_train_over).value_counts())
```

    0    199020
    1       342
    Name: Class, dtype: int64
    0    199020
    1    199020
    Name: Class, dtype: int64



```python
#오버 샘플링된 데이터를 가지고 로지스틱 회귀를 수행
lr_clf = LogisticRegression()
get_model_train_eval(lr_clf, X_train_over, X_test, y_train_over, y_test)
```

    오차 행렬
    [[83305  1990]
     [   16   130]]
    정확도: 0.9765218103720696
    정밀도: 0.06132075471698113
    재현율: 0.8904109589041096
    F1: 0.1147396293027361
    AUC: 0.9692843210549688



```python
## 타겟의 레이블이 불균형 분포를 이룰 때는 boost_from_average를 False 로 설정
lgbm_clf = LGBMClassifier(n_estimators=1000, num_leaves=64,
                          boost_from_average=False)
get_model_train_eval(lgbm_clf, X_train_over, X_test, y_train_over, y_test)

```

    오차 행렬
    [[85279    16]
     [   27   119]]
    정확도: 0.9994967287367891
    정밀도: 0.8814814814814815
    재현율: 0.815068493150685
    F1: 0.8469750889679716
    AUC: 0.9699531922650398


## 오버샘플링을 하게되면 재현율은 증가하지만 정밀도가 떨어집니다.
## 좋은 오버샘플링은 재현율은 증가시키면서 정밀도 감소를 적게 해야 합니다.


```python

```


```python

```

# 신용 위험 모델링

## 데이터 가져오기


```python
#경고 제거하기
import warnings
warnings.filterwarnings('ignore')

#첫번째 열을 제거하고 데이터 가져오기
training_data = pd.read_csv("./data/cs-training.csv")
print(training_data.info())

training_data.drop('Unnamed: 0', axis=1, inplace=True)
print(training_data.info())


#컬럼이름에서 -를 제거하고 모두 소문자로 변경하기
#새로운 컬럼이름을 저장하기 위한 list
cleancolumn = []

#0부터 10까지(len 은 길이 - 11)
for i in range(len(training_data.columns)):
    cleancolumn.append(training_data.columns[i].replace('-', '').lower())

#새로 만든 컬럼 이름들을 데이터프레임에 반영
training_data.columns = cleancolumn

print(training_data.head())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150000 entries, 0 to 149999
    Data columns (total 12 columns):
     #   Column                                Non-Null Count   Dtype  
    ---  ------                                --------------   -----  
     0   Unnamed: 0                            150000 non-null  int64  
     1   SeriousDlqin2yrs                      150000 non-null  int64  
     2   RevolvingUtilizationOfUnsecuredLines  150000 non-null  float64
     3   age                                   150000 non-null  int64  
     4   NumberOfTime30-59DaysPastDueNotWorse  150000 non-null  int64  
     5   DebtRatio                             150000 non-null  float64
     6   MonthlyIncome                         120269 non-null  float64
     7   NumberOfOpenCreditLinesAndLoans       150000 non-null  int64  
     8   NumberOfTimes90DaysLate               150000 non-null  int64  
     9   NumberRealEstateLoansOrLines          150000 non-null  int64  
     10  NumberOfTime60-89DaysPastDueNotWorse  150000 non-null  int64  
     11  NumberOfDependents                    146076 non-null  float64
    dtypes: float64(4), int64(8)
    memory usage: 13.7 MB
    None
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150000 entries, 0 to 149999
    Data columns (total 11 columns):
     #   Column                                Non-Null Count   Dtype  
    ---  ------                                --------------   -----  
     0   SeriousDlqin2yrs                      150000 non-null  int64  
     1   RevolvingUtilizationOfUnsecuredLines  150000 non-null  float64
     2   age                                   150000 non-null  int64  
     3   NumberOfTime30-59DaysPastDueNotWorse  150000 non-null  int64  
     4   DebtRatio                             150000 non-null  float64
     5   MonthlyIncome                         120269 non-null  float64
     6   NumberOfOpenCreditLinesAndLoans       150000 non-null  int64  
     7   NumberOfTimes90DaysLate               150000 non-null  int64  
     8   NumberRealEstateLoansOrLines          150000 non-null  int64  
     9   NumberOfTime60-89DaysPastDueNotWorse  150000 non-null  int64  
     10  NumberOfDependents                    146076 non-null  float64
    dtypes: float64(4), int64(7)
    memory usage: 12.6 MB
    None
       seriousdlqin2yrs  revolvingutilizationofunsecuredlines  age  \
    0                 1                              0.766127   45   
    1                 0                              0.957151   40   
    2                 0                              0.658180   38   
    3                 0                              0.233810   30   
    4                 0                              0.907239   49   
    
       numberoftime3059dayspastduenotworse  debtratio  monthlyincome  \
    0                                    2   0.802982         9120.0   
    1                                    0   0.121876         2600.0   
    2                                    1   0.085113         3042.0   
    3                                    0   0.036050         3300.0   
    4                                    1   0.024926        63588.0   
    
       numberofopencreditlinesandloans  numberoftimes90dayslate  \
    0                               13                        0   
    1                                4                        0   
    2                                2                        1   
    3                                5                        0   
    4                                7                        0   
    
       numberrealestateloansorlines  numberoftime6089dayspastduenotworse  \
    0                             6                                    0   
    1                             0                                    0   
    2                             0                                    0   
    3                             0                                    0   
    4                             1                                    0   
    
       numberofdependents  
    0                 2.0  
    1                 1.0  
    2                 0.0  
    3                 0.0  
    4                 0.0  


## EDA(탐색적 분석)

### 기술 통계 확인


```python
#피처들의 기술 통계량 확인
training_data[training_data.columns[1:]].describe()
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
      <th>revolvingutilizationofunsecuredlines</th>
      <th>age</th>
      <th>numberoftime3059dayspastduenotworse</th>
      <th>debtratio</th>
      <th>monthlyincome</th>
      <th>numberofopencreditlinesandloans</th>
      <th>numberoftimes90dayslate</th>
      <th>numberrealestateloansorlines</th>
      <th>numberoftime6089dayspastduenotworse</th>
      <th>numberofdependents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150000.000000</td>
      <td>150000.000000</td>
      <td>150000.000000</td>
      <td>150000.000000</td>
      <td>1.202690e+05</td>
      <td>150000.000000</td>
      <td>150000.000000</td>
      <td>150000.000000</td>
      <td>150000.000000</td>
      <td>146076.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.048438</td>
      <td>52.295207</td>
      <td>0.421033</td>
      <td>353.005076</td>
      <td>6.670221e+03</td>
      <td>8.452760</td>
      <td>0.265973</td>
      <td>1.018240</td>
      <td>0.240387</td>
      <td>0.757222</td>
    </tr>
    <tr>
      <th>std</th>
      <td>249.755371</td>
      <td>14.771866</td>
      <td>4.192781</td>
      <td>2037.818523</td>
      <td>1.438467e+04</td>
      <td>5.145951</td>
      <td>4.169304</td>
      <td>1.129771</td>
      <td>4.155179</td>
      <td>1.115086</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.029867</td>
      <td>41.000000</td>
      <td>0.000000</td>
      <td>0.175074</td>
      <td>3.400000e+03</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.154181</td>
      <td>52.000000</td>
      <td>0.000000</td>
      <td>0.366508</td>
      <td>5.400000e+03</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.559046</td>
      <td>63.000000</td>
      <td>0.000000</td>
      <td>0.868254</td>
      <td>8.249000e+03</td>
      <td>11.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>50708.000000</td>
      <td>109.000000</td>
      <td>98.000000</td>
      <td>329664.000000</td>
      <td>3.008750e+06</td>
      <td>58.000000</td>
      <td>98.000000</td>
      <td>54.000000</td>
      <td>98.000000</td>
      <td>20.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#분류에서의 레이블은 분포를 확인
print(training_data['seriousdlqin2yrs'].value_counts())

print(training_data['seriousdlqin2yrs'].value_counts() / 
      len(training_data['seriousdlqin2yrs']))

```

    0    139974
    1     10026
    Name: seriousdlqin2yrs, dtype: int64
    0    0.93316
    1    0.06684
    Name: seriousdlqin2yrs, dtype: float64



```python
#분류에서의 레이블은 분포를 그래프로 시각화 : 비율 차이에 대한 효과를 높임
sns.set()
total_len = len(training_data['seriousdlqin2yrs'])
#막대 그래프
sns.countplot(training_data['seriousdlqin2yrs']).set_title("Data Distribution")

ax = plt.gca()

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2, height + 2, 
           100*(height/total_len), fontsize=12, ha='center', va='bottom')

import platform
if platform.system() == 'Darwin':
    fontname = 'AppleGothic'
else:
    fontname = 'Malgun Gothic'
    
sns.set(font=fontname,
        rc={'axes.unicode_minus':False},
        style='darkgrid',
        font_scale=1.5)

ax.set_xlabel("Labels for seriousdlqin2yrs attribute")
ax.set_ylabel("레코드 개수")
plt.show()
```


    
![png](output_70_0.png)
    


### 결측치 확인


```python
training_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150000 entries, 0 to 149999
    Data columns (total 11 columns):
     #   Column                                Non-Null Count   Dtype  
    ---  ------                                --------------   -----  
     0   seriousdlqin2yrs                      150000 non-null  int64  
     1   revolvingutilizationofunsecuredlines  150000 non-null  float64
     2   age                                   150000 non-null  int64  
     3   numberoftime3059dayspastduenotworse   150000 non-null  int64  
     4   debtratio                             150000 non-null  float64
     5   monthlyincome                         120269 non-null  float64
     6   numberofopencreditlinesandloans       150000 non-null  int64  
     7   numberoftimes90dayslate               150000 non-null  int64  
     8   numberrealestateloansorlines          150000 non-null  int64  
     9   numberoftime6089dayspastduenotworse   150000 non-null  int64  
     10  numberofdependents                    146076 non-null  float64
    dtypes: float64(4), int64(7)
    memory usage: 12.6 MB



```python
training_data.isnull().sum()
```




    seriousdlqin2yrs                            0
    revolvingutilizationofunsecuredlines        0
    age                                         0
    numberoftime3059dayspastduenotworse         0
    debtratio                                   0
    monthlyincome                           29731
    numberofopencreditlinesandloans             0
    numberoftimes90dayslate                     0
    numberrealestateloansorlines                0
    numberoftime6089dayspastduenotworse         0
    numberofdependents                       3924
    dtype: int64




```python
### 결측치 처리 
```


```python
#평균으로 대체
training_data_mean_replace = training_data.fillna(training_data.mean())
training_data_mean_replace.isnull().sum()
```




    seriousdlqin2yrs                        0
    revolvingutilizationofunsecuredlines    0
    age                                     0
    numberoftime3059dayspastduenotworse     0
    debtratio                               0
    monthlyincome                           0
    numberofopencreditlinesandloans         0
    numberoftimes90dayslate                 0
    numberrealestateloansorlines            0
    numberoftime6089dayspastduenotworse     0
    numberofdependents                      0
    dtype: int64




```python
#중위값으로 대체
training_data_median_replace = training_data.fillna(training_data.median())
training_data_median_replace.isnull().sum()
```




    seriousdlqin2yrs                        0
    revolvingutilizationofunsecuredlines    0
    age                                     0
    numberoftime3059dayspastduenotworse     0
    debtratio                               0
    monthlyincome                           0
    numberofopencreditlinesandloans         0
    numberoftimes90dayslate                 0
    numberrealestateloansorlines            0
    numberoftime6089dayspastduenotworse     0
    numberofdependents                      0
    dtype: int64



### 상관관계 확인


```python
# 피처들 간의 상관 계수 확인
training_data.fillna(training_data.median(), inplace=True)
training_data[training_data.columns[1:]].corr()
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
      <th>revolvingutilizationofunsecuredlines</th>
      <th>age</th>
      <th>numberoftime3059dayspastduenotworse</th>
      <th>debtratio</th>
      <th>monthlyincome</th>
      <th>numberofopencreditlinesandloans</th>
      <th>numberoftimes90dayslate</th>
      <th>numberrealestateloansorlines</th>
      <th>numberoftime6089dayspastduenotworse</th>
      <th>numberofdependents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>revolvingutilizationofunsecuredlines</th>
      <td>1.000000</td>
      <td>-0.005898</td>
      <td>-0.001314</td>
      <td>0.003961</td>
      <td>0.006513</td>
      <td>-0.011281</td>
      <td>-0.001061</td>
      <td>0.006235</td>
      <td>-0.001048</td>
      <td>0.001193</td>
    </tr>
    <tr>
      <th>age</th>
      <td>-0.005898</td>
      <td>1.000000</td>
      <td>-0.062995</td>
      <td>0.024188</td>
      <td>0.027581</td>
      <td>0.147705</td>
      <td>-0.061005</td>
      <td>0.033150</td>
      <td>-0.057159</td>
      <td>-0.215693</td>
    </tr>
    <tr>
      <th>numberoftime3059dayspastduenotworse</th>
      <td>-0.001314</td>
      <td>-0.062995</td>
      <td>1.000000</td>
      <td>-0.006542</td>
      <td>-0.008370</td>
      <td>-0.055312</td>
      <td>0.983603</td>
      <td>-0.030565</td>
      <td>0.987005</td>
      <td>-0.004590</td>
    </tr>
    <tr>
      <th>debtratio</th>
      <td>0.003961</td>
      <td>0.024188</td>
      <td>-0.006542</td>
      <td>1.000000</td>
      <td>-0.018006</td>
      <td>0.049565</td>
      <td>-0.008320</td>
      <td>0.120046</td>
      <td>-0.007533</td>
      <td>-0.044476</td>
    </tr>
    <tr>
      <th>monthlyincome</th>
      <td>0.006513</td>
      <td>0.027581</td>
      <td>-0.008370</td>
      <td>-0.018006</td>
      <td>1.000000</td>
      <td>0.086949</td>
      <td>-0.010500</td>
      <td>0.116273</td>
      <td>-0.009252</td>
      <td>0.066314</td>
    </tr>
    <tr>
      <th>numberofopencreditlinesandloans</th>
      <td>-0.011281</td>
      <td>0.147705</td>
      <td>-0.055312</td>
      <td>0.049565</td>
      <td>0.086949</td>
      <td>1.000000</td>
      <td>-0.079984</td>
      <td>0.433959</td>
      <td>-0.071077</td>
      <td>0.074026</td>
    </tr>
    <tr>
      <th>numberoftimes90dayslate</th>
      <td>-0.001061</td>
      <td>-0.061005</td>
      <td>0.983603</td>
      <td>-0.008320</td>
      <td>-0.010500</td>
      <td>-0.079984</td>
      <td>1.000000</td>
      <td>-0.045205</td>
      <td>0.992796</td>
      <td>-0.011962</td>
    </tr>
    <tr>
      <th>numberrealestateloansorlines</th>
      <td>0.006235</td>
      <td>0.033150</td>
      <td>-0.030565</td>
      <td>0.120046</td>
      <td>0.116273</td>
      <td>0.433959</td>
      <td>-0.045205</td>
      <td>1.000000</td>
      <td>-0.039722</td>
      <td>0.129399</td>
    </tr>
    <tr>
      <th>numberoftime6089dayspastduenotworse</th>
      <td>-0.001048</td>
      <td>-0.057159</td>
      <td>0.987005</td>
      <td>-0.007533</td>
      <td>-0.009252</td>
      <td>-0.071077</td>
      <td>0.992796</td>
      <td>-0.039722</td>
      <td>1.000000</td>
      <td>-0.012678</td>
    </tr>
    <tr>
      <th>numberofdependents</th>
      <td>0.001193</td>
      <td>-0.215693</td>
      <td>-0.004590</td>
      <td>-0.044476</td>
      <td>0.066314</td>
      <td>0.074026</td>
      <td>-0.011962</td>
      <td>0.129399</td>
      <td>-0.012678</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#상관계수 시각화
sns.set()
sns.heatmap(training_data[training_data.columns[1:]].corr(), 
            annot=True, fmt='.2f', 
            cmap=(sns.cubehelix_palette(8, start=.5, rot=-7.5)))
plt.show()
```


    
![png](output_79_0.png)
    


### 이상치를 검출하는 함수


```python
#백분율 기반 - 기본 95% 밖의 데이터를 이상치로 간주
def percentile_based_outlier(data, threshold=95):
    diff = (100 - threshold) / 2
    (minval, maxval) = np.percentile(data, [diff, 100-diff])
    return ((data < minval) | (data > maxval))
    
```


```python
#IQR 기반 - 1사분위수에서 IQR*1.5 를 뺀 값부터 3사분위수에서 IQR*1.5를 더한값 을 
#정상적인 범위의 데이터로 간주
def outlier_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))
```


```python
#중위절대편차 이용
def mad_based_outlier(points, threshold=3.5):
    median_y = np.median(point)
    #중위 절대 편차
    median_absolute_derivation_y = 
        np.median([np.abs(y - median_y) for y in points])
    #z-score
    modified_z_scores = 
        [0.6745 * (y - median_y)/median_absolute_derivation_y for y in points]
    #z-score 가 3.5 보다 큰 것을 이상치로 간주
    return np.abs(modified_z_scores > threshold)
```


```python
#표준편차 이용 
#정규 분포는 표준 편차의 2배 이내에 약 95%, 3배 이내에 99% 정도의 데이터가 배치됨
def std_div(data, threshold=3):
    std = data.std()
    mean = data.mean()
    isOutlier = []
    for val in data:
        if val > mean * std * threshold or val < mean * std * (-1 * threshold):
            isOutlier.append(True)
        else:
            isOutlier.append(False)
    return isOutlier
```


```python
#다수결 이용
def outlier_vote(data):
    #3개의 함수를 호출해서 False 가 2개 이상이면 정상
    #그렇지 않으면 이상치로 간주
    x = percentile_based_outlier(data)
    y = mad_based_outlier(data)
    z = std_div(data)
    
    temp = list(zip(data.index, x, y, z))
    final = []
    
    for i in range(len(temp)):
        if temp[i].count(False) >= 2:
            final.append(False)
        else:
            final.append(True)
    return final
```

### 이상치 처리


```python
#revolvingutilizationofunsecuredlines
print(training_data.revolvingutilizationofunsecuredlines.describe())

#대부분 1.0 아래인데 아주 특별한 값(이상치) 때문에 평균이 6이 넘음
#0.99999 보다 큰 값을 0.99999 로 대체
revNew = []

for val in training_data.revolvingutilizationofunsecuredlines:
    if val <= 0.99999:
        revNew.append(val)
    else:
        revNew.append(0.99999)
training_data.revolvingutilizationofunsecuredlines = revNew

print(training_data.revolvingutilizationofunsecuredlines.describe())
```

    count    150000.000000
    mean          6.048438
    std         249.755371
    min           0.000000
    25%           0.029867
    50%           0.154181
    75%           0.559046
    max       50708.000000
    Name: revolvingutilizationofunsecuredlines, dtype: float64
    count    150000.000000
    mean          0.319195
    std           0.349480
    min           0.000000
    25%           0.029867
    50%           0.154181
    75%           0.559046
    max           0.999990
    Name: revolvingutilizationofunsecuredlines, dtype: float64



```python
#age 열의 값을 boxplot 으로 확인
training_data.age.plot.box()
```




    <AxesSubplot:>




    
![png](output_88_1.png)
    



```python
training_data.age.value_counts()
```




    49     3837
    48     3806
    50     3753
    63     3719
    47     3719
           ... 
    102       3
    109       2
    105       1
    107       1
    0         1
    Name: age, Length: 86, dtype: int64




```python
#0살인 데이터를 가장 많이 나온 49로 설정
ageNew = []

for val in training_data.age:
    if val < 20:
        ageNew.append(49)
    else:
        ageNew.append(val)
        
training_data.age = ageNew

print(training_data.age.describe())
```

    count    150000.000000
    mean         52.295533
    std          14.771251
    min          21.000000
    25%          41.000000
    50%          52.000000
    75%          63.000000
    max         109.000000
    Name: age, dtype: float64



```python
#numberoftime3059dayspastduenotworse 의 데이터에서 이상치를 중위수로 변경
training_data.numberoftime3059dayspastduenotworse.value_counts().sort_index()

#96 과 98 이라는 이상한 데이터를 발견
```




    0     126018
    1      16033
    2       4598
    3       1754
    4        747
    5        342
    6        140
    7         54
    8         25
    9         12
    10         4
    11         1
    12         2
    13         1
    96         5
    98       264
    Name: numberoftime3059dayspastduenotworse, dtype: int64




```python
#numberoftime3059dayspastduenotworse 의 값이 96 이나 98인 값을 중위수로 대체
newNum = []

med = training_data.numberoftime3059dayspastduenotworse.median()

for val in training_data.numberoftime3059dayspastduenotworse:
    if(val == 96) | (val == 98):
        newNum.append(med)
    else:
        newNum.append(val)
training_data.numberoftime3059dayspastduenotworse = newNum

training_data.numberoftime3059dayspastduenotworse.value_counts().sort_index()
```




    0.0     126287
    1.0      16033
    2.0       4598
    3.0       1754
    4.0        747
    5.0        342
    6.0        140
    7.0         54
    8.0         25
    9.0         12
    10.0         4
    11.0         1
    12.0         2
    13.0         1
    Name: numberoftime3059dayspastduenotworse, dtype: int64




```python
#numberoftimes90dayslate 열의 이상치 확인
training_data.numberoftimes90dayslate.value_counts().sort_index()
```




    0     141662
    1       5243
    2       1555
    3        667
    4        291
    5        131
    6         80
    7         38
    8         21
    9         19
    10         8
    11         5
    12         2
    13         4
    14         2
    15         2
    17         1
    96         5
    98       264
    Name: numberoftimes90dayslate, dtype: int64




```python
#numberoftimes90dayslate 의 값이 96 이나 98인 값을 중위수로 대체
newNum = []

med = training_data.numberoftimes90dayslate.median()

for val in training_data.numberoftimes90dayslate:
    if(val == 96) | (val == 98):
        newNum.append(med)
    else:
        newNum.append(val)
training_data.numberoftimes90dayslate = newNum

training_data.numberoftimes90dayslate.value_counts().sort_index()
```




    0.0     141931
    1.0       5243
    2.0       1555
    3.0        667
    4.0        291
    5.0        131
    6.0         80
    7.0         38
    8.0         21
    9.0         19
    10.0         8
    11.0         5
    12.0         2
    13.0         4
    14.0         2
    15.0         2
    17.0         1
    Name: numberoftimes90dayslate, dtype: int64




```python
#numberrealestateloansorlines 이상치 확인 및 처리
#담보 대출 및 한도 대출 횟수
training_data.numberrealestateloansorlines.value_counts().sort_index()


#17이 넘는 데이터를 17로 대체
newNum = []

med = 17

for val in training_data.numberrealestateloansorlines:
    ifval > 17:
        newNum.append(med)
    else:
        newNum.append(val)
training_data.numberrealestateloansorlines = newNum

training_data.numberrealestateloansorlines.value_counts().sort_index()
```




    0     56188
    1     52338
    2     31522
    3      6300
    4      2170
    5       689
    6       320
    7       171
    8        93
    9        78
    10       37
    11       23
    12       18
    13       15
    14        7
    15        7
    16        4
    17        4
    18        2
    19        2
    20        2
    21        1
    23        2
    25        3
    26        1
    29        1
    32        1
    54        1
    Name: numberrealestateloansorlines, dtype: int64




```python
training_data.numberoftime6089dayspastduenotworse.value_counts().sort_index()    
```




    0     142396
    1       5731
    2       1118
    3        318
    4        105
    5         34
    6         16
    7          9
    8          2
    9          1
    11         1
    96         5
    98       264
    Name: numberoftime6089dayspastduenotworse, dtype: int64



### 특성 중요도 확인


```python
from sklearn.ensemble import RandomForestClassifier

#타깃 속성과 피처 분리
X = training_data.drop('seriousdlqin2yrs', axis=1)
y = training_data.seriousdlqin2yrs

feature_label = training_data.columns[1:]

forest = RandomForestClassifier(n_estimators=10000, random_state=42, n_jobs=-1)
forest.fit(X, y)
```




    RandomForestClassifier(n_estimators=10000, n_jobs=-1, random_state=42)




```python
#특성 중요도 저장
importances = forest.feature_importances_
#정렬
indices = np.argsort(importances)[::-1]
#특성 중요도 출력
for i in range(X.shape[1]):
    print(feature_label[i], importances[indices[i]])
```

    revolvingutilizationofunsecuredlines 0.18569006887706263
    age 0.17715137822024382
    numberoftime3059dayspastduenotworse 0.1497934914944211
    debtratio 0.13065687537982196
    monthlyincome 0.09132624824840796
    numberofopencreditlinesandloans 0.08853404348394732
    numberoftimes90dayslate 0.05106902982911809
    numberrealestateloansorlines 0.04891730662314978
    numberoftime6089dayspastduenotworse 0.042659149108972665
    numberofdependents 0.03420240873485472



```python
#특성 중요도 시각화
plt.title('Feature Importance')
plt.bar(range(X.shape[1]), importances[indices], color='green', align='center')
plt.xticks(range(X.shape[1]), feature_label, rotation=90)
plt.show()
```


    
![png](output_100_0.png)
    



```python
# 훈련 데이터와 테스트 데이터를 분리
from sklearn.model_selection import train_test_split

X = training_data.drop('seriousdlqin2yrs', axis=1)
y = training_data.seriousdlqin2yrs

X_train, X_test, y_train, y_test =  train_test_split(X, y,
                                test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
```

    (120000, 10)
    (30000, 10)
    (120000,)
    (30000,)



```python
# 평가지표 확인
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-107-40e379f0ee7b> in <module>
          1 # 평가지표 확인
          2 from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
    ----> 3 print('선형 회귀 정확도:', accuracy_score(y_test,lin_reg_pred ))
    

    ~\anaconda3\lib\site-packages\sklearn\utils\validation.py in inner_f(*args, **kwargs)
         61             extra_args = len(args) - len(all_args)
         62             if extra_args <= 0:
    ---> 63                 return f(*args, **kwargs)
         64 
         65             # extra_args > 0


    ~\anaconda3\lib\site-packages\sklearn\metrics\_classification.py in accuracy_score(y_true, y_pred, normalize, sample_weight)
        200 
        201     # Compute accuracy for each possible representation
    --> 202     y_type, y_true, y_pred = _check_targets(y_true, y_pred)
        203     check_consistent_length(y_true, y_pred, sample_weight)
        204     if y_type.startswith('multilabel'):


    ~\anaconda3\lib\site-packages\sklearn\metrics\_classification.py in _check_targets(y_true, y_pred)
         90 
         91     if len(y_type) > 1:
    ---> 92         raise ValueError("Classification metrics can't handle a mix of {0} "
         93                          "and {1} targets".format(type_true, type_pred))
         94 


    ValueError: Classification metrics can't handle a mix of binary and continuous targets



```python
# 경사 하강법을 사용하는 선형 분류 - 데이터가 선형으로 많은 경우 사용
from sklearn.linear_model import SGDClassifier

sgd_reg = SGDClassifier(max_iter=1000, tol=1e-3, random_state = 42)
sgd_reg.fit(X_train, y_train)
sgd_reg_pred = sgd_reg.predict(X_test)

print('경사하강법 선형 분류 정확도:', accuracy_score(y_test,sgd_reg_pred))
print('경사하강법 선형 분류 정밀도:', precision_score(y_test,sgd_reg_pred))
print('경사하강법 선형 분류 재현율:', recall_score(y_test,sgd_reg_pred))
print('경사하강법 선형 분류 F1:', f1_score(y_test,sgd_reg_pred))
```

    경사하강법 선형 분류 정확도: 0.9243333333333333
    경사하강법 선형 분류 정밀도: 0.05142857142857143
    경사하강법 선형 분류 재현율: 0.009202453987730062
    경사하강법 선형 분류 F1: 0.015611448395490026



```python
# 랜덤 분류기
from sklearn.dummy import DummyClassifier

dummy_reg = DummyClassifier(strategy='stratified', random_state = 42)
dummy_reg.fit(X_train, y_train)
dummy_reg_pred = dummy_reg.predict(X_test)

print('랜덤 분류 정확도:', accuracy_score(y_test,dummy_reg_pred))
print('랜덤 분류 정밀도:', precision_score(y_test,dummy_reg_pred))
print('랜덤 분류 재현율:', recall_score(y_test,dummy_reg_pred))
print('랜덤 분류 F1:', f1_score(y_test,dummy_reg_pred))
```

    랜덤 분류 정확도: 0.8754666666666666
    랜덤 분류 정밀도: 0.06071076011846002
    랜덤 분류 재현율: 0.06288343558282208
    랜덤 분류 F1: 0.06177800100452034



```python
#KNN - 결측치 설정 등에 많이 사용, 정규화를 반드시 수행해 주어야 합니다.
from sklearn.neighbors import KNeighborsClassifier

knn_reg = KNeighborsClassifier(n_neighbors=20)
knn_reg.fit(X_train, y_train)
knn_reg_pred = knn_reg.predict(X_test)

print('KNN 분류 정확도:', accuracy_score(y_test,knn_reg_pred))
print('KNN 분류 정밀도:', precision_score(y_test,knn_reg_pred))
print('KNN 분류 재현율:', recall_score(y_test,knn_reg_pred))
print('KNN 분류 F1:', f1_score(y_test,knn_reg_pred))
```

    KNN 분류 정확도: 0.9350333333333334
    KNN 분류 정밀도: 0.8181818181818182
    KNN 분류 재현율: 0.004601226993865031
    KNN 분류 F1: 0.009150991357397052



```python
#Naive Bayes - 텍스트 분류에서 많이 이용, 등장 확률을 가지고 분류
from sklearn.naive_bayes import MultinomialNB

#fit_prior 를 True 로 설정하면 분포가 균등하지 않은 것이고 False 이면 균등
nb_reg = MultinomialNB(alpha=0.1, fit_prior=True)
nb_reg.fit(X_train, y_train)
nb_reg_pred = nb_reg.predict(X_test)

print('Naive Bayes 분류 정확도:', accuracy_score(y_test,nb_reg_pred))
print('Naive Bayes 분류 정밀도:', precision_score(y_test,nb_reg_pred))
print('Naive Bayes 분류 재현율:', recall_score(y_test,nb_reg_pred))
print('Naive Bayes 분류 F1:', f1_score(y_test,nb_reg_pred))
```

    KNN 분류 정확도: 0.21986666666666665
    KNN 분류 정밀도: 0.06789427028769442
    KNN 분류 재현율: 0.8614519427402862
    KNN 분류 F1: 0.12586837977142004



```python
#Logistic 회귀 - 회귀지만 분류에 이용
#odds 라는 값을 이용해서 분류
from sklearn.linear_model import LogisticRegression

#solver 가 solver 특이값 분해, sag 를 설정하면 경사하강법
logit_reg = LogisticRegression(solver='lbfgs', random_state=42)
logit_reg.fit(X_train, y_train)
logit_reg_pred = logit_reg.predict(X_test)

print('LogisticRegression 분류 정확도:', accuracy_score(y_test,logit_reg_pred))
print('LogisticRegression 분류 정밀도:', precision_score(y_test,logit_reg_pred))
print('LogisticRegression 분류 재현율:', recall_score(y_test,logit_reg_pred))
print('LogisticRegression 분류 F1:', f1_score(y_test,logit_reg_pred))
```

    Naive Bayes 분류 정확도: 0.9359666666666666
    Naive Bayes 분류 정밀도: 0.5343811394891945
    Naive Bayes 분류 재현율: 0.1390593047034765
    Naive Bayes 분류 F1: 0.22068965517241382



```python
#초평면을 이용하는 SVM
from sklearn.svm import SVC
#선형 SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
svm_model_pred = svm_model.predict(X_test)

print('선형 SVC 분류 정확도:', accuracy_score(y_test,svm_model_pred))
print('선형 SVC 분류 정밀도:', precision_score(y_test,svm_model_pred))
print('선형 SVC 분류 재현율:', recall_score(y_test,svm_model_pred))
print('선형 SVC 분류 F1:', f1_score(y_test,svm_model_pred))
```


```python
#초평면을 이용하는 SVM
from sklearn.svm import SVC
#비선형 SVM
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train)
svm_model_pred = svm_model.predict(X_test)

print('선형 SVC 분류 정확도:', accuracy_score(y_test,svm_model_pred))
print('선형 SVC 분류 정밀도:', precision_score(y_test,svm_model_pred))
print('선형 SVC 분류 재현율:', recall_score(y_test,svm_model_pred))
print('선형 SVC 분류 F1:', f1_score(y_test,svm_model_pred))
```
