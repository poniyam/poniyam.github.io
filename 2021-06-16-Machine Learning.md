---
layout: post
title:  "실습 예제 입니다."
---

# 머신러닝 개요


```python
import pandas as pd
import numpy as np

#그래프를 그리기 위한 설정
#화려한 시각화 와 통계를 적용한 시각화 그리고 기본 데이터셋을 제공하는 패키지
import seaborn as sns

#그래프를 그릴 때 한글을 사용하기 위한 설정
import matplotlib.pyplot as plt

import platform
from matplotlib import font_manager, rc

if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    font_name = font_manager.FontProperties(
        fname='c:/Windows/Fonts/malgun.ttf').get_name()
    rc('font', family=font_name)

#음수 출력 설정
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

#통계에 많이 사용하는 패키지
import scipy as sp

#경고 제거
import warnings
warnings.filterwarnings(action='ignore')
```

# 일반화에 따른 머신러닝 분류

## 모델 기반 학습 - 알고리즘을 생성해서 새로운 데이터가 오면 이 알고리즘에 대입해서 결과를 예측하는 방식


```python
#데이터 가져오기: oecd_bli_2015.csv, gdp_per_capita.csv - value 가 삶의 만족도
oecd_bli = pd.read_csv('./data/oecd_bli_2015.csv', thousands=',')
print(oecd_bli.head())
oecd_bli.info()

#구분자는 tab - 2015 가 1인당 GDP 
#delimiter 는 구분자로 기본값은 ,
#thousands 는 천단위 구분 기호
#encoding 은 인코딩 방식 - ms949(cp949-윈도우즈), utf-8, euc-kr(웹), latin(iso_latin-)
#na_values 는 NaN 으로 처리할 문자열을 설정
#구분자 탭 인 경우 na_values를 설정하는 경우가 종종 있음

gdp_per_capita = pd.read_csv('./data/gdp_per_capita.csv', delimiter='\t', 
                            thousands=',',  encoding='latin1',
                            na_values='n/a')
print(gdp_per_capita)
gdp_per_capita.info()
```

      LOCATION         Country INDICATOR                           Indicator  \
    0      AUS       Australia   HO_BASE  Dwellings without basic facilities   
    1      AUT         Austria   HO_BASE  Dwellings without basic facilities   
    2      BEL         Belgium   HO_BASE  Dwellings without basic facilities   
    3      CAN          Canada   HO_BASE  Dwellings without basic facilities   
    4      CZE  Czech Republic   HO_BASE  Dwellings without basic facilities   
    
      MEASURE Measure INEQUALITY Inequality Unit Code        Unit  PowerCode Code  \
    0       L   Value        TOT      Total        PC  Percentage               0   
    1       L   Value        TOT      Total        PC  Percentage               0   
    2       L   Value        TOT      Total        PC  Percentage               0   
    3       L   Value        TOT      Total        PC  Percentage               0   
    4       L   Value        TOT      Total        PC  Percentage               0   
    
      PowerCode  Reference Period Code  Reference Period  Value Flag Codes  \
    0     units                    NaN               NaN    1.1          E   
    1     units                    NaN               NaN    1.0        NaN   
    2     units                    NaN               NaN    2.0        NaN   
    3     units                    NaN               NaN    0.2        NaN   
    4     units                    NaN               NaN    0.9        NaN   
    
                 Flags  
    0  Estimated value  
    1              NaN  
    2              NaN  
    3              NaN  
    4              NaN  
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3292 entries, 0 to 3291
    Data columns (total 17 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   LOCATION               3292 non-null   object 
     1   Country                3292 non-null   object 
     2   INDICATOR              3292 non-null   object 
     3   Indicator              3292 non-null   object 
     4   MEASURE                3292 non-null   object 
     5   Measure                3292 non-null   object 
     6   INEQUALITY             3292 non-null   object 
     7   Inequality             3292 non-null   object 
     8   Unit Code              3292 non-null   object 
     9   Unit                   3292 non-null   object 
     10  PowerCode Code         3292 non-null   int64  
     11  PowerCode              3292 non-null   object 
     12  Reference Period Code  0 non-null      float64
     13  Reference Period       0 non-null      float64
     14  Value                  3292 non-null   float64
     15  Flag Codes             1120 non-null   object 
     16  Flags                  1120 non-null   object 
    dtypes: float64(3), int64(1), object(13)
    memory usage: 437.3+ KB
                                                   Country  \
    0                                          Afghanistan   
    1                                              Albania   
    2                                              Algeria   
    3                                               Angola   
    4                                  Antigua and Barbuda   
    ..                                                 ...   
    185                                            Vietnam   
    186                                              Yemen   
    187                                             Zambia   
    188                                           Zimbabwe   
    189  International Monetary Fund, World Economic Ou...   
    
                                        Subject Descriptor         Units  Scale  \
    0    Gross domestic product per capita, current prices  U.S. dollars  Units   
    1    Gross domestic product per capita, current prices  U.S. dollars  Units   
    2    Gross domestic product per capita, current prices  U.S. dollars  Units   
    3    Gross domestic product per capita, current prices  U.S. dollars  Units   
    4    Gross domestic product per capita, current prices  U.S. dollars  Units   
    ..                                                 ...           ...    ...   
    185  Gross domestic product per capita, current prices  U.S. dollars  Units   
    186  Gross domestic product per capita, current prices  U.S. dollars  Units   
    187  Gross domestic product per capita, current prices  U.S. dollars  Units   
    188  Gross domestic product per capita, current prices  U.S. dollars  Units   
    189                                                NaN           NaN    NaN   
    
                             Country/Series-specific Notes       2015  \
    0    See notes for:  Gross domestic product, curren...    599.994   
    1    See notes for:  Gross domestic product, curren...   3995.383   
    2    See notes for:  Gross domestic product, curren...   4318.135   
    3    See notes for:  Gross domestic product, curren...   4100.315   
    4    See notes for:  Gross domestic product, curren...  14414.302   
    ..                                                 ...        ...   
    185  See notes for:  Gross domestic product, curren...   2088.344   
    186  See notes for:  Gross domestic product, curren...   1302.940   
    187  See notes for:  Gross domestic product, curren...   1350.151   
    188  See notes for:  Gross domestic product, curren...   1064.350   
    189                                                NaN        NaN   
    
         Estimates Start After  
    0                   2013.0  
    1                   2010.0  
    2                   2014.0  
    3                   2014.0  
    4                   2011.0  
    ..                     ...  
    185                 2012.0  
    186                 2008.0  
    187                 2010.0  
    188                 2012.0  
    189                    NaN  
    
    [190 rows x 7 columns]
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 190 entries, 0 to 189
    Data columns (total 7 columns):
     #   Column                         Non-Null Count  Dtype  
    ---  ------                         --------------  -----  
     0   Country                        190 non-null    object 
     1   Subject Descriptor             189 non-null    object 
     2   Units                          189 non-null    object 
     3   Scale                          189 non-null    object 
     4   Country/Series-specific Notes  188 non-null    object 
     5   2015                           187 non-null    float64
     6   Estimates Start After          188 non-null    float64
    dtypes: float64(2), object(5)
    memory usage: 10.5+ KB



```python
# 데이터 가공 - filtering, merge, transform(변환)

# 필터링 - oecd_bli 에서 INEQUALITY 가 TOT 인 데이터만 추출
oecd_bli = oecd_bli[oecd_bli['INEQUALITY'] == 'TOT']
#pivot 테이블을 생성 - index 는 Country, columns 는 Indicator, values는 Value
oecd_bli = oecd_bli.pivot(index='Country', columns='Indicator', values='Value')
print(oecd_bli)

#지금 처럼 원본 데이터에 변경 내용을 반영하면 2번 실행하면 에러가 발생할 수 있음
```

    Indicator        Air pollution  Assault rate  Consultation on rule-making  \
    Country                                                                     
    Australia                 13.0           2.1                         10.5   
    Austria                   27.0           3.4                          7.1   
    Belgium                   21.0           6.6                          4.5   
    Brazil                    18.0           7.9                          4.0   
    Canada                    15.0           1.3                         10.5   
    Chile                     46.0           6.9                          2.0   
    Czech Republic            16.0           2.8                          6.8   
    Denmark                   15.0           3.9                          7.0   
    Estonia                    9.0           5.5                          3.3   
    Finland                   15.0           2.4                          9.0   
    France                    12.0           5.0                          3.5   
    Germany                   16.0           3.6                          4.5   
    Greece                    27.0           3.7                          6.5   
    Hungary                   15.0           3.6                          7.9   
    Iceland                   18.0           2.7                          5.1   
    Ireland                   13.0           2.6                          9.0   
    Israel                    21.0           6.4                          2.5   
    Italy                     21.0           4.7                          5.0   
    Japan                     24.0           1.4                          7.3   
    Korea                     30.0           2.1                         10.4   
    Luxembourg                12.0           4.3                          6.0   
    Mexico                    30.0          12.8                          9.0   
    Netherlands               30.0           4.9                          6.1   
    New Zealand               11.0           2.2                         10.3   
    Norway                    16.0           3.3                          8.1   
    OECD - Total              20.0           3.9                          7.3   
    Poland                    33.0           1.4                         10.8   
    Portugal                  18.0           5.7                          6.5   
    Russia                    15.0           3.8                          2.5   
    Slovak Republic           13.0           3.0                          6.6   
    Slovenia                  26.0           3.9                         10.3   
    Spain                     24.0           4.2                          7.3   
    Sweden                    10.0           5.1                         10.9   
    Switzerland               20.0           4.2                          8.4   
    Turkey                    35.0           5.0                          5.5   
    United Kingdom            13.0           1.9                         11.5   
    United States             18.0           1.5                          8.3   
    
    Indicator        Dwellings without basic facilities  Educational attainment  \
    Country                                                                       
    Australia                                       1.1                    76.0   
    Austria                                         1.0                    83.0   
    Belgium                                         2.0                    72.0   
    Brazil                                          6.7                    45.0   
    Canada                                          0.2                    89.0   
    Chile                                           9.4                    57.0   
    Czech Republic                                  0.9                    92.0   
    Denmark                                         0.9                    78.0   
    Estonia                                         8.1                    90.0   
    Finland                                         0.6                    85.0   
    France                                          0.5                    73.0   
    Germany                                         0.1                    86.0   
    Greece                                          0.7                    68.0   
    Hungary                                         4.8                    82.0   
    Iceland                                         0.4                    71.0   
    Ireland                                         0.2                    75.0   
    Israel                                          3.7                    85.0   
    Italy                                           1.1                    57.0   
    Japan                                           6.4                    94.0   
    Korea                                           4.2                    82.0   
    Luxembourg                                      0.1                    78.0   
    Mexico                                          4.2                    37.0   
    Netherlands                                     0.0                    73.0   
    New Zealand                                     0.2                    74.0   
    Norway                                          0.3                    82.0   
    OECD - Total                                    2.4                    75.0   
    Poland                                          3.2                    90.0   
    Portugal                                        0.9                    38.0   
    Russia                                         15.1                    94.0   
    Slovak Republic                                 0.6                    92.0   
    Slovenia                                        0.5                    85.0   
    Spain                                           0.1                    55.0   
    Sweden                                          0.0                    88.0   
    Switzerland                                     0.0                    86.0   
    Turkey                                         12.7                    34.0   
    United Kingdom                                  0.2                    78.0   
    United States                                   0.1                    89.0   
    
    Indicator        Employees working very long hours  Employment rate  \
    Country                                                               
    Australia                                    14.02             72.0   
    Austria                                       7.61             72.0   
    Belgium                                       4.57             62.0   
    Brazil                                       10.41             67.0   
    Canada                                        3.94             72.0   
    Chile                                        15.42             62.0   
    Czech Republic                                6.98             68.0   
    Denmark                                       2.03             73.0   
    Estonia                                       3.30             68.0   
    Finland                                       3.58             69.0   
    France                                        8.15             64.0   
    Germany                                       5.25             73.0   
    Greece                                        6.16             49.0   
    Hungary                                       3.19             58.0   
    Iceland                                      12.25             82.0   
    Ireland                                       4.20             60.0   
    Israel                                       16.03             67.0   
    Italy                                         3.66             56.0   
    Japan                                        22.26             72.0   
    Korea                                        18.72             64.0   
    Luxembourg                                    3.47             66.0   
    Mexico                                       28.83             61.0   
    Netherlands                                   0.45             74.0   
    New Zealand                                  13.87             73.0   
    Norway                                        2.82             75.0   
    OECD - Total                                 12.51             65.0   
    Poland                                        7.41             60.0   
    Portugal                                      9.62             61.0   
    Russia                                        0.16             69.0   
    Slovak Republic                               7.02             60.0   
    Slovenia                                      5.63             63.0   
    Spain                                         5.89             56.0   
    Sweden                                        1.13             74.0   
    Switzerland                                   6.72             80.0   
    Turkey                                       40.86             50.0   
    United Kingdom                               12.70             71.0   
    United States                                11.30             67.0   
    
    Indicator        Homicide rate  Household net adjusted disposable income  \
    Country                                                                    
    Australia                  0.8                                   31588.0   
    Austria                    0.4                                   31173.0   
    Belgium                    1.1                                   28307.0   
    Brazil                    25.5                                   11664.0   
    Canada                     1.5                                   29365.0   
    Chile                      4.4                                   14533.0   
    Czech Republic             0.8                                   18404.0   
    Denmark                    0.3                                   26491.0   
    Estonia                    4.8                                   15167.0   
    Finland                    1.4                                   27927.0   
    France                     0.6                                   28799.0   
    Germany                    0.5                                   31252.0   
    Greece                     1.6                                   18575.0   
    Hungary                    1.3                                   15442.0   
    Iceland                    0.3                                   23965.0   
    Ireland                    0.8                                   23917.0   
    Israel                     2.3                                   22104.0   
    Italy                      0.7                                   25166.0   
    Japan                      0.3                                   26111.0   
    Korea                      1.1                                   19510.0   
    Luxembourg                 0.4                                   38951.0   
    Mexico                    23.4                                   13085.0   
    Netherlands                0.9                                   27888.0   
    New Zealand                1.2                                   23815.0   
    Norway                     0.6                                   33492.0   
    OECD - Total               4.0                                   25908.0   
    Poland                     0.9                                   17852.0   
    Portugal                   1.1                                   20086.0   
    Russia                    12.8                                   19292.0   
    Slovak Republic            1.2                                   17503.0   
    Slovenia                   0.4                                   19326.0   
    Spain                      0.6                                   22477.0   
    Sweden                     0.7                                   29185.0   
    Switzerland                0.5                                   33491.0   
    Turkey                     1.2                                   14095.0   
    United Kingdom             0.3                                   27029.0   
    United States              5.2                                   41355.0   
    
    Indicator        Household net financial wealth  ...  \
    Country                                          ...   
    Australia                               47657.0  ...   
    Austria                                 49887.0  ...   
    Belgium                                 83876.0  ...   
    Brazil                                   6844.0  ...   
    Canada                                  67913.0  ...   
    Chile                                   17733.0  ...   
    Czech Republic                          17299.0  ...   
    Denmark                                 44488.0  ...   
    Estonia                                  7680.0  ...   
    Finland                                 18761.0  ...   
    France                                  48741.0  ...   
    Germany                                 50394.0  ...   
    Greece                                  14579.0  ...   
    Hungary                                 13277.0  ...   
    Iceland                                 43045.0  ...   
    Ireland                                 31580.0  ...   
    Israel                                  52933.0  ...   
    Italy                                   54987.0  ...   
    Japan                                   86764.0  ...   
    Korea                                   29091.0  ...   
    Luxembourg                              61765.0  ...   
    Mexico                                   9056.0  ...   
    Netherlands                             77961.0  ...   
    New Zealand                             28290.0  ...   
    Norway                                   8797.0  ...   
    OECD - Total                            67139.0  ...   
    Poland                                  10919.0  ...   
    Portugal                                31245.0  ...   
    Russia                                   3412.0  ...   
    Slovak Republic                          8663.0  ...   
    Slovenia                                18465.0  ...   
    Spain                                   24774.0  ...   
    Sweden                                  60328.0  ...   
    Switzerland                            108823.0  ...   
    Turkey                                   3251.0  ...   
    United Kingdom                          60778.0  ...   
    United States                          145769.0  ...   
    
    Indicator        Long-term unemployment rate  Personal earnings  \
    Country                                                           
    Australia                               1.08            50449.0   
    Austria                                 1.19            45199.0   
    Belgium                                 3.88            48082.0   
    Brazil                                  1.97            17177.0   
    Canada                                  0.90            46911.0   
    Chile                                   1.59            22101.0   
    Czech Republic                          3.12            20338.0   
    Denmark                                 1.78            48347.0   
    Estonia                                 3.82            18944.0   
    Finland                                 1.73            40060.0   
    France                                  3.99            40242.0   
    Germany                                 2.37            43682.0   
    Greece                                 18.39            25503.0   
    Hungary                                 5.10            20948.0   
    Iceland                                 1.18            55716.0   
    Ireland                                 8.39            49506.0   
    Israel                                  0.79            28817.0   
    Italy                                   6.94            34561.0   
    Japan                                   1.67            35405.0   
    Korea                                   0.01            36354.0   
    Luxembourg                              1.78            56021.0   
    Mexico                                  0.08            16193.0   
    Netherlands                             2.40            47590.0   
    New Zealand                             0.75            35609.0   
    Norway                                  0.32            50282.0   
    OECD - Total                            2.79            36118.0   
    Poland                                  3.77            22655.0   
    Portugal                                9.11            23688.0   
    Russia                                  1.70            20885.0   
    Slovak Republic                         9.46            20307.0   
    Slovenia                                5.15            32037.0   
    Spain                                  12.96            34824.0   
    Sweden                                  1.37            40818.0   
    Switzerland                             1.46            54236.0   
    Turkey                                  2.37            16919.0   
    United Kingdom                          2.77            41192.0   
    United States                           1.91            56340.0   
    
    Indicator        Quality of support network  Rooms per person  \
    Country                                                         
    Australia                              92.0               2.3   
    Austria                                89.0               1.6   
    Belgium                                94.0               2.2   
    Brazil                                 90.0               1.6   
    Canada                                 92.0               2.5   
    Chile                                  86.0               1.2   
    Czech Republic                         85.0               1.4   
    Denmark                                95.0               1.9   
    Estonia                                89.0               1.5   
    Finland                                95.0               1.9   
    France                                 87.0               1.8   
    Germany                                94.0               1.8   
    Greece                                 83.0               1.2   
    Hungary                                87.0               1.1   
    Iceland                                96.0               1.5   
    Ireland                                96.0               2.1   
    Israel                                 87.0               1.2   
    Italy                                  90.0               1.4   
    Japan                                  89.0               1.8   
    Korea                                  72.0               1.4   
    Luxembourg                             87.0               2.0   
    Mexico                                 77.0               1.0   
    Netherlands                            90.0               2.0   
    New Zealand                            94.0               2.4   
    Norway                                 94.0               2.0   
    OECD - Total                           88.0               1.8   
    Poland                                 91.0               1.1   
    Portugal                               86.0               1.6   
    Russia                                 90.0               0.9   
    Slovak Republic                        90.0               1.1   
    Slovenia                               90.0               1.5   
    Spain                                  95.0               1.9   
    Sweden                                 92.0               1.7   
    Switzerland                            96.0               1.8   
    Turkey                                 86.0               1.1   
    United Kingdom                         91.0               1.9   
    United States                          90.0               2.4   
    
    Indicator        Self-reported health  Student skills  \
    Country                                                 
    Australia                        85.0           512.0   
    Austria                          69.0           500.0   
    Belgium                          74.0           509.0   
    Brazil                           69.0           402.0   
    Canada                           89.0           522.0   
    Chile                            59.0           436.0   
    Czech Republic                   60.0           500.0   
    Denmark                          72.0           498.0   
    Estonia                          54.0           526.0   
    Finland                          65.0           529.0   
    France                           67.0           500.0   
    Germany                          65.0           515.0   
    Greece                           74.0           466.0   
    Hungary                          57.0           487.0   
    Iceland                          77.0           484.0   
    Ireland                          82.0           516.0   
    Israel                           80.0           474.0   
    Italy                            66.0           490.0   
    Japan                            30.0           540.0   
    Korea                            35.0           542.0   
    Luxembourg                       72.0           490.0   
    Mexico                           66.0           417.0   
    Netherlands                      76.0           519.0   
    New Zealand                      90.0           509.0   
    Norway                           76.0           496.0   
    OECD - Total                     68.0           497.0   
    Poland                           58.0           521.0   
    Portugal                         46.0           488.0   
    Russia                           37.0           481.0   
    Slovak Republic                  66.0           472.0   
    Slovenia                         65.0           499.0   
    Spain                            72.0           490.0   
    Sweden                           81.0           482.0   
    Switzerland                      81.0           518.0   
    Turkey                           68.0           462.0   
    United Kingdom                   74.0           502.0   
    United States                    88.0           492.0   
    
    Indicator        Time devoted to leisure and personal care  Voter turnout  \
    Country                                                                     
    Australia                                            14.41           93.0   
    Austria                                              14.46           75.0   
    Belgium                                              15.71           89.0   
    Brazil                                               14.97           79.0   
    Canada                                               14.25           61.0   
    Chile                                                14.41           49.0   
    Czech Republic                                       14.98           59.0   
    Denmark                                              16.06           88.0   
    Estonia                                              14.90           64.0   
    Finland                                              14.89           69.0   
    France                                               15.33           80.0   
    Germany                                              15.31           72.0   
    Greece                                               14.91           64.0   
    Hungary                                              15.04           62.0   
    Iceland                                              14.61           81.0   
    Ireland                                              15.19           70.0   
    Israel                                               14.48           68.0   
    Italy                                                14.98           75.0   
    Japan                                                14.93           53.0   
    Korea                                                14.63           76.0   
    Luxembourg                                           15.12           91.0   
    Mexico                                               13.89           63.0   
    Netherlands                                          15.44           75.0   
    New Zealand                                          14.87           77.0   
    Norway                                               15.56           78.0   
    OECD - Total                                         14.97           68.0   
    Poland                                               14.20           55.0   
    Portugal                                             14.95           58.0   
    Russia                                               14.97           65.0   
    Slovak Republic                                      14.99           59.0   
    Slovenia                                             14.62           52.0   
    Spain                                                16.06           69.0   
    Sweden                                               15.11           86.0   
    Switzerland                                          14.98           49.0   
    Turkey                                               13.42           88.0   
    United Kingdom                                       14.83           66.0   
    United States                                        14.27           68.0   
    
    Indicator        Water quality  Years in education  
    Country                                             
    Australia                 91.0                19.4  
    Austria                   94.0                17.0  
    Belgium                   87.0                18.9  
    Brazil                    72.0                16.3  
    Canada                    91.0                17.2  
    Chile                     73.0                16.5  
    Czech Republic            85.0                18.1  
    Denmark                   94.0                19.4  
    Estonia                   79.0                17.5  
    Finland                   94.0                19.7  
    France                    82.0                16.4  
    Germany                   95.0                18.2  
    Greece                    69.0                18.6  
    Hungary                   77.0                17.6  
    Iceland                   97.0                19.8  
    Ireland                   80.0                17.6  
    Israel                    68.0                15.8  
    Italy                     71.0                16.8  
    Japan                     85.0                16.3  
    Korea                     78.0                17.5  
    Luxembourg                86.0                15.1  
    Mexico                    67.0                14.4  
    Netherlands               92.0                18.7  
    New Zealand               89.0                18.1  
    Norway                    94.0                17.9  
    OECD - Total              81.0                17.7  
    Poland                    79.0                18.4  
    Portugal                  86.0                17.6  
    Russia                    56.0                16.0  
    Slovak Republic           81.0                16.3  
    Slovenia                  88.0                18.4  
    Spain                     71.0                17.6  
    Sweden                    95.0                19.3  
    Switzerland               96.0                17.3  
    Turkey                    62.0                16.4  
    United Kingdom            88.0                16.4  
    United States             85.0                17.2  
    
    [37 rows x 24 columns]



```python
#열이름 변경
gdp_per_capita.rename(columns={"2015":"GDP per capita"}, inplace=True)
#인덱스 설정 - 앞의 데이터와 합치기 위해서 index를 동일한 의미를 갖는 컬럼으로 설정
gdp_per_capita.set_index("Country", inplace=True)
print(gdp_per_capita)
```

                                                                                       Subject Descriptor  \
    Country                                                                                                 
    Afghanistan                                         Gross domestic product per capita, current prices   
    Albania                                             Gross domestic product per capita, current prices   
    Algeria                                             Gross domestic product per capita, current prices   
    Angola                                              Gross domestic product per capita, current prices   
    Antigua and Barbuda                                 Gross domestic product per capita, current prices   
    ...                                                                                               ...   
    Vietnam                                             Gross domestic product per capita, current prices   
    Yemen                                               Gross domestic product per capita, current prices   
    Zambia                                              Gross domestic product per capita, current prices   
    Zimbabwe                                            Gross domestic product per capita, current prices   
    International Monetary Fund, World Economic Out...                                                NaN   
    
                                                               Units  Scale  \
    Country                                                                   
    Afghanistan                                         U.S. dollars  Units   
    Albania                                             U.S. dollars  Units   
    Algeria                                             U.S. dollars  Units   
    Angola                                              U.S. dollars  Units   
    Antigua and Barbuda                                 U.S. dollars  Units   
    ...                                                          ...    ...   
    Vietnam                                             U.S. dollars  Units   
    Yemen                                               U.S. dollars  Units   
    Zambia                                              U.S. dollars  Units   
    Zimbabwe                                            U.S. dollars  Units   
    International Monetary Fund, World Economic Out...           NaN    NaN   
    
                                                                            Country/Series-specific Notes  \
    Country                                                                                                 
    Afghanistan                                         See notes for:  Gross domestic product, curren...   
    Albania                                             See notes for:  Gross domestic product, curren...   
    Algeria                                             See notes for:  Gross domestic product, curren...   
    Angola                                              See notes for:  Gross domestic product, curren...   
    Antigua and Barbuda                                 See notes for:  Gross domestic product, curren...   
    ...                                                                                               ...   
    Vietnam                                             See notes for:  Gross domestic product, curren...   
    Yemen                                               See notes for:  Gross domestic product, curren...   
    Zambia                                              See notes for:  Gross domestic product, curren...   
    Zimbabwe                                            See notes for:  Gross domestic product, curren...   
    International Monetary Fund, World Economic Out...                                                NaN   
    
                                                        GDP per capita  \
    Country                                                              
    Afghanistan                                                599.994   
    Albania                                                   3995.383   
    Algeria                                                   4318.135   
    Angola                                                    4100.315   
    Antigua and Barbuda                                      14414.302   
    ...                                                            ...   
    Vietnam                                                   2088.344   
    Yemen                                                     1302.940   
    Zambia                                                    1350.151   
    Zimbabwe                                                  1064.350   
    International Monetary Fund, World Economic Out...             NaN   
    
                                                        Estimates Start After  
    Country                                                                    
    Afghanistan                                                        2013.0  
    Albania                                                            2010.0  
    Algeria                                                            2014.0  
    Angola                                                             2014.0  
    Antigua and Barbuda                                                2011.0  
    ...                                                                   ...  
    Vietnam                                                            2012.0  
    Yemen                                                              2008.0  
    Zambia                                                             2010.0  
    Zimbabwe                                                           2012.0  
    International Monetary Fund, World Economic Out...                    NaN  
    
    [190 rows x 6 columns]



```python
# 2개의 데이터프레임을 인덱스를 이용해서 합치기 
full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita, 
                             left_index = True, right_index=True)
#데이터 정렬
full_country_stats.sort_values(by='GDP per capita', inplace=True)
print(full_country_stats)
```

                     Air pollution  Assault rate  Consultation on rule-making  \
    Country                                                                     
    Brazil                    18.0           7.9                          4.0   
    Mexico                    30.0          12.8                          9.0   
    Russia                    15.0           3.8                          2.5   
    Turkey                    35.0           5.0                          5.5   
    Hungary                   15.0           3.6                          7.9   
    Poland                    33.0           1.4                         10.8   
    Chile                     46.0           6.9                          2.0   
    Slovak Republic           13.0           3.0                          6.6   
    Czech Republic            16.0           2.8                          6.8   
    Estonia                    9.0           5.5                          3.3   
    Greece                    27.0           3.7                          6.5   
    Portugal                  18.0           5.7                          6.5   
    Slovenia                  26.0           3.9                         10.3   
    Spain                     24.0           4.2                          7.3   
    Korea                     30.0           2.1                         10.4   
    Italy                     21.0           4.7                          5.0   
    Japan                     24.0           1.4                          7.3   
    Israel                    21.0           6.4                          2.5   
    New Zealand               11.0           2.2                         10.3   
    France                    12.0           5.0                          3.5   
    Belgium                   21.0           6.6                          4.5   
    Germany                   16.0           3.6                          4.5   
    Finland                   15.0           2.4                          9.0   
    Canada                    15.0           1.3                         10.5   
    Netherlands               30.0           4.9                          6.1   
    Austria                   27.0           3.4                          7.1   
    United Kingdom            13.0           1.9                         11.5   
    Sweden                    10.0           5.1                         10.9   
    Iceland                   18.0           2.7                          5.1   
    Australia                 13.0           2.1                         10.5   
    Ireland                   13.0           2.6                          9.0   
    Denmark                   15.0           3.9                          7.0   
    United States             18.0           1.5                          8.3   
    Norway                    16.0           3.3                          8.1   
    Switzerland               20.0           4.2                          8.4   
    Luxembourg                12.0           4.3                          6.0   
    
                     Dwellings without basic facilities  Educational attainment  \
    Country                                                                       
    Brazil                                          6.7                    45.0   
    Mexico                                          4.2                    37.0   
    Russia                                         15.1                    94.0   
    Turkey                                         12.7                    34.0   
    Hungary                                         4.8                    82.0   
    Poland                                          3.2                    90.0   
    Chile                                           9.4                    57.0   
    Slovak Republic                                 0.6                    92.0   
    Czech Republic                                  0.9                    92.0   
    Estonia                                         8.1                    90.0   
    Greece                                          0.7                    68.0   
    Portugal                                        0.9                    38.0   
    Slovenia                                        0.5                    85.0   
    Spain                                           0.1                    55.0   
    Korea                                           4.2                    82.0   
    Italy                                           1.1                    57.0   
    Japan                                           6.4                    94.0   
    Israel                                          3.7                    85.0   
    New Zealand                                     0.2                    74.0   
    France                                          0.5                    73.0   
    Belgium                                         2.0                    72.0   
    Germany                                         0.1                    86.0   
    Finland                                         0.6                    85.0   
    Canada                                          0.2                    89.0   
    Netherlands                                     0.0                    73.0   
    Austria                                         1.0                    83.0   
    United Kingdom                                  0.2                    78.0   
    Sweden                                          0.0                    88.0   
    Iceland                                         0.4                    71.0   
    Australia                                       1.1                    76.0   
    Ireland                                         0.2                    75.0   
    Denmark                                         0.9                    78.0   
    United States                                   0.1                    89.0   
    Norway                                          0.3                    82.0   
    Switzerland                                     0.0                    86.0   
    Luxembourg                                      0.1                    78.0   
    
                     Employees working very long hours  Employment rate  \
    Country                                                               
    Brazil                                       10.41             67.0   
    Mexico                                       28.83             61.0   
    Russia                                        0.16             69.0   
    Turkey                                       40.86             50.0   
    Hungary                                       3.19             58.0   
    Poland                                        7.41             60.0   
    Chile                                        15.42             62.0   
    Slovak Republic                               7.02             60.0   
    Czech Republic                                6.98             68.0   
    Estonia                                       3.30             68.0   
    Greece                                        6.16             49.0   
    Portugal                                      9.62             61.0   
    Slovenia                                      5.63             63.0   
    Spain                                         5.89             56.0   
    Korea                                        18.72             64.0   
    Italy                                         3.66             56.0   
    Japan                                        22.26             72.0   
    Israel                                       16.03             67.0   
    New Zealand                                  13.87             73.0   
    France                                        8.15             64.0   
    Belgium                                       4.57             62.0   
    Germany                                       5.25             73.0   
    Finland                                       3.58             69.0   
    Canada                                        3.94             72.0   
    Netherlands                                   0.45             74.0   
    Austria                                       7.61             72.0   
    United Kingdom                               12.70             71.0   
    Sweden                                        1.13             74.0   
    Iceland                                      12.25             82.0   
    Australia                                    14.02             72.0   
    Ireland                                       4.20             60.0   
    Denmark                                       2.03             73.0   
    United States                                11.30             67.0   
    Norway                                        2.82             75.0   
    Switzerland                                   6.72             80.0   
    Luxembourg                                    3.47             66.0   
    
                     Homicide rate  Household net adjusted disposable income  \
    Country                                                                    
    Brazil                    25.5                                   11664.0   
    Mexico                    23.4                                   13085.0   
    Russia                    12.8                                   19292.0   
    Turkey                     1.2                                   14095.0   
    Hungary                    1.3                                   15442.0   
    Poland                     0.9                                   17852.0   
    Chile                      4.4                                   14533.0   
    Slovak Republic            1.2                                   17503.0   
    Czech Republic             0.8                                   18404.0   
    Estonia                    4.8                                   15167.0   
    Greece                     1.6                                   18575.0   
    Portugal                   1.1                                   20086.0   
    Slovenia                   0.4                                   19326.0   
    Spain                      0.6                                   22477.0   
    Korea                      1.1                                   19510.0   
    Italy                      0.7                                   25166.0   
    Japan                      0.3                                   26111.0   
    Israel                     2.3                                   22104.0   
    New Zealand                1.2                                   23815.0   
    France                     0.6                                   28799.0   
    Belgium                    1.1                                   28307.0   
    Germany                    0.5                                   31252.0   
    Finland                    1.4                                   27927.0   
    Canada                     1.5                                   29365.0   
    Netherlands                0.9                                   27888.0   
    Austria                    0.4                                   31173.0   
    United Kingdom             0.3                                   27029.0   
    Sweden                     0.7                                   29185.0   
    Iceland                    0.3                                   23965.0   
    Australia                  0.8                                   31588.0   
    Ireland                    0.8                                   23917.0   
    Denmark                    0.3                                   26491.0   
    United States              5.2                                   41355.0   
    Norway                     0.6                                   33492.0   
    Switzerland                0.5                                   33491.0   
    Luxembourg                 0.4                                   38951.0   
    
                     Household net financial wealth  ...  \
    Country                                          ...   
    Brazil                                   6844.0  ...   
    Mexico                                   9056.0  ...   
    Russia                                   3412.0  ...   
    Turkey                                   3251.0  ...   
    Hungary                                 13277.0  ...   
    Poland                                  10919.0  ...   
    Chile                                   17733.0  ...   
    Slovak Republic                          8663.0  ...   
    Czech Republic                          17299.0  ...   
    Estonia                                  7680.0  ...   
    Greece                                  14579.0  ...   
    Portugal                                31245.0  ...   
    Slovenia                                18465.0  ...   
    Spain                                   24774.0  ...   
    Korea                                   29091.0  ...   
    Italy                                   54987.0  ...   
    Japan                                   86764.0  ...   
    Israel                                  52933.0  ...   
    New Zealand                             28290.0  ...   
    France                                  48741.0  ...   
    Belgium                                 83876.0  ...   
    Germany                                 50394.0  ...   
    Finland                                 18761.0  ...   
    Canada                                  67913.0  ...   
    Netherlands                             77961.0  ...   
    Austria                                 49887.0  ...   
    United Kingdom                          60778.0  ...   
    Sweden                                  60328.0  ...   
    Iceland                                 43045.0  ...   
    Australia                               47657.0  ...   
    Ireland                                 31580.0  ...   
    Denmark                                 44488.0  ...   
    United States                          145769.0  ...   
    Norway                                   8797.0  ...   
    Switzerland                            108823.0  ...   
    Luxembourg                              61765.0  ...   
    
                     Time devoted to leisure and personal care  Voter turnout  \
    Country                                                                     
    Brazil                                               14.97           79.0   
    Mexico                                               13.89           63.0   
    Russia                                               14.97           65.0   
    Turkey                                               13.42           88.0   
    Hungary                                              15.04           62.0   
    Poland                                               14.20           55.0   
    Chile                                                14.41           49.0   
    Slovak Republic                                      14.99           59.0   
    Czech Republic                                       14.98           59.0   
    Estonia                                              14.90           64.0   
    Greece                                               14.91           64.0   
    Portugal                                             14.95           58.0   
    Slovenia                                             14.62           52.0   
    Spain                                                16.06           69.0   
    Korea                                                14.63           76.0   
    Italy                                                14.98           75.0   
    Japan                                                14.93           53.0   
    Israel                                               14.48           68.0   
    New Zealand                                          14.87           77.0   
    France                                               15.33           80.0   
    Belgium                                              15.71           89.0   
    Germany                                              15.31           72.0   
    Finland                                              14.89           69.0   
    Canada                                               14.25           61.0   
    Netherlands                                          15.44           75.0   
    Austria                                              14.46           75.0   
    United Kingdom                                       14.83           66.0   
    Sweden                                               15.11           86.0   
    Iceland                                              14.61           81.0   
    Australia                                            14.41           93.0   
    Ireland                                              15.19           70.0   
    Denmark                                              16.06           88.0   
    United States                                        14.27           68.0   
    Norway                                               15.56           78.0   
    Switzerland                                          14.98           49.0   
    Luxembourg                                           15.12           91.0   
    
                     Water quality  Years in education  \
    Country                                              
    Brazil                    72.0                16.3   
    Mexico                    67.0                14.4   
    Russia                    56.0                16.0   
    Turkey                    62.0                16.4   
    Hungary                   77.0                17.6   
    Poland                    79.0                18.4   
    Chile                     73.0                16.5   
    Slovak Republic           81.0                16.3   
    Czech Republic            85.0                18.1   
    Estonia                   79.0                17.5   
    Greece                    69.0                18.6   
    Portugal                  86.0                17.6   
    Slovenia                  88.0                18.4   
    Spain                     71.0                17.6   
    Korea                     78.0                17.5   
    Italy                     71.0                16.8   
    Japan                     85.0                16.3   
    Israel                    68.0                15.8   
    New Zealand               89.0                18.1   
    France                    82.0                16.4   
    Belgium                   87.0                18.9   
    Germany                   95.0                18.2   
    Finland                   94.0                19.7   
    Canada                    91.0                17.2   
    Netherlands               92.0                18.7   
    Austria                   94.0                17.0   
    United Kingdom            88.0                16.4   
    Sweden                    95.0                19.3   
    Iceland                   97.0                19.8   
    Australia                 91.0                19.4   
    Ireland                   80.0                17.6   
    Denmark                   94.0                19.4   
    United States             85.0                17.2   
    Norway                    94.0                17.9   
    Switzerland               96.0                17.3   
    Luxembourg                86.0                15.1   
    
                                                    Subject Descriptor  \
    Country                                                              
    Brazil           Gross domestic product per capita, current prices   
    Mexico           Gross domestic product per capita, current prices   
    Russia           Gross domestic product per capita, current prices   
    Turkey           Gross domestic product per capita, current prices   
    Hungary          Gross domestic product per capita, current prices   
    Poland           Gross domestic product per capita, current prices   
    Chile            Gross domestic product per capita, current prices   
    Slovak Republic  Gross domestic product per capita, current prices   
    Czech Republic   Gross domestic product per capita, current prices   
    Estonia          Gross domestic product per capita, current prices   
    Greece           Gross domestic product per capita, current prices   
    Portugal         Gross domestic product per capita, current prices   
    Slovenia         Gross domestic product per capita, current prices   
    Spain            Gross domestic product per capita, current prices   
    Korea            Gross domestic product per capita, current prices   
    Italy            Gross domestic product per capita, current prices   
    Japan            Gross domestic product per capita, current prices   
    Israel           Gross domestic product per capita, current prices   
    New Zealand      Gross domestic product per capita, current prices   
    France           Gross domestic product per capita, current prices   
    Belgium          Gross domestic product per capita, current prices   
    Germany          Gross domestic product per capita, current prices   
    Finland          Gross domestic product per capita, current prices   
    Canada           Gross domestic product per capita, current prices   
    Netherlands      Gross domestic product per capita, current prices   
    Austria          Gross domestic product per capita, current prices   
    United Kingdom   Gross domestic product per capita, current prices   
    Sweden           Gross domestic product per capita, current prices   
    Iceland          Gross domestic product per capita, current prices   
    Australia        Gross domestic product per capita, current prices   
    Ireland          Gross domestic product per capita, current prices   
    Denmark          Gross domestic product per capita, current prices   
    United States    Gross domestic product per capita, current prices   
    Norway           Gross domestic product per capita, current prices   
    Switzerland      Gross domestic product per capita, current prices   
    Luxembourg       Gross domestic product per capita, current prices   
    
                            Units  Scale  \
    Country                                
    Brazil           U.S. dollars  Units   
    Mexico           U.S. dollars  Units   
    Russia           U.S. dollars  Units   
    Turkey           U.S. dollars  Units   
    Hungary          U.S. dollars  Units   
    Poland           U.S. dollars  Units   
    Chile            U.S. dollars  Units   
    Slovak Republic  U.S. dollars  Units   
    Czech Republic   U.S. dollars  Units   
    Estonia          U.S. dollars  Units   
    Greece           U.S. dollars  Units   
    Portugal         U.S. dollars  Units   
    Slovenia         U.S. dollars  Units   
    Spain            U.S. dollars  Units   
    Korea            U.S. dollars  Units   
    Italy            U.S. dollars  Units   
    Japan            U.S. dollars  Units   
    Israel           U.S. dollars  Units   
    New Zealand      U.S. dollars  Units   
    France           U.S. dollars  Units   
    Belgium          U.S. dollars  Units   
    Germany          U.S. dollars  Units   
    Finland          U.S. dollars  Units   
    Canada           U.S. dollars  Units   
    Netherlands      U.S. dollars  Units   
    Austria          U.S. dollars  Units   
    United Kingdom   U.S. dollars  Units   
    Sweden           U.S. dollars  Units   
    Iceland          U.S. dollars  Units   
    Australia        U.S. dollars  Units   
    Ireland          U.S. dollars  Units   
    Denmark          U.S. dollars  Units   
    United States    U.S. dollars  Units   
    Norway           U.S. dollars  Units   
    Switzerland      U.S. dollars  Units   
    Luxembourg       U.S. dollars  Units   
    
                                         Country/Series-specific Notes  \
    Country                                                              
    Brazil           See notes for:  Gross domestic product, curren...   
    Mexico           See notes for:  Gross domestic product, curren...   
    Russia           See notes for:  Gross domestic product, curren...   
    Turkey           See notes for:  Gross domestic product, curren...   
    Hungary          See notes for:  Gross domestic product, curren...   
    Poland           See notes for:  Gross domestic product, curren...   
    Chile            See notes for:  Gross domestic product, curren...   
    Slovak Republic  See notes for:  Gross domestic product, curren...   
    Czech Republic   See notes for:  Gross domestic product, curren...   
    Estonia          See notes for:  Gross domestic product, curren...   
    Greece           See notes for:  Gross domestic product, curren...   
    Portugal         See notes for:  Gross domestic product, curren...   
    Slovenia         See notes for:  Gross domestic product, curren...   
    Spain            See notes for:  Gross domestic product, curren...   
    Korea            See notes for:  Gross domestic product, curren...   
    Italy            See notes for:  Gross domestic product, curren...   
    Japan            See notes for:  Gross domestic product, curren...   
    Israel           See notes for:  Gross domestic product, curren...   
    New Zealand      See notes for:  Gross domestic product, curren...   
    France           See notes for:  Gross domestic product, curren...   
    Belgium          See notes for:  Gross domestic product, curren...   
    Germany          See notes for:  Gross domestic product, curren...   
    Finland          See notes for:  Gross domestic product, curren...   
    Canada           See notes for:  Gross domestic product, curren...   
    Netherlands      See notes for:  Gross domestic product, curren...   
    Austria          See notes for:  Gross domestic product, curren...   
    United Kingdom   See notes for:  Gross domestic product, curren...   
    Sweden           See notes for:  Gross domestic product, curren...   
    Iceland          See notes for:  Gross domestic product, curren...   
    Australia        See notes for:  Gross domestic product, curren...   
    Ireland          See notes for:  Gross domestic product, curren...   
    Denmark          See notes for:  Gross domestic product, curren...   
    United States    See notes for:  Gross domestic product, curren...   
    Norway           See notes for:  Gross domestic product, curren...   
    Switzerland      See notes for:  Gross domestic product, curren...   
    Luxembourg       See notes for:  Gross domestic product, curren...   
    
                     GDP per capita  Estimates Start After  
    Country                                                 
    Brazil                 8669.998                 2014.0  
    Mexico                 9009.280                 2015.0  
    Russia                 9054.914                 2015.0  
    Turkey                 9437.372                 2013.0  
    Hungary               12239.894                 2015.0  
    Poland                12495.334                 2014.0  
    Chile                 13340.905                 2014.0  
    Slovak Republic       15991.736                 2015.0  
    Czech Republic        17256.918                 2015.0  
    Estonia               17288.083                 2014.0  
    Greece                18064.288                 2014.0  
    Portugal              19121.592                 2014.0  
    Slovenia              20732.482                 2015.0  
    Spain                 25864.721                 2014.0  
    Korea                 27195.197                 2014.0  
    Italy                 29866.581                 2015.0  
    Japan                 32485.545                 2015.0  
    Israel                35343.336                 2015.0  
    New Zealand           37044.891                 2015.0  
    France                37675.006                 2015.0  
    Belgium               40106.632                 2014.0  
    Germany               40996.511                 2014.0  
    Finland               41973.988                 2014.0  
    Canada                43331.961                 2015.0  
    Netherlands           43603.115                 2014.0  
    Austria               43724.031                 2015.0  
    United Kingdom        43770.688                 2015.0  
    Sweden                49866.266                 2014.0  
    Iceland               50854.583                 2014.0  
    Australia             50961.865                 2014.0  
    Ireland               51350.744                 2014.0  
    Denmark               52114.165                 2015.0  
    United States         55805.204                 2015.0  
    Norway                74822.106                 2015.0  
    Switzerland           80675.308                 2015.0  
    Luxembourg           101994.093                 2014.0  
    
    [36 rows x 30 columns]



```python
#행 추출

#삭제할 행 번호 만들기
remove_indices = [0,1,6,8,33,34,35]
#유지할 행 번호 만들기
keep_indices = list(set(range(36)) - set(remove_indices))
#추출
country_stats = full_country_stats[['GDP per capita', 'Life satisfaction']].iloc[keep_indices]
print(country_stats)
```

                     GDP per capita  Life satisfaction
    Country                                           
    Russia                 9054.914                6.0
    Turkey                 9437.372                5.6
    Hungary               12239.894                4.9
    Poland                12495.334                5.8
    Slovak Republic       15991.736                6.1
    Estonia               17288.083                5.6
    Greece                18064.288                4.8
    Portugal              19121.592                5.1
    Slovenia              20732.482                5.7
    Spain                 25864.721                6.5
    Korea                 27195.197                5.8
    Italy                 29866.581                6.0
    Japan                 32485.545                5.9
    Israel                35343.336                7.4
    New Zealand           37044.891                7.3
    France                37675.006                6.5
    Belgium               40106.632                6.9
    Germany               40996.511                7.0
    Finland               41973.988                7.4
    Canada                43331.961                7.3
    Netherlands           43603.115                7.3
    Austria               43724.031                6.9
    United Kingdom        43770.688                6.8
    Sweden                49866.266                7.2
    Iceland               50854.583                7.5
    Australia             50961.865                7.3
    Ireland               51350.744                7.0
    Denmark               52114.165                7.5
    United States         55805.204                7.2



```python
#데이터 탐색 - 기술 통계량을 출력하거나 시각화
#데이터의 분포 형태를 확인하기 위해서 산점도 생성
ax = country_stats.plot(kind='scatter', x='GDP per capita', y='Life satisfaction')
ax.set(xlabel='1인당 GDP', ylabel='삶의 만족도')
plt.show()
```


    
![png](output_9_0.png)
    



```python
#GDP 대비 삶의 만족도를 예측
#사용할 데이터는 GDP(연속형 데이터) 와 삶의 만족도(연속형 데이터)
#연속형 데이터를 가지고 연속형 데이터를 예측 - 회귀분석
#범주형 데이터를 예측한다면 분류분석

#데이터의 분포가 선형을 이루기 때문에 선형회귀를 수행

#선형회귀 클래스를 import
import sklearn.linear_model

#피처(예측하기 위한 독립변수) 와 레이블(결과로 사용될 종속변수)을 생성
#데이터를 바로 추출하지 않고 np.c_를 이용한 이유는 추출한 데이터를 이차원 배열로
#만들기 위해서
X = np.c_[country_stats['GDP per capita']]
y = np.c_[country_stats['Life satisfaction']]

#print(X)
#print(y)

#모델을 생성 - 훈련 데이터를 가지고 모델을 생성
model = sklearn.linear_model.LinearRegression()
model.fit(X, y)
print(model)
```

    LinearRegression()



```python
# 테스트를 가지고 평가

# 실제 적용
X_new = [[22587]] #키프로스의 GDP
print(model.predict(X_new))

#모델 기반은 알고리즘을 생성하고 그 알고리즘을 이용해서 연산작업을 수행해서 예측
```

    [[5.96242338]]


## 사례기반 머신러닝
### 알고리즘을 만드는 것이 아니고 필요할 때 유사도를 측정해서 판단


```python
#KNN 알고리즘을 예측

import sklearn.neighbors

#입력된 값에서 가장 가까운 3개를 구해서 평균을 내는 회귀 모델
knn = sklearn.neighbors.KNeighborsRegressor(n_neighbors = 3)

#훈련
knn.fit(X, y)

#예측
print(knn.predict(X_new))
```

    [[5.76666667]]


# 머신러닝 프로젝트 연습
## 주택 가격을 예측해보는 다변량 회귀 분석

### 데이터 가져오기


```python
#housing.csv 파일의 내용 가져오기
#파일을 열어서 구분자화 한글 포함 여부 그리고 첫번째의 레이블 여부를 확인
#숫자가 큰 경우에는 천단위 구분기호가 있는지 확인

housing = pd.read_csv('./data/housing.csv')
print(housing.head())
```

       longitude  latitude  housing_median_age  total_rooms  total_bedrooms  \
    0    -122.23     37.88                41.0        880.0           129.0   
    1    -122.22     37.86                21.0       7099.0          1106.0   
    2    -122.24     37.85                52.0       1467.0           190.0   
    3    -122.25     37.85                52.0       1274.0           235.0   
    4    -122.25     37.85                52.0       1627.0           280.0   
    
       population  households  median_income  median_house_value ocean_proximity  
    0       322.0       126.0         8.3252            452600.0        NEAR BAY  
    1      2401.0      1138.0         8.3014            358500.0        NEAR BAY  
    2       496.0       177.0         7.2574            352100.0        NEAR BAY  
    3       558.0       219.0         5.6431            341300.0        NEAR BAY  
    4       565.0       259.0         3.8462            342200.0        NEAR BAY  


### 데이터의 개략적인 정보 확인


```python
housing.info()

#위도: 실수, null 없음
#경도: 실수, null 없음
#중간 연령(지은지 지나온 햇수): 실수 null 없음
#방의 개수: 실수 null 없음
#침실의 개수: 실수, null 이 있음
#인구: 실수, null 이 없음
#가구수: 실수, null 이 없음
#중간 소득: 실수, null 이 없음
#중간 가격: 실수, null 이 없음
#바다 근접성: 객체, null이 없음, 범주형일 가능성이 높음
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20640 entries, 0 to 20639
    Data columns (total 10 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   longitude           20640 non-null  float64
     1   latitude            20640 non-null  float64
     2   housing_median_age  20640 non-null  float64
     3   total_rooms         20640 non-null  float64
     4   total_bedrooms      20433 non-null  float64
     5   population          20640 non-null  float64
     6   households          20640 non-null  float64
     7   median_income       20640 non-null  float64
     8   median_house_value  20640 non-null  float64
     9   ocean_proximity     20640 non-null  object 
    dtypes: float64(9), object(1)
    memory usage: 1.6+ MB


### 데이터 분포 확인


```python
#바다 근접성이 범주형 인지 확인 - value_counts()로 빈도수 확인
#중복된 데이터가 많음 - unique()로 확인 가능
print(housing['ocean_proximity'].value_counts())
#중복된 값이 많고 값의 종류가 몇가지 안되면 범주형
```

    <1H OCEAN     9136
    INLAND        6551
    NEAR OCEAN    2658
    NEAR BAY      2290
    ISLAND           5
    Name: ocean_proximity, dtype: int64



```python
#숫자 열은 describe() 를 이용해서 기술 통계량 확인
print(housing.describe())
#기술통계량을 가지고 데이터의 범위를 확인
#각 피쳐별로 숫자의 범위가 많이 다름 - 정규화를 해주어야 할 것 같음
```

              longitude      latitude  housing_median_age   total_rooms  \
    count  20640.000000  20640.000000        20640.000000  20640.000000   
    mean    -119.569704     35.631861           28.639486   2635.763081   
    std        2.003532      2.135952           12.585558   2181.615252   
    min     -124.350000     32.540000            1.000000      2.000000   
    25%     -121.800000     33.930000           18.000000   1447.750000   
    50%     -118.490000     34.260000           29.000000   2127.000000   
    75%     -118.010000     37.710000           37.000000   3148.000000   
    max     -114.310000     41.950000           52.000000  39320.000000   
    
           total_bedrooms    population    households  median_income  \
    count    20433.000000  20640.000000  20640.000000   20640.000000   
    mean       537.870553   1425.476744    499.539680       3.870671   
    std        421.385070   1132.462122    382.329753       1.899822   
    min          1.000000      3.000000      1.000000       0.499900   
    25%        296.000000    787.000000    280.000000       2.563400   
    50%        435.000000   1166.000000    409.000000       3.534800   
    75%        647.000000   1725.000000    605.000000       4.743250   
    max       6445.000000  35682.000000   6082.000000      15.000100   
    
           median_house_value  
    count        20640.000000  
    mean        206855.816909  
    std         115395.615874  
    min          14999.000000  
    25%         119600.000000  
    50%         179700.000000  
    75%         264725.000000  
    max         500001.000000  



```python
#숫자 열은 histogram 이나 scatter를 이용해서도 확인
housing.hist(bins=50, figsize=(20,15))
#시각화 한 내용을 저장
plt.savefig('./data/attribute_histogram_plot', format='png', dpi=300)
plt.show()

#데이터의 치우침을 확인
#데이터의 치우침이 심하면 정규 분포와 유사해지도록 
#로그 변환을 해주어야 할 수 도 있음
```


    
![png](output_22_0.png)
    


### 훈련 데이터 와 검증 데이터를 나누는 작업 - 보통의 경우는 8:2
### 여러 모델을 가지고 테스트 할 거라면 seed 를 고정시켜야 합니다.


```python
#seed 고정
np.random.seed(42)

#데이터 와 테스트 데이터의 비율을 매개변수로 받아서 데이터를 분할해서 리턴해주는 함수
def split_train_test(data, test_ratio):
    #무작위로 섞어서 인덱스를 추출
    shuffled_indices = np.random.permutation(len(data))
    #테스트 데이터의 크기 계산
    test_set_size = int(len(data) * test_ratio)
    #테스트 데이터의 인덱스 추출
    test_indices = shuffled_indices[:test_set_size]
    #훈련 데이터의 인덱스 추출
    train_indices = shuffled_indices[test_set_size:]
    #각 인덱스에 해당하는 데이터 리턴
    return data.iloc[train_indices], data.iloc[test_indices]

#데이터 분할
train_set, test_set = split_train_test(housing, 0.2)
print("훈련 데이터 개수:", len(train_set))
print("검증 데이터 개수:", len(test_set))
```

    훈련 데이터 개수: 16512
    검증 데이터 개수: 4128



```python
#데이터에는 다른 데이터를 구별할 수 있도록 해주는 컬럼을 사용하는 것이 좋습니다.
#데이터베이스에 기본키의 개념입니다.
#현재 데이터프레임에 기본키로 설정할 만한 것이 없다면 새로운 컬럼을 만들거나
#불가피한 경우 행 번호를 사용합니다.
#기본키는 변경이 안됩니다.
#이번 데이터의 경우는 위도 와 경도를 합친 값을 사용하는 것이 좋습니다.
```


```python
#머신러닝 프로젝트

#1. 데이터에는 하나의 행을 구분하기 위한 컬럼이 존재하는 것이 좋습니다.

#2. 훈련 데이터와 검증 데이터 분리하기
#sklearn.model_selection.train_test_split 메소드 이용 가능
#첫번째 매개변수는 DataSet, test_size ㅡ 검증 데이터의 비율 : 0.0 ~ 1.0, random_state 는 seed 번호

#3. 계층적 샘플림
#샘플링을 할 때 모수의 비율과 일치하는 형태로 샘플링이 되어야 합니다.
#남자의 비율이 48.7% 이고 여성의 비율이 51.3% 라면 샘플링 데이터에서도 이 비율을 유지해야 합니다.
#중요한 피처가 있다면 이 피처는 샘플링 비율을 잘 조정해야 합니다.
#계층 분할을 할 떄 사용할 수 있는 함수 중 하ㅏ가 pandas 의 cut(연속형 데이터를 카테고리화 할 때 주로 이용) 이라는 함수입니다.
#첫번째 매개변수로 분할할 데이터를 설정
#bins 에 구간을 설정, labels 에 각 구간의 이름을 설정

#sklearn.model_selection 의 StratifiedshuffleSplit 클래스를 이용해서 계층적 샘플링 가능
#StratifiedshuffleSplit(n_split=서브셋의 개수, test_size=검증 데이터의 비율, train_size=훈련 데이터의 비율, random_state=시드번호)

#StratifiedshuffleSplit객체.split(분할할 데이터, 적용할 계층 데이터)을 호출하면 
#n_split 의 개수만큼 훈련 데이터와 검증 데이터의 인덱스를 리턴합니다.

#4.훈련 데이터 탐색적 분석
#데이터의 분포나 기술 통계값을 확인해서 의미를 찾아내는 직업 -

#housing 데이터를 8:2 비율로 훈련 데이터와 검증 데이터로 분할
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state = 42)
print(train_set.shape)
print(test_set.shape)
```

    (16512, 10)
    (4128, 10)


## 연속형 데이터의 구간 분한
### median_income 항목을 0, 1.5, 3.0, 4.5, 6 으로 분할


```python
#np.inf 는 마지막 숫자를 몰라서 쓴거, 알면은 숫자로 표기해도 무관하다.
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0, 1.5, 3.0, 4.5, 6, np.inf], 
                              labels=[1,2,3,4,5])
print(housing['income_cat'].value_counts())
```

    3    7236
    2    6581
    4    3639
    5    2362
    1     822
    Name: income_cat, dtype: int64



```python
housing["income_cat"].hist()
```




    <AxesSubplot:>




    
![png](output_29_1.png)
    



```python
#원본 데이터의 비율
print(housing['income_cat'].value_counts()/len(housing))
```

    3    0.350581
    2    0.318847
    4    0.176308
    5    0.114438
    1    0.039826
    Name: income_cat, dtype: float64


# 계층적 샘플링


```python
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size=0.2, random_state = 42)
print(split)

#분할 
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#분할 비율 확인
print(strat_test_set['income_cat'].value_counts() / len(strat_test_set))
```

    StratifiedShuffleSplit(n_splits=1, random_state=42, test_size=0.2,
                train_size=None)
    3    0.350533
    2    0.318798
    4    0.176357
    5    0.114583
    1    0.039729
    Name: income_cat, dtype: float64



```python
#원본 데이터와 랜덤 샘플링을 한 경우 와 계층적 샘플링을 한 경우의 비교

#데이터를 주면 비율을 리턴해주는 함수
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

#랜덤 샘플링
train_set, test_set = train_test_split(housing, test_size=0.2, random_state = 42)

#데이터의 비율을 데이터프레임으로 생성
compare_props = pd.DataFrame({
    "원본": income_cat_proportions(housing),
    "랜덤": income_cat_proportions(test_set),
    "계층": income_cat_proportions(strat_test_set),
}).sort_index()

#일치하는 비율 계산
compare_props["랜덤 비율"] = 100 * compare_props["랜덤"] / compare_props["원본"]
compare_props["계층 비율"] = 100 * compare_props["계층"] / compare_props["원본"]
print(compare_props)
```

             원본        랜덤        계층       랜덤 비율       계층 비율
    1  0.039826  0.040213  0.039729  100.973236   99.756691
    2  0.318847  0.324370  0.318798  101.732260   99.984805
    3  0.350581  0.358527  0.350533  102.266446   99.986180
    4  0.176308  0.167393  0.176357   94.943666  100.027480
    5  0.114438  0.109496  0.114583   95.681626  100.127011


## 데이터의 탐색적 분석(EDA)
### 위도와 경도를 이용한 산포도 작성 - 데이터의 분포를 확인하기 위해서 사용


```python
#데이터를 복제
housing = strat_train_set.copy()

ax = housing.plot(kind='scatter', x = 'longitude', y='latitude')
ax.set(xlabel='경도', ylabel='위도')
#그림 저장 - 나중에 보고서에 사용할 지 모르고 다른 곳에서 이미지를 확인하고자 할 때
#코드를 실행하는 것보다는 이미지 파일을 실행하는 것이 깔끔함
plt.savefig("bad_visulization_plot.png", format='png', dpi=300)
```


    
![png](output_35_0.png)
    



```python
#데이터를 복제
housing = strat_train_set.copy()

#alpha를 0.1로 낮춰서 데이터의 크기를 조금 더 명확하게 출력
ax = housing.plot(kind='scatter', x = 'longitude', y='latitude', alpha=0.1)
ax.set(xlabel='경도', ylabel='위도')
#그림 저장 - 나중에 보고서에 사용할 지 모르고 다른 곳에서 이미지를 확인하고자 할 때
#코드를 실행하는 것보다는 이미지 파일을 실행하는 것이 깔끔함
plt.savefig("bad_visulization_plot.png", format='png', dpi=300)
```


    
![png](output_36_0.png)
    



```python
#alpha를 0.1로 낮춰서 데이터의 크기를 조금 더 명확하게 출력
#점의 크기를 이용해서 인구 수를 출력
#점의 색상을 이용해서 중간 주택 가격을 나타냄 : 빨간색이 비쌈
ax = housing.plot(kind='scatter', x = 'longitude', y='latitude', alpha=0.4,
                 s = housing['population']/100, label='인구',
                 c = "median_house_value", cmap=plt.get_cmap('jet'),
                 colorbar=True, figsize=(10,8))
ax.set(xlabel='경도', ylabel='위도')
plt.legend()
#그림 저장 - 나중에 보고서에 사용할 지 모르고 다른 곳에서 이미지를 확인하고자 할 때
#코드를 실행하는 것보다는 이미지 파일을 실행하는 것이 깔끔함
plt.savefig("bad_visulization_plot.png", format='png', dpi=300)
```


    
![png](output_37_0.png)
    


# 상관 관계(계수) 확인


```python
#모든 열의 상관관계 확인
corr_matrix = housing.corr()
#print(corr_matrix)

#중간 주택 가격과 상관관계가 높은 순으로 출력
print(corr_matrix['median_house_value'].sort_values(ascending=False))

#중간 주택 가격 과 가장 관련성이 높은 항목은 아마도 중간 소득
#0.7 이상이면 매우 높은 상관관계
#0.4 이상이면 높은 상관관계
#나머지는 낮거나 상관관계가 없음
```

    median_house_value    1.000000
    median_income         0.687160
    total_rooms           0.135097
    housing_median_age    0.114110
    households            0.064506
    total_bedrooms        0.047689
    population           -0.026920
    longitude            -0.047432
    latitude             -0.142724
    Name: median_house_value, dtype: float64



```python
#산포도를 이용해서 피쳐들의 관계를 확인
#여러 개의 산포도를 그릴 때는 pandas.plotting 의 scatter_matrix 를 이용할 수 있음
from pandas.plotting import scatter_matrix

#관계를 파악할 속성을 추출
attributes = ["median_house_value", "median_income", 
              "total_rooms", "housing_median_age"]

scatter_matrix(housing[attributes], figsize=(12,9))

plt.savefig('scatter_matrix.png', format='png', dpi=300)
```


    
![png](output_40_0.png)
    



```python
#중간 주택 가격 과 중간 소득 간의 산점도를 그리기
housing.plot(kind = 'scatter', x='median_income', y='median_house_value',
            alpha = 0.1)
plt.axis=([0,16,0, 550000])
plt.savefig('scatter_income_vs_value.png', format='png', dpi=300)

#상관관계가 높을 때 직선으로 보여지는 데이터 집단이 있다면 제거하는 것이 
#좋은 모델을 만드는데 도움이 됩니다.
```


    
![png](output_41_0.png)
    


### 여러 특성조합으로 새로운 특성을 생성


```python
#total_rooms 를 households 로 나눈 특성
housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
#total_bedrooms를 total_rooms 로 나눈 특성
housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
#population 을 households로 나눈 특성
housing['population_per_household'] = housing['population'] / housing['households']

#모든 열의 상관관계 확인
corr_matrix = housing.corr()

#중간 주택 가격과 상관관계가 높은 순으로 출력
print(corr_matrix['median_house_value'].sort_values(ascending=False))
```

    median_house_value          1.000000
    median_income               0.687160
    rooms_per_household         0.146285
    total_rooms                 0.135097
    housing_median_age          0.114110
    households                  0.064506
    total_bedrooms              0.047689
    population_per_household   -0.021985
    population                 -0.026920
    longitude                  -0.047432
    latitude                   -0.142724
    bedrooms_per_room          -0.259984
    Name: median_house_value, dtype: float64


### 레이블을 분리


```python
#훈련에 사용할 피처를 분리
housing = strat_train_set.drop('median_house_value', axis=1)

#레이블로 사용할 피처를 분리
housing_labels = strat_train_set['median_house_value'].copy()
```

### 누락된 처리


```python
#누락된 데이터 찾기
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
print(sample_incomplete_rows)
```

           longitude  latitude  housing_median_age  total_rooms  total_bedrooms  \
    4629     -118.30     34.07                18.0       3759.0             NaN   
    6068     -117.86     34.01                16.0       4632.0             NaN   
    17923    -121.97     37.35                30.0       1955.0             NaN   
    13656    -117.30     34.05                 6.0       2155.0             NaN   
    19252    -122.79     38.48                 7.0       6837.0             NaN   
    
           population  households  median_income ocean_proximity income_cat  
    4629       3296.0      1462.0         2.2708       <1H OCEAN          2  
    6068       3038.0       727.0         5.1762       <1H OCEAN          4  
    17923       999.0       386.0         4.6328       <1H OCEAN          4  
    13656      1039.0       391.0         1.6675          INLAND          2  
    19252      3468.0      1405.0         3.1662       <1H OCEAN          3  



```python
#누락된 데이터가 많지 않은 경우 행을 제거
housing.dropna(subset=["total_bedrooms"]).info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 16354 entries, 17606 to 15775
    Data columns (total 10 columns):
     #   Column              Non-Null Count  Dtype   
    ---  ------              --------------  -----   
     0   longitude           16354 non-null  float64 
     1   latitude            16354 non-null  float64 
     2   housing_median_age  16354 non-null  float64 
     3   total_rooms         16354 non-null  float64 
     4   total_bedrooms      16354 non-null  float64 
     5   population          16354 non-null  float64 
     6   households          16354 non-null  float64 
     7   median_income       16354 non-null  float64 
     8   ocean_proximity     16354 non-null  object  
     9   income_cat          16354 non-null  category
    dtypes: category(1), float64(8), object(1)
    memory usage: 1.3+ MB



```python
#누락된 데이터가 많은 열이 있으면 그 열을 제거
sample_incomplete_rows.drop('total_bedrooms', axis=1)
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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>ocean_proximity</th>
      <th>income_cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4629</th>
      <td>-118.30</td>
      <td>34.07</td>
      <td>18.0</td>
      <td>3759.0</td>
      <td>3296.0</td>
      <td>1462.0</td>
      <td>2.2708</td>
      <td>&lt;1H OCEAN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6068</th>
      <td>-117.86</td>
      <td>34.01</td>
      <td>16.0</td>
      <td>4632.0</td>
      <td>3038.0</td>
      <td>727.0</td>
      <td>5.1762</td>
      <td>&lt;1H OCEAN</td>
      <td>4</td>
    </tr>
    <tr>
      <th>17923</th>
      <td>-121.97</td>
      <td>37.35</td>
      <td>30.0</td>
      <td>1955.0</td>
      <td>999.0</td>
      <td>386.0</td>
      <td>4.6328</td>
      <td>&lt;1H OCEAN</td>
      <td>4</td>
    </tr>
    <tr>
      <th>13656</th>
      <td>-117.30</td>
      <td>34.05</td>
      <td>6.0</td>
      <td>2155.0</td>
      <td>1039.0</td>
      <td>391.0</td>
      <td>1.6675</td>
      <td>INLAND</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19252</th>
      <td>-122.79</td>
      <td>38.48</td>
      <td>7.0</td>
      <td>6837.0</td>
      <td>3468.0</td>
      <td>1405.0</td>
      <td>3.1662</td>
      <td>&lt;1H OCEAN</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
#다른 값으로 대체
#중간값으로 대체
median = housing['total_bedrooms'].median()
housing['total_bedrooms'].fillna(median, inplace=True)
housing.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 16512 entries, 17606 to 15775
    Data columns (total 10 columns):
     #   Column              Non-Null Count  Dtype   
    ---  ------              --------------  -----   
     0   longitude           16512 non-null  float64 
     1   latitude            16512 non-null  float64 
     2   housing_median_age  16512 non-null  float64 
     3   total_rooms         16512 non-null  float64 
     4   total_bedrooms      16512 non-null  float64 
     5   population          16512 non-null  float64 
     6   households          16512 non-null  float64 
     7   median_income       16512 non-null  float64 
     8   ocean_proximity     16512 non-null  object  
     9   income_cat          16512 non-null  category
    dtypes: category(1), float64(8), object(1)
    memory usage: 1.3+ MB



```python
#sklearn 의 결측값 대체 클래스 이용하기
from sklearn.impute import SimpleImputer

#중간값으로 채워주는 객체 생성
imputer = SimpleImputer(strategy='median')

#숫자 데이터가 아닌 열을 제거
housing_num = housing.drop('ocean_proximity', axis=1)

#데이터 추정
print(imputer.fit(housing_num))
#데이터 확인
print(imputer.statistics_)
print(housing_num.median().values)

#데이터 변환
X = imputer.transform(housing_num)
#변환된 데이터로 새로운 데이터프레임을 생성
housing_tr = pd.DataFrame(X, columns=housing_num.columns, 
                         index = list(housing.index.values))
print(housing_tr.loc[sample_incomplete_rows.index.values])
```

    SimpleImputer(strategy='median')
    [-118.51     34.26     29.     2119.5     433.     1164.      408.
        3.5409    3.    ]
    [-118.51     34.26     29.     2119.5     433.     1164.      408.
        3.5409]
           longitude  latitude  housing_median_age  total_rooms  total_bedrooms  \
    4629     -118.30     34.07                18.0       3759.0           433.0   
    6068     -117.86     34.01                16.0       4632.0           433.0   
    17923    -121.97     37.35                30.0       1955.0           433.0   
    13656    -117.30     34.05                 6.0       2155.0           433.0   
    19252    -122.79     38.48                 7.0       6837.0           433.0   
    
           population  households  median_income  income_cat  
    4629       3296.0      1462.0         2.2708         2.0  
    6068       3038.0       727.0         5.1762         4.0  
    17923       999.0       386.0         4.6328         4.0  
    13656      1039.0       391.0         1.6675         2.0  
    19252      3468.0      1405.0         3.1662         3.0  



```python
#하이퍼 파라미터 확인 - 하이퍼 파라미터는 파라미터 이름을 호출하면 확인 가능
#모델 파라미터는 이름뒤에 _를 붙여서 호출하면 확인 가능
print(imputer.strategy)
```

    median


### 범주형 데이터를 정수화


```python
#문자열로 만들어진 ocean_proximity 를 숫자로 변환
housing_cat = housing['ocean_proximity']
print(housing_cat.head(10))
print()

housing_cat_encoded, housing_categories = housing_cat.factorize()
print(housing_cat_encoded[:10])
print()
print(housing_categories)
```

    17606     <1H OCEAN
    18632     <1H OCEAN
    14650    NEAR OCEAN
    3230         INLAND
    3555      <1H OCEAN
    19480        INLAND
    8879      <1H OCEAN
    13685        INLAND
    4937      <1H OCEAN
    4861      <1H OCEAN
    Name: ocean_proximity, dtype: object
    
    [0 0 1 2 0 2 0 2 0 0]
    
    Index(['<1H OCEAN', 'NEAR OCEAN', 'INLAND', 'NEAR BAY', 'ISLAND'], dtype='object')



```python
#sklearn 의 OrdinalEncoder 를 이용한 방식
from sklearn.preprocessing import OrdinalEncoder

#sklearn 은 머신러닝 패키지라서 데이터가 2차원 배열이어야 합니다.
housing_cat = housing[["ocean_proximity"]]

#encoder 객체를 생성
ordinal_encoder = OrdinalEncoder()
#훈련과 변환을 한번에 처리
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[:10])

#OrdinalEncoder 객체는 categories 라는 프로퍼티로 카테고리를 저장하고 있음
print(ordinal_encoder.categories_)
```

    [[0.]
     [0.]
     [4.]
     [1.]
     [0.]
     [1.]
     [0.]
     [1.]
     [0.]
     [0.]]
    [array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
          dtype=object)]



```python
#원 핫 인코딩을 위해서 2차원 배열을 생성
housing_cat_1hot = housing_cat_encoded.reshape(-1, 1)
print(housing_cat_1hot[:10])

#원핫 인코딩 수행
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(categories = 'auto', sparse=False)
housing_cat_onehot = encoder.fit_transform(housing_cat_1hot)

#밀집 배열로 출력 - toarray()를 호출하지 않으면 희소행렬로 리턴
print(housing_cat_onehot[:10])
```

    [[0.]
     [0.]
     [4.]
     [1.]
     [0.]
     [1.]
     [0.]
     [1.]
     [0.]
     [0.]]
    [[1. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0.]
     [0. 0. 0. 0. 1.]
     [0. 1. 0. 0. 0.]
     [1. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0.]
     [1. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0.]
     [1. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0.]]



```python
#PipeLine 처리
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#Pipeline 생성
#결측값을 중간값으로 대체
#표준화 작업
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), 
                        ('std_scaler', StandardScaler())])

#결측값을 중간값으로 대체하고 표준화 작업을 수행해서 결과를 리턴
housing_num_tr = num_pipeline.fit_transform(housing_num)
print(housing_num_tr)
```

    [[-1.15604281  0.77194962  0.74333089 ... -0.42069842 -0.61493744
      -0.95445595]
     [-1.17602483  0.6596948  -1.1653172  ... -1.02222705  1.33645936
       1.89030518]
     [ 1.18684903 -1.34218285  0.18664186 ... -0.0933178  -0.5320456
      -0.95445595]
     ...
     [ 1.58648943 -0.72478134 -1.56295222 ...  0.71315642 -0.3167053
      -0.00620224]
     [ 0.78221312 -0.85106801  0.18664186 ... -0.37545069  0.09812139
      -0.00620224]
     [-1.43579109  0.99645926  1.85670895 ...  0.3777909  -0.15779865
      -0.00620224]]



```python
#sklearn 버전 확인
import sklearn
print(sklearn.__version__)
```

    0.24.1



```python
from sklearn.compose import ColumnTransformer

#적용할 열의 리스트
num_attribs = list(housing_num) #Pipeline 적용할 컬럼 목록
cat_attribs = ["ocean_proximity"] #OneHotEncoder 적용할 컬럼 목록

#컬럼 별로 서로 다른 변환기를 적용할 수 있는 클래스의 객체 생성
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(categories='auto'), cat_attribs)
])

#변환 작업 수행
housing_propared = full_pipeline.fit_transform(housing)
print(housing_propared)
```

    [[-1.15604281  0.77194962  0.74333089 ...  0.          0.
       0.        ]
     [-1.17602483  0.6596948  -1.1653172  ...  0.          0.
       0.        ]
     [ 1.18684903 -1.34218285  0.18664186 ...  0.          0.
       1.        ]
     ...
     [ 1.58648943 -0.72478134 -1.56295222 ...  0.          0.
       0.        ]
     [ 0.78221312 -0.85106801  0.18664186 ...  0.          0.
       0.        ]
     [-1.43579109  0.99645926  1.85670895 ...  0.          1.
       0.        ]]



```python
#머신 러닝 모델에 적용할 데이터 확인
print(housing_propared) #독립 변수 - 2차원 배열 확인
print()
print(housing_labels) #종속 변수 - 결과에 사용될 변수 - label 이나 target 이라고 함
```

    [[-1.15604281  0.77194962  0.74333089 ...  0.          0.
       0.        ]
     [-1.17602483  0.6596948  -1.1653172  ...  0.          0.
       0.        ]
     [ 1.18684903 -1.34218285  0.18664186 ...  0.          0.
       1.        ]
     ...
     [ 1.58648943 -0.72478134 -1.56295222 ...  0.          0.
       0.        ]
     [ 0.78221312 -0.85106801  0.18664186 ...  0.          0.
       0.        ]
     [-1.43579109  0.99645926  1.85670895 ...  0.          1.
       0.        ]]
    
    17606    286600.0
    18632    340600.0
    14650    196900.0
    3230      46300.0
    3555     254500.0
               ...   
    6563     240200.0
    12053    113000.0
    13908     97800.0
    11159    225900.0
    15775    500001.0
    Name: median_house_value, Length: 16512, dtype: float64



```python
#모델을 생성해서 훈련
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_propared, housing_labels)
```




    LinearRegression()




```python
#예측 - 테스트
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

#처음 5개의 예측 값과 실제 값을 출력
print("예측 값:", lin_reg.predict(some_data_prepared))
print("실제 값:", list(some_labels))
```

    예측 값: [205841.96139906 329060.05217695 205219.96047357  61859.01340291
     196908.23636333]
    실제 값: [286600.0, 340600.0, 196900.0, 46300.0, 254500.0]


### 평가 지표 확인


```python
#RMSE 확인
from sklearn.metrics import mean_squared_error

#검증 데이터를 가지고 예측값을 생성
housing_predictions = lin_reg.predict(housing_propared)
#실제 값과 비교해서 RMSE를 측정
lin_mse = mean_squared_error(housing_labels, housing_predictions)
print("제곱근 하기 전의 MSE:", lin_mse)
print("RMSE:", np.sqrt(lin_mse))
```

    제곱근 하기 전의 MSE: 4742665159.4621315
    RMSE: 68867.01067610044



```python
#MAE 확인
from sklearn.metrics import mean_absolute_error
lin_mae = mean_absolute_error(housing_labels, housing_predictions)
print("MAE:", lin_mae)
```

    MAE: 50005.718400295045


## 다른 모델 적용 - DecisionTreeRegressor


```python
#의사결정나무 모델 적용
from sklearn.tree import DecisionTreeRegressor

#훈련
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_propared, housing_labels)

#예측
housing_predictions = tree_reg.predict(housing_propared)

#평가 지표 확인
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("RMSE:", tree_rmse)
```

    RMSE: 0.0


## 교차 검증 수행


```python
from sklearn.model_selection import cross_val_score

#교차 검증 수행
#tree_reg 은 평가할 모델
#housing_propared : 훈련 데이터
#housing_labels : 타겟 데이터
#scoring: 평가 지표
#cv: k 겹
scores = cross_val_score(tree_reg, housing_propared, housing_labels, 
                         scoring = 'neg_mean_squared_error', 
                         cv = 10)

#결과에 제곱근을 취함
tree_rmse_scores = np.sqrt(-scores)

#평균과 표준 편차 및 값을 출력
print("Score:", tree_rmse_scores)
print("Score 평균:", tree_rmse_scores.mean())
print("Score 편차:", tree_rmse_scores.std())
```

    Score: [66511.83866866 65218.24365445 72073.11298807 69065.87982634
     69044.08430761 75497.52594563 67274.5960707  69372.51908292
     70886.10880051 69172.98810858]
    Score 평균: 69411.68974534725
    Score 편차: 2777.9311422791166



```python
scores = cross_val_score(lin_reg, housing_propared, housing_labels, 
                         scoring = 'neg_mean_squared_error', 
                         cv = 10)

#결과에 제곱근을 취함
lin_rmse_scores = np.sqrt(-scores)

#평균과 표준 편차 및 값을 출력
print("Score:", lin_rmse_scores)
print("Score 평균:", lin_rmse_scores.mean())
print("Score 편차:", lin_rmse_scores.std())
```

    Score: [67621.36459192 67050.2893707  68159.77945126 74185.17069359
     68112.25073467 71610.00721757 65235.24278175 68160.85292722
     72191.44396288 68163.77349381]
    Score 평균: 69049.01752253593
    Score 편차: 2581.324004019166



```python
#새로운 모델 학습
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_propared, housing_labels)

scores = cross_val_score(forest_reg, housing_propared, housing_labels, 
                         scoring = 'neg_mean_squared_error', 
                         cv = 10)

#결과에 제곱근을 취함
forest_rmse_scores = np.sqrt(-scores)

#평균과 표준 편차 및 값을 출력
print("Score:", forest_rmse_scores)
print("Score 평균:", forest_rmse_scores.mean())
print("Score 편차:", forest_rmse_scores.std())
```

## 하이퍼 파라미터 튜닝 - 최적의 하이퍼 파라미터를 찾는 일


```python
from sklearn.model_selection import GridSearchCV

#하이퍼 파라미터 조합 생성
#첫번째 디셔너리의 조합: 3*4 = 12가지 조합 테스트
#두번째 디셔너리의 조합: 1*2*3 = 6가지 조합 테스트
#12 + 6 = 18 가지 조합을 테스트하기 위한 파라미터 집합
param_grid = [{'n_estimators':[3, 10, 30], 'max_features': [2,4,6,8]},
              {'bootstrap': [False], 'n_estimators':[3,10],
               'max_features': [2,3,4]}]

#하이퍼 파라미터 튜닝을 할 모델을 생성
forest_reg = RandomForestRegressor(random_state =  42)

#하이퍼 파라미터 튜닝을 수행할 객체 생성
#cv 는 수행할 횟수 : 5 * 18 = 90번
#scoring 은 이전과 동일
#return_train_score는 점수를 반환할지 여부를 설정
#n_jobs 는 프로세서 개수
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, 
                          scoring='neg_mean_squared_error',
                          return_train_score=True, n_jobs=-1)
#훈련
grid_search.fit(housing_propared, housing_labels)
```


```python
#최적의 파라미터 확인
print(grid_search.best_params_)
#max_features 와 n_estimators 가 8 과 30입니다.
#파라미터 리스트를 만들 때 8 과 30이 가장 큰 값이었습니다.
#이런 경우라면 파라미터 의 값을 더 크게 해서 테스트를 해봐야 합니다.
```


```python
#평가 점수 확인
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```


```python
#피처의 중요도 확인
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)

#덜 중요한 피처들을 제거하고 모델을 학습해봐야 합니다.
```


```python

```
