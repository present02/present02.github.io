---
layout: single
title:  "jupyter notebook 변환하기!"
categories: coding
tag: [python, blog, jekyll]
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


**목차**  

로지스틱 회귀 분석 소개  

로지스틱 회귀 직관  

로지스틱 회귀 분석의 가정  

로지스틱 회귀 분석 유형  

라이브러리 가져오기  

데이터 집합 가져오기  

탐색적 데이터 분석  

피쳐 벡터 및 대상 변수 선언  

데이터를 별도의 교육 및 테스트 세트로 분할   

피쳐 엔지니어링  

피쳐 스케일링  

모델 교육  

결과 예측  

정확도 점수 확인   

혼동 행렬  

분류 메트릭  

임계값 레벨 조정  

ROC - AUC  

k-Fold 교차 검증  

그리드 검색 CV를 사용한 하이퍼 파라미터 최적화  

결과 및 결론  

레퍼런스  


1. 로지스틱 회귀 분석 소개   

데이터 과학자들이 새로운 분류 문제를 발견할 수 있는 경우, 가장 먼저 떠오르는 알고리즘은 로지스틱 회귀 분석입니다. 개별 클래스 집합에 대한 관찰을 예측하는 데 사용되는 지도 학습 분류 알고리즘입니다. 실제로 관측치를 여러 범주로 분류하는 데 사용됩니다. 따라서, 그것의 출력은 본질적으로 별개입니다. 로지스틱 회귀 분석을 로짓 회귀 분석이라고도 합니다. 분류 문제를 해결하는 데 사용되는 가장 단순하고 간단하며 다용도의 분류 알고리즘 중 하나입니다.


2. 로지스틱 회귀 직관   

통계학에서 로지스틱 회귀 모형은 주로 분류 목적으로 사용되는 널리 사용되는 통계 모형입니다. 즉, 관측치 집합이 주어지면 로지스틱 회귀 알고리즘을 사용하여 관측치를 두 개 이상의 이산 클래스로 분류할 수 있습니다. 따라서 대상 변수는 본질적으로 이산적입니다.

로지스틱 회귀 분석 알고리즘은 다음과 같이 작동합니다


선형 방정식 구현  

로지스틱 회귀 분석 알고리즘은 반응 값을 예측하기 위해 독립 변수 또는 설명 변수가 있는 선형 방정식을 구현하는 방식으로 작동합니다. 예를 들어, 우리는 공부한 시간의 수와 시험에 합격할 확률의 예를 고려합니다. 여기서 연구된 시간 수는 설명 변수이며 x1로 표시됩니다. 합격 확률은 반응 변수 또는 목표 변수이며 z로 표시됩니다.



만약 우리가 하나의 설명 변수(x1)와 하나의 반응 변수(z)를 가지고 있다면, 선형 방정식은 다음과 같은 방정식으로 수학적으로 주어질 것입니다


z = β0 + β1x1   

여기서 계수 β0과 β1은 모형의 모수입니다.



설명 변수가 여러 개인 경우, 위의 방정식은 다음과 같이 확장될 수 있습니다.  

z = β0 + β1x1+ β2x2+……..+ βnxn  

여기서 계수 β0, β1, β2 및 βn은 모델의 매개변수입니다.



따라서 예측 반응 값은 위의 방정식에 의해 주어지며 z로 표시됩니다.


시그모이드 함수  

z로 표시된 이 예측 반응 값은 0과 1 사이에 있는 확률 값으로 변환됩니다. 우리는 예측 값을 확률 값에 매핑하기 위해 시그모이드 함수를 사용합니다. 그런 다음 이 시그모이드 함수는 실제 값을 0과 1 사이의 확률 값으로 매핑합니다.



기계 학습에서 시그모이드 함수는 예측을 확률에 매핑하는 데 사용됩니다. 시그모이드 함수는 S자형 곡선을 가지고 있습니다. 그것은 시그모이드 곡선이라고도 불립니다.



Sigmoid 함수는 로지스틱 함수의 특수한 경우입니다. 그것은 다음과 같은 수학 공식에 의해 주어집니다.



다음 그래프로 시그모이드 함수를 그래픽으로 표현할 수 있습니다.


의사결정경계  

시그모이드 함수는 0과 1 사이의 확률 값을 반환합니다. 그런 다음 이 확률 값은 "0" 또는 "1"인 이산 클래스에 매핑됩니다. 이 확률 값을 이산 클래스(통과/실패, 예/아니오, 참/거짓)에 매핑하기 위해 임계값을 선택합니다. 이 임계값을 의사결정 경계라고 합니다. 이 임계값을 초과하면 확률 값을 클래스 1에 매핑하고 클래스 0에 매핑합니다.



수학적으로 다음과 같이 표현할 수 있습니다



p ◦ 0.5 => 클래스 = 1



p < 0.5 => 클래스 = 0



일반적으로 의사 결정 경계는 0.5로 설정됩니다. 따라서 확률 값이 0.8(> 0.5)이면 이 관측치를 클래스 1에 매핑합니다. 마찬가지로 확률 값이 0.2(< 0.5)이면 이 관측치를 클래스 0에 매핑합니다. 이것은 아래 그래프에 나와 있습니다


예측하기  

이제 우리는 로지스틱 회귀 분석에서 시그모이드 함수와 결정 경계에 대해 알고 있습니다. 우리는 시그모이드 함수와 결정 경계에 대한 지식을 사용하여 예측 함수를 작성할 수 있습니다. 로지스틱 회귀 분석의 예측 함수는 관측치가 양수, 예 또는 참일 확률을 반환합니다. 이를 클래스 1이라고 하며 P(클래스 = 1)로 표시합니다. 확률이 1에 가까우면 관측치가 클래스 1에 있고 그렇지 않으면 클래스 0에 있다는 것을 모형에 대해 더 확신할 수 있습니다.


3. 로지스틱 회귀 분석의 가정   



로지스틱 회귀 분석 모형에는 몇 가지 주요 가정이 필요합니다. 다음은 다음과 같습니다



1. 로지스틱 회귀 분석 모형에서는 종속 변수가 이항, 다항식 또는 순서형이어야 합니다.



2. 관측치가 서로 독립적이어야 합니다. 따라서 관측치는 반복적인 측정에서 나와서는 안 됩니다.



3. 로지스틱 회귀 분석 알고리즘에는 독립 변수 간의 다중 공선성이 거의 또는 전혀 필요하지 않습니다. 즉, 독립 변수들이 서로 너무 높은 상관 관계를 맺어서는 안 됩니다.



4. 로지스틱 회귀 모형은 독립 변수와 로그 승산의 선형성을 가정합니다.



5. 로지스틱 회귀 분석 모형의 성공 여부는 표본 크기에 따라 달라집니다. 일반적으로 높은 정확도를 얻으려면 큰 표본 크기가 필요합니다.


4. 로지스틱 회귀 분석의 유형   



로지스틱 회귀 분석 모형은 대상 변수 범주를 기준으로 세 그룹으로 분류할 수 있습니다. 이 세 그룹은 아래에 설명되어 있습니다



1. 이항 로지스틱 회귀 분석

이항 로지스틱 회귀 분석에서 대상 변수에는 두 가지 범주가 있습니다. 범주의 일반적인 예는 예 또는 아니오, 양호 또는 불량, 참 또는 거짓, 스팸 또는 스팸 없음, 통과 또는 실패입니다.



2. 다항 로지스틱 회귀 분석

다항 로지스틱 회귀 분석에서 대상 변수에는 특정 순서가 아닌 세 개 이상의 범주가 있습니다. 따라서 세 개 이상의 공칭 범주가 있습니다. 그 예들은 사과, 망고, 오렌지 그리고 바나나와 같은 과일의 종류를 포함합니다.



3. 순서형 로지스틱 회귀 분석

순서형 로지스틱 회귀 분석에서 대상 변수에는 세 개 이상의 순서형 범주가 있습니다. 그래서, 범주와 관련된 본질적인 순서가 있습니다. 예를 들어, 학생들의 성적은 불량, 평균, 양호, 우수로 분류될 수 있습니다.


5. 라이브러리 가져오기  



```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```


```python
import warnings

warnings.filterwarnings('ignore')
```

6. 데이터 집합 가져오기  



```python
data = '/content/sample_data/weatherAUS.csv'

df = pd.read_csv(data)
```

7. 탐색적 데이터 분석  



```python
df.shape
```

<pre>
(86877, 23)
</pre>

```python
df.head()
```

<pre>
         Date Location  MinTemp  MaxTemp  Rainfall  Evaporation  Sunshine  \
0  2008-12-01   Albury     13.4     22.9       0.6          NaN       NaN   
1  2008-12-02   Albury      7.4     25.1       0.0          NaN       NaN   
2  2008-12-03   Albury     12.9     25.7       0.0          NaN       NaN   
3  2008-12-04   Albury      9.2     28.0       0.0          NaN       NaN   
4  2008-12-05   Albury     17.5     32.3       1.0          NaN       NaN   

  WindGustDir  WindGustSpeed WindDir9am  ... Humidity9am  Humidity3pm  \
0           W           44.0          W  ...        71.0         22.0   
1         WNW           44.0        NNW  ...        44.0         25.0   
2         WSW           46.0          W  ...        38.0         30.0   
3          NE           24.0         SE  ...        45.0         16.0   
4           W           41.0        ENE  ...        82.0         33.0   

   Pressure9am  Pressure3pm  Cloud9am  Cloud3pm  Temp9am  Temp3pm  RainToday  \
0       1007.7       1007.1       8.0       NaN     16.9     21.8         No   
1       1010.6       1007.8       NaN       NaN     17.2     24.3         No   
2       1007.6       1008.7       NaN       2.0     21.0     23.2         No   
3       1017.6       1012.8       NaN       NaN     18.1     26.5         No   
4       1010.8       1006.0       7.0       8.0     17.8     29.7         No   

   RainTomorrow  
0            No  
1            No  
2            No  
3            No  
4            No  

[5 rows x 23 columns]
</pre>

```python
col_names = df.columns

col_names
```

<pre>
Index(['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
       'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm', 'RainToday', 'RainTomorrow'],
      dtype='object')
</pre>

```python
df.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 86877 entries, 0 to 86876
Data columns (total 23 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Date           86877 non-null  object 
 1   Location       86877 non-null  object 
 2   MinTemp        85681 non-null  float64
 3   MaxTemp        85865 non-null  float64
 4   Rainfall       84535 non-null  float64
 5   Evaporation    47217 non-null  float64
 6   Sunshine       40297 non-null  float64
 7   WindGustDir    80596 non-null  object 
 8   WindGustSpeed  80604 non-null  float64
 9   WindDir9am     79092 non-null  object 
 10  WindDir3pm     83907 non-null  object 
 11  WindSpeed9am   85429 non-null  float64
 12  WindSpeed3pm   84810 non-null  float64
 13  Humidity9am    84811 non-null  float64
 14  Humidity3pm    84390 non-null  float64
 15  Pressure9am    76221 non-null  float64
 16  Pressure3pm    76268 non-null  float64
 17  Cloud9am       53399 non-null  float64
 18  Cloud3pm       52375 non-null  float64
 19  Temp9am        85314 non-null  float64
 20  Temp3pm        84837 non-null  float64
 21  RainToday      84535 non-null  object 
 22  RainTomorrow   84537 non-null  object 
dtypes: float64(16), object(7)
memory usage: 15.2+ MB
</pre>
변수 유형  

이 섹션에서는 데이터 세트를 범주형 변수와 숫자 변수로 분리합니다. 데이터 집합에는 범주형 변수와 숫자 변수가 혼합되어 있습니다. 범주형 변수에는 데이터 유형 개체가 있습니다. 숫자 변수의 데이터 유형은 float64입니다.



우선 범주형 변수를 찾아보겠습니다.



```python
categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)
```

<pre>
There are 7 categorical variables

The categorical variables are : ['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
</pre>

```python
df[categorical].head()
```

<pre>
         Date Location WindGustDir WindDir9am WindDir3pm RainToday  \
0  2008-12-01   Albury           W          W        WNW        No   
1  2008-12-02   Albury         WNW        NNW        WSW        No   
2  2008-12-03   Albury         WSW          W        WSW        No   
3  2008-12-04   Albury          NE         SE          E        No   
4  2008-12-05   Albury           W        ENE         NW        No   

  RainTomorrow  
0           No  
1           No  
2           No  
3           No  
4           No  
</pre>
범주형 변수 요약  

날짜 변수가 있습니다. 날짜 열로 표시됩니다.

6개의 범주형 변수가 있습니다. 이것들은 위치, 윈드 구스트 다이어, 윈드 다이어 9am, 윈드 다이어 3pm, 비 투데이 그리고 비 투데이에 의해 주어집니다.

두 개의 이진 범주형 변수인 RainToday와 RainTomorrow가 있습니다.

내일 비가 목표 변수입니다.


범주형 변수 내의 문제 탐색  

먼저 범주형 변수에 대해 알아보겠습니다.



```python
df[categorical].isnull().sum()
```

<pre>
Date               0
Location           0
WindGustDir     6281
WindDir9am      7785
WindDir3pm      2970
RainToday       2342
RainTomorrow    2340
dtype: int64
</pre>

```python
cat1 = [var for var in categorical if df[var].isnull().sum()!=0]

print(df[cat1].isnull().sum())
```

<pre>
WindGustDir     6281
WindDir9am      7785
WindDir3pm      2970
RainToday       2342
RainTomorrow    2340
dtype: int64
</pre>
데이터 세트에 결측값이 포함된 범주형 변수는 4개뿐임을 알 수 있습니다. 윈드구스트디어, 윈드디어9am, 윈드디어3pm, 레인투데이입니다.


범주형 변수의 빈도 카운트

이제 범주형 변수의 빈도 수를 확인하겠습니다.



```python
for var in categorical: 
    
    print(df[var].value_counts())
```

<pre>
2013-11-12    29
2014-05-01    29
2014-04-24    29
2014-04-25    29
2014-04-26    29
              ..
2007-11-29     1
2007-11-28     1
2007-11-27     1
2007-11-26     1
2008-01-31     1
Name: Date, Length: 3436, dtype: int64
Canberra            3436
Sydney              3344
Melbourne           3193
Albury              3040
Bendigo             3040
Ballarat            3040
MountGinini         3040
Wollongong          3040
Penrith             3039
Tuggeranong         3039
Newcastle           3039
CoffsHarbour        3009
Dartmoor            3009
Watsonia            3009
Portland            3009
Mildura             3009
Cobar               3009
MelbourneAirport    3009
Sale                3009
Moree               3009
Richmond            3009
BadgerysCreek       3009
Williamtown         3009
WaggaWagga          3009
SydneyAirport       3009
NorfolkIsland       3009
NorahHead           3004
Brisbane            2870
Nhil                1578
Name: Location, dtype: int64
W      7070
S      6225
N      6222
WSW    5707
SSW    5454
WNW    5431
SW     5320
SSE    5091
E      4874
ENE    4627
NE     4524
NW     4449
SE     4391
NNE    4378
ESE    3596
NNW    3237
Name: WindGustDir, dtype: int64
N      7609
W      6243
SW     6203
NW     5418
WNW    5200
SSW    5158
WSW    5052
S      4893
NNE    4865
SE     4411
SSE    4385
E      4281
NE     4083
NNW    4049
ENE    3947
ESE    3295
Name: WindDir9am, dtype: int64
S      6648
W      6571
N      5840
SE     5808
NE     5738
WNW    5625
WSW    5314
SSE    5206
E      5048
SSW    5024
SW     4999
NW     4946
ESE    4341
NNE    4334
ENE    4321
NNW    4144
Name: WindDir3pm, dtype: int64
No     65220
Yes    19315
Name: RainToday, dtype: int64
No     65220
Yes    19317
Name: RainTomorrow, dtype: int64
</pre>

```python
for var in categorical: 
    
    print(df[var].value_counts()/np.float(len(df)))
```

<pre>
2013-11-12    0.000334
2014-05-01    0.000334
2014-04-24    0.000334
2014-04-25    0.000334
2014-04-26    0.000334
                ...   
2007-11-29    0.000012
2007-11-28    0.000012
2007-11-27    0.000012
2007-11-26    0.000012
2008-01-31    0.000012
Name: Date, Length: 3436, dtype: float64
Canberra            0.039550
Sydney              0.038491
Melbourne           0.036753
Albury              0.034992
Bendigo             0.034992
Ballarat            0.034992
MountGinini         0.034992
Wollongong          0.034992
Penrith             0.034980
Tuggeranong         0.034980
Newcastle           0.034980
CoffsHarbour        0.034635
Dartmoor            0.034635
Watsonia            0.034635
Portland            0.034635
Mildura             0.034635
Cobar               0.034635
MelbourneAirport    0.034635
Sale                0.034635
Moree               0.034635
Richmond            0.034635
BadgerysCreek       0.034635
Williamtown         0.034635
WaggaWagga          0.034635
SydneyAirport       0.034635
NorfolkIsland       0.034635
NorahHead           0.034578
Brisbane            0.033035
Nhil                0.018164
Name: Location, dtype: float64
W      0.081379
S      0.071653
N      0.071618
WSW    0.065691
SSW    0.062778
WNW    0.062514
SW     0.061236
SSE    0.058600
E      0.056102
ENE    0.053259
NE     0.052074
NW     0.051210
SE     0.050543
NNE    0.050393
ESE    0.041392
NNW    0.037260
Name: WindGustDir, dtype: float64
N      0.087584
W      0.071860
SW     0.071400
NW     0.062364
WNW    0.059855
SSW    0.059371
WSW    0.058151
S      0.056321
NNE    0.055999
SE     0.050773
SSE    0.050474
E      0.049277
NE     0.046997
NNW    0.046606
ENE    0.045432
ESE    0.037927
Name: WindDir9am, dtype: float64
S      0.076522
W      0.075636
N      0.067221
SE     0.066853
NE     0.066047
WNW    0.064747
WSW    0.061167
SSE    0.059924
E      0.058105
SSW    0.057829
SW     0.057541
NW     0.056931
ESE    0.049967
NNE    0.049887
ENE    0.049737
NNW    0.047700
Name: WindDir3pm, dtype: float64
No     0.750717
Yes    0.222326
Name: RainToday, dtype: float64
No     0.750717
Yes    0.222349
Name: RainTomorrow, dtype: float64
</pre>

```python
for var in categorical:
    
    print(var, ' contains ', len(df[var].unique()), ' labels')
```

<pre>
Date  contains  3436  labels
Location  contains  29  labels
WindGustDir  contains  17  labels
WindDir9am  contains  17  labels
WindDir3pm  contains  17  labels
RainToday  contains  3  labels
RainTomorrow  contains  3  labels
</pre>
우리는 전처리가 필요한 날짜 변수가 있다는 것을 알 수 있습니다. 저는 다음 섹션에서 전처리를 할 것입니다.



다른 모든 변수에는 상대적으로 적은 수의 변수가 포함되어 있습니다.


날짜 변수의 피쳐 엔지니어링



```python
df['Date'].dtypes
```

<pre>
dtype('O')
</pre>
날짜 변수의 데이터 유형이 개체임을 알 수 있습니다. 현재 객체로 코딩된 날짜를 datetime 형식으로 구문 분석하겠습니다.



```python
df['Date'] = pd.to_datetime(df['Date'])
```


```python
df['Year'] = df['Date'].dt.year

df['Year'].head()
```

<pre>
0    2008
1    2008
2    2008
3    2008
4    2008
Name: Year, dtype: int64
</pre>

```python
df['Month'] = df['Date'].dt.month

df['Month'].head()
```

<pre>
0    12
1    12
2    12
3    12
4    12
Name: Month, dtype: int64
</pre>

```python
df['Day'] = df['Date'].dt.day

df['Day'].head()
```

<pre>
0    1
1    2
2    3
3    4
4    5
Name: Day, dtype: int64
</pre>

```python
df.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 86877 entries, 0 to 86876
Data columns (total 26 columns):
 #   Column         Non-Null Count  Dtype         
---  ------         --------------  -----         
 0   Date           86877 non-null  datetime64[ns]
 1   Location       86877 non-null  object        
 2   MinTemp        85681 non-null  float64       
 3   MaxTemp        85865 non-null  float64       
 4   Rainfall       84535 non-null  float64       
 5   Evaporation    47217 non-null  float64       
 6   Sunshine       40297 non-null  float64       
 7   WindGustDir    80596 non-null  object        
 8   WindGustSpeed  80604 non-null  float64       
 9   WindDir9am     79092 non-null  object        
 10  WindDir3pm     83907 non-null  object        
 11  WindSpeed9am   85429 non-null  float64       
 12  WindSpeed3pm   84810 non-null  float64       
 13  Humidity9am    84811 non-null  float64       
 14  Humidity3pm    84390 non-null  float64       
 15  Pressure9am    76221 non-null  float64       
 16  Pressure3pm    76268 non-null  float64       
 17  Cloud9am       53399 non-null  float64       
 18  Cloud3pm       52375 non-null  float64       
 19  Temp9am        85314 non-null  float64       
 20  Temp3pm        84837 non-null  float64       
 21  RainToday      84535 non-null  object        
 22  RainTomorrow   84537 non-null  object        
 23  Year           86877 non-null  int64         
 24  Month          86877 non-null  int64         
 25  Day            86877 non-null  int64         
dtypes: datetime64[ns](1), float64(16), int64(3), object(6)
memory usage: 17.2+ MB
</pre>

```python
df.drop('Date', axis=1, inplace = True)
```


```python
df.head()
```

<pre>
  Location  MinTemp  MaxTemp  Rainfall  Evaporation  Sunshine WindGustDir  \
0   Albury     13.4     22.9       0.6          NaN       NaN           W   
1   Albury      7.4     25.1       0.0          NaN       NaN         WNW   
2   Albury     12.9     25.7       0.0          NaN       NaN         WSW   
3   Albury      9.2     28.0       0.0          NaN       NaN          NE   
4   Albury     17.5     32.3       1.0          NaN       NaN           W   

   WindGustSpeed WindDir9am WindDir3pm  ...  Pressure3pm  Cloud9am  Cloud3pm  \
0           44.0          W        WNW  ...       1007.1       8.0       NaN   
1           44.0        NNW        WSW  ...       1007.8       NaN       NaN   
2           46.0          W        WSW  ...       1008.7       NaN       2.0   
3           24.0         SE          E  ...       1012.8       NaN       NaN   
4           41.0        ENE         NW  ...       1006.0       7.0       8.0   

   Temp9am  Temp3pm  RainToday  RainTomorrow  Year  Month  Day  
0     16.9     21.8         No            No  2008     12    1  
1     17.2     24.3         No            No  2008     12    2  
2     21.0     23.2         No            No  2008     12    3  
3     18.1     26.5         No            No  2008     12    4  
4     17.8     29.7         No            No  2008     12    5  

[5 rows x 25 columns]
</pre>
데이터 집합에서 날짜 변수가 제거된 것을 확인할 수 있습니다.



```python
categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)
```

<pre>
There are 6 categorical variables

The categorical variables are : ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
</pre>
우리는 데이터 세트에 6개의 범주형 변수가 있다는 것을 알 수 있습니다. 날짜 변수가 제거되었습니다. 먼저 범주형 변수의 결측값을 확인하겠습니다.



```python
df[categorical].isnull().sum()
```

<pre>
Location           0
WindGustDir     6281
WindDir9am      7785
WindDir3pm      2970
RainToday       2342
RainTomorrow    2340
dtype: int64
</pre>
WindGustDir, WindDir9am, WindDir3pm, RainToday 변수에 결측값이 포함되어 있음을 알 수 있습니다. 저는 이 변수들을 하나씩 탐색할 것입니다.


위치 변수 탐색



```python
print('Location contains', len(df.Location.unique()), 'labels')
```

<pre>
Location contains 29 labels
</pre>

```python
df.Location.unique()
```

<pre>
array(['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree',
       'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond',
       'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown',
       'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat',
       'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura',
       'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane'],
      dtype=object)
</pre>

```python
df.Location.value_counts()
```

<pre>
Canberra            3436
Sydney              3344
Melbourne           3193
Albury              3040
Bendigo             3040
Ballarat            3040
MountGinini         3040
Wollongong          3040
Penrith             3039
Tuggeranong         3039
Newcastle           3039
CoffsHarbour        3009
Dartmoor            3009
Watsonia            3009
Portland            3009
Mildura             3009
Cobar               3009
MelbourneAirport    3009
Sale                3009
Moree               3009
Richmond            3009
BadgerysCreek       3009
Williamtown         3009
WaggaWagga          3009
SydneyAirport       3009
NorfolkIsland       3009
NorahHead           3004
Brisbane            2870
Nhil                1578
Name: Location, dtype: int64
</pre>

```python
pd.get_dummies(df.Location, drop_first=True).head()
```

<pre>
   BadgerysCreek  Ballarat  Bendigo  Brisbane  Canberra  Cobar  CoffsHarbour  \
0              0         0        0         0         0      0             0   
1              0         0        0         0         0      0             0   
2              0         0        0         0         0      0             0   
3              0         0        0         0         0      0             0   
4              0         0        0         0         0      0             0   

   Dartmoor  Melbourne  MelbourneAirport  ...  Portland  Richmond  Sale  \
0         0          0                 0  ...         0         0     0   
1         0          0                 0  ...         0         0     0   
2         0          0                 0  ...         0         0     0   
3         0          0                 0  ...         0         0     0   
4         0          0                 0  ...         0         0     0   

   Sydney  SydneyAirport  Tuggeranong  WaggaWagga  Watsonia  Williamtown  \
0       0              0            0           0         0            0   
1       0              0            0           0         0            0   
2       0              0            0           0         0            0   
3       0              0            0           0         0            0   
4       0              0            0           0         0            0   

   Wollongong  
0           0  
1           0  
2           0  
3           0  
4           0  

[5 rows x 28 columns]
</pre>

```python
print('WindGustDir contains', len(df['WindGustDir'].unique()), 'labels')
```

<pre>
WindGustDir contains 17 labels
</pre>

```python
df['WindGustDir'].unique()
```

<pre>
array(['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', nan, 'ENE',
       'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'], dtype=object)
</pre>

```python
df.WindGustDir.value_counts()
```

<pre>
W      7070
S      6225
N      6222
WSW    5707
SSW    5454
WNW    5431
SW     5320
SSE    5091
E      4874
ENE    4627
NE     4524
NW     4449
SE     4391
NNE    4378
ESE    3596
NNW    3237
Name: WindGustDir, dtype: int64
</pre>

```python
pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).head()
```

<pre>
   ENE  ESE  N  NE  NNE  NNW  NW  S  SE  SSE  SSW  SW  W  WNW  WSW  NaN
0    0    0  0   0    0    0   0  0   0    0    0   0  1    0    0    0
1    0    0  0   0    0    0   0  0   0    0    0   0  0    1    0    0
2    0    0  0   0    0    0   0  0   0    0    0   0  0    0    1    0
3    0    0  0   1    0    0   0  0   0    0    0   0  0    0    0    0
4    0    0  0   0    0    0   0  0   0    0    0   0  1    0    0    0
</pre>

```python
pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).sum(axis=0)
```

<pre>
ENE    4627
ESE    3596
N      6222
NE     4524
NNE    4378
NNW    3237
NW     4449
S      6225
SE     4391
SSE    5091
SSW    5454
SW     5320
W      7070
WNW    5431
WSW    5707
NaN    6281
dtype: int64
</pre>
우리는 바람이 불고 있는 9330명이 있는 가치가 있다고 볼 수 있습니다.


WindDir9am 변수 탐색



```python
print('WindDir9am contains', len(df['WindDir9am'].unique()), 'labels')
```

<pre>
WindDir9am contains 17 labels
</pre>

```python
df['WindDir9am'].unique()
```

<pre>
array(['W', 'NNW', 'SE', 'ENE', 'SW', 'SSE', 'S', 'NE', nan, 'SSW', 'N',
       'WSW', 'ESE', 'E', 'NW', 'WNW', 'NNE'], dtype=object)
</pre>

```python
df['WindDir9am'].value_counts()
```

<pre>
N      7609
W      6243
SW     6203
NW     5418
WNW    5200
SSW    5158
WSW    5052
S      4893
NNE    4865
SE     4411
SSE    4385
E      4281
NE     4083
NNW    4049
ENE    3947
ESE    3295
Name: WindDir9am, dtype: int64
</pre>

```python
pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).head()
```

<pre>
   ENE  ESE  N  NE  NNE  NNW  NW  S  SE  SSE  SSW  SW  W  WNW  WSW  NaN
0    0    0  0   0    0    0   0  0   0    0    0   0  1    0    0    0
1    0    0  0   0    0    1   0  0   0    0    0   0  0    0    0    0
2    0    0  0   0    0    0   0  0   0    0    0   0  1    0    0    0
3    0    0  0   0    0    0   0  0   1    0    0   0  0    0    0    0
4    1    0  0   0    0    0   0  0   0    0    0   0  0    0    0    0
</pre>

```python
pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).sum(axis=0)
```

<pre>
ENE    3947
ESE    3295
N      7609
NE     4083
NNE    4865
NNW    4049
NW     5418
S      4893
SE     4411
SSE    4385
SSW    5158
SW     6203
W      6243
WNW    5200
WSW    5052
NaN    7785
dtype: int64
</pre>
WindDir9am 변수에 결측값이 10013개 있음을 알 수 있습니다.




WindDir3pm 변수 탐색



```python
print('WindDir3pm contains', len(df['WindDir3pm'].unique()), 'labels')
```

<pre>
WindDir3pm contains 17 labels
</pre>

```python
df['WindDir3pm'].unique()
```

<pre>
array(['WNW', 'WSW', 'E', 'NW', 'W', 'SSE', 'ESE', 'ENE', 'NNW', 'SSW',
       'SW', 'SE', 'N', 'S', 'NNE', nan, 'NE'], dtype=object)
</pre>

```python
df['WindDir3pm'].value_counts()
```

<pre>
S      6648
W      6571
N      5840
SE     5808
NE     5738
WNW    5625
WSW    5314
SSE    5206
E      5048
SSW    5024
SW     4999
NW     4946
ESE    4341
NNE    4334
ENE    4321
NNW    4144
Name: WindDir3pm, dtype: int64
</pre>

```python
pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).head()
```

<pre>
   ENE  ESE  N  NE  NNE  NNW  NW  S  SE  SSE  SSW  SW  W  WNW  WSW  NaN
0    0    0  0   0    0    0   0  0   0    0    0   0  0    1    0    0
1    0    0  0   0    0    0   0  0   0    0    0   0  0    0    1    0
2    0    0  0   0    0    0   0  0   0    0    0   0  0    0    1    0
3    0    0  0   0    0    0   0  0   0    0    0   0  0    0    0    0
4    0    0  0   0    0    0   1  0   0    0    0   0  0    0    0    0
</pre>

```python
pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).sum(axis=0)
```

<pre>
ENE    4321
ESE    4341
N      5840
NE     5738
NNE    4334
NNW    4144
NW     4946
S      6648
SE     5808
SSE    5206
SSW    5024
SW     4999
W      6571
WNW    5625
WSW    5314
NaN    2970
dtype: int64
</pre>
WindDir3pm 변수에는 3778개의 결측값이 있습니다.



```python
print('RainToday contains', len(df['RainToday'].unique()), 'labels')
```

<pre>
RainToday contains 3 labels
</pre>

```python
df['RainToday'].unique()
```

<pre>
array(['No', 'Yes', nan], dtype=object)
</pre>

```python
df.RainToday.value_counts()
```

<pre>
No     65220
Yes    19315
Name: RainToday, dtype: int64
</pre>

```python
pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).head()
```

<pre>
   Yes  NaN
0    0    0
1    0    0
2    0    0
3    0    0
4    0    0
</pre>

```python
pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).sum(axis=0)
```

<pre>
Yes    19315
NaN     2342
dtype: int64
</pre>
RainToday 변수에는 1406개의 결측값이 있습니다.


수치 변수 탐색



```python
numerical = [var for var in df.columns if df[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)
```

<pre>
There are 19 numerical variables

The numerical variables are : ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'Year', 'Month', 'Day']
</pre>

```python
df[numerical].head()
```

<pre>
   MinTemp  MaxTemp  Rainfall  Evaporation  Sunshine  WindGustSpeed  \
0     13.4     22.9       0.6          NaN       NaN           44.0   
1      7.4     25.1       0.0          NaN       NaN           44.0   
2     12.9     25.7       0.0          NaN       NaN           46.0   
3      9.2     28.0       0.0          NaN       NaN           24.0   
4     17.5     32.3       1.0          NaN       NaN           41.0   

   WindSpeed9am  WindSpeed3pm  Humidity9am  Humidity3pm  Pressure9am  \
0          20.0          24.0         71.0         22.0       1007.7   
1           4.0          22.0         44.0         25.0       1010.6   
2          19.0          26.0         38.0         30.0       1007.6   
3          11.0           9.0         45.0         16.0       1017.6   
4           7.0          20.0         82.0         33.0       1010.8   

   Pressure3pm  Cloud9am  Cloud3pm  Temp9am  Temp3pm  Year  Month  Day  
0       1007.1       8.0       NaN     16.9     21.8  2008     12    1  
1       1007.8       NaN       NaN     17.2     24.3  2008     12    2  
2       1008.7       NaN       2.0     21.0     23.2  2008     12    3  
3       1012.8       NaN       NaN     18.1     26.5  2008     12    4  
4       1006.0       7.0       8.0     17.8     29.7  2008     12    5  
</pre>
수치 변수 요약  

16개의 숫자 변수가 있습니다.

이것들은 MinTemp, MaxTemp, 강우량, 증발, 햇빛, 풍속, 풍속 9am, 풍속 3pm, 습도 9am, 습도 3pm, 압력 9am, 구름 3pm, 구름 3pm, 온도 9am 및 온도 3pm에 의해 제공됩니다.

모든 숫자 변수는 연속형입니다.


수치 변수 내의 문제 탐색

이제 수치 변수를 살펴보겠습니다.





숫자 변수의 결측값



```python
df[numerical].isnull().sum()
```

<pre>
MinTemp           1196
MaxTemp           1012
Rainfall          2342
Evaporation      39660
Sunshine         46580
WindGustSpeed     6273
WindSpeed9am      1448
WindSpeed3pm      2067
Humidity9am       2066
Humidity3pm       2487
Pressure9am      10656
Pressure3pm      10609
Cloud9am         33478
Cloud3pm         34502
Temp9am           1563
Temp3pm           2040
Year                 0
Month                0
Day                  0
dtype: int64
</pre>
16개의 수치 변수에 결측값이 모두 포함되어 있음을 알 수 있습니다.


숫자 변수의 특이치



```python
print(round(df[numerical].describe()),2)
```

<pre>
       MinTemp  MaxTemp  Rainfall  Evaporation  Sunshine  WindGustSpeed  \
count  85681.0  85865.0   84535.0      47217.0   40297.0        80604.0   
mean      11.0     22.0       2.0          5.0       7.0           40.0   
std        6.0      7.0       8.0          4.0       4.0           14.0   
min       -8.0     -5.0       0.0          0.0       0.0            7.0   
25%        7.0     17.0       0.0          2.0       4.0           30.0   
50%       11.0     22.0       0.0          4.0       8.0           37.0   
75%       16.0     27.0       1.0          7.0      10.0           48.0   
max       32.0     47.0     371.0        145.0      14.0          135.0   

       WindSpeed9am  WindSpeed3pm  Humidity9am  Humidity3pm  Pressure9am  \
count       85429.0       84810.0      84811.0      84390.0      76221.0   
mean           14.0          18.0         72.0         53.0       1018.0   
std             9.0           9.0         18.0         21.0          7.0   
min             0.0           0.0          3.0          1.0        980.0   
25%             7.0          11.0         60.0         38.0       1014.0   
50%            13.0          17.0         73.0         53.0       1018.0   
75%            19.0          24.0         86.0         67.0       1023.0   
max           130.0          83.0        100.0        100.0       1041.0   

       Pressure3pm  Cloud9am  Cloud3pm  Temp9am  Temp3pm     Year    Month  \
count      76268.0   53399.0   52375.0  85314.0  84837.0  86877.0  86877.0   
mean        1016.0       5.0       5.0     16.0     21.0   2013.0      6.0   
std            7.0       3.0       3.0      6.0      7.0      3.0      3.0   
min          979.0       0.0       0.0     -7.0     -5.0   2007.0      1.0   
25%         1011.0       1.0       2.0     11.0     16.0   2010.0      3.0   
50%         1016.0       6.0       5.0     16.0     20.0   2013.0      6.0   
75%         1021.0       7.0       7.0     20.0     25.0   2015.0      9.0   
max         1038.0       9.0       8.0     38.0     47.0   2017.0     12.0   

           Day  
count  86877.0  
mean      16.0  
std        9.0  
min        1.0  
25%        8.0  
50%       16.0  
75%       23.0  
max       31.0   2
</pre>
자세히 살펴보면 강우량, 증발량, 풍속 9am 및 풍속 3pm 열에 특이치가 포함되어 있을 수 있습니다.



상자 그림을 그려 위 변수의 특이치를 시각화합니다.



```python
plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')


plt.subplot(2, 2, 2)
fig = df.boxplot(column='Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')


plt.subplot(2, 2, 3)
fig = df.boxplot(column='WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')


plt.subplot(2, 2, 4)
fig = df.boxplot(column='WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')
```

<pre>
Text(0, 0.5, 'WindSpeed3pm')
</pre>
<pre>
<Figure size 1500x1000 with 4 Axes>
</pre>
위의 상자 그림은 이러한 변수에 특이치가 많다는 것을 확인합니다.


변수 분포 확인  

이제 히스토그램을 그려 분포가 정규 분포인지 치우쳐 있는지 확인합니다. 변수가 정규 분포를 따르는 경우 극단값 분석을 수행하고, 그렇지 않은 경우 치우친 경우 IQR(양자 간 범위)을 찾습니다.



```python
plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.Rainfall.hist(bins=10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 2)
fig = df.Evaporation.hist(bins=10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 3)
fig = df.WindSpeed9am.hist(bins=10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 4)
fig = df.WindSpeed3pm.hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('RainTomorrow')
```

<pre>
Text(0, 0.5, 'RainTomorrow')
</pre>
<pre>
<Figure size 1500x1000 with 4 Axes>
</pre>
네 가지 변수가 모두 치우쳐 있음을 알 수 있습니다. 따라서 특이치를 찾기 위해 분위수 범위를 사용합니다.



```python
IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)
Lower_fence = df.Rainfall.quantile(0.25) - (IQR * 3)
Upper_fence = df.Rainfall.quantile(0.75) + (IQR * 3)
print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

<pre>
Rainfall outliers are values < -2.4000000000000004 or > 3.2
</pre>
강우량의 경우 최소값과 최대값은 0.0과 371.0입니다. 따라서 특이치는 3.2보다 큰 값입니다.



```python
IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)
Lower_fence = df.Evaporation.quantile(0.25) - (IQR * 3)
Upper_fence = df.Evaporation.quantile(0.75) + (IQR * 3)
print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

<pre>
Evaporation outliers are values < -10.8 or > 20.0
</pre>
강우량의 경우 최소값과 최대값은 0.0과 371.0입니다. 따라서 특이치는 3.2보다 큰 값입니다.



```python
IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)
Lower_fence = df.Evaporation.quantile(0.25) - (IQR * 3)
Upper_fence = df.Evaporation.quantile(0.75) + (IQR * 3)
print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

<pre>
Evaporation outliers are values < -10.8 or > 20.0
</pre>
증발의 경우 최소값과 최대값은 0.0과 145.0입니다. 따라서 특이치는 21.8보다 큰 값입니다.



```python
IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)
Lower_fence = df.WindSpeed9am.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed9am.quantile(0.75) + (IQR * 3)
print('WindSpeed9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

<pre>
WindSpeed9am outliers are values < -29.0 or > 55.0
</pre>
풍속 9am의 경우 최소값과 최대값은 0.0과 130.0입니다. 따라서 특이치는 55.0보다 큰 값입니다.



```python
IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)
Lower_fence = df.WindSpeed3pm.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed3pm.quantile(0.75) + (IQR * 3)
print('WindSpeed3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

<pre>
WindSpeed3pm outliers are values < -28.0 or > 63.0
</pre>
풍속 3pm의 경우 최소값과 최대값은 0.0과 87.0입니다. 따라서 특이치는 57.0보다 큰 값입니다.


8. 피쳐 벡터 및 대상 변수 선언



```python
X = df.drop(['RainTomorrow'], axis=1)

y = df['RainTomorrow']
```

9. 데이터를 별도의 교육 및 테스트 세트로 분할



```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```


```python
X_train.shape, X_test.shape
```

<pre>
((69501, 24), (17376, 24))
</pre>
10. Feature Engineering


기능 엔지니어링은 원시 데이터를 유용한 기능으로 변환하여 모델을 더 잘 이해하고 예측력을 높이는 데 도움이 됩니다. 저는 다양한 유형의 변수에 대해 피쳐 엔지니어링을 수행할 것입니다.



먼저 범주형 변수와 숫자형 변수를 다시 별도로 표시하겠습니다.



```python
X_train.dtypes
```

<pre>
Location          object
MinTemp          float64
MaxTemp          float64
Rainfall         float64
Evaporation      float64
Sunshine         float64
WindGustDir       object
WindGustSpeed    float64
WindDir9am        object
WindDir3pm        object
WindSpeed9am     float64
WindSpeed3pm     float64
Humidity9am      float64
Humidity3pm      float64
Pressure9am      float64
Pressure3pm      float64
Cloud9am         float64
Cloud3pm         float64
Temp9am          float64
Temp3pm          float64
RainToday         object
Year               int64
Month              int64
Day                int64
dtype: object
</pre>

```python
categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical
```

<pre>
['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
</pre>

```python
numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical
```

<pre>
['MinTemp',
 'MaxTemp',
 'Rainfall',
 'Evaporation',
 'Sunshine',
 'WindGustSpeed',
 'WindSpeed9am',
 'WindSpeed3pm',
 'Humidity9am',
 'Humidity3pm',
 'Pressure9am',
 'Pressure3pm',
 'Cloud9am',
 'Cloud3pm',
 'Temp9am',
 'Temp3pm',
 'Year',
 'Month',
 'Day']
</pre>
숫자 변수의 결측값 엔지니어링



```python
X_train[numerical].isnull().sum()
```

<pre>
MinTemp            960
MaxTemp            811
Rainfall          1878
Evaporation      31687
Sunshine         37249
WindGustSpeed     5019
WindSpeed9am      1137
WindSpeed3pm      1672
Humidity9am       1633
Humidity3pm       2009
Pressure9am       8536
Pressure3pm       8496
Cloud9am         26837
Cloud3pm         27697
Temp9am           1231
Temp3pm           1654
Year                 0
Month                0
Day                  0
dtype: int64
</pre>

```python
X_test[numerical].isnull().sum()
```

<pre>
MinTemp           236
MaxTemp           201
Rainfall          464
Evaporation      7973
Sunshine         9331
WindGustSpeed    1254
WindSpeed9am      311
WindSpeed3pm      395
Humidity9am       433
Humidity3pm       478
Pressure9am      2120
Pressure3pm      2113
Cloud9am         6641
Cloud3pm         6805
Temp9am           332
Temp3pm           386
Year                0
Month               0
Day                 0
dtype: int64
</pre>

```python
for col in numerical:
    if X_train[col].isnull().mean()>0:
        print(col, round(X_train[col].isnull().mean(),4))
```

<pre>
MinTemp 0.0138
MaxTemp 0.0117
Rainfall 0.027
Evaporation 0.4559
Sunshine 0.5359
WindGustSpeed 0.0722
WindSpeed9am 0.0164
WindSpeed3pm 0.0241
Humidity9am 0.0235
Humidity3pm 0.0289
Pressure9am 0.1228
Pressure3pm 0.1222
Cloud9am 0.3861
Cloud3pm 0.3985
Temp9am 0.0177
Temp3pm 0.0238
</pre>
추정

데이터가 랜덤으로 완전히 누락되었다고 가정합니다(MCAR). 결측값을 귀속시키는 데 사용할 수 있는 두 가지 방법이 있습니다. 하나는 평균 또는 중위수 귀책이고 다른 하나는 랜덤 표본 귀책입니다. 데이터 집합에 특이치가 있을 경우 중위수 귀책을 사용해야 합니다. 중위수 귀인은 특이치에 강하므로 중위수 귀인을 사용합니다.



결측값을 데이터의 적절한 통계적 측도(이 경우 중위수)로 귀속시킵니다. 귀속은 교육 세트에 대해 수행된 다음 테스트 세트에 전파되어야 합니다. 즉, 트레인과 테스트 세트 모두에서 결측값을 채우기 위해 사용되는 통계적 측정값은 트레인 세트에서만 추출되어야 합니다. 이는 과적합을 방지하기 위한 것입니다.



```python
for df1 in [X_train, X_test]:
    for col in numerical:
        col_median=X_train[col].median()
        df1[col].fillna(col_median, inplace=True)
```


```python
X_train[numerical].isnull().sum()
```

<pre>
MinTemp          0
MaxTemp          0
Rainfall         0
Evaporation      0
Sunshine         0
WindGustSpeed    0
WindSpeed9am     0
WindSpeed3pm     0
Humidity9am      0
Humidity3pm      0
Pressure9am      0
Pressure3pm      0
Cloud9am         0
Cloud3pm         0
Temp9am          0
Temp3pm          0
Year             0
Month            0
Day              0
dtype: int64
</pre>

```python
X_test[numerical].isnull().sum()
```

<pre>
MinTemp          0
MaxTemp          0
Rainfall         0
Evaporation      0
Sunshine         0
WindGustSpeed    0
WindSpeed9am     0
WindSpeed3pm     0
Humidity9am      0
Humidity3pm      0
Pressure9am      0
Pressure3pm      0
Cloud9am         0
Cloud3pm         0
Temp9am          0
Temp3pm          0
Year             0
Month            0
Day              0
dtype: int64
</pre>
이제 훈련 및 테스트 세트의 숫자 열에 결측값이 없음을 알 수 있습니다.


범주형 변수의 결측값 엔지니어링



```python
X_train[categorical].isnull().mean()
```

<pre>
Location       0.000000
WindGustDir    0.072301
WindDir9am     0.089207
WindDir3pm     0.034359
RainToday      0.027021
dtype: float64
</pre>

```python
for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, (X_train[col].isnull().mean()))
```

<pre>
WindGustDir 0.07230111796952562
WindDir9am 0.08920734953453907
WindDir3pm 0.03435921785298053
RainToday 0.027021193939655543
</pre>

```python
for df2 in [X_train, X_test]:
    df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)
    df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
    df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)
    df2['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)
```


```python
X_train[categorical].isnull().sum()
```

<pre>
Location       0
WindGustDir    0
WindDir9am     0
WindDir3pm     0
RainToday      0
dtype: int64
</pre>

```python
X_test[categorical].isnull().sum()
```

<pre>
Location       0
WindGustDir    0
WindDir9am     0
WindDir3pm     0
RainToday      0
dtype: int64
</pre>

```python
X_train.isnull().sum()
```

<pre>
Location         0
MinTemp          0
MaxTemp          0
Rainfall         0
Evaporation      0
Sunshine         0
WindGustDir      0
WindGustSpeed    0
WindDir9am       0
WindDir3pm       0
WindSpeed9am     0
WindSpeed3pm     0
Humidity9am      0
Humidity3pm      0
Pressure9am      0
Pressure3pm      0
Cloud9am         0
Cloud3pm         0
Temp9am          0
Temp3pm          0
RainToday        0
Year             0
Month            0
Day              0
dtype: int64
</pre>

```python
X_test.isnull().sum()
```

<pre>
Location         0
MinTemp          0
MaxTemp          0
Rainfall         0
Evaporation      0
Sunshine         0
WindGustDir      0
WindGustSpeed    0
WindDir9am       0
WindDir3pm       0
WindSpeed9am     0
WindSpeed3pm     0
Humidity9am      0
Humidity3pm      0
Pressure9am      0
Pressure3pm      0
Cloud9am         0
Cloud3pm         0
Temp9am          0
Temp3pm          0
RainToday        0
Year             0
Month            0
Day              0
dtype: int64
</pre>
X_train 및 X_test에서 결측값이 없음을 알 수 있습니다.


숫자 변수의 공학적 특이치

강우량, 증발량, 풍속 9am 및 풍속 3pm 열에 특이치가 포함되어 있는 것을 확인했습니다. 최상위 코드화 방법을 사용하여 최대값을 상한으로 설정하고 위 변수에서 특이치를 제거합니다.



```python
def max_value(df3, variable, top):
    return np.where(df3[variable]>top, top, df3[variable])

for df3 in [X_train, X_test]:
    df3['Rainfall'] = max_value(df3, 'Rainfall', 3.2)
    df3['Evaporation'] = max_value(df3, 'Evaporation', 21.8)
    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)
    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)
```


```python
X_train.Rainfall.max(), X_test.Rainfall.max()
```

<pre>
(3.2, 3.2)
</pre>

```python
X_train.Evaporation.max(), X_test.Evaporation.max()
```

<pre>
(21.8, 21.8)
</pre>

```python
X_train.WindSpeed9am.max(), X_test.WindSpeed9am.max()
```

<pre>
(55.0, 55.0)
</pre>

```python
X_train.WindSpeed3pm.max(), X_test.WindSpeed3pm.max()
```

<pre>
(57.0, 57.0)
</pre>

```python
X_train[numerical].describe()
```

<pre>
            MinTemp       MaxTemp      Rainfall   Evaporation      Sunshine  \
count  69501.000000  69501.000000  69501.000000  69501.000000  69501.000000   
mean      11.256306     22.149077      0.677774      4.624131      7.644891   
std        6.076202      6.847693      1.181797      2.627143      2.657073   
min       -8.500000     -4.800000      0.000000      0.000000      0.000000   
25%        7.000000     17.300000      0.000000      4.000000      8.000000   
50%       11.300000     21.700000      0.000000      4.200000      8.000000   
75%       15.900000     26.500000      0.600000      4.600000      8.000000   
max       31.900000     47.300000      3.200000     21.800000     14.500000   

       WindGustSpeed  WindSpeed9am  WindSpeed3pm   Humidity9am   Humidity3pm  \
count   69501.000000  69501.000000  69501.000000  69501.000000  69501.000000   
mean       39.532352     13.514237     18.187379     71.845254     52.988230   
std        13.827590      9.131182      9.103827     17.650405     20.233695   
min         7.000000      0.000000      0.000000      3.000000      1.000000   
25%        30.000000      7.000000     11.000000     60.000000     39.000000   
50%        37.000000     13.000000     17.000000     73.000000     53.000000   
75%        46.000000     19.000000     24.000000     85.000000     66.000000   
max       135.000000     55.000000     57.000000    100.000000    100.000000   

        Pressure9am   Pressure3pm      Cloud9am      Cloud3pm       Temp9am  \
count  69501.000000  69501.000000  69501.000000  69501.000000  69501.000000   
mean    1018.279151   1015.951965      5.185523      4.841815     15.746578   
std        6.677312      6.564234      2.352161      2.087499      5.992743   
min      980.500000    979.000000      0.000000      0.000000     -7.200000   
25%     1014.400000   1012.000000      4.000000      4.000000     11.500000   
50%     1018.400000   1016.000000      6.000000      5.000000     15.800000   
75%     1022.300000   1020.000000      7.000000      6.000000     20.100000   
max     1040.600000   1037.900000      9.000000      8.000000     37.600000   

            Temp3pm          Year         Month           Day  
count  69501.000000  69501.000000  69501.000000  69501.000000  
mean      20.678746   2012.729802      6.391994     15.722004  
std        6.616623      2.535347      3.424067      8.797447  
min       -5.400000   2007.000000      1.000000      1.000000  
25%       16.100000   2010.000000      3.000000      8.000000  
50%       20.200000   2013.000000      6.000000     16.000000  
75%       24.800000   2015.000000      9.000000     23.000000  
max       46.700000   2017.000000     12.000000     31.000000  
</pre>
이제 강우량, 증발량, 풍속 9am 및 풍속 3pm 열의 특이치가 상한선임을 알 수 있습니다.


범주형 변수 인코딩



```python
categorical
```

<pre>
['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
</pre>

```python
X_train[categorical].head()
```

<pre>
          Location WindGustDir WindDir9am WindDir3pm RainToday
61963         Sale           W          W         SW        No
40957  Williamtown           S         SE        SSE       Yes
53176  MountGinini           W          N         NW        No
37872   WaggaWagga           W          N          W        No
274         Albury         WNW        WNW          W       Yes
</pre>

```python
!pip install --upgrade category_encoders
import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['RainToday'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)
```

<pre>
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting category_encoders
  Downloading category_encoders-2.6.0-py2.py3-none-any.whl (81 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m81.2/81.2 kB[0m [31m2.7 MB/s[0m eta [36m0:00:00[0m
[?25hRequirement already satisfied: numpy>=1.14.0 in /usr/local/lib/python3.9/dist-packages (from category_encoders) (1.22.4)
Requirement already satisfied: pandas>=1.0.5 in /usr/local/lib/python3.9/dist-packages (from category_encoders) (1.5.3)
Requirement already satisfied: scikit-learn>=0.20.0 in /usr/local/lib/python3.9/dist-packages (from category_encoders) (1.2.2)
Requirement already satisfied: scipy>=1.0.0 in /usr/local/lib/python3.9/dist-packages (from category_encoders) (1.10.1)
Requirement already satisfied: patsy>=0.5.1 in /usr/local/lib/python3.9/dist-packages (from category_encoders) (0.5.3)
Requirement already satisfied: statsmodels>=0.9.0 in /usr/local/lib/python3.9/dist-packages (from category_encoders) (0.13.5)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.9/dist-packages (from pandas>=1.0.5->category_encoders) (2022.7.1)
Requirement already satisfied: python-dateutil>=2.8.1 in /usr/local/lib/python3.9/dist-packages (from pandas>=1.0.5->category_encoders) (2.8.2)
Requirement already satisfied: six in /usr/local/lib/python3.9/dist-packages (from patsy>=0.5.1->category_encoders) (1.16.0)
Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.9/dist-packages (from scikit-learn>=0.20.0->category_encoders) (3.1.0)
Requirement already satisfied: joblib>=1.1.1 in /usr/local/lib/python3.9/dist-packages (from scikit-learn>=0.20.0->category_encoders) (1.2.0)
Requirement already satisfied: packaging>=21.3 in /usr/local/lib/python3.9/dist-packages (from statsmodels>=0.9.0->category_encoders) (23.0)
Installing collected packages: category_encoders
Successfully installed category_encoders-2.6.0
</pre>

```python
X_train.head()
```

<pre>
          Location  MinTemp  MaxTemp  Rainfall  Evaporation  Sunshine  \
61963         Sale     10.6     19.6       0.0         10.1       7.5   
40957  Williamtown     18.9     24.3       3.2         21.8       8.0   
53176  MountGinini     12.1     23.1       0.0          4.2       8.0   
37872   WaggaWagga      6.9     28.3       0.0          7.6      11.1   
274         Albury      5.1     14.2       3.0          4.2       8.0   

      WindGustDir  WindGustSpeed WindDir9am WindDir3pm  ...  Pressure3pm  \
61963           W           54.0          W         SW  ...       1018.2   
40957           S           24.0         SE        SSE  ...       1015.7   
53176           W           30.0          N         NW  ...       1016.0   
37872           W           35.0          N          W  ...       1014.0   
274           WNW           24.0        WNW          W  ...       1021.7   

       Cloud9am  Cloud3pm  Temp9am  Temp3pm  RainToday_0  RainToday_1  Year  \
61963       4.0       6.0     14.2     18.1            0            1  2011   
40957       8.0       8.0     20.3     20.4            1            0  2013   
53176       6.0       5.0     17.0     22.0            0            1  2012   
37872       1.0       1.0     19.7     27.9            0            1  2012   
274         8.0       1.0      9.7     12.5            1            0  2009   

       Month  Day  
61963      2   21  
40957      1   20  
53176      1   19  
37872     10    5  
274        9    1  

[5 rows x 25 columns]
</pre>
RainToday_0 및 RainToday_1 변수 두 개가 RainToday 변수에서 생성되었음을 알 수 있습니다.



이제 X_train 교육 세트를 생성하겠습니다.



```python
X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_train.Location), 
                     pd.get_dummies(X_train.WindGustDir),
                     pd.get_dummies(X_train.WindDir9am),
                     pd.get_dummies(X_train.WindDir3pm)], axis=1)
```


```python
X_train.head()
```

<pre>
       MinTemp  MaxTemp  Rainfall  Evaporation  Sunshine  WindGustSpeed  \
61963     10.6     19.6       0.0         10.1       7.5           54.0   
40957     18.9     24.3       3.2         21.8       8.0           24.0   
53176     12.1     23.1       0.0          4.2       8.0           30.0   
37872      6.9     28.3       0.0          7.6      11.1           35.0   
274        5.1     14.2       3.0          4.2       8.0           24.0   

       WindSpeed9am  WindSpeed3pm  Humidity9am  Humidity3pm  ...  NNW  NW  S  \
61963          31.0          31.0         64.0         49.0  ...    0   0  0   
40957          11.0           9.0         93.0         95.0  ...    0   0  0   
53176           6.0          11.0         60.0         54.0  ...    0   1  0   
37872           6.0          24.0         34.0         19.0  ...    0   0  0   
274             7.0          15.0         96.0         58.0  ...    0   0  0   

       SE  SSE  SSW  SW  W  WNW  WSW  
61963   0    0    0   1  0    0    0  
40957   0    1    0   0  0    0    0  
53176   0    0    0   0  0    0    0  
37872   0    0    0   0  1    0    0  
274     0    0    0   0  1    0    0  

[5 rows x 98 columns]
</pre>
마찬가지로 X_test testing set도 만들겠습니다.



```python
X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_test.Location), 
                     pd.get_dummies(X_test.WindGustDir),
                     pd.get_dummies(X_test.WindDir9am),
                     pd.get_dummies(X_test.WindDir3pm)], axis=1)
```


```python
X_test.head()
```

<pre>
       MinTemp  MaxTemp  Rainfall  Evaporation  Sunshine  WindGustSpeed  \
19573     20.5     27.5       0.0          4.2       8.0           35.0   
27987     12.2     22.8       2.4          2.6       8.0           24.0   
62734      2.6     16.1       0.2          0.2       8.2           28.0   
68371     15.6     24.0       0.0          5.4       9.2           59.0   
54837     -2.2      1.5       3.2          4.2       8.0           96.0   

       WindSpeed9am  WindSpeed3pm  Humidity9am  Humidity3pm  ...  NNW  NW  S  \
19573          15.0          24.0         78.0         75.0  ...    0   0  0   
27987           0.0           9.0         86.0         44.0  ...    0   0  0   
62734          15.0          19.0         92.0         66.0  ...    0   0  0   
68371          13.0          35.0         60.0         59.0  ...    0   0  0   
54837          26.0          28.0         98.0         98.0  ...    0   0  0   

       SE  SSE  SSW  SW  W  WNW  WSW  
19573   0    0    0   0  0    0    0  
27987   0    0    0   0  0    0    1  
62734   0    0    0   0  1    0    0  
68371   0    0    0   1  0    0    0  
54837   0    0    0   0  0    1    0  

[5 rows x 98 columns]
</pre>
이제 모델 구축을 위한 교육 및 테스트가 준비되었습니다. 그 전에 모든 형상 변수를 동일한 척도에 매핑해야 합니다. 이를 형상 스케일링이라고 합니다. 다음과 같이 하겠습니다.


11. 피쳐 스케일링



```python
X_train.describe()
```

<pre>
            MinTemp       MaxTemp      Rainfall   Evaporation      Sunshine  \
count  69501.000000  69501.000000  69501.000000  69501.000000  69501.000000   
mean      11.256306     22.149077      0.677774      4.624131      7.644891   
std        6.076202      6.847693      1.181797      2.627143      2.657073   
min       -8.500000     -4.800000      0.000000      0.000000      0.000000   
25%        7.000000     17.300000      0.000000      4.000000      8.000000   
50%       11.300000     21.700000      0.000000      4.200000      8.000000   
75%       15.900000     26.500000      0.600000      4.600000      8.000000   
max       31.900000     47.300000      3.200000     21.800000     14.500000   

       WindGustSpeed  WindSpeed9am  WindSpeed3pm   Humidity9am   Humidity3pm  \
count   69501.000000  69501.000000  69501.000000  69501.000000  69501.000000   
mean       39.532352     13.514237     18.187379     71.845254     52.988230   
std        13.827590      9.131182      9.103827     17.650405     20.233695   
min         7.000000      0.000000      0.000000      3.000000      1.000000   
25%        30.000000      7.000000     11.000000     60.000000     39.000000   
50%        37.000000     13.000000     17.000000     73.000000     53.000000   
75%        46.000000     19.000000     24.000000     85.000000     66.000000   
max       135.000000     55.000000     57.000000    100.000000    100.000000   

       ...           NNW            NW             S            SE  \
count  ...  69501.000000  69501.000000  69501.000000  69501.000000   
mean   ...      0.047208      0.056906      0.111034      0.067625   
std    ...      0.212085      0.231664      0.314177      0.251103   
min    ...      0.000000      0.000000      0.000000      0.000000   
25%    ...      0.000000      0.000000      0.000000      0.000000   
50%    ...      0.000000      0.000000      0.000000      0.000000   
75%    ...      0.000000      0.000000      0.000000      0.000000   
max    ...      1.000000      1.000000      1.000000      1.000000   

                SSE           SSW            SW             W           WNW  \
count  69501.000000  69501.000000  69501.000000  69501.000000  69501.000000   
mean       0.060157      0.057812      0.057352      0.075668      0.064460   
std        0.237780      0.233390      0.232515      0.264468      0.245571   
min        0.000000      0.000000      0.000000      0.000000      0.000000   
25%        0.000000      0.000000      0.000000      0.000000      0.000000   
50%        0.000000      0.000000      0.000000      0.000000      0.000000   
75%        0.000000      0.000000      0.000000      0.000000      0.000000   
max        1.000000      1.000000      1.000000      1.000000      1.000000   

                WSW  
count  69501.000000  
mean       0.061208  
std        0.239713  
min        0.000000  
25%        0.000000  
50%        0.000000  
75%        0.000000  
max        1.000000  

[8 rows x 98 columns]
</pre>

```python
cols = X_train.columns
```


```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
```


```python
X_train = pd.DataFrame(X_train, columns=[cols])
```


```python
X_test = pd.DataFrame(X_test, columns=[cols])
```


```python
X_train.describe()
```

<pre>
            MinTemp       MaxTemp      Rainfall   Evaporation      Sunshine  \
count  69501.000000  69501.000000  69501.000000  69501.000000  69501.000000   
mean       0.489017      0.517257      0.211805      0.212116      0.527234   
std        0.150401      0.131434      0.369312      0.120511      0.183246   
min        0.000000      0.000000      0.000000      0.000000      0.000000   
25%        0.383663      0.424184      0.000000      0.183486      0.551724   
50%        0.490099      0.508637      0.000000      0.192661      0.551724   
75%        0.603960      0.600768      0.187500      0.211009      0.551724   
max        1.000000      1.000000      1.000000      1.000000      1.000000   

      WindGustSpeed  WindSpeed9am  WindSpeed3pm   Humidity9am   Humidity3pm  \
count  69501.000000  69501.000000  69501.000000  69501.000000  69501.000000   
mean       0.254159      0.245713      0.319077      0.709745      0.525134   
std        0.108028      0.166021      0.159716      0.181963      0.204381   
min        0.000000      0.000000      0.000000      0.000000      0.000000   
25%        0.179688      0.127273      0.192982      0.587629      0.383838   
50%        0.234375      0.236364      0.298246      0.721649      0.525253   
75%        0.304688      0.345455      0.421053      0.845361      0.656566   
max        1.000000      1.000000      1.000000      1.000000      1.000000   

       ...           NNW            NW             S            SE  \
count  ...  69501.000000  69501.000000  69501.000000  69501.000000   
mean   ...      0.047208      0.056906      0.111034      0.067625   
std    ...      0.212085      0.231664      0.314177      0.251103   
min    ...      0.000000      0.000000      0.000000      0.000000   
25%    ...      0.000000      0.000000      0.000000      0.000000   
50%    ...      0.000000      0.000000      0.000000      0.000000   
75%    ...      0.000000      0.000000      0.000000      0.000000   
max    ...      1.000000      1.000000      1.000000      1.000000   

                SSE           SSW            SW             W           WNW  \
count  69501.000000  69501.000000  69501.000000  69501.000000  69501.000000   
mean       0.060157      0.057812      0.057352      0.075668      0.064460   
std        0.237780      0.233390      0.232515      0.264468      0.245571   
min        0.000000      0.000000      0.000000      0.000000      0.000000   
25%        0.000000      0.000000      0.000000      0.000000      0.000000   
50%        0.000000      0.000000      0.000000      0.000000      0.000000   
75%        0.000000      0.000000      0.000000      0.000000      0.000000   
max        1.000000      1.000000      1.000000      1.000000      1.000000   

                WSW  
count  69501.000000  
mean       0.061208  
std        0.239713  
min        0.000000  
25%        0.000000  
50%        0.000000  
75%        0.000000  
max        1.000000  

[8 rows x 98 columns]
</pre>
이제 X_train 데이터 세트를 로지스틱 회귀 분류기에 입력할 준비가 되었습니다. 다음과 같이 하겠습니다.



```python
df.isnull().sum()
```

<pre>
Location             0
MinTemp           1196
MaxTemp           1012
Rainfall          2342
Evaporation      39660
Sunshine         46580
WindGustDir       6281
WindGustSpeed     6273
WindDir9am        7785
WindDir3pm        2970
WindSpeed9am      1448
WindSpeed3pm      2067
Humidity9am       2066
Humidity3pm       2487
Pressure9am      10656
Pressure3pm      10609
Cloud9am         33478
Cloud3pm         34502
Temp9am           1563
Temp3pm           2040
RainToday         2342
RainTomorrow      2340
Year                 0
Month                0
Day                  0
dtype: int64
</pre>

```python
df.replace([np.inf, -np.inf], np.nan).dropna()
```

<pre>
       Location  MinTemp  MaxTemp  Rainfall  Evaporation  Sunshine  \
6049      Cobar     17.9     35.2       0.0         12.0      12.3   
6050      Cobar     18.4     28.9       0.0         14.8      13.0   
6052      Cobar     19.4     37.6       0.0         10.8      10.6   
6053      Cobar     21.9     38.4       0.0         11.4      12.2   
6054      Cobar     24.2     41.0       0.0         11.2       8.4   
...         ...      ...      ...       ...          ...       ...   
86871  Brisbane      9.1     25.6       0.0          2.6      10.0   
86872  Brisbane     12.6     23.8       0.0          4.0       2.3   
86874  Brisbane     11.7     20.8       0.8          6.8       4.4   
86875  Brisbane     11.6     20.9       0.2          4.0       9.1   
86876  Brisbane     10.9     22.4       0.0          3.8       9.6   

      WindGustDir  WindGustSpeed WindDir9am WindDir3pm  ...  Pressure3pm  \
6049          SSW           48.0        ENE         SW  ...       1004.4   
6050            S           37.0        SSE        SSE  ...       1012.1   
6052          NNE           46.0        NNE        NNW  ...       1009.2   
6053          WNW           31.0        WNW        WSW  ...       1009.1   
6054          WNW           35.0         NW        WNW  ...       1007.4   
...           ...            ...        ...        ...  ...          ...   
86871         ENE           15.0        WSW         NE  ...       1014.7   
86872         NNW           22.0        SSE         NW  ...       1010.5   
86874         WSW           41.0         SW          S  ...       1016.4   
86875         SSW           30.0        SSW         SE  ...       1022.8   
86876           E           20.0         SW        SSE  ...       1023.1   

       Cloud9am  Cloud3pm  Temp9am  Temp3pm  RainToday  RainTomorrow  Year  \
6049        2.0       5.0     26.6     33.4         No            No  2009   
6050        1.0       1.0     20.3     27.0         No            No  2009   
6052        1.0       6.0     28.7     34.9         No            No  2009   
6053        1.0       5.0     29.1     35.6         No            No  2009   
6054        1.0       6.0     33.6     37.6         No            No  2009   
...         ...       ...      ...      ...        ...           ...   ...   
86871       0.0       0.0     14.2     24.9         No            No  2016   
86872       7.0       7.0     16.6     22.3         No           Yes  2016   
86874       7.0       7.0     17.5     18.8         No            No  2016   
86875       1.0       3.0     16.7     20.4         No            No  2016   
86876       4.0       2.0     15.3     22.0         No            No  2016   

       Month  Day  
6049       1    1  
6050       1    2  
6052       1    4  
6053       1    5  
6054       1    6  
...      ...  ...  
86871      8    1  
86872      8    2  
86874      8    4  
86875      8    5  
86876      8    6  

[31893 rows x 25 columns]
</pre>
12. 모델 교육



```python
from sklearn.linear_model import LogisticRegression


# instantiate the model
logreg = LogisticRegression(solver='liblinear', random_state=0)


# fit the model
logreg.fit(X_train, y_train)
```

13. 결과 예측



```python
y_pred_test = logreg.predict(X_test)

y_pred_test
```

predict_proba 방법  

predict_proba 메서드는 이 경우 대상 변수(0 및 1)에 대한 확률을 배열 형식으로 제공합니다.



0은 비가 오지 않을 확률이고 1은 비가 올 확률입니다.



```python
logreg.predict_proba(X_test)[:,0]
```


```python
logreg.predict_proba(X_test)[:,1]
```

14.정확도 점수 확인



```python
from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))
```

여기서 y_test는 참 클래스 레이블이고 y_pred_test는 테스트 세트의 예측 클래스 레이블입니다.


열차 세트와 테스트 세트 정확도 비교  

이제 트레인 세트와 테스트 세트 정확도를 비교하여 과적합 여부를 확인하겠습니다.



```python
y_pred_train = logreg.predict(X_train)

y_pred_train
```


```python
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
```

과적합 및 과소적합 여부 점검



```python
print('Training set score: {:.4f}'.format(logreg.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg.score(X_test, y_test)))
```

교육 세트 정확도 점수는 0.8476인 반면 테스트 세트 정확도는 0.8501입니다. 이 두 값은 상당히 비슷합니다. 따라서 과적합의 문제는 없습니다.


로지스틱 회귀 분석에서는 C = 1의 기본값을 사용합니다. 교육 및 테스트 세트 모두에서 약 85%의 정확도로 우수한 성능을 제공합니다. 그러나 교육 및 테스트 세트의 모델 성능은 매우 유사합니다. 그것은 아마도 부족한 경우일 것입니다.



저는 C를 늘리고 좀 더 유연한 모델을 맞출 것입니다.



```python
logreg100 = LogisticRegression(C=100, solver='liblinear', random_state=0)


# fit the model
logreg100.fit(X_train, y_train)
```


```python
print('Training set score: {:.4f}'.format(logreg100.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg100.score(X_test, y_test)))
```

우리는 C=100이 테스트 세트 정확도를 높이고 교육 세트 정확도를 약간 높인다는 것을 알 수 있습니다. 따라서 더 복잡한 모델이 더 나은 성능을 발휘해야 한다는 결론을 내릴 수 있습니다.



이제 C=0.01을 설정하여 기본값인 C=1보다 정규화된 모델을 사용하면 어떻게 되는지 알아보겠습니다.



```python
logreg001 = LogisticRegression(C=0.01, solver='liblinear', random_state=0)


# fit the model
logreg001.fit(X_train, y_train)
```


```python
print('Training set score: {:.4f}'.format(logreg001.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg001.score(X_test, y_test)))
```

따라서 C=0.01을 설정하여 보다 정규화된 모델을 사용하면 교육 및 테스트 세트 정확도가 기본 매개 변수에 비해 모두 감소합니다.


모델 정확도와 null 정확도 비교

따라서 모형 정확도는 0.8501입니다. 그러나 위의 정확도에 근거하여 우리의 모델이 매우 좋다고 말할 수는 없습니다. 우리는 그것을 null 정확도와 비교해야 합니다. Null 정확도는 항상 가장 빈도가 높은 클래스를 예측하여 얻을 수 있는 정확도입니다.



그래서 우리는 먼저 테스트 세트의 클래스 분포를 확인해야 합니다.



```python
y_test.value_counts()
```

우리는 가장 빈번한 수업의 발생 횟수가 22067회임을 알 수 있습니다. 따라서 22067을 총 발생 횟수로 나누어 null 정확도를 계산할 수 있습니다.



```python
null_accuracy = (22067/(22067+6372))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))
```

우리의 모델 정확도 점수는 0.8501이지만 null 정확도 점수는 0.7759임을 알 수 있습니다. 따라서 로지스틱 회귀 분석 모형이 클래스 레이블을 예측하는 데 매우 효과적이라는 결론을 내릴 수 있습니다.



이제 위의 분석을 바탕으로 분류 모델 정확도가 매우 우수하다는 결론을 내릴 수 있습니다. 우리 모델은 클래스 레이블을 예측하는 측면에서 매우 잘 수행하고 있습니다.



그러나 기본적인 값 분포는 제공하지 않습니다. 또한, 그것은 우리 반 학생들이 저지르는 오류의 유형에 대해서는 아무 것도 말해주지 않습니다.



우리는 혼란 매트릭스라고 불리는 또 다른 도구를 가지고 있습니다.


15. 혼동 행렬  

혼동 행렬은 분류 알고리즘의 성능을 요약하는 도구입니다. 혼동 행렬은 분류 모델 성능과 모델에 의해 생성되는 오류 유형에 대한 명확한 그림을 제공합니다. 각 범주별로 분류된 정확한 예측과 잘못된 예측의 요약을 제공합니다. 요약은 표 형식으로 표시됩니다.



분류 모델 성능을 평가하는 동안 네 가지 유형의 결과가 가능합니다. 이 네 가지 결과는 아래에 설명되어 있습니다



참 양성(TP) – 참 양성은 관측치가 특정 클래스에 속하고 관측치가 실제로 해당 클래스에 속한다고 예측할 때 발생합니다.



True Negatives(TN) – True Negatives는 관측치가 특정 클래스에 속하지 않고 실제로 관측치가 해당 클래스에 속하지 않을 때 발생합니다.



False Positives(FP) – False Positives는 관측치가 특정 클래스에 속하지만 실제로는 해당 클래스에 속하지 않는다고 예측할 때 발생합니다. 이러한 유형의 오류를 유형 I 오류라고 합니다.



FN(False Negatives) – 관측치가 특정 클래스에 속하지 않지만 실제로는 해당 클래스에 속한다고 예측할 때 False Negatives가 발생합니다. 이것은 매우 심각한 오류이며 Type II 오류라고 합니다.



이 네 가지 결과는 아래에 제시된 혼동 매트릭스로 요약됩니다.



```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_test)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])
```

혼동 행렬은 20892 + 3285 = 24177 정확한 예측과 3087 + 1175 = 4262 부정확한 예측을 나타냅니다.



이 경우, 우리는



참 양성(실제 양성:1 및 예측 양성:1) - 20892

참 음수(실제 음수:0 및 예측 음수:0) - 3285

거짓 양성(실제 음성: 0이지만 예측 양성: 1) - 1175(유형 I 오류)

거짓 음성(실제 양의 1이지만 예측 음의 0) - 3087(타입 II 오류)



```python
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
```

16.분류측정지표  

분류 보고서  

분류 보고서는 분류 모델 성능을 평가하는 또 다른 방법입니다. 모형의 정밀도, 호출, f1 및 지원 점수가 표시됩니다. 저는 이 용어들을 나중에 설명했습니다.



다음과 같이 분류 보고서를 인쇄할 수 있습니다



```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_test))
```

분류정확도



```python
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
```


```python
classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
```

분류오류



```python
classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))
```

정확  

정밀도는 모든 예측 긍정 결과 중 정확하게 예측된 긍정 결과의 백분율로 정의할 수 있습니다. 참 및 거짓 양성의 합계에 대한 참 양성(TP + FP)의 비율로 지정할 수 있습니다.



따라서 정밀도는 정확하게 예측된 양성 결과의 비율을 식별합니다. 그것은 부정적인 계층보다 긍정적인 계층에 더 관심이 있습니다.



수학적으로 정밀도는 TP 대 (TP + FP)의 비율로 정의할 수 있습니다.



```python
precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))
```

리콜  

리콜은 모든 실제 긍정적 결과 중 정확하게 예측된 긍정적 결과의 비율로 정의할 수 있습니다. 참 양성과 거짓 음성의 합(TP + FN)에 대한 참 양성(TP)의 비율로 지정할 수 있습니다. 리콜은 민감도라고도 합니다.



호출은 정확하게 예측된 실제 긍정의 비율을 식별합니다.



수학적으로 호출은 TP 대 (TP + FN)의 비율로 지정할 수 있습니다.



```python
recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))
```

실제 양의 비율  

True Positive Rate는 Recall과 동의어입니다.



```python
true_positive_rate = TP / float(TP + FN)


print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
```

거짓 긍정률



```python
false_positive_rate = FP / float(FP + TN)


print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
```

특수성



```python
specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))
```

f1 점수  

f1-점수는 정밀도와 호출의 가중 조화 평균입니다. 가능한 최상의 f1-점수는 1.0이고 최악의 f1-점수는 0.0입니다. f1-점수는 정밀도와 호출의 조화 평균입니다. 따라서 f1-점수는 정확도와 리콜을 계산에 포함시키기 때문에 정확도 측도보다 항상 낮습니다. f1-점수의 가중 평균은 전역 정확도가 아닌 분류기 모델을 비교하는 데 사용되어야 합니다.


지지  

지원은 데이터 세트에서 클래스의 실제 발생 횟수입니다.


17. 임계값 레벨 조정



```python
y_pred_prob = logreg.predict_proba(X_test)[0:10]

y_pred_prob
```

관찰

각 행에서 숫자는 1이 됩니다.

2개의 클래스(0 및 1)에 해당하는 2개의 열이 있습니다.



클래스 0 - 내일 비가 오지 않을 확률을 예측합니다.



클래스 1 - 내일 비가 올 확률을 예측합니다.



예측 확률의 중요성



비가 오거나 오지 않을 확률로 관측치의 순위를 매길 수 있습니다.

predict_proba 공정



확률을 예측합니다



확률이 가장 높은 클래스 선택



분류 임계값 레벨



분류 임계값 레벨은 0.5입니다.



클래스 1 - 확률이 0.5 이상일 경우 비가 올 확률이 예측됩니다.



클래스 0 - 확률이 0.5 미만일 경우 비가 오지 않을 확률이 예측됩니다.



```python
y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - No rain tomorrow (0)', 'Prob of - Rain tomorrow (1)'])

y_pred_prob_df
```


```python
logreg.predict_proba(X_test)[0:10, 1]
```


```python
y_pred1 = logreg.predict_proba(X_test)[:, 1]
```


```python
plt.rcParams['font.size'] = 12


# plot histogram with 10 bins
plt.hist(y_pred1, bins = 10)


# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of rain')


# set the x-axis limit
plt.xlim(0,1)


# set the title
plt.xlabel('Predicted probabilities of rain')
plt.ylabel('Frequency')
```

관찰  

위의 히스토그램이 매우 양으로 치우쳐 있음을 알 수 있습니다.

첫 번째 열은 확률이 0.0과 0.1 사이인 관측치가 약 15,000개임을 나타냅니다.

확률이 0.5보다 작은 관측치가 있습니다.

그래서 이 소수의 관측치들은 내일 비가 올 것이라고 예측하고 있습니다.

내일은 비가 오지 않을 것이라는 관측이 대다수입니다.


임계값을 낮춥니다



```python
from sklearn.preprocessing import binarize

for i in range(1,5):
    
    cm1=0
    
    y_pred1 = logreg.predict_proba(X_test)[:,1]
    
    y_pred1 = y_pred1.reshape(-1,1)
    
    y_pred2 = binarize(y_pred1, i/10)
    
    y_pred2 = np.where(y_pred2 == 1, 'Yes', 'No')
    
    cm1 = confusion_matrix(y_test, y_pred2)
        
    print ('With',i/10,'threshold the Confusion Matrix is ','\n\n',cm1,'\n\n',
           
            'with',cm1[0,0]+cm1[1,1],'correct predictions, ', '\n\n', 
           
            cm1[0,1],'Type I errors( False Positives), ','\n\n',
           
            cm1[1,0],'Type II errors( False Negatives), ','\n\n',
           
           'Accuracy score: ', (accuracy_score(y_test, y_pred2)), '\n\n',
           
           'Sensitivity: ',cm1[1,1]/(float(cm1[1,1]+cm1[1,0])), '\n\n',
           
           'Specificity: ',cm1[0,0]/(float(cm1[0,0]+cm1[0,1])),'\n\n',
          
            '====================================================', '\n\n')
```

평.  

이항 문제에서는 예측 확률을 클래스 예측으로 변환하는 데 임계값 0.5가 기본적으로 사용됩니다.

임계값을 조정하여 감도 또는 특수성을 높일 수 있습니다.

민감도와 특수성은 역관계가 있습니다. 하나를 늘리면 다른 하나는 항상 감소하고 그 반대도 마찬가지입니다.

임계값 레벨을 높이면 정확도가 높아진다는 것을 알 수 있습니다.

임계값 레벨 조정은 모델 작성 프로세스에서 수행하는 마지막 단계 중 하나여야 합니다.


18. ROC - AUC  

ROC 곡선

분류 모델 성능을 시각적으로 측정하는 또 다른 도구는 ROC 곡선입니다. ROC 곡선은 수신기 작동 특성 곡선을 나타냅니다. ROC 곡선은 다양한 분류 임계값 수준에서 분류 모델의 성능을 보여주는 그림입니다.



ROC 곡선은 다양한 임계값 레벨에서 FPR(False Positive Rate)에 대한 True Positive Rate(TPR)를 표시합니다.



실제 양의 비율(TPR)은 리콜이라고도 합니다. TP 대 (TP + FN)의 비율로 정의됩니다.



FPR(False Positive Rate)은 FP 대 (FP + TN)의 비율로 정의됩니다.



ROC 곡선에서는 단일 지점의 TPR(True Positive Rate)과 FPR(False Positive Rate)에 초점을 맞출 것입니다. 이를 통해 다양한 임계값 레벨에서 TPR과 FPR로 구성된 ROC 곡선의 일반적인 성능을 얻을 수 있습니다. 따라서 ROC 곡선은 여러 분류 임계값 수준에서 TPR 대 FPR을 표시합니다. 임계값 레벨을 낮추면 더 많은 항목이 포지티브로 분류될 수 있습니다. 그러면 True Positives(TP)와 False Positives(FP)가 모두 증가합니다.



```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred1, pos_label = 'Yes')

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for RainTomorrow classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()
```

ROC 곡선은 특정 컨텍스트에 대한 민감도와 특수성의 균형을 맞추는 임계값 레벨을 선택하는 데 도움이 됩니다.


ROC-AUC

ROC AUC는 수신기 작동 특성 - 곡선 아래 영역을 나타냅니다. 분류기 성능을 비교하는 기술입니다. 이 기술에서 우리는 곡선 아래의 면적을 측정합니다. 완벽한 분류기는 ROC AUC가 1인 반면, 순수한 무작위 분류기는 ROC AUC가 0.5입니다.



즉, ROCAUC는 곡선 아래에 있는 ROC 그림의 백분율입니다.



```python
from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred1)

print('ROC AUC : {:.4f}'.format(ROC_AUC))
```

평.  

ROC AUC는 분류기 성능의 단일 숫자 요약입니다. 값이 높을수록 분류기가 더 좋습니다.



우리 모델의 ROCAUC는 1에 접근합니다. 그래서, 우리는 우리의 분류기가 내일 비가 올지 안 올지 예측하는 것을 잘한다는 결론을 내릴 수 있습니다.



```python
from sklearn.model_selection import cross_val_score

Cross_validated_ROC_AUC = cross_val_score(logreg, X_train, y_train, cv=5, scoring='roc_auc').mean()

print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))
```

19. k-Fold 교차 검증



```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(logreg, X_train, y_train, cv = 5, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))
```

평균을 계산하여 교차 검증 정확도를 요약할 수 있습니다.



```python
print('Average cross-validation score: {:.4f}'.format(scores.mean()))
```

우리의 원래 모델 점수는 0.8476입니다. 교차 검증 평균 점수는 0.8474입니다. 따라서 교차 검증을 통해 성능이 향상되지 않는다는 결론을 내릴 수 있습니다.


20. GridSearch CV를 이용한 하이퍼파라미터 최적화



```python
from sklearn.model_selection import GridSearchCV


parameters = [{'penalty':['l1','l2']}, 
              {'C':[1, 10, 100, 1000]}]



grid_search = GridSearchCV(estimator = logreg,  
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           verbose=0)


grid_search.fit(X_train, y_train)
```


```python
print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))

# print parameters that give the best results
print('Parameters that give the best results :','\n\n', (grid_search.best_params_))

# print estimator that was chosen by the GridSearch
print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))
```

평.  

우리의 원래 모델 테스트 정확도는 0.8501인 반면 그리드 검색 CV 정확도는 0.8507입니다.

그리드 검색 CV가 이 특정 모델의 성능을 향상시킨다는 것을 알 수 있습니다.


21. 결과 및 결론


1. 로지스틱 회귀 모형 정확도 점수는 0.8501입니다. 그래서, 이 모델은 호주에 내일 비가 올지 안 올지 예측하는 데 매우 좋은 역할을 합니다.



2. 내일 비가 올 것이라는 관측은 소수입니다. 내일은 비가 오지 않을 것이라는 관측이 대다수입니다.



3. 모형에 과적합 징후가 없습니다.



4. C 값을 늘리면 테스트 세트 정확도가 높아지고 교육 세트 정확도가 약간 높아집니다. 따라서 더 복잡한 모델이 더 나은 성능을 발휘해야 한다는 결론을 내릴 수 있습니다.



5. 임계값 레벨을 높이면 정확도가 높아집니다.



6. 우리 모델의 ROCAUC는 1에 접근합니다. 그래서, 우리는 우리의 분류기가 내일 비가 올지 안 올지 예측하는 것을 잘한다는 결론을 내릴 수 있습니다.



7. 우리의 원래 모델 정확도 점수는 0.8501인 반면 RFECV 이후 정확도 점수는 0.8500입니다. 따라서 기능 집합을 줄이면 거의 유사한 정확도를 얻을 수 있습니다.



8. 원래 모델에서는 FP = 1175인 반면 FP1 = 1174입니다. 그래서 우리는 대략 같은 수의 오검출을 얻습니다. 또한 FN = 3087인 반면 FN1 = 3091입니다. 그래서 우리는 약간 더 높은 거짓 음성을 얻습니다.



9. 우리의 원래 모델 점수는 0.8476입니다. 교차 검증 평균 점수는 0.8474입니다. 따라서 교차 검증을 통해 성능이 향상되지 않는다는 결론을 내릴 수 있습니다.



10. 우리의 원래 모델 테스트 정확도는 0.8501인 반면 그리드 검색 CV 정확도는 0.8507입니다. 그리드 검색 CV가 이 특정 모델의 성능을 향상시킨다는 것을 알 수 있습니다.

