---
layout: single
title:  "Tackle the Titanic dataset"
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


### 3. Tackle the Titanic dataset



```python
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_titanic_data():
    tarball_path = Path("datasets/titanic.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/titanic.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as titanic_tarball:
            titanic_tarball.extractall(path="datasets")
    return [pd.read_csv(Path("datasets/titanic") / filename)
            for filename in ("train.csv", "test.csv")]
```


```python
train_data, test_data = load_titanic_data()
```

데이터는 이미 학습 세트와 테스트 세트로 분할되어 있습니다. 그러나 테스트 데이터에는 레이블이 포함되어 있지 않습니다. 목표는 훈련 데이터를 사용하여 가능한 최고의 모델을 훈련한 다음 테스트 데이터에 대한 예측을 만들고 Kaggle에 업로드하여 최종 점수를 확인하는 것입니다 .



```python
train_data.head()
```

<pre>
   PassengerId  Survived  Pclass  \
0            1         0       3   
1            2         1       1   
2            3         1       3   
3            4         1       1   
4            5         0       3   

                                                Name     Sex   Age  SibSp  \
0                            Braund, Mr. Owen Harris    male  22.0      1   
1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
2                             Heikkinen, Miss. Laina  female  26.0      0   
3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
4                           Allen, Mr. William Henry    male  35.0      0   

   Parch            Ticket     Fare Cabin Embarked  
0      0         A/5 21171   7.2500   NaN        S  
1      0          PC 17599  71.2833   C85        C  
2      0  STON/O2. 3101282   7.9250   NaN        S  
3      0            113803  53.1000  C123        S  
4      0            373450   8.0500   NaN        S  
</pre>
PassengerId : 각 승객의 고유 식별자

Survived : 그것이 목표이며 0은 승객이 생존하지 못했음을 의미하고 1은 생존했음을 의미합니다.

Pclass : 여객 등급.

Name, Sex, Age : 자명

SibSp : 타이타닉에 탑승한 승객의 형제 및 배우자 수.

Parch : 타이타닉호에 탑승한 승객의 자녀와 부모의 수

Ticket : 티켓아이디

Fare : 지불한 가격(파운드)

Cabin : 승객의 캐빈 번호

Embarked : 승객이 타이타닉호에 승선한 곳


목표는 승객의 연령, 성별, 승객 등급, 승선 장소 등과 같은 속성을 기반으로 승객의 생존 여부를 예측하는 것입니다.


PassengerId열을 인덱스 열로 명시적으로 설정해 보겠습니다 .



```python
train_data = train_data.set_index("PassengerId")
test_data = test_data.set_index("PassengerId")
```

누락된 데이터의 양을 확인하기 위해 더 많은 정보를 얻겠습니다.



```python
train_data.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
Int64Index: 891 entries, 1 to 891
Data columns (total 11 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   Survived  891 non-null    int64  
 1   Pclass    891 non-null    int64  
 2   Name      891 non-null    object 
 3   Sex       891 non-null    object 
 4   Age       714 non-null    float64
 5   SibSp     891 non-null    int64  
 6   Parch     891 non-null    int64  
 7   Ticket    891 non-null    object 
 8   Fare      891 non-null    float64
 9   Cabin     204 non-null    object 
 10  Embarked  889 non-null    object 
dtypes: float64(2), int64(4), object(5)
memory usage: 83.5+ KB
</pre>

```python
train_data[train_data["Sex"]=="female"]["Age"].median()
```

<pre>
27.0
</pre>
연령, 캐빈 및 선내 특성은 때때로 null(null이 아닌 891개 미만)이며, 특히 캐빈(77%가 null)입니다. 일단 캐빈은 무시하고 나머지는 집중하겠습니다. Age 속성의 null 값은 약 19%이므로 이 값으로 수행할 작업을 결정해야 합니다. null 값을 중위수 연령으로 대체하는 것이 합리적인 것 같습니다. 다른 열을 기준으로 나이를 예측하면 조금 더 현명해질 수 있습니다(예: 중위 연령은 1등 37세, 2등 29세, 3등 24세). 하지만 우리는 단순하게 유지하고 전체 중위 연령을 사용할 것입니다.


이름 및 티켓 특성 에는 약간의 값이 있을 수 있지만 모델이 사용할 수 있는 유용한 숫자로 변환하기가 약간 까다로울 수 있습니다. 따라서 지금은 무시하겠습니다.


숫자 속성을 살펴보겠습니다.



```python
train_data.describe()
```

<pre>
         Survived      Pclass         Age       SibSp       Parch        Fare
count  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
mean     0.383838    2.308642   29.699113    0.523008    0.381594   32.204208
std      0.486592    0.836071   14.526507    1.102743    0.806057   49.693429
min      0.000000    1.000000    0.416700    0.000000    0.000000    0.000000
25%      0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
50%      0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
75%      1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200
</pre>
 38%만 살아남았습니다. 40%에 가깝기 때문에 정확도는 우리 모델을 평가하는 합리적인 척도가 될 것입니다.

평균 요금은 32.20파운드로 그다지 비싸지 않은 것 같습니다(당시에는 아마 많은 돈이었을 것입니다).

평균 연령은 30세 미만이었다.


대상이 실제로 0 또는 1인지 확인합시다.



```python
train_data["Survived"].value_counts()
```

<pre>
0    549
1    342
Name: Survived, dtype: int64
</pre>

```python
train_data["Pclass"].value_counts()
```

<pre>
3    491
1    216
2    184
Name: Pclass, dtype: int64
</pre>

```python
train_data["Sex"].value_counts()
```

<pre>
male      577
female    314
Name: Sex, dtype: int64
</pre>

```python
train_data["Embarked"].value_counts()
```

<pre>
S    644
C    168
Q     77
Name: Embarked, dtype: int64
</pre>
Embarked 속성은 승객이 승선한 위치를 알려줍니다. C=Cherbourg, Q=Queenstown, S=Southampton.




이제 숫자 속성에 대한 파이프라인부터 시작하여 전처리 파이프라인을 빌드해 보겠습니다.



```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
```

이제 범주 속성에 대한 파이프라인을 구축할 수 있습니다.



```python
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
```


```python
cat_pipeline = Pipeline([
        ("ordinal_encoder", OrdinalEncoder()),    
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])
```

마지막으로 수치 및 범주 파이프라인을 연결해 보겠습니다.



```python
from sklearn.compose import ColumnTransformer

num_attribs = ["Age", "SibSp", "Parch", "Fare"]
cat_attribs = ["Pclass", "Sex", "Embarked"]

preprocess_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])
```

원시 데이터를 가져오고 우리가 원하는 기계 학습 모델에 공급할 수 있는 숫자 입력 기능을 출력하는 멋진 전처리 파이프라인을 가지고 있습니다.



```python
X_train = preprocess_pipeline.fit_transform(train_data)
X_train
```

<pre>
/usr/local/lib/python3.9/dist-packages/sklearn/preprocessing/_encoders.py:868: FutureWarning: `sparse` was renamed to `sparse_output` in version 1.2 and will be removed in 1.4. `sparse_output` is ignored unless you leave `sparse` to its default value.
  warnings.warn(
</pre>
<pre>
array([[-0.56573582,  0.43279337, -0.47367361, ...,  0.        ,
         0.        ,  1.        ],
       [ 0.6638609 ,  0.43279337, -0.47367361, ...,  1.        ,
         0.        ,  0.        ],
       [-0.25833664, -0.4745452 , -0.47367361, ...,  0.        ,
         0.        ,  1.        ],
       ...,
       [-0.10463705,  0.43279337,  2.00893337, ...,  0.        ,
         0.        ,  1.        ],
       [-0.25833664, -0.4745452 , -0.47367361, ...,  1.        ,
         0.        ,  0.        ],
       [ 0.20276213, -0.4745452 , -0.47367361, ...,  0.        ,
         1.        ,  0.        ]])
</pre>

```python
y_train = train_data["Survived"]
```


```python
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train, y_train)
```

<pre>
RandomForestClassifier(random_state=42)
</pre>
모델이 훈련되었으니 이를 사용하여 테스트 세트에 대한 예측을 해봅시다.



```python
X_test = preprocess_pipeline.transform(test_data)
y_pred = forest_clf.predict(X_test)
```


```python
from sklearn.model_selection import cross_val_score
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
forest_scores.mean()
```

<pre>
0.8137578027465668
</pre>

```python
from sklearn.svm import SVC

svm_clf = SVC(gamma="auto")
svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
svm_scores.mean()
```

<pre>
0.8249313358302123
</pre>

```python
plt.figure(figsize=(8, 4))
plt.plot([1]*10, svm_scores, ".")
plt.plot([2]*10, forest_scores, ".")
plt.boxplot([svm_scores, forest_scores], labels=("SVM", "Random Forest"))
plt.ylabel("Accuracy")
plt.show()
```

<pre>
<Figure size 800x400 with 1 Axes>
</pre>
랜덤 포레스트 분류기는 10개의 접기 중 하나에서 매우 높은 점수를 받았지만 전반적으로 평균 점수가 낮고 산포도가 더 커서 SVM 분류기가 더 잘 일반화될 가능성이 있는 것으로 보입니다.


이 결과를 더 개선하려면 다음을 수행할 수 있습니다.



교차 검증 및 그리드 검색을 사용하여 더 많은 모델을 비교하고 하이퍼파라미터를 조정합니다.

예를 들어 더 많은 기능 엔지니어링을 수행하십시오.

숫자 속성을 범주 속성으로 변환해 보십시오. 예를 들어, 연령 그룹마다 생존율이 매우 다르기 때문에(아래 참조) 연령 버킷 범주를 만들어 연령 대신 사용하는 것이 도움이 될 수 있습니다. 마찬가지로 혼자 여행하는 사람들 중 30%만이 살아남았기 때문에 혼자 여행하는 사람들을 위한 특별 범주가 있는 것이 유용할 수 있습니다(아래 참조).

SibSp 와 Parch를 합계로 바꿉니다 .

Survived 속성 과 잘 연관되는 이름 부분을 식별하십시오 .

예를 들어 Cabin 열을 사용하여 첫 글자를 범주 속성으로 처리합니다.



```python
train_data["AgeBucket"] = train_data["Age"] // 15 * 15
train_data[["AgeBucket", "Survived"]].groupby(['AgeBucket']).mean()
```

<pre>
           Survived
AgeBucket          
0.0        0.576923
15.0       0.362745
30.0       0.423256
45.0       0.404494
60.0       0.240000
75.0       1.000000
</pre>

```python
train_data["RelativesOnboard"] = train_data["SibSp"] + train_data["Parch"]
train_data[["RelativesOnboard", "Survived"]].groupby(
    ['RelativesOnboard']).mean()
```

<pre>
                  Survived
RelativesOnboard          
0                 0.303538
1                 0.552795
2                 0.578431
3                 0.724138
4                 0.200000
5                 0.136364
6                 0.333333
7                 0.000000
10                0.000000
</pre>
