{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "### 3. Tackle the Titanic dataset"
      ],
      "metadata": {
        "id": "JU7PktxJZIXO"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from pathlib import Path\n",
        "import pandas as pd\n",
        "import tarfile\n",
        "import urllib.request\n",
        "\n",
        "def load_titanic_data():\n",
        "    tarball_path = Path(\"datasets/titanic.tgz\")\n",
        "    if not tarball_path.is_file():\n",
        "        Path(\"datasets\").mkdir(parents=True, exist_ok=True)\n",
        "        url = \"https://github.com/ageron/data/raw/main/titanic.tgz\"\n",
        "        urllib.request.urlretrieve(url, tarball_path)\n",
        "        with tarfile.open(tarball_path) as titanic_tarball:\n",
        "            titanic_tarball.extractall(path=\"datasets\")\n",
        "    return [pd.read_csv(Path(\"datasets/titanic\") / filename)\n",
        "            for filename in (\"train.csv\", \"test.csv\")]"
      ],
      "metadata": {
        "id": "7xQxLoDQZH_0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_data, test_data = load_titanic_data()"
      ],
      "metadata": {
        "id": "X1CZVQsgZNar"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "데이터는 이미 학습 세트와 테스트 세트로 분할되어 있습니다. 그러나 테스트 데이터에는 레이블이 포함되어 있지 않습니다. 목표는 훈련 데이터를 사용하여 가능한 최고의 모델을 훈련한 다음 테스트 데이터에 대한 예측을 만들고 Kaggle에 업로드하여 최종 점수를 확인하는 것입니다 ."
      ],
      "metadata": {
        "id": "Fpstl59jZO3Q"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "train_data.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "OyTd047uZQUZ",
        "outputId": "e2824d77-a7f4-4a1a-f61b-eadf1c582dd3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   PassengerId  Survived  Pclass  \\\n",
              "0            1         0       3   \n",
              "1            2         1       1   \n",
              "2            3         1       3   \n",
              "3            4         1       1   \n",
              "4            5         0       3   \n",
              "\n",
              "                                                Name     Sex   Age  SibSp  \\\n",
              "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
              "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
              "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
              "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
              "4                           Allen, Mr. William Henry    male  35.0      0   \n",
              "\n",
              "   Parch            Ticket     Fare Cabin Embarked  \n",
              "0      0         A/5 21171   7.2500   NaN        S  \n",
              "1      0          PC 17599  71.2833   C85        C  \n",
              "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
              "3      0            113803  53.1000  C123        S  \n",
              "4      0            373450   8.0500   NaN        S  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-eb3e2734-3660-4c70-b470-6f169062c15a\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Name</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Cabin</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Braund, Mr. Owen Harris</td>\n",
              "      <td>male</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>A/5 21171</td>\n",
              "      <td>7.2500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
              "      <td>female</td>\n",
              "      <td>38.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>PC 17599</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>C85</td>\n",
              "      <td>C</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>Heikkinen, Miss. Laina</td>\n",
              "      <td>female</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>STON/O2. 3101282</td>\n",
              "      <td>7.9250</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
              "      <td>female</td>\n",
              "      <td>35.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>113803</td>\n",
              "      <td>53.1000</td>\n",
              "      <td>C123</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Allen, Mr. William Henry</td>\n",
              "      <td>male</td>\n",
              "      <td>35.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>373450</td>\n",
              "      <td>8.0500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-eb3e2734-3660-4c70-b470-6f169062c15a')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-eb3e2734-3660-4c70-b470-6f169062c15a button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-eb3e2734-3660-4c70-b470-6f169062c15a');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 36
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "PassengerId : 각 승객의 고유 식별자\n",
        "Survived : 그것이 목표이며 0은 승객이 생존하지 못했음을 의미하고 1은 생존했음을 의미합니다.\n",
        "Pclass : 여객 등급.\n",
        "Name, Sex, Age : 자명\n",
        "SibSp : 타이타닉에 탑승한 승객의 형제 및 배우자 수.\n",
        "Parch : 타이타닉호에 탑승한 승객의 자녀와 부모의 수\n",
        "Ticket : 티켓아이디\n",
        "Fare : 지불한 가격(파운드)\n",
        "Cabin : 승객의 캐빈 번호\n",
        "Embarked : 승객이 타이타닉호에 승선한 곳"
      ],
      "metadata": {
        "id": "-J0_zDVLZRzk"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "목표는 승객의 연령, 성별, 승객 등급, 승선 장소 등과 같은 속성을 기반으로 승객의 생존 여부를 예측하는 것입니다."
      ],
      "metadata": {
        "id": "DL41vvBUZTYM"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "PassengerId열을 인덱스 열로 명시적으로 설정해 보겠습니다 ."
      ],
      "metadata": {
        "id": "72sMJ-h4aCpk"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "train_data = train_data.set_index(\"PassengerId\")\n",
        "test_data = test_data.set_index(\"PassengerId\")"
      ],
      "metadata": {
        "id": "gUrNCDxCaHjD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "누락된 데이터의 양을 확인하기 위해 더 많은 정보를 얻겠습니다."
      ],
      "metadata": {
        "id": "P9N_KMnSaQcc"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "train_data.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "e3U2X7UpaMUV",
        "outputId": "36e0ceb2-5c79-47dc-b290-8476a6c8eefd"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "Int64Index: 891 entries, 1 to 891\n",
            "Data columns (total 11 columns):\n",
            " #   Column    Non-Null Count  Dtype  \n",
            "---  ------    --------------  -----  \n",
            " 0   Survived  891 non-null    int64  \n",
            " 1   Pclass    891 non-null    int64  \n",
            " 2   Name      891 non-null    object \n",
            " 3   Sex       891 non-null    object \n",
            " 4   Age       714 non-null    float64\n",
            " 5   SibSp     891 non-null    int64  \n",
            " 6   Parch     891 non-null    int64  \n",
            " 7   Ticket    891 non-null    object \n",
            " 8   Fare      891 non-null    float64\n",
            " 9   Cabin     204 non-null    object \n",
            " 10  Embarked  889 non-null    object \n",
            "dtypes: float64(2), int64(4), object(5)\n",
            "memory usage: 83.5+ KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_data[train_data[\"Sex\"]==\"female\"][\"Age\"].median()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RTAoEJ2YaN35",
        "outputId": "af65341c-10a0-4d32-f0db-a7ded471effa"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "27.0"
            ]
          },
          "metadata": {},
          "execution_count": 39
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "연령, 캐빈 및 선내 특성은 때때로 null(null이 아닌 891개 미만)이며, 특히 캐빈(77%가 null)입니다. 일단 캐빈은 무시하고 나머지는 집중하겠습니다. Age 속성의 null 값은 약 19%이므로 이 값으로 수행할 작업을 결정해야 합니다. null 값을 중위수 연령으로 대체하는 것이 합리적인 것 같습니다. 다른 열을 기준으로 나이를 예측하면 조금 더 현명해질 수 있습니다(예: 중위 연령은 1등 37세, 2등 29세, 3등 24세). 하지만 우리는 단순하게 유지하고 전체 중위 연령을 사용할 것입니다."
      ],
      "metadata": {
        "id": "-8JnsEisaaiS"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "이름 및 티켓 특성 에는 약간의 값이 있을 수 있지만 모델이 사용할 수 있는 유용한 숫자로 변환하기가 약간 까다로울 수 있습니다. 따라서 지금은 무시하겠습니다."
      ],
      "metadata": {
        "id": "S8tgAkF3adMF"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "숫자 속성을 살펴보겠습니다."
      ],
      "metadata": {
        "id": "MlvyLs1faeqW"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "train_data.describe()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 300
        },
        "id": "yn3B8Ag-ag5_",
        "outputId": "5644f432-3505-455a-b105-e4b79605415a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "         Survived      Pclass         Age       SibSp       Parch        Fare\n",
              "count  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000\n",
              "mean     0.383838    2.308642   29.699113    0.523008    0.381594   32.204208\n",
              "std      0.486592    0.836071   14.526507    1.102743    0.806057   49.693429\n",
              "min      0.000000    1.000000    0.416700    0.000000    0.000000    0.000000\n",
              "25%      0.000000    2.000000   20.125000    0.000000    0.000000    7.910400\n",
              "50%      0.000000    3.000000   28.000000    0.000000    0.000000   14.454200\n",
              "75%      1.000000    3.000000   38.000000    1.000000    0.000000   31.000000\n",
              "max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-e6c2fbf2-408e-4f5a-a668-ad0730581e54\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Fare</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>891.000000</td>\n",
              "      <td>891.000000</td>\n",
              "      <td>714.000000</td>\n",
              "      <td>891.000000</td>\n",
              "      <td>891.000000</td>\n",
              "      <td>891.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>0.383838</td>\n",
              "      <td>2.308642</td>\n",
              "      <td>29.699113</td>\n",
              "      <td>0.523008</td>\n",
              "      <td>0.381594</td>\n",
              "      <td>32.204208</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>0.486592</td>\n",
              "      <td>0.836071</td>\n",
              "      <td>14.526507</td>\n",
              "      <td>1.102743</td>\n",
              "      <td>0.806057</td>\n",
              "      <td>49.693429</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.416700</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>0.000000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>20.125000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>7.910400</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>0.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>28.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>14.454200</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>1.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>38.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>31.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>1.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>80.000000</td>\n",
              "      <td>8.000000</td>\n",
              "      <td>6.000000</td>\n",
              "      <td>512.329200</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-e6c2fbf2-408e-4f5a-a668-ad0730581e54')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-e6c2fbf2-408e-4f5a-a668-ad0730581e54 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-e6c2fbf2-408e-4f5a-a668-ad0730581e54');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 40
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        " 38%만 살아남았습니다. 40%에 가깝기 때문에 정확도는 우리 모델을 평가하는 합리적인 척도가 될 것입니다.\n",
        "평균 요금은 32.20파운드로 그다지 비싸지 않은 것 같습니다(당시에는 아마 많은 돈이었을 것입니다).\n",
        "평균 연령은 30세 미만이었다."
      ],
      "metadata": {
        "id": "QqEQDEqbajX7"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "대상이 실제로 0 또는 1인지 확인합시다."
      ],
      "metadata": {
        "id": "UzetA0fnanT2"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "train_data[\"Survived\"].value_counts()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qjufMj2_apos",
        "outputId": "e7dbfa8f-b415-47c9-adc2-4a2a739b7b51"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0    549\n",
              "1    342\n",
              "Name: Survived, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 41
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_data[\"Pclass\"].value_counts()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "dt4oGsNoarC1",
        "outputId": "85ba291e-997e-4ddf-8003-80b3cb4ecda5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "3    491\n",
              "1    216\n",
              "2    184\n",
              "Name: Pclass, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 42
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_data[\"Sex\"].value_counts()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7OfyO4brasDn",
        "outputId": "0ca8cbde-df4e-425f-aad3-f572faa38fb3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "male      577\n",
              "female    314\n",
              "Name: Sex, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 43
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_data[\"Embarked\"].value_counts()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "eaxKwl8ZatFv",
        "outputId": "d2f096b1-1e77-48a3-faff-4c022646f653"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "S    644\n",
              "C    168\n",
              "Q     77\n",
              "Name: Embarked, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 44
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Embarked 속성은 승객이 승선한 위치를 알려줍니다. C=Cherbourg, Q=Queenstown, S=Southampton."
      ],
      "metadata": {
        "id": "bvJJJDXha10Z"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "이제 숫자 속성에 대한 파이프라인부터 시작하여 전처리 파이프라인을 빌드해 보겠습니다."
      ],
      "metadata": {
        "id": "WfndlLBKa3IT"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.impute import SimpleImputer\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "\n",
        "num_pipeline = Pipeline([\n",
        "        (\"imputer\", SimpleImputer(strategy=\"median\")),\n",
        "        (\"scaler\", StandardScaler())\n",
        "    ])"
      ],
      "metadata": {
        "id": "YGXAZJjkauhw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "이제 범주 속성에 대한 파이프라인을 구축할 수 있습니다."
      ],
      "metadata": {
        "id": "oUZPh9E_a45Y"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder"
      ],
      "metadata": {
        "id": "MMXLHCesavnP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cat_pipeline = Pipeline([\n",
        "        (\"ordinal_encoder\", OrdinalEncoder()),    \n",
        "        (\"imputer\", SimpleImputer(strategy=\"most_frequent\")),\n",
        "        (\"cat_encoder\", OneHotEncoder(sparse=False)),\n",
        "    ])"
      ],
      "metadata": {
        "id": "4fiztBDNaw1P"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "마지막으로 수치 및 범주 파이프라인을 연결해 보겠습니다."
      ],
      "metadata": {
        "id": "pWv0FDyCa69m"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.compose import ColumnTransformer\n",
        "\n",
        "num_attribs = [\"Age\", \"SibSp\", \"Parch\", \"Fare\"]\n",
        "cat_attribs = [\"Pclass\", \"Sex\", \"Embarked\"]\n",
        "\n",
        "preprocess_pipeline = ColumnTransformer([\n",
        "        (\"num\", num_pipeline, num_attribs),\n",
        "        (\"cat\", cat_pipeline, cat_attribs),\n",
        "    ])"
      ],
      "metadata": {
        "id": "NOGm9wpHa82J"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "원시 데이터를 가져오고 우리가 원하는 기계 학습 모델에 공급할 수 있는 숫자 입력 기능을 출력하는 멋진 전처리 파이프라인을 가지고 있습니다."
      ],
      "metadata": {
        "id": "sV7mURSxbJUy"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "X_train = preprocess_pipeline.fit_transform(train_data)\n",
        "X_train"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CnO64rS2a-Ih",
        "outputId": "21716526-85f1-4926-fa7f-56dc6bb5255b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.9/dist-packages/sklearn/preprocessing/_encoders.py:868: FutureWarning: `sparse` was renamed to `sparse_output` in version 1.2 and will be removed in 1.4. `sparse_output` is ignored unless you leave `sparse` to its default value.\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[-0.56573582,  0.43279337, -0.47367361, ...,  0.        ,\n",
              "         0.        ,  1.        ],\n",
              "       [ 0.6638609 ,  0.43279337, -0.47367361, ...,  1.        ,\n",
              "         0.        ,  0.        ],\n",
              "       [-0.25833664, -0.4745452 , -0.47367361, ...,  0.        ,\n",
              "         0.        ,  1.        ],\n",
              "       ...,\n",
              "       [-0.10463705,  0.43279337,  2.00893337, ...,  0.        ,\n",
              "         0.        ,  1.        ],\n",
              "       [-0.25833664, -0.4745452 , -0.47367361, ...,  1.        ,\n",
              "         0.        ,  0.        ],\n",
              "       [ 0.20276213, -0.4745452 , -0.47367361, ...,  0.        ,\n",
              "         1.        ,  0.        ]])"
            ]
          },
          "metadata": {},
          "execution_count": 51
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y_train = train_data[\"Survived\"]"
      ],
      "metadata": {
        "id": "FhXUGWd8a_R8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.ensemble import RandomForestClassifier\n",
        "forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)\n",
        "forest_clf.fit(X_train, y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 74
        },
        "id": "_PSazLq8bARe",
        "outputId": "2bb6d894-7f77-4457-b17d-77d660093b28"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "RandomForestClassifier(random_state=42)"
            ],
            "text/html": [
              "<style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-3\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>RandomForestClassifier(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-5\" type=\"checkbox\" checked><label for=\"sk-estimator-id-5\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestClassifier</label><div class=\"sk-toggleable__content\"><pre>RandomForestClassifier(random_state=42)</pre></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 54
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "모델이 훈련되었으니 이를 사용하여 테스트 세트에 대한 예측을 해봅시다."
      ],
      "metadata": {
        "id": "nyLolx1IbOcJ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "X_test = preprocess_pipeline.transform(test_data)\n",
        "y_pred = forest_clf.predict(X_test)"
      ],
      "metadata": {
        "id": "aYHCJeBYbBoU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import cross_val_score\n",
        "forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)\n",
        "forest_scores.mean()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LDHqdemKbBdn",
        "outputId": "b837a3e3-a812-44cb-d80d-958828342a0c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.8137578027465668"
            ]
          },
          "metadata": {},
          "execution_count": 70
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.svm import SVC\n",
        "\n",
        "svm_clf = SVC(gamma=\"auto\")\n",
        "svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)\n",
        "svm_scores.mean()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MCYNqQnibE6m",
        "outputId": "46a7a4cf-0c63-4179-f031-e6357029788b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.8249313358302123"
            ]
          },
          "metadata": {},
          "execution_count": 72
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize=(8, 4))\n",
        "plt.plot([1]*10, svm_scores, \".\")\n",
        "plt.plot([2]*10, forest_scores, \".\")\n",
        "plt.boxplot([svm_scores, forest_scores], labels=(\"SVM\", \"Random Forest\"))\n",
        "plt.ylabel(\"Accuracy\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 368
        },
        "id": "xii4qBDFbYg0",
        "outputId": "642065a1-d7e3-4205-d361-9dd9c3cdb15a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 800x400 with 1 Axes>"
            ],
            "image/png": "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EiJvb2/FxMRoy5YtF+2fkZGhzp07y8fHR8HBwZoxY4ZOnTpVY9+//vWvMplMmj59ei1UDtinfYCfzCbbNg+TSSEBvs4pCACuVNteNbcH31S3dQA1sCsEb9682dF12MjOzlZSUpLmzJmj/Px8devWTQMHDtSRI0dq7L98+XIlJydrzpw5KigoUFZWlrKzszVz5szz+m7dulUvvviibrzxxlqdA3C5Wvn7KG1YuDxM1UnYw2TS/GFd1crfx8mVAYCd2kRK3UbatnUbWd0OOJldIbhXr17q1q2bnnvuOR07dszBJUnp6emaNGmSEhIS1KVLF2VmZsrX11eLFi2qsf/GjRsVGxurkSNHKiQkRAMGDNCIESPOWz0+ceKERo0apYULF+qaa65xeN3AlYqPaqsNybfo1Uk3aUPyLTwUB+Dq98cXpKEvVv/z0Bd5KA4uw64QPHr0aH377be6//771bp1a40dO1affvqpQwo6ffq08vLy1L9//5+LNJvVv39/bdpU871FvXv3Vl5enjX07t27V2vXrtWgQYNs+k2bNk133nmnzdgXUllZqbKyMpsPUBda+fuoV4drWQEGUH8EhtkeARdg1xZpL7/8sp599lktW7ZMWVlZWrZsmV555RV17NhRkyZN0rhx4xQQEGBXQSUlJaqqqlJgYKBNe2BgoHbs2FHjNSNHjlRJSYn69OkjwzB09uxZTZ061eZ2iBUrVig/P19bt269pDrS0tL0+OOP2zUHAACuRrt379bx48cdPm5BQYHN0dEaN26sjh071srYqL/s3ifY399f06ZN07Rp05Sfn6+FCxdqxYoVeuihh/Too49qyJAhmjRp0iWtul6p3NxczZ8/X88//7xiYmL07bff6oEHHtDcuXOVmpqqAwcO6IEHHtD69evl7e19SWOmpKQoKSnJel5WVqbg4ODamgIAAE61e/duderUqVZ/Y/To0bU29q5duwjCuCwOeVlGRESEXnjhBaWnp+u1117TzJkz9frrr+v1119Xu3btNHXqVN17771q3Ljxb44VEBAgDw8PFRcX27QXFxerZcuWNV6TmpqqMWPGaOLEiZKk8PBwlZeXa/LkyXr00UeVl5enI0eOKCIiwnpNVVWVPvnkEz333HOqrKyUh4eHzZheXl7y8vK63H8VAABclc6tAC9btkxhYY69beHkyZPav3+/QkJC5OPj2Fu9CgoKNHr06FpZwUb95rA3xv300096+eWX9a9//UsHDx6UyWRSbGysCgoKlJycrIyMDL399tuKioq66DgNGzZUZGSkcnJyNHToUEmSxWJRTk6OEhMTa7ymoqJCZrPt7c3nQq1hGLrtttv0zTff2HyfkJCg0NBQPfLII+cFYAAA3FVYWJjNopGjxMbGOnxM4EpccQj+6KOPtHDhQr311ls6deqUmjdvroceekhTpkzRddddp8rKSi1atEgPP/yw7rvvPn3++W9vkJ2UlKRx48apZ8+eio6OVkZGhsrLy5WQkCBJGjt2rIKCgpSWliZJiouLU3p6unr06GG9HSI1NVVxcXHy8PBQ48aN1bVrV5vf8PPz07XXXnteOwAAAOo/u0JwcXGxFi9erKysLO3du1eGYejmm2/W1KlTNWzYMDVo0MDa18vLS/fee6++/fZbLViw4JLGj4+P19GjRzV79mwdPnxY3bt317p166wPyxUWFtqs/M6aNUsmk0mzZs1SUVGRmjdvrri4OM2bN8+e6QEAAKCesysEt2nTRhaLRddcc42mT5+uyZMnq3Pnzhe9pnnz5jp9+vQl/0ZiYuIFb3/Izc21Off09NScOXM0Z86cSx7/12MAAADAfdi1T3BMTIyWLl2qoqIiPf30078ZgCUpOTlZFovFnp8DAAAAHMquleANGzY4ug4AAACgzti1Evz9999r9erVF3xl8k8//aTVq1erqKjoSmoDAAAAaoVdIfjJJ59UQkLCBff68/X11YQJE6y7NwAAAACuxK4Q/OGHH2rAgAEXfJmEl5eXBgwYoA8++OCKigMAAABqg10huKioSCEhIRft065dO26HAAAAgEuyKwQ3bNhQZWVlF+1TVlYmk8lkV1EAAABAbbIrBIeHh+udd95RZWVljd+fOnVKq1evVnh4+BUVBwAAANQGu0JwQkKCvv/+ew0ePFh79+61+W7Pnj0aMmSIDh48qIkTJzqkSMCdHCo9qY17SnSo9KSzSwEAx/g+T9r4XPURcBF27ROckJCgtWvX6o033lBoaKjat2+voKAgFRUVad++fTp79qzi4+OVkJDg6HqBei17a6FSVn0jiyGZTVLasHDFR7V1dlkAYL8375W2Lf/5vNtI6Y8vOK8e4L/sWgmWpJUrV+qf//ynrr/+eu3evVu5ubnavXu3OnXqpAULFujVV191ZJ1AvXeo9KQ1AEuSxZBmrtrOijCAq9f3ebYBWKo+Z0UYLsCulWBJMplMSkxMVGJiosrLy1VaWip/f3/5+fk5sj7AbewrKbcG4HOqDEP7SyrUyr/mPbkBwKUVbqq5/cDnUpvIuq0F+BW7Q/Av+fn5EX6BK9Q+wE9mk2yCsIfJpJAAX+cVBQBXom2vmtuDb6rbOoAa2H07BADHauXvo7Rh4fL479aCHiaT5g/ryiowgKtXm8jqe4B/qdtIVoHhEuxeCT5w4ICefPJJffDBBzp48KBOnz59Xh+TyaSzZ89eUYGAO4mPaqu+nZprf0mFQgJ8CcAArn5/fEGKmlh9C0TwTQRguAy7QvDevXsVExOjn376STfccIMqKyvVrl07eXt7a+/evTpz5oy6deumpk2bOrhcoP5r5e9D+AVQv7SJJPzC5dh1O8Tjjz+u0tJS5eTkaNu2bZKqt00rKCjQ/v37NXjwYJWXl+v11193aLEAAACAI9gVgj/44AMNGjRIN998s7XNMKqf5mnVqpWys7MlSTNnznRAiQAAAIBj2RWCS0pKFBoaaj339PRURUWF9dzLy0u///3v9e677155hQAAAICD2RWCAwICVF5ebnO+f/9+mz6enp46duzYldQGAAAA1Aq7QnDHjh21Z88e63l0dLTee+897d27V5J09OhRvf766+rQoYNjqgQAAAAcyK4QfMcdd+ijjz6yrvROnz5dx48f14033qioqCh16tRJhw8f1n333efIWgEAAACHsGuLtHvvvVf9+vWTh4eHJKlfv35asWKFHnvsMW3fvl3t2rXTk08+qUmTJjm0WMBVmM6eUo+WZvkc2yUdvHreOeNzbJd6tDTLdPaUs0sBAMCp7ArBTZo0UUxMjE3b8OHDNXz4cIcUBbg67xOFyp/SSPpkivSJs6u5dGGS8qc0UsGJQkm9nV0OAABOY1cIvvXWWxUbG6u5c+c6uh7gqnCqUVtFvHhCr7zyisJ+sVOKqyvYsUOjRo1S1qC2zi4FAACnsisEb968WTfddJOjawGuGoant748bNHJpp2k1t2dXc4lO3nYoi8PW2R4eju7FAAAnMqumxlDQ0P13XffOboWAAAAoE7YFYLvu+8+vf322/q///s/R9cDAAAA1Dq7boe47rrr1K9fP910002aMmWKoqKiFBgYKJPJdF7fvn37XnGRAAAAgCPZFYL79esnk8kkwzD09NNP1xh+z6mqqrK7OAAAAKA22BWCZ8+efdHgCwAAALgyu0LwY4895uAyAAAAgLrjsq+6WrBggUJCQuTt7a2YmBht2bLlov0zMjLUuXNn+fj4KDg4WDNmzNCpUz+/FSstLU1RUVFq3LixWrRooaFDh2rnzp21PQ3gsh0qPamNe0p0qPSks0sBAKDecskQnJ2draSkJM2ZM0f5+fnq1q2bBg4cqCNHjtTYf/ny5UpOTtacOXNUUFCgrKwsZWdna+bMmdY+H3/8saZNm6bPP/9c69ev15kzZzRgwACVl5fX1bSA35S9tVCxf/1QIxduVuxfP1T21kJnlwQAQL1k1+0QZrP5ku4JNplMOnv27GWPn56erkmTJikhIUGSlJmZqTVr1mjRokVKTk4+r//GjRsVGxurkSNHSpJCQkI0YsQIbd682dpn3bp1NtcsWbJELVq0UF5eHjtYwCUcKj2plFXfyGJUn1sMaeaq7erbqbla+fs4tzgA9Z7p7Cn1aGmWz7Fd0kGXXCOrkc+xXerR0izT2VO/3Rn4BbtCcN++fWsMwaWlpdq9e7fKy8vVrVs3NW3a9LLHPn36tPLy8pSSkmJtM5vN6t+/vzZt2lTjNb1799ayZcu0ZcsWRUdHa+/evVq7dq3GjBlzwd8pLS2VJDVr1qzG7ysrK1VZWWk9Lysru+y5AJdjX0m5NQCfU2UY2l9SQQgGUOu8TxQqf0oj6ZMp0ifOrubShUnKn9JIBScKJfV2djm4itgVgnNzcy/4XUVFhZKTk7Vu3TqtX7/+sscuKSlRVVWVAgMDbdoDAwO1Y8eOGq8ZOXKkSkpK1KdPHxmGobNnz2rq1Kk2t0P8ksVi0fTp0xUbG6uuXbvW2CctLU2PP/74ZdcP2Kt9gJ/MJtkEYQ+TSSEBvs4rCoDbONWorSJePKFXXnlFYaGhzi7nkhXs2KFRo0Ypa1BbZ5eCq4xdIfhifH199c9//lNRUVF66KGHtHjxYkf/xHlyc3M1f/58Pf/884qJidG3336rBx54QHPnzlVqaup5/adNm6bt27drw4YNFxwzJSVFSUlJ1vOysjIFBwfXSv2AJLXy91HasHDNXLVdVYYhD5NJ84d1ZRUYQJ0wPL315WGLTjbtJLXu7uxyLtnJwxZ9edgiw9Pb2aXgKuPwEHzO7373Oy1btuyyrwsICJCHh4eKi4tt2ouLi9WyZcsar0lNTdWYMWM0ceJESVJ4eLjKy8s1efJkPfroozKbf763KTExUe+++64++eQTtWnT5oJ1eHl5ycvL67LrB65EfFRb9e3UXPtLKhQS4EsABgCgltTane9Hjx7ViRMnLvu6hg0bKjIyUjk5OdY2i8WinJwc9erVq8ZrKioqbIKuJHl4eEiSDMOwHhMTE/Xmm2/qww8/VPv27S+7NqAutPL3Ua8O1xKAAQCoRQ5fCbZYLHrllVeUnZ2tnj172jVGUlKSxo0bp549eyo6OloZGRkqLy+37hYxduxYBQUFKS0tTZIUFxen9PR09ejRw3o7RGpqquLi4qxheNq0aVq+fLnefvttNW7cWIcPH5Yk+fv7y8eHsAEAAOBO7ArB1113XY3tZ8+e1ZEjR3TmzBk1aNDAGlIvV3x8vI4eParZs2fr8OHD6t69u9atW2d9WK6wsNBm5XfWrFkymUyaNWuWioqK1Lx5c8XFxWnevHnWPi+88IIkqV+/fja/tXjxYo0fP96uOgEAAHB1sisEWyyWGrdIa9Cggbp27aqoqCglJibqhhtusLuwxMREJSYm1vjdr3en8PT01Jw5czRnzpwLjnfutggAAADArhC8f/9+B5cBAAAA1J2r55UwAAAAgIPYFYK///57rV69WseOHavx+59++kmrV69WUVHRldQGAAAA1Aq7QvCTTz6phISEC+6q4OvrqwkTJtj9YBwAAABQm+wKwR9++KEGDBhwwZdJeHl5acCAAfrggw+uqDgAAACgNtgVgouKihQSEnLRPu3ateN2CAAAALgku0Jww4YNVVZWdtE+ZWVlNW6jBgAAADibXSE4PDxc77zzjiorK2v8/tSpU1q9erXCw8OvqDgAAACgNtgVghMSEvT9999r8ODB2rt3r813e/bs0ZAhQ3Tw4EFNnDjRIUUCAAAAjmTXyzISEhK0du1avfHGGwoNDVX79u0VFBSkoqIi7du3T2fPnlV8fLwSEhIcXS9Q7x0qPal9JeVqH+CnVv4178ACAACujF0hWJJWrlypBQsW6Pnnn9eOHTu0e/duSVKXLl00bdo03XvvvQ4rEnAX2VsLlbLqG1kMyWyS0oaFKz6qrbPLAgCg3rE7BJtMJiUmJioxMVHl5eUqLS2Vv7+//Pz8HFkf4DYOlZ60BmBJshjSzFXb1bdTc1aEAQBwMLtD8C/5+fkRfoErtK+k3BqAz6kyDO0vqSAEAwDgYHY9GPfZZ58pKSlJhw8frvH7Q4cOKSkpSZ9//vkVFQe4k/YBfjL/aldBD5NJIQG+zikIAByltEja90n1EXARdoXg9PR0vfPOO2rZsmWN37dq1UrvvvuunnnmmSsqDnAnrfx9lDYsXB7/3V/bw2TS/GFdWQUGcHXLf1nK6Cotjas+5r/s7IoASXbeDrF161bddtttF+3Tt29frV+/3q6iAHcVH9VWfTs11/6SCoUE+BKAAVzdSoukdx6QDEv1uWGR3pkudbhN8g9yammAXSH4yJEjCgq6+P94W7ZsqSNHjthVFODOWvn7EH4B1A8/7vk5AJ9jVEk/7iUEw+nsuh2iadOmKiwsvGif7777To0aNbKrKAAAUA806yCZfhU1TB5Ss+ucUw/wC3aF4JtuuklvvvmmDhw4UOP3hYWFeuutt9S7d+8rKg4AAFzF/IOkuH9UB1+p+hiXwSowXIJdITgpKUkVFRWKjY3Vyy+/rEOHDkmq3hVi6dKlio2N1cmTJ/Xggw86tFgAAHCViRgrTf9GGvdu9TFirLMrAiTZeU9w3759lZ6ergcffND6amSTySTDqN7k1Gw26x//+If69u3ruEoBAMDVyT+I1V+4HLtflvHAAw/olltuUWZmprZu3arS0lI1bdpU0dHRmjp1qrp27arKykp5eXk5sl4AAADgil3RG+NuvPFGPf/88+e15+fna9q0aVqxYoV++OGHK/kJAAAAwOEc8tpkSTp27JiWLVumrKwsff311zIMQz4+bPMEAAAA13PFIfiDDz5QVlaW3n77bVVWVsowDPXq1UsJCQmKj493RI0AAACAQ9kVgg8cOKDFixdr8eLFKiwslGEYCgoKUlFRkcaPH69FixY5uk4AAADAYS45BJ85c0ZvvfWWsrKylJOTo6qqKvn5+WnUqFEaO3asbr31Vnl6esrT02F3WAAAAAC14pITa+vWrfXjjz/KZDLplltu0dixYzVs2DD5+fnVZn0AAACAw11yCP7hhx9kNps1Y8YMPfzww2revHlt1gUAAADUmkt+Y9z48ePl4+Oj9PR0tWnTRoMHD9Zrr72m06dP12Z9AAAAgMNdcghetGiRDh06pBdffFERERF69913dffddyswMFBTpkzRhg0barNOAAAAwGEuOQRLUqNGjTRx4kRt2rRJ//nPfzR9+nQ1bNhQCxcu1M033yyTyaSdO3fqu+++u+LCFixYoJCQEHl7eysmJkZbtmy5aP+MjAx17txZPj4+Cg4O1owZM3Tq1KkrGhMAAAD1k91bOYSFhenpp5/WU089Zd01Yv369fr000/VoUMH3XzzzRo/frzGjBlz2WNnZ2crKSlJmZmZiomJUUZGhgYOHKidO3eqRYsW5/Vfvny5kpOTtWjRIvXu3Vu7du3S+PHjZTKZlJ6ebteYwMVUVFRIqn47oqOdPHlS+/fvV0hIiMNfOFNQUODQ8QAAuGoZDnTgwAHjiSeeMK677jrDZDIZZrPZrnGio6ONadOmWc+rqqqM1q1bG2lpaTX2nzZtmnHrrbfatCUlJRmxsbF2j/lrpaWlhiSjtLT0cqaCemrhwoWGpKv2s2vXLmf/KwTgYvLy8gxJRl5enrNLuSxXa92oHZeT1xy6qW+bNm2Umpqq1NRU5eTk2PXSjNOnTysvL08pKSnWNrPZrP79+2vTpk01XtO7d28tW7ZMW7ZsUXR0tPbu3au1a9daV6HtGbOyslKVlZXW87KyssueC+qvoUOHSpJCQ0Pl6+vr0LELCgo0evRoLVu2TGFhYQ4dW5IaN26sjh07OnxcAACuJrX2ZovbbrtNt91222VfV1JSoqqqKgUGBtq0BwYGaseOHTVeM3LkSJWUlKhPnz4yDENnz57V1KlTNXPmTLvHTEtL0+OPP37Z9cM9BAQEaOLEibX6G2FhYYqIiKjV3wAAwF1d1oNxrio3N1fz58/X888/r/z8fK1atUpr1qzR3Llz7R4zJSVFpaWl1s+BAwccWDEAAACcyeXecRwQECAPDw8VFxfbtBcXF6tly5Y1XpOamqoxY8ZYV+bCw8NVXl6uyZMn69FHH7VrTC8vL3l5eTlgRsDlOXr8lM0RAK56pUXSj3ukZh0k/yBnVwNIcsGV4IYNGyoyMlI5OTnWNovFopycHPXq1avGayoqKmQ2207Fw8NDkmQYhl1jAs6QvbVQE5ZslSRNWLJV2VsLnVwRAFyh/JeljK7S0rjqY/7Lzq4IkOSCIViSkpKStHDhQi1dulQFBQW69957VV5eroSEBEnS2LFjbR5yi4uL0wsvvKAVK1Zo3759Wr9+vVJTUxUXF2cNw781JuBsh0pPKmXVN7IY1ecWQ5q5arsOlZ50bmEAYK/SIumdByTDUn1uWKR3ple3A07mcrdDSFJ8fLyOHj2q2bNn6/Dhw+revbvWrVtnfbCtsLDQZuV31qxZMplMmjVrloqKitS8eXPFxcVp3rx5lzwm4Gz7SsqtAficKsPQ/pIKtfJ37H7BAFAnftzzcwA+x6iSftzLbRFwOpcMwZKUmJioxMTEGr/Lzc21Off09NScOXM0Z84cu8cEnK19gJ/MJts2D5NJIQGO3YINAOpMsw6SyWwbhE0eUrPrnFcT8F8ueTsE4I5a+fsobVi4PEzVSdjDZNL8YV1ZBQZw9fIPkuL+UR18pepjXAarwHAJLrsSDLij+Ki2ajq+p25fImWN76mBUW2dXRIAXJmIsVKH26pvgWh2HQEYLoMQDLiY5o29bY4AcNXzDyL8wuVwOwQAAADcDiEYAAAAbocQDAAAALdDCAYAAIDbIQQDAADA7RCCAQAA4HYIwQAAAHA7hGAAAAC4HUIwAAAA3A4hGAAAAG6HEAwAAAC3QwgGAACA2yEEAwAAwO0QggEAAOB2CMEAAABwO4RgAAAAuB1CMOBijh4/ZXMEAACORwgGXEj21kJNWLJVkjRhyVZlby10ckUAANRPhGDARRwqPamUVd/IYlSfWwxp5qrtOlR60rmFAQBQDxGCARexr6TcGoDPqTIM7S+pcE5BAADUY4RgwEW0D/CT2WTb5mEyKSTA1zkFAQBQjxGCARfRyt9HacPC5WGqTsIeJpPmD+uqVv4+Tq4MAID6hxAMuJD4qLbKGt9TkpQ1vqfio9o6uSIAAOonQjDgYpo39rY5AgAAxyMEAwAAwO0QggEAAOB2CMEAAABwO4RgAAAAuB1CMAAAANyOy4bgBQsWKCQkRN7e3oqJidGWLVsu2Ldfv34ymUznfe68805rnxMnTigxMVFt2rSRj4+PunTposzMzLqYCgAAAFyMp7MLqEl2draSkpKUmZmpmJgYZWRkaODAgdq5c6datGhxXv9Vq1bp9OnT1vMffvhB3bp10/Dhw61tSUlJ+vDDD7Vs2TKFhITo/fff11/+8he1bt1agwcPrpN5AQDgqioqql/Rnp+f7/CxT548qf379yskJEQ+Po59AVBBQYFDx4P7cMkQnJ6erkmTJikhIUGSlJmZqTVr1mjRokVKTk4+r3+zZs1szlesWCFfX1+bELxx40aNGzdO/fr1kyRNnjxZL774orZs2VJjCK6srFRlZaX1vKyszBFTAwDAJe3YsUOSNGnSJCdXYp/GjRs7uwRcZVwuBJ8+fVp5eXlKSUmxtpnNZvXv31+bNm26pDGysrJ09913y8/Pz9rWu3dvrV69WhMmTFDr1q2Vm5urXbt26ZlnnqlxjLS0ND3++ONXNhkAAK4SQ4cOlSSFhobK19fXoWMXFBRo9OjRWrZsmcLCwhw6tlQdgDt27OjwcVG/uVwILikpUVVVlQIDA23aAwMDrf8v9WK2bNmi7du3Kysry6b92Wef1eTJk9WmTRt5enrKbDZr4cKF6tu3b43jpKSkKCkpyXpeVlam4OBgO2YEAIDrCwgI0MSJE2v1N8LCwhQREVGrvwFcKpcLwVcqKytL4eHhio6Otml/9tln9fnnn2v16tVq166dPvnkE02bNk2tW7dW//79zxvHy8tLXl5edVU2AAAA6pDLheCAgAB5eHiouLjYpr24uFgtW7a86LXl5eVasWKFnnjiCZv2kydPaubMmXrzzTetO0bceOON+uqrr/T3v/+9xhAMAACA+svltkhr2LChIiMjlZOTY22zWCzKyclRr169Lnrta6+9psrKSo0ePdqm/cyZMzpz5ozMZtvpenh4yGKxOK54AAAAXBVcbiVYqt7ObNy4cerZs6eio6OVkZGh8vJy624RY8eOVVBQkNLS0myuy8rK0tChQ3XttdfatDdp0kQ333yzHnroIfn4+Khdu3b6+OOP9fLLLys9Pb3O5gUAAADX4JIhOD4+XkePHtXs2bN1+PBhde/eXevWrbM+LFdYWHjequ7OnTu1YcMGvf/++zWOuWLFCqWkpGjUqFH68ccf1a5dO82bN09Tp06t9fkAAADAtbhkCJakxMREJSYm1vhdbm7ueW2dO3eWYRgXHK9ly5ZavHixo8oDas3R46dsjgBw1Tt+xPYIuACXuycYcGfZWws1YclWSdKEJVuVvbXQyRUBwBXKf1la/t+XVy0fXn0OuACXXQkG6pOKiorf3Of66PFTSlqyVZUlByRJlSUH9ODzb6np+J5q3tj7N3+jNja4B4ArUlokvfOApHN/U2tI70yXOtwm+Qc5sTCAEAzUiR07digyMvKyrvnh3aclSbcvubT+eXl5bEIPwLX8uEcyfrULk1El/biXEAynIwQDdSA0NFR5eXkX7XP0+ClNWLJVVWdO62xpsTz9A+XZwEtZl7ESDAAupVkHyWSWVPVzm8lDanad00oCziEEA3XA19f3klZp033baOaq7apq00UeJpPmD+uqgVFt66BCAKgF/kFS3D+kF8896G6W4jJYBYZLIAQDLiQ+qq36dmqu/SUVCgnwVSt/H2eXBABXJmKsNLKF9NId0siVUsTtzq4IkEQIBlxOK38fwi+A+qVxC9sj4ALYIg0AAABuhxAMAAAAt0MIBgAAgNshBAMAAMDtEIIBAADgdgjBAAAAcDuEYAAAALgdQjAAAADcDiEYAAAAbocQDAAAALdDCAYAAIDbIQQDAADA7RCCAQAA4HYIwQAAAHA7hGAAAAC4HUIwAAAA3A4hGAAA1K7jR2yPgAsgBAMAgNqT/7K0fHj1Py8fXn0OuABCMAAAqB2lRdI7D0gy/ttgSO9Mr24HnIwQDAAAasePeyTDYttmVEk/7nVOPcAvEIIBAEDtaNZBMv0qapg8pGbXOace4BcIwQAAoHb4B0lx/9DPccMsxWVUtwNORggGAAC1J2KsNHJl9T+PXFl9DrgAQjAAAKhdjVvYHgEX4LIheMGCBQoJCZG3t7diYmK0ZcuWC/bt16+fTCbTeZ8777zTpl9BQYEGDx4sf39/+fn5KSoqSoWFhbU9FQAAALgYlwzB2dnZSkpK0pw5c5Sfn69u3bpp4MCBOnKk5k22V61apUOHDlk/27dvl4eHh4YPH27ts2fPHvXp00ehoaHKzc3V119/rdTUVHl7e9fVtAAAAOAiTIZhGL/drW7FxMQoKipKzz33nCTJYrEoODhY9913n5KTk3/z+oyMDM2ePVuHDh2Sn5+fJOnuu+9WgwYN9L//+7921VRWViZ/f3+VlpaqSZMmdo0BAIA7ys/PV2RkpPLy8hQREeHsclCPXU5ec7mV4NOnTysvL0/9+/e3tpnNZvXv31+bNm26pDGysrJ09913WwOwxWLRmjVr1KlTJw0cOFAtWrRQTEyM3nrrrQuOUVlZqbKyMpsPAAAA6geXC8ElJSWqqqpSYGCgTXtgYKAOHz78m9dv2bJF27dv18SJE61tR44c0YkTJ/TXv/5Vt99+u95//3398Y9/1LBhw/Txxx/XOE5aWpr8/f2tn+Dg4CubGAAAAFyGy4XgK5WVlaXw8HBFR0db2yyW6rfVDBkyRDNmzFD37t2VnJysP/zhD8rMzKxxnJSUFJWWllo/Bw4cqJP6AQAAUPtcLgQHBATIw8NDxcXFNu3FxcVq2bLlRa8tLy/XihUrdM8995w3pqenp7p06WLTHhYWdsHdIby8vNSkSRObDwAAAOoHlwvBDRs2VGRkpHJycqxtFotFOTk56tWr10Wvfe2111RZWanRo0efN2ZUVJR27txp075r1y61a9fOccUDAADgquDp7AJqkpSUpHHjxqlnz56Kjo5WRkaGysvLlZCQIEkaO3asgoKClJaWZnNdVlaWhg4dqmuvvfa8MR966CHFx8erb9++uuWWW7Ru3Tq98847ys3NrYspAQAAwIW4ZAiOj4/X0aNHNXv2bB0+fFjdu3fXunXrrA/LFRYWymy2XcTeuXOnNmzYoPfff7/GMf/4xz8qMzNTaWlpuv/++9W5c2e98cYb6tOnT63PBwAAAK7FJfcJdkXsEwwAgH3yP16nyH53KC/334q4+XZnl4N67KreJxgAANQj+S9Ly//7Btflw6vPARfgkrdDAAAA11ZRUaEdO3ZcvNPxI9LyaSooOStJ1ccXE6WRLaTGLX7zN0JDQ+Xr6+uIcoHzEIIBAMBl27FjhyIjIy/rmtGrTkk6Jb10xyX15zXLqE2EYAAAcNlCQ0OVl5d38U7Hj0jLh+vkWYv2H7MopKlZPp4e0siVl7wSDNQWQjAAALhsvr6+l7ZK23iB9M50xQZXSSYPKS5DiuDhODgfIRgAANSeiLFSh9ukH/dKza6T/IOcXREgiRAMAABqm38Q4Rcuhy3SAAAA4HYIwQAAAHA7hGAAAAC4HUIwAAAA3A4hGAAAAG6HEAwAAAC3QwgGAACA2yEEAwAAwO3wsoxLZBiGJKmsrMzJlQAAAKAm53Laudx2MYTgS3T8+HFJUnBwsJMrAQAAwMUcP35c/v7+F+1jMi4lKkMWi0UHDx5U48aNZTKZnF0O6rGysjIFBwfrwIEDatKkibPLAYArxp9rqCuGYej48eNq3bq1zOaL3/XLSvAlMpvNatOmjbPLgBtp0qQJ/7EAUK/w5xrqwm+tAJ/Dg3EAAABwO4RgAAAAuB1CMOBivLy8NGfOHHl5eTm7FABwCP5cgyviwTgAAAC4HVaCAQAA4HYIwQAAAHA7hGAAAAC4HUIwAAAA3A4hGAAA1MhkMumtt95ydhlArSAEA7Xs6NGjuvfee9W2bVt5eXmpZcuWGjhwoD7++GMFBATor3/9a43XzZ07V4GBgTpz5oyWLFkik8mksLCw8/q99tprMplMCgkJqeWZAKhr48ePl8lkkslkUoMGDdS+fXs9/PDDOnXqlLNLq1W/nPcvP99++61Taxo6dKjTfh+ORwgGatmf/vQnffnll1q6dKl27dql1atXq1+/fiotLdXo0aO1ePHi864xDENLlizR2LFj1aBBA0mSn5+fjhw5ok2bNtn0zcrKUtu2betkLgDq3u23365Dhw5p7969euaZZ/Tiiy9qzpw5zi6r1p2b9y8/7du3t2us06dPO7g61AeEYKAWHTt2TJ9++qmeeuop3XLLLWrXrp2io6OVkpKiwYMH65577tGuXbu0YcMGm+s+/vhj7d27V/fcc4+1zdPTUyNHjtSiRYusbd9//71yc3M1cuTIOpsTgLp17m+QgoODNXToUPXv31/r16+3fv/DDz9oxIgRCgoKkq+vr8LDw/Xqq6/ajNGvXz/df//9evjhh9WsWTO1bNlSjz32mE2f3bt3q2/fvvL29laXLl1sfuOcb775Rrfeeqt8fHx07bXXavLkyTpx4oT1+3OrpfPnz1dgYKCaNm2qJ554QmfPntVDDz2kZs2aqU2bNjX+n/8LzfuXHw8PD0nVf0ZGR0fLy8tLrVq1UnJyss6ePWsz38TERE2fPl0BAQEaOHCgJGn79u2644471KhRIwUGBmrMmDEqKSmxXvf6668rPDzcOr/+/furvLxcjz32mJYuXaq3337buiqdm5v7m3OAayMEA7WoUaNGatSokd566y1VVlae9314eLiioqJsgq0kLV68WL1791ZoaKhN+4QJE7Ry5UpVVFRIkpYsWaLbb79dgYGBtTcJAC5j+/bt2rhxoxo2bGhtO3XqlCIjI7VmzRpt375dkydP1pgxY7Rlyxaba5cuXSo/Pz9t3rxZf/vb3/TEE09Yg67FYtGwYcPUsGFDbd68WZmZmXrkkUdsri8vL9fAgQN1zTXXaOvWrXrttdf0wQcfKDEx0abfhx9+qIMHD+qTTz5Renq65syZoz/84Q+65pprtHnzZk2dOlVTpkzR999/b9e/g6KiIg0aNEhRUVHatm2bXnjhBWVlZenJJ588b74NGzbUZ599pszMTB07dky33nqrevTooS+++ELr1q1TcXGx7rrrLknSoUOHNGLECE2YMEEFBQXKzc3VsGHDZBiG/t//+3+66667bFane/fubVf9cCEGgFr1+uuvG9dcc43h7e1t9O7d20hJSTG2bdtm/T4zM9No1KiRcfz4ccMwDKOsrMzw9fU1/vWvf1n7LF682PD39zcMwzC6d+9uLF261LBYLEaHDh2Mt99+23jmmWeMdu3a1eW0ANSBcePGGR4eHoafn5/h5eVlSDLMZrPx+uuvX/S6O++803jwwQet5zfffLPRp08fmz5RUVHGI488YhiGYbz33nuGp6enUVRUZP3+3//+tyHJePPNNw3DMIyXXnrJuOaaa4wTJ05Y+6xZs8Ywm83G4cOHrfW2a9fOqKqqsvbp3Lmz8bvf/c56fvbsWcPPz8949dVXL2ne5z5//vOfDcMwjJkzZxqdO3c2LBaLtf+CBQuMRo0aWX/35ptvNnr06GEz5ty5c40BAwbYtB04cMCQZOzcudPIy8szJBn79++/YE1Dhgy5YM24+rASDNSyP/3pTzp48KBWr16t22+/Xbm5uYqIiNCSJUskSSNGjFBVVZVWrlwpScrOzpbZbFZ8fHyN402YMEGLFy/Wxx9/rPLycg0aNKiupgLACW655RZ99dVX2rx5s8aNG6eEhAT96U9/sn5fVVWluXPnKjw8XM2aNVOjRo303nvvqbCw0GacG2+80ea8VatWOnLkiCSpoKBAwcHBat26tfX7Xr162fQvKChQt27d5OfnZ22LjY2VxWLRzp07rW033HCDzOaf40VgYKDCw8Ot5x4eHrr22mutv/1b8z73+ec//2mto1evXjKZTDZ1nDhxwmZ1OTIy0ma8bdu26aOPPrL+DV2jRo2sf9u2Z88edevWTbfddpvCw8M1fPhwLVy4UD/99NNFa8TVjRAM1AFvb2/9/ve/V2pqqjZu3Kjx48dbH2xp0qSJ/vznP1vvkVu8eLHuuusuNWrUqMaxRo0apc8//1yPPfaYxowZI09PzzqbB4C65+fnp+uvv17dunXTokWLtHnzZmVlZVm//5//+R/94x//0COPPKKPPvpIX331lQYOHHjew2DnHrI9x2QyyWKxOLzemn7Hnt8+N+9zn1atWl1WHb8M65J04sQJxcXF2QTrr776ynovtIeHh9avX69///vf6tKli5599ll17txZ+/btu6zfxdWDEAw4QZcuXVReXm49v+eee7Rhwwa9++672rhxo80Dcb/WrFkzDR48WB9//LEmTJhQF+UCcBFms1kzZ87UrFmzdPLkSUnSZ599piFDhmj06NHq1q2brrvuOu3ateuyxg0LC9OBAwd06NAha9vnn39+Xp9t27bZ/Nn12WefyWw2q3Pnzlcwq8sTFhamTZs2yTAMmzoaN26sNm3aXPC6iIgI/ec//1FISIhNuL7++uutgdlkMik2NlaPP/64vvzySzVs2FBvvvmmJKlhw4aqqqqq3cmhThGCgVr0ww8/6NZbb9WyZcv09ddfa9++fXrttdf0t7/9TUOGDLH269u3r66//nqNHTtWoaGhv/nAxZIlS1RSUnLeg3MA6r/hw4fLw8NDCxYskCR17NhR69ev18aNG1VQUKApU6aouLj4ssbs37+/OnXqpHHjxmnbtm369NNP9eijj9r0GTVqlLy9vTVu3Dht375dH330ke677z6NGTOmTh/O/ctf/qIDBw7ovvvu044dO/T2229rzpw5SkpKsrkN49emTZumH3/8USNGjNDWrVu1Z88evffee0pISFBVVZU2b96s+fPn64svvlBhYaFWrVqlo0ePWvdnDwkJ0ddff62dO3eqpKREZ86cqaspo5YQgoFa1KhRI8XExOiZZ55R37591bVrV6WmpmrSpEl67rnnrP1MJpMmTJign3766ZJWd89t3wPA/Xh6eioxMVF/+9vfVF5erlmzZikiIkIDBw5Uv3791LJly8t+qYPZbNabb76pkydPKjo6WhMnTtS8efNs+vj6+uq9997Tjz/+qKioKP35z3/WbbfdZvNnWV0ICgrS2rVrtWXLFnXr1k1Tp07VPffco1mzZl30utatW+uzzz5TVVWVBgwYoPDwcE2fPl1NmzaV2WxWkyZN9Mknn2jQoEHq1KmTZs2apaefflp33HGHJGnSpEnq3LmzevbsqebNm+uzzz6ri+miFpmMX/59AgAAAOAGWAkGAACA2yEEAwAAwO0QggEAAOB2CMEAAABwO4RgAAAAuB1CMAAAANwOIRgAAABuhxAMAAAAt0MIBgAAgNshBAMAAMDtEIIBAADgdv4/JUDZv+zj9yAAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "랜덤 포레스트 분류기는 10개의 접기 중 하나에서 매우 높은 점수를 받았지만 전반적으로 평균 점수가 낮고 산포도가 더 커서 SVM 분류기가 더 잘 일반화될 가능성이 있는 것으로 보입니다."
      ],
      "metadata": {
        "id": "Zp1J_AZHbiRV"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "이 결과를 더 개선하려면 다음을 수행할 수 있습니다.\n",
        "\n",
        "교차 검증 및 그리드 검색을 사용하여 더 많은 모델을 비교하고 하이퍼파라미터를 조정합니다.\n",
        "예를 들어 더 많은 기능 엔지니어링을 수행하십시오.\n",
        "숫자 속성을 범주 속성으로 변환해 보십시오. 예를 들어, 연령 그룹마다 생존율이 매우 다르기 때문에(아래 참조) 연령 버킷 범주를 만들어 연령 대신 사용하는 것이 도움이 될 수 있습니다. 마찬가지로 혼자 여행하는 사람들 중 30%만이 살아남았기 때문에 혼자 여행하는 사람들을 위한 특별 범주가 있는 것이 유용할 수 있습니다(아래 참조).\n",
        "SibSp 와 Parch를 합계로 바꿉니다 .\n",
        "Survived 속성 과 잘 연관되는 이름 부분을 식별하십시오 .\n",
        "예를 들어 Cabin 열을 사용하여 첫 글자를 범주 속성으로 처리합니다."
      ],
      "metadata": {
        "id": "mcZiY8B6blNB"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "train_data[\"AgeBucket\"] = train_data[\"Age\"] // 15 * 15\n",
        "train_data[[\"AgeBucket\", \"Survived\"]].groupby(['AgeBucket']).mean()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 269
        },
        "id": "ZfvvpeeSbbNX",
        "outputId": "e6e2ea69-212e-4179-8525-6a4059fb2663"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "           Survived\n",
              "AgeBucket          \n",
              "0.0        0.576923\n",
              "15.0       0.362745\n",
              "30.0       0.423256\n",
              "45.0       0.404494\n",
              "60.0       0.240000\n",
              "75.0       1.000000"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-722a78a4-37f4-4fb0-9366-8bef0acaff5c\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Survived</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>AgeBucket</th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0.0</th>\n",
              "      <td>0.576923</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>15.0</th>\n",
              "      <td>0.362745</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>30.0</th>\n",
              "      <td>0.423256</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>45.0</th>\n",
              "      <td>0.404494</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>60.0</th>\n",
              "      <td>0.240000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75.0</th>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-722a78a4-37f4-4fb0-9366-8bef0acaff5c')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-722a78a4-37f4-4fb0-9366-8bef0acaff5c button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-722a78a4-37f4-4fb0-9366-8bef0acaff5c');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 74
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_data[\"RelativesOnboard\"] = train_data[\"SibSp\"] + train_data[\"Parch\"]\n",
        "train_data[[\"RelativesOnboard\", \"Survived\"]].groupby(\n",
        "    ['RelativesOnboard']).mean()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 363
        },
        "id": "btAT8g3Tbceb",
        "outputId": "302bd680-e9cb-435f-faba-2edff2818243"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                  Survived\n",
              "RelativesOnboard          \n",
              "0                 0.303538\n",
              "1                 0.552795\n",
              "2                 0.578431\n",
              "3                 0.724138\n",
              "4                 0.200000\n",
              "5                 0.136364\n",
              "6                 0.333333\n",
              "7                 0.000000\n",
              "10                0.000000"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-a48c5505-576c-4f95-a4a1-7096b9ee00dd\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Survived</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>RelativesOnboard</th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0.303538</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0.552795</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>0.578431</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0.724138</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0.200000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>0.136364</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>0.333333</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>10</th>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-a48c5505-576c-4f95-a4a1-7096b9ee00dd')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-a48c5505-576c-4f95-a4a1-7096b9ee00dd button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-a48c5505-576c-4f95-a4a1-7096b9ee00dd');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 75
        }
      ]
    }
  ]
}