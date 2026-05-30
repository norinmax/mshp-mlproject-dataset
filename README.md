# 📊 hh-vacancies-dataset

Датасет вакансий с **hh.ru**, собранный через официальный API для задачи многоклассовой классификации: предсказание требуемого уровня опыта работы по характеристикам вакансии.

---

## 📁 Структура репозитория

```
hh-vacancies-dataset/
├── hh_vacancies.csv          # Датасет
├── data_collection.ipynb     # Парсинг + EDA + baseline-модель
├── eda_overview.png          # Графики разведочного анализа
├── baseline_results.png      # Матрица ошибок и важность признаков
└── README.md
```

---

## 🎯 Задача

**Многоклассовая классификация** — предсказать требуемый опыт работы для вакансии по её характеристикам.

| Класс | Описание |
|-------|----------|
| `noExperience` | Без опыта |
| `between1And3` | от 1 до 3 лет |
| `between3And6` | от 3 до 6 лет |
| `moreThan6` | более 6 лет |

---

## 🗃️ Описание датасета

Файл: `hh_vacancies.csv` (~5 500 вакансий из 5 городов России)

| Колонка | Тип | Описание |
|---------|-----|----------|
| `id` | str | Уникальный ID вакансии на hh.ru |
| `title` | str | Название вакансии |
| `city` | str | Город |
| `experience` | str | ⭐ **Целевая переменная** (4 класса) |
| `employment` | str | Тип занятости (полная, частичная и др.) |
| `schedule` | str | График работы (офис, удалёнка и др.) |
| `salary_from` | float | Минимум зарплатной вилки (руб.) |
| `salary_to` | float | Максимум зарплатной вилки (руб.) |
| `salary_currency` | str | Валюта зарплаты |
| `has_salary` | int | Указана ли зарплата (0/1) |
| `salary_gross` | int | Зарплата до вычета налогов (0/1) |
| `is_premium` | int | Платное продвижение вакансии (0/1) |
| `employer_trusted` | int | Верифицированный работодатель (0/1) |
| `has_address` | int | Указан физический адрес офиса (0/1) |
| `role_id` | int | ID профессиональной роли (категория должности) |
| `published_weekday` | int | День недели публикации (0 = понедельник) |
| `published_hour` | int | Час публикации |
| `title_word_count` | int | Количество слов в названии вакансии |

---

## 🔧 Сбор данных

Данные собраны через **официальный API hh.ru** с OAuth 2.0 авторизацией (`client_credentials`).

**Охват:**
- 5 городов: Москва, Санкт-Петербург, Екатеринбург, Новосибирск, Казань
- До 100 вакансий на страницу, до 20 страниц на регион
- Дедупликация по уникальному `id`

**Обоснование признаков:**
- `title` — косвенно кодирует уровень (Junior/Senior/Lead)
- `salary_from/to` — зарплатная вилка отражает грейд
- `role_id` — профессиональная категория
- `employment` / `schedule` — тип и режим работы
- `published_weekday` / `published_hour` — паттерны HR-активности

---

## 🚀 Быстрый старт

### 1. Зависимости

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Загрузка и предобработка

```python
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

df = pd.read_csv('hh_vacancies.csv')

CATEGORICAL = ['city', 'employment', 'schedule', 'salary_currency']
NUMERIC     = ['salary_from', 'salary_to', 'title_word_count',
               'published_weekday', 'published_hour', 'role_id']
BINARY      = ['has_salary', 'salary_gross', 'is_premium',
               'employer_trusted', 'has_address']

df[NUMERIC]     = df[NUMERIC].fillna(-1)
df[CATEGORICAL] = df[CATEGORICAL].fillna('unknown')
df[BINARY]      = df[BINARY].fillna(0).astype(int)

enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df[CATEGORICAL] = enc.fit_transform(df[CATEGORICAL])

le = LabelEncoder()
y  = le.fit_transform(df['experience'])
```

### 3. Baseline-модель (Random Forest)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

FEATURES = CATEGORICAL + NUMERIC + BINARY
X = df[FEATURES]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=300, max_depth=15,
    min_samples_leaf=2, random_state=42, n_jobs=-1
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f'Accuracy:      {accuracy_score(y_test, y_pred):.3f}')
print(f'F1 (weighted): {f1_score(y_test, y_pred, average="weighted"):.3f}')
```

---

## 📓 Ноутбук

`data_collection.ipynb` содержит три части:

1. **Парсинг** — сбор данных через API hh.ru
2. **EDA** — распределение классов, топ городов, анализ зарплат
3. **Baseline** — обучение `RandomForestClassifier`, матрица ошибок, важность признаков

Также доступен в [Google Colab](https://colab.research.google.com/drive/1GNk8vn1M2EJDe9jR1-LbQkcn-aGQ9dY4).

---

## ⚖️ Правила использования

Данные получены через официальный API hh.ru в соответствии с [условиями использования](https://dev.hh.ru/articles/terms). Датасет предназначен **только для учебных и исследовательских целей**.
