import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# --- Настройка Streamlit страницы ---
st.set_page_config(layout="wide", page_title="Анализ землетрясений в Центральной Азии")
st.title("🌍 Анализ и прогнозирование землетрясений в Центральной Азии")

# --- Загрузка и подготовка данных ---
# В Streamlit, для простоты, предполагаем, что файл находится в той же папке
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    # Убедимся, что все числовые столбцы имеют правильный тип
    data['Magnitude'] = pd.to_numeric(data['Magnitude'], errors='coerce')
    data['Depth'] = pd.to_numeric(data['Depth'], errors='coerce')
    data = data.set_index('Date')
    return data

DATA_FILE = 'Central-Asian-earthquake-dataset.csv'
try:
    data = load_data(DATA_FILE)
    df = data.copy()
except FileNotFoundError:
    st.error(f"Файл данных '{DATA_FILE}' не найден. Пожалуйста, убедитесь, что он находится в той же директории, что и app.py.")
    st.stop()


# Дополнительные поля для DataFrame
df['Описание'] = (
    'Magnitude:' + df['Magnitude'].astype(str) +
    '<br>Depth: ' + df['Depth'].astype(str) + ' км' 
    # Время и Дата были исключены, так как они не являются частью текущего индекса
)


# --- 1. Анализ и визуализация частоты ---
st.header("📊 Анализ частоты землетрясений")

col1, col2 = st.columns(2)

# Месячный график
with col1:
    st.subheader("Ежемесячная частота")
    # Используем df.index (который является 'Date')
    df['month_year'] = df.index.to_period('M')
    monthly_counts = df.groupby('month_year').size().reset_index(name='count')
    monthly_counts['month_year'] = monthly_counts['month_year'].dt.to_timestamp()
    monthly_counts = monthly_counts.set_index('month_year')

    fig_month, ax_month = plt.subplots(figsize=(10, 5))
    ax_month.plot(monthly_counts.index, monthly_counts['count'])
    ax_month.set_title("Ежемесячная частота землетрясений с течением времени")
    ax_month.set_xlabel("Дата")
    ax_month.set_ylabel("Количество землетрясений")
    ax_month.grid(True, linestyle='--')
    st.pyplot(fig_month)

# Ежегодный график
with col2:
    st.subheader("Ежегодная частота")
    df['year'] = df.index.to_period('Y') 
    yearly_counts = df.groupby('year').size().reset_index(name='count')
    yearly_counts['year'] = yearly_counts['year'].dt.to_timestamp()
    yearly_counts = yearly_counts.set_index('year')

    fig_year, ax_year = plt.subplots(figsize=(10, 5))
    ax_year.plot(yearly_counts.index, yearly_counts['count'])
    ax_year.set_title("Ежегодная частота землетрясений с течением времени")
    ax_year.set_xlabel("Год")
    ax_year.set_ylabel("Общее количество землетрясений")
    ax_year.grid(True, linestyle='--')
    st.pyplot(fig_year)


# --- Анализ по странам и описательная статистика ---
st.header("📜 Описательная статистика и анализ по странам")

col3, col4 = st.columns([1, 2])

with col3:
    st.subheader("Статистика данных")
    st.dataframe(data.describe())

with col4:
    st.subheader("Распределение по странам")
    earthquake_counts_by_country = data['Country'].value_counts().reset_index()
    earthquake_counts_by_country.columns = ['Country', 'count']
    st.dataframe(earthquake_counts_by_country)

    fig_country, ax_country = plt.subplots(figsize=(12, 6))
    ax_country.bar(earthquake_counts_by_country['Country'], earthquake_counts_by_country['count'])
    ax_country.set_title('Частота землетрясений по странам')
    ax_country.set_xlabel('Страна')
    ax_country.set_ylabel('Количество землетрясений')
    ax_country.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig_country)


# --- 2. Модель классификации ---
st.header("🤖 Модель классификации (Random Forest)")

# Целевая переменная: 1, если магнитуда >= 4.0 (высокий риск), иначе 0
data['Risk'] = (data['Magnitude'] >= 4.0).astype(int)
y = data['Risk']

# Признаки: Latitude, Longitude, Depth
X = data[['Latitude', 'Longitude', 'Depth']] 
X = X.fillna(0) # Замена NaN для простоты, в реальном проекте нужна более сложная импутация

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Обучение модели
st.markdown("**Обучение модели...**")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
st.success("Модель RandomForest обучена!")


# --- Результаты модели ---
st.subheader("Результаты предсказания")

# Предсказание классов
y_pred = model.predict(X_test)
st.write(f"**Точность (Accuracy):** {accuracy_score(y_test, y_pred):.4f}")

# Предсказание вероятности
y_pred_proba = model.predict_proba(X_test)[:, 1] 

# Расчет ROC AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
st.write(f"**ROC AUC Score:** {roc_auc:.4f}")

st.markdown("---")
st.subheader("Детальный отчет по классификации")
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(2)

# Отображаем отчет в виде таблицы Streamlit
st.dataframe(report_df)

st.markdown("---")
st.subheader("Пример предсказания вероятности (Класс 1: Высокий риск)")
# Создаем DataFrame для удобного отображения
predictions_df = pd.DataFrame({
    'Фактический риск': y_test,
    'Предсказанный класс': y_pred,
    'Вероятность (Риск 1)': y_pred_proba.round(4)
})

st.dataframe(predictions_df.head(10))

# --- Инструкция по запуску ---
st.sidebar.title("Как запустить приложение")
st.sidebar.info(
    """
    1. **Сохраните** этот код в файл под названием `app.py`.
    2. **Поместите** файл данных `Central-Asian-earthquake-dataset.csv` в ту же папку.
    3. **Установите** необходимые библиотеки:
       ```bash
       pip install streamlit pandas matplotlib scikit-learn
       ```
    4. **Запустите** приложение из терминала:
       ```bash
       streamlit run app.py
       ```
    """
)
