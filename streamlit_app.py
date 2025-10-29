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
)


# --- 1. Анализ и визуализация частоты ---
st.header("📊 Анализ частоты землетрясений")

col1, col2 = st.columns(2)

# Месячный график
with col1:
    st.subheader("Ежемесячная частота")
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


# --- 2. Модель классификации (Обучение) ---
st.header("🤖 Модель классификации (Random Forest)")

# Целевая переменная: 1, если магнитуда >= 4.0 (высокий риск), иначе 0
data['Risk'] = (data['Magnitude'] >= 4.0).astype(int)
y = data['Risk']

# Признаки: Latitude, Longitude, Depth
X = data[['Latitude', 'Longitude', 'Depth']] 
X = X.fillna(0) 

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Обучение модели
st.markdown("**Обучение модели...**")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
st.success("Модель RandomForest обучена!")


# --- 🔮 НОВЫЙ БЛОК: Интерактивное предсказание риска ---
st.header("🔮 Интерактивное предсказание риска")
st.markdown("Введите параметры, чтобы получить оценку риска землетрясения (Риск **1** = Магнитуда **>= 4.0**):")

with st.form("prediction_form"):
    # Поля ввода с разумными значениями по умолчанию для Центральной Азии
    col_lat, col_lon, col_depth = st.columns(3)
    
    with col_lat:
        # Типичный диапазон широты для региона
        input_lat = st.number_input("Широта (Latitude)", value=38.0, min_value=30.0, max_value=45.0, format="%.4f")
    
    with col_lon:
        # Типичный диапазон долготы для региона
        input_lon = st.number_input("Долгота (Longitude)", value=71.0, min_value=60.0, max_value=85.0, format="%.4f")
        
    with col_depth:
        # Глубина в км
        input_depth = st.number_input("Глубина (Depth, км)", value=10.0, min_value=0.0, format="%.2f")

    submitted = st.form_submit_button("Сделать предсказание")

    if submitted:
        # Создание DataFrame для предсказания (должен соответствовать X_train)
        input_data = pd.DataFrame({
            'Latitude': [input_lat],
            'Longitude': [input_lon],
            'Depth': [input_depth]
        })
        
        # Предсказание класса и вероятности
        prediction_class = model.predict(input_data)[0]
        # Вероятность класса 1
        prediction_proba = model.predict_proba(input_data)[0, 1]
        
        st.subheader("Результат предсказания:")
        
        # Отображение класса риска
        if prediction_class == 1:
            st.error(f"Класс риска: **1 (Высокий риск)**")
            st.markdown("Модель предсказывает, что это землетрясение, **скорее всего, будет иметь магнитуду 4.0 или выше.**")
        else:
            st.success(f"Класс риска: **0 (Низкий риск)**")
            st.markdown("Модель предсказывает, что это землетрясение, **скорее всего, будет иметь магнитуду ниже 4.0.**")
            
        # Отображение вероятности
        st.info(f"Вероятность класса 1 (Высокий риск): **{prediction_proba:.4f}**")
# -------------------------------------------------------------------------


# --- Результаты модели (Оценка) ---
st.header("🎯 Оценка производительности модели")

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
       pip install -r requirements.txt
       ```
    4. **Запустите** приложение из терминала:
       ```bash
       streamlit run app.py
       ```
    """
)
