import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración inicial
st.set_page_config(page_title="Análisis de Cáncer de Pulmón", layout="wide")

st.title("🩺 Predicción de Supervivencia - Cáncer de Pulmón")

# Cargar los datos
st.header("📂 Cargar y explorar los datos")
uploaded_file = st.file_uploader("Sube el archivo CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    columnas_utiles = [
        'Age', 'Gender', 'Smoking_Status', 'Years_Smoking', 'Cigarettes_Per_Day',
        'Secondhand_Smoke_Exposure', 'Air_Pollution_Level', 'Family_History',
        'BMI', 'Physical_Activity_Level', 'Alcohol_Consumption',
        'Chronic_Lung_Disease', 'Survival_Status'
    ]

    df_filtrado = df[columnas_utiles].copy()

    st.subheader("Primeras filas del dataset filtrado")
    st.dataframe(df_filtrado.head())

    # Limpieza de datos
    df_filtrado['Alcohol_Consumption'] = df_filtrado['Alcohol_Consumption'].fillna('Unknown')

    # Codificación de variables categóricas
    cat_cols = [col for col in df_filtrado.select_dtypes(include='object').columns if col != 'Survival_Status']
    for col in cat_cols:
        df_filtrado[col] = LabelEncoder().fit_transform(df_filtrado[col])

    # Separar características y objetivo
    X = df_filtrado.drop('Survival_Status', axis=1)
    y = df_filtrado['Survival_Status']

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Grid Search
    # with st.spinner("Buscando los mejores hiperparámetros..."):
    #     param_grid = {
    #         'n_estimators': [50, 100],
    #         'max_depth': [None, 10]
    #     }
    #     grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
    #     grid.fit(X_train, y_train)
    #     best_params = grid.best_params_

    # st.success(f"🏆 Mejores parámetros encontrados: {best_params}")

    # Entrenar modelo final
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Predicción
    prediccion = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediccion)

    st.header("📊 Evaluación del modelo")
    st.write(f"**Precisión del modelo:** {accuracy:.2f}")

    st.header("📌 Importancia de las características")

    importances = model.feature_importances_
    features = X.columns
    feature_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_df = feature_df.sort_values(by='Importance', ascending=False)

    st.bar_chart(feature_df.set_index('Feature'))

    # Supervivencia por Smoking_Status
    st.subheader("🟤 Smoking Status")

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='Smoking_Status', hue='Survival_Status', ax=ax1, palette='Set2')
    ax1.set_title('Supervivencia por Estado de Fumador')
    ax1.set_xlabel('Smoking Status')
    ax1.set_ylabel('Número de personas')
    ax1.legend(title='Survival Status')
    st.pyplot(fig1)


    st.subheader("🔢 Matriz de Confusión")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, prediccion), annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor Real")
    st.pyplot(fig)

    st.subheader("🧾 Reporte de Clasificación")
    st.text(classification_report(y_test, prediccion))

    # Predicción individual
    st.header("🔍 Prueba una predicción manual")
    input_data = {}

    for col in X.columns:
        if col in cat_cols:
            opciones = df[col].dropna().unique().tolist()
            input_data[col] = st.selectbox(f"{col}", opciones)
        else:
            input_data[col] = st.number_input(f"{col}", value=float(df[col].mean()))

    if st.button("Predecir supervivencia"):
        input_df = pd.DataFrame([input_data])

        # Codificar variables categóricas
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(df[col])  # Ajustamos usando el df original
            input_df[col] = le.transform([input_data[col]])  # Transformamos directamente desde input_data

        # Reordenar columnas como en X
        input_df = input_df[X.columns]

        pred = model.predict(input_df)[0]
        st.success(f"🎯 Predicción: {'Sobrevivió' if pred == 1 else 'No sobrevivió'}")



else:
    st.info("Por favor, sube un archivo CSV para comenzar.")
