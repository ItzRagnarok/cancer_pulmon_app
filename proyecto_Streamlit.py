import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración inicial
st.set_page_config(page_title="Análisis de Cáncer de Pulmón", layout="wide")

st.title("🩺 Predicción de Supervivencia - Cáncer de Pulmón")

# Cargar los datos
st.header("📂 Cargar y explorar los datos")
uploaded_file = st.file_uploader("Sube tu propio archivo CSV (Opcional)", type="csv")

df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("💾 ¡Archivo subido correctamente por el usuario!")
else:
    # INTENTA CARGAR EL ARCHIVO POR DEFECTO DEL REPOSITORIO
    # Cambia "cancer_data.csv" por el nombre exacto de tu archivo CSV
    nombre_archivo_defecto = "Lung_Cancer_Trends_Realistic.csv" 
    try:
        df = pd.read_csv(nombre_archivo_defecto)
        st.info(f"📦 Cargando automáticamente el dataset por defecto (`{nombre_archivo_defecto}`) del repositorio.")
    except FileNotFoundError:
        st.error(f"❌ No se encontró el archivo `{nombre_archivo_defecto}` en el repositorio. Por favor, sube un CSV manualmente.")

# Si el dataframe se ha cargado (ya sea por el usuario o por defecto)
if df is not None:
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

    # Entrenar modelo
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

    # Visualizaciones organizadas
    st.header("📊 Visualizaciones")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🟤 Supervivencia por Estado de Fumador")
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        sns.countplot(data=df, x='Smoking_Status', hue='Survival_Status', ax=ax1, palette='Set2')
        ax1.set_title('Fumador vs Supervivencia')
        ax1.set_xlabel('Estado de Fumador')
        ax1.set_ylabel('Cantidad')
        ax1.legend(title='Supervivencia')
        st.pyplot(fig1)

    with col2:
        st.markdown("#### 🔢 Matriz de Confusión")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_test, prediccion), annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_xlabel("Predicción")
        ax2.set_ylabel("Real")
        ax2.set_title("Matriz de Confusión")
        st.pyplot(fig2)

    # Reporte
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

        for col in cat_cols:
            le = LabelEncoder()
            le.fit(df[col])
            input_df[col] = le.transform([input_data[col]])

        input_df = input_df[X.columns]

        pred = model.predict(input_df)[0]
        st.success(f"🎯 Predicción: {'Sobrevivió' if pred == 1 else 'No sobrevivió'}")
