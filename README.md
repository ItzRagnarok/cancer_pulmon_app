# 🩺 Predicción de Supervivencia - Cáncer de Pulmón

Aplicación interactiva de Machine Learning diseñada para predecir la probabilidad de supervivencia de pacientes con cáncer de pulmón. El modelo analiza diversos factores de riesgo como el historial de tabaquismo, la exposición pasiva al humo, la edad y el nivel de contaminación, utilizando algoritmos de clasificación.

🚀 **[Prueba la aplicación en vivo aquí](https://cancerpulmonapp.streamlit.app/)**

## 🛠️ Stack Tecnológico

*   **Lenguaje:** Python
*   **Interfaz y Despliegue:** Streamlit
*   **Machine Learning:** Scikit-Learn (Random Forest Classifier)
*   **Procesamiento de Datos:** Pandas
*   **Visualización:** Seaborn, Matplotlib

## 📊 Características y Funcionalidades

*   **Análisis Exploratorio Automatizado:** Permite cargar un archivo CSV con datos de pacientes y visualizar instantáneamente una muestra limpia y filtrada.
*   **Preprocesamiento de Datos:** Imputación automática de valores nulos y codificación de variables categóricas mediante `LabelEncoder`.
*   **Entrenamiento del Modelo:** Utiliza un clasificador *Random Forest* entrenado en tiempo real con partición de datos (80% entrenamiento / 20% prueba).
*   **Métricas de Evaluación:** Muestra la precisión del modelo, el reporte de clasificación y una matriz de confusión para evaluar el rendimiento de las predicciones.
*   **Visualización de Importancia:** Gráfico de barras que destaca qué características (ej. años fumando, edad) tienen mayor peso en la decisión del algoritmo.
*   **Predicción Interactiva:** Panel manual donde el usuario puede introducir parámetros personalizados y obtener una predicción instantánea de supervivencia.

## 📸 Interfaz de la Aplicación

*Añade aquí un par de capturas de tu aplicación funcionando en Streamlit.*
![Panel de Carga y Evaluación](https://github.com/user-attachments/assets/ce9064e8-2370-4cd2-8ef1-74cfa429504a)
![Visualizaciones](https://github.com/user-attachments/assets/a4d0bab7-112e-4b4b-89c2-64da44ad4ec0)
![Panel de Predicción Manual](https://github.com/user-attachments/assets/1786546f-ed75-4589-a816-b2139aac5b5d)

## 🚀 Instalación y Despliegue en Local

Si deseas ejecutar este proyecto en tu entorno local, sigue estos pasos:

### Prerrequisitos
*   Tener Python 3.8 o superior instalado.
*   Se recomienda el uso de un entorno virtual (venv o conda).

### Pasos de ejecución
1. Clonar el repositorio:
   `git clone [tu-enlace-de-github]`
2. Instalar las dependencias necesarias:
   `pip install pandas scikit-learn streamlit seaborn matplotlib`
3. Ejecutar la aplicación de Streamlit:
   `streamlit run app.py`
4. La aplicación se abrirá automáticamente en tu navegador web en `http://localhost:8501`.
