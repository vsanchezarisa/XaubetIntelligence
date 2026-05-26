# 🏋️‍♂️ Xaubet Intelligence: Sistema Predictivo de Retención y GenAI

## 📌 Descripción del Proyecto
Este proyecto es el trabajo final del curso de IA y Big Data. Implementa una solución integral para un centro deportivo con el objetivo de anticipar las posibles bajas de socios (Churn Prediction) a partir de datos demográficos y de comportamiento, y automatizar estrategias de retención personalizadas. 

El valor diferencial de esta arquitectura es la integración de un **modelo predictivo (XGBoost)** con un **motor de IA Generativa en local (Ollama - Gemma 2)** a través de una aplicación web interactiva. Esto permite generar campañas persuasivas de rescate garantizando al 100% la privacidad de los datos de los clientes, ya que ninguna información sensible abandona el equipo.

## ⚙️ Arquitectura de la Solución (Pipeline)

El proyecto está estructurado en 5 fases analíticas y una fase de despliegue:

1. **ETL y Unificación (Notebooks 01 & 02):** Integración de registros históricos de altas/bajas y extracción de logs masivos de accesos diarios mediante `pandas`.
2. **Feature Engineering (Notebook 03):** Creación de variables de comportamiento clave (frecuencia, recencia, distancia al centro, día favorito) y consolidación de la Tabla Base Analítica (ABT).
3. **Machine Learning Predictivo (Notebook 04):** Entrenamiento, validación y exportación de un modelo `XGBoost` optimizado para detectar usuarios en riesgo crítico de abandono.
4. **IA Generativa Prescriptiva (Notebook 05):** Integración con el LLM `gemma2:9b` vía Ollama para redactar correos de retención altamente personalizados según la trayectoria de cada socio.
5. **Despliegue e Interfaz Web (`app_retencion.py`):** Plataforma interactiva en Streamlit con tres módulos: simulador de riesgo individual, análisis de bases de datos masivas por lotes y automatizador masivo de emails de rescate.

## 🛠️ Stack Tecnológico
* **Lenguaje:** Python 3.10+
* **Procesamiento y Análisis:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost, Joblib
* **IA Generativa (Local):** Ollama, framework OpenAI (Python), Gemma 2 (9B)
* **Frontend / UI:** Streamlit, Plotly (Gráficos interactivos)
* **Base de Datos & BI:** SQLite, exportaciones en CSV/Excel para Power BI

## 📂 Estructura del Repositorio
```text
├── app/
│   ├── app_retencion.py
│   ├── requirements.txt
│   └── modelo_xgboost.pkl
├── datos_pruebas/
│   ├── prueba_carga_usuarios.csv
├── notebooks/
│   ├── 01_ETL_Unificacion_Datos.ipynb
│   ├── 02_ETL_Maestro_Socios.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Preparacion_y_Modelado.ipynb
│   └── 05_IA_Generativa_Ollama.ipynb
├── .gitignore
└── README.md
```

## 🚀 Cómo ejecutar este proyecto en local
Para garantizar la reproducibilidad y no sobrecargar la memoria RAM del sistema, la aplicación está configurada para ejecutarse dentro de un Entorno Virtual de Python.

### Requisitos previos:
* Tener Python instalado en el sistema.
* Tener instalado y ejecutándose Ollama con el modelo descargado (ejecutar en terminal: `ollama run gemma2:9b`).

### Pasos de instalación:

1. **Clonar el repositorio y entrar en la carpeta de la app:**
   ```bash
   git clone https://github.com/vsanchezarisa/XaubetIntelligence.git
   cd XaubetIntelligence
   ```

2. **Crear el entorno virtual:**
   ```bash
   python -m venv venv
   ```

3. **Activar el entorno virtual:**
   * En Windows (CMD/PowerShell):
     ```bash
     venv\Scripts\activate
     ```
   * En Mac/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Instalar las dependencias:**
   ```bash
   pip install -r app/requirements.txt
   ```

5. **Ejecutar la plataforma:**
   ```bash
   streamlit run app/app_retencion.py
   ```

Se abrirá automáticamente una pestaña en tu navegador web (típicamente en `http://localhost:8501`) con la aplicación funcionando.

---
*Proyecto desarrollado como Trabajo Final del curso de IA y Big Data.*
