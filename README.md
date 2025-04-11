# 🛍️ Online Shoppers Purchasing Intention - MLOps Project with Amazon SageMaker

Este proyecto implementa un **pipeline completo de machine learning automatizado** utilizando Amazon SageMaker, para predecir la **intención de compra** de usuarios en línea con el dataset [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset).

---

## 📦 Estructura del Proyecto

```
.
├── code/
│   ├── preprocess.py                  # Preprocesamiento y balanceo con SMOTE
│   ├── train.py                       # Entrenamiento del modelo XGBoost
│   ├── evaluate.py                    # Evaluación de métricas, matriz de confusión y predicciones
├── sagemaker_pipeline.py              # Definición del pipeline completo de SageMaker
├── mlops-online-shopping.ipynb        # Notebook de ejecución y análisis
└── README.md                         
```

---

## 🔄 Flujo del pipeline (MLOps)

1. **Preprocesamiento de datos (`preprocess.py`)**
   - Limpieza, codificación, escalado
   - Balanceo con **SMOTE**
   - División en `train`, `validation` y `test`

2. **Entrenamiento con XGBoost (`train.py`)**
   - Modelo binario con métricas: `auc`, `logloss`, `error`
   - Ajuste automático de hiperparámetros con `HyperparameterTuner`

3. **Evaluación del modelo (`evaluate.py`)**
   - Métricas: `accuracy`, `precision`, `recall`, `f1-score`
   - Matriz de confusión y archivo de predicciones
   - Resultados guardados en `/opt/ml/processing/evaluation`

---

## 📈 Resultados

- **Accuracy**: 88.8%
- **Precision**: 62.6%
- **Recall**: 62.0%
- **F1 Score**: 62.3%

**Matriz de confusión:**

|               | Pred. Neg | Pred. Pos |
|---------------|-----------|-----------|
| Real Negativo |   981     |   68      |
| Real Positivo |   70      |   114     |

---

## 💡 Tecnologías utilizadas

- 🧠 **XGBoost** como modelo de clasificación
- 🧪 **SMOTE** para balancear clases
- 🔁 **Amazon SageMaker Pipelines**
- 🗂️ **S3** como repositorio de datos
- 📊 **Scikit-learn** para métricas y evaluación
- 🐍 **Python** 3.8+

---

## 🚀 Cómo ejecutar el pipeline

1. Sube los archivos a tu instancia de SageMaker Studio.
2. Ejecuta el archivo `sagemaker_pipeline.py` o el notebook `trabajo-final.ipynb`.
3. Monitorea el estado del pipeline y evalúa los resultados.

---

## 📁 Salidas importantes

- `evaluation.json`: métricas principales
- `confusion_matrix.csv`: matriz de confusión
- `predictions.csv`: predicciones individuales

---

## 🧠 Autoras

- **Angie Rodríguez Gómez**
- **Daniela Benjumea Restrepo**
- **Natalia Vélez Vanegas**

**Profesor:** María Camila Durango Barrera  
**Materia:** Aprendizaje Automático en la Nube

---

## 📌 Notas adicionales

- Este proyecto se desarrolló como parte de un ejercicio de MLOps con Amazon SageMaker.
- El pipeline es completamente reproducible y escalable.

