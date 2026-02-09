import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv('data/bmw.csv')

"""# **Entrenamiento de Modelos**

# Selección de variables
Definimos explícitamente las variables de entrada **(X)** y la variable objetivo **(y)**. Las variables seleccionadas como entrada incluyen características relevantes del vehículo, como el modelo, tipo de combustible, transmisión, tamaño del motor, kilometraje y año de fabricación. La variable objetivo es el precio del vehículo. Esta selección se realiza de forma razonada, descartando columnas que no aportan información directa a la predicción del precio y evitando introducir ruido en el modelo.
"""

X = df[
    [
        "model",
        "fuelType",
        "transmission",
        "engineSize",
        "mileage",
        "year"
    ]
]

y = df["price"]

"""# Preprocesamiento
En esta celda definimos el preprocesamiento de los datos mediante un ***ColumnTransforme***. Las variables categóricas se transforman usando **OneHotEncoding**, lo que permite convertir categorías de texto en variables numéricas binarias que pueden ser utilizadas por los modelos de Machine Learning que vamos a entrenar. Las variables numéricas se mantienen sin modificar. Este paso es fundamental, ya que los algoritmos no pueden trabajar directamente con texto y necesitan datos numéricos estructurados.
"""

cat = [
    "model",
    "fuelType",
    "transmission"
]

num = [
    "engineSize",
    "mileage",
    "year"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", "passthrough", num)
    ]
)

"""# Train / Test Split
En esta celda se realiza la división del dataset en un conjunto de **entrenamiento** y otro de **test**, utilizando un 80 % de los datos para entrenar y un 20 % para evaluar el modelo. Esta separación permite medir la capacidad de generalización del modelo sobre datos no vistos previamente. El uso de una semilla fija **(random_state)** garantiza que los resultados sean reproducibles
"""

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

"""# Función de Evaluación
Definimos una función que calcula las **métricas de evaluación** del modelo: ***MAE***, ***MSE*** y ***R²***. Estas métricas permiten evaluar cuantitativamente el rendimiento del modelo. El ***MAE*** mide el **error medio** en unidades reales, el ***MSE*** penaliza más los **errores grandes** y el ***R²*** indica qué **porcentaje de la variabilidad** del precio es explicado por el modelo.
"""

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }

"""# Entrenamiento de Modelos de Base
En esta celda se **entrenan** y **comparan** tres modelos de **aprendizaje automático** con el objetivo de predecir el precio de vehículos de segunda mano: **Regresión Lineal**, **Árbol de Decisión** y **Random Forest**. Todos los modelos se integran dentro de **pipelines** que incluyen el **preprocesamiento** de los datos, garantizando que las transformaciones se apliquen de forma consistente tanto en entrenamiento como en predicción.

La comparación de estos modelos se realiza mediante métricas cuantitativas como ***MAE***, ***MSE*** y ***R²***, permitiendo seleccionar el modelo más adecuado en función de su **rendimiento** y **estabilidad**.
"""

# Regresión Lineal
lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

lr_pipeline.fit(X_train, y_train)
lr_metrics = evaluate_model(lr_pipeline, X_test, y_test)

# Árbol de Decisión
dt_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeRegressor(random_state=42))
])

dt_pipeline.fit(X_train, y_train)
dt_metrics = evaluate_model(dt_pipeline, X_test, y_test)

# Random Forest
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

rf_pipeline.fit(X_train, y_train)
rf_metrics = evaluate_model(rf_pipeline, X_test, y_test)

"""# Tabla Comparativa
**Análisis de resultados**

A partir de la tabla de métricas se observa una mejora progresiva del rendimiento a medida que se utilizan modelos más complejos. La **Regresión Lineal** presenta los valores más altos de ***MAE*** y ***MSE***, lo que indica que, aunque explica una parte importante de la variabilidad del precio (***R²*** **≈ 0.83**), no es capaz de capturar adecuadamente las relaciones no lineales existentes en los datos. Esto limita su precisión en la predicción de precios reales.

El **Árbol de Decisión** mejora notablemente los resultados respecto a la regresión lineal, reduciendo tanto el ***MAE*** como el ***MSE*** y aumentando el coeficiente ***R²***. Esto indica que el modelo es capaz de adaptarse mejor a la estructura del problema al capturar relaciones más complejas entre las variables. No obstante, su rendimiento sigue siendo inferior al del modelo más avanzado.

El **Random Forest** obtiene los mejores resultados en todas las métricas evaluadas. Presenta el menor ***MAE*** y ***MSE***, lo que implica predicciones más precisas y errores más reducidos, además de un valor de ***R²*** cercano a **0.91**, lo que indica que el modelo explica más del **91 %** de la variabilidad del precio del vehículo. Esto demuestra una mayor capacidad de generalización y estabilidad frente a los otros modelos.
"""

results = pd.DataFrame.from_dict(
    {
        "Linear Regression": lr_metrics,
        "Decision Tree": dt_metrics,
        "Random Forest": rf_metrics
    },
    orient="index"
)

results

"""# Validación Cruzada
Aplicamos **validación cruzada** al modelo **Random Forest**, utilizando varios folds. Este procedimiento permite evaluar la estabilidad del modelo y reducir la dependencia de una única partición del dataset. El resultado es una estimación más robusta del error medio del modelo.

* **El error promedio (*MAE*: 2,114.54)**
  * El ***MAE*** (**Error Absoluto Medio**) nos muestra que, en promedio, las predicciones del modelo se desvían unos **2,114 €** del precio real de los coches.

* **La consistencia (*Desviación Estándar*: 340.50)**

  * Este valor es clave para la confianza ya que indica cuánto varía el error dependiendo de los datos que le des.El error suele oscilar entre los **1,774 €** y los **2,455 €** . Como la **desviación estándar** es relativamente pequeña en comparación con el ***MAE*** (representa un **16%** del error), podemos decir que el modelo es estable. No cambia drásticamente su comportamiento cuando ve diferentes grupos de datos.
"""

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    rf_pipeline,
    X,
    y,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

print("MAE medio (CV):", -cv_scores.mean())
print("Desviación estándar:", cv_scores.std())

"""# Ajuste de hiperparámetros
Para realizar el **ajuste de hiperparámetros** utilizamos **GridSearchCV** y **RandomizedSearchCV** para ajustar los **hiperparámetros** del modelo **Random Forest**. Este proceso busca automáticamente la combinación de parámetros que minimiza el error, mejorando el rendimiento del modelo y controlando su complejidad.

**GridSearchCV**

Los resultados nos muestra unos resultados muy claros sobre cómo está aprendiendo el modelo. Hemos logrado reducir el error de **2,114** a aproximadamente **1,921**, lo cual es una mejora muy positiva
"""

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 10],
    "model__min_samples_leaf": [1, 5]
}

grid_search = GridSearchCV(
    rf_pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print("Mejores parámetros (GridSearch):")
print(grid_search.best_params_)

print("Mejor MAE (GridSearch):", -grid_search.best_score_)

from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    "model__n_estimators": [100, 200, 300, 400],
    "model__max_depth": [None, 10, 20, 30],
    "model__min_samples_split": [2, 5, 10, 20],
    "model__min_samples_leaf": [1, 2, 5, 10]
}

random_search = RandomizedSearchCV(
    rf_pipeline,
    param_distributions=param_dist,
    n_iter=15,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    random_state=42,
    verbose=2
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

print("Mejores parámetros (RandomizedSearch):")
print(random_search.best_params_)

print("Mejor MAE (RandomizedSearch):", -random_search.best_score_)

"""# Gráfica Real vs Predicho
El modelo muestra un buen desempeño, ya que la mayoría de los puntos se concentran cerca de la línea diagonal, indicando que las predicciones están razonablemente alineadas con los valores reales.
"""

y_pred = best_model.predict(X_test)

plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)
plt.xlabel("Precio real")
plt.ylabel("Precio predicho")
plt.title("Precio real vs Precio predicho")
plt.show()

"""# Exportación del modelo"""

joblib.dump(best_model, "modelo_precio_bmw.pkl")