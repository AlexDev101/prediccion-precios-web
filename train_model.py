# train_model.py
# ============================================
# ENTRENAMIENTO DEL MODELO ML (VS CODE)
# Compatible con Python 3.11 + Streamlit
# ============================================

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

# ============================================
# 1. CARGA DE DATOS
# ============================================
DATA_PATH = Path('data/bmw.csv')
MODEL_PATH = Path('models/modelo_precio_bmw.pkl')

MODEL_PATH.parent.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

# ============================================
# 2. PREPROCESAMIENTO
# ============================================
# Conversión de millas a kilómetros

df['km'] = df['mileage'] * 1.60934

# Variable objetivo
y = df['price']

# Variables predictoras
X = df[['model', 'fuelType', 'transmission', 'km', 'year']]

categorical_features = ['model', 'fuelType', 'transmission']
numerical_features = ['km', 'year']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ]
)

# ============================================
# 3. TRAIN / TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# 4. COMPARACIÓN DE MODELOS
# ============================================
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

results = []

for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    results.append([
        name,
        mean_absolute_error(y_test, preds),
        mean_squared_error(y_test, preds),
        r2_score(y_test, preds)
    ])

results_df = pd.DataFrame(results, columns=['Modelo', 'MAE', 'MSE', 'R2'])
print('\nRESULTADOS DE LOS MODELOS:')
print(results_df)

# ============================================
# 5. AJUSTE DE HIPERPARÁMETROS (RANDOM FOREST)
# ============================================
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 10, 20]
}

grid = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=3,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# ============================================
# 6. EVALUACIÓN FINAL
# ============================================
final_preds = best_model.predict(X_test)

print('\nMODELO FINAL (Random Forest Ajustado)')
print('MAE:', mean_absolute_error(y_test, final_preds))
print('MSE:', mean_squared_error(y_test, final_preds))
print('R2:', r2_score(y_test, final_preds))

# ============================================
# 7. EXPORTACIÓN DEL MODELO
# ============================================
joblib.dump(best_model, MODEL_PATH)
print(f'\nModelo guardado en: {MODEL_PATH}')
