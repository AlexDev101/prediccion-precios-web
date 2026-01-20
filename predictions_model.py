import pandas as pd
import joblib

# 1. Cargamos el modelo
model = joblib.load('models/modelo_precio_bmw.pkl')

print("Introduce los datos del vehículo:\n")

# 2. Pedimos los datos dek modelo al usuario
model_car = input("Modelo (ej: 3 Series, X5, 1 Series): ")
fuel_type = input("Tipo de combustible (Petrol / Diesel / Hybrid / Electric): ")
transmission = input("Transmisión (Manual / Automatic / Semi-Auto): ")

mileage_miles = float(input("Kilometraje en millas: "))
year = int(input("Año del vehículo: "))

# Conversión a km (IGUAL que en el entrenamiento)
km = mileage_miles * 1.60934

# 3.Creamos el Dataframe 
user_data = pd.DataFrame([{
    'model': model_car,
    'fuelType': fuel_type,
    'transmission': transmission,
    'km': km,
    'year': year
}])


# 4. Predicción
prediction = model.predict(user_data)

print("\n💰 Precio estimado del vehículo:")
print(f"{prediction[0]:.2f} €")
