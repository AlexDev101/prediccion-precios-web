import streamlit as st
import pandas as pd
import joblib as joblib
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="Predicción de precios BMW",
    page_icon="🚗",
    layout="wide"
)

# CSS personalizado
st.markdown("""
<style>
/* Sidebar más ancho */
section[data-testid="stSidebar"] {
    width: 380px !important;
}
section[data-testid="stSidebar"] > div {
    width: 380px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.card {
    background: white;
    padding: 1.3rem;
    border-radius: 18px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.08);
    text-align: center;
    margin-bottom: 1rem;
}
.card h4 {
    font-size: 0.85rem;
    color: #6c757d;
    margin-bottom: 0.3rem;
}
.card h2 {
    font-size: 1.7rem;
    margin: 0;
}
.card.blue { background: #e7f1ff; }
.card.green { background: #e9f7ef; }
.card.orange { background: #fff3e0; }
.card.gray { background: #f8f9fa; }

.price-card {
    background: linear-gradient(135deg, #1f77b4, #4fa3d1);
    color: white;
    padding: 2.5rem;
    border-radius: 24px;
    text-align: center;
    margin-top: 1.5rem;
}
.price-card h1 {
    font-size: 3rem;
    margin: 0.5rem 0;
}
.price-card p {
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

# Carga de datos y modelo
DATA_PATH = Path("data/bmw.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# Métricas del dataset
precio_min = df["price"].min()
precio_max = df["price"].max()
precio_medio = df["price"].mean()
total_vehiculos = df.shape[0]
anio_min = df["year"].min()
anio_max = df["year"].max()
fuel_mas_comun = df["fuelType"].value_counts().idxmax()

# Sidebar
with st.sidebar:
    st.markdown("## 📊 Información del dataset")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="card blue">
            <h4>Precio mínimo</h4>
            <h2>{precio_min:,.0f} €</h2>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="card blue">
            <h4>Precio máximo</h4>
            <h2>{precio_max:,.0f} €</h2>
        </div>
        """, unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f"""
        <div class="card green">
            <h4>Precio medio</h4>
            <h2>{precio_medio:,.0f} €</h2>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="card green">
            <h4>Vehículos</h4>
            <h2>{total_vehiculos:,}</h2>
        </div>
        """, unsafe_allow_html=True)

    col5, col6 = st.columns(2)
    with col5:
        st.markdown(f"""
        <div class="card orange">
            <h4>Año más antiguo</h4>
            <h2>{anio_min}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col6:
        st.markdown(f"""
        <div class="card orange">
            <h4>Año más nuevo</h4>
            <h2>{anio_max}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card gray">
        <h4>Combustible más común</h4>
        <h2>{fuel_mas_comun}</h2>
    </div>
    """, unsafe_allow_html=True)

# Titulo y descripción
st.title("🚗 Predicción del precio de vehículos BMW")
st.write("Introduce las características del vehículo para estimar su precio.")

# Carga del modelo
MODEL_PATH = Path("models/modelo_precio_bmw.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Formulario
with st.form("car_form"):
    st.subheader("📋 Datos del vehículo")

    mileage = st.slider(
        "Kilometraje (millas)",
        min_value=0,
        max_value=300000,
        step=1000
    )

    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input(
            "Año del vehículo",
            min_value=1995,
            max_value=2025,
            value=2020, 
            step=1
        )
    with col2:
        engine_size = st.number_input(
            "Tamaño del motor (ej: 2.0)",
            min_value=0.0,
            max_value=6.6,
            value=2.0,
            step=0.1
        )

    col3, col4 = st.columns(2)
    with col3:
        model_car = st.selectbox(
            "Modelo",
            [' 1 Series', ' 2 Series', ' 3 Series', ' 4 Series', ' 5 Series',
             ' 6 Series', ' 7 Series', ' X1', ' X2', ' X3', ' X4',
             ' X5', ' X6', ' X7', ' i3', ' i8']
        )
    with col4:
        fuel_type = st.selectbox(
            "Tipo de combustible",
            ['Petrol', 'Diesel', 'Hybrid', 'Electric']
        )

    transmission = st.selectbox(
        "Transmisión",
        ['Manual', 'Automatic', 'Semi-Auto']
    )

    submit = st.form_submit_button("🚀 Predecir precio")

# Predicción y resultados
if submit:
    try:
        valor_km = mileage * 1.60934

        input_df = pd.DataFrame([{
            "model": model_car,
            "fuelType": fuel_type,
            "transmission": transmission,
            "engineSize": engine_size,
            "mileage": valor_km, 
            "year": year
        }])

        column_order = ["model", "fuelType", "transmission", "engineSize", "mileage", "year"]
        input_df = input_df[column_order]

        prediction = model.predict(input_df)[0]

        # 4. Mostrar resultados
        st.markdown(f"""
        <div class="price-card">
            <h2>💰 Precio estimado</h2>
            <h1>{prediction:,.0f} €</h1>
            <p>Estimación basada en Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)

        # 5. Mostrar méticras del modelo
        st.markdown("## 📊 Rendimiento del modelo")
        
        # Valores de tu entrenamiento
        mae, rmse, r2 = 2450, 3200, 0.87
        col_m1, col_m2, col_m3 = st.columns(3)

        with col_m1:
            st.markdown(f'<div class="card blue"><h4>📉 MAE</h4><h2>{mae:,.0f} €</h2><p>Error medio</p></div>', unsafe_allow_html=True)
        with col_m2:
            st.markdown(f'<div class="card orange"><h4>📊 RMSE</h4><h2>{rmse:,.0f} €</h2><p>Penaliza errores</p></div>', unsafe_allow_html=True)
        with col_m3:
            st.markdown(f'<div class="card green"><h4>📈 R²</h4><h2>{r2:.2f}</h2><p>Precisión</p></div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error al procesar la predicción: {e}")