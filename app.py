import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import os
import plotly.express as px

# Configuración de la página
st.set_page_config(
    page_title="Valuador de Autos - Proyecto VII",
    page_icon="🚗",
    layout="wide"
)

# --- 1. CARGA DE DATOS Y MODELOS ---

@st.cache_data
def load_data():
    """Carga y limpia el dataset para usarlo en los filtros y gráficos."""
    # Intentamos cargar el CSV desde la ruta raíz o rutas comunes
    paths = ['used_cars.csv', 'data/used_cars.csv']
    df = None
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            break
            
    if df is None:
        st.error("No se encontró 'used_cars.csv'. Asegúrate de poner app.py en la misma carpeta que el csv.")
        return pd.DataFrame()

    # Limpieza básica de columnas numéricas (igual que en tu notebook)
    def clean_currency(x):
        if isinstance(x, str):
            return float(x.replace('$', '').replace(',', ''))
        return x

    def clean_miles(x):
        if isinstance(x, str):
            return float(x.replace(' mi.', '').replace(',', ''))
        return x

    df['price'] = df['price'].apply(clean_currency)
    df['milage'] = df['milage'].apply(clean_miles)
    
    # Rellenar nulos para evitar errores en los selectbox
    df = df.fillna({
        'clean_title': 'No',
        'accident': 'None reported',
        'fuel_type': 'Unknown',
        'ext_col': 'Unknown',
        'int_col': 'Unknown'
    })
    
    return df

@st.cache_resource
def load_artifacts():
    """Carga los modelos entrenados (.pkl)."""
    # Rutas esperadas
    path_dict = {
        'model': 'models/best_model.pkl',
        'scaler': 'models/scaler.pkl',
        'encoders': 'models/label_encoders.pkl',
        'features': 'models/feature_names.pkl'
    }
    
    artifacts = {}
    try:
        for key, path in path_dict.items():
            if not os.path.exists(path):
                # Intento de corrección si la carpeta models no está directa
                path = os.path.join('ProyectoVIIRegression-main', path)
            
            with open(path, 'rb') as f:
                artifacts[key] = pickle.load(f)
    except Exception as e:
        st.error(f"Error cargando los modelos: {e}. Verifica que la carpeta 'models' exista.")
        return None
        
    return artifacts

# --- 2. FUNCIONES DE INGENIERÍA DE CARACTERÍSTICAS ---

def extract_horsepower(engine_str):
    """Extrae los HP del texto del motor."""
    if pd.isna(engine_str): return np.nan
    match = re.search(r'(\d+\.?\d*)HP', str(engine_str))
    if match:
        return float(match.group(1))
    return 0.0 # Valor por defecto si no encuentra HP

def prepare_input_data(user_inputs, df_full, artifacts):
    """Prepara los datos del usuario para que el modelo pueda leerlos."""
    
    # Crear DataFrame de una sola fila
    input_df = pd.DataFrame([user_inputs])
    
    # 1. Calcular car_age (Antigüedad)
    current_year = 2025 # Ajustable
    input_df['car_age'] = current_year - input_df['model_year']
    
    # 2. Extraer Horsepower
    input_df['horsepower'] = input_df['engine'].apply(extract_horsepower)
    
    # 3. Calcular Brand Frequency (Frecuencia de la marca)
    # Necesitamos calcular esto basado en el dataset original, tal como se entrenó
    brand_counts = df_full['brand'].value_counts(normalize=True)
    brand_freq = brand_counts.get(user_inputs['brand'], 0)
    input_df['brand_frequency'] = brand_freq
    
    # 4. Codificar variables categóricas (Label Encoding)
    encoders = artifacts['encoders']
    cat_cols = ['brand', 'model', 'fuel_type', 'transmission', 
                'ext_col', 'int_col', 'accident', 'clean_title']
    
    for col in cat_cols:
        le = encoders.get(col)
        val = input_df[col].iloc[0]
        if le:
            if val in le.classes_:
                input_df[col] = le.transform([val])[0]
            else:
                # Manejo de valores nuevos no vistos (asignamos el primero o un valor seguro)
                input_df[col] = 0 
                
    # 5. Escalar los datos
    feature_order = artifacts['features'] # El orden importa
    input_df = input_df[feature_order] # Reordenar columnas
    
    scaler = artifacts['scaler']
    scaled_data = scaler.transform(input_df)
    
    return scaled_data

# --- 3. INTERFAZ PRINCIPAL (MAIN) ---

def main():
    st.title("📊 Análisis y Predicción de Precios de Coches")
    st.markdown("Herramienta interactiva para valorar vehículos usados basado en sus características y el mercado actual.")
    
    # Cargar recursos
    df = load_data()
    artifacts = load_artifacts()
    
    if df.empty or artifacts is None:
        return # Detener si falla la carga

    # --- BARRA LATERAL: FILTROS EN CASCADA ---
    st.sidebar.header("🛠️ Configuración del Vehículo")
    st.sidebar.info("Selecciona en orden: Marca -> Modelo -> Año...")

    # 1. Marca
    brands = sorted(df['brand'].unique())
    selected_brand = st.sidebar.selectbox("Marca", brands)
    
    # Filtrar datos para el siguiente paso
    df_step1 = df[df['brand'] == selected_brand]

    # 2. Modelo (Solo modelos de esa marca)
    models = sorted(df_step1['model'].unique())
    selected_model = st.sidebar.selectbox("Modelo", models)
    
    # Filtrar datos
    df_step2 = df_step1[df_step1['model'] == selected_model]

    # 3. Año (Solo años disponibles para ese modelo)
    years = sorted(df_step2['model_year'].unique(), reverse=True)
    selected_year = st.sidebar.selectbox("Año", years)
    
    # Filtrar datos
    df_step3 = df_step2[df_step2['model_year'] == selected_year]

    # 4. Motor (Solo motores vistos en ese año/modelo)
    engines = sorted(df_step3['engine'].unique())
    if len(engines) == 0: engines = ["Unknown"] # Fallback
    selected_engine = st.sidebar.selectbox("Motor", engines)

    # 5. Transmisión
    transmissions = sorted(df_step3['transmission'].unique())
    if len(transmissions) == 0: transmissions = ["Unknown"]
    selected_transmission = st.sidebar.selectbox("Transmisión", transmissions)
    
    st.sidebar.markdown("---")
    st.sidebar.write("**Otros Detalles**")

    # Para colores y combustible, damos un poco más de libertad mostrando opciones del modelo en general
    # para no restringir demasiado si el usuario tiene un coche raro.
    fuels = sorted(df_step2['fuel_type'].unique())
    selected_fuel = st.sidebar.selectbox("Combustible", fuels)
    
    ext_colors = sorted(df_step2['ext_col'].unique())
    selected_ext_col = st.sidebar.selectbox("Color Exterior", ext_colors)
    
    int_colors = sorted(df_step2['int_col'].unique())
    selected_int_col = st.sidebar.selectbox("Color Interior", int_colors)

    # Kilometraje
    avg_mileage = int(df_step3['milage'].mean()) if not df_step3.empty else 50000
    mileage = st.sidebar.number_input("Kilometraje (Millas)", min_value=0, value=avg_mileage, step=5000)

    accident = st.sidebar.selectbox("Accidentes", df['accident'].unique())
    clean_title = st.sidebar.selectbox("Título Limpio", df['clean_title'].unique())

    # --- ÁREA CENTRAL: PREDICCIÓN Y GRÁFICOS ---

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("💰 Predicción de Precio")
        if st.button("Calcular Precio", use_container_width=True, type="primary"):
            user_data = {
                'brand': selected_brand, 'model': selected_model, 'model_year': selected_year,
                'milage': mileage, 'fuel_type': selected_fuel, 'engine': selected_engine,
                'transmission': selected_transmission, 'ext_col': selected_ext_col,
                'int_col': selected_int_col, 'accident': accident, 'clean_title': clean_title
            }
            
            # Procesar y predecir
            try:
                X_pred = prepare_input_data(user_data, df, artifacts)
                predicted_price = artifacts['model'].predict(X_pred)[0]
                
                st.success(f"### ${predicted_price:,.2f}")
                st.caption(f"Valor estimado para un {selected_brand} {selected_model} del {selected_year}.")
            except Exception as e:
                st.error(f"Error en la predicción: {e}")

    with col2:
        st.subheader("📊 Datos de Referencia")
        # Mostrar cuántos coches similares hay en la base de datos
        similar_cars = df_step2 # Mismo modelo
        count = len(similar_cars)
        st.metric("Coches similares en base de datos", count)
        
        avg_price = similar_cars['price'].mean()
        st.metric("Precio Promedio de Mercado", f"${avg_price:,.2f}")

    # --- SECCIÓN GRÁFICA ---
    st.markdown("---")
    st.header(f"Análisis de Mercado: {selected_brand} {selected_model}")
    
    if not similar_cars.empty:
        # Gráfico 1: Relación Precio vs Kilometraje
        # Esto ayuda al usuario a ver dónde cae su coche
        fig_scatter = px.scatter(
            similar_cars, 
            x='milage', 
            y='price',
            color='model_year',
            title=f'Precio vs. Millaje ({selected_brand} {selected_model})',
            labels={'milage': 'Millas', 'price': 'Precio ($)', 'model_year': 'Año'},
            hover_data=['transmission', 'fuel_type']
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Gráfico 2: Distribución de precios por año
        # Agrupamos para ver la tendencia
        avg_price_by_year = similar_cars.groupby('model_year')['price'].mean().reset_index()
        fig_line = px.line(
            avg_price_by_year,
            x='model_year',
            y='price',
            title='Tendencia de Precio Promedio por Año',
            markers=True
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.warning("No hay suficientes datos para generar gráficos de este modelo específico.")

if __name__ == "__main__":
    main()