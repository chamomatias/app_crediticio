import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo entrenado
modelo = joblib.load('modelo_crediticio.joblib')

# Título
st.title("🧠 Predicción de Incumplimiento Crediticio")

st.write("Ingresá los datos del cliente para predecir si podría incumplir el crédito.")

# Inputs del usuario
antiguedad_empleo = st.number_input("Antigüedad en empleo (años)", min_value=0.0, max_value=40.0, value=5.0)
tasa_interes = st.number_input("Tasa de interés (%)", min_value=0.0, max_value=50.0, value=12.0)
porcentaje_ingreso = st.slider("Porcentaje de ingreso destinado al préstamo", 0.0, 0.5, 0.1)
edad = st.slider("Edad", 18, 100, 30)
ingreso_anual = st.number_input("Ingreso anual", min_value=0, value=50000)
monto_prestamo = st.number_input("Monto del préstamo", min_value=0, value=10000)
historial_crediticio = st.slider("Años de historial crediticio", 0, 30, 5)

# Variables categóricas dummy
tenencia_vivienda = st.selectbox("Tenencia de vivienda", ['OWN', 'RENT', 'OTHER'])
motivo_prestamo = st.selectbox("Motivo del préstamo", ['EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE'])
calificacion = st.selectbox("Calificación crediticia", ['B', 'C', 'D', 'E', 'F', 'G'])
incumplimiento_previo = st.checkbox("Tuvo incumplimiento previo")

# Crear DataFrame con los valores ingresados
input_dict = {
    'antiguedad_empleo': [antiguedad_empleo],
    'tasa_interes': [tasa_interes],
    'porcentaje_ingreso': [porcentaje_ingreso],
    'edad': [edad],
    'ingreso_anual': [ingreso_anual],
    'monto_prestamo': [monto_prestamo],
    'historial_crediticio_anios': [historial_crediticio],
    f'tenencia_vivienda_{tenencia_vivienda}': [True],
    f'motivo_prestamo_{motivo_prestamo}': [True],
    f'calificacion_prestamo_{calificacion}': [True],
    'incumplimiento_previo_Y': [incumplimiento_previo]
}

# Completar variables faltantes con False
columnas_esperadas = modelo.feature_names_in_
for col in columnas_esperadas:
    if col not in input_dict:
        input_dict[col] = [False]

input_df = pd.DataFrame(input_dict)

# Botón para predecir
if st.button("Predecir"):
    proba = modelo.predict_proba(input_df)[0][1]
    pred = modelo.predict(input_df)[0]

    st.write(f"🔍 **Probabilidad de incumplimiento:** {proba:.2%}")

    if pred == 1:
        st.error("⚠️ El modelo predice que el cliente INCUMPLIRÍA el crédito.")
    else:
        st.success("✅ El modelo predice que el cliente NO incumpliría el crédito.")
