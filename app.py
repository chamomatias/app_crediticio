import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- 1. Cargar datos ---
@st.cache_data
def cargar_datos():
    df = pd.read_csv("data.csv")
    return df

data = cargar_datos()

# --- 2. Entrenar modelo en backend ---
X = data.drop("incumplimiento", axis=1)
y = data["incumplimiento"]
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

preprocesador = ColumnTransformer(transformers=[
    ("escala", StandardScaler(), num_cols)
], remainder="passthrough")

modelo = Pipeline([
    ("preprocesamiento", preprocesador),
    ("modelo", LogisticRegression(class_weight="balanced", max_iter=2000))
])

modelo.fit(X, y)

# --- 3. Interfaz de usuario ---
st.title("🧠 Evaluación de Riesgo Crediticio")
st.write("Ingresá los datos del cliente para predecir si podría incumplir el crédito.")

# Inputs del usuario
antiguedad_empleo = st.slider("Antigüedad en empleo (años)", 0, 30, 5)
tasa_interes = st.slider("Tasa de interés (%)", 0.0, 30.0, 12.0)
porcentaje_ingreso = st.slider("Porcentaje del ingreso", 0.0, 0.5, 0.2)
edad = st.slider("Edad", 18, 70, 35)
ingreso_anual = st.number_input("Ingreso anual", 0, 500000, 50000)
monto_prestamo = st.number_input("Monto solicitado", 0, 100000, 10000)
historial = st.slider("Historial crediticio (años)", 0, 30, 5)

# Dummy encoding de ejemplo (deberías adaptarlo a tu dataset real)
tenencia_vivienda = st.selectbox("Tenencia de vivienda", ["OWN", "RENT", "OTHER"])
motivo = st.selectbox("Motivo del préstamo", ["EDUCATION", "PERSONAL", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT"])
calificacion = st.selectbox("Calificación", ["B", "C", "D", "E", "F", "G"])
inc_previo = st.checkbox("Incumplimiento previo", value=False)

# --- 4. Armar input con columnas como en el dataset ---
input_dict = {
    "antiguedad_empleo": [antiguedad_empleo],
    "tasa_interes": [tasa_interes],
    "porcentaje_ingreso": [porcentaje_ingreso],
    "edad": [edad],
    "ingreso_anual": [ingreso_anual],
    "monto_prestamo": [monto_prestamo],
    "historial_crediticio_anios": [historial],
    "tenencia_vivienda_" + tenencia_vivienda: [1],
    "motivo_prestamo_" + motivo: [1],
    "calificacion_prestamo_" + calificacion: [1],
    "incumplimiento_previo_Y": [int(inc_previo)]
}

# Completar las demás columnas como 0 si no fueron seleccionadas
for col in modelo.named_steps["preprocesamiento"].get_feature_names_out():
    col_name = col.split("__")[-1]
    if col_name not in input_dict:
        input_dict[col_name] = [0]

input_df = pd.DataFrame(input_dict)

# --- 5. Predicción ---
if st.button("Predecir"):
    proba = modelo.predict_proba(input_df)[0][1]
    pred = modelo.predict(input_df)[0]

    st.write(f"🔍 Probabilidad de incumplimiento: {proba:.2%}")
    if pred == 1:
        st.error("⚠️ El cliente probablemente incumpla el crédito.")
    else:
        st.success("✅ El cliente probablemente cumpla con el crédito.")
