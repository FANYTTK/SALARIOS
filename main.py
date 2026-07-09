"""
Salary Predictor - App de Streamlit
====================================
Carga el modelo entrenado en Salary_Prediction_v15_NN.ipynb (Red Neuronal
Keras): el archivo del modelo (salary_nn_model_v15.keras) y su metadata
(salary_nn_metadata_v15.pkl, con scalers de X y de y, mapeos, bins, métricas
e importancia de variables) y predice el salario anual sugerido a partir de
los datos del candidato/empleado.

Diferencia clave respecto a la versión anterior (modelo de árboles): aquí el
modelo predice en una escala normalizada, así que la predicción SIEMPRE se
revierte a dólares reales con scaler_y.inverse_transform() antes de usarse
en cualquier parte de la app.

Ejecutar con:
    streamlit run main.py
"""

import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# --------------------------------------------------------------------------
# Configuración general
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Salary Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "salary_nn_model_v15.keras"
METADATA_PATH = "salary_nn_metadata_v15.pkl"

# Tasa de cambio USD -> MXN. En producción, reemplazar por una llamada a una
# API de tipo de cambio en tiempo real (p. ej. exchangerate.host, banxico, etc.)
USD_TO_MXN = 17.5
ANNUAL_WORK_HOURS = 2080  # 40 h/semana x 52 semanas, estándar para calcular salario/hora

# --------------------------------------------------------------------------
# Iconos (SVG en línea, monocromáticos, sin emojis)
# --------------------------------------------------------------------------
ICONS = {
    "user": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
    "calendar": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/></svg>',
    "briefcase": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>',
    "building": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 22V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v18"/><path d="M2 22h20M10 6h4M10 10h4M10 14h4"/></svg>',
    "graduation": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 10 12 5 2 10l10 5 10-5Z"/><path d="M6 12v5c0 1.5 2.7 3 6 3s6-1.5 6-3v-5"/></svg>',
    "star": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 2 3.1 6.6 7.2.9-5.3 5 1.4 7.2L12 18.3 5.6 21.7 7 14.5 1.7 9.5l7.2-.9L12 2Z"/></svg>',
    "badge": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="6"/><path d="M9 13.5 7 22l5-3 5 3-2-8.5"/></svg>',
    "chart": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M7 15l4-6 3 3 5-8"/></svg>',
    "clock": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/></svg>',
    "coin": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v10M9 9.5c0-1.4 1.3-2.5 3-2.5s3 1 3 2c0 3-6 1.5-6 4.5 0 1.4 1.3 2.5 3 2.5s3-1.1 3-2.5"/></svg>',
    "info": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 8h.01M11 12h1v5h1"/></svg>',
    "history": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 2.6-6.3"/><path d="M3 4v5h5M12 7v5l4 2"/></svg>',
    "cpu": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M2 15h2M20 9h2M20 15h2"/></svg>',
}


def icon(name: str, size: int = 16, color: str = "currentColor") -> str:
    svg = ICONS[name].replace('stroke="currentColor"', f'stroke="{color}"')
    return f'<span style="display:inline-flex;vertical-align:middle;width:{size}px;height:{size}px;margin-right:6px;color:{color};">{svg}</span>'


def field_label(name: str, text: str) -> None:
    st.markdown(
        f'<div class="field-label">{icon(name, 16, "#1e3a5f")}{text}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Estilos (CSS) - alto contraste, sin texto blanco sobre blanco
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-family: "Segoe UI", "Inter", sans-serif;
    }
    .stApp {
        background: #eef1f6;
    }

    /* ---------- HERO ---------- */
    .hero {
        background: linear-gradient(135deg, #16324f 0%, #1f4e79 55%, #2b6ca3 100%);
        border-radius: 18px;
        padding: 2.4rem 2.4rem;
        color: #ffffff;
        margin-bottom: 1.6rem;
        box-shadow: 0 10px 30px rgba(22, 50, 79, 0.28);
    }
    .hero-title {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.35rem;
    }
    .hero-title svg { width: 30px; height: 30px; }
    .hero p {
        font-size: 1.02rem;
        color: #dce6f2;
        margin: 0;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 999px;
        padding: 0.3rem 0.9rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.9rem;
    }
    .hero-badge svg { width: 14px; height: 14px; margin-right: 6px; }

    /* ---------- CARDS (fondo blanco, texto oscuro forzado) ---------- */
    .card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.7rem 1.9rem;
        box-shadow: 0 4px 18px rgba(20, 30, 60, 0.07);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.2rem;
        color: #1e293b;
    }
    .card * { color: #1e293b; }

    .card-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.15rem;
        font-weight: 700;
        color: #16324f !important;
        margin-bottom: 1.1rem;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.7rem;
    }
    .card-title svg { width: 20px; height: 20px; color: #1f4e79; }

    .field-label {
        display: flex;
        align-items: center;
        font-size: 0.92rem;
        font-weight: 600;
        color: #334155 !important;
        margin-bottom: 0.3rem;
        margin-top: 0.6rem;
    }

    .field-hint {
        font-size: 0.78rem;
        color: #64748b !important;
        margin-top: -0.2rem;
        margin-bottom: 0.4rem;
    }

    /* Inputs de Streamlit: forzar fondo claro y texto oscuro siempre */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
    }
    .stSelectbox div[data-baseweb="select"] span { color: #0f172a !important; }
    .stRadio label, .stRadio div { color: #1e293b !important; }
    div[role="radiogroup"] label span { color: #1e293b !important; }
    .stTextInput label, .stNumberInput label, .stSelectbox label, .stRadio label { color: #1e293b !important; }

    /* ---------- RESULT CARD ---------- */
    .result-card {
        background: linear-gradient(135deg, #0f766e 0%, #0d9488 100%);
        border-radius: 18px;
        padding: 2rem 1.8rem;
        color: #ffffff;
        text-align: center;
        box-shadow: 0 10px 25px rgba(13, 148, 136, 0.32);
    }
    .result-card * { color: #ffffff !important; }
    .result-label {
        font-size: 0.95rem;
        letter-spacing: 0.4px;
        text-transform: uppercase;
        opacity: 0.9;
    }
    .result-big {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0.25rem 0 0.1rem 0;
    }
    .result-sub {
        font-size: 1.05rem;
        opacity: 0.95;
    }
    .metric-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 999px;
        padding: 0.4rem 1rem;
        margin: 0.25rem;
        font-size: 0.88rem;
        font-weight: 600;
    }
    .metric-pill svg { width: 14px; height: 14px; margin-right: 6px; }

    .placeholder-card {
        background: #ffffff;
        border: 1px dashed #cbd5e1;
        border-radius: 16px;
        padding: 3.2rem 1.5rem;
        text-align: center;
        color: #64748b;
    }

    .error-note {
        background: #fff7ed;
        border: 1px solid #fdba74;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #7c2d12 !important;
        margin-top: 0.6rem;
    }
    .error-note * { color: #7c2d12 !important; }

    section[data-testid="stSidebar"] { display: none; }

    div.stButton > button {
        background: #1f4e79;
        color: #ffffff;
        font-weight: 700;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        border: none;
    }
    div.stButton > button:hover {
        background: #16324f;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Carga del modelo (red neuronal Keras) + su metadata (scalers, mapeos, etc.)
# --------------------------------------------------------------------------
@st.cache_resource
def load_metadata(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_keras_model(path: str):
    # compile=False porque solo hacemos predict(), no reentrenamos en la app;
    # evita advertencias/errores si la versión de TF de la app difiere un poco
    # de la usada al entrenar.
    return tf.keras.models.load_model(path, compile=False)


def build_features(model_data: dict, age, experience, qualification, university,
                    role_level, has_cert) -> pd.DataFrame:
    """Reconstruye exactamente el mismo pipeline de feature engineering
    usado en el notebook de entrenamiento, en el mismo orden de columnas."""

    qualification_map = model_data["qualification_map"]
    university_map = model_data["university_map"]
    age_bins = model_data["age_bins"]
    exp_bins = model_data["exp_bins"]

    qualification_level = qualification_map[qualification]
    university_tier = university_map[university]
    is_senior = 1 if role_level == "Senior" else 0
    has_cert_flag = 1 if has_cert == "Si" else 0

    experience_age_ratio = experience / (age + 1e-6)
    exp_qual_interaction = experience * qualification_level
    age_qual_interaction = age * qualification_level
    experience_log = np.log1p(experience)

    age_bin = pd.cut([age], bins=age_bins, labels=[1, 2, 3, 4, 5])[0]
    experience_bin = pd.cut([experience], bins=exp_bins, labels=[1, 2, 3, 4, 5])[0]
    age_bin = int(age_bin) if not pd.isna(age_bin) else 3
    experience_bin = int(experience_bin) if not pd.isna(experience_bin) else 3

    seniority_score = (
        qualification_level * 2 + is_senior * 3 + has_cert_flag * 1 + experience_log
    )

    row = {
        "Experience": experience,
        "Age": age,
        "Qualification_Level": qualification_level,
        "University_Tier": university_tier,
        "Is_Senior": is_senior,
        "Has_Cert": has_cert_flag,
        "Experience_Age_Ratio": experience_age_ratio,
        "Exp_Qual_Interaction": exp_qual_interaction,
        "Age_Qual_Interaction": age_qual_interaction,
        "Experience_Log": experience_log,
        "Age_Bin": age_bin,
        "Experience_Bin": experience_bin,
        "Seniority_Score": seniority_score,
    }

    feature_names = model_data["feature_names"]
    missing = [col for col in feature_names if col not in row]
    if missing:
        st.warning(
            f"Aviso: el modelo espera las columnas {missing}, que no se generaron en la app. "
            "Se están enviando como 0, lo que puede distorsionar la predicción. Revisa que "
            "build_features() cubra exactamente las mismas columnas que el notebook de entrenamiento."
        )
    ordered_row = {col: row.get(col, 0) for col in feature_names}
    return pd.DataFrame([ordered_row])


def predict_salary(model_data: dict, keras_model, features_df: pd.DataFrame) -> float:
    """Predice el salario en dólares reales.

    Diferencia clave frente al modelo de árboles: la red fue entrenada sobre
    el target ESCALADO (scaler_y), así que su salida cruda no está en dólares.
    Por eso aquí SIEMPRE se revierte con scaler_y.inverse_transform() antes de
    devolver el número — nunca se debe usar la salida cruda de keras_model.predict()
    directamente en ninguna otra parte de la app.
    """
    scaler_X = model_data["scaler_X"]
    scaler_y = model_data["scaler_y"]

    X_scaled = scaler_X.transform(features_df)
    prediction_scaled = keras_model.predict(X_scaled, verbose=0)
    prediction_real = scaler_y.inverse_transform(prediction_scaled)
    return float(prediction_real[0][0])


# --------------------------------------------------------------------------
# Estado de sesión (historial)
# --------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-title">{icon('coin', 30, '#ffffff')}Salary Predictor</div>
        <p>Estima el sueldo anual sugerido para un puesto a partir del perfil del candidato,
        usando un modelo de Machine Learning entrenado y validado.</p>
        <div class="hero-badge">{icon('cpu', 14, '#ffffff')}Modelo: Red Neuronal (Keras)</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Carga de modelo con manejo de errores
# --------------------------------------------------------------------------
try:
    model_data = load_metadata(METADATA_PATH)
except FileNotFoundError:
    st.markdown(
        f"""
        <div class="error-note">
            {icon('info', 16, '#7c2d12')}No se encontro el archivo de metadata <b>{METADATA_PATH}</b>.
            Ejecuta primero el notebook <b>Salary_Prediction_v15_NN.ipynb</b>
            (requiere conexion a internet para descargar el dataset y entrenar)
            y coloca los archivos generados en la misma carpeta que este main.py.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

try:
    keras_model = load_keras_model(MODEL_PATH)
except (FileNotFoundError, OSError):
    st.markdown(
        f"""
        <div class="error-note">
            {icon('info', 16, '#7c2d12')}No se encontro el archivo del modelo <b>{MODEL_PATH}</b>.
            Ejecuta primero el notebook <b>Salary_Prediction_v13_NN.ipynb</b>
            y coloca el .keras generado en la misma carpeta que este main.py.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# --------------------------------------------------------------------------
# Formulario
# --------------------------------------------------------------------------
col_form, col_result = st.columns([1.1, 1], gap="large")

with col_form:
    st.markdown(
        f'<div class="card"><div class="card-title">{icon("badge", 20, "#1f4e79")}Datos del candidato</div>',
        unsafe_allow_html=True,
    )

    field_label("user", "Nombre (opcional)")
    name = st.text_input("Nombre", label_visibility="collapsed", placeholder="Ej. Juan Perez")

    c1, c2 = st.columns(2)
    with c1:
        field_label("calendar", "Edad")
        age = st.number_input("Edad", label_visibility="collapsed", min_value=18, max_value=70,
                               value=30, step=1)
    with c2:
        field_label("clock", "Anios de experiencia")
        experience = st.number_input("Experiencia", label_visibility="collapsed", min_value=0,
                                      max_value=50, value=5, step=1)

    field_label("building", "Departamento")
    department = st.selectbox(
        "Departamento", label_visibility="collapsed",
        options=["Tecnologia", "Ventas", "Marketing", "Finanzas", "Operaciones", "Recursos Humanos", "Otro"],
    )
    st.markdown(
        f'<p class="field-hint">{icon("info", 12, "#64748b")}El dataset de entrenamiento no incluye '
        "Departamento; este campo se guarda solo como referencia en tu historial y no afecta la prediccion.</p>",
        unsafe_allow_html=True,
    )

    field_label("graduation", "Nivel de formacion")
    qualification = st.selectbox(
        "Formacion", label_visibility="collapsed",
        options=["Bsc", "Msc", "PhD"],
        format_func=lambda x: {"Bsc": "Licenciatura (Bsc)", "Msc": "Maestria (Msc)", "PhD": "Doctorado (PhD)"}[x],
    )

    field_label("star", "Prestigio de la universidad")
    university = st.selectbox(
        "Universidad", label_visibility="collapsed",
        options=["Tier1", "Tier2", "Tier3"],
        format_func=lambda x: {"Tier1": "Tier 1 (alto prestigio)", "Tier2": "Tier 2 (medio)", "Tier3": "Tier 3 (regional)"}[x],
    )

    field_label("briefcase", "Nivel del puesto")
    role_level = st.selectbox(
        "Puesto", label_visibility="collapsed",
        options=["Senior", "Junior/Mid"],
    )

    field_label("badge", "Certificacion profesional")
    has_cert = st.radio("Certificacion", label_visibility="collapsed", options=["Si", "No"], horizontal=True)

    st.markdown("</div>", unsafe_allow_html=True)

    calculate = st.button("Calcular Sueldo Sugerido", use_container_width=True, type="primary")

with col_result:
    if calculate:
        features_df = build_features(
            model_data, age, experience, qualification, university, role_level, has_cert
        )
        raw_prediction = predict_salary(model_data, keras_model, features_df)

        # --- Salvaguarda transparente contra extrapolaciones absurdas ---
        # Aunque la red ya no tiene el problema de escala del modelo anterior
        # (aqui la salida SIEMPRE se revierte con scaler_y antes de llegar aqui),
        # sigue siendo posible que, para una combinacion de entradas muy alejada
        # de lo visto en entrenamiento, la prediccion caiga fuera de un rango
        # razonable. Por eso se compara contra el rango de salarios REALMENTE
        # observado en el entrenamiento (guardado en la metadata) y, si se sale
        # de ahi, se ajusta al limite mas cercano y se deja explicito en pantalla.
        salary_range = model_data.get("salary_range")
        out_of_range = False
        if salary_range is not None:
            floor = salary_range["p01"]
            ceiling = salary_range["p99"]
            if raw_prediction < floor or raw_prediction > ceiling:
                out_of_range = True
                predicted_salary = min(max(raw_prediction, floor), ceiling)
            else:
                predicted_salary = raw_prediction
        else:
            predicted_salary = raw_prediction

        predicted_salary = max(predicted_salary, 0)
        salary_mxn = predicted_salary * USD_TO_MXN
        monthly_usd = predicted_salary / 12
        monthly_mxn = salary_mxn / 12
        hourly_usd = predicted_salary / ANNUAL_WORK_HOURS  # solo informativo, no es feature del modelo

        test_metrics = model_data["metrics"]["test"]
        mape = test_metrics["MAPE"]
        mae = test_metrics["MAE"]

        if out_of_range:
            st.markdown(
                f"""
                <div class="error-note">
                    {icon('info', 16, '#7c2d12')}El modelo, para esta combinacion exacta de datos,
                    calculo un valor de <b>${raw_prediction:,.0f} USD</b>, fuera del rango de
                    salarios observados durante el entrenamiento
                    (${salary_range['p01']:,.0f} - ${salary_range['p99']:,.0f} USD).
                    Es una senial de que esta combinacion de entradas es poco comun en los datos de
                    entrenamiento (extrapolacion), asi que el resultado se ajusto al limite mas
                    cercano de ese rango en vez de mostrar el numero sin contexto.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Sueldo anual sugerido (USD)</div>
                <div class="result-big">${predicted_salary:,.0f}</div>
                <div class="result-sub">Equivalente a ${salary_mxn:,.0f} MXN</div>
                <hr style="border-color: rgba(255,255,255,0.3); margin: 1.1rem 0;">
                <span class="metric-pill">{icon('coin', 14, '#ffffff')}Mensual: ${monthly_usd:,.0f} USD</span>
                <span class="metric-pill">{icon('coin', 14, '#ffffff')}Mensual: ${monthly_mxn:,.0f} MXN</span>
                <span class="metric-pill">{icon('clock', 14, '#ffffff')}${hourly_usd:,.1f} USD / hora</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="card"><div class="card-title">{icon("chart", 20, "#1f4e79")}Variacion y margen de error</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            Este modelo (<b>Red Neuronal - Keras</b>) tiene, sobre datos de prueba no vistos:
            <ul>
                <li><b>MAPE:</b> {mape:.2f} por ciento (error porcentual promedio)</li>
                <li><b>MAE:</b> ${mae:,.0f} USD (error promedio en dolares)</li>
            </ul>
            En terminos practicos, el sueldo real podria variar aproximadamente
            <b>+/- ${mae:,.0f} USD</b> respecto a la cifra sugerida.
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.session_state.history.append(
            {
                "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Nombre": name if name else "-",
                "Departamento": department,
                "Edad": age,
                "Experiencia": experience,
                "Formacion": qualification,
                "Universidad": university,
                "Puesto": role_level,
                "Certificacion": has_cert,
                "Salario (USD)": round(predicted_salary, 0),
                "Salario (MXN)": round(salary_mxn, 0),
            }
        )

        with st.expander("Ver por que el modelo dio este resultado (transparencia)"):
            st.markdown("**Datos exactos enviados al modelo** (ya con las variables derivadas):")
            st.dataframe(features_df.T.rename(columns={0: "Valor"}), use_container_width=True)

            importances = model_data.get("feature_importances")
            if importances:
                st.markdown(
                    "**Importancia de cada variable (Permutation Importance) para el modelo entrenado:**"
                )
                imp_series = pd.Series(importances).sort_values(ascending=False)
                st.bar_chart(imp_series)
                st.caption(
                    "Calculada mezclando cada variable en el set de prueba y midiendo cuanto empeora "
                    "el error del modelo; a diferencia de un arbol, los pesos de una red neuronal no "
                    "se interpretan directamente como importancia."
                )
                exp_rank = list(imp_series.index).index("Experience") + 1 if "Experience" in imp_series.index else None
                if exp_rank:
                    st.caption(
                        f"'Experience' ocupa el puesto {exp_rank} de {len(imp_series)} en importancia "
                        "para este modelo. Si aqui aparece con peso bajo, es una caracteristica real "
                        "de los datos de entrenamiento (ver Seccion 3.1 del notebook), no un error de la app."
                    )
            else:
                st.caption(
                    "Este modelo se guardo antes de agregar el registro de importancia de variables; "
                    "vuelve a ejecutar el notebook para verlo aqui."
                )
    else:
        st.markdown(
            f"""
            <div class="placeholder-card">
                {icon('chart', 34, '#94a3b8')}
                <p style="margin-top:0.8rem;">Completa el formulario y presiona
                <b>Calcular Sueldo Sugerido</b> para ver el resultado aqui.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------------------
# Historial de predicciones
# --------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    f'<div class="card-title" style="border-bottom:none;">{icon("history", 20, "#1f4e79")}Historial de predicciones</div>',
    unsafe_allow_html=True,
)

if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True, hide_index=True)
    if st.button("Limpiar historial"):
        st.session_state.history = []
        st.rerun()
else:
    st.caption("Aun no hay predicciones en esta sesion.")