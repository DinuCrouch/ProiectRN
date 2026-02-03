import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import math

# --- CONFIGURĂRI ---
CALE_MODEL = 'models/model_meteo.keras'
CALE_SCALER = 'models/scaler_meteo.pkl'

# Funcție pentru încărcarea resurselor (cu cache pentru viteză)
@st.cache_resource
def incarca_resurse():
    try:
        model = tf.keras.models.load_model(CALE_MODEL)
        scaler = joblib.load(CALE_SCALER)
        return model, scaler
    except Exception as e:
        return None, None

def preproceseaza_input(ora_tinta, seninatate, vant, presiune, luna_curenta, scaler):
    """
    Transformă datele introduse de utilizator în formatul acceptat de rețeaua neurală.
    """
    # 1. Calculăm Sin/Cos pentru Ora Țintă și Luna Curentă
    ora_sin = np.sin(2 * np.pi * ora_tinta / 24.0)
    ora_cos = np.cos(2 * np.pi * ora_tinta / 24.0)
    luna_sin = np.sin(2 * np.pi * luna_curenta / 12.0)
    luna_cos = np.cos(2 * np.pi * luna_curenta / 12.0)

    # 2. Scalăm valorile meteo (Seninatate, Vânt, Presiune)
    # Creăm un rând dummy cu temperatura 0, doar pentru a putea folosi scalerul
    input_brut = np.array([[0, seninatate, vant, presiune]])
    input_scaled = scaler.transform(input_brut)
    
    # Extragem valorile scalate
    seninatate_s = input_scaled[0, 1]
    vant_s       = input_scaled[0, 2]
    presiune_s   = input_scaled[0, 3]

    # 3. Construim vectorul final de 7 trăsături
    vector_final = np.array([[
        luna_sin, luna_cos, 
        ora_sin, ora_cos, 
        seninatate_s, vant_s, presiune_s
    ]])
    
    return vector_final

# --- INTERFAȚA STREAMLIT ---
st.set_page_config(page_title="Meteo AI Neural", page_icon="🌤️")

st.title("🌤️ Predicție Meteo cu Rețele Neurale")
st.markdown("Acest sistem folosește AI pentru a prezice temperatura viitoare bazată pe condițiile actuale.")

# Încărcare model
model, scaler = incarca_resurse()

if model is None or scaler is None:
    st.error("⚠️ Nu am găsit modelul sau scalerul! Te rog rulează întâi `preprocess.py` și `train_neural.py`.")
else:
    # --- COLTARE INPUT (Sidebar) ---
    st.sidebar.header("Parametrii Acum")
    
    # --- MODIFICARE AICI: Luna este fixată, nu selectată ---
    luna = 3  # Luna Martie este setată implicit ("hardcoded")
    # Am afișat doar un text informativ ca să știi că e setat pe Martie
    st.sidebar.info("📅 Luna curentă: **Martie** (Implicit)")

    # Restul intrărilor
    ora_curenta = st.sidebar.slider("Ora Curentă (0-23)", 0, 23, 12)
    
    st.sidebar.markdown("---")
    temp_acum = st.sidebar.number_input("🌡️ Temperatura Acum (°C)", value=15.0, step=0.5)
    presiune = st.sidebar.number_input("⏲️ Presiune (hPa)", value=1015, step=1)
    vant = st.sidebar.slider("💨 Viteza Vânt (km/h)", 0, 100, 10)
    seninatate = st.sidebar.slider("☀️ Grad Seninătate (%)", 0, 100, 80)

    st.sidebar.markdown("---")
    offset_ore = st.sidebar.slider("🔮 Peste câte ore vrei predicția?", 1, 24, 1)

    # --- ZONA PRINCIPALĂ ---
    
    # Calculăm ora viitoare
    ora_viitoare = (ora_curenta + offset_ore) % 24
    
    st.subheader(f"Analiză pentru ora {ora_viitoare}:00 (Peste {offset_ore} ore)")
    
    if st.button("Calculează Temperatura Viitoare"):
        # 1. Predicție "Teoretică" pentru momentul ACUM
        X_acum = preproceseaza_input(ora_curenta, seninatate, vant, presiune, luna, scaler)
        pred_acum_scaled = model.predict(X_acum, verbose=0)
        
        # Denormalizăm
        dummy_matrix_acum = np.zeros((1, 4))
        dummy_matrix_acum[0, 0] = pred_acum_scaled[0, 0]
        temp_teoretica_acum = scaler.inverse_transform(dummy_matrix_acum)[0, 0]
        
        # 2. Calculăm BIAS-ul (Diferența dintre realitate și teorie)
        bias = temp_acum - temp_teoretica_acum
        
        # 3. Predicție pentru VIITOR
        X_viitor = preproceseaza_input(ora_viitoare, seninatate, vant, presiune, luna, scaler)
        pred_viitor_scaled = model.predict(X_viitor, verbose=0)
        
        # Denormalizăm
        dummy_matrix_viitor = np.zeros((1, 4))
        dummy_matrix_viitor[0, 0] = pred_viitor_scaled[0, 0]
        temp_teoretica_viitor = scaler.inverse_transform(dummy_matrix_viitor)[0, 0]
        
        # 4. Aplicăm Bias-ul la predicția viitoare
        temp_finala = temp_teoretica_viitor + bias

        # --- AFIȘARE REZULTATE ---
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="Temperatura Acum (Real)", value=f"{temp_acum} °C")
            
        with col2:
            st.metric(label="Predicție AI (Brut)", value=f"{temp_teoretica_viitor:.1f} °C", 
                      delta=f"{temp_teoretica_viitor - temp_teoretica_acum:.1f} °C vs Acum")
            
        with col3:
            st.metric(label=f"Prognoză Finală (+{offset_ore}h)", value=f"{temp_finala:.1f} °C",
                      help="Include calibrarea bazată pe temperatura curentă introdusă de tine.")

        # Explicație vizuală
        st.info(f"Modelul a detectat o diferență de **{bias:.2f}°C** față de media statistică pentru luna Martie. Această ajustare a fost aplicată prognozei.")
        
        # Mic grafic de trend
        st.write("### Trend estimat (următoarele 24h)")
        trend_data = []
        for h in range(1, 25):
            h_target = (ora_curenta + h) % 24
            # Predicție folosind luna fixată (3)
            X_h = preproceseaza_input(h_target, seninatate, vant, presiune, luna, scaler)
            p_scaled = model.predict(X_h, verbose=0)[0,0]
            # Denormalizare
            d_mat = np.zeros((1, 4))
            d_mat[0, 0] = p_scaled
            val = scaler.inverse_transform(d_mat)[0, 0] + bias
            trend_data.append(val)
            
        chart_data = pd.DataFrame({
            "Ore de acum": range(1, 25),
            "Temperatura": trend_data
        })
        st.line_chart(chart_data, x="Ore de acum", y="Temperatura")