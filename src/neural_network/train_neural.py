import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score, classification_report, mean_absolute_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import os

# --- CONFIGURĂRI ---
CALE_TRAIN = 'data/train/train.csv'
CALE_VAL   = 'data/validation/validation.csv'
CALE_TEST  = 'data/test/test.csv'
CALE_SCALER = 'models/scaler_meteo.pkl'
CALE_MODEL_SALVARE = 'models/model_meteo.keras' 
CALE_GRAFIC = 'docs/grafic_performanta.png'
CALE_MATRICE = 'docs/confusion_matrix.png' 

EPOCHS = 200
BATCH_SIZE = 32
PRAG_RISC = 2.0 # Sub 2 grade considerăm Risc de Îngheț

def incarca_dataset(cale):
    if not os.path.exists(cale):
        raise FileNotFoundError(f"Lipsă fișier: {cale}")
    df = pd.read_csv(cale)
    feature_cols = [
        'luna_sin', 'luna_cos', 'ora_sin', 'ora_cos',
        'grad_seninatate', 'viteza_vant', 'presiune'
    ]
    X = df[feature_cols].values
    y = df['temperatura'].values
    return X, y

def antreneaza_retea():
    print("--> Încărcare seturi de date separate...")
    try:
        X_train, y_train = incarca_dataset(CALE_TRAIN)
        X_val, y_val     = incarca_dataset(CALE_VAL)
        X_test, y_test   = incarca_dataset(CALE_TEST)
        print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    except Exception as e:
        print(f"EROARE: {e}. Rulează preprocess.py!")
        return

    # 1. Definire Model
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear') 
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    # 2. Antrenare
    print("--> Start Antrenare...")
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val),
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        verbose=1
    )
    
    # 3. Salvare Model
    os.makedirs(os.path.dirname(CALE_MODEL_SALVARE), exist_ok=True)
    model.save(CALE_MODEL_SALVARE)
    print(f"--> Model salvat în {CALE_MODEL_SALVARE}")

    # 4. Evaluare Completă (Regresie + Clasificare)
    calculeaza_metrici_finale(model, history, X_test, y_test)

def calculeaza_metrici_finale(model, history, X_test, y_test):
    print("\n--> Generare metrici și grafice pe TEST SET...")
    os.makedirs(os.path.dirname(CALE_GRAFIC), exist_ok=True)

    # Predicție și Denormalizare
    y_pred_scaled = model.predict(X_test)
    
    if os.path.exists(CALE_SCALER):
        scaler = joblib.load(CALE_SCALER)
        def denormalizeaza_temp(y_vector):
            mat = np.zeros((len(y_vector), 4))
            mat[:, 0] = y_vector.flatten()
            return scaler.inverse_transform(mat)[:, 0]

        y_pred_real = denormalizeaza_temp(y_pred_scaled)
        y_test_real = denormalizeaza_temp(y_test)
    else:
        y_pred_real = y_pred_scaled.flatten()
        y_test_real = y_test

    # --- 1. METRICI REGRESIE ---
    r2 = r2_score(y_test_real, y_pred_real)
    mae = mean_absolute_error(y_test_real, y_pred_real)
    print(f"\n[METRICI REGRESIE]")
    print(f"   MAE (Eroare grade): {mae:.2f} °C")
    print(f"   R2 Score: {r2:.4f}")

    # --- 2. METRICI CLASIFICARE (ÎNGHEȚ) ---
    # Transformăm temperaturile în clase (1=Îngheț, 0=Sigur)
    y_test_class = (y_test_real < PRAG_RISC).astype(int)
    y_pred_class = (y_pred_real < PRAG_RISC).astype(int)

    f1 = f1_score(y_test_class, y_pred_class)
    cm = confusion_matrix(y_test_class, y_pred_class)
    
    print(f"\n[METRICI CLASIFICARE - RISC ÎNGHEȚ (<{PRAG_RISC}°C)]")
    print(f"   F1 Score: {f1:.4f}")
    print("   Matrice Confuzie (Text):")
    print(cm)
    print("-" * 40)
    
    # Afișăm raport detaliat
    target_names = ['Sigur (>=2°C)', 'PERICOL (<2°C)']
    print(classification_report(y_test_class, y_pred_class, target_names=target_names))

    # --- 3. GENERARE GRAFICE ---
    
    # GRAFIC 1: Performanța Modelului (Curbe)
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    limit = 80 
    plt.scatter(range(limit), y_test_real[:limit], label='Real', color='navy', alpha=0.7)
    plt.plot(range(limit), y_pred_real[:limit], label='Predicție AI', color='orange', linewidth=2)
    # Tragem o linie roșie la pragul de îngheț
    plt.axhline(y=PRAG_RISC, color='red', linestyle=':', label=f'Prag Îngheț ({PRAG_RISC}°C)')
    plt.title(f'Testare Regresie (F1 Score pe Îngheț: {f1:.2f})')
    plt.ylabel('Grade Celsius')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Val Loss', color='red', linestyle='--')
    plt.title('Curba de Învățare')
    plt.xlabel('Epoci')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CALE_GRAFIC)
    print(f"--> Grafic performanță salvat: {CALE_GRAFIC}")

    # GRAFIC 2: Confusion Matrix Vizuală
    plt.figure(figsize=(6, 5))
    # Folosim funcția din sklearn pentru a desena frumos
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix\n(Detectare Îngheț < {PRAG_RISC}°C)')
    plt.savefig(CALE_MATRICE)
    print(f"--> Confusion Matrix salvată: {CALE_MATRICE}")

if __name__ == "__main__":
    antreneaza_retea()