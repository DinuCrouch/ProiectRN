import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

CALE_INTRARE = 'data/raw/date_meteo.txt'
# Definim căile pentru cele 3 seturi
CALE_TRAIN = 'data/train/train.csv'
CALE_VAL   = 'data/validation/validation.csv'
CALE_TEST  = 'data/test/test.csv'
CALE_SCALER = 'models/scaler_meteo.pkl'

def procesare_inteligenta():
    if not os.path.exists(CALE_INTRARE):
        print(f"Nu găsesc: {CALE_INTRARE}")
        return

    # 1. Încărcare
    print("--> Încărcare și curățare date...")
    cols = ['data', 'ora', 'temperatura', 'grad_seninatate', 'viteza_vant', 'presiune']
    df = pd.read_csv(CALE_INTRARE, sep=r'\s+', names=cols, engine='python')

    # Conversie numerică
    cols_meteo = ['temperatura', 'grad_seninatate', 'viteza_vant', 'presiune']
    for col in cols_meteo:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # 2. Feature Engineering (Timp -> Sin/Cos)
    df['dt'] = pd.to_datetime(df['data'] + ' ' + df['ora'])
    df['ora_sin'] = np.sin(2 * np.pi * df['dt'].dt.hour / 24.0)
    df['ora_cos'] = np.cos(2 * np.pi * df['dt'].dt.hour / 24.0)
    df['luna_sin'] = np.sin(2 * np.pi * df['dt'].dt.month / 12.0)
    df['luna_cos'] = np.cos(2 * np.pi * df['dt'].dt.month / 12.0)

    # 3. Normalizare
    print("--> Normalizare și salvare Scaler...")
    scaler = MinMaxScaler()
    scaler.fit(df[cols_meteo]) # Învățăm min/max de pe tot setul
    
    os.makedirs(os.path.dirname(CALE_SCALER), exist_ok=True)
    joblib.dump(scaler, CALE_SCALER)
    
    df[cols_meteo] = scaler.transform(df[cols_meteo])

    # 4. SPLIT DATE (Train / Validation / Test)
    # Amestecăm datele aleatoriu pentru a nu avea doar "ianuarie" în train și "decembrie" în test
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(df)
    train_stop = int(n * 0.70) # 70%
    val_stop   = int(n * 0.85) # 15% (de la 70 la 85)
    
    cols_finale = [
        'luna_sin', 'luna_cos', 'ora_sin', 'ora_cos',
        'temperatura', 'grad_seninatate', 'viteza_vant', 'presiune'
    ]
    df_final = df[cols_finale].round(4)

    df_train = df_final.iloc[:train_stop]
    df_val   = df_final.iloc[train_stop:val_stop]
    df_test  = df_final.iloc[val_stop:]

    # 5. Salvare Fizică
    os.makedirs(os.path.dirname(CALE_TRAIN), exist_ok=True)
    df_train.to_csv(CALE_TRAIN, index=False)
    df_val.to_csv(CALE_VAL, index=False)
    df_test.to_csv(CALE_TEST, index=False)

    print(f"--> Split finalizat:")
    print(f"    Train:      {len(df_train)} linii -> {CALE_TRAIN}")
    print(f"    Validation: {len(df_val)} linii -> {CALE_VAL}")
    print(f"    Test:       {len(df_test)} linii -> {CALE_TEST}")

if __name__ == "__main__":
    procesare_inteligenta()