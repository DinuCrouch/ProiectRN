import pandas as pd
import os

CALE_INTRARE = 'data/raw/date_meteo.txt'
CALE_IESIRE  = 'data/processed/date_meteo_prelucrat.csv'

def prelucreaza_ora():
    if not os.path.exists(CALE_INTRARE):
        print(f"Eroare: Nu găsesc fișierul {CALE_INTRARE}")
        return

    cols = ['data', 'ora', 'temperatura', 'grad_seninatate', 'viteza_vant', 'presiune']
    
    try:
        df = pd.read_csv(CALE_INTRARE, sep=r'\s+', names=cols, engine='python')
    except Exception as e:
        print(f"Eroare: {e}")
        return

    if str(df.iloc[0]['temperatura']).strip().lower().startswith('temp'):
        df = df.iloc[1:].reset_index(drop=True)

    try:
        data_clean = df['data'].astype(str).str.replace('-', '')
        ora_clean = df['ora'].astype(str).apply(lambda x: x.split(':')[0].zfill(2))
        
        df['timestamp_id'] = data_clean + ora_clean
        
        cols_finale = ['timestamp_id', 'temperatura', 'grad_seninatate', 'viteza_vant', 'presiune']
        df_final = df[cols_finale]
        
    except Exception as e:
        print(f"Eroare la transformare: {e}")
        return

    os.makedirs(os.path.dirname(CALE_IESIRE), exist_ok=True)
    df_final.to_csv(CALE_IESIRE, index=False)
    
    print(f"Succes. Fișier salvat: {CALE_IESIRE}")

if __name__ == "__main__":
    prelucreaza_ora()