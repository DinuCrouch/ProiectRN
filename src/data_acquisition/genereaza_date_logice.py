import pandas as pd
import numpy as np
import os

def genereaza_date_logice():
    NUMAR_INTRARI = 2200
    timeline = pd.date_range(start='2024-03-01 00:00', periods=NUMAR_INTRARI, freq='H')
    
    # 1. Generăm factori de bază
    ora = timeline.hour
    luna = timeline.month
    
    # 2. Generăm Vântul și Presiunea (aleatoriu, căci ele variază haotic)
    np.random.seed(42)
    vant = np.random.randint(0, 60, NUMAR_INTRARI)
    presiune = np.random.randint(990, 1030, NUMAR_INTRARI)
    seninatate = np.random.randint(0, 101, NUMAR_INTRARI) # 0 = Nor, 100 = Soare

    # 3. CALCULĂM TEMPERATURA BAZATĂ PE FORMULE FIZICE 
    # Temperatura de bază (media primăverii)
    temp = 10.0 
    
    # + Ciclul Zi/Noapte (mai cald la prânz, ora 14)
    # Folosim o funcție sin care are maximul pe la prânz
    temp += 10 * np.sin((ora - 8) * np.pi / 12)
    
    # + Efectul Soarelui (dacă e senin, e mai cald ziua)
    # Doar între orele 06 și 20
    ziua = (ora > 6) & (ora < 20)
    temp += (seninatate / 100) * 5 * ziua
    
    # - Efectul Vântului (vântul răcește)
    temp -= (vant / 60) * 3
    
    # + Zgomot aleatoriu (ca să nu fie perfect matematic)
    temp += np.random.normal(0, 2, NUMAR_INTRARI)

    # Rotunjim la întregi pentru a păstra formatul tău
    temp = temp.astype(int)

    # 4. Salvare
    df = pd.DataFrame({
        'data': timeline.strftime('%Y-%m-%d'),
        'ora': timeline.strftime('%H:%M'),
        'temperatura': temp,
        'grad_seninatate': seninatate,
        'viteza_vant': vant,
        'presiune': presiune
    })

    CALE_IESIRE = 'data/raw/date_meteo.txt'
    os.makedirs(os.path.dirname(CALE_IESIRE), exist_ok=True)
    df.to_csv(CALE_IESIRE, sep='\t', index=False, header=False)
    print("S-au generat date LOGICE . Acum modelul va putea învăța!")

if __name__ == "__main__":
    genereaza_date_logice()