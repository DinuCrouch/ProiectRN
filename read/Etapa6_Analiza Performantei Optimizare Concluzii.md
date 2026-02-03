# README – Etapa 6: Analiza Performanței, Optimizarea și Concluzii Finale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Proiect:** Sistem AI pentru Predicția Temperaturii la o Fermă Locală  
**Student:** [Nume Prenume]  
**Data:** 03 Februarie 2026

---

## 1. Actualizarea Aplicației Software în Etapa 6

Sistemul a fost maturizat dintr-un simplu model de regresie într-o aplicație interactivă (`app_ferma.py`) capabilă să asiste deciziile zilnice la o fermă locală prin monitorizarea microclimatului.

### Tabel Modificări Aplicație Software

| **Componenta** | **Stare Etapa 5** | **Modificare Etapa 6** | **Justificare** |
|----------------|-------------------|------------------------|-----------------|
| **Model încărcat** | `model_meteo.keras` | `model_meteo.keras` (Optimizat) | Utilizarea unei arhitecturi Dense 128-64-32 cu Dropout pentru stabilitate. |
| **Input Date** | Date sintetice simple | Date logice (fizice) | Utilizarea `genereaza_date_logice.py` pentru a simula corect influența vântului și a soarelui. |
| **Interfață UI** | Formular de test | Streamlit cu Bias Correction | Permite fermierului să calibreze AI-ul în funcție de temperatura reală citită la sol. |
| **Feature Engineering** | Timp brut | Transformare Sin/Cos | Ameliorarea învățării ciclicității zi/noapte și a succesiunii lunilor. |
| **Vizualizare** | Fără grafic | Trend 24h Interactiv | Adăugarea unui grafic de evoluție pentru planificarea activităților în fermă. |

---

## 2. Analiza Detaliată a Performanței

### 2.1 Analiza Regresiei
Deoarece proiectul vizează o valoare continuă (temperatura), analiza se bazează pe metricile **MAE** (Mean Absolute Error) și **MSE** (Mean Squared Error) raportate în scriptul de antrenare.

* **Performanță optimă**: Modelul prezice excelent variațiile diurne datorită componentelor `ora_sin` și `ora_cos`.
* **Stabilitate**: Introducerea stratului de `Dropout(0.2)` a redus riscul de overfitting pe datele generate logic.

### 2.2 Analiza Reziduurilor (Exemple Greșite)

| **Index** | **Real (°C)** | **Predicție (°C)** | **Eroare** | **Cauză probabilă** | **Soluție** |
|-----------|---------------|-------------------|------------|---------------------|-------------|
| #1 | 10.0 | 13.5 | +3.5 | Rafală de vânt intensă (haotică) | Calibrare manuală "Bias" în UI. |
| #2 | 22.0 | 18.5 | -3.5 | Cer senin la prânz (maxim solar) | Feature `grad_seninatate` ponderat. |

---

## 3. Optimizarea Parametrilor și Experimentare

### 3.1 Strategia de Optimizare
S-a utilizat un script de antrenare cu **150 de epoci** și un **batch size de 32**.

**Axe de optimizare:**
1.  **Arhitectură**: Trei straturi Dense (128, 64, 32) pentru a captura non-liniaritatea fenomenelor meteo.
2.  **Regularizare**: Strat de Dropout pentru a gestiona zgomotul din datele de antrenare.
3.  **Optimizator**: Adam (LR=0.001) pentru o convergență rapidă spre minimul funcției de cost.



---

## 4. Concluzii Finale și Lecții Învățate

### 4.1 Evaluarea Performanței Finale
* **Target atins**: Predicția temperaturii cu o eroare medie scăzută după calibrarea în timp real.
* **Utilizabilitate**: Sistemul de "Bias Correction" din `app_ferma.py` transformă un model statistic într-un instrument practic pentru fermier.

### 4.2 Limitări Identificate
* **Date**: Modelul este antrenat pe date generate matematic; în realitate, fenomenele meteo extreme pot depăși logica liniară a scriptului de generare.
* **Localizare**: Modelul este calibrat implicit pentru luna Martie (hardcoded în aplicație), necesitând extindere pentru un ciclu anual complet.

### 4.3 Lecții Învățate
1.  **Ingineria caracteristicilor**: Transformarea Sin/Cos a timpului este esențială pentru modelele meteorologice.
2.  **Interfața contează**: Un model AI bun devine util doar dacă utilizatorul final poate interacționa simplu cu el (Streamlit).

---

## Instrucțiuni de Rulare (Pipeline Complet)

1.  **Generare Date**: `python genereaza_date_logice.py`.
2.  **Preprocesare**: `python preprocess.py` (creează seturile Train/Val/Test și Scalerul).
3.  **Antrenare**: `python train_neural.py` (generează modelul optimizat).
4.  **Aplicație**: `streamlit run app_ferma.py`.