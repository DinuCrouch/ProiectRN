# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Ionescu Dinu
**Link Repository GitHub:** [Adaugă Link-ul Tău Aici]  
**Data predării:** [Data Curentă]

---

## Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN** din lista de 9 etape.

**Obiectiv principal:** Antrenarea efectivă a modelului RN (Regresie) pentru predicția temperaturii, evaluarea erorii medii și integrarea modelului antrenat în interfața Streamlit.

---

## PREREQUISITE – Verificare Etapa 4 (OBLIGATORIU)

**Înainte de a începe Etapa 5, am verificat:**

- [x] **State Machine** definit și documentat în `docs/state_machine.png`
- [x] **Contribuție 100% date originale** (Generare prin simulare fizică în `genereaza_date_logice.py`)
- [x] **Modul 1 (Data Logging)** funcțional - produce `date_meteo.txt` și CSV-uri procesate
- [x] **Modul 2 (RN)** cu arhitectură definită (`models/model_meteo.keras`)
- [x] **Modul 3 (UI/Web Service)** funcțional (`app_meteo.py`)

---

## Pregătire Date pentru Antrenare

Deoarece proiectul utilizează **100% date originale generate prin simulare** (conform Etapei 4), nu a fost necesară o fuziune (merge) cu seturi externe.

**Procesul realizat:**
1.  **Generare:** Scriptul `genereaza_date_logice.py` a creat 5000 de intrări (ore simulate).
2.  **Preprocesare:** Scriptul `preprocess.py` a normalizat datele (MinMax) și a creat trăsături ciclice (`sin/cos` pentru oră și lună).
3.  **Split:** Datele au fost împărțite stratificat:
    * **Train:** 70% (3500 mostre)
    * **Validation:** 15% (750 mostre)
    * **Test:** 15% (750 mostre)

---

##  Cerințe Structurate pe 3 Niveluri

### Nivel 1 – Obligatoriu pentru Toți

1. **Antrenare model:** Realizată pe 5000 de ore simulate.
2. **Epoci:** 200 (necesare pentru convergența regresiei fine).
3. **Metrici test set (Adaptate pentru Regresie):**
   - **R2 Score (Echivalent Acuratețe):** ~0.85 (Target atins)
   - **MAE (Eroare Medie Absolută):** < 1.5°C
4. **Integrare UI:** Modelul `model_meteo.keras` este încărcat dinamic în Streamlit.

#### Tabel Hiperparametri și Justificări (OBLIGATORIU - Nivel 1)

| **Hiperparametru** | **Valoare Aleasă** | **Justificare** |
|--------------------|-------------------|-----------------|
| **Learning rate** | 0.001 | Valoare standard pentru Adam. O valoare mai mare (0.01) cauza oscilații în Loss (MSE), iar mai mică (0.0001) antrena prea lent. |
| **Batch size** | 32 | Dataset-ul are 5000 linii. Batch 32 oferă actualizări frecvente ale gradienților fără a depăși memoria sau a încetini procesul. |
| **Number of epochs** | 150 | Regresia temperaturii necesită mai multe epoci decât clasificarea simplă pentru a minimiza eroarea absolută sub 1 grad. |
| **Optimizer** | Adam | Cel mai robust optimizator pentru date tabulare nestructurate, gestionează bine minimele locale. |
| **Loss function** | **MSE (Mean Squared Error)** | Specific pentru **Regresie**. Penalizează erorile mari mai mult decât MAE, forțând modelul să evite predicțiile aberante. |
| **Activation functions** | ReLU (hidden), **Linear (output)** | ReLU pentru straturile ascunse. **Linear** la ieșire este obligatoriu pentru regresie (temperatura poate fi orice număr real, inclusiv negativ). |

---

### Nivel 2 – Analiză Avansată

1. **Early Stopping:** Implementat (monitorizare `val_loss`, patience=10).
2. **Grafic Loss:** Salvat în `docs/grafic_performanta.png`.

### Analiză Erori în Context Industrial (OBLIGATORIU Nivel 2)

**Nu e suficient să raportați doar acuratețea globală.** Analiza performanței în contextul predicției meteo:

#### 1. Pe ce cazuri greșește cel mai mult modelul?
Modelul are tendința de a micsora extremele.
* **Eroare:** Subestimează temperaturile foarte ridicate (ex: 35°C) și supraestimează temperaturile foarte scăzute (-5°C).
* **Cauză:** Majoritatea datelor simulate sunt în intervalul mediu (10-25°C), iar rețeaua optimizează MSE pentru majoritate.

#### 2. Ce caracteristici ale datelor cauzează erori?
Combinațiile rare de factori contradictorii.
* **Exemplu:** "Soare puternic" (Seninătate 100%) dar "Vânt puternic" (50 km/h). Modelul poate fi confuz dacă radiația solară ar trebui să crească temperatura, dar vântul o scade drastic.

#### 3. Ce implicații are pentru aplicația industrială?
* **Scenariu:** Automatizare climatizare seră (control temperatură/ventilație) pentru culturi sensibile.
* **Impact:** Dacă modelul prezice 8°C (zonă sigură) dar realitatea este -2°C (îngheț), sistemul de încălzire nu pornește automat, ducând la pierderea totală a răsadurilor. Similar, neactivarea ventilației pe caniculă poate "arde" plantele.
* **Prioritate:** Este critic să evităm erorile mari (>3°C), mai ales în zonele de temperatură extremă, unde daunele sunt ireversibile.

#### 4. Ce măsuri corective propuneți?
1.  **Bias Correction (Implementat în UI):** Utilizatorul calibrează predicția cu temperatura curentă reală, anulând eroarea sistematică a zilei (offset-ul).
2.  **Augmentare Date Extreme:** Generarea artificială a mai multor date pentru zile caniculare și geroase în `genereaza_date_logice.py`.
3.  **Feature Engineering:** Adăugarea indicelui de confort termic ca intrare explicită.

---

**Verificare consistenta cu state machine**
|Stare din Etapa 4 | Implementare în Etapa 5 |
|ACQUIRE_INPUT | "UI Streamlit preia datele (Oră, Vânt, etc.) de la utilizator." |
|PREPROCESS | Aplicare scaler_meteo.pkl și transformare Sin/Cos pentru timp. |
|RN_INFERENCE | Forward pass cu modelul antrenat model_meteo.keras (nu random). |
|BIAS_CORRECTION | Ajustarea rezultatului brut pe baza temperaturii curente (logică UI). |
|DISPLAY | Afișare metrici și grafic în Dashboard. |