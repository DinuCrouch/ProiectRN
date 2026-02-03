# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Ionescu Dinu  
**Data:** 4/12/2025

---

## Scopul Etapei 4

Această etapă definește arhitectura completă a sistemului software. Am dezvoltat scheletul funcțional al celor 3 module principale (Achiziție, Neural Network, UI) și am definit fluxul de date prin State Machine.

---

##  Livrabile Obligatorii

### 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul vostru** | **Modul software responsabil** |
|---------------------------|--------------------------------|--------------------------------|
| Predicția temperaturii locale bazată pe condiții meteo specifice momentului (vânt, presiune, oră) | Model de regresie neurală antrenat pe date istorice simulate → output: temperatura în °C | **Modul 2: Neural Network** (Arhitectură Dense) |
| Vizualizarea impactului factorilor meteo asupra temperaturii în timp real pentru utilizator | Interfață grafică interactivă ce permite modificarea parametrilor și afișarea predicției calibrate | **Modul 3: Web Service / UI** (Streamlit App) |
| Generarea unui dataset consistent pentru antrenare, lipsit de erori de măsurare reale | Simulare fizică a ciclurilor termodinamice (zi/noapte, anotimp) și generarea a 2000+ ore de date sintetice | **Modul 1: Data Acquisition** (Generator Date Logice) |

---

### 2. Contribuția Originală la Setul de Date – 100% Original

### Contribuția originală la setul de date:

**Total observații finale:** 2400 linii (ore simulate)
**Observații originale:** 2400 (100%)

**Tipul contribuției:**
[X] Date generate prin simulare fizică  
[ ] Date achiziționate cu senzori proprii  
[ ] Etichetare/adnotare manuală  
[ ] Date sintetice prin metode avansate  

**Descriere detaliată:**
Am creat un generator algoritmic (`genereaza_date_logice.py`) care simulează legile fizice de bază ale meteorologiei, nu doar date aleatorii. 
- **Ciclul Zi/Noapte:** Implementat folosind o funcție sinusoidală cu maxim la ora 14:00 și minim la 04:00.
- **Ciclul Anual:** Variație sezonieră sinusoidală (maxim în Iunie, minim în Martie).
- **Interacțiuni Fizice:** Am implementat logică de corelație: Vântul puternic scade temperatura (factor de răcire), iar Seninătatea o crește în timpul zilei (radiație solară).
- **Zgomot:** Am adăugat zgomot Gaussian pentru a simula variațiile naturale imprevizibile și erorile de măsurare, forțând rețeaua să generalizeze.

**Locația codului:** `src/data_acquisition/genereaza_date_logice.py` (redenumit din rădăcină pentru structură)
**Locația datelor:** `data/raw/date_meteo.txt`

**Dovezi:**
- Scriptul python de generare este funcțional și produce fișierul raw.
- Distribuția datelor reflectă curbe gaussiene și sinusoidale (vizibile în EDA Etapa 3).

---

### 3. Diagrama State Machine a Întregului Sistem

**Legendă obligatorie (Justificare):**

### Justificarea State Machine-ului ales:

Am ales o arhitectură de tip **"Interactive Prediction Loop"** deoarece aplicația este destinată utilizării umane directe, unde utilizatorul vrea să testeze scenarii ("Ce se întâmplă dacă e vânt puternic?").

**Stările principale sunt:**
1. **IDLE (Așteptare):** Aplicația așteaptă input de la utilizator prin UI (Streamlit).
2. **ACQUIRE_USER_INPUT:** Preia valorile slider-elor (Oră, Vânt, Presiune).
3. **PREPROCESS:** Transformă ora în Sin/Cos și normalizează valorile meteo folosind Scaler-ul salvat.
4. **INFERENCE (Predicție):** Rețeaua Neurală calculează temperatura brută.
5. **CALIBRATE (Bias Correction):** (Specific proiectului meu) Ajustează predicția brută pe baza temperaturii curente introduse de utilizator pentru a corecta offset-ul zilei.
6. **DISPLAY:** Afișează rezultatul și graficul de trend.

**Tranzițiile critice:**
- `CALCULATE_BTN_PRESS` -> Declanșează fluxul de predicție.
- `ERROR` -> Dacă modelul sau scalerul lipsesc, aplicația afișează un mesaj de eroare și cere rularea scripturilor de setup.

**Locație Diagramă:** `docs/state_machine.png` (Va trebui să adaugi o imagine simplă cu stările de mai sus în folderul docs).

---

### 4. Scheletul Complet al celor 3 Module

#### **Modul 1: Data Logging / Acquisition**
* **Locație:** `genereaza_date_logice.py` (integrat acum în fluxul de proiect)
* **Status:** Funcțional. Generează 5000 linii de date cu logică fizică.
* **Format Output:** CSV/TXT cu coloane: Data, Ora, Temperatura, Vânt, Presiune, Seninătate.

#### **Modul 2: Neural Network Module**
* **Locație:** `train_neural.py`
* **Status:** Arhitectură definită (Sequential, Dense Layers).
* **Compilare:** Modelul este compilat cu optimizatorul Adam și loss MSE.
* **Salvare:** Modelul neantrenat (sau antrenat sumar) se salvează în `models/model_meteo.keras`.

#### **Modul 3: Web Service / UI**
* **Locație:** `app_meteo.py`
* **Tehnologie:** Streamlit
* **Status:** Funcțional. 
    - Interfața se încarcă în browser.
    - Are slidere pentru input.
    - Comunică cu modelul salvat (`.keras`) și scaler-ul (`.pkl`).
    - Afișează predicția numerică și grafică.

---


---

## Checklist Final

- [x] Tabelul Nevoie → Soluție completat.
- [x] Declarație contribuție 100% date originale (Simulare).
- [x] Cod generare date funcțional (`genereaza_date_logice.py`).
- [x] Diagrama State Machine descrisă în text (urmează imaginea în docs).
- [x] Modul 1 (Generare) funcțional.
- [x] Modul 2 (Rețea Neurală) definit și funcțional (script antrenare).
- [x] Modul 3 (UI Streamlit) funcțional și testat (vezi screenshot).
- [x] Repository structurat corect.

---