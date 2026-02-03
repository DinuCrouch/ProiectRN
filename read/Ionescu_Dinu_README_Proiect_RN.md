# README - Proiect Final Rețele Neuronale

## 1. Identificare Proiect

| Câmp | Valoare |
|------|---------|
| **Student** | Ionescu Dinu |
| **Grupa / Specializare** | 633AB |
| **Disciplina** | Rețele Neuronale |
| **Instituție** | POLITEHNICA București – FIIR |
| **Link Repository GitHub** | https://github.com/DinuCrouch/Proiect-RN |
| **Acces Repository** | Public |
| **Stack Tehnologic** | Python (TensorFlow, Streamlit, Pandas, Scikit-Learn) |
| **Domeniul Industrial de Interes (DII)** | Smart Farming / Agricultură de Precizie |
| **Tip Rețea Neuronală** | MLP (Multi-Layer Perceptron) pentru Regresie |

### Rezultate Cheie (Versiunea Finală vs Etapa 6)

| Metric | Țintă Minimă | Rezultat Etapa 6 | Rezultat Final | Îmbunătățire | Status |
|--------|--------------|------------------|----------------|--------------|--------|
| R2 Score (Acuratețe Regresie) | ≥0.80 | 0.88 | **0.91** | +0.03 | [✓] |
| F1-Score (Detectare Îngheț) | ≥0.65 | 0.85 | **0.92** | +0.07 | [✓] |
| MAE (Eroare Absolută) | ≤1.5°C | 1.1°C | **0.8°C** | -0.3°C | [✓] |
| Latență Inferență | < 50ms | 45 ms | **35 ms** | -10 ms | [✓] |
| Contribuție Date Originale | ≥40% | 100% | **100%** | - | [✓] |
| Nr. Experimente Optimizare | ≥4 | 4 | **5** | +1 | [✓] |

### Declarație de Originalitate & Politica de Utilizare AI

**Acest proiect reflectă munca, gândirea și deciziile mele proprii.**

Utilizarea asistenților de inteligență artificială (ChatGPT, Claude, etc.) a fost folosită pentru debugging, generarea de boilerplate code (structură fișiere) și rafinarea documentației.

**Confirmare explicită:**

| Nr. | Cerință | Confirmare |
|-----|---------|------------|
| 1 | Modelul RN a fost antrenat **de la zero** (weights inițializate random). | [x] DA |
| 2 | Minimum **40% din date sunt contribuție originală** (generate de mine). | [x] DA |
| 3 | Codul este propriu sau sursele externe sunt **citate explicit**. | [x] DA |
| 4 | Arhitectura, codul și interpretarea rezultatelor reprezintă **muncă proprie**. | [x] DA |
| 5 | Pot explica și justifica **fiecare decizie importantă** cu argumente proprii. | [x] DA |

**Semnătură student:** *Ionescu Dinu*

---

## 2. Descrierea Nevoii și Soluția SIA

### 2.1 Nevoia Reală / Studiul de Caz

Fermele locale de dimensiuni mici și medii se confruntă cu pierderi semnificative din cauza **înghețului neprevăzut** și a variațiilor bruște de temperatură. Prognozele meteo generale (TV/Radio) oferă medii regionale, dar nu iau în calcul microclimatul specific fermei (vânt local, presiune, orientare).

Fermierii au nevoie de un sistem care să prezică temperatura exactă **în locația lor** pentru următoarele ore, permițându-le să activeze sistemele de protecție (încălzire sere, aspersoare anti-îngheț) doar atunci când este strict necesar, economisind energie.

### 2.2 Beneficii Măsurabile Urmărite

1. **Reducerea pierderilor de recoltă cu 80%** prin alertare timpurie a riscului de îngheț (< 2°C).
2. **Optimizarea costurilor energetice cu 30%** prin evitarea pornirii inutile a centralelor termice.
3. **Precizie locală:** Eroare medie sub 1°C față de termometrul din fermă (vs 3-4°C eroare la prognoza TV).

### 2.3 Tabel: Nevoie → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul** | **Modul software responsabil** | **Metric măsurabil** |
|---------------------------|--------------------------|--------------------------------|----------------------|
| Predicție precisă temperatură seră | Regresie Neurală pe date istorice + input curent | RN Module (`model_meteo.keras`) | MAE < 1.0 °C |
| Alertă risc îngheț | Logică de post-procesare (Threshold < 2°C) | UI (`app_ferma.py`) | F1-Score > 0.80 |
| Calibrare cu condițiile curente | Bias Correction (Utilizatorul introduce temp. acum) | UI Logic | Reducere eroare offset cu 100% |

---

## 3. Dataset și Contribuție Originală

### 3.1 Sursa și Caracteristicile Datelor

| Caracteristică | Valoare |
|----------------|---------|
| **Origine date** | Simulare Fizică (Algoritmică) |
| **Sursa concretă** | Script propriu `genereaza_date_logice.py` |
| **Număr total observații finale (N)** | 2400 ore simulate |
| **Număr features** | 7 (Luna_sin, Luna_cos, Ora_sin, Ora_cos, Vânt, Presiune, Seninătate) |
| **Tipuri de date** | Numerice (Float) + Temporale |
| **Format fișiere** | CSV Procesat / TXT Raw |
| **Perioada simulată** | Martie - iunie 2023 |

### 3.2 Contribuția Originală (100%)

| Câmp | Valoare |
|------|---------|
| **Total observații finale (N)** | 2400 |
| **Observații originale (M)** | 2400 |
| **Procent contribuție originală** | **100%** |
| **Tip contribuție** | Date generate prin simulare fizică |
| **Locație cod generare** | `src/data_acquisition/genereaza_date_logice.py` |

**Descriere metodă generare:**
Am implementat un algoritm care simulează termodinamica atmosferică:
1.  **Cicluri Sinusoidale:** Pentru variația zi/noapte (maxim ora 14) și sezonieră.
2.  **Corelații Fizice:** Vântul reduce temperatura , Seninătatea o crește ziua  și o scade noaptea (Pierdere căldură).
3.  **Zgomot Gaussian:** Adăugat pentru a simula erorile de măsurare și imprevizibilitatea naturii.

### 3.3 Preprocesare și Split Date

| Set | Procent | Număr Observații |
|-----|---------|------------------|
| Train | 70% | 1680 |
| Validation | 15% | 360 |
| Test | 15% | 360 |

**Preprocesări aplicate:**
- **Feature Engineering:** Transformarea timpului (0-23 ore) în coordonate ciclice (`sin`, `cos`) pentru a păstra continuitatea (ora 23 aproape de 0).
- **Normalizare:** MinMaxScaler (0-1) pentru toate variabilele de intrare.
- **Salvare Scaler:** `joblib` pentru consistență în UI.

---

## 4. Arhitectura SIA și State Machine

### 4.1 Cele 3 Module Software

| Modul | Tehnologie | Funcționalitate Principală | Locație în Repo |
|-------|------------|---------------------------|-----------------|
| **Data Acquisition** | Python (Pandas/Numpy) | Generare date logice corelate fizic | `src/data_aquisition/genereaza_date_logice.py` |
| **Neural Network** | TensorFlow/Keras | Model MLP Regresie + Salvare Scaler | `src/neural_network/train_neural.py` |
| **Web Service / UI** | Streamlit | Interfață interactivă cu Bias Correction | `src/apps/app_fema.py` |

### 4.2 State Machine

**Stări principale:**

| Stare | Descriere | Condiție Intrare |
|-------|-----------|------------------|
| `IDLE` | Așteptare input utilizator | Start aplicație |
| `ACQUIRE_INPUT` | Preluare date slidere (Vânt, Oră, etc.) | Interacțiune UI |
| `PREPROCESS` | Scalare și transformare trigonometrică | Buton "Calculează" |
| `INFERENCE` | Predicție cu Modelul RN (Brut) | Date preprocesate |
| `BIAS_CORRECTION` | Ajustare predicție cu temp. curentă | Predicție brută + Input Temp |
| `DECISION` | Verificare prag îngheț (< 2°C) | Temp. Finală |
| `DISPLAY` | Afișare metrici și grafic trend | Decizie luată |

**Justificare:** Arhitectura include starea critică de **BIAS_CORRECTION** care combină inteligența modelului (trend) cu realitatea din teren (punct de start), esențială pentru precizia în agricultură.

---

## 5. Modelul RN – Antrenare și Optimizare

### 5.1 Arhitectura Rețelei Neuronale

Input (7 features) → Dense(128, ReLU) → Dropout(0.2) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Linear) Output: Temperatura (°C)

**Justificare:** Am ales MLP (Multi-Layer Perceptron) deoarece datele sunt tabulare și relațiile sunt neliniare dar nu spațiale (nu necesită CNN) sau secvențiale complexe (deoarece folosim input-ul curent pentru calibrare).

### 5.2 Hiperparametri Finali

| Hiperparametru | Valoare Finală | Justificare |
|----------------|----------------|-------------|
| Learning Rate | 0.001 | Convergență stabilă fără oscilații mari. |
| Batch Size | 32 | Echilibru bun între viteză și actualizarea gradienților. |
| Epochs | 200 | Regresia necesită multe iterații pentru fine-tuning eroare. |
| Optimizer | Adam | Standardul pentru date tabulare. |
| Loss Function | MSE | Penalizează erorile mari (outlierii). |

### 5.3 Experimente de Optimizare

| Exp# | Modificare | R2 Score | MAE | Observații |
|------|------------|----------|-----|------------|
| Baseline | Model Simplu (64 neuroni) | 0.72 | 3.8°C | Underfitting masiv. |
| Exp 1 | +Feature Eng (Sin/Cos) | 0.85 | 1.5°C | Modelul înțelege ciclul zi/noapte. |
| Exp 2 | Deep Arch (128+64+32) | 0.88 | 1.1°C | Captură mai bună a neliniarităților. |
| Exp 3 | Dropout 0.2 | 0.89 | 1.0°C | Generalizare mai bună pe test. |
| **FINAL** | **Exp 3 + Augmentare Extreme** | **0.91** | **0.8°C** | **Performanță maximă pe îngheț.** |

---

## 6. Performanță Finală și Analiză Erori

### 6.1 Metrici pe Test Set (Model Optimizat)

| Metric | Valoare | Target Minim | Status |
|--------|---------|--------------|--------|
| **R2 Score** | **0.91** | ≥0.80 | [✓] |
| **MAE** | **0.8°C** | ≤1.0°C | [✓] |
| **F1-Score (Îngheț)** | **0.92** | ≥0.65 | [✓] |

### 6.2 Confusion Matrix (Risc Îngheț < 2°C)

**Locație:** `docs/confusion_matrix.png`

**Interpretare:**
- **Recall Îngheț:** Modelul detectează 95% din cazurile de îngheț reale.
- **False Negatives:** Foarte puține (Critic pentru fermă - nu vrem să ratăm un îngheț).
- **False Positives:** Acceptabile (Mai bine pornim centrala degeaba decât să pierdem recolta).

### 6.3 Validare în Context Industrial

Pentru o seră de legume:
- Un **False Negative** (Model zice Cald, Real e Îngheț) costă **10.000 EUR** (recoltă distrusă).
- Un **False Positive** (Model zice Îngheț, Real e Cald) costă **50 EUR** (gaz consumat inutil).
- Modelul optimizat prioritizează Recall-ul, minimizând riscul de 10.000 EUR.

---

## 7. Aplicația Software Finală

### 7.1 Modificări Implementate în Etapa 6

| Componentă | Modificare | Justificare |
|------------|------------|-------------|
| **Logică UI** | Adăugare **Bias Correction** | Calibrarea predicției cu temperatura curentă elimină eroarea de offset a zilei. |
| **Vizualizare** | Grafic Trend 24h | Fermierul vede nu doar temperatura, ci și tendința (scade brusc?). |
| **Feedback** | Alertă Vizuală (Diferență Temp) | Arată clar cât de mult diferă prognoza AI de realitatea momentană. |

### 7.2 Screenshot UI
**Locație:** `docs\interfata_utilizator_intrare.png`
Demonstrează interfața Streamlit rulând cu modelul final, afișând temperatura calibrată și graficul de evoluție.

---

## 8. Structura Repository-ului Final

---

## 9. Instrucțiuni de Instalare și Rulare

1.  **Instalare dependențe:**
    ```bash
    pip install pandas numpy tensorflow scikit-learn matplotlib streamlit
    ```

2.  **Generare și Procesare Date:**
    ```bash
    python src/generate_data/genereaza_date_logice.py
    python src/preprocessing/preprocess.py
    ```

3.  **Antrenare Model (produce metrici și grafice):**
    ```bash
    python src/neural_network/train_neural.py
    ```

4.  **Rulare Aplicație UI:**
    ```bash
    python -m streamlit run src/apps/app_ferma.py
    ```

---

## 10. Concluzii

Am dezvoltat un sistem SIA robust pentru protecția culturilor agricole. Prin combinarea unei Rețele Neurale capabile să învețe modele fizice (zi/noapte, vânt) cu o logică de calibrare în timp real (Bias Correction), am atins o precizie de sub 1°C, suficientă pentru automatizarea serelor.

**Lecția principală:** Într-o problemă de regresie aplicată (meteo), calitatea datelor (Feature Engineering - Sin/Cos) și adaptarea la context (Bias Correction) sunt mai importante decât complexitatea brută a modelului.