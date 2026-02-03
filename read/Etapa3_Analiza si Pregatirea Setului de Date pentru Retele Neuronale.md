# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Ionescu Dinu  
**Data:** 20/11/2025

---

## Introducere

Acest document descrie activitățile realizate în **Etapa 3**, în care s-a analizat și preprocesat setul de date necesar pentru proiectul de **Predicție a Temperaturii** (Problemă de Regresie). Scopul a fost transformarea datelor brute (serii de timp simulate/reale) într-un format numeric normalizat, optim pentru antrenarea unei rețele neuronale de tip Feed-Forward (Dense), asigurând separarea corectă a datelor pentru a evita overfitting-ul.

---

##  1. Structura Repository-ului Github (versiunea Etapei 3)

project-name/
├── README.md
├── docs/
│   └── datasets/          # descriere seturi de date, surse, diagrame
├── data/
│   ├── raw/               # date brute
│   ├── processed/         # date curățate și transformate
│   ├── train/             # set de instruire
│   ├── validation/        # set de validare
│   └── test/              # set de testare
├── src/
│   ├── preprocessing/     # funcții pentru preprocesare
│   ├── data_acquisition/  # generare / achiziție date (dacă există)
│   └── neural_network/    # implementarea RN (în etapa următoare)
└── requirements.txt       # dependențe Python (dacă aplicabil)

---

##  2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** Simulare programatică bazată pe legi fizice simplificate (cicluri termodinamice zilnice și anuale).
* **Modul de achiziție:** Generare programatică (Script: `genereaza_date_logice.py`).
* **Perioada / condițiile colectării:** S-au simulat date orare pentru o perioadă de aproximativ 7 luni (5000 de intrări), acoperind tranziția iarnă-primăvară-vară pentru a captura varianța sezonieră.

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** 2.400 linii.
* **Număr de caracteristici (features):** * *Inițial:* 4 (Data, Ora, Vânt, Presiune, Seninătate).
    * *După procesare:* 7 features de intrare + 1 target.
* **Tipuri de date:** Numerice (float/int) și Temporale (convertite ulterior).
* **Format fișiere:** CSV (procesat), TXT (raw).

### 2.3 Descrierea fiecărei caracteristici

| **Caracteristică** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
|-------------------|---------|-------------|---------------|--------------------|
| data | temporal | YYYY-MM-DD | Data calendaristică | 2023-01-01 -> 2023-08-01 |
| ora | numeric | 0-23 | Ora din zi | 0 – 23 |
| grad_seninatate | numeric | % | Cât de senin este cerul (impact solar) | 0 – 100 |
| viteza_vant | numeric | km/h | Viteza vântului (efect de răcire) | 0 – 60 |
| presiune | numeric | hPa | Presiunea atmosferică | 990 – 1030 |
| **temperatura** | **numeric (Target)** | **°C** | **Variabila prezisă** | **-5.0 – +35.0** |

---

##  3. Analiza Exploratorie a Datelor (EDA)

### 3.1 Statistici descriptive aplicate

* **Distribuția Targetului:** Temperatura urmează o distribuție pseudo-normală, centrată pe mediile sezoniere, cu variații zilnice (ciclu zi/noapte).
* **Corelații:** * Corelație pozitivă puternică între `ora` (ziua) și `temperatura`.
    * Corelație pozitivă între `grad_seninatate` și `temperatura` (ziua).
    * Corelație negativă între `viteza_vant` și `temperatura`.

### 3.2 Analiza calității datelor

* **Valori lipsă:** 0% (Dat fiind că datele sunt generate controlat, nu există valori NaN).
* **Outlieri:** Introduși intenționat prin zgomot gaussian (`np.random.normal`).

### 3.3 Probleme identificate și Soluții

1.  **Ciclicitatea Timpului:** Ora 23:00 este foarte apropiată de Ora 00:00, dar numeric (23 vs 0) sunt departe. 
    * *Soluție:* Transformare în coordonate polare (Sinus/Cosinus).
2.  **Diferențe de Scară:** Presiunea (1000+) domină numeric Vântul (0-60).
    * *Soluție:* Normalizare Min-Max în intervalul [0, 1].

---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

* Eliminarea coloanelor textuale brute (`data`, `ora`) după extragerea informației numerice.
* Verificare consistență tipuri de date (conversie la `float32` pentru TensorFlow).

### 4.2 Transformarea caracteristicilor (Feature Engineering)

Pentru a ajuta rețeaua să înțeleagă periodicitatea, am aplicat transformări trigonometrice:
* **Ora:** Transformată în `ora_sin` și `ora_cos`.
* **Luna:** Transformată în `luna_sin` și `luna_cos`.
* **Normalizare:** S-a utilizat `MinMaxScaler` pe toate coloanele numerice (Vânt, Presiune, Seninătate, Temperatură) pentru a aduce valorile între 0 și 1. Scaler-ul a fost salvat (`scaler_meteo.pkl`) pentru a putea denormaliza predicțiile ulterior.

### 4.3 Structurarea seturilor de date

S-a realizat o împărțire **randomizată** (shuffle) pentru a evita bias-ul temporal (ex: să nu avem doar iarnă în Train și vară în Test).

**Împărțire realizată:**
* **Train (70%):** Folosit pentru ajustarea ponderilor (backpropagation).
* **Validation (15%):** Folosit la finalul fiecărei epoci pentru monitorizarea Loss-ului și prevenirea Overfitting-ului.
* **Test (15%):** Date complet noi, folosite doar la final pentru generarea graficelor de performanță.

### 4.4 Salvarea rezultatelor preprocesării

* Datele au fost salvate fizic în CSV-uri separate pentru trasabilitate:
    * `data/train/train.csv`
    * `data/validation/validation.csv`
    * `data/test/test.csv`

---

##  5. Fișiere Generate în Această Etapă

* `data/raw/date_meteo.txt` – Dataset brut.
* `models/scaler_meteo.pkl` – Obiectul de scalare (esențial pentru UI).
* `data/processed/*.csv` – Seturile de date pregătite pentru `train_neural.py`.
* `src/preprocess.py` – Codul sursă pentru pipeline-ul de date.

---

##  6. Stare Etapă

- [x] Structură repository configurată
- [x] Dataset analizat (EDA realizată implicit prin generare controlată)
- [x] Date preprocesate (Encoding Ciclic + Normalizare)
- [x] Seturi train/val/test generate și salvate pe disc
- [x] Documentație actualizată