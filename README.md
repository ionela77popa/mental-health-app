# Teen Mental Health — Depression Prediction System

## Abstract

Sistemul implementează un motor de decizie pentru detectarea depresiei la adolescenți pe baza unui set de variabile comportamentale și psihologice. Selecția modelului optim se realizează prin benchmarking comparativ pe 8 clasificatori supervizați, cu evaluare prin validare încrucișată stratificată și metrici multi-dimensionale.

---

## 1. Formularea problemei

Fie un vector de intrare $\mathbf{x} \in \mathbb{R}^8$:

$$\mathbf{x} = [x_1,\ x_2,\ x_3,\ x_4,\ x_5,\ x_6,\ x_7,\ x_8]$$

unde componentele corespund variabilelor:

| Index | Variabilă                       | Interval     |
| ----- | ------------------------------- | ------------ |
| $x_1$ | ore social media / zi           | $[1, 8]$     |
| $x_2$ | ore somn / zi                   | $[4, 9]$     |
| $x_3$ | utilizare ecran înainte de somn | $[0.5, 3.0]$ |
| $x_4$ | performanță academică (GPA)     | $[2.0, 4.0]$ |
| $x_5$ | activitate fizică (ore / zi)    | $[0.0, 2.0]$ |
| $x_6$ | nivel stres                     | $[1, 10]$    |
| $x_7$ | nivel anxietate                 | $[1, 10]$    |
| $x_8$ | nivel dependență social media   | $[1, 10]$    |

Funcția de decizie $f: \mathbb{R}^8 \to \{0, 1\}$ este:

$$\hat{y} = f(\mathbf{x}) = \begin{cases} 1 & \text{depresie prezentă} \\ 0 & \text{fără depresie} \end{cases}$$

---

## 2. Preprocesarea datelor

### 2.1 Normalizare (Z-score)

Fiecare caracteristică este standardizată pentru a elimina efectele de scală:

$$\tilde{x}_j = \frac{x_j - \mu_j}{\sigma_j}$$

unde $\mu_j$ și $\sigma_j$ sunt media și deviația standard **estimate exclusiv pe setul de antrenament**, evitând data leakage.

### 2.2 Partiționarea datelor

Setul de 1200 înregistrări este împărțit prin **Stratified HoldOut** cu proporția 70/30:

$$|D_{train}| = 840,\quad |D_{test}| = 360$$

Stratificarea asigură că distribuția etichetei $y$ (rată depresie: **2.6%**) este păstrată în ambele partiții — echivalent cu `cvpartition(Y, 'HoldOut', 0.3)` din MATLAB.

---

## 3. Evaluarea modelelor

### 3.1 Validare încrucișată stratificată

Fiecare clasificator candidat este evaluat prin **5-fold Stratified K-Fold CV** pe $D_{train}$:

$$\text{CV-Accuracy} = \frac{1}{K} \sum_{k=1}^{K} \text{Acc}(f_k,\ D_{val}^{(k)})$$

cu $K = 5$. Media și deviația standard oferă o estimare robustă a generalizării.

### 3.2 Metrici de performanță

Pentru seturi dezechilibrate (96.7% clasa 0, 3.3% clasa 1), acuratețea singură este insuficientă. Se calculează:

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

$$\text{ROC-AUC} = \int_0^1 TPR(FPR)\, d(FPR)$$

### 3.3 Criterii de selecție

Modelul optim $f^*$ este selectat după:

$$f^* = \underset{f \in \mathcal{F}}{\arg\max}\ \bigl(\text{CV-Acc}(f),\ F_1(f)\bigr)$$

---

## 4. Rezultate benchmark

Rulat pe `Teen_Mental_Health_Dataset.csv` (1200 instanțe, 8 predictori):

| Model                     | CV Accuracy       | HO Accuracy | F1         | ROC-AUC    |
| ------------------------- | ----------------- | ----------- | ---------- | ---------- |
| **★ Gradient Boosting**   | **99.76% ±0.29%** | **100.00%** | **1.0000** | **1.0000** |
| Logistic Regression       | 98.33% ±0.24%     | 98.06%      | 0.4615     | 0.9864     |
| KNN-5 _(MATLAB baseline)_ | 97.86% ±0.48%     | 98.06%      | 0.3636     | 0.9846     |
| KNN-7                     | 97.86% ±0.71%     | 97.78%      | 0.2000     | 0.9934     |
| SVC-RBF                   | 97.86% ±0.71%     | 97.78%      | 0.2000     | 0.9870     |
| Random Forest             | 97.62% ±0.38%     | 98.06%      | 0.3636     | 1.0000     |
| Extra Trees               | 97.62% ±0.53%     | 97.78%      | 0.2000     | 0.9997     |
| KNN-3                     | 97.62% ±1.00%     | 97.78%      | 0.3333     | 0.8257     |

**Concluzie:** Gradient Boosting depășește MATLAB KNN-5 cu **+1.94% CV-Accuracy** și **+0.6364 F1-score**, eliminând complet erorile pe setul de test.

---

## 5. Modelul câștigător — Gradient Boosting

Clasificatorul construiește un ansamblu aditiv de arbori de decizie slabi $h_t$:

$$F_T(\mathbf{x}) = \sum_{t=1}^{T} \alpha_t \cdot h_t(\mathbf{x})$$

unde fiecare $h_t$ minimizează reziduul:

$$h_t = \underset{h}{\arg\min}\ \sum_{i=1}^{n} \left[ y_i - F_{t-1}(\mathbf{x}_i) \right]^2$$

**Hiperparametri utilizați:** $T = 200$, $\eta = 0.05$ (learning rate), profunzime maximă $= 4$.

---

## 6. Exemplu real de predicție

**Caz MATLAB** (`nouCopil`): profil cu risc ridicat

$$\mathbf{x} = [7,\ 5,\ 2.5,\ 2.3,\ 1,\ 8,\ 9,\ 8]$$

| Variabilă             | Valoare | Interpretare |
| --------------------- | ------- | ------------ |
| ore social media      | 7 h     | ridicat      |
| ore somn              | 5 h     | sub medie    |
| utilizare ecran seara | 2.5 h   | ridicat      |
| GPA                   | 2.3     | scăzut       |
| activitate fizică     | 1 h     | redusă       |
| stres                 | 8/10    | sever        |
| anxietate             | 9/10    | sever        |
| dependență            | 8/10    | ridicat      |

$$\hat{y} = f(\mathbf{x}) = 1 \quad (\text{Depresie prezentă}),\quad P(\hat{y}=1 \mid \mathbf{x}) = 100\%$$

**Profil sănătos:**

$$\mathbf{x} = [2,\ 8,\ 0.5,\ 3.8,\ 1.8,\ 2,\ 1,\ 1] \implies \hat{y} = 0\quad (\text{Fără depresie})$$

---

## 7. Arhitectura sistemului

```
app.py                      ← interfață Streamlit
└── prediction.py           ← motor de predicție universal
    ├── PredictionEngine
    │   ├── _prepare_data()     Stratified HoldOut 70/30 + StandardScaler
    │   ├── _run_benchmark()    8 clasificatori × 5-fold CV
    │   ├── _refit_winner()     reantrenare model final
    │   ├── predict()           inferență sincronă  O(1)
    │   └── predict_async()     inferență async (ThreadPoolExecutor)
    ├── predict_sync()          shortcut Streamlit
    └── predict_async()         shortcut FastAPI / asyncio
```

Predicția asincronă rulează în `ThreadPoolExecutor` cu 4 workers, neblocând event loop-ul:

```python
result = await engine.predict_async([7, 5, 2.5, 2.3, 1, 8, 9, 8])
# → 1  (Depresie prezentă)
```

---

## 8. Instalare și rulare

```bash
pip install -r requirements.txt

# Benchmark standalone
python prediction.py

# Aplicație web
streamlit run app.py
```

**requirements.txt** include: `streamlit`, `scikit-learn`, `pandas`, `numpy`.

---

## Date

Dataset: `Teen_Mental_Health_Dataset.csv` — 1200 instanțe sintetice, 13 coloane, etichetă binară `depression_label`.
