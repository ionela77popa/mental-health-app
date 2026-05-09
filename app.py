import base64
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
CSS_FILE = ASSETS_DIR / "css" / "style.css"
DATA_FILE = BASE_DIR / "Teen_Mental_Health_Dataset.csv"
DEPRESSION_IMAGE = ASSETS_DIR / "img" / "1.png"
HEALTHY_IMAGE = ASSETS_DIR / "img" / "0.png"


# ==================================
# CONFIGURARE PAGINA
# ==================================

st.set_page_config(
    page_title="KNN Depression Predictor",
    layout="centered"
)

# ==================================
# CSS
# ==================================

with open(CSS_FILE, encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ==================================
# TITLU
# ==================================

st.title("KNN Depression Predictor")

# ==================================
# INCARCARE DATE
# ==================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

data = load_data()

# ==================================
# PREDICTORI
# ==================================

features = [
    "daily_social_media_hours",
    "sleep_hours",
    "screen_time_before_sleep",
    "academic_performance",
    "physical_activity",
    "stress_level",
    "anxiety_level",
    "addiction_level"
]

X = data[features]
Y = data["depression_label"]

# ==================================
# NORMALIZARE
# ==================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==================================
# IMPARTIRE DATE
# ==================================

XTrain, XTest, YTrain, YTest = train_test_split(
    X_scaled,
    Y,
    test_size=0.3,
    random_state=42
)

# ==================================
# MODEL KNN
# ==================================

k = 5

model = KNeighborsClassifier(
    n_neighbors=k
)

model.fit(XTrain, YTrain)

# ==================================
# ACURATETE
# ==================================

YPred = model.predict(XTest)

accuracy = accuracy_score(YTest, YPred)

# ==================================
# INPUTURI
# ==================================

st.markdown("## Introdu datele profilului")

social_media = st.slider(
    "Număr ore social media / zi",
    1.0,
    8.0,
    5.0,
    0.1
)

sleep_hours = st.slider(
    "Număr ore somn / zi",
    4.0,
    9.0,
    7.0,
    0.1
)

screen_before_sleep = st.slider(
    "Utilizare telefon înainte de somn",
    0.5,
    3.0,
    1.5,
    0.1
)

academic = st.slider(
    "Performanță academică",
    2.0,
    4.0,
    3.0,
    0.01
)

physical = st.slider(
    "Nivel activitate fizică",
    0.0,
    2.0,
    1.0,
    0.1
)

stress = st.slider(
    "Nivel stres",
    1,
    10,
    5
)

anxiety = st.slider(
    "Nivel anxietate",
    1,
    10,
    5
)

addiction = st.slider(
    "Nivel dependență social media",
    1,
    10,
    5
)

# ==================================
# BUTON PREDICT
# ==================================

if st.button("PREDICT", key="predict_button"):

    user_data = np.array([[
        social_media,
        sleep_hours,
        screen_before_sleep,
        academic,
        physical,
        stress,
        anxiety,
        addiction
    ]])

    user_scaled = scaler.transform(user_data)

    prediction = model.predict(user_scaled)[0]

    class TypesStatus:
        DEPRESION = 1
        SANATOS = 0

    class ImagesForTypes:
        DEPRESION = str(DEPRESSION_IMAGE)
        SANATOS = str(HEALTHY_IMAGE)

    def img_html(path: str) -> str:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return (
            f'<div style="display:flex;justify-content:center;margin-top:1em">'
            f'<img src="data:image/png;base64,{b64}" width="250"/>'
            f'</div>'
        )

    if prediction == TypesStatus.DEPRESION:

        st.markdown(
            """
            <div class="result-box">
            Depresie prezentă
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(img_html(ImagesForTypes.DEPRESION), unsafe_allow_html=True)

    else:

        st.markdown(
            """
            <div class="result-box">
            Fără depresie
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(img_html(ImagesForTypes.SANATOS), unsafe_allow_html=True)

    components.html(
        """
        <script>
        const main = window.parent.document.querySelector('[data-testid="stMain"]');
        const scrollTarget = main ?? window.parent;
        scrollTarget.scrollBy({ top: 1500, behavior: "smooth" });
        </script>
        """,
        height=0,
    )