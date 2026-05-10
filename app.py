import base64
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

from prediction import FEATURES, PredictionEngine


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

st.markdown(
    """
    <h1 class="app-title">
        <span class="app-title-line">KNN</span>
        <span class="app-title-line">Depression</span>
        <span class="app-title-line app-title-line--predictor">Predictor</span>
    </h1>
    """,
    unsafe_allow_html=True
)

# ==================================
# INCARCARE MODEL
# ==================================

@st.cache_resource(show_spinner="Antrenare model...")
def load_engine():
    e = PredictionEngine(DATA_FILE)
    e.initialize()
    return e

engine = load_engine()
accuracy = engine.holdout_accuracy

# ==================================
# BENCHMARK EXPANDER
# ==================================

# with st.expander(
#     f"Model activ: **{engine.best_model_name}** — Acuratețe: **{accuracy*100:.2f}%**",
#     expanded=False,
# ):
#     report = engine.report
#     if report:
#         rows = []
#         for m in report.ranked:
#             rows.append({
#                 "Model": ("★ " if m.name == report.best.name else "   ") + m.name,
#                 "CV Accuracy": f"{m.accuracy_cv:.4f} ±{m.accuracy_cv_std:.4f}",
#                 "HO Accuracy": f"{m.accuracy_holdout:.4f}",
#                 "F1": f"{m.f1:.4f}",
#                 "ROC-AUC": f"{m.roc_auc:.4f}",
#                 "Fit (s)": f"{m.fit_time_s:.3f}",
#                 "Pred (ms)": f"{m.predict_time_ms:.3f}",
#             })
#         import pandas as _pd
#         st.dataframe(_pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ==================================
# INPUTURI
# ==================================

st.markdown("## Introdu datele profilului:")

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

    prediction = engine.predict(
        [
            social_media,
            sleep_hours,
            screen_before_sleep,
            academic,
            physical,
            stress,
            anxiety,
            addiction,
        ]
    )

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
        const result = window.parent.document.querySelector('.result-box');

        if (main && result && window.parent.innerWidth <= 768) {
            main.scrollBy({ top: 500, behavior: "smooth" });
        } else if (main && result) {
            const mainRect = main.getBoundingClientRect();
            const resultRect = result.getBoundingClientRect();
            const offset = 16;

            main.scrollBy({
                top: resultRect.top - mainRect.top - offset,
                behavior: "smooth"
            });
        }
        </script>
        """,
        height=0,
    )
