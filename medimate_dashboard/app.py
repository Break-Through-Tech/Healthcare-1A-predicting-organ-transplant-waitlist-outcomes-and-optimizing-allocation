from pathlib import Path
import json
import mimetypes
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MediMate Kidney Waitlist Risk Explorer",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "medimate_pipeline"

EDA = PIPELINE / "01_eda"
CLEAN = PIPELINE / "02_data_cleaning"
PREP = PIPELINE / "03_model_preparation"
CLASS = PIPELINE / "04_classification"
SURV = PIPELINE / "05_survival"
FINAL = PIPELINE / "06_final_analysis"

CLASS_OUT = CLASS / "outputs"
SURV_OUT = SURV / "outputs"
FINAL_OUT = FINAL / "outputs"

GITHUB = (
    "https://github.com/Break-Through-Tech/"
    "Healthcare-1A-predicting-organ-transplant-waitlist-outcomes-and-optimizing-allocation"
    "/tree/Prince-Subedi"
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
<style>
.block-container {
    max-width: 1320px;
    padding-top: 1.8rem;
    padding-bottom: 4rem;
}

h1, h2, h3 {
    letter-spacing: -0.025em;
}

.hero {
    padding: 2.2rem 2.4rem;
    border: 1px solid #dfe6ea;
    border-radius: 18px;
    background: linear-gradient(135deg, #f2fafb 0%, #ffffff 68%);
    margin-bottom: 1.6rem;
}

.hero-title {
    font-size: 2.6rem;
    font-weight: 760;
    color: #14212b;
    line-height: 1.08;
}

.hero-subtitle {
    margin-top: 0.8rem;
    color: #566573;
    max-width: 900px;
    font-size: 1.05rem;
    line-height: 1.65;
}

.simple {
    border-left: 4px solid #176B87;
    background: #f3f9fa;
    padding: 1rem 1.1rem;
    border-radius: 0 10px 10px 0;
    margin: 0.9rem 0 1.3rem 0;
}

.caution {
    border-left: 4px solid #b66a00;
    background: #fff9ef;
    padding: 1rem 1.1rem;
    border-radius: 0 10px 10px 0;
    margin: 0.9rem 0 1.3rem 0;
}

.research {
    border: 1px solid #edcaca;
    background: #fff8f8;
    padding: 1.05rem 1.15rem;
    border-radius: 12px;
    margin: 1rem 0;
}

.stage-card {
    border: 1px solid #e0e6ea;
    background: #fbfcfd;
    padding: 1rem;
    border-radius: 13px;
    min-height: 135px;
}

.stage-number {
    color: #176B87;
    font-weight: 700;
    font-size: .85rem;
}

.stage-title {
    font-weight: 700;
    margin-top: .15rem;
}

.stage-text {
    color: #677581;
    font-size: .86rem;
    line-height: 1.4;
    margin-top: .4rem;
}

.term {
    border-bottom: 1px dotted #176B87;
    cursor: help;
}

.footer {
    border-top: 1px solid #e2e7ea;
    color: #7a8791;
    font-size: .82rem;
    margin-top: 3rem;
    padding-top: 1.25rem;
}

[data-testid="stMetric"] {
    border: 1px solid #e1e7ea;
    border-radius: 12px;
    padding: .75rem .85rem;
    background: white;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# GLOSSARY
# ============================================================

GLOSSARY = {
    "Adverse outcome":
        "Death while waiting or removal from the waitlist because the patient became too sick for transplant.",

    "Censoring":
        "The final outcome was not observed during the available follow-up period.",

    "ROC-AUC":
        "Measures how well a classification model ranks higher-risk cases above lower-risk cases. "
        "0.5 is random ranking and 1.0 is perfect ranking.",

    "PR-AUC":
        "Precision-Recall Area Under the Curve. This focuses on finding the positive outcome and is especially useful when that outcome is uncommon.",

    "Precision":
        "Among registrations predicted above the selected risk threshold, the proportion that actually experienced the adverse outcome.",

    "Recall":
        "Among registrations that actually experienced the adverse outcome, the proportion identified by the model.",

    "Top-decile lift":
        "Compares the adverse-event rate among the highest predicted-risk 10% with the overall adverse-event rate.",

    "C-index":
        "A survival-model ranking measure. Higher values mean the model more often assigns higher risk to the patient who experiences the event earlier.",

    "Competing risk":
        "An event that prevents the event being studied from occurring. Receiving a transplant competes with dying or becoming too sick while still waiting.",

    "Cumulative incidence":
        "The estimated probability of experiencing a particular event by a given time while accounting for competing events.",

    "Hazard ratio":
        "A relative comparison of how quickly an event occurs between groups in a survival model.",

    "SHAP":
        "A model-explanation method that estimates how much each feature pushes predictions higher or lower. It explains model behavior, not causation.",

    "Temporal split":
        "Training on earlier years and evaluating on later years instead of randomly mixing all years.",

    "Patient-safe split":
        "A split where the same patient cannot appear in both training and evaluation data.",

    "CPRA":
        "Calculated Panel Reactive Antibody. A transplant-related measure of sensitization that can affect compatibility with potential donors.",

    "XGBoost":
        "A machine-learning method that combines many decision trees to make predictions.",

    "Cox model":
        "A traditional survival-analysis model that estimates how patient characteristics are associated with the rate of experiencing an event.",

    "Aalen-Johansen estimator":
        "A method for estimating event probabilities when more than one competing event can occur.",

    "Calibration":
        "How closely predicted probabilities agree with the outcomes that actually occur.",

    "Feature importance":
        "A measure of how much a model relied on a variable when making predictions. Importance does not mean causation.",
}


# ============================================================
# HELPERS
# ============================================================

def simple(text):
    st.markdown(
        f'<div class="simple"><b>In simple terms</b><br>{text}</div>',
        unsafe_allow_html=True,
    )


def caution(text):
    st.markdown(
        f'<div class="caution">{text}</div>',
        unsafe_allow_html=True,
    )


def research_warning():
    st.markdown(
        """
        <div class="research">
        <b>Research use only</b><br>
        These analyses describe patterns in historical waitlist data.
        The models are not clinically validated and should not be used to determine
        transplant eligibility, organ allocation, treatment, or individual patient care.
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric(col, label, value, note=None, glossary=None):
    with col:
        st.metric(
            label,
            value,
            help=GLOSSARY.get(glossary or label),
            border=True,
        )
        if note:
            st.caption(note)


def read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_text(path):
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def show_csv(path, title=None):
    df = read_csv(path)
    if df is None:
        st.info(f"{path.name} is not available.")
        return
    if title:
        st.markdown(f"**{title}**")
    st.dataframe(df, hide_index=True)


def show_json(path):
    obj = read_json(path)
    if obj is not None:
        st.json(obj)
    else:
        st.info(f"{path.name} is not available.")


def show_image(path, caption=None):
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"{path.name} is not available.")


def download_file(path, label=None, key=None):
    if not path.exists():
        return
    mime, _ = mimetypes.guess_type(str(path))
    st.download_button(
        label or f"Download {path.name}",
        data=path.read_bytes(),
        file_name=path.name,
        mime=mime or "application/octet-stream",
        key=key,
    )


def notebook_panel(stage, filename, key):
    path = stage / filename
    if not path.exists():
        st.warning(f"Notebook not found: {filename}")
        return

    st.success(f"Notebook available: `{filename}`")
    download_file(
        path,
        f"Download {filename}",
        key=f"nb_{key}",
    )


def output_files(directory):
    if not directory.exists():
        return []

    allowed = {".csv", ".json", ".png", ".txt", ".md"}

    return sorted(
        [
            p for p in directory.iterdir()
            if p.is_file()
            and p.suffix.lower() in allowed
            and not p.name.startswith(".")
        ],
        key=lambda p: p.name.lower(),
    )


def output_explorer(directory, key):
    files = output_files(directory)

    if not files:
        st.info("There are no separate public output files for this stage.")
        return

    selected_name = st.selectbox(
        "Select an output file",
        [p.name for p in files],
        key=f"select_{key}",
    )

    path = directory / selected_name
    suffix = path.suffix.lower()

    size_kb = path.stat().st_size / 1024
    st.caption(f"{path.name} · {size_kb:,.1f} KB")

    if suffix == ".csv":
        show_csv(path)

    elif suffix == ".png":
        show_image(path)

    elif suffix == ".json":
        show_json(path)

    elif suffix in {".txt", ".md"}:
        text = read_text(path)
        if text:
            st.markdown(text)

    download_file(
        path,
        f"Download {path.name}",
        key=f"download_{key}_{path.name}",
    )


def stage_readme(stage):
    path = stage / "README.md"
    if path.exists():
        with st.expander("About this pipeline stage"):
            st.markdown(read_text(path) or "")


def technical_only(title, body):
    if TECHNICAL:
        with st.expander(title):
            body()


def section(title, subtitle=None):
    st.header(title)
    if subtitle:
        st.caption(subtitle)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("MediMate")
    st.caption("Kidney Waitlist Risk Explorer")

    PAGE = st.radio(
        "Project sections",
        [
            "Overview",
            "Project Pipeline",
            "01 · Exploratory Data Analysis",
            "02 · Data Cleaning",
            "03 · Model Preparation",
            "04 · Classification Modeling",
            "05 · Survival Modeling",
            "06 · Final Analysis",
            "Fairness & Explainability",
            "Reproducibility & Files",
            "Glossary",
        ],
    )

    TECHNICAL = st.toggle(
        "Show technical detail",
        value=True,
        help="Turn this off for a simpler overview. Turn it on to see model tables, settings, and detailed outputs.",
    )

    st.divider()

    with st.expander("Quick glossary"):
        for name, definition in GLOSSARY.items():
            st.markdown(f"**{name}**")
            st.caption(definition)

    st.divider()

    st.link_button(
        "View GitHub repository",
        GITHUB,
    )

    st.warning(
        "Research use only — not a clinical or organ-allocation tool."
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">
    <div class="hero-title">MediMate Kidney Waitlist Risk Explorer</div>
    <div class="hero-subtitle">
        A complete view of the kidney waitlist analysis — from exploratory data
        analysis and preprocessing through classification, survival modeling,
        subgroup evaluation, explainability, and the final combined findings.
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# OVERVIEW
# ============================================================

if PAGE == "Overview":

    section(
        "Project Overview",
        "What problem was studied, what was built, and what the main results mean.",
    )

    st.markdown(
        """
        Patients on the kidney transplant waitlist can experience very different
        outcomes. Some receive a transplant, some continue waiting, and others
        may die or become too sick for transplant.

        This project asks two main questions:

        **1. Can we identify registrations at higher risk of an adverse outcome within one year?**

        **2. Can we model how that risk changes over time while accounting for transplantation?**
        """
    )

    simple(
        "The project is about understanding and ranking risk in historical waitlist data. "
        "It is not an organ-allocation system and it does not decide who should receive a transplant."
    )

    st.subheader("Main results")

    cols = st.columns(4)

    metric(
        cols[0],
        "ROC-AUC",
        "0.674",
        "2024 classification test set",
        "ROC-AUC",
    )

    metric(
        cols[1],
        "Top-decile lift",
        "2.53×",
        "Highest predicted-risk 10%",
        "Top-decile lift",
    )

    metric(
        cols[2],
        "Survival C-index",
        "0.678",
        "2024 survival test set",
        "C-index",
    )

    metric(
        cols[3],
        "High vs low 1-year risk",
        "3.87×",
        "Survival risk groups",
        "Cumulative incidence",
    )

    st.subheader("Strongest shared model signals")

    left, right = st.columns([1, 1.3])

    with left:
        st.markdown(
            """
            The most consistent variables across the classification and survival
            approaches were:

            1. **Age**
            2. **Functional status**
            3. **Region**
            4. **Ethnicity code**
            5. **Dialysis status**
            """
        )

        simple(
            "Age and functional status were the two strongest signals that appeared "
            "across both modeling approaches."
        )

    with right:
        show_image(
            FINAL_OUT / "combined_feature_importance.png",
            "Combined feature importance across classification and survival modeling.",
        )

    research_warning()


# ============================================================
# PIPELINE
# ============================================================

elif PAGE == "Project Pipeline":

    section(
        "Complete Analysis Pipeline",
        "How the project moved from raw waitlist data to the final findings.",
    )

    stages = [
        (
            "01",
            "Exploratory Data Analysis",
            "Understand patients, outcomes, missingness, repeated registrations, and follow-up.",
        ),
        (
            "02",
            "Data Cleaning",
            "Create reliable dates and variables and flag invalid or unusual values.",
        ),
        (
            "03",
            "Model Preparation",
            "Define the one-year target, select features, and make patient-safe temporal splits.",
        ),
        (
            "04",
            "Classification",
            "Predict which registrations have higher one-year adverse-outcome risk.",
        ),
        (
            "05",
            "Survival",
            "Study when adverse outcomes occur while treating transplant as a competing event.",
        ),
        (
            "06",
            "Final Analysis",
            "Combine both approaches and create the final tables, plots, and conclusions.",
        ),
    ]

    for i in range(0, 6, 3):
        cols = st.columns(3)
        for col, (number, name, description) in zip(cols, stages[i:i+3]):
            with col:
                st.markdown(
                    f"""
<div class="stage-card">
<div class="stage-number">STAGE {number}</div>
<div class="stage-title">{name}</div>
<div class="stage-text">{description}</div>
</div>
""",
                    unsafe_allow_html=True,
                )

    st.markdown("### Modeling flow")

    st.code(
        """OPTN waitlist data
        ↓
Exploratory data analysis
        ↓
Data cleaning
        ↓
Model preparation
        ↓
┌───────────────────────┬────────────────────────┐
│ One-year classification│ Time-to-event survival │
└───────────────────────┴────────────────────────┘
        ↓
Fairness + explainability
        ↓
Final combined analysis"""
    )

    simple(
        "Classification answers whether an adverse event happens within a fixed period. "
        "Survival analysis adds the question of when it happens."
    )


# ============================================================
# 01 EDA
# ============================================================

elif PAGE == "01 · Exploratory Data Analysis":

    section(
        "01 · Exploratory Data Analysis",
        "Understanding the waitlist population before making modeling decisions.",
    )

    cols = st.columns(4)

    metric(
        cols[0],
        "Waitlist registrations",
        "494,862",
        "Analytic cohort",
    )

    metric(
        cols[1],
        "Unique patients",
        "414,187",
        "Some patients had repeated registrations",
    )

    metric(
        cols[2],
        "Transplanted",
        "47.37%",
        "234,429 registrations",
    )

    metric(
        cols[3],
        "Raw adverse outcome",
        "14.09%",
        "Death or removal-too-sick",
        "Adverse outcome",
    )

    simple(
        "A registration is not the same thing as a patient. Some patients entered "
        "the waitlist more than once, so the dataset contains more registrations than unique patients."
    )

    st.subheader("Observed outcomes")

    outcomes = pd.DataFrame(
        {
            "Outcome": [
                "Transplanted",
                "Still waiting",
                "Removed - administrative",
                "Transplanted elsewhere",
                "Removed too sick",
                "Died",
                "Unknown",
            ],
            "Registrations": [
                234429,
                103300,
                45332,
                37243,
                35219,
                34524,
                4815,
            ],
            "Percent": [
                47.37,
                20.87,
                9.16,
                7.53,
                7.12,
                6.98,
                0.97,
            ],
        }
    )

    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.dataframe(outcomes, hide_index=True)

    with c2:
        chart_data = outcomes.set_index("Outcome")[["Percent"]]
        st.bar_chart(chart_data)

    st.subheader("What stood out during EDA")

    st.markdown(
        """
        - Older age groups showed higher raw adverse-event rates.
        - Patients with poorer functional status had higher adverse-event rates.
        - Dialysis status was associated with outcome differences.
        - Blood type affected transplant patterns and waiting time.
        - Regional differences were present.
        - Recent listing years had substantially shorter follow-up.
        - Some patients had multiple waitlist registrations.
        """
    )

    st.subheader("CPRA missingness changed over time")

    st.markdown(
        """
        CPRA was essentially complete from **2015 through 2022**, but approximately
        **80% was missing in 2023–2026**.
        """
    )

    caution(
        "<b>Why this matters:</b> A feature can look useful in historical data but "
        "become unreliable when its data collection changes in later years."
    )

    if TECHNICAL:
        with st.expander("Technical EDA notes"):
            st.markdown(
                """
                - Initial analytic dataframe: **494,862 × 38**
                - 349,609 patients had one registration.
                - 52,358 patients had two registrations.
                - Maximum observed registrations for one patient: 11.
                - 59 records had an end date earlier than the initial listing date.
                - Recent cohorts are heavily censored because they have had less follow-up.
                """
            )

    stage_readme(EDA)

    st.subheader("Notebook")
    notebook_panel(
        EDA,
        "01_EDA.ipynb",
        "eda",
    )

    if TECHNICAL:
        with st.expander("Public outputs"):
            output_explorer(EDA / "outputs", "eda")


# ============================================================
# 02 CLEANING
# ============================================================

elif PAGE == "02 · Data Cleaning":

    section(
        "02 · Data Cleaning",
        "Preparing reliable variables while keeping the original patient-level data private.",
    )

    cols = st.columns(4)

    metric(
        cols[0],
        "Starting registrations",
        "494,862",
        "Same analytic cohort as EDA",
    )

    metric(
        cols[1],
        "Survival-valid",
        "494,803",
        "Valid time-to-event records",
    )

    metric(
        cols[2],
        "Invalid timelines",
        "59",
        "End date earlier than listing date",
    )

    metric(
        cols[3],
        "Modeling-base columns",
        "25",
        "Selected cleaned fields",
    )

    simple(
        "Cleaning did not mean deleting everything unusual. Important issues were "
        "flagged, corrected when appropriate, or converted to missing values so later "
        "models would not silently treat bad values as real measurements."
    )

    st.subheader("Main cleaning decisions")

    cleaning = pd.DataFrame(
        {
            "Area": [
                "Dates",
                "Survival time",
                "BMI",
                "Functional status",
                "CPRA",
                "A2/A2B status",
                "Missingness",
            ],
            "What was done": [
                "Converted listing and end dates to usable date values.",
                "Flagged negative or invalid survival durations.",
                "Values ≤10 or >100 were treated as missing.",
                "Invalid code 1 was treated as missing; special codes were preserved.",
                "Restricted valid values to 0–100 and created a missingness indicator.",
                "Converted compatibility information into a structured feature.",
                "Created indicators where missingness itself could be informative.",
            ],
        }
    )

    st.dataframe(cleaning, hide_index=True)

    st.subheader("Cleaning log")

    log_path = CLEAN / "outputs" / "medimate_cleaning_log.csv"

    if log_path.exists():
        show_csv(log_path)
    else:
        st.info("Cleaning log is not available.")

    caution(
        "The cleaned patient-level datasets are intentionally not published in GitHub."
    )

    stage_readme(CLEAN)

    st.subheader("Notebook")
    notebook_panel(
        CLEAN,
        "02_Data_Cleaning.ipynb",
        "clean",
    )

    if TECHNICAL:
        with st.expander("All public cleaning outputs"):
            output_explorer(CLEAN / "outputs", "clean")


# ============================================================
# 03 MODEL PREPARATION
# ============================================================

elif PAGE == "03 · Model Preparation":

    section(
        "03 · Model Preparation",
        "Creating a fair evaluation setup before training the classification models.",
    )

    cols = st.columns(4)

    metric(
        cols[0],
        "Classification cohort",
        "450,764",
        "Registrations with usable 1-year labels",
    )

    metric(
        cols[1],
        "1-year adverse events",
        "17,587",
        "3.90% positive outcome",
    )

    metric(
        cols[2],
        "Known negatives",
        "433,177",
        "96.10% of classification cohort",
    )

    metric(
        cols[3],
        "Excluded",
        "44,098",
        "Insufficient or unknown 1-year outcome",
    )

    simple(
        "A patient who was still waiting after only a few months could not fairly be called "
        "a one-year negative. Those registrations were excluded until enough follow-up was available."
    )

    st.subheader("Temporal and patient-safe split")

    cols = st.columns(3)

    metric(
        cols[0],
        "Training",
        "2015–2022",
        "314,982 registrations",
        "Temporal split",
    )

    metric(
        cols[1],
        "Validation",
        "2023",
        "39,402 registrations",
        "Temporal split",
    )

    metric(
        cols[2],
        "Test",
        "2024",
        "41,471 registrations",
        "Temporal split",
    )

    st.success(
        "Patient overlap between train, validation, and test: 0"
    )

    st.markdown(
        """
        The final primary classification feature set contained **12 original features**
        before preprocessing:

        - Age
        - Gender
        - Ethnicity code
        - Blood type
        - Dialysis status
        - BMI
        - BMI missingness
        - Functional status
        - Functional-status missingness
        - Initial status
        - Region
        - A2/A2B status
        """
    )

    st.subheader("Feature files")

    tabs = st.tabs(["Without CPRA", "With CPRA"])

    with tabs[0]:
        p = PREP / "outputs" / "features_no_cpra.csv"
        if p.exists():
            show_csv(p)

    with tabs[1]:
        p = PREP / "outputs" / "features_with_cpra.csv"
        if p.exists():
            show_csv(p)

    st.subheader("Why CPRA was excluded")

    st.markdown(
        """
        CPRA gave almost no validation improvement while recent years had very high
        missingness. The final models therefore used the more stable feature set
        without CPRA.
        """
    )

    stage_readme(PREP)

    st.subheader("Notebook")
    notebook_panel(
        PREP,
        "03_Model_Preparation.ipynb",
        "prep",
    )

    if TECHNICAL:
        with st.expander("All public model-preparation outputs"):
            output_explorer(PREP / "outputs", "prep")


# ============================================================
# 04 CLASSIFICATION
# ============================================================

elif PAGE == "04 · Classification Modeling":

    section(
        "04 · Classification Modeling",
        "Predicting whether an adverse waitlist outcome occurs within one year.",
    )

    st.markdown(
        """
        Several classification approaches were compared. The final primary model was
        **XGBoost without CPRA**.
        """
    )

    cols = st.columns(5)

    metric(
        cols[0],
        "ROC-AUC",
        "0.6742",
        "2024 untouched test set",
        "ROC-AUC",
    )

    metric(
        cols[1],
        "PR-AUC",
        "0.0860",
        "Rare-event metric",
        "PR-AUC",
    )

    metric(
        cols[2],
        "Precision",
        "7.18%",
        "At validation-selected threshold",
        "Precision",
    )

    metric(
        cols[3],
        "Recall",
        "48.22%",
        "At validation-selected threshold",
        "Recall",
    )

    metric(
        cols[4],
        "Top-decile lift",
        "2.53×",
        "Highest predicted-risk 10%",
        "Top-decile lift",
    )

    simple(
        "The strongest use of the classifier is ranking risk. The highest predicted-risk "
        "10% contained roughly 2.5 times the adverse-event rate expected from the population overall."
    )

    caution(
        "<b>Why is precision low?</b> The one-year adverse event is uncommon. "
        "When an outcome is rare, even a useful ranking model can have many false-positive flags."
    )

    st.subheader("Validation vs test performance")

    c1, c2 = st.columns(2)

    with c1:
        show_image(
            FINAL_OUT / "classification_validation_vs_test.png",
            "Performance stability from validation to the final 2024 test set.",
        )

    with c2:
        show_image(
            FINAL_OUT / "classification_top_features.png",
            "Most important classification features.",
        )

    if TECHNICAL:
        st.subheader("Model comparison")

        tabs = st.tabs(
            [
                "Models",
                "Validation vs test",
                "CPRA experiment",
                "Settings",
            ]
        )

        with tabs[0]:
            show_csv(
                CLASS_OUT / "multi_model_validation_comparison.csv"
            )

        with tabs[1]:
            show_csv(
                CLASS_OUT / "validation_vs_test_summary.csv"
            )

        with tabs[2]:
            show_csv(
                CLASS_OUT / "cpra_sensitivity_comparison.csv"
            )

        with tabs[3]:
            show_json(
                CLASS_OUT / "final_model_settings.json"
            )

        st.subheader("Confusion matrix at the fixed threshold")

        cm = pd.DataFrame(
            [
                ["Actual negative", 29867, 9999],
                ["Actual adverse", 831, 774],
            ],
            columns=[
                "Actual class",
                "Predicted negative",
                "Predicted adverse",
            ],
        )

        st.dataframe(cm, hide_index=True)

        st.subheader("Explainability results")

        show_csv(
            CLASS_OUT / "shap_original_feature_importance.csv"
        )

    stage_readme(CLASS)

    st.subheader("Notebook")
    notebook_panel(
        CLASS,
        "04_Classification_Modeling.ipynb",
        "classification",
    )

    if TECHNICAL:
        with st.expander("Explore every public classification output"):
            output_explorer(
                CLASS_OUT,
                "classification",
            )


# ============================================================
# 05 SURVIVAL
# ============================================================

elif PAGE == "05 · Survival Modeling":

    section(
        "05 · Survival Modeling",
        "Modeling how adverse-outcome risk changes over time while accounting for transplantation.",
    )

    st.markdown(
        """
        Survival analysis adds time to the problem.

        A major issue is that receiving a transplant prevents a patient from later
        experiencing death or removal-too-sick while still waiting. Transplant was
        therefore treated as a **competing event**.
        """
    )

    cols = st.columns(4)

    metric(
        cols[0],
        "XGBoost Survival C-index",
        "0.6784",
        "2024 test set",
        "C-index",
    )

    metric(
        cols[1],
        "Low-risk 1-year incidence",
        "1.70%",
        "2024 risk group",
        "Cumulative incidence",
    )

    metric(
        cols[2],
        "High-risk 1-year incidence",
        "6.60%",
        "2024 risk group",
        "Cumulative incidence",
    )

    metric(
        cols[3],
        "High / low risk ratio",
        "3.87×",
        "One-year adverse incidence",
        "Cumulative incidence",
    )

    simple(
        "The survival model separated the test population into groups with clearly "
        "different one-year adverse-event incidence."
    )

    st.subheader("Competing events")

    c1, c2 = st.columns(2)

    with c1:
        show_image(
            FINAL_OUT / "overall_competing_risk.png",
            "Overall adverse and transplant cumulative incidence.",
        )

    with c2:
        show_image(
            FINAL_OUT / "survival_1yr_risk_groups.png",
            "One-year adverse incidence for low-, medium-, and high-risk groups.",
        )

    st.subheader("Overall competing-risk estimates")

    risk_summary = pd.DataFrame(
        {
            "Time": ["1 year", "2 years", "3 years", "5 years"],
            "Adverse incidence": [
                "3.88%",
                "8.31%",
                "12.68%",
                "20.00%",
            ],
            "Transplant incidence": [
                "30.07%",
                "40.74%",
                "48.15%",
                "58.95%",
            ],
        }
    )

    st.dataframe(risk_summary, hide_index=True)

    st.subheader("Risk by patient characteristics")

    c1, c2 = st.columns(2)

    with c1:
        show_image(
            FINAL_OUT / "survival_age_risk.png",
            "Adverse-event incidence across age groups.",
        )

    with c2:
        show_image(
            FINAL_OUT / "functional_status_1yr_risk.png",
            "One-year adverse incidence by functional-status group.",
        )

    if TECHNICAL:
        st.subheader("Survival models")

        tabs = st.tabs(
            [
                "Model comparison",
                "Cox hazard ratios",
                "Final settings",
                "Risk groups",
            ]
        )

        with tabs[0]:
            show_csv(
                SURV_OUT / "survival_model_comparison.csv"
            )

        with tabs[1]:
            show_csv(
                SURV_OUT / "final_cox_hazard_ratios.csv"
            )

        with tabs[2]:
            show_json(
                SURV_OUT / "final_survival_settings.json"
            )

        with tabs[3]:
            show_csv(
                SURV_OUT / "final_1yr_risk_groups.csv"
            )

        st.markdown("### Selected interpretable Cox results")

        cox = pd.DataFrame(
            {
                "Factor": [
                    "Age +10 years",
                    "Dialysis",
                    "Moderate functional status",
                    "Poor functional status",
                ],
                "Hazard ratio": [
                    1.305,
                    1.150,
                    1.329,
                    2.355,
                ],
                "Plain meaning": [
                    "About 30.5% higher modeled hazard per 10 years of age.",
                    "About 15% higher modeled hazard.",
                    "About 32.9% higher than good functional status.",
                    "About 2.36 times the modeled hazard of good functional status.",
                ],
            }
        )

        st.dataframe(cox, hide_index=True)

        caution(
            "Hazard ratios describe associations in this model. They do not establish causation."
        )

    stage_readme(SURV)

    st.subheader("Notebook")
    notebook_panel(
        SURV,
        "05_Survival_Modeling.ipynb",
        "survival",
    )

    if TECHNICAL:
        with st.expander("Explore every public survival output"):
            output_explorer(
                SURV_OUT,
                "survival",
            )


# ============================================================
# 06 FINAL ANALYSIS
# ============================================================

elif PAGE == "06 · Final Analysis":

    section(
        "06 · Final Combined Analysis",
        "Bringing classification and survival results together into one project-level interpretation.",
    )

    cols = st.columns(4)

    metric(
        cols[0],
        "Classification ROC-AUC",
        "0.6742",
        "XGBoost",
        "ROC-AUC",
    )

    metric(
        cols[1],
        "Top-decile lift",
        "2.53×",
        "Classification",
        "Top-decile lift",
    )

    metric(
        cols[2],
        "Survival C-index",
        "0.6784",
        "XGBoost Survival",
        "C-index",
    )

    metric(
        cols[3],
        "High / low 1-year risk",
        "3.87×",
        "Survival risk groups",
        "Cumulative incidence",
    )

    st.subheader("Shared feature signals")

    show_image(
        FINAL_OUT / "combined_feature_importance.png",
        "Features that were important across both modeling approaches.",
    )

    simple(
        "Age and functional status were the most consistent shared signals. "
        "Region, ethnicity code, and dialysis status were also important across the final analyses."
    )

    st.subheader("Final figures")

    figures = [
        (
            "Classification validation vs test",
            "classification_validation_vs_test.png",
        ),
        (
            "Classification top features",
            "classification_top_features.png",
        ),
        (
            "Overall competing risk",
            "overall_competing_risk.png",
        ),
        (
            "Survival risk groups",
            "survival_1yr_risk_groups.png",
        ),
        (
            "Survival age risk",
            "survival_age_risk.png",
        ),
        (
            "Functional status risk",
            "functional_status_1yr_risk.png",
        ),
        (
            "Survival top features",
            "survival_top_features.png",
        ),
    ]

    for i in range(0, len(figures), 2):
        cols = st.columns(2)
        for col, (title, filename) in zip(cols, figures[i:i+2]):
            with col:
                show_image(
                    FINAL_OUT / filename,
                    title,
                )

    if TECHNICAL:
        st.subheader("Final project tables")

        tabs = st.tabs(
            [
                "Executive summary",
                "Key findings",
                "Subgroups",
                "Primary models",
                "Age synthesis",
                "Metrics",
            ]
        )

        with tabs[0]:
            show_csv(
                FINAL_OUT / "executive_summary.csv"
            )

        with tabs[1]:
            show_csv(
                FINAL_OUT / "final_key_findings.csv"
            )

        with tabs[2]:
            show_csv(
                FINAL_OUT / "final_subgroup_findings.csv"
            )

        with tabs[3]:
            show_csv(
                FINAL_OUT / "primary_model_summary.csv"
            )

        with tabs[4]:
            show_csv(
                FINAL_OUT / "age_subgroup_synthesis.csv"
            )

        with tabs[5]:
            show_json(
                FINAL_OUT / "final_project_metrics.json"
            )

        summary_path = FINAL_OUT / "final_project_summary.txt"
        if summary_path.exists():
            with st.expander("Final written project summary"):
                st.text(read_text(summary_path))

    research_warning()

    stage_readme(FINAL)

    st.subheader("Notebook")
    notebook_panel(
        FINAL,
        "06_Final_Analysis.ipynb",
        "final",
    )

    if TECHNICAL:
        with st.expander("Explore every public final-analysis output"):
            output_explorer(
                FINAL_OUT,
                "final",
            )


# ============================================================
# FAIRNESS
# ============================================================

elif PAGE == "Fairness & Explainability":

    section(
        "Fairness & Explainability",
        "Understanding model behavior and subgroup performance without treating model explanations as causal effects.",
    )

    caution(
        "<b>Important:</b> A difference in performance between groups deserves investigation, "
        "but it is not by itself proof that a model is biased."
    )

    st.subheader("Model explainability")

    c1, c2 = st.columns(2)

    with c1:
        show_image(
            FINAL_OUT / "classification_top_features.png",
            "Classification feature importance.",
        )

    with c2:
        show_image(
            FINAL_OUT / "survival_top_features.png",
            "Survival feature importance.",
        )

    st.markdown(
        """
        **Feature importance and SHAP values describe the model.**

        They do not tell us that a feature causes the patient's outcome.
        """
    )

    st.subheader("Classification subgroup evaluation")

    group = st.selectbox(
        "Choose a subgroup",
        [
            "Age",
            "Gender",
            "Blood type",
            "Ethnicity code",
            "Region",
        ],
    )

    subgroup_files = {
        "Age": "fairness_age.csv",
        "Gender": "fairness_gender.csv",
        "Blood type": "fairness_blood_type.csv",
        "Ethnicity code": "fairness_ethcat.csv",
        "Region": "fairness_region.csv",
    }

    show_csv(
        CLASS_OUT / subgroup_files[group]
    )

    if group == "Age":
        st.markdown(
            """
            Recall varied substantially by age at the fixed global threshold.
            This is one reason subgroup evaluation matters even when a model's
            overall metrics appear stable.
            """
        )

    if group == "Ethnicity code":
        st.info(
            "The analysis keeps the dataset's numeric ethnicity codes. "
            "They are not assigned names without an official code mapping."
        )

    if TECHNICAL:
        st.subheader("SHAP and contribution tables")

        tabs = st.tabs(
            [
                "Classification importance",
                "Age direction",
                "Functional status",
                "Dialysis",
                "Region",
                "Survival explainability",
            ]
        )

        with tabs[0]:
            show_csv(
                CLASS_OUT / "shap_original_feature_importance.csv"
            )

        with tabs[1]:
            show_csv(
                CLASS_OUT / "shap_age_direction.csv"
            )

        with tabs[2]:
            show_csv(
                CLASS_OUT / "shap_functional_status_direction.csv"
            )

        with tabs[3]:
            show_csv(
                CLASS_OUT / "shap_dialysis_direction.csv"
            )

        with tabs[4]:
            show_csv(
                CLASS_OUT / "shap_region_direction.csv"
            )

        with tabs[5]:
            show_csv(
                SURV_OUT / "xgboost_survival_explainability.csv"
            )


# ============================================================
# REPRODUCIBILITY
# ============================================================

elif PAGE == "Reproducibility & Files":

    section(
        "Reproducibility & Project Files",
        "How the public analysis is organized and what can be inspected or downloaded.",
    )

    st.subheader("Pipeline notebooks")

    notebook_specs = [
        (
            EDA,
            "01_EDA.ipynb",
            "Exploratory Data Analysis",
            "r_eda",
        ),
        (
            CLEAN,
            "02_Data_Cleaning.ipynb",
            "Data Cleaning",
            "r_clean",
        ),
        (
            PREP,
            "03_Model_Preparation.ipynb",
            "Model Preparation",
            "r_prep",
        ),
        (
            CLASS,
            "04_Classification_Modeling.ipynb",
            "Classification",
            "r_class",
        ),
        (
            SURV,
            "05_Survival_Modeling.ipynb",
            "Survival",
            "r_surv",
        ),
        (
            FINAL,
            "06_Final_Analysis.ipynb",
            "Final Analysis",
            "r_final",
        ),
    ]

    for stage, filename, title, key in notebook_specs:
        with st.expander(title):
            notebook_panel(
                stage,
                filename,
                key,
            )

    st.subheader("Public output inventory")

    inventory_rows = []

    stage_outputs = {
        "01 EDA": EDA / "outputs",
        "02 Data cleaning": CLEAN / "outputs",
        "03 Model preparation": PREP / "outputs",
        "04 Classification": CLASS_OUT,
        "05 Survival": SURV_OUT,
        "06 Final analysis": FINAL_OUT,
    }

    for stage_name, folder in stage_outputs.items():
        for path in output_files(folder):
            inventory_rows.append(
                {
                    "Stage": stage_name,
                    "File": path.name,
                    "Type": path.suffix.lower().replace(".", "").upper(),
                    "Size (KB)": round(path.stat().st_size / 1024, 1),
                }
            )

    if inventory_rows:
        inventory = pd.DataFrame(inventory_rows)
        st.dataframe(inventory, hide_index=True)

    st.subheader("Explore outputs by stage")

    stage_name = st.selectbox(
        "Pipeline stage",
        list(stage_outputs.keys()),
    )

    output_explorer(
        stage_outputs[stage_name],
        "repro_" + stage_name.replace(" ", "_"),
    )

    st.subheader("Files intentionally excluded")

    excluded = pd.DataFrame(
        {
            "File type": [
                "Raw OPTN data",
                "Cleaned patient-level datasets",
                "Patient-level predictions",
                "Train/validation/test matrices",
                "Preprocessor objects",
                "Trained model binaries",
            ],
            "Reason": [
                "Protected under the data-use agreement.",
                "Contains row-level waitlist information.",
                "Contains patient/registration-level model outputs.",
                "Intermediate modeling artifact.",
                "Not needed to understand the public analysis.",
                "Not needed for the public results website.",
            ],
        }
    )

    st.dataframe(excluded, hide_index=True)

    simple(
        "The public repository contains enough information to understand the analysis, "
        "methods, aggregate results, and modeling decisions without publishing protected row-level data."
    )

    st.link_button(
        "Open the GitHub branch",
        GITHUB,
    )


# ============================================================
# GLOSSARY
# ============================================================

elif PAGE == "Glossary":

    section(
        "Technical Terms in Plain English",
        "Short explanations for the main terms used throughout the website.",
    )

    for term, definition in GLOSSARY.items():
        with st.expander(term):
            st.write(definition)

    st.markdown(
        """
        If a term appears next to a metric with a **?** icon, hover over the icon
        to see the same short explanation without leaving the page.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">
MediMate Kidney Waitlist Analysis · Research and educational use only ·
Raw OPTN patient-level data is not included in this website.
</div>
""",
    unsafe_allow_html=True,
)
