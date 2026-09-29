import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════

st.set_page_config(
    page_title="JoSAA College Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ════════════════════════════════════════════════════
# CUSTOM CSS
# ════════════════════════════════════════════════════

st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background: #0f172a;
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background: #1e293b !important;
    border-right: 1px solid #334155;
}

[data-testid="stHeader"] {
    background: #1e293b;
    border-bottom: 1px solid #334155;
}

.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    width: 100% !important;
    font-size: 15px !important;
    box-shadow: 0 4px 16px rgba(59,130,246,.3) !important;
}

.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

[data-testid="stMetric"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px !important;
}

</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# LOAD MODEL
# ════════════════════════════════════════════════════

@st.cache_resource
def load_model():

    model = joblib.load(
        "model/final_random_forest.pkl"
    )

    encoders = joblib.load(
        "model/encoders.pkl"
    )

    features = joblib.load(
        "model/features.pkl"
    )

    return model, encoders, features


# ════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════

@st.cache_data
def load_data():

    return pd.read_pickle(
        "model/processed_data.pkl"
    )


# ════════════════════════════════════════════════════
# LOAD PROJECT FILES
# ════════════════════════════════════════════════════

try:

    model, encoders, features = load_model()
    df = load_data()

except Exception as e:

    st.error(
        "❌ Project files could not be loaded."
    )

    st.code(str(e))

    st.stop()


# ════════════════════════════════════════════════════
# COLUMN NORMALIZATION
# ════════════════════════════════════════════════════

if (
    "Academic_Program_Name" not in df.columns
    and
    "Academic Program Name" in df.columns
):

    df = df.rename(
        columns={
            "Academic Program Name":
            "Academic_Program_Name"
        }
    )


if (
    "Opening_Rank" not in df.columns
    and
    "Opening Rank" in df.columns
):

    df = df.rename(
        columns={
            "Opening Rank":
            "Opening_Rank"
        }
    )


if (
    "Closing_Rank" not in df.columns
    and
    "Closing Rank" in df.columns
):

    df = df.rename(
        columns={
            "Closing Rank":
            "Closing_Rank"
        }
    )


# ════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════

if "page" not in st.session_state:

    st.session_state.page = "landing"


# ════════════════════════════════════════════════════
# PAGE 1 — HOME
# ════════════════════════════════════════════════════

def landing_page():

    st.markdown(
        '<div style="text-align: center;">'
        '<span style="background:rgba(59,130,246,.15); '
        'border:1px solid rgba(59,130,246,.3); '
        'border-radius:999px; padding:6px 18px; '
        'font-size:13px; color:#60a5fa; '
        'font-weight:500;">'
        'JoSAA College Prediction'
        '</span>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h1 style='text-align: center;
        font-size: 52px;
        font-weight: 700;
        color: #ffffff;'>
        Find your college<br>
        <span style='color: #60a5fa;'>
        before counselling begins
        </span>
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p style='text-align: center;
        font-size: 18px;
        color: #94a3b8;
        max-width: 600px;
        margin: 0 auto 30px;'>
        Enter your JEE rank, category and quota —
        explore IITs, NITs, IIITs and GFTIs using
        historical JoSAA closing-rank data.
        </p>
        """,
        unsafe_allow_html=True
    )


    # ====================================================
    # DATASET METRICS
    # ====================================================

    institutes = (
        df["Institute"].nunique()
        if "Institute" in df.columns
        else 0
    )

    branches = (
        df["Academic_Program_Name"].nunique()
        if "Academic_Program_Name" in df.columns
        else 0
    )

    years = (
        df["Year"].nunique()
        if "Year" in df.columns
        else 0
    )

    rounds = (
        df["Round"].nunique()
        if "Round" in df.columns
        else 0
    )


    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Institutes",
            f"{institutes:,}"
        )

    with c2:
        st.metric(
            "Branches",
            f"{branches:,}"
        )

    with c3:
        st.metric(
            "Years",
            years
        )

    with c4:
        st.metric(
            "Rounds",
            rounds
        )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    # ====================================================
    # FEATURES
    # ====================================================

    f1, f2, f3, f4 = st.columns(4)

    with f1:

        with st.container(border=True):

            st.markdown("### 🎯")

            st.markdown(
                "**Rank-based Prediction**"
            )

            st.caption(
                "Find colleges based on your actual "
                "JEE rank and preferences."
            )


    with f2:

        with st.container(border=True):

            st.markdown("### 📈")

            st.markdown(
                "**Cutoff Analysis**"
            )

            st.caption(
                "Compare your rank with historical "
                "JoSAA closing ranks."
            )


    with f3:

        with st.container(border=True):

            st.markdown("### 🟢")

            st.markdown(
                "**Chance Indicator**"
            )

            st.caption(
                "HIGH and MEDIUM options based "
                "on your rank safety."
            )


    with f4:

        with st.container(border=True):

            st.markdown("### 📊")

            st.markdown(
                "**Dataset Insights**"
            )

            st.caption(
                "Explore quota, category, gender, "
                "rounds and institute data."
            )


    st.markdown(
        "<br><br>",
        unsafe_allow_html=True
    )


    # ====================================================
    # NAVIGATION BUTTONS
    # ====================================================

    col1, col2, col3, col4 = st.columns(
        [1, 2, 2, 1]
    )

    with col2:

        if st.button(
            "🔍  Start Predicting  →",
            key="home_predict"
        ):

            st.session_state.page = "predictor"

            st.rerun()


    with col3:

        if st.button(
            "📊  Explore Insights  →",
            key="home_insights"
        ):

            st.session_state.page = "insights"

            st.rerun()


    st.markdown(
        "<br><br>",
        unsafe_allow_html=True
    )

    st.divider()


    # ====================================================
    # FOOTER
    # ====================================================

    ft1, ft2 = st.columns(2)

    with ft1:

        st.markdown(
            "Built by **Shreshtha Pandey** · "
            "[⭐ GitHub Repo]"
            "(https://github.com/"
            "ShreshthaPandey/"
            "JOSAA-collage-predictor)"
        )


    with ft2:

        st.markdown(
            "<div style='text-align: right;'>"
            "JoSAA Historical Data · Streamlit"
            "</div>",
            unsafe_allow_html=True
        )


# ════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════

def clean_text(value):

    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def normalize_pwd(value):

    if pd.isna(value):
        return 0

    value = clean_text(value)

    if value in [
        "1",
        "1.0",
        "yes",
        "true",
        "pwd"
    ]:

        return 1

    return 0


def gender_matches(
    series,
    selected_gender
):

    values = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    selected_gender = clean_text(
        selected_gender
    )


    if selected_gender == "female":

        return (
            values.str.contains(
                "female",
                na=False
            )
            |
            values.str.contains(
                "women",
                na=False
            )
        )


    if selected_gender == "gender-neutral":

        return (
            values.str.contains(
                "gender-neutral",
                na=False
            )
            |
            values.str.contains(
                "gender neutral",
                na=False
            )
            |
            values.str.contains(
                "neutral",
                na=False
            )
        )


    return values == selected_gender


def quota_matches(
    series,
    selected_quota
):

    values = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    selected_quota = clean_text(
        selected_quota
    )


    if selected_quota == "ai":
        return values == "ai"


    if selected_quota == "hs":
        return values == "hs"


    if selected_quota == "os":
        return values == "os"


    if selected_quota == "other":

        return ~values.isin(
            ["ai", "hs", "os"]
        )


    return values == selected_quota


# ════════════════════════════════════════════════════
# CHANCE LOGIC
# ════════════════════════════════════════════════════

def get_chances(margin):

    if margin >= 1000:

        return "🟢 HIGH", "high"

    elif margin >= 300:

        return "🟢 HIGH", "high"

    elif margin >= 0:

        return "🟡 MEDIUM", "medium"

    elif margin >= -500:

        return "🔴 LOW", "low"

    else:

        return None, None


# ════════════════════════════════════════════════════
# PREDICTION FUNCTION
# ════════════════════════════════════════════════════

def predict_colleges(
    rank,
    category,
    quota,
    gender,
    is_pwd,
    exam
):

    # ====================================================
    # FILTER BY EXAM
    # ====================================================

    if exam == "JEE Advanced":

        data = df[
            df["Institute_Type"]
            .astype(str)
            .str.upper()
            == "IIT"
        ].copy()

    else:

        data = df[
            df["Institute_Type"]
            .astype(str)
            .str.upper()
            != "IIT"
        ].copy()


    # ====================================================
    # LATEST YEAR
    # ====================================================

    if data.empty:
        return pd.DataFrame()


    latest_year = data["Year"].max()

    latest_year_data = data[
        data["Year"] == latest_year
    ].copy()


    if latest_year_data.empty:
        return pd.DataFrame()


    # ====================================================
    # LAST ROUND
    # ====================================================

    last_round = latest_year_data[
        "Round"
    ].max()

    data = latest_year_data[
        latest_year_data["Round"]
        == last_round
    ].copy()


    # ====================================================
    # USER FILTERS
    # ====================================================

    data = data[
        (
            data["Category"]
            .astype(str)
            .str.strip()
            ==
            str(category).strip()
        )
        &
        (
            data["Quota"]
            .astype(str)
            .str.strip()
            ==
            str(quota).strip()
        )
        &
        (
            data["Gender"]
            .astype(str)
            .str.strip()
            ==
            str(gender).strip()
        )
        &
        (
            data["is_pwd"]
            == is_pwd
        )
    ].copy()


    # ====================================================
    # NO DATA
    # ====================================================

    if data.empty:
        return pd.DataFrame()


    # ====================================================
    # PREDICTED CLOSING
    # ====================================================

    if "hist_avg_closing" in data.columns:

        data["Predicted Closing"] = (
            pd.to_numeric(
                data["hist_avg_closing"],
                errors="coerce"
            )
        )

        data["Predicted Closing"] = (
            data["Predicted Closing"]
            .fillna(
                pd.to_numeric(
                    data["Closing_Rank"],
                    errors="coerce"
                )
            )
        )

    else:

        data["Predicted Closing"] = (
            pd.to_numeric(
                data["Closing_Rank"],
                errors="coerce"
            )
        )


    # ====================================================
    # REMOVE INVALID CLOSING RANKS
    # ====================================================

    data = data[
        data["Predicted Closing"].notna()
        &
        (
            data["Predicted Closing"]
            > 0
        )
    ].copy()


    if data.empty:
        return pd.DataFrame()


    # ====================================================
    # MARGIN
    # ====================================================

    data["Margin"] = (
        data["Predicted Closing"]
        - rank
    )


    # ====================================================
    # CHANCES
    # ====================================================

    data["Chances"] = (
        data["Margin"]
        .apply(
            lambda margin:
            get_chances(margin)[0]
        )
    )


    # ====================================================
    # REMOVE NO-CHANCE OPTIONS
    # ====================================================

    data = data[
        data["Chances"].notna()
    ].copy()


    if data.empty:
        return pd.DataFrame()


    # ====================================================
    # REMOVE DUPLICATES
    # ====================================================

    data = data.drop_duplicates(
        subset=[
            "Institute",
            "Academic_Program_Name"
        ],
        keep="first"
    )


    # ====================================================
    # SORT SAFEST FIRST
    # ====================================================

    data = data.sort_values(
        by="Margin",
        ascending=False
    )


    # ====================================================
    # FINAL RESULT
    # ====================================================

    result = data[
        [
            "Institute",
            "Academic_Program_Name",
            "Institute_Type",
            "Opening_Rank",
            "Predicted Closing",
            "Chances"
        ]
    ].copy()


    result.columns = [
        "College",
        "Branch",
        "Type",
        "Opening Rank",
        "Predicted Closing",
        "Chances"
    ]


    return result.reset_index(
        drop=True
    )


# ════════════════════════════════════════════════════
# NO COLLEGE MESSAGE
# ════════════════════════════════════════════════════

def show_no_college_message(
    rank,
    exam,
    category,
    quota,
    gender
):

    with st.container(
        border=True
    ):

        st.markdown(
            "### 🎓 No College Found"
        )

        st.markdown(
            "We couldn't find a college whose "
            "historical closing rank covers "
            "your current rank."
        )

        st.markdown(
            f"**Rank:** {rank:,} · "
            f"**{exam}** · "
            f"**{category}** · "
            f"**Quota:** {quota} · "
            f"**Gender:** {gender}"
        )

        st.info(
            "Try exploring a different quota, "
            "category, gender preference or "
            "exam to see more options."
        )


# ════════════════════════════════════════════════════
# PAGE 2 — PREDICTOR
# ════════════════════════════════════════════════════

def predictor_page():

    col_logo, col_home = st.columns(
        [5, 1]
    )


    with col_logo:

        st.markdown(
            "### 🎓 JoSAA Predictor"
        )


    with col_home:

        if st.button(
            "← Home",
            key="predictor_home"
        ):

            st.session_state.page = "landing"

            st.rerun()


    st.divider()


    sidebar_col, results_col = st.columns(
        [1, 3]
    )


    # ====================================================
    # LEFT SIDE — INPUTS
    # ====================================================

    with sidebar_col:

        st.subheader(
            "Your Details"
        )


        # ------------------------------------------------
        # EXAM
        # ------------------------------------------------

        exam = st.radio(
            "Exam",
            options=[
                "JEE Main",
                "JEE Advanced"
            ],
            horizontal=True
        )


        # ------------------------------------------------
        # DISCLAIMER
        # ------------------------------------------------

        st.warning(
            "⚠️ DISCLAIMER: If you belong "
            "to a reserved category "
            "(SC / ST / OBC-NCL / EWS), "
            "please enter your Category Rank "
            "from your JEE Main scorecard. "
            "Do NOT enter your Common Rank List "
            "(CRL) or All India Rank, otherwise "
            'the system may show "No colleges found."'
        )


        # ------------------------------------------------
        # RANK
        # ------------------------------------------------

        rank_input = st.text_input(
            "Your Rank",
            placeholder="Example: 12500",
            help="Enter your JEE rank"
        )


        # ------------------------------------------------
        # CATEGORY
        # ------------------------------------------------

        category = st.selectbox(
            "Category",
            [
                "OPEN",
                "OBC-NCL",
                "SC",
                "ST",
                "EWS"
            ]
        )


        # ------------------------------------------------
        # GENDER
        # ------------------------------------------------

        gender = st.selectbox(
            "Gender",
            [
                "Gender-Neutral",
                "Female"
            ]
        )


        # ------------------------------------------------
        # QUOTA
        # ------------------------------------------------

        quota = st.selectbox(
            "Quota",
            [
                "AI",
                "HS",
                "OS",
                "Other"
            ]
        )


        # ------------------------------------------------
        # PWD
        # ------------------------------------------------

        pwd_option = st.selectbox(
            "PwD",
            [
                "No",
                "Yes"
            ]
        )


        is_pwd = (
            1
            if pwd_option == "Yes"
            else 0
        )


        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )


        # ------------------------------------------------
        # PREDICT BUTTON
        # ------------------------------------------------

        predict_clicked = st.button(
            "🔍 Predict Colleges",
            key="predict_button"
        )


    # ====================================================
    # RIGHT SIDE — RESULTS
    # ====================================================

    with results_col:

        if not predict_clicked:

            with st.container(
                border=True
            ):

                st.markdown(
                    "<h3 style='text-align: center;'>"
                    "🎓 Ready to predict"
                    "</h3>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    "<p style='text-align: center; "
                    "color: #94a3b8;'>"
                    "Enter your rank and preferences, "
                    "then click Predict Colleges."
                    "</p>",
                    unsafe_allow_html=True
                )


        else:

            # =================================================
            # VALIDATE RANK
            # =================================================

            try:

                rank = int(
                    rank_input
                    .replace(",", "")
                    .strip()
                )

            except ValueError:

                st.warning(
                    "⚠️ Please enter a valid "
                    "numeric JEE rank."
                )

                return


            if rank <= 0:

                st.warning(
                    "⚠️ Rank must be greater than 0."
                )

                return


            # =================================================
            # PREDICTION
            # =================================================

            with st.spinner(
                "Finding suitable colleges..."
            ):

                results = predict_colleges(
                    rank,
                    category,
                    quota,
                    gender,
                    is_pwd,
                    exam
                )


            # =================================================
            # NO RESULTS
            # =================================================

            if results.empty:

                show_no_college_message(
                    rank,
                    exam,
                    category,
                    quota,
                    gender
                )

                return


            # =================================================
            # CHANCE COUNTS
            # =================================================

            high = len(
                results[
                    results["Chances"]
                    == "🟢 HIGH"
                ]
            )


            medium = len(
                results[
                    results["Chances"]
                    == "🟡 MEDIUM"
                ]
            )


            low = len(
                results[
                    results["Chances"]
                    == "🔴 LOW"
                ]
            )


            # =================================================
            # HEADER
            # =================================================

            st.markdown(
                "### Predicted Colleges"
            )

            st.caption(
                f"Rank {rank:,} · "
                f"{exam} · "
                f"{len(results)} colleges found"
            )


            # =================================================
            # METRICS
            # =================================================

            m1, m2, m3 = st.columns(3)


            with m1:

                st.metric(
                    "🎓 Colleges",
                    len(results)
                )


            with m2:

                st.metric(
                    "🟢 High Chance",
                    high
                )


            with m3:

                st.metric(
                    "🟡 Medium Chance",
                    medium
                )


            st.markdown(
                "<br>",
                unsafe_allow_html=True
            )


            # =================================================
            # TABS
            # =================================================

            tab_all, tab_high, tab_medium = st.tabs(
                [
                    f"All ({len(results)})",
                    f"🟢 High ({high})",
                    f"🟡 Medium ({medium})"
                ]
            )


            # =================================================
            # TABLE FUNCTION
            # =================================================

            def show_table(data):

                if data.empty:

                    st.info(
                        "No colleges in this category."
                    )

                    return


                st.dataframe(
                    data,
                    use_container_width=True,
                    hide_index=True,

                    column_config={

                        "College":
                            st.column_config.TextColumn(
                                "College",
                                width="large"
                            ),

                        "Branch":
                            st.column_config.TextColumn(
                                "Branch",
                                width="large"
                            ),

                        "Type":
                            st.column_config.TextColumn(
                                "Type",
                                width="small"
                            ),

                        "Opening Rank":
                            st.column_config.NumberColumn(
                                "Opening Rank",
                                format="%d"
                            ),

                        "Predicted Closing":
                            st.column_config.NumberColumn(
                                "Predicted Closing",
                                format="%d"
                            ),

                        "Chances":
                            st.column_config.TextColumn(
                                "Chances",
                                width="medium"
                            )
                    }
                )


            # =================================================
            # ALL
            # =================================================

            with tab_all:

                show_table(
                    results
                )


            # =================================================
            # HIGH
            # =================================================

            with tab_high:

                show_table(
                    results[
                        results["Chances"]
                        == "🟢 HIGH"
                    ].reset_index(
                        drop=True
                    )
                )


            # =================================================
            # MEDIUM
            # =================================================

            with tab_medium:

                show_table(
                    results[
                        results["Chances"]
                        == "🟡 MEDIUM"
                    ].reset_index(
                        drop=True
                    )
                )


    st.divider()


# ════════════════════════════════════════════════════
# PAGE 3 — INSIGHTS
# ════════════════════════════════════════════════════

def insights_page():

    col_logo, col_home = st.columns(
        [5, 1]
    )


    with col_logo:

        st.markdown(
            "### 📊 JoSAA Dataset Insights"
        )


    with col_home:

        if st.button(
            "← Home",
            key="insights_home"
        ):

            st.session_state.page = "landing"

            st.rerun()


    st.divider()


    st.markdown(
        "Explore distribution metrics and "
        "breakdowns from historical JoSAA "
        "counseling records."
    )


    # ====================================================
    # HIGH LEVEL BREAKDOWN
    # ====================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        with st.container(
            border=True
        ):

            st.markdown(
                "#### 🏛️ Institute Types"
            )

            if "Institute_Type" in df.columns:

                type_counts = (
                    df["Institute_Type"]
                    .value_counts()
                )

                st.bar_chart(
                    type_counts
                )

            else:

                st.info(
                    "Institute type data "
                    "not available."
                )


    with col2:

        with st.container(
            border=True
        ):

            st.markdown(
                "#### 📂 Categories"
            )

            if "Category" in df.columns:

                cat_counts = (
                    df["Category"]
                    .value_counts()
                )

                st.bar_chart(
                    cat_counts
                )

            else:

                st.info(
                    "Category data "
                    "not available."
                )


    with col3:

        with st.container(
            border=True
        ):

            st.markdown(
                "#### 🌐 Quotas"
            )

            if "Quota" in df.columns:

                quota_counts = (
                    df["Quota"]
                    .value_counts()
                )

                st.bar_chart(
                    quota_counts
                )

            else:

                st.info(
                    "Quota data "
                    "not available."
                )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    st.subheader(
        "Explore Raw Data Sample"
    )


    st.dataframe(
        df.head(100),
        use_container_width=True
    )


    st.divider()


# ════════════════════════════════════════════════════
# ROUTER
# ════════════════════════════════════════════════════

if st.session_state.page == "landing":

    landing_page()

elif st.session_state.page == "predictor":

    predictor_page()

elif st.session_state.page == "insights":

    insights_page()