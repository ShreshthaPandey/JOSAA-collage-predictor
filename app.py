import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="JoSAA College Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# DARK BLUE GRADIENT THEME & CUSTOM STYLING
# ============================================================

st.markdown("""
<style>
/* =========================
   GLOBAL APP BACKGROUND
========================= */
.stApp {
    background:
        radial-gradient(
            circle at 0% 0%,
            rgba(30, 58, 138, 0.18),
            transparent 35%
        ),
        radial-gradient(
            circle at 100% 10%,
            rgba(29, 78, 216, 0.14),
            transparent 32%
        ),
        linear-gradient(
            135deg,
            #0f172a 0%,
            #1e293b 45%,
            #0f172a 100%
        );
    color: #f8fafc;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* =========================
   LANDING PAGE HERO
========================= */
.hero-box {
    padding: 65px 30px 45px 30px;
    text-align: center;
}

.hero-logo-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 9px 20px;
    border-radius: 50px;
    background: linear-gradient(135deg, #1e3a8a, #1d4ed8);
    color: #eff6ff;
    font-size: 14px;
    font-weight: 700;
    border: 1px solid #3b82f6;
    margin-bottom: 22px;
    box-shadow: 0 4px 20px rgba(37, 99, 235, 0.3);
}

.hero-title {
    font-size: 56px;
    font-weight: 800;
    letter-spacing: -2px;
    background: linear-gradient(90deg, #60a5fa, #93c5fd, #ffffff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 15px;
}

.hero-description {
    max-width: 720px;
    margin: auto;
    color: #94a3b8;
    font-size: 17px;
    line-height: 1.7;
}

/* =========================
   FEATURE CARDS
========================= */
.card {
    background: rgba(30, 41, 59, 0.85);
    border-radius: 22px;
    padding: 28px;
    border: 1px solid rgba(59, 130, 246, 0.25);
    box-shadow: 0 10px 35px rgba(15, 23, 42, 0.5);
    min-height: 185px;
    transition: 0.25s ease;
}

.card:hover {
    transform: translateY(-4px);
    border-color: rgba(96, 165, 250, 0.5);
    box-shadow: 0 18px 40px rgba(37, 99, 235, 0.2);
}

.card-icon {
    width: 48px;
    height: 48px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #1e3a8a, #2563eb);
    font-size: 22px;
    margin-bottom: 16px;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
}

.card-title {
    font-size: 18px;
    font-weight: 750;
    color: #f8fafc;
    margin-bottom: 8px;
}

.card-text {
    color: #94a3b8;
    font-size: 14px;
    line-height: 1.6;
}

/* =========================
   DASHBOARD CARD
========================= */
.dashboard-card {
    background: rgba(30, 41, 59, 0.9);
    border-radius: 24px;
    padding: 32px;
    border: 1px solid rgba(59, 130, 246, 0.3);
    box-shadow: 0 12px 40px rgba(15, 23, 42, 0.6);
}

.dashboard-title {
    font-size: 30px;
    font-weight: 800;
    color: #f8fafc;
}

.dashboard-subtitle {
    color: #94a3b8;
    margin-top: 5px;
    margin-bottom: 25px;
}

/* =========================
   CUSTOM BUTTONS
========================= */
.stButton > button {
    width: 100%;
    border: none;
    border-radius: 14px;
    padding: 13px 20px;
    font-size: 15px;
    font-weight: 700;
    color: white;
    background: linear-gradient(135deg, #2563eb, #1d4ed8, #1e40af);
    box-shadow: 0 8px 25px rgba(37, 99, 235, 0.4);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(59, 130, 246, 0.6);
}

/* =========================
   INPUT FIELDS
========================= */
.stSelectbox > div > div, .stTextInput input {
    background-color: rgba(15, 23, 42, 0.6) !important;
    color: #f8fafc !important;
    border-radius: 12px !important;
    border: 1px solid rgba(59, 130, 246, 0.3) !important;
}

/* =========================
   RESULTS & METRICS
========================= */
.result-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border-radius: 20px;
    padding: 20px 25px;
    border: 1px solid rgba(59, 130, 246, 0.4);
    margin-top: 25px;
}

.result-title {
    font-size: 22px;
    font-weight: 750;
    color: #f8fafc;
}

.metric-card {
    background: rgba(30, 41, 59, 0.8);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
    border: 1px solid rgba(59, 130, 246, 0.25);
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.4);
}

.metric-number {
    font-size: 26px;
    font-weight: 800;
    color: #60a5fa;
}

.metric-label {
    font-size: 12px;
    color: #94a3b8;
    margin-top: 5px;
    letter-spacing: 0.5px;
}

/* =========================
   FOOTER
========================= */
.footer {
    text-align: center;
    margin-top: 55px;
    padding: 25px 10px;
    border-top: 1px solid rgba(59, 130, 246, 0.2);
    color: #94a3b8;
    font-size: 13px;
}

.footer-name {
    color: #60a5fa;
    font-weight: 700;
}

.footer-repo {
    color: #60a5fa;
    font-weight: 600;
}

/* =========================
   RESPONSIVE DESIGN
========================= */
@media(max-width: 768px) {
    .hero-title { font-size: 38px; }
    .hero-description { font-size: 15px; }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL & DATA
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load("model/final_random_forest.pkl")
    encoders = joblib.load("model/encoders.pkl")
    features = joblib.load("model/features.pkl")
    return model, encoders, features

@st.cache_data
def load_data():
    return pd.read_pickle("model/processed_data.pkl")

model, encoders, features = load_model()
df = load_data()

# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

# ============================================================
# FOOTER COMPONENT
# ============================================================

def footer():
    st.markdown("""
        <div class="footer">
            Built with ❤️ by <span class="footer-name">Shreshtha Pandey</span>
            &nbsp; • &nbsp;
            <span class="footer-repo">JoSAA College Predictor</span>
            <br><br>
            Machine Learning Project &nbsp; • &nbsp; JoSAA 2020–2026 Dataset
        </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE 1 — HOME
# ============================================================

if st.session_state.page == "home":
    st.markdown("""
        <div class="hero-box">
            <div class="hero-logo-badge">⚡ JoSAA • MACHINE LEARNING ENGINE</div>
            <div class="hero-title">JoSAA College Predictor</div>
            <div class="hero-description">
                Discover your best-fit engineering colleges and branches based on your JEE rank, category, quota, and specific preferences. 
                Powered by historical JoSAA counselling datasets and advanced machine learning models.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Feature Cards
    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown("""
            <div class="card">
                <div class="card-icon">📊</div>
                <div class="card-title">Data Driven Insights</div>
                <div class="card-text">Analyzes historical opening and closing rank records across multiple counseling rounds.</div>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
            <div class="card">
                <div class="card-icon">🤖</div>
                <div class="card-title">Random Forest Model</div>
                <div class="card-text">Trained regression pipeline predicts customized closing ranks with high accuracy.</div>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
            <div class="card">
                <div class="card-icon">🎯</div>
                <div class="card-title">Smart Recommendations</div>
                <div class="card-text">Categorizes admission probability into High, Medium, and Low chances instantly.</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Start Button Layout
    _, center, _ = st.columns([1.5, 2, 1.5])
    with center:
        if st.button("LAUNCH PREDICTOR ➔", use_container_width=True):
            st.session_state.page = "predictor"
            st.rerun()

    footer()

# ============================================================
# PAGE 2 — PREDICTOR DASHBOARD
# ============================================================

else:
    st.markdown("""
        <div class="dashboard-card">
            <div class="dashboard-title">🎓 Admission Prediction Dashboard</div>
            <div class="dashboard-subtitle">Input your JEE exam details to query suitable colleges and branch choices.</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
        <div class="dashboard-card">
            <div class="dashboard-title">Enter Your Credentials</div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        exam = st.selectbox("Exam", ["JEE Main", "JEE Advanced"])
        
        # Rank entered via a text box (No increment/decrement arrows)
        rank_input = st.text_input("Your Rank", value="10000")
        
        # Safely parse text input to integer
        try:
            rank = int(rank_input.replace(",", "").strip())
        except ValueError:
            rank = 10000
            st.warning("⚠️ Please enter a valid number for your rank.")
    
    with col2:
        category = st.selectbox("Category", sorted(df["Category"].dropna().astype(str).unique()))
        quota = st.selectbox("Quota", sorted(df["Quota"].dropna().astype(str).unique()))
    
    with col3:
        gender = st.selectbox("Gender", sorted(df["Gender"].dropna().astype(str).unique()))
        pwd = st.selectbox("PwD", ["No", "Yes"])

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 PREDICT COLLEGES", use_container_width=True):
        if exam == "JEE Advanced":
            filtered_df = df[df["Institute_Type"] == "IIT"].copy()
        else:
            filtered_df = df[df["Institute_Type"] != "IIT"].copy()

        filtered_df = filtered_df[
            (filtered_df["Category"].astype(str) == str(category)) &
            (filtered_df["Quota"].astype(str) == str(quota)) &
            (filtered_df["Gender"].astype(str) == str(gender)) &
            (filtered_df["is_pwd"] == (1 if pwd == "Yes" else 0))
        ].copy()

        if filtered_df.empty:
            st.warning("No matching colleges found for these specific preferences.")
        else:
            latest_year = filtered_df["Year"].max()
            candidates = filtered_df[filtered_df["Year"] == latest_year].copy()

            X_candidates = candidates[features]
            candidates["Predicted_Closing_Rank"] = model.predict(X_candidates)

            def chance(predicted, student):
                if student <= predicted * 0.80:
                    return "HIGH"
                elif student <= predicted * 1.10:
                    return "MEDIUM"
                else:
                    return "LOW"

            candidates["Chances"] = candidates["Predicted_Closing_Rank"].apply(lambda x: chance(x, rank))

            chance_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            candidates["chance_order"] = candidates["Chances"].map(chance_order)

            candidates = candidates.sort_values(["chance_order", "Predicted_Closing_Rank"])
            candidates = candidates.drop_duplicates(subset=["Institute", "Academic_Program_Name"])

            st.markdown("""
                <div class="result-card">
                    <div class="result-title">🎯 Recommended College Options</div>
                </div>
            """, unsafe_allow_html=True)

            st.caption(f"Based on available {latest_year} JoSAA counseling insights")

            result = candidates[["Institute", "Academic_Program_Name", "Institute_Type", "Chances"]].copy()
            result.columns = ["College Name", "Branch", "College Type", "Chances"]

            st.dataframe(result, use_container_width=True, hide_index=True, height=500)

            st.markdown("<br>", unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3, gap="large")

            with m1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-number">{rank:,}</div>
                        <div class="metric-label">YOUR RANK</div>
                    </div>
                """, unsafe_allow_html=True)

            with m2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-number">{len(result)}</div>
                        <div class="metric-label">COLLEGE OPTIONS</div>
                    </div>
                """, unsafe_allow_html=True)

            with m3:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-number">{latest_year}</div>
                        <div class="metric-label">LATEST DATA YEAR</div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.info("⚠️ Chances are machine learning estimates based on past cutoffs and do not guarantee official admission allocation.")

    footer()