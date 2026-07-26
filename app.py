import streamlit as st
import pandas as pd
import numpy as np
import joblib
from google import genai

# --- PAGE SETUP ---
st.set_page_config(
    page_title="AgriSoil AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- TRANSLATION DICTIONARY ---
T = {
    "English": {
        "features": "Features",
        "how_it_works": "How It Works",
        "ai_chat": "AI Chat",
        "analyze_btn": "Analyze Now ↗",
        "ask_btn": "Ask Chatbot ↗",
        "hero_title": 'Grow Smarter with <span class="serif-gold">Data-</span><br><span class="serif-gold">Driven</span> Soil Intelligence',
        "hero_tagline": '<span class="logo-highlight">AgriSoil</span> turns soil test numbers into instant AI diagnostics and clear fertilizer recommendations so you know exactly what your crops need.',
        
        # Features Text
        "f1_title": "Lab Diagnostics",
        "f1_desc": "Automated ML classifier for primary parameters & micronutrients in seconds.",
        "f2_title": "Gemini AI Advisor",
        "f2_desc": "Adaptive, concise AI advice on soil chemistry, fertilizer dosing, and field queries.",
        "f3_title": "Fertilizer Prescriptions",
        "f3_desc": "Actionable soil recovery advice and exact fertilizer dosage recommendations.",
        
        # How It Works Text
        "h1_title": "Enter Parameters",
        "h1_desc": "Input test parameters like pH, EC, NPK, and micronutrients to analyze soil health.",
        "h2_title": "Run Diagnostics",
        "h2_desc": "The ML model classifies soil state and displays visual progress indicators.",
        "h3_title": "Consult Gemini AI",
        "h3_desc": "Ask follow-up queries to refine field management, soil recovery, and fertilizer dosage.",

        # Diagnostic Page
        "primary_header": "🧪 Primary & Secondary Parameters",
        "micro_header": "🔬 Micronutrients (ppm)",
        "page_heading": "Enter Soil Parameters for Analysis",
        "analyze_now_btn": "Analyze Now",
        "breakdown_header": "📊 Parameter Status Breakdown",
        "prescriptions_header": "💡 Actionable Fertilizer Prescriptions",
        "optimal_msg": "✅ All parameters are in optimal range!",
        "ph_label": "pH Level (0 - 14)",
        "ec_label": "Electrical Conductivity - E.C. (dS/m)",
        "oc_label": "Organic Carbon (%)",
        "p_label": "Available P₂O₅ (kg/ha)",
        "k_label": "Available K₂O (kg/ha)",
        "zn_label": "Available Zinc - Zn (ppm)",
        "fe_label": "Available Iron - Fe (ppm)",
        "cu_label": "Available Copper - Cu (ppm)",
        "mn_label": "Available Manganese - Mn (ppm)",
        "back_home": "← Back to Home",

        # Parameter Names for Breakdown
        "param_ph": "pH Level",
        "param_ec": "Electrical Conductivity (EC)",
        "param_oc": "Organic Carbon",
        "param_p": "Phosphorus (P)",
        "param_k": "Potassium (K)",
        "param_zn": "Zinc (Zn)",
        "param_fe": "Iron (Fe)",
        "param_cu": "Copper (Cu)",
        "param_mn": "Manganese (Mn)"
    },
    "हिंदी": {
        "features": "विशेषताएं",
        "how_it_works": "यह कैसे काम करता है",
        "ai_chat": "एआई चैट",
        "analyze_btn": "अभी विश्लेषण करें ↗",
        "ask_btn": "चैटबॉट से पूछें ↗",
        "hero_title": 'डेटा-संचालित मृदा बुद्धिमत्ता के साथ <span class="serif-gold">स्मार्ट तरीके</span> से फसल उगाएं',
        "hero_tagline": '<span class="logo-highlight">एग्रीसोइल</span> मिट्टी परीक्षण के नंबरों को तुरंत एआई निदान और स्पष्ट उर्वरक सिफारिशों में बदल देता है ताकि आप ठीक से जान सकें कि आपकी फसलों को क्या चाहिए।',
        
        # Features Text
        "f1_title": "प्रयोगशाला निदान",
        "f1_desc": "सेकंडों में प्राथमिक मापदंडों और सूक्ष्म पोषक तत्वों के लिए स्वचालित मशीन लर्निंग वर्गीकरण।",
        "f2_title": "जेमिनी एआई सलाहकार",
        "f2_desc": "मृदा रसायन शास्त्र, उर्वरक खुराक और कृषि प्रश्नों पर अनुकूल और संक्षिप्त एआई सलाह।",
        "f3_title": "उर्वरक नुस्खे",
        "f3_desc": "पोषक तत्वों की कमी के लिए व्यावहारिक मृदा सुधार सलाह और सटीक उर्वरक मात्रा सिफारिशें।",

        # How It Works Text
        "h1_title": "मापदंड दर्ज करें",
        "h1_desc": "मृदा स्वास्थ्य का विश्लेषण करने के लिए पीएच, ईसी, एनपीके और सूक्ष्म पोषक तत्वों के मान दर्ज करें।",
        "h2_title": "निदान चलाएं",
        "h2_desc": "मशीन लर्निंग मॉडल मृदा की स्थिति का वर्गीकरण करता है और दृश्य प्रगति संकेतक दिखाता है।",
        "h3_title": "जेमिनी एआई से सलाह लें",
        "h3_desc": "खेत प्रबंधन, मृदा सुधार और उर्वरक मात्रा को परिष्कृत करने के लिए प्रश्न पूछें।",

        # Diagnostic Page
        "primary_header": "🧪 प्राथमिक एवं द्वितीयक मापदंड",
        "micro_header": "🔬 सूक्ष्म पोषक तत्व (ppm)",
        "page_heading": "विश्लेषण के लिए मिट्टी के मापदंड दर्ज करें",
        "analyze_now_btn": "अभी विश्लेषण करें",
        "breakdown_header": "📊 मापदंड स्थिति विवरण",
        "prescriptions_header": "💡 व्यावहारिक उर्वरक नुस्खे",
        "optimal_msg": "✅ सभी मापदंड इष्टतम सीमा में हैं!",
        "ph_label": "पीएच स्तर (0 - 14)",
        "ec_label": "विद्युत चालकता - E.C. (dS/m)",
        "oc_label": "जैविक कार्बन (%)",
        "p_label": "उपलब्ध P₂O₅ (kg/ha)",
        "k_label": "उपलब्ध K₂O (kg/ha)",
        "zn_label": "उपलब्ध जिंक - Zn (ppm)",
        "fe_label": "उपलब्ध लोहा - Fe (ppm)",
        "cu_label": "उपलब्ध तांबा - Cu (ppm)",
        "mn_label": "उपलब्ध मैंगनीज - Mn (ppm)",
        "back_home": "← मुख्य पृष्ठ पर वापस जाएं",

        # Parameter Names for Breakdown
        "param_ph": "पीएच स्तर (pH Level)",
        "param_ec": "विद्युत चालकता (EC)",
        "param_oc": "जैविक कार्बन (Organic Carbon)",
        "param_p": "फास्फोरस (P)",
        "param_k": "पोटाश / पोटेशियम (K)",
        "param_zn": "जिंक (Zn)",
        "param_fe": "लोहा / आयरन (Fe)",
        "param_cu": "तांबा (Cu)",
        "param_mn": "मैंगनीज (Mn)"
    }
}

# --- ML CONDITION TRANSLATIONS ---
condition_translations = {
    "Normal Soil Reaction": "सामान्य मृदा (Normal Soil)",
    "Alkaline Soil": "क्षारीय मृदा (Alkaline Soil)",
    "Saline Soil": "लवणीय मृदा (Saline Soil)",
    "Acidic Soil": "अम्लीय मृदा (Acidic Soil)"
}

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;0,700;1,600&family=Inter:wght@400;500;600&family=Outfit:wght@500;600;700;800&display=swap');

    html { scroll-behavior: smooth; }
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif; 
        background: linear-gradient(90deg, #1b4332 0%, #2d6a4f 50%, #74a57f 100%) !important; 
        color: #FFFFFF !important; 
    }
    .stApp { 
        background: linear-gradient(90deg, #1b4332 0%, #2d6a4f 50%, #74a57f 100%) !important; 
        background-attachment: fixed !important;
    }

    /* HIDE ANCHOR LINK ICONS ON ALL HEADINGS */
    .anchor-link, [data-testid="stHeaderActionElements"], a.aria-hidden {
        display: none !important;
    }

    /* TYPOGRAPHY FOR HERO */
    .hero-title {
        font-family: 'Playfair Display', serif !important;
        font-size: 3.4rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        line-height: 1.25 !important;
        margin-bottom: 1.2rem !important;
    }
    .serif-gold {
        font-family: 'Playfair Display', serif !important;
        font-style: italic !important;
        color: #E9B44C !important;
        font-weight: 600 !important;
    }
    .logo-highlight {
        color: #E9B44C !important;
        font-weight: 700 !important;
    }
    .hero-sub {
        font-family: 'Inter', sans-serif !important;
        font-size: 1.1rem !important;
        color: #E2F0E7 !important;
        line-height: 1.6 !important;
        margin-bottom: 2rem !important;
        max-width: 92% !important;
    }

    /* CLEAN MODERN FLOATING IMAGE CONTAINER WITHOUT BADGES */
    .hero-image-card {
        background: #F5EBE0;
        border-radius: 24px;
        padding: 1.2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.25);
        border: 1px solid rgba(255,255,255,0.2);
        display: flex;
        justify-content: center;
        align-items: center;
        overflow: hidden;
    }
    .hero-image-card img {
        width: 100%;
        height: 320px;
        object-fit: cover;
        border-radius: 16px;
    }

    /* YELLOW / GOLDEN BUTTONS */
    div.stButton > button[kind="primary"] {
        background-color: #E9B44C !important;
        color: #1E3A2F !important;
        border-radius: 50px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        padding: 0.85rem 2rem !important;
        border: none !important;
        transition: all 0.25s ease-in-out !important;
        box-shadow: 0 6px 20px rgba(233, 180, 76, 0.3) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #f3c66b !important;
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 25px rgba(233, 180, 76, 0.45) !important;
    }

    div.stButton > button[kind="secondary"] {
        background-color: transparent !important;
        color: #FFFFFF !important;
        border: 2px solid #FFFFFF !important;
        border-radius: 50px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 0.75rem 1.8rem !important;
        transition: all 0.25s ease-in-out !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: #E9B44C !important;
        color: #1E3A2F !important;
        border-color: #E9B44C !important;
        transform: translateY(-2px);
    }

    /* BEIGE BENTO FEATURE & HOW IT WORKS CARDS */
    .bento-card-beige {
        background-color: #F5EBE0;
        color: #1E3A2F !important;
        border-radius: 20px;
        padding: 2rem;
        height: 250px !important;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        overflow: hidden;
    }
    .bento-card-beige h4, .bento-card-beige h5, .bento-card-beige h3 {
        color: #1E3A2F !important;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 1.4rem;
        margin-bottom: 0.5rem;
    }
    .bento-card-beige p {
        color: #4A5759 !important;
        font-family: 'Inter', sans-serif;
        font-size: 1.0rem;
        font-weight: 600;
        line-height: 1.4;
        margin: 0;
    }

    /* DARK CARD CONTAINER FOR DIAGNOSTIC BREAKDOWN */
    .diagnostic-dark-card {
        background-color: #1E3A2F;
        border: 1px solid #3A5A40;
        color: #FFFFFF !important;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 1.5rem;
    }
    .diagnostic-dark-card h3 {
        color: #E9B44C !important;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 1.4rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3A5A40;
        padding-bottom: 0.5rem;
    }
    
    .diagnostic-dark-card li, .diagnostic-dark-card ul li, div[data-testid="stMarkdownContainer"] ul li {
        color: #FFD166 !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem;
    }

    /* FORM INPUTS & INCREASED NUMBER FONT SIZE */
    div[data-baseweb="input"] {
        background-color: #F5EBE0 !important;
        border: 1px solid #D6C3B2 !important;
        border-radius: 12px !important;
    }
    div[data-baseweb="input"] input {
        color: #1E3A2F !important;
        font-size: 1.5rem !important;
        font-weight: 800 !important;
    }
    div[data-testid="stNumberInput"] label p, div[data-testid="stTextInput"] label p {
        color: #FFFFFF !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
    }

    /* ABSOLUTE FORCE: STRICT DARK GREEN TEXT FOR ALL CHAT MESSAGES, ZERO YELLOW OR LIGHT TEXT */
    div[data-testid="stChatMessage"] {
        background-color: #F5EBE0 !important;
        border: 1px solid #D6C3B2 !important;
        border-radius: 16px !important;
        padding: 1.2rem 1.6rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    }
    div[data-testid="stChatMessage"] p, 
    div[data-testid="stChatMessage"] li, 
    div[data-testid="stChatMessage"] span, 
    div[data-testid="stChatMessage"] strong, 
    div[data-testid="stChatMessage"] em, 
    div[data-testid="stChatMessage"] b, 
    div[data-testid="stChatMessage"] i, 
    div[data-testid="stChatMessage"] h1, 
    div[data-testid="stChatMessage"] h2, 
    div[data-testid="stChatMessage"] h3, 
    div[data-testid="stChatMessage"] h4, 
    div[data-testid="stChatMessage"] * {
        font-size: 1.15rem !important;
        line-height: 1.65 !important;
        color: #1E3A2F !important;
        -webkit-text-fill-color: #1E3A2F !important;
        background: transparent !important;
    }

    /* PROGRESS BARS & HIGH VISIBILITY STATUS BADGES */
    .bar-container { width: 100%; background-color: #3A5A40; border-radius: 8px; height: 14px; margin: 6px 0 14px 0; overflow: hidden; }
    .bar-fill-opt { height: 100%; background-color: #00F5D4; border-radius: 8px; }
    .bar-fill-warn { height: 100%; background-color: #FFB703; border-radius: 8px; }
    .bar-fill-danger { height: 100%; background-color: #FF6B6B; border-radius: 8px; }

    /* FLOATING YELLOW AI BUTTON */
    .floating-ai-btn {
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 999999;
        background-color: #E9B44C;
        color: #1E3A2F !important;
        border-radius: 50px;
        padding: 12px 26px;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 1.1rem;
        text-decoration: none !important;
        box-shadow: 0 10px 30px rgba(233, 180, 76, 0.4);
        display: flex;
        align-items: center;
        gap: 8px;
        transition: all 0.25s ease-in-out;
    }
    .floating-ai-btn:hover {
        background-color: #f3c66b;
        transform: translateY(-3px) scale(1.03);
        box-shadow: 0 15px 35px rgba(233, 180, 76, 0.6);
    }
    </style>
""", unsafe_allow_html=True)

# --- QUERY PARAMS ROUTING FOR FLOATING BUTTON ---
if "view" in st.query_params and st.query_params["view"] == "chat":
    st.session_state.active_view = "chat"
    st.query_params.clear()

# --- LOAD ML ASSETS ---
@st.cache_resource
def load_ml_assets():
    model = joblib.load('soil_classifier_model.pkl')
    features = joblib.load('feature_names.pkl')
    return model, features

try:
    model, feature_names = load_ml_assets()
except Exception as e:
    st.error("⚠️ Model files not found! Ensure 'soil_classifier_model.pkl' and 'feature_names.pkl' exist in the project directory.")
    st.stop()

if 'active_view' not in st.session_state:
    st.session_state.active_view = 'home'

# ==========================================
# 🌟 TOP NAVBAR
# ==========================================
col_brand, col_f, col_h, col_chat, col_lang = st.columns([2.8, 1, 1.2, 1, 1.2])

with col_lang:
    lang = st.selectbox("🌐 Language", ["English", "हिंदी"], label_visibility="collapsed")

with col_brand:
    st.markdown('<a href="?view=home" target="_self" style="text-decoration:none; color:#E9B44C; font-weight:800; font-family:\'Outfit\', sans-serif; font-size:2rem; display:inline-block;">🌱 AgriSoil</a>', unsafe_allow_html=True)

with col_f:
    st.markdown(f'<a href="#features" style="text-decoration:none; color:#E2F0E7; font-weight:600; font-family:\'Outfit\', sans-serif; display:inline-block; margin-top:12px;">{T[lang]["features"]}</a>', unsafe_allow_html=True)

with col_h:
    st.markdown(f'<a href="#how-it-works" style="text-decoration:none; color:#E2F0E7; font-weight:600; font-family:\'Outfit\', sans-serif; display:inline-block; margin-top:12px;">{T[lang]["how_it_works"]}</a>', unsafe_allow_html=True)

with col_chat:
    if st.button(T[lang]["ai_chat"], type="secondary"):
        st.session_state.active_view = 'chat'
        st.rerun()

st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.2); margin: 0.8rem 0 1.5rem 0;'>", unsafe_allow_html=True)

# ==========================================
# VIEW 1: HERO HOMEPAGE
# ==========================================
if st.session_state.active_view == 'home':
    
    col_h_text, col_h_img = st.columns([1.3, 1.1], gap="large")

    with col_h_text:
        st.markdown(f"""
            <div style="padding-top: 0.5rem;">
                <h1 class="hero-title">{T[lang]["hero_title"]}</h1>
                <p class="hero-sub">{T[lang]["hero_tagline"]}</p>
            </div>
        """, unsafe_allow_html=True)

        b_c1, b_c2 = st.columns(2)
        with b_c1:
            if st.button(T[lang]["analyze_btn"], type="primary", use_container_width=True):
                st.session_state.active_view = 'lab'
                st.rerun()
        with b_c2:
            if st.button(T[lang]["ask_btn"], type="secondary", use_container_width=True):
                st.session_state.active_view = 'chat'
                st.rerun()

    with col_h_img:
        st.markdown("""
            <div class="hero-image-card">
                <img src="https://images.unsplash.com/photo-1625246333195-78d9c38ad449?auto=format&fit=crop&w=1000&q=80" alt="Agriculture 3D Farm">
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- SUBSECTION: FEATURES (BEIGE CARDS) ---
    st.markdown("<div id='features'></div><br><hr style='border-color:rgba(255,255,255,0.2);'><br>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='font-family: Outfit; color: #E9B44C; text-align: center; font-size:2.2rem; font-weight:800; margin-bottom: 2rem;'>{T[lang]['features']}</h2>", unsafe_allow_html=True)
    
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(f"""
            <div class="bento-card-beige">
                <div style="font-size:2rem; margin-bottom:0.3rem;">🧪</div>
                <h4>{T[lang]['f1_title']}</h4>
                <p>{T[lang]['f1_desc']}</p>
            </div>
        """, unsafe_allow_html=True)

    with f2:
        st.markdown(f"""
            <div class="bento-card-beige">
                <div style="font-size:2rem; margin-bottom:0.3rem;">🤖</div>
                <h4>{T[lang]['f2_title']}</h4>
                <p>{T[lang]['f2_desc']}</p>
            </div>
        """, unsafe_allow_html=True)

    with f3:
        st.markdown(f"""
            <div class="bento-card-beige">
                <div style="font-size:2rem; margin-bottom:0.3rem;">🌾</div>
                <h4>{T[lang]['f3_title']}</h4>
                <p>{T[lang]['f3_desc']}</p>
            </div>
        """, unsafe_allow_html=True)

    # --- SUBSECTION: HOW IT WORKS (BEIGE CARDS) ---
    st.markdown("<div id='how-it-works'></div><br><hr style='border-color:rgba(255,255,255,0.2);'><br>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='font-family: Outfit; color: #E9B44C; text-align: center; font-size:2.2rem; font-weight:800; margin-bottom: 2rem;'>{T[lang]['how_it_works']}</h2>", unsafe_allow_html=True)
    
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown(f"""
            <div class="bento-card-beige">
                <h3 style="color:#1E3A2F; margin:0; font-size:1.8rem; font-weight:800;">01</h3>
                <h5>{T[lang]['h1_title']}</h5>
                <p>{T[lang]['h1_desc']}</p>
            </div>
        """, unsafe_allow_html=True)

    with h2:
        st.markdown(f"""
            <div class="bento-card-beige">
                <h3 style="color:#1E3A2F; margin:0; font-size:1.8rem; font-weight:800;">02</h3>
                <h5>{T[lang]['h2_title']}</h5>
                <p>{T[lang]['h2_desc']}</p>
            </div>
        """, unsafe_allow_html=True)

    with h3:
        st.markdown(f"""
            <div class="bento-card-beige">
                <h3 style="color:#1E3A2F; margin:0; font-size:1.8rem; font-weight:800;">03</h3>
                <h5>{T[lang]['h3_title']}</h5>
                <p>{T[lang]['h3_desc']}</p>
            </div>
        """, unsafe_allow_html=True)

# ==========================================
# VIEW 2: DEDICATED NUMERIC ANALYSIS PAGE
# ==========================================
elif st.session_state.active_view == 'lab':
    
    if st.button(T[lang]["back_home"], type="secondary"):
        st.session_state.active_view = 'home'
        st.rerun()

    st.markdown(f"<h2 style='font-family: Outfit; color: #E9B44C; margin-top: 1rem; font-size:2.2rem; font-weight:800;'>{T[lang]['page_heading']}</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col_macro, col_micro = st.columns(2)

    with col_macro:
        st.markdown(f'<h3 style="color:#E9B44C; font-size:1.35rem; font-weight:700; border-bottom:2px solid rgba(255,255,255,0.2); padding-bottom:0.5rem;">{T[lang]["primary_header"]}</h3>', unsafe_allow_html=True)
        val_ph = st.number_input(T[lang]["ph_label"], min_value=0.0, max_value=14.0, value=None, step=0.05, placeholder="e.g. 7.2")
        val_ec = st.number_input(T[lang]["ec_label"], min_value=0.0, max_value=20.0, value=None, step=0.01, placeholder="e.g. 0.5")
        val_oc = st.number_input(T[lang]["oc_label"], min_value=0.0, max_value=5.00, value=None, step=0.01, placeholder="e.g. 0.6")
        val_p  = st.number_input(T[lang]["p_label"], min_value=0.0, max_value=500.0, value=None, step=1.0, placeholder="e.g. 30")
        val_k  = st.number_input(T[lang]["k_label"], min_value=0.0, max_value=1000.0, value=None, step=1.0, placeholder="e.g. 150")

    with col_micro:
        st.markdown(f'<h3 style="color:#E9B44C; font-size:1.35rem; font-weight:700; border-bottom:2px solid rgba(255,255,255,0.2); padding-bottom:0.5rem;">{T[lang]["micro_header"]}</h3>', unsafe_allow_html=True)
        val_zn = st.number_input(T[lang]["zn_label"], min_value=0.0, max_value=20.0, value=None, step=0.01, placeholder="e.g. 1.2")
        val_fe = st.number_input(T[lang]["fe_label"], min_value=0.0, max_value=50.0, value=None, step=0.05, placeholder="e.g. 5.0")
        val_cu = st.number_input(T[lang]["cu_label"], min_value=0.0, max_value=20.0, value=None, step=0.01, placeholder="e.g. 0.4")
        val_mn = st.number_input(T[lang]["mn_label"], min_value=0.0, max_value=50.0, value=None, step=0.05, placeholder="e.g. 4.0")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(T[lang]["analyze_now_btn"], type="primary", use_container_width=True):
        
        inputs_list = [val_ph, val_ec, val_oc, val_p, val_k, val_zn, val_fe, val_cu, val_mn]
        if any(v is None for v in inputs_list):
            st.error("⚠️ Please fill in all soil parameter fields before running the analysis!" if lang == "English" else "⚠️ कृपया विश्लेषण चलाने से पहले सभी मिट्टी के मापदंड भरें!")
        else:
            input_data = pd.DataFrame([[val_ph, val_ec, val_oc, val_p, val_k, val_zn, val_fe, val_cu, val_mn]], columns=feature_names)
            predicted_condition = model.predict(input_data)[0]

            # STACKED RESULTS LAYOUT WITH HIGH VISIBILITY DARK CARDS
            st.markdown(f'<div class="diagnostic-dark-card"><h3>{T[lang]["breakdown_header"]}</h3>', unsafe_allow_html=True)
            
            def render_custom_param_bar(name, val, status_text, badge_color, bar_class, max_scale, unit=""):
                pct = min(100, max(5, int((val / max_scale) * 100)))
                st.markdown(f"""
                    <div style="margin-bottom: 1rem;">
                        <div style="display:flex; justify-content:space-between; font-weight:700; color:#FFFFFF; font-size: 1.05rem;">
                            <span>{name} ({val} {unit})</span>
                            <span style="color:{badge_color}; font-weight:800;">{status_text}</span>
                        </div>
                        <div class="bar-container">
                            <div class="{bar_class}" style="width: {pct}%;"></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            # 1. pH Evaluation (< 7: Low, 7 - 8.5: Normal, > 8.5: High)
            if val_ph < 7.0:
                ph_status = "Low" if lang == "English" else "कम"
                ph_badge = "#FFB703"
                ph_bar = "bar-fill-warn"
            elif val_ph <= 8.5:
                ph_status = "Normal" if lang == "English" else "सामान्य"
                ph_badge = "#00F5D4"
                ph_bar = "bar-fill-opt"
            else:
                ph_status = "High" if lang == "English" else "अधिक"
                ph_badge = "#FF6B6B"
                ph_bar = "bar-fill-danger"
            render_custom_param_bar(T[lang]["param_ph"], val_ph, ph_status, ph_badge, ph_bar, 14.0)

            # 2. EC Evaluation (< 1.5: Normal, 1.5 - 3.0: Critical, > 3.0: Injurious)
            if val_ec < 1.5:
                ec_status = "Normal" if lang == "English" else "सामान्य"
                ec_badge = "#00F5D4"
                ec_bar = "bar-fill-opt"
            elif val_ec <= 3.0:
                ec_status = "Critical" if lang == "English" else "गंभीर"
                ec_badge = "#FF6B6B"
                ec_bar = "bar-fill-danger"
            else:
                ec_status = "Injurious" if lang == "English" else "हानिकारक"
                ec_badge = "#FF6B6B"
                ec_bar = "bar-fill-danger"
            render_custom_param_bar(T[lang]["param_ec"], val_ec, ec_status, ec_badge, ec_bar, 5.0, "dS/m")

            # Helper for other parameters: Deficient, Optimal, High
            def render_param_bar(name_key, val, low_thresh, high_thresh, max_scale, unit=""):
                name = T[lang][name_key]
                pct = min(100, max(5, int((val / max_scale) * 100)))
                
                if val < low_thresh:
                    status_text = "Deficient" if lang == "English" else "कम"
                    bar_class = "bar-fill-warn"
                    badge_style = "color:#FFB703; font-weight:800;"
                elif val > high_thresh:
                    status_text = "High" if lang == "English" else "अधिक"
                    bar_class = "bar-fill-danger"
                    badge_style = "color:#FF6B6B; font-weight:800;"
                else:
                    status_text = "Optimal" if lang == "English" else "इष्टतम"
                    bar_class = "bar-fill-opt"
                    badge_style = "color:#00F5D4; font-weight:800;"

                st.markdown(f"""
                    <div style="margin-bottom: 1rem;">
                        <div style="display:flex; justify-content:space-between; font-weight:700; color:#FFFFFF; font-size: 1.05rem;">
                            <span>{name} ({val} {unit})</span>
                            <span style="{badge_style}">{status_text}</span>
                        </div>
                        <div class="bar-container">
                            <div class="{bar_class}" style="width: {pct}%;"></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            # 3. Organic Carbon (< 0.5: Deficient, 0.5 - 0.75: Optimal, > 0.75: High)
            render_param_bar("param_oc", val_oc, 0.5, 0.75, 2.0, "%")

            # 4. Phosphorus P2O5 (< 23: Deficient, 23 - 56: Optimal, > 56: High)
            render_param_bar("param_p", val_p, 23, 56, 100.0, "kg/ha")

            # 5. Potassium K2O (< 144: Deficient, 144 - 336: Optimal, > 336: High)
            render_param_bar("param_k", val_k, 144, 336, 500.0, "kg/ha")

            # 6. Zinc (Critical Limit: 0.6 ppm)
            render_param_bar("param_zn", val_zn, 0.6, 2.0, 5.0, "ppm")

            # 7. Iron (Critical Limit: Fe < 4.5 is Deficient, otherwise Optimal)
            fe_status_text = "Deficient" if val_fe < 4.5 else "Optimal"
            if lang != "English":
                fe_status_text = "कम" if val_fe < 4.5 else "इष्टतम"
            fe_badge = "#FFB703" if val_fe < 4.5 else "#00F5D4"
            fe_bar = "bar-fill-warn" if val_fe < 4.5 else "bar-fill-opt"
            render_custom_param_bar(T[lang]["param_fe"], val_fe, fe_status_text, fe_badge, fe_bar, 20.0, "ppm")

            # 8. Copper (Critical Limit: 0.2 ppm)
            render_param_bar("param_cu", val_cu, 0.2, 2.0, 5.0, "ppm")

            # 9. Manganese (Critical Limit: 2.0 ppm)
            render_param_bar("param_mn", val_mn, 2.0, 10.0, 20.0, "ppm")

            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="diagnostic-dark-card"><h3>{T[lang]["prescriptions_header"]}</h3>', unsafe_allow_html=True)
            prescriptions = []
            
            if lang == "English":
                if val_ph < 7.0: prescriptions.append("🔴 **pH Low (< 7.0):** Apply Agricultural Lime treatment to raise pH.")
                elif val_ph > 8.5: prescriptions.append("🟡 **pH High (> 8.5):** Apply Gypsum @ 500 kg/ha or elemental sulfur.")
                
                if 1.5 <= val_ec <= 3.0: prescriptions.append("⚠️ **EC Critical (1.5 - 3.0 dS/m):** Moderate salinity risk; leach soil with good quality irrigation water.")
                elif val_ec > 3.0: prescriptions.append("🚨 **EC Injurious (> 3.0 dS/m):** High salinity toxicity! Heavy leaching and organic amendments required before planting.")

                if val_oc < 0.5: prescriptions.append("📉 **Organic Carbon Deficient (< 0.5%):** Add Farmyard Manure (FYM) @ 10 tonnes/ha.")
                elif val_oc > 0.75: prescriptions.append("📈 **Organic Carbon High (> 0.75%):** Excellent organic status; maintain regular crop residue management.")
                
                if val_p < 23: prescriptions.append("📉 **Phosphorus Deficient (< 23 kg/ha):** Apply DAP or SSP.")
                elif val_p > 56: prescriptions.append("📈 **Phosphorus High (> 56 kg/ha):** Withhold P fertilizer applications; grow cover crops to absorb excess P.")
                
                if val_k < 144: prescriptions.append("📉 **Potassium Deficient (< 144 kg/ha):** Apply Muriate of Potash (MOP).")
                elif val_k > 336: prescriptions.append("📈 **Potassium High (> 336 kg/ha):** Avoid potash fertilizers; ensure good drainage to prevent salt accumulation.")

                if val_zn < 0.6: prescriptions.append("🌾 **Zinc Deficient (< 0.6 ppm):** Apply Zinc Sulfate (ZnSO₄) @ 25 kg/ha.")
                if val_fe < 4.5: prescriptions.append("🌾 **Iron Deficient (< 4.5 ppm):** Foliar spray of 0.5% Ferrous Sulfate solution.")
                if val_cu < 0.2: prescriptions.append("🌾 **Copper Deficient (< 0.2 ppm):** Apply Copper Sulfate solution.")
                if val_mn < 2.0: prescriptions.append("🌾 **Manganese Deficient (< 2.0 ppm):** Apply Manganese Sulfate spray.")
            else:
                if val_ph < 7.0: prescriptions.append("🔴 **पीएच कम (< 7.0):** पीएच बढ़ाने के लिए कृषि चूना (Lime) का प्रयोग करें।")
                elif val_ph > 8.5: prescriptions.append("🟡 **पीएच अधिक (> 8.5):** 500 किग्रा/हेक्टेयर जिप्सम या सल्फर का प्रयोग करें।")
                
                if 1.5 <= val_ec <= 3.0: prescriptions.append("⚠️ **ईसी गंभीर (1.5 - 3.0 dS/m):** मध्यम लवणता जोखिम; अच्छी गुणवत्ता वाले पानी से सिंचाई करें।")
                elif val_ec > 3.0: prescriptions.append("🚨 **ईसी हानिकारक (> 3.0 dS/m):** उच्च लवणता विषाक्तता! रोपण से पहले भारी सिंचाई और लीचिंग आवश्यक है।")

                if val_oc < 0.5: prescriptions.append("📉 **जैविक कार्बन अपूर्ण (< 0.5%):** 10 टन/हेक्टेयर गोबर की खाद (FYM) डालें।")
                elif val_oc > 0.75: prescriptions.append("📈 **जैविक कार्बन अधिक (> 0.75%):** उत्कृष्ट जैविक स्थिति; नियमित फसल अवशेष प्रबंधन बनाए रखें।")
                
                if val_p < 23: prescriptions.append("📉 **फास्फोरस अपूर्ण (< 23 kg/ha):** डीएपी या एसएसपी का प्रयोग करें।")
                elif val_p > 56: prescriptions.append("📈 **फास्फोरस अधिक (> 56 kg/ha):** फॉस्फोरस उर्वरक का प्रयोग बंद करें और कवर फसलें उगाएं।")
                
                if val_k < 144: prescriptions.append("📉 **पोटाश अपूर्ण (< 144 kg/ha):** एमओपी (MOP) का प्रयोग करें।")
                elif val_k > 336: prescriptions.append("📈 **पोटाश अधिक (> 336 kg/ha):** पोटाश उर्वरक से बचें और जल निकासी का ध्यान रखें।")

                if val_zn < 0.6: prescriptions.append("🌾 **जिंक अपूर्ण (< 0.6 ppm):** 25 किलोग्राम/हेक्टेयर जिंक सल्फेट डालें।")
                if val_fe < 4.5: prescriptions.append("🌾 **लोहा अपूर्ण (< 4.5 ppm):** 0.5% फेरस सल्फेट का छिड़काव करें।")
                if val_cu < 0.2: prescriptions.append("🌾 **तांबा अपूर्ण (< 0.2 ppm):** कॉपर सल्फेट का प्रयोग करें।")
                if val_mn < 2.0: prescriptions.append("🌾 **मैंगनीज अपूर्ण (< 2.0 ppm):** मैंगनीज सल्फेट का छिड़काव करें।")

            if prescriptions:
                for item in prescriptions:
                    st.markdown(f"- {item}")
            else:
                st.markdown(f'<span style="color: #FFD166; font-weight: 700; font-size: 1.15rem;">{T[lang]["optimal_msg"]}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# VIEW 3: DEDICATED GEMINI AI CHATBOT PAGE
# ==========================================
elif st.session_state.active_view == 'chat':
    
    if st.button(T[lang]["back_home"], type="secondary"):
        st.session_state.active_view = 'home'
        st.rerun()

    st.markdown('<h2 style="font-family: \'Outfit\', sans-serif; color: #E9B44C; font-weight:800; font-size:2.2rem; margin-top: 1rem;">🤖 Gemini AI Advisory Assistant</h2>', unsafe_allow_html=True)
    
    caption_txt = "Ask anything about soil chemistry, pH effects, fertilizer dosages, or field management." if lang == "English" else "सोइल केमिस्ट्री, पीएच प्रभाव, उर्वरक मात्रा, या खेत प्रबंधन के बारे में कुछ भी पूछें।"
    st.markdown(f'<p style="color:#E2F0E7; font-size:1.15rem; margin-bottom:1.5rem;">{caption_txt}</p>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        avatar_icon = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar_icon):
            st.markdown(message["content"])

    chat_placeholder = "Enter your question (e.g., What is soil pH? How to fix pH 8.5?):" if lang == "English" else "सवाल दर्ज करें (उदाहरण: मिट्टी का पीएच क्या है?):"

    user_query = st.chat_input(chat_placeholder)
    
    if user_query:
        api_key = st.secrets.get("GEMINI_API_KEY")
        
        if not api_key:
            st.error("⚠️ Gemini API key missing from `.streamlit/secrets.toml`. Please add it to enable AI responses!")
        else:
            try:
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                
                client = genai.Client(api_key=api_key)
                
                prompt = f"""
                You are AgriBot, an expert Soil Science and Agricultural AI Assistant exclusively for AgriSoil AI.
                Target Language: {lang}.
                
                STRICT FORMATTING RULES:
                1. MATCH THE USER'S STYLE:
                   - If the user's prompt explicitly includes words like "in points", "bullets", or "steps", you MUST format your answer as a clear bulleted or numbered list.
                   - If the user's prompt includes words like "explain in detail" or "deep" WITHOUT mentioning points/bullets, you MUST format your answer as a cohesive, well-structured paragraph (NO bullet points).
                   - If it is a short question, give a short 1-2 sentence response.
                2. NO YELLOW OR LIGHT TEXT COLOR: Do not use yellow, gold, orange, or low-contrast highlights. All text must be neutral, dark, and high-contrast.
                3. DOMAIN LOCK: Only answer agriculture, soil health, and farming queries.
                
                User Query: {user_query}
                """
                
                response = client.models.generate_content(
                    model='gemini-3.1-flash-lite',
                    contents=prompt
                )
                
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})
                st.rerun()

            except Exception as e:
                st.error(f"AI Service Error: {str(e)}")

# ==========================================
# 💬 FLOATING YELLOW AI BUTTON
# ==========================================
if st.session_state.active_view != 'chat':
    st.markdown("""
        <a href="?view=chat" target="_self" class="floating-ai-btn">
            <span style="font-size: 1.2rem;">🤖</span>
            <span>AI Chat Assistant</span>
        </a>
    """, unsafe_allow_html=True)