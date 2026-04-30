import streamlit as st
from google import genai
import requests
import os
import json
import re
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import datetime

# ─── Load Environment ───────────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="COUTURE AI — Fashion Intelligence Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Premium CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root & Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 0.01em;
}
.main { background: #FAF7F4; color: #2C2420; }
.block-container { padding: 2.5rem 3rem 4rem; max-width: 1280px; }

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #FDF0E8 0%, #FAF7F4 45%, #EEF5F0 100%);
    border: 1px solid #E8DDD5;
    border-radius: 16px;
    padding: 3.5rem 3rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(196,154,100,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 40px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(120,180,150,0.09) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.25em;
    color: #B4845A;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.4rem;
    font-weight: 300;
    line-height: 1.1;
    color: #2C2420;
    margin: 0 0 0.5rem;
}
.hero-title em {
    font-style: italic;
    color: #B4845A;
}
.hero-sub {
    font-size: 0.95rem;
    color: #8A7A72;
    font-weight: 300;
    margin-top: 0.5rem;
}
.hero-badges {
    display: flex;
    gap: 0.5rem;
    margin-top: 1.5rem;
    flex-wrap: wrap;
}
.badge {
    display: inline-block;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 500;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    border: 1px solid;
}
.badge-gold { color: #A0723A; border-color: #DFC9A8; background: #FBF3E8; }
.badge-green { color: #4A9070; border-color: #B0D8C4; background: #EEF8F3; }
.badge-gray  { color: #7A6E68; border-color: #D8CFC8; background: #F5F2EF; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #F5EFE8;
    border-right: 1px solid #E0D4CA;
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem; }
.sidebar-logo {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.4rem;
    font-weight: 300;
    color: #A0723A;
    letter-spacing: 0.08em;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid #DDD0C4;
    margin-bottom: 1.5rem;
}
.sidebar-section {
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #A09088;
    font-weight: 500;
    margin: 1.2rem 0 0.6rem;
}

/* ── Inputs ── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
textarea {
    background: #FFFFFF !important;
    border: 1px solid #DDD0C4 !important;
    border-radius: 8px !important;
    color: #2C2420 !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-baseweb="select"] > div:hover,
div[data-baseweb="input"] > div:hover {
    border-color: #B4845A !important;
}
.stTextArea textarea { min-height: 100px; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #C9935A, #7AAF92) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    height: 2.8em !important;
    transition: all 0.25s ease;
    width: 100%;
    box-shadow: 0 2px 10px rgba(180,132,90,0.25);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #B47A45, #5A9A7A) !important;
    box-shadow: 0 4px 16px rgba(180,132,90,0.35) !important;
    transform: translateY(-1px);
}

/* ── Section Cards ── */
.section-card {
    background: #FFFFFF;
    border: 1px solid #EAE0D8;
    border-radius: 12px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 8px rgba(44,36,32,0.05);
}
.section-card-header {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.4rem;
    font-weight: 400;
    color: #A0723A;
    margin-bottom: 1rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid #EAE0D8;
    letter-spacing: 0.04em;
}

/* ── Tab Navigation ── */
.stTabs [data-baseweb="tab-list"] {
    background: #FAF7F4;
    border-bottom: 1px solid #E0D4CA;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #A09088 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
    padding: 0.8rem 1.5rem !important;
    border: none !important;
    transition: color 0.2s;
}
.stTabs [aria-selected="true"] {
    color: #A0723A !important;
    border-bottom: 2px solid #C9935A !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem; }

/* ── Expanders / History ── */
.streamlit-expanderHeader {
    background: #FBF7F3 !important;
    border: 1px solid #E8DDD5 !important;
    border-radius: 8px !important;
    color: #A0723A !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1rem !important;
}
.streamlit-expanderContent {
    background: #FFFFFF !important;
    border: 1px solid #E8DDD5 !important;
    border-top: none !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border: 1px solid #E8DDD5 !important;
    border-radius: 10px !important;
    padding: 1rem 1.2rem !important;
    box-shadow: 0 1px 6px rgba(44,36,32,0.06);
}
[data-testid="metric-container"] label {
    font-size: 0.65rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #A09088 !important;
    font-weight: 500 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.8rem !important;
    color: #A0723A !important;
    font-weight: 400 !important;
}

/* ── Alerts & Info ── */
.stAlert, .stInfo, .stWarning, .stSuccess, .stError {
    border-radius: 8px !important;
    border: 1px solid !important;
}
.stInfo { border-color: #B0D8C4 !important; background: #EEF8F3 !important; color: #3A8060 !important; }

/* ── Download Button ── */
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid #DDD0C4 !important;
    color: #8A7A72 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
.stDownloadButton > button:hover {
    border-color: #C9935A !important;
    color: #A0723A !important;
    background: #FBF3E8 !important;
}

/* ── Caption & Small Text ── */
.stCaption, small { color: #A09088 !important; font-size: 0.78rem !important; }

/* ── Divider ── */
hr { border-color: #E8DDD5 !important; margin: 2rem 0 !important; }

/* ── Radio ── */
.stRadio label { color: #7A6E68 !important; font-size: 0.85rem !important; }
.stRadio [data-checked="true"] label { color: #A0723A !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #C9935A !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #FAF7F4; }
::-webkit-scrollbar-thumb { background: #DDD0C4; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #B4845A; }
</style>
""", unsafe_allow_html=True)

# ─── Hero Banner ────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-eyebrow">✦ Powered by Gemini 2.5 & Stable Diffusion</div>
    <div class="hero-title">COUTURE <em>AI</em></div>
    <div class="hero-title" style="margin-top:-0.2rem; font-size:2rem; color:#8A8075;">Fashion Intelligence Studio</div>
    <div class="hero-sub">End-to-end AI pipeline for concept generation, trend forecasting, virtual styling & cost intelligence</div>
    <div class="hero-badges">
        <span class="badge badge-gold">✦ Generative Design</span>
        <span class="badge badge-green">◈ Trend Analytics</span>
        <span class="badge badge-gray">◉ Style Intelligence</span>
        <span class="badge badge-gold">✧ Cost Estimator</span>
        <span class="badge badge-green">⊕ Outfit Builder</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Utility Functions ───────────────────────────────────────────────────────
def extract_hex_colors(text):
    hex_colors = re.findall(r"#(?:[0-9a-fA-F]{6})", text)
    unique = list(dict.fromkeys(hex_colors))
    while len(unique) < 5:
        unique.append("#1A1A1A")
    return unique[:5]

HISTORY_FILE = "design_history.json"

def save_to_history(data):
    history = load_history()
    history.append(data)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)

def render_palette(palette):
    fig, ax = plt.subplots(figsize=(8, 1.8))
    fig.patch.set_facecolor('#FAF7F4')
    ax.set_facecolor('#FAF7F4')
    for i, color in enumerate(palette):
        rect = plt.Rectangle((i * 1.05, 0.15), 0.95, 0.7, color=color, linewidth=0)
        ax.add_patch(rect)
        ax.text(i * 1.05 + 0.475, 0.05, color, ha='center', va='top',
                fontsize=7, color='#8A7A72', fontfamily='monospace')
    ax.set_xlim(-0.05, 5.3)
    ax.set_ylim(-0.05, 1.0)
    ax.axis('off')
    plt.tight_layout(pad=0.2)
    return fig

def generate_image_hf(prompt):
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    return None

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">COUTURE AI ✦</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Core Settings</div>', unsafe_allow_html=True)
    category = st.selectbox("Category", ["Women", "Men", "Kids", "Unisex", "Accessories"])
    style = st.selectbox("Style", ["Casual", "Formal", "Streetwear", "Ethnic", "Party Wear", "Bridal Wear", "Avant-Garde", "Resort Wear"])
    style_vibe = st.selectbox("Aesthetic Vibe", ["Elegant", "Minimalist", "Luxury Glam", "Modern Chic", "Traditional", "Bold", "Dark Romantic", "Soft Feminine"])
    fabric = st.selectbox("Primary Fabric", ["Cotton", "Silk", "Denim", "Linen", "Velvet", "Chiffon", "Satin", "Organza", "Wool", "Leather"])
    season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter", "Resort"])
    occasion = st.selectbox("Occasion", ["Wedding", "Office", "Party", "Festival", "Casual Outing", "Red Carpet", "Beachwear", "Date Night", "Travel"])
    budget = st.selectbox("Budget Range", ["Under ₹2000", "₹2000–₹5000", "₹5000–₹10,000", "₹10,000–₹25,000", "₹25,000+", "Bespoke / No Limit"])

    st.markdown('<div class="sidebar-section">Advanced</div>', unsafe_allow_html=True)
    target_audience = st.selectbox("Target Audience", ["Gen Z (18–25)", "Millennials (26–40)", "Premium Adults (35–55)", "All Ages"])
    sustainability = st.checkbox("Sustainable / Eco-Conscious", value=False)
    generate_image = st.checkbox("Generate AI Visual (slower)", value=True)

    st.markdown("---")
    st.markdown('<p style="color:#C0B0A0; font-size:0.72rem; text-align:center; letter-spacing:0.1em;">COUTURE AI • v2.0</p>', unsafe_allow_html=True)

# ─── Main Tabs ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "✦  Design Studio",
    "◈  Trend Analyzer",
    "⊕  Outfit Builder",
    "✧  Cost Intelligence",
    "◉  History"
])

# ═══════════════════════════════════════════════════════
# TAB 1 — DESIGN STUDIO
# ═══════════════════════════════════════════════════════
with tab1:
    col_input, col_preview = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown('<div class="section-card-header">Design Brief</div>', unsafe_allow_html=True)
        description = st.text_area(
            "Describe your fashion concept",
            placeholder="e.g. A floor-length ivory silk gown with delicate gold thread embroidery at the neckline, structured bodice, and a dramatic cathedral train…",
            height=130,
            label_visibility="collapsed"
        )

        st.markdown("**Mood Keywords** *(optional)*")
        mood_col1, mood_col2, mood_col3 = st.columns(3)
        with mood_col1:
            moodA = st.text_input("Keyword 1", placeholder="ethereal", label_visibility="collapsed")
        with mood_col2:
            moodB = st.text_input("Keyword 2", placeholder="structured", label_visibility="collapsed")
        with mood_col3:
            moodC = st.text_input("Keyword 3", placeholder="timeless", label_visibility="collapsed")
        mood_keywords = ", ".join(filter(None, [moodA, moodB, moodC]))

        eco_note = " Prioritize sustainable, ethically sourced materials." if sustainability else ""

        gen_btn = st.button("✦  Generate Fashion Concept", key="gen_btn")

    with col_preview:
        st.markdown('<div class="section-card-header">Quick Reference</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Category", category)
        m2.metric("Style", style)
        m3.metric("Season", season)
        st.markdown("")
        m4, m5 = st.columns(2)
        m4.metric("Fabric", fabric)
        m5.metric("Occasion", occasion)
        st.caption(f"Budget: {budget}  •  Audience: {target_audience}{'  •  🌿 Eco Mode' if sustainability else ''}")

    if gen_btn:
        if not description.strip():
            st.warning("Please enter a design description.")
        else:
            with st.spinner("Generating fashion concept…"):
                prompt = f"""You are a world-class fashion designer and creative director.

Create a comprehensive fashion design document with the following parameters:

Category: {category}
Style: {style}
Aesthetic Vibe: {style_vibe}
Primary Fabric: {fabric}
Season: {season}
Occasion: {occasion}
Budget: {budget}
Target Audience: {target_audience}
Mood Keywords: {mood_keywords if mood_keywords else 'Not specified'}
Designer's Concept: {description}
{eco_note}

Structure your response with these sections:
## Design Overview
## Fabric & Material Details
## Silhouette, Fit & Construction
## Styling Guide & Lookbook Notes
## Accessories & Complementary Pieces
## Care Instructions
## Affordable Alternatives (same look, lower budget)
## Brand & Market Positioning
## Color Palette
Provide exactly 5 HEX color codes relevant to this design in the exact format:
#XXXXXX, #XXXXXX, #XXXXXX, #XXXXXX, #XXXXXX
Only valid 6-digit HEX codes, no other format.
"""
                response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                design_text = response.text

            st.markdown("---")
            st.markdown('<div class="section-card-header">✦ AI Fashion Concept</div>', unsafe_allow_html=True)
            st.write(design_text)

            # Color Palette
            st.markdown('<div class="section-card-header">Color Palette</div>', unsafe_allow_html=True)
            palette = extract_hex_colors(design_text)
            fig = render_palette(palette)
            st.pyplot(fig, use_container_width=True)

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button("↓ Download Concept (TXT)", design_text, file_name="couture_ai_design.txt", mime="text/plain")
            with col_dl2:
                st.download_button("↓ Download Brief (JSON)", json.dumps({
                    "description": description,
                    "category": category, "style": style,
                    "fabric": fabric, "season": season,
                    "occasion": occasion, "budget": budget,
                    "palette": palette, "date": str(datetime.datetime.now())
                }, indent=2), file_name="design_brief.json", mime="application/json")

            # Image Generation
            if generate_image:
                with st.spinner("Generating AI visual…"):
                    img_prompt = f"High fashion editorial photograph, {style_vibe.lower()} {style.lower()} {category.lower()} fashion, {fabric.lower()} fabric, {description[:150]}, studio lighting, Vogue magazine quality"
                    image = generate_image_hf(img_prompt)
                if image:
                    st.markdown('<div class="section-card-header">AI Visual</div>', unsafe_allow_html=True)
                    st.image(image, use_container_width=True)
                    buf = BytesIO()
                    image.save(buf, format="PNG")
                    st.download_button("↓ Download Visual (PNG)", buf.getvalue(), file_name="couture_visual.png", mime="image/png")
                else:
                    st.info("Visual generation unavailable — try again or check your HF API key.")

            # Save history
            save_to_history({
                "type": "design",
                "description": description,
                "category": category, "style": style,
                "style_vibe": style_vibe, "fabric": fabric,
                "season": season, "occasion": occasion,
                "budget": budget, "palette": palette,
                "date": str(datetime.datetime.now())
            })

# ═══════════════════════════════════════════════════════
# TAB 2 — TREND ANALYZER
# ═══════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-card-header">◈ Fashion Trend Intelligence</div>', unsafe_allow_html=True)

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        market_segment = st.selectbox("Market Segment", ["Luxury / Haute Couture", "Premium", "Contemporary", "Budget / Fast Fashion", "Sustainable / Slow Fashion", "Youth / Gen-Z"])
    with col_t2:
        region = st.selectbox("Region", ["Global", "India", "Europe", "USA", "East Asia", "Middle East"])
    with col_t3:
        horizon = st.selectbox("Forecast Horizon", ["Current Season", "Next Season", "12 Months", "3-Year Outlook"])

    trend_btn = st.button("◈  Run Trend Analysis", key="trend_btn")

    if trend_btn:
        with st.spinner("Analyzing global trend signals…"):
            trend_prompt = f"""You are a senior fashion trend forecaster at a top Paris consultancy.

Provide a detailed trend analysis report:

Season: {season}
Market Segment: {market_segment}
Category: {category}
Region: {region}
Occasion: {occasion}
Forecast Horizon: {horizon}

Structure your report:
## Executive Summary
## Macro Cultural Drivers
## Key Color Directions (name 6–8 colors with HEX codes in #XXXXXX format)
## Fabric & Texture Trends
## Silhouette Evolution
## Print & Pattern Forecast
## Consumer Behaviour Shifts
## Brands to Watch
## Commercial Opportunities
## Risk Factors
## 3 Actionable Recommendations for Designers
"""
            trend_response = client.models.generate_content(model="gemini-2.5-flash", contents=trend_prompt)
            trend_text = trend_response.text

        st.write(trend_text)

        # Trend Popularity Chart
        st.markdown('<div class="section-card-header">Predicted Trend Popularity Scores</div>', unsafe_allow_html=True)
        trends = ["Minimalist", "Neo-Vintage", "Sustainable", "Bold Maximalism", "Dark Romance", "Cottagecore", "Tech-Wear"]
        base_scores = [72, 85, 91, 78, 68, 63, 80]
        # Vary slightly based on market_segment
        if "Luxury" in market_segment:
            scores = [s + np.random.randint(-5, 10) for s in base_scores]
        elif "Youth" in market_segment:
            scores = [s + np.random.randint(-8, 12) for s in base_scores]
        else:
            scores = [s + np.random.randint(-6, 8) for s in base_scores]
        scores = [min(100, max(30, s)) for s in scores]

        fig2, ax2 = plt.subplots(figsize=(9, 3.5))
        fig2.patch.set_facecolor('#FAF7F4')
        ax2.set_facecolor('#FFFFFF')
        bar_colors = ['#C9935A' if s == max(scores) else '#E8DDD5' for s in scores]
        bars = ax2.barh(trends, scores, color=bar_colors, height=0.55, edgecolor='#DDD0C4', linewidth=0.5)
        for bar, score in zip(bars, scores):
            ax2.text(score + 1, bar.get_y() + bar.get_height() / 2,
                     f'{score}', va='center', ha='left', color='#8A7A72', fontsize=8)
        ax2.set_xlim(0, 115)
        ax2.set_xlabel('Popularity Score', color='#A09088', fontsize=8, labelpad=8)
        ax2.tick_params(colors='#A09088', labelsize=8)
        ax2.spines['bottom'].set_color('#DDD0C4')
        ax2.spines['left'].set_color('#DDD0C4')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

        st.download_button("↓ Download Trend Report", trend_text, file_name="trend_report.txt")

        save_to_history({"type": "trend", "market": market_segment, "region": region,
                         "season": season, "category": category, "date": str(datetime.datetime.now())})

# ═══════════════════════════════════════════════════════
# TAB 3 — OUTFIT BUILDER (NEW)
# ═══════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-card-header">⊕ Complete Outfit Builder</div>', unsafe_allow_html=True)
    st.caption("Build a full head-to-toe look for any scenario")

    col_ob1, col_ob2 = st.columns([1, 1], gap="large")
    with col_ob1:
        body_type = st.selectbox("Body Type", ["Hourglass", "Pear / Triangle", "Apple / Inverted Triangle", "Rectangle", "Petite", "Plus Size / Full Figure"])
        skin_tone = st.selectbox("Skin Tone", ["Fair / Porcelain", "Light / Ivory", "Medium / Olive", "Tan / Caramel", "Brown / Bronze", "Deep / Ebony"])
        persona = st.selectbox("Style Persona", ["The Power Professional", "The Free Spirit", "The Minimalist", "The Fashion Risk-Taker", "The Classic Elegante", "The Street Artist"])
    with col_ob2:
        event_context = st.text_area("Event/Context", placeholder="e.g. Investor pitch meeting in Mumbai in July, expecting 35°C heat…", height=100)
        wardrobe_items = st.text_area("Existing wardrobe pieces *(optional)*", placeholder="e.g. Navy blazer, white silk blouse, black trousers…", height=70)

    outfit_btn = st.button("⊕  Build Complete Outfit", key="outfit_btn")

    if outfit_btn:
        with st.spinner("Curating your look…"):
            outfit_prompt = f"""You are a personal stylist to A-list celebrities and executives.

Build a complete, styled outfit for:

Body Type: {body_type}
Skin Tone: {skin_tone}
Style Persona: {persona}
Category: {category}
Occasion: {occasion}
Season: {season}
Budget: {budget}
Event/Context: {event_context if event_context.strip() else 'General ' + occasion}
Existing Wardrobe: {wardrobe_items if wardrobe_items.strip() else 'Not specified'}

Include:
## The Complete Look (item by item: top, bottom/dress, outerwear, footwear, bag, accessories)
## Why This Works For Your Body Type & Skin Tone
## Where To Shop (mix of budget & premium options with estimated prices in INR)
## Styling Tips & Tricks
## 2 Alternative Variations of the Same Outfit
## What to Avoid
## Confidence Tips for This Look
"""
            outfit_response = client.models.generate_content(model="gemini-2.5-flash", contents=outfit_prompt)
            st.write(outfit_response.text)
            save_to_history({"type": "outfit", "persona": persona, "body_type": body_type,
                             "occasion": occasion, "date": str(datetime.datetime.now())})

# ═══════════════════════════════════════════════════════
# TAB 4 — COST INTELLIGENCE (NEW)
# ═══════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-card-header">✧ Cost & Production Intelligence</div>', unsafe_allow_html=True)
    st.caption("AI-powered pricing, production cost breakdown & market sizing")

    col_ci1, col_ci2 = st.columns([1, 1], gap="large")
    with col_ci1:
        design_desc_cost = st.text_area("Design Description for Costing", placeholder="Describe the garment you want costed…", height=100)
        production_qty = st.selectbox("Production Quantity", ["1 (Custom/Bespoke)", "10–50 (Small Batch)", "50–200 (Limited Run)", "200–1000 (Medium Scale)", "1000+ (Mass Production)"])
    with col_ci2:
        target_market = st.selectbox("Target Retail Market", ["Direct-to-Consumer Online", "Boutique Retail", "Department Store", "Export / International", "Luxury Multi-Brand"])
        production_loc = st.selectbox("Production Location", ["Mumbai / Delhi NCR", "Jaipur / Jodhpur (Handicraft)", "Surat / Ahmedabad (Textiles)", "Bangalore (Premium)", "Export Factory (Generic)"])

    cost_btn = st.button("✧  Analyze Costs & Pricing", key="cost_btn")

    if cost_btn:
        with st.spinner("Running cost analysis…"):
            cost_prompt = f"""You are a fashion business consultant and production manager with 20 years of experience in the Indian fashion industry.

Provide a detailed cost intelligence report for:

Design: {design_desc_cost if design_desc_cost.strip() else description if 'description' in dir() else 'A ' + style + ' ' + fabric + ' garment'}
Category: {category}
Fabric: {fabric}
Production Quantity: {production_qty}
Production Location: {production_loc}
Target Market: {target_market}
Budget Range: {budget}

Structure your report:
## Bill of Materials (BOM) — itemized fabric, lining, trims, hardware costs in INR
## Labour Cost Breakdown (cutting, stitching, finishing, embellishment)
## Overhead & Indirect Costs
## Total Cost of Production (per unit)
## Recommended Retail Pricing (3 tiers: value, standard, premium markup)
## Gross Margin Analysis
## Break-Even Units
## Competitor Pricing Benchmarks
## Cost Optimization Suggestions
## Market Opportunity (potential revenue at target scale)
"""
            cost_response = client.models.generate_content(model="gemini-2.5-flash", contents=cost_prompt)
            cost_text = cost_response.text

        st.write(cost_text)

        # Simple cost breakdown chart
        st.markdown('<div class="section-card-header">Illustrative Cost Breakdown</div>', unsafe_allow_html=True)
        labels = ['Fabric & Materials', 'Labour', 'Overheads', 'Finishing & Trims', 'Packaging']
        sizes = [42, 28, 12, 10, 8]
        colors_pie = ['#C9935A', '#E8B080', '#A0C8B0', '#DDD0C4', '#F5EFE8']
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        fig3.patch.set_facecolor('#FAF7F4')
        ax3.set_facecolor('#FAF7F4')
        wedges, texts, autotexts = ax3.pie(
            sizes, labels=labels, colors=colors_pie,
            autopct='%1.0f%%', startangle=140,
            textprops={'color': '#6A5A52', 'fontsize': 8},
            wedgeprops={'edgecolor': '#FAF7F4', 'linewidth': 2}
        )
        for at in autotexts:
            at.set_color('#2C2420')
            at.set_fontsize(8)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        st.caption("Indicative split — actual values from the AI analysis above supersede this chart.")

        st.download_button("↓ Download Cost Report", cost_text, file_name="cost_intelligence.txt")
        save_to_history({"type": "cost", "qty": production_qty, "location": production_loc,
                         "fabric": fabric, "date": str(datetime.datetime.now())})

# ═══════════════════════════════════════════════════════
# TAB 5 — HISTORY
# ═══════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-card-header">◉ Session Archive</div>', unsafe_allow_html=True)
    history = load_history()
    if history:
        total = len(history)
        designs = sum(1 for h in history if h.get("type") == "design")
        trends = sum(1 for h in history if h.get("type") == "trend")
        outfits = sum(1 for h in history if h.get("type") == "outfit")
        costs = sum(1 for h in history if h.get("type") == "cost")

        hm1, hm2, hm3, hm4 = st.columns(4)
        hm1.metric("Total Generations", total)
        hm2.metric("Design Concepts", designs)
        hm3.metric("Trend Reports", trends)
        hm4.metric("Outfit Builds + Cost", outfits + costs)

        st.markdown("---")
        if st.button("🗑  Clear All History"):
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.rerun()

        for item in reversed(history[-10:]):
            icon = {"design": "✦", "trend": "◈", "outfit": "⊕", "cost": "✧"}.get(item.get("type", ""), "◉")
            label = item.get("description", item.get("market", item.get("persona", item.get("qty", "Entry"))))
            with st.expander(f"{icon} {label[:60]}  ·  {item.get('date', '')[:16]}"):
                for k, v in item.items():
                    if k not in ("date", "type", "palette"):
                        st.write(f"**{k.replace('_',' ').title()}:** {v}")
                if "palette" in item and item["palette"]:
                    fig_h = render_palette(item["palette"])
                    st.pyplot(fig_h, use_container_width=True)
    else:
        st.info("No history yet. Start generating in the Design Studio, Trend Analyzer, Outfit Builder or Cost Intelligence tabs.")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding: 1rem 0 0.5rem;">
    <p style="font-family:'Cormorant Garamond',serif; font-size:1.1rem; color:#C9B8A8; font-weight:300; letter-spacing:0.1em;">
        COUTURE AI ✦ Fashion Intelligence Studio
    </p>
    <p style="font-size:0.7rem; color:#C0B4AC; letter-spacing:0.15em; text-transform:uppercase; margin-top:0.3rem;">
        Built by Samruddhi Mahale &nbsp;·&nbsp; Gemini 2.5 Flash + Stable Diffusion 2 &nbsp;·&nbsp; v2.0
    </p>
</div>
""", unsafe_allow_html=True)