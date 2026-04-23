import streamlit as st
import numpy as np
import re
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI-Driven Sentiment Intelligence",
    page_icon="🧠",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
[data-testid="stMetric"] {
    background: rgba(0,0,0,0.03);
    padding: 16px 20px;
    border-radius: 12px;
    border: 1px solid rgba(0,0,0,0.06);
}
[data-testid="stMetricLabel"] { font-size: 13px; color: #666; }
[data-testid="stMetricValue"] { font-size: 24px; font-weight: 700; color: #2E7D32; }
.big-title { font-size: 32px; font-weight: 800; line-height: 1.2; margin-bottom: 2px; color: #1E1E1E; }
.sub-title { font-size: 15px; color: #666; margin-bottom: 20px; }
.section-header { font-size: 20px; font-weight: 700; margin: 20px 0 10px 0; border-left: 5px solid #2E7D32; padding-left: 10px; }
div[data-testid="stSidebar"] { background: #0f1117; }
div[data-testid="stSidebar"] * { color: #fafafa !important; }
.stButton > button { border-radius: 8px; font-weight: 600; padding: 10px 24px; }
</style>
""", unsafe_allow_html=True)

# --- PREPROCESSING LOGIC ---
NEGATION_TRIGGERS = {"not","no","never","nothing","nobody","nowhere","neither","nor","isnt","wasnt","shouldnt","wouldnt","couldnt","doesnt","didnt","dont","cant","wont","aint","without","lacking","fails","failed","lack"}
CLAUSE_BREAKS = {"but","however","although","though","yet","so","because","if","while","and","or","then","still","despite","except"}
INTENSIFIERS = {"very","so","really","extremely","absolutely","completely","totally","utterly","quite","super","pretty","rather","highly","truly","deeply"}
NEGATION_MAP = {
    "great":"terrible","amazing":"awful","excellent":"poor","love":"hate","best":"worst","perfect":"terrible","good":"bad","nice":"unpleasant",
    "happy":"unhappy","enjoy":"dislike","like":"dislike","beautiful":"ugly","easy":"difficult","fast":"slow","helpful":"unhelpful","smart":"stupid",
    "bad":"good","terrible":"great","awful":"good","horrible":"pleasant","boring":"interesting","hate":"love","worst":"best","ugly":"beautiful"
}

def mark_negation(tokens):
    result, negate, i, neg_count = [], False, 0, 0
    MAX_SCOPE = 4
    while i < len(tokens):
        tok = tokens[i]
        if tok in NEGATION_TRIGGERS:
            negate = True; neg_count = 0; i += 1; continue
        elif tok in INTENSIFIERS and negate:
            i += 1; continue
        elif tok in CLAUSE_BREAKS or tok in {".", ",", "!", "?", ";", ":"}:
            negate = False; neg_count = 0; result.append(tok)
        elif negate and neg_count < MAX_SCOPE:
            if tok in NEGATION_MAP:
                result.append(NEGATION_MAP[tok]); negate = False; neg_count = 0
            else:
                result.append(tok + "_NEG"); neg_count += 1
        else:
            negate = False; result.append(tok)
        i += 1
    return result

def clean_text_ml(text):
    text = str(text).lower()
    text = re.sub(r"@\w+|http\S+|www\S+", "", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"[^a-zA-Z\s_]", "", text)
    tokens = text.split()
    tokens = mark_negation(tokens)
    return " ".join(tokens).strip() or "unknown"

def clean_text_simple(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|@\w+|[^\w\s]", "", text)
    return text.strip()

vader_analyzer = SentimentIntensityAnalyzer()

def get_hybrid_sentiment(text, model, tfidf, source_type="general"):
    raw_text = str(text)
    cleaned = clean_text_simple(raw_text) if source_type in ["news", "social"] else clean_text_ml(raw_text)
    
    v_scores = vader_analyzer.polarity_scores(raw_text)
    v_score = v_scores["compound"]
    
    try:
        vec = tfidf.transform([cleaned])
        vader_f = csr_matrix(np.array([[v_scores["pos"], v_scores["neg"], v_score]]))
        features = hstack([vec, vader_f])
        ml_prob = model.predict_proba(features)[0]
        ml_score = float(ml_prob[1] - ml_prob[0])
        confidence = float(abs(ml_prob[1] - ml_prob[0]))
    except:
        ml_score, confidence = 0.0, 0.0

    # Dynamic Weighting Logic
    if source_type == "review": wv, wm = 0.2, 0.8
    elif source_type in ["news", "social"]: wv, wm = 0.6, 0.4
    else: wv, wm = 0.3, 0.7

    final_score = (v_score * wv) + (ml_score * wm)
    label = "Positive" if final_score >= 0.1 else ("Negative" if final_score <= -0.1 else "Neutral")
    
    return label, round(confidence * 100, 2), cleaned, round(final_score, 4)

@st.cache_resource(show_spinner="Waking up the brain...")
def load_assets():
    model = joblib.load("sentiment_model_final.pkl")
    tfidf = joblib.load("tfidf_vectorizer_final.pkl")
    return model, tfidf

model, tfidf = load_assets()

# --- SIDEBAR ---
st.sidebar.markdown("## 🧠 Intelligence Engine")
st.sidebar.divider()
option = st.sidebar.selectbox("Navigate Module", ["Home", "Live Monitor", "Bulk Analysis", "Performance Metrics", "About"])
st.sidebar.divider()
st.sidebar.caption("Champion: **Hybrid Hybrid (Trigrams)**")
st.sidebar.caption("Best Accuracy: **80.55%**")
st.sidebar.divider()
st.sidebar.info("BCA Capstone | CAN304\nDIT University")

# --- HOME ---
if option == "Home":
    st.markdown('<p class="big-title">🧠 AI-Driven Sentiment Intelligence</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Brand & Product Intelligence System | 80.55% Accuracy Champion Model</p>', unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", "Hybrid Trigrams")
    c2.metric("Accuracy", "80.55%", "+13.24% vs Base")
    c3.metric("Samples", "100,000")
    c4.metric("Engine", "ML + VADER")

    st.markdown('<p class="section-header">📝 Analyze Individual Sentiment</p>', unsafe_allow_html=True)
    user_input = st.text_area("Enter your text below:", height=100, placeholder="e.g., The battery life is not bad, but the screen is amazing!")
    source_type = st.radio("Source Context:", ["general", "review", "news", "social"], horizontal=True)
    
    if st.button("Predict Sentiment", type="primary"):
        if user_input.strip():
            label, conf, cleaned, score = get_hybrid_sentiment(user_input, model, tfidf, source_type)
            emoji = "🟢" if label=="Positive" else ("🔴" if label=="Negative" else "⚪")
            
            st.divider()
            res_c1, res_c2, res_c3 = st.columns(3)
            res_c1.metric("Result", f"{emoji} {label}")
            res_c2.metric("Confidence", f"{conf}%")
            res_c3.metric("Hybrid Score", f"{score}")
            
            with st.expander("🔍 Deep Trace (Preprocessing)"):
                st.write(f"**Cleaned Text:** `{cleaned}`")
                st.caption("The model used Trigrams (3-word patterns) to handle the negations in this sentence.")
        else:
            st.warning("Please enter text first!")

# --- LIVE MONITOR ---
elif option == "Live Monitor":
    st.markdown('<p class="big-title">🌐 Live Brand Market Monitor</p>', unsafe_allow_html=True)
    query = st.text_input("Enter Brand/Product Name:", placeholder="e.g. Tesla, iPhone 15, Zomato")
    
    if st.button("Fetch & Analyze Headlines", type="primary"):
        try:
            from gnews import GNews
            gn = GNews(language="en", max_results=15)
            articles = gn.get_news(query)
            if articles:
                news_data = []
                for art in articles:
                    lbl, cf, _, _ = get_hybrid_sentiment(art['title'], model, tfidf, "news")
                    news_data.append({"Headline": art['title'], "Sentiment": lbl, "Confidence": f"{cf}%", "Link": art['url']})
                st.dataframe(pd.DataFrame(news_data), use_container_width=True)
            else:
                st.info("No recent news found for this query.")
        except:
            st.error("GNews library not found. Please install via requirements.txt")

# --- PERFORMANCE ---
elif option == "Performance Metrics":
    st.markdown('<p class="section-header">📊 Model Comparison Leaderboard</p>', unsafe_allow_html=True)
    perf_df = pd.DataFrame({
        "Model Architecture": ["Hybrid Champion (Trigrams)", "Logistic Regression + TF-IDF", "BiLSTM (Deep Learning)", "Naive Bayes + TF-IDF", "TextCNN (Deep Learning)", "VADER (Baseline)"],
        "Accuracy Score": [80.55, 79.57, 78.92, 78.83, 77.16, 67.31],
        "Selection": ["✅ Deployment Choice", "Solo ML Leader", "Experimental", "Baseline ML", "Experimental", "Rule-based"]
    })
    st.table(perf_df)
    st.success("Our Hybrid Champion (80.55%) outperformed Deep Learning (BiLSTM) by ~1.6% due to better generalization on this dataset size.")

# --- ABOUT ---
elif option == "About":
    st.markdown("### BCA Capstone Project")
    st.write("**Team:** Ashish Pal (1000021731), Shaurya Pundir, Abhishek")
    st.write("**Advisor:** Riya Dhama | DIT University")
    st.info("This project demonstrates that carefully tuned Machine Learning with Lexicon features (Hybrid) can often outperform complex Deep Learning on medium-sized datasets.")

else:
    st.write("Module under construction.")
