import streamlit as st
import numpy as np
import re
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="Brand & Product Intelligence",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stMetric"] {
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.08);
}
.big-title { font-size: 28px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

NEGATION_TRIGGERS = {
    "not","no","never","nothing","nobody","nowhere","neither","nor",
    "hardly","barely","scarcely","isnt","wasnt","shouldnt","wouldnt",
    "couldnt","doesnt","didnt","dont","cant","wont","aint",
    "without","lacking","fails","failed","lack"
}
CLAUSE_BREAKS = {
    "but","however","although","though","yet","so","because",
    "if","while","and","or","then","still","despite","except"
}
INTENSIFIERS = {
    "very","so","really","extremely","absolutely","completely",
    "totally","utterly","quite","super","pretty","rather","highly",
    "truly","deeply"
}
NEGATION_MAP = {
    "great":"terrible","amazing":"awful","excellent":"poor",
    "fantastic":"horrible","wonderful":"dreadful","love":"hate",
    "best":"worst","perfect":"terrible","awesome":"awful",
    "brilliant":"stupid","good":"bad","nice":"unpleasant","fine":"poor",
    "happy":"unhappy","glad":"sad","pleased":"displeased","fun":"boring",
    "enjoy":"dislike","like":"dislike","beautiful":"ugly",
    "easy":"difficult","fast":"slow","helpful":"unhelpful",
    "useful":"useless","safe":"unsafe","clean":"dirty",
    "fresh":"stale","smart":"stupid","kind":"unkind",
    "bad":"good","terrible":"great","awful":"good","horrible":"pleasant",
    "boring":"interesting","hate":"love","worst":"best","ugly":"beautiful",
    "stupid":"smart","useless":"useful","difficult":"easy","slow":"fast",
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
                result.append(NEGATION_MAP[tok])
                negate = False; neg_count = 0
            else:
                result.append(tok + "_NEG"); neg_count += 1
                if neg_count >= MAX_SCOPE:
                    negate = False; neg_count = 0
        else:
            negate = False; result.append(tok)
        i += 1
    return result

def clean_text_ml(text):
    text = str(text).lower()
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"[^a-zA-Z\s_]", "", text)
    tokens = text.split()
    tokens = mark_negation(tokens)
    result = " ".join(tokens).strip()
    return result if result else "unknown"

vader_analyzer = SentimentIntensityAnalyzer()

def get_hybrid_sentiment(text, model, tfidf, source_type="general"):
    raw_text = str(text)
    cleaned  = clean_text_ml(raw_text)
    v_scores = vader_analyzer.polarity_scores(raw_text)
    v_score  = v_scores["compound"]
    ml_score = 0.0
    confidence = 0.0
    if model is not None:
        try:
            vec      = tfidf.transform([cleaned])
            vader_f  = csr_matrix(np.array([[v_scores["pos"], v_scores["neg"], v_score]]))
            features = hstack([vec, vader_f])
            ml_prob  = model.predict_proba(features)[0]
            ml_score   = float(ml_prob[1] - ml_prob[0])
            confidence = float(abs(ml_prob[1] - ml_prob[0]))
        except Exception:
            ml_score = 0.0; confidence = 0.0
    if confidence < 0.3:
        wv, wm = 0.8, 0.2
    elif source_type == "news":
        wv, wm = 0.6, 0.4
    elif source_type == "social":
        wv, wm = 0.5, 0.5
    elif source_type == "review":
        wv, wm = 0.2, 0.8
    else:
        wv, wm = 0.35, 0.65
    conflict   = (ml_score * v_score) < 0
    short_text = len(raw_text.split()) <= 4
    if conflict and short_text and abs(v_score) > 0.2:
        final_score = (v_score * 0.85) + (ml_score * 0.15)
    else:
        final_score = (v_score * wv) + (ml_score * wm)
    if final_score >= 0.15:
        label = "Positive"
    elif final_score <= -0.15:
        label = "Negative"
    else:
        label = "Neutral"
    return label, round(confidence * 100, 1), cleaned, round(final_score, 4)

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model = joblib.load("sentiment_model_final.pkl")
    tfidf = joblib.load("tfidf_vectorizer_final.pkl")
    return model, tfidf

model, tfidf = load_model()

st.sidebar.title("🧠 Intelligence Engine")
st.sidebar.divider()
option = st.sidebar.selectbox(
    "Select Module",
    ["Home", "Live Brand Monitor", "Product Intelligence", "Model Performance", "About"]
)
st.sidebar.divider()
st.sidebar.caption("BCA Capstone | DIT University | CAN304")
st.sidebar.caption("Model: Hybrid LR + TF-IDF + VADER")
st.sidebar.caption("Accuracy: 80.20%")

# HOME
if option == "Home":
    st.markdown('<p class="big-title">🧠 AI-Driven Brand & Product Intelligence</p>', unsafe_allow_html=True)
    st.caption("Hybrid sentiment system — Logistic Regression (Sentiment140, 100k) combined with VADER lexicon. Source-aware weighting, negation handling, live brand monitoring.")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model",    "Hybrid LR + TF-IDF + VADER")
    c2.metric("Accuracy", "80.20%")
    c3.metric("Dataset",  "Sentiment140")
    c4.metric("Samples",  "100,000")
    st.divider()

    st.subheader("📝 Analyze Text")
    user_input  = st.text_area("Enter text (single or multi-line):", height=130,
                                placeholder="Paste a review, tweet, feedback, or multiple lines here...")
    source_type = st.selectbox("Text Source", ["general", "review", "social", "news"],
                                help="review=ML heavy | news=VADER heavy | social=balanced | general=default")

    if st.button("Analyze Sentiment", type="primary"):
        if user_input.strip():
            lines = [l.strip() for l in user_input.strip().split("\n") if l.strip()]
            if len(lines) == 1:
                label, conf, cleaned, score = get_hybrid_sentiment(lines[0], model, tfidf, source_type)
                color = "#28a745" if label=="Positive" else ("#dc3545" if label=="Negative" else "#6c757d")
                emoji = "🟢" if label=="Positive" else ("🔴" if label=="Negative" else "⚪")
                st.divider()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Sentiment",   f"{emoji} {label}")
                c2.metric("Confidence",  f"{conf}%")
                c3.metric("Final Score", f"{score:+.4f}")
                c4.metric("Word Count",  len(lines[0].split()))
                st.markdown("**Confidence Level**")
                st.markdown(
                    f'<div style="background:#e9ecef;border-radius:10px;height:22px;width:100%">'
                    f'<div style="background:{color};width:{min(conf,100)}%;height:22px;border-radius:10px;'
                    f'display:flex;align-items:center;justify-content:center;color:white;font-size:12px;'
                    f'font-weight:bold">{conf}%</div></div>',
                    unsafe_allow_html=True
                )
                with st.expander("🔍 Preprocessing Details"):
                    st.write(f"**Original:** {lines[0]}")
                    st.write(f"**Cleaned:**  {cleaned}")
                    neg_words = [w for w in cleaned.split() if w in NEGATION_MAP.values() or w.endswith("_NEG")]
                    st.write(f"**Negation applied:** {', '.join(neg_words) if neg_words else 'none'}")
            else:
                st.divider()
                results = []
                for line in lines:
                    lbl, cf, cl, sc = get_hybrid_sentiment(line, model, tfidf, source_type)
                    results.append({
                        "Text":       line,
                        "Sentiment":  f"{'🟢' if lbl=='Positive' else ('🔴' if lbl=='Negative' else '⚪')} {lbl}",
                        "Confidence": f"{cf}%",
                        "Score":      f"{sc:+.4f}"
                    })
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
                pos = sum(1 for r in results if "Positive" in r["Sentiment"])
                neg = sum(1 for r in results if "Negative" in r["Sentiment"])
                neu = len(results) - pos - neg
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total", len(results))
                c2.metric("Positive 🟢", pos)
                c3.metric("Negative 🔴", neg)
                c4.metric("Neutral ⚪",  neu)
        else:
            st.warning("Please enter some text!")

    st.divider()
    st.markdown("#### Source Type — When to use which")
    st.dataframe(pd.DataFrame({
        "Source Type":  ["review", "social", "news", "general"],
        "ML Weight":    ["80%", "50%", "40%", "65%"],
        "VADER Weight": ["20%", "50%", "60%", "35%"],
        "Best For":     ["Amazon/product reviews", "Tweets, Reddit, Instagram", "News headlines", "Mixed/unknown source"]
    }), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Negation Handling — Key Innovation")
    st.table(pd.DataFrame({
        "Phrase":            ["not bad at all", "nothing great about this", "not very good", "nothing good came from this", "can not complain really"],
        "Without Negation":  ["❌ Negative", "❌ Positive", "❌ Positive", "❌ Positive", "❌ Negative"],
        "Our Model":         ["✅ Positive", "✅ Negative", "✅ Negative", "✅ Negative", "✅ Positive"]
    }))

# LIVE BRAND MONITOR
elif option == "Live Brand Monitor":
    st.title("🌐 Live Brand Market Monitor")
    st.caption("Real-time news sentiment — search any brand or product.")
    query  = st.text_input("Brand / Product Name", placeholder="e.g. Apple, Tesla, Samsung")
    period = st.selectbox("Time Period", ["1d", "7d", "1m"], index=1)

    if st.button("Generate Report", type="primary"):
        if not query.strip():
            st.warning("Enter a brand or product name!")
        else:
            try:
                from GoogleNews import GoogleNews
                with st.spinner(f"Fetching news for '{query}'..."):
                    gn = GoogleNews(lang="en", region="IN", period=period)
                    gn.clear()
                    gn.search(query)
                    results = gn.result()
                if not results:
                    st.warning("No news found. Try different query or time period.")
                else:
                    data = []
                    for art in results:
                        if not art.get("title"):
                            continue
                        link = art["link"]
                        if "&ved=" in link:
                            link = link.split("&ved=")[0]
                        if link.startswith("./"):
                            link = "https://news.google.com/" + link[2:]
                        lbl, cf, _, sc = get_hybrid_sentiment(art["title"], model, tfidf, "news")
                        data.append({
                            "Headline":   art["title"],
                            "Sentiment":  f"{'🟢' if lbl=='Positive' else ('🔴' if lbl=='Negative' else '⚪')} {lbl}",
                            "Confidence": f"{cf}%",
                            "Date":       art.get("date", ""),
                            "Link":       link
                        })
                    df_news = pd.DataFrame(data)
                    pos = sum(1 for d in data if "Positive" in d["Sentiment"])
                    neg = sum(1 for d in data if "Negative" in d["Sentiment"])
                    neu = len(data) - pos - neg
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total", len(data))
                    c2.metric("Positive 🟢", pos)
                    c3.metric("Negative 🔴", neg)
                    c4.metric("Neutral ⚪",  neu)
                    try:
                        import plotly.express as px
                        counts_df = pd.DataFrame({
                            "Sentiment": ["Positive","Negative","Neutral"],
                            "Count":     [pos, neg, neu]
                        })
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            fig = px.pie(counts_df, names="Sentiment", values="Count",
                                         color="Sentiment",
                                         color_discrete_map={"Positive":"#28a745","Negative":"#dc3545","Neutral":"#6c757d"},
                                         title=f"Sentiment — {query}")
                            st.plotly_chart(fig, use_container_width=True)
                        with col2:
                            st.dataframe(df_news,
                                         column_config={"Link": st.column_config.LinkColumn("Open Article")},
                                         hide_index=True, use_container_width=True)
                    except Exception:
                        st.dataframe(df_news, hide_index=True, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

# PRODUCT INTELLIGENCE
elif option == "Product Intelligence":
    st.title("📦 Product Intelligence")
    st.caption("Upload any CSV of reviews for bulk sentiment analysis.")
    st.info("Supports Amazon, Flipkart, or any custom review CSV.")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df_rev   = pd.read_csv(file)
        st.write(f"Loaded {len(df_rev):,} rows")
        text_col = st.selectbox("Select Review/Text Column", df_rev.columns)
        src_type = st.selectbox("Source Type", ["review", "general", "social", "news"])
        limit    = st.slider("Max rows to analyze", 50, min(500, len(df_rev)), 100)
        st.dataframe(df_rev[[text_col]].head(5), use_container_width=True)
        if st.button("Run Analysis", type="primary"):
            with st.spinner("Analyzing..."):
                df_sample = df_rev[[text_col]].dropna().head(limit).copy()
                df_sample["Sentiment"] = df_sample[text_col].apply(
                    lambda x: get_hybrid_sentiment(str(x), model, tfidf, src_type)[0]
                )
                counts = df_sample["Sentiment"].value_counts()
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Positive 🟢", counts.get("Positive", 0))
            c2.metric("Negative 🔴", counts.get("Negative", 0))
            c3.metric("Neutral ⚪",  counts.get("Neutral",  0))
            try:
                import plotly.express as px
                col1, col2 = st.columns(2)
                with col1:
                    fig1 = px.pie(counts.reset_index(), names="Sentiment", values="count",
                                  color="Sentiment",
                                  color_discrete_map={"Positive":"#28a745","Negative":"#dc3545","Neutral":"#6c757d"},
                                  title="Sentiment Distribution")
                    st.plotly_chart(fig1, use_container_width=True)
                with col2:
                    fig2 = px.bar(counts.reset_index(), x="Sentiment", y="count",
                                  color="Sentiment",
                                  color_discrete_map={"Positive":"#28a745","Negative":"#dc3545","Neutral":"#6c757d"},
                                  title="Count by Sentiment")
                    st.plotly_chart(fig2, use_container_width=True)
            except Exception:
                st.bar_chart(counts)
            st.dataframe(df_sample[[text_col, "Sentiment"]], use_container_width=True, hide_index=True)
            csv_out = df_sample[[text_col, "Sentiment"]].to_csv(index=False)
            st.download_button("Download Results CSV", csv_out, "sentiment_results.csv", "text/csv")

# MODEL PERFORMANCE
elif option == "Model Performance":
    st.title("📊 Model Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best Accuracy",      "80.20%")
    m2.metric("vs VADER baseline",  "+12.89%")
    m3.metric("Precision / Recall", "0.80 / 0.80")
    m4.metric("F1-Score",           "0.80")
    st.divider()
    model_data = pd.DataFrame({
        "Model":    ["VADER","NB + BoW","LR + BoW","NB + TF-IDF","LR + TF-IDF","Hybrid (Ours)"],
        "Accuracy": [67.31,   77.45,     77.85,      78.83,        79.56,         80.20],
        "Type":     ["Lexicon","ML","ML","ML","ML","Hybrid"]
    })
    try:
        import plotly.express as px
        fig = px.bar(model_data, x="Model", y="Accuracy", color="Type", text="Accuracy",
                     color_discrete_map={"Lexicon":"#6c757d","ML":"#0d6efd","Hybrid":"#28a745"},
                     title="Model Accuracy Comparison — Sentiment140 (100k sample)")
        fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="80% target")
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(yaxis_range=[60, 86], showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.bar_chart(model_data.set_index("Model")["Accuracy"])
    st.divider()
    model_data["Status"] = model_data["Accuracy"].apply(lambda x: "✅ Best" if x == 80.20 else "")
    st.dataframe(model_data[["Model","Accuracy","Type","Status"]].sort_values("Accuracy", ascending=False),
                 use_container_width=True, hide_index=True)
    st.divider()
    st.subheader("Classification Report — Hybrid Model")
    st.dataframe(pd.DataFrame({
        "Class":     ["Negative","Positive","Macro Avg"],
        "Precision": [0.80, 0.80, 0.80],
        "Recall":    [0.80, 0.81, 0.80],
        "F1-Score":  [0.80, 0.80, 0.80],
        "Support":   [9989, 10011, 20000],
    }), use_container_width=True, hide_index=True)
    st.divider()
    st.subheader("Why VADER alone scores 67.31%?")
    st.markdown("VADER is a rule-based lexicon with no training phase. It cannot learn dataset-specific patterns. Included as baseline to show supervised ML outperforms lexicon methods. In our hybrid, VADER contributes as extra signal alongside TF-IDF — not as primary classifier.")

# ABOUT
elif option == "About":
    st.title("ℹ️ About This Project")
    st.markdown("""
### AI-Driven Sentiment Analysis for Brand & Product Intelligence
**Course:** CAN304 | **University:** DIT University | **Mode:** A

---
### Team
| Name | SAP ID |
|------|--------|
| Ashish Pal | 1000021731 |
| Shaurya Pundir | 1000021913 |
| Abhishek | 1000022005 |

**Faculty Advisor:** Riya Dhama

---
### Tech Stack
| Component | Technology |
|-----------|-----------|
| Language | Python 3 |
| ML Library | Scikit-learn |
| NLP | TF-IDF, VADER, Negation Handling |
| Dataset | Sentiment140 (100k balanced sample) |
| Dashboard | Streamlit |
| Deployment | Streamlit Community Cloud |
| Version Control | GitHub |

---
### Key Innovation — 3 Layer Negation Handling
**Layer 1 — Antonym Replacement:** not good → bad | nothing great → terrible

**Layer 2 — VADER as Extra Features:** pos/neg/compound scores appended to TF-IDF matrix

**Layer 3 — Intensifier Skipping + Conflict Resolution:** not very good → bad

---
### GitHub
[github.com/Ashishpal0219/sentiment-analysis-capstone](https://github.com/Ashishpal0219/sentiment-analysis-capstone)
    """)
