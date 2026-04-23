import streamlit as st
import numpy as np
import re
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="AI-Driven Sentiment Analysis",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stMetric"] {
    background: rgba(0,0,0,0.03);
    padding: 16px 20px;
    border-radius: 12px;
    border: 1px solid rgba(0,0,0,0.06);
}
[data-testid="stMetricLabel"] { font-size: 13px; color: #666; }
[data-testid="stMetricValue"] { font-size: 24px; font-weight: 700; }
.big-title { font-size: 30px; font-weight: 800; line-height: 1.3; margin-bottom: 4px; }
.sub-title { font-size: 14px; color: #888; margin-bottom: 0; }
.section-header { font-size: 18px; font-weight: 700; margin: 8px 0 4px 0; }
div[data-testid="stSidebar"] { background: #0f1117; }
div[data-testid="stSidebar"] * { color: #fafafa !important; }
div[data-testid="stSidebar"] hr { border-color: #333 !important; }
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    padding: 8px 24px;
}
.stSelectbox > div > div { border-radius: 8px; }
.stTextArea > div > div { border-radius: 8px; }
.stTextInput > div > div { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Preprocessing ──
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
    """Negation-aware cleaning for reviews/general."""
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

def clean_text_simple(text):
    """Simple cleaning for news/social — no negation mapping."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+|\#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

vader_analyzer = SentimentIntensityAnalyzer()

def get_hybrid_sentiment(text, model, tfidf, source_type="general"):
    raw_text = str(text)

    if source_type in ["news", "social"]:
        cleaned = clean_text_simple(raw_text)
    else:
        cleaned = clean_text_ml(raw_text)

    v_scores = vader_analyzer.polarity_scores(raw_text)
    v_score  = v_scores["compound"]
    ml_score = 0.0
    confidence = 0.0

    if model is not None:
        try:
            vec      = tfidf.transform([cleaned])
            vader_f  = csr_matrix(np.array([[
                v_scores["pos"], v_scores["neg"], v_score
            ]]))
            features = hstack([vec, vader_f])
            ml_prob  = model.predict_proba(features)[0]
            ml_score   = float(ml_prob[1] - ml_prob[0])
            confidence = float(abs(ml_prob[1] - ml_prob[0]))
        except Exception:
            ml_score = 0.0; confidence = 0.0

    if confidence < 0.3:
        wv, wm = 0.8, 0.2
    elif source_type in ["news", "social"]:
        wv, wm = 0.6, 0.4
    elif source_type == "review":
        wv, wm = 0.2, 0.8
    else:
        wv, wm = 0.35, 0.65

    conflict   = (ml_score * v_score) < 0
    short_text = len(raw_text.split()) <= 3
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

# ── SIDEBAR ──
st.sidebar.markdown("## 🧠 Intelligence Engine")
st.sidebar.divider()
option = st.sidebar.selectbox(
    "Select Module",
    ["Home", "Live Brand Monitor", "Product Intelligence", "Model Performance", "About"]
)
st.sidebar.divider()
st.sidebar.markdown("**Project**")
st.sidebar.caption("AI-Driven Sentiment Analysis for\nBrand & Product Intelligence")
st.sidebar.divider()
st.sidebar.markdown("**Model**")
st.sidebar.caption("Hybrid (Logistic Regression + TF-IDF + VADER)")
st.sidebar.caption("Accuracy: **80.55%**")  # CHANGE 1
st.sidebar.divider()
st.sidebar.caption("BCA Capstone | CAN304\nDIT University")

# ════════════════════════════════════════
# HOME
# ════════════════════════════════════════
if option == "Home":
    st.markdown('<p class="big-title">🧠 AI-Driven Sentiment Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Brand & Product Intelligence System &nbsp;|&nbsp; DIT University &nbsp;|&nbsp; CAN304 &nbsp;|&nbsp; BCA Capstone</p>', unsafe_allow_html=True)
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", "Hybrid (Logistic Regression + TF-IDF + VADER)")
    c2.metric("Accuracy", "80.55%", "+13.24% vs VADER")  # CHANGE 2
    c3.metric("Training Dataset", "Sentiment140")
    c4.metric("Training Samples", "100,000")
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📝 Text Analysis**\n\nAnalyze single or multiple texts with confidence score, VADER score, and preprocessing breakdown. Supports negation-aware classification.")
    with col2:
        st.info("**🌐 Live Brand Monitor**\n\nFetch real-time news headlines for any brand or product. Sentiment marked with clickable source links.")
    with col3:
        st.info("**📦 Product Intelligence**\n\nUpload any review CSV — Amazon, Flipkart, or custom. Bulk analysis with downloadable results.")

    st.divider()
    st.markdown('<p class="section-header">📝 Analyze Text</p>', unsafe_allow_html=True)

    with st.form(key="analyze_form", clear_on_submit=False):
        user_input = st.text_area(
            "Enter text — single line or paste multiple reviews (one per line):",
            height=140,
            placeholder="e.g.\nnot bad at all\nnothing great about this product\nthis is absolutely amazing!"
        )
        col1, col2 = st.columns([2, 1])
        with col1:
            source_type = st.selectbox(
                "Text Source Type",
                ["general", "review", "social", "news"],
                help="Affects ML vs VADER weighting. review=ML 80% | news=VADER 60% | social=balanced | general=default"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            analyze = st.form_submit_button("Analyze Sentiment", type="primary", use_container_width=True)

    if analyze:
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
                    f'<div style="background:#e9ecef;border-radius:10px;height:24px;width:100%">'
                    f'<div style="background:{color};width:{min(conf,100)}%;height:24px;border-radius:10px;'
                    f'display:flex;align-items:center;justify-content:center;color:white;'
                    f'font-size:12px;font-weight:700">{conf}%</div></div>',
                    unsafe_allow_html=True
                )
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("🔍 Preprocessing Details"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Original:**")
                        st.code(lines[0])
                    with col2:
                        st.write("**Cleaned (fed to model):**")
                        st.code(cleaned)
                    neg_words = [w for w in cleaned.split()
                                 if w in NEGATION_MAP.values() or w.endswith("_NEG")]
                    if neg_words:
                        st.success(f"Negation applied on: {', '.join(neg_words)}")
                    else:
                        st.info("No negation detected")
            else:
                st.divider()
                with st.spinner("Analyzing all lines..."):
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

                st.divider()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total",       len(results))
                c2.metric("Positive 🟢", pos)
                c3.metric("Negative 🔴", neg)
                c4.metric("Neutral ⚪",  neu)

                try:
                    import plotly.express as px
                    counts_df = pd.DataFrame({
                        "Sentiment": ["Positive", "Negative", "Neutral"],
                        "Count":     [pos, neg, neu]
                    })
                    fig = px.pie(
                        counts_df, names="Sentiment", values="Count",
                        color="Sentiment",
                        color_discrete_map={"Positive":"#28a745","Negative":"#dc3545","Neutral":"#6c757d"},
                        title="Batch Sentiment Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass
        else:
            st.warning("Please enter some text!")

    st.divider()
    st.markdown('<p class="section-header">ℹ️ Source Type Weighting</p>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Source Type":  ["review", "social", "news", "general"],
        "ML Weight":    ["80%",    "40%",    "40%",  "65%"],
        "VADER Weight": ["20%",    "60%",    "60%",  "35%"],
        "Best For":     [
            "Amazon / Flipkart / product reviews",
            "Tweets, Reddit, Instagram posts",
            "News headlines, articles",
            "Mixed or unknown source"
        ]
    }), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown('<p class="section-header">✅ Negation Handling — Key Innovation</p>', unsafe_allow_html=True)
    st.table(pd.DataFrame({
        "Phrase": [
            "not bad at all",
            "nothing great about this",
            "not very good",
            "nothing good came from this",
            "can not complain really"
        ],
        "Without Negation Handling": [
            "❌ Negative (wrong)",
            "❌ Positive (wrong)",
            "❌ Positive (wrong)",
            "❌ Positive (wrong)",
            "❌ Negative (wrong)"
        ],
        "Our Hybrid Model": [
            "✅ Positive",
            "✅ Negative",
            "✅ Negative",
            "✅ Negative",
            "✅ Positive"
        ]
    }))

# ════════════════════════════════════════
# LIVE BRAND MONITOR
# ════════════════════════════════════════
elif option == "Live Brand Monitor":
    st.markdown('<p class="big-title">🌐 Live Brand Market Monitor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Real-time news sentiment tracking — search any brand or product</p>', unsafe_allow_html=True)
    st.divider()

    query = st.text_input("Brand / Product Name",
                           placeholder="e.g. Apple, iPhone, Samsung, Tesla")

    col1, col2 = st.columns(2)
    with col1:
        filter_sentiment = st.selectbox(
            "Filter by Sentiment",
            ["All", "Positive 🟢", "Negative 🔴", "Neutral ⚪"]
        )
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            ["Date (Newest First)", "Date (Oldest First)",
             "Confidence (High to Low)", "Confidence (Low to High)"]
        )

    if st.button("Generate Report", type="primary"):
        if not query.strip():
            st.warning("Enter a brand or product name!")
        else:
            try:
                from gnews import GNews
                with st.spinner(f"Fetching live news for '{query}'..."):
                    gn = GNews(language="en", country="IN", max_results=50)
                    articles = gn.get_news(query)

                if not articles:
                    st.warning("No news found. Try a different query.")
                else:
                    query_words = [w.lower() for w in query.strip().split() if len(w) > 2]
                    data = []

                    for art in articles:
                        title = art.get("title", "")
                        link  = art.get("url", art.get("link", ""))
                        date  = art.get("published date", "")
                        if not title:
                            continue
                        title_lower = title.lower()
                        if not all(w in title_lower for w in query_words):
                            continue
                        lbl, cf, _, sc = get_hybrid_sentiment(title, model, tfidf, "news")
                        data.append({
                            "Headline":       title,
                            "Sentiment":      f"{'🟢' if lbl=='Positive' else ('🔴' if lbl=='Negative' else '⚪')} {lbl}",
                            "Label":          lbl,
                            "Confidence":     f"{cf}%",
                            "Confidence_raw": cf,
                            "Date":           date,
                            "Link":           link
                        })

                    # Fallback to partial match
                    if not data:
                        for art in articles:
                            title = art.get("title", "")
                            link  = art.get("url", art.get("link", ""))
                            date  = art.get("published date", "")
                            if not title:
                                continue
                            title_lower = title.lower()
                            if not any(w in title_lower for w in query_words):
                                continue
                            lbl, cf, _, sc = get_hybrid_sentiment(title, model, tfidf, "news")
                            data.append({
                                "Headline":       title,
                                "Sentiment":      f"{'🟢' if lbl=='Positive' else ('🔴' if lbl=='Negative' else '⚪')} {lbl}",
                                "Label":          lbl,
                                "Confidence":     f"{cf}%",
                                "Confidence_raw": cf,
                                "Date":           date,
                                "Link":           link
                            })

                    if not data:
                        st.warning("No relevant articles found.")
                    else:
                        pos = sum(1 for d in data if d["Label"] == "Positive")
                        neg = sum(1 for d in data if d["Label"] == "Negative")
                        neu = sum(1 for d in data if d["Label"] == "Neutral")

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Total Headlines", len(data))
                        c2.metric("Positive 🟢",     pos)
                        c3.metric("Negative 🔴",     neg)
                        c4.metric("Neutral ⚪",      neu)
                        st.divider()

                        filter_map = {
                            "All":         data,
                            "Positive 🟢": [d for d in data if d["Label"] == "Positive"],
                            "Negative 🔴": [d for d in data if d["Label"] == "Negative"],
                            "Neutral ⚪":  [d for d in data if d["Label"] == "Neutral"],
                        }
                        filtered = filter_map[filter_sentiment]

                        if sort_by == "Date (Newest First)":
                            filtered = sorted(filtered, key=lambda x: x["Date"], reverse=True)
                        elif sort_by == "Date (Oldest First)":
                            filtered = sorted(filtered, key=lambda x: x["Date"])
                        elif sort_by == "Confidence (High to Low)":
                            filtered = sorted(filtered, key=lambda x: x["Confidence_raw"], reverse=True)
                        elif sort_by == "Confidence (Low to High)":
                            filtered = sorted(filtered, key=lambda x: x["Confidence_raw"])

                        import plotly.express as px
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            counts_df = pd.DataFrame({
                                "Sentiment": ["Positive", "Negative", "Neutral"],
                                "Count":     [pos, neg, neu]
                            })
                            fig = px.pie(
                                counts_df, names="Sentiment", values="Count",
                                color="Sentiment",
                                color_discrete_map={
                                    "Positive":"#28a745",
                                    "Negative":"#dc3545",
                                    "Neutral": "#6c757d"
                                },
                                title=f"Sentiment Distribution — {query}",
                                hole=0.4
                            )
                            fig.update_layout(
                                legend=dict(orientation="h", y=-0.1),
                                margin=dict(t=40, b=0, l=0, r=0)
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            PAGE_SIZE  = 10
                            total_pages = max(1, -(-len(filtered) // PAGE_SIZE))
                            page = st.number_input(
                                f"Page (1–{total_pages})",
                                min_value=1, max_value=total_pages,
                                value=1, step=1
                            )
                            start = (page - 1) * PAGE_SIZE
                            end   = start + PAGE_SIZE

                            st.caption(
                                f"{'Filtered: ' + filter_sentiment + ' | ' if filter_sentiment != 'All' else ''}"
                                f"Showing {start+1}–{min(end, len(filtered))} of {len(filtered)} articles"
                            )

                            df_show = pd.DataFrame([{
                                "Headline":   d["Headline"],
                                "Sentiment":  d["Sentiment"],
                                "Confidence": d["Confidence"],
                                "Date":       d["Date"],
                                "Link":       d["Link"]
                            } for d in filtered[start:end]])

                            st.dataframe(
                                df_show,
                                column_config={
                                    "Link": st.column_config.LinkColumn("Open Article")
                                },
                                hide_index=True,
                                use_container_width=True
                            )

            except Exception as e:
                st.error(f"Error fetching news: {e}")

# ════════════════════════════════════════
# PRODUCT INTELLIGENCE
# ════════════════════════════════════════
elif option == "Product Intelligence":
    st.markdown('<p class="big-title">📦 Product Intelligence</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Upload any review CSV for bulk sentiment analysis</p>', unsafe_allow_html=True)
    st.divider()

    st.info(
        "**Supported formats:** Amazon reviews, Flipkart reviews, or any custom CSV with a text column.  \n"
        "For Amazon reviews: select `reviews.text` as the text column and `review` as source type."
    )

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df_rev = pd.read_csv(file)
        st.success(f"Loaded **{len(df_rev):,}** rows, **{len(df_rev.columns)}** columns")

        col1, col2, col3 = st.columns(3)
        with col1:
            text_col = st.selectbox("Select Text Column", df_rev.columns)
        with col2:
            src_type = st.selectbox("Source Type",
                                     ["review", "general", "social", "news"])
        with col3:
            limit = st.slider("Max rows to analyze",
                               50, min(500, len(df_rev)), 100)

        st.dataframe(df_rev[[text_col]].head(5), use_container_width=True)

        if st.button("Run Analysis", type="primary"):
            with st.spinner(f"Analyzing {limit} reviews..."):
                df_sample = df_rev[[text_col]].dropna().head(limit).copy()
                df_sample["Sentiment"] = df_sample[text_col].apply(
                    lambda x: get_hybrid_sentiment(str(x), model, tfidf, src_type)[0]
                )
                counts = df_sample["Sentiment"].value_counts()

            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Analyzed",    limit)
            c2.metric("Positive 🟢", counts.get("Positive", 0))
            c3.metric("Negative 🔴", counts.get("Negative", 0))
            c4.metric("Neutral ⚪",  counts.get("Neutral",  0))

            try:
                import plotly.express as px
                col1, col2 = st.columns(2)
                with col1:
                    fig1 = px.pie(
                        counts.reset_index(),
                        names="Sentiment", values="count",
                        color="Sentiment",
                        color_discrete_map={
                            "Positive":"#28a745",
                            "Negative":"#dc3545",
                            "Neutral": "#6c757d"
                        },
                        title="Sentiment Distribution",
                        hole=0.4
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                with col2:
                    fig2 = px.bar(
                        counts.reset_index(),
                        x="Sentiment", y="count",
                        color="Sentiment",
                        color_discrete_map={
                            "Positive":"#28a745",
                            "Negative":"#dc3545",
                            "Neutral": "#6c757d"
                        },
                        title="Count by Sentiment",
                        text="count"
                    )
                    fig2.update_traces(textposition="outside")
                    fig2.update_layout(showlegend=False)
                    st.plotly_chart(fig2, use_container_width=True)
            except Exception:
                st.bar_chart(counts)

            st.divider()
            st.markdown("**Results Preview**")
            st.dataframe(
                df_sample[[text_col, "Sentiment"]],
                use_container_width=True, hide_index=True
            )
            csv_out = df_sample[[text_col, "Sentiment"]].to_csv(index=False)
            st.download_button(
                "⬇️ Download Results CSV",
                csv_out, "sentiment_results.csv", "text/csv",
                use_container_width=True
            )

# ════════════════════════════════════════
# MODEL PERFORMANCE
# ════════════════════════════════════════
elif option == "Model Performance":
    st.markdown('<p class="big-title">📊 Model Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Comparison of all models trained on Sentiment140 (100k sample)</p>', unsafe_allow_html=True)
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Accuracy",      "80.55%")   # CHANGE 5
    c2.metric("vs VADER baseline",  "+13.24%")
    c3.metric("Precision",          "0.81")     # CHANGE 4
    c4.metric("F1-Score",           "0.81")     # CHANGE 4
    st.divider()

    # updated model leaderboard with full names
    model_data = pd.DataFrame({
        "Model":    [
            "VADER",
            "TextCNN (Deep Learning)",
            "Naive Bayes + Bag of Words",
            "BiLSTM (Deep Learning)",
            "Naive Bayes + TF-IDF",
            "Logistic Regression + TF-IDF",
            "Hybrid (Logistic Regression + TF-IDF + VADER)"
        ],
        "Accuracy": [67.31, 77.16, 77.45, 78.92, 78.83, 79.57, 80.55],
        "Type":     ["Lexicon", "Deep Learning", "ML", "Deep Learning", "ML", "ML", "Hybrid"]
    })

    try:
        import plotly.express as px
        fig = px.bar(
            model_data, x="Model", y="Accuracy",
            color="Type", text="Accuracy",
            color_discrete_map={
                "Lexicon":       "#adb5bd",
                "ML":            "#0d6efd",
                "Deep Learning": "#fd7e14",
                "Hybrid":        "#28a745"
            },
            title="Model Accuracy Comparison — Sentiment140 (100k sample)"
        )
        fig.add_hline(
            y=80, line_dash="dash", line_color="red",
            annotation_text="80% target",
            annotation_position="top right"
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(
            yaxis_range=[60, 86],
            showlegend=True,
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="rgba(0,0,0,0.05)")
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.bar_chart(model_data.set_index("Model")["Accuracy"])

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**All Models — Ranked by Accuracy**")
        display_df = model_data.copy()
        display_df["Status"] = display_df["Accuracy"].apply(
            lambda x: "✅ Best" if x == 80.55 else ""
        )
        st.dataframe(
            display_df[["Model","Type","Accuracy","Status"]].sort_values(
                "Accuracy", ascending=False
            ),
            use_container_width=True, hide_index=True
        )

    with col2:
        st.markdown("**Classification Report — Hybrid (Logistic Regression + TF-IDF + VADER)**")
        st.dataframe(pd.DataFrame({
            "Class":     ["Negative", "Positive", "Macro Avg"],
            "Precision": [0.81, 0.81, 0.81],   # CHANGE 4
            "Recall":    [0.81, 0.81, 0.81],   # CHANGE 4
            "F1-Score":  [0.81, 0.81, 0.81],   # CHANGE 4
            "Support":   [9989, 10011, 20000],
        }), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Why VADER alone scores 67.31%?")
    st.markdown("""
    VADER is a **rule-based lexicon** — it has no training phase and cannot learn
    dataset-specific patterns (Twitter slang, sarcasm, domain-specific terms).
    It is included as a **baseline** to demonstrate that supervised ML outperforms
    unsupervised lexicon methods.

    In our Hybrid (Logistic Regression + TF-IDF + VADER) model, VADER contributes as **extra signal** (3 features: pos, neg,
    compound scores) alongside 100,000 TF-IDF features — not as the primary classifier.
    This combination gives us +13.24% over VADER standalone.
    """)

# ════════════════════════════════════════
# ABOUT
# ════════════════════════════════════════
elif option == "About":
    st.markdown('<p class="big-title">ℹ️ About This Project</p>', unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
### AI-Driven Sentiment Analysis for Brand & Product Intelligence
**Course:** CAN304 &nbsp;|&nbsp; **University:** DIT University &nbsp;|&nbsp; **Mode:** A

This system is a hybrid sentiment intelligence platform combining statistical
machine learning with lexicon-based analysis. It enables real-time brand monitoring,
product review intelligence, and negation-aware sentiment classification.

---
### Team
| Name | SAP ID |
|------|--------|
| Ashish Pal | 1000021731 |
| Shaurya Pundir | 1000021913 |
| Abhishek | 1000022005 |

**Faculty Advisor:** Riya Dhama &nbsp;|&nbsp; riya.dhama@dituniversity.edu.in
        """)

    with col2:
        st.markdown("### Tech Stack")
        st.dataframe(pd.DataFrame({
            "Component": ["Language", "ML Library", "NLP", "Dataset",
                          "Dashboard", "Deployment", "Version Control"],
            "Technology": ["Python 3", "Scikit-learn", "TF-IDF + VADER",
                           "Sentiment140", "Streamlit", "Streamlit Cloud", "GitHub"]
        }), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### System Architecture")
    st.code("""
Raw Text Input
    │
    ├── [News/Social] clean_text_simple()   ──► TF-IDF ──► LR Model ──► ML Score
    │   (no negation mapping)                                              │
    │                                                                      ▼
    ├── [Review/General] clean_text_ml()                     Dynamic Weighting
    │   (negation + antonym replacement)                     (source-aware)
    │                                                                      │
    └── VADER (always on raw text) ──────────────────────► VADER Score    │
        pos, neg, compound as features                                     │
                                                              Conflict Resolution
                                                              (short negation fix)
                                                                           │
                                                                    Final Label
                                                         Positive / Negative / Neutral
    """, language="")

    st.divider()
    st.markdown("### Key Innovation — 3 Layer Negation Handling")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("**Layer 1 — Antonym Replacement**\n\nnot good → bad\n\nnothing great → terrible\n\nnot bad → good")
    with col2:
        st.info("**Layer 2 — VADER Extra Features**\n\nVADER pos/neg/compound scores appended as 3 extra columns to TF-IDF matrix")
    with col3:
        st.warning("**Layer 3 — Intensifier Skip + Conflict Resolution**\n\nnot very good → bad\n\nShort negations trust VADER over ML")

    st.divider()
    st.markdown("### GitHub Repository")
    st.markdown("[github.com/Ashishpal0219/sentiment-analysis-capstone](https://github.com/Ashishpal0219/sentiment-analysis-capstone)")
