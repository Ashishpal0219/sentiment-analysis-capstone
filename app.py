
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
    padding:15px;
    border-radius:12px;
    box-shadow:0px 2px 6px rgba(0,0,0,0.08);
}
.big-title { font-size:26px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ── Preprocessing ──
NEGATION_TRIGGERS = {
    "not","no","never","nothing","nobody","nowhere","neither","nor",
    "hardly","barely","scarcely","isnt","wasnt","shouldnt","wouldnt",
    "couldnt","doesnt","didnt","dont","cant","wont","aint","without",
    "lacking","fails","failed","lack"
}
CLAUSE_BREAKS = {
    "but","however","although","though","yet","so","because","if",
    "while","and","or","then","still","despite","except"
}
INTENSIFIERS = {
    "very","so","really","extremely","absolutely","completely","totally",
    "utterly","quite","super","pretty","rather","highly","truly","deeply"
}
NEGATION_MAP = {
    "great":"terrible","amazing":"awful","excellent":"poor",
    "fantastic":"horrible","wonderful":"dreadful","love":"hate",
    "best":"worst","perfect":"terrible","awesome":"awful",
    "brilliant":"stupid","good":"bad","nice":"unpleasant","fine":"poor",
    "happy":"unhappy","glad":"sad","pleased":"displeased","fun":"boring",
    "enjoy":"dislike","like":"dislike","beautiful":"ugly","easy":"difficult",
    "fast":"slow","helpful":"unhelpful","useful":"useless","safe":"unsafe",
    "clean":"dirty","fresh":"stale","smart":"stupid","kind":"unkind",
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

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"@\w+",           "",         text)
    text = re.sub(r"won't",         "will not", text)
    text = re.sub(r"can't",         "cannot",   text)
    text = re.sub(r"n't",           " not",     text)
    text = re.sub(r"'re",           " are",     text)
    text = re.sub(r"'ve",           " have",    text)
    text = re.sub(r"'ll",           " will",    text)
    text = re.sub(r"'d",            " would",   text)
    text = re.sub(r"http\S+|www\S+", "",         text)
    text = re.sub(r"#(\w+)",         r"",      text)
    text = re.sub(r"(.){2,}",      r"",    text)
    text = re.sub(r"[^a-zA-Z\s_]",  "",         text)
    tokens = text.split()
    tokens = mark_negation(tokens)
    result = " ".join(tokens).strip()
    return result if result else "unknown"

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model      = joblib.load("sentiment_model_final.pkl")
    vectorizer = joblib.load("tfidf_vectorizer_final.pkl")
    return model, vectorizer

model, tfidf   = load_model()
vader_analyzer = SentimentIntensityAnalyzer()

def predict(text, source_type="general"):
    cleaned    = clean_text(text)
    tfidf_feat = tfidf.transform([cleaned])
    scores     = vader_analyzer.polarity_scores(str(text))
    vader_feat = csr_matrix(np.array([[
        scores["pos"], scores["neg"], scores["compound"]
    ]]))
    features = hstack([tfidf_feat, vader_feat])
    pred     = model.predict(features)[0]
    proba    = model.predict_proba(features)[0]
    label    = "Positive" if pred == 1 else "Negative"
    conf     = round(proba[pred] * 100, 1)
    return label, conf, cleaned, scores["compound"]

# ── SIDEBAR ──
st.sidebar.title("🧠 Intelligence Engine")
st.sidebar.divider()
option = st.sidebar.selectbox(
    "Select Module",
    ["Home", "Single Analysis", "Product Intelligence", "Model Performance", "About"]
)
st.sidebar.divider()
st.sidebar.caption("BCA Capstone | DIT University | CAN304")
st.sidebar.caption("Model: Hybrid LR + TF-IDF + VADER")
st.sidebar.caption("Accuracy: 80.20%")

# ════════════════════════════════════════
# HOME
# ════════════════════════════════════════
if option == "Home":
    st.markdown('<p class="big-title">🧠 AI-Driven Brand & Product Intelligence</p>', unsafe_allow_html=True)
    st.caption(
        "A hybrid sentiment intelligence system combining Logistic Regression "
        "(trained on Sentiment140, 100k sample) and VADER lexicon analysis. "
        "Enables brand monitoring, product review intelligence, and negation-aware sentiment classification."
    )
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model",    "Hybrid LR + TF-IDF + VADER")
    c2.metric("Accuracy", "80.20%")
    c3.metric("Dataset",  "Sentiment140")
    c4.metric("Samples",  "100,000")
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📝 Single Analysis**\nAnalyze any text with confidence score, VADER score, and preprocessing breakdown.")
    with col2:
        st.info("**📦 Product Intelligence**\nUpload a CSV of reviews for bulk sentiment analysis with visual charts.")
    with col3:
        st.info("**📈 Model Performance**\nCompare all 6 models trained, view confusion matrix and classification report.")

    st.divider()
    st.markdown("#### Negation Handling — Key Innovation")
    st.markdown("Our model correctly handles negated phrases that break traditional models:")
    neg_data = {
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
        "Our Model": [
            "✅ Positive (correct)",
            "✅ Negative (correct)",
            "✅ Negative (correct)",
            "✅ Negative (correct)",
            "✅ Positive (correct)"
        ]
    }
    st.table(pd.DataFrame(neg_data))

# ════════════════════════════════════════
# SINGLE ANALYSIS
# ════════════════════════════════════════
elif option == "Single Analysis":
    st.title("📝 Single Text Analysis")
    st.caption("Analyze sentiment of any review, tweet, or feedback.")

    user_input = st.text_area("Enter text:", height=130,
                               placeholder="e.g. nothing great about this product, not worth the money...")
    text_type  = st.selectbox("Text Type", ["general", "social", "news", "review"])
    st.caption("Select text type for context.")

    if st.button("Analyze Sentiment", type="primary"):
        if user_input.strip():
            label, conf, cleaned, compound = predict(user_input, text_type)
            color = "green" if label == "Positive" else "red"

            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sentiment",   f"{'🟢' if label=='Positive' else '🔴'} {label}")
            c2.metric("Confidence",  f"{conf}%")
            c3.metric("VADER Score", f"{compound:+.3f}")
            c4.metric("Word Count",  len(user_input.split()))

            st.markdown("**Confidence Level**")
            st.markdown(
                f"""<div style="background:#e9ecef;border-radius:10px;height:22px;width:100%">
                <div style="background:{"#28a745" if label=="Positive" else "#dc3545"};
                width:{conf}%;height:22px;border-radius:10px;
                display:flex;align-items:center;justify-content:center;
                color:white;font-size:12px;font-weight:bold">{conf}%</div></div>""",
                unsafe_allow_html=True
            )
            st.progress(int(conf))

            st.divider()
            with st.expander("🔍 Preprocessing Details"):
                st.write(f"**Original:**  {user_input}")
                st.write(f"**Cleaned:**   {cleaned}")
                neg_words = [w for w in cleaned.split()
                             if w in NEGATION_MAP.values() or w.endswith("_NEG")]
                st.write(f"**Negation applied:** {', '.join(neg_words) if neg_words else 'none detected'}")
        else:
            st.warning("Please enter some text!")

    st.divider()
    st.markdown("#### Quick Examples")
    examples_test = [
        "not bad at all",
        "nothing great about this",
        "absolutely love this product",
        "worst purchase ever",
        "can not complain really",
        "nothing good came from this",
    ]
    cols = st.columns(3)
    for idx, ex in enumerate(examples_test):
        with cols[idx % 3]:
            if st.button(ex, key=f"ex_{idx}"):
                lbl, cf, cl, comp = predict(ex)
                emoji = "🟢" if lbl == "Positive" else "🔴"
                st.write(f"{emoji} **{lbl}** ({cf}%)")

# ════════════════════════════════════════
# PRODUCT INTELLIGENCE
# ════════════════════════════════════════
elif option == "Product Intelligence":
    st.title("📦 Product Intelligence")
    st.caption("Upload a CSV of product reviews for bulk sentiment analysis.")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df_rev   = pd.read_csv(file).head(200)
        text_col = st.selectbox("Select Review Column", df_rev.columns)
        st.dataframe(df_rev[[text_col]].head(5), use_container_width=True)

        if st.button("Run Analysis", type="primary"):
            with st.spinner("Analyzing reviews..."):
                df_rev["Sentiment"] = df_rev[text_col].apply(
                    lambda x: predict(str(x))[0]
                )
                counts = df_rev["Sentiment"].value_counts()

                st.divider()
                m1, m2 = st.columns(2)
                m1.metric("Positive 🟢", counts.get("Positive", 0))
                m2.metric("Negative 🔴", counts.get("Negative", 0))

                try:
                    import plotly.express as px
                    c1, c2 = st.columns(2)
                    with c1:
                        fig = px.pie(df_rev, names="Sentiment",
                                     color="Sentiment",
                                     color_discrete_map={"Positive":"#28a745","Negative":"#dc3545"},
                                     title="Sentiment Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    with c2:
                        fig2 = px.bar(counts.reset_index(),
                                      x="Sentiment", y="count",
                                      color="Sentiment",
                                      color_discrete_map={"Positive":"#28a745","Negative":"#dc3545"},
                                      title="Count by Sentiment")
                        st.plotly_chart(fig2, use_container_width=True)
                except:
                    st.bar_chart(counts)

                st.divider()
                st.dataframe(df_rev[[text_col, "Sentiment"]],
                             use_container_width=True, hide_index=True)

                csv_out = df_rev[[text_col, "Sentiment"]].to_csv(index=False)
                st.download_button("Download Results CSV", csv_out,
                                   "sentiment_results.csv", "text/csv")

# ════════════════════════════════════════
# MODEL PERFORMANCE
# ════════════════════════════════════════
elif option == "Model Performance":
    st.title("📊 Model Performance")

    m1, m2, m3 = st.columns(3)
    m1.metric("Best Accuracy", "80.20%")
    m2.metric("Improvement over VADER", "+12.89%")
    m3.metric("Dataset", "Sentiment140 — 100k")
    st.divider()

    model_data = pd.DataFrame({
        "Model":    ["VADER", "NB + BoW", "LR + BoW",
                     "NB + TF-IDF", "LR + TF-IDF", "Hybrid (Ours)"],
        "Accuracy": [67.31,   77.45,      77.85,
                     78.83,    79.56,       80.20],
    })

    try:
        import plotly.express as px
        fig = px.bar(model_data, x="Model", y="Accuracy",
                     color="Model", text="Accuracy",
                     color_discrete_sequence=["#adb5bd","#6c757d","#495057",
                                               "#343a40","#212529","#28a745"],
                     title="Model Accuracy Comparison (Sentiment140, 100k sample)")
        fig.add_hline(y=80, line_dash="dash", line_color="red",
                      annotation_text="80% target")
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(yaxis_range=[60, 85], showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.bar_chart(model_data.set_index("Model"))

    st.divider()
    model_data["Status"] = model_data["Accuracy"].apply(
        lambda x: "✅ Best" if x == 80.20 else ""
    )
    st.dataframe(model_data.sort_values("Accuracy", ascending=False),
                 use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Classification Report — Hybrid Model")
    st.dataframe(pd.DataFrame({
        "Class":     ["Negative", "Positive", "Macro Avg"],
        "Precision": [0.80, 0.80, 0.80],
        "Recall":    [0.80, 0.81, 0.80],
        "F1-Score":  [0.80, 0.80, 0.80],
        "Support":   [9989, 10011, 20000],
    }), use_container_width=True, hide_index=True)

# ════════════════════════════════════════
# ABOUT
# ════════════════════════════════════════
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
    | NLP | TF-IDF, VADER |
    | Dataset | Sentiment140 (100k) |
    | Dashboard | Streamlit |
    | Environment | Google Colab |

    ---
    ### Key Innovation — Negation Handling
    Traditional BoW/TF-IDF models fail on negated phrases because words are 
    treated independently. Our 3-layer approach:

    **Layer 1 — Antonym Replacement**
    `"not good"` → `"bad"` | `"nothing great"` → `"terrible"`

    **Layer 2 — VADER as Extra Features**
    VADER compound/pos/neg scores appended to TF-IDF matrix.

    **Layer 3 — Intensifier Skipping**
    `"not very good"` → `"bad"` (intensifier skipped)

    ---
    ### Why VADER alone scores 67.31%
    VADER is rule-based with no training — it cannot learn dataset-specific 
    patterns. Included as baseline to show supervised ML outperforms lexicon 
    methods. Most useful as extra signal combined with TF-IDF.
    """)
