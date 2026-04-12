
import streamlit as st
import numpy as np
import re
import joblib
from scipy.sparse import hstack, csr_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

st.set_page_config(
    page_title="Sentiment Analysis — Brand & Product Intelligence",
    page_icon="🎯",
    layout="wide"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main { padding: 0rem 1rem; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .positive { color: #28a745; font-weight: bold; font-size: 1.3rem; }
    .negative { color: #dc3545; font-weight: bold; font-size: 1.3rem; }
    .neutral  { color: #6c757d; font-weight: bold; font-size: 1.3rem; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 500; }
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
    "but","however","although","though","yet","so","because","if","while",
    "and","or","then","still","despite","except"
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
    "unhappy":"happy","sad":"happy","poor":"excellent","dreadful":"wonderful",
}

def mark_negation(tokens):
    result  = []
    negate  = False
    i       = 0
    neg_count = 0          # track how many words flipped
    MAX_SCOPE = 4          # max words to flip after negation trigger

    while i < len(tokens):
        tok = tokens[i]

        if tok in NEGATION_TRIGGERS:
            negate    = True
            neg_count = 0
            i += 1
            continue

        elif tok in INTENSIFIERS and negate:
            i += 1
            continue

        elif tok in CLAUSE_BREAKS or tok in {".", ",", "!", "?", ";", ":"}:
            negate    = False
            neg_count = 0
            result.append(tok)

        elif negate and neg_count < MAX_SCOPE:
            if tok in NEGATION_MAP:
                result.append(NEGATION_MAP[tok])
                negate    = False   # stop after first sentiment word found
                neg_count = 0
            else:
                result.append(tok + "_NEG")
                neg_count += 1
                if neg_count >= MAX_SCOPE:
                    negate    = False
                    neg_count = 0
        else:
            negate = False
            result.append(tok)

        i += 1
    return result

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"@\w+",            "",          text)
    text = re.sub(r"won't",          "will not",  text)
    text = re.sub(r"can't",          "cannot",    text)
    text = re.sub(r"n't",            " not",      text)
    text = re.sub(r"'re",            " are",      text)
    text = re.sub(r"'ve",            " have",     text)
    text = re.sub(r"'ll",            " will",     text)
    text = re.sub(r"'d",             " would",    text)
    text = re.sub(r"http\S+|www\S+",  "",          text)
    text = re.sub(r"#(\w+)",          r"",       text)
    text = re.sub(r"(.){2,}",       r"",     text)
    text = re.sub(r"[^a-zA-Z\s_]",   "",          text)
    tokens = text.split()
    tokens = mark_negation(tokens)
    result = " ".join(tokens).strip()
    return result if result else "unknown"

# ── Load Model ──
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model      = joblib.load("sentiment_model_final.pkl")
    vectorizer = joblib.load("tfidf_vectorizer_final.pkl")
    return model, vectorizer

model, tfidf   = load_model()
vader_analyzer = SentimentIntensityAnalyzer()

def predict(text):
    cleaned    = clean_text(text)
    tfidf_feat = tfidf.transform([cleaned])
    scores     = vader_analyzer.polarity_scores(str(text))
    vader_feat = csr_matrix(np.array([[
        scores["pos"], scores["neg"], scores["compound"]
    ]]))
    features   = hstack([tfidf_feat, vader_feat])
    pred       = model.predict(features)[0]
    proba      = model.predict_proba(features)[0]
    label      = "Positive" if pred == 1 else "Negative"
    conf       = proba[pred] * 100
    return label, conf, cleaned, scores["compound"]

# ── TABS ──
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Home",
    "📝 Single Analysis",
    "📊 Batch Analysis",
    "📈 Model Performance",
    "ℹ️ About"
])

# ════════════════════════════════════════
# TAB 1 — HOME
# ════════════════════════════════════════
with tab1:
    st.title("🎯 AI-Driven Sentiment Analysis")
    st.markdown("### Brand & Product Intelligence System")
    st.markdown("**DIT University | CAN304 | BCA Capstone Project**")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model",    "Hybrid LR + TF-IDF + VADER")
    c2.metric("Accuracy", "80.20%")
    c3.metric("Dataset",  "Sentiment140")
    c4.metric("Samples",  "100,000")
    st.divider()

    st.markdown("#### What this system does")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📝 Single Analysis**\nAnalyze sentiment of any text with confidence score and preprocessing breakdown.")
    with col2:
        st.info("**📊 Batch Analysis**\nAnalyze multiple texts at once and get a summary with positive/negative counts.")
    with col3:
        st.info("**📈 Model Performance**\nView accuracy comparison of all models, confusion matrix, and EDA charts.")

    st.divider()
    st.markdown("#### Negation Handling — Key Innovation")
    st.markdown("Traditional models fail on negated phrases. Our hybrid model handles them correctly:")

    examples = {
        "not bad at all":               ("❌ Negative (wrong)", "✅ Positive (correct)"),
        "nothing great about this":     ("❌ Positive (wrong)", "✅ Negative (correct)"),
        "can not believe how bad":      ("❌ Positive (wrong)", "✅ Negative (correct)"),
        "not very good":                ("❌ Positive (wrong)", "✅ Negative (correct)"),
    }
    df_ex = pd.DataFrame(examples, index=["Without Negation Handling","With Negation Handling"]).T
    df_ex.index.name = "Phrase"
    st.table(df_ex)

# ════════════════════════════════════════
# TAB 2 — SINGLE ANALYSIS
# ════════════════════════════════════════
with tab2:
    st.header("📝 Single Text Analysis")
    st.markdown("Enter any product review, tweet, or feedback below.")

    user_input = st.text_area("Enter text:", height=120,
                               placeholder="e.g. nothing great about this product, not worth the money...")

    col_btn, col_clear = st.columns([1, 5])
    analyze_clicked = col_btn.button("Analyze", type="primary")

    if analyze_clicked:
        if user_input.strip():
            label, conf, cleaned, compound = predict(user_input)

            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            emoji = "🟢" if label == "Positive" else "🔴"
            c1.metric("Sentiment",  f"{emoji} {label}")
            c2.metric("Confidence", f"{conf:.1f}%")
            c3.metric("VADER Score",f"{compound:+.3f}")
            c4.metric("Word Count", len(user_input.split()))

            # Confidence bar
            st.markdown("**Confidence Level**")
            color = "green" if label == "Positive" else "red"
            st.markdown(
                f"""<div style="background:#e9ecef;border-radius:10px;height:20px;width:100%">
                <div style="background:{'#28a745' if label=='Positive' else '#dc3545'};
                width:{conf:.0f}%;height:20px;border-radius:10px;
                display:flex;align-items:center;justify-content:center;
                color:white;font-size:12px;font-weight:bold">{conf:.1f}%</div>
                </div>""",
                unsafe_allow_html=True
            )

            st.divider()
            with st.expander("🔍 See preprocessing details"):
                st.write(f"**Original:**  {user_input}")
                st.write(f"**Cleaned:**   {cleaned}")
                neg_words = [w for w in cleaned.split() 
                             if w in NEGATION_MAP.values() or w.endswith("_NEG")]
                if neg_words:
                    st.write(f"**Negation applied on:** {', '.join(neg_words)}")
                else:
                    st.write("**Negation applied on:** none detected")
        else:
            st.warning("Please enter some text!")

    st.divider()
    st.markdown("#### Quick Test Examples")
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
                st.write(f"{emoji} **{lbl}** ({cf:.1f}%)")

# ════════════════════════════════════════
# TAB 3 — BATCH ANALYSIS
# ════════════════════════════════════════
with tab3:
    st.header("📊 Batch Analysis")
    st.markdown("Enter multiple texts — one per line.")

    batch = st.text_area("Batch input:", height=200,
                          placeholder="This product is amazing!\nNothing good about this.\nNot bad at all.\nWorst experience ever.")

    if st.button("Analyze All", type="primary"):
        if batch.strip():
            lines = [l.strip() for l in batch.strip().split("\n") if l.strip()]
            results = []
            for line in lines:
                lbl, cf, cl, comp = predict(line)
                results.append({
                    "Text":       line,
                    "Sentiment":  f"{'🟢' if lbl=='Positive' else '🔴'} {lbl}",
                    "Confidence": f"{cf:.1f}%",
                    "VADER":      f"{comp:+.3f}"
                })

            df_r = pd.DataFrame(results)
            st.dataframe(df_r, use_container_width=True)

            pos = sum(1 for r in results if "Positive" in r["Sentiment"])
            neg = len(results) - pos
            st.divider()

            c1, c2, c3 = st.columns(3)
            c1.metric("Total",    len(results))
            c2.metric("Positive 🟢", pos)
            c3.metric("Negative 🔴", neg)

            # Pie chart
            if pos + neg > 0:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie([pos, neg] if neg > 0 else [pos, 0.001],
                       labels=["Positive", "Negative"],
                       colors=["#28a745", "#dc3545"],
                       autopct="%1.1f%%", startangle=90)
                ax.set_title("Sentiment Distribution")
                st.pyplot(fig)
                plt.close()
        else:
            st.warning("Please enter some texts!")

# ════════════════════════════════════════
# TAB 4 — MODEL PERFORMANCE
# ════════════════════════════════════════
with tab4:
    st.header("📈 Model Performance")

    # Accuracy comparison
    st.subheader("Model Accuracy Comparison")
    model_data = {
        "Model":    ["VADER", "NB + BoW", "LR + BoW",
                     "NB + TF-IDF", "LR + TF-IDF", "Hybrid (Ours)"],
        "Accuracy": [67.31,   77.45,      77.85,
                     78.83,    79.56,       80.20],
    }
    df_models = pd.DataFrame(model_data)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#adb5bd","#6c757d","#495057","#343a40","#212529","#28a745"]
    bars   = ax.bar(df_models["Model"], df_models["Accuracy"],
                    color=colors, edgecolor="black", width=0.5)
    ax.axhline(y=80, color="red", linestyle="--",
               linewidth=1.5, label="80% target")
    for bar, acc in zip(bars, df_models["Accuracy"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.2,
                f"{acc:.2f}%", ha="center",
                fontweight="bold", fontsize=9)
    ax.set_ylim(60, 85)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Comparison — Sentiment140 (100k sample)")
    ax.legend()
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # Table
    st.subheader("Detailed Results")
    df_models["Status"] = df_models["Accuracy"].apply(
        lambda x: "✅ Best" if x == 80.20 else ("🎯 Target met" if x >= 80 else "")
    )
    st.dataframe(df_models.sort_values("Accuracy", ascending=False),
                 use_container_width=True, hide_index=True)

    st.divider()

    # Classification report
    st.subheader("Best Model — Classification Report")
    report_data = {
        "Class":     ["Negative", "Positive", "Macro Avg"],
        "Precision": [0.80, 0.80, 0.80],
        "Recall":    [0.80, 0.81, 0.80],
        "F1-Score":  [0.80, 0.80, 0.80],
        "Support":   [9989, 10011, 20000],
    }
    st.dataframe(pd.DataFrame(report_data),
                 use_container_width=True, hide_index=True)

# ════════════════════════════════════════
# TAB 5 — ABOUT
# ════════════════════════════════════════
with tab5:
    st.header("ℹ️ About This Project")

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
    Traditional BoW/TF-IDF models fail on negated phrases because they treat
    each word independently. Our system uses a 3-layer approach:

    **Layer 1 — Antonym Replacement**
    Negation triggers flip the next sentiment word to its antonym:
    `"not good"` → `"bad"`, `"nothing great"` → `"terrible"`

    **Layer 2 — VADER as Extra Features**
    VADER compound/pos/neg scores appended to TF-IDF matrix —
    gives rule-based backup signal to the ML model.

    **Layer 3 — Intensifier Skipping**
    Words like "very", "so", "really" are skipped during negation scope:
    `"not very good"` → `"bad"` (not `"very_NEG good"`)

    ---
    ### Why VADER alone has low accuracy (67.31%)
    VADER is a **rule-based lexicon** — it has no training, so it cannot
    learn dataset-specific patterns. We include it as a baseline to
    demonstrate that supervised ML (which learns from data) outperforms
    unsupervised lexicon methods. VADER is most useful as extra signal
    combined with TF-IDF, not standalone.
    """)
