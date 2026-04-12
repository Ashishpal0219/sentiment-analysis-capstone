
import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
from scipy.sparse import hstack, csr_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="Sentiment Analysis — Brand & Product Intelligence",
    page_icon="🎯",
    layout="wide"
)

NEGATION_TRIGGERS = {
    "not","no","never","nothing","nobody","nowhere","neither","nor",
    "hardly","barely","scarcely","isnt","wasnt","shouldnt","wouldnt",
    "couldnt","doesnt","didnt","dont","cant","wont","aint"
}
CLAUSE_BREAKS = {"but","however","although","though","yet","so","because","if","while"}
INTENSIFIERS  = {"very","so","really","extremely","absolutely","completely",
                 "totally","utterly","quite","super"}
NEGATION_MAP  = {
    "great":"terrible","amazing":"awful","excellent":"poor","fantastic":"horrible",
    "wonderful":"dreadful","love":"hate","best":"worst","perfect":"terrible",
    "awesome":"awful","good":"bad","nice":"unpleasant","fine":"poor",
    "happy":"unhappy","glad":"sad","fun":"boring","enjoy":"dislike",
    "like":"dislike","beautiful":"ugly","bad":"good","terrible":"great",
    "awful":"good","horrible":"pleasant","boring":"interesting","hate":"love",
    "worst":"best","ugly":"beautiful","stupid":"smart","useless":"useful",
}

def mark_negation(tokens):
    result, negate, i = [], False, 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in NEGATION_TRIGGERS:
            negate = True; i += 1; continue
        elif tok in INTENSIFIERS and negate:
            i += 1; continue
        elif tok in CLAUSE_BREAKS or tok in {".", ",", "!", "?", ";"}:
            negate = False; result.append(tok)
        elif negate:
            result.append(NEGATION_MAP.get(tok, tok + "_NEG"))
            negate = False
        else:
            result.append(tok)
        i += 1
    return result

def clean_text(text):
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
    text = re.sub(r"#(\w+)", r"", text)
    text = re.sub(r"(.){2,}", r"", text)
    text = re.sub(r"[^a-zA-Z\s_]", "", text)
    tokens = text.split()
    tokens = mark_negation(tokens)
    result = " ".join(tokens).strip()
    return result if result else "unknown"

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model      = joblib.load("sentiment_model_final.pkl")
    vectorizer = joblib.load("tfidf_vectorizer_final.pkl")
    return model, vectorizer

model, tfidf = load_model()
vader_analyzer = SentimentIntensityAnalyzer()

def predict(text):
    cleaned    = clean_text(text)
    tfidf_feat = tfidf.transform([cleaned])
    scores     = vader_analyzer.polarity_scores(str(text))
    vader_feat = csr_matrix(np.array([[scores["pos"], scores["neg"], scores["compound"]]]))
    features   = hstack([tfidf_feat, vader_feat])
    pred       = model.predict(features)[0]
    proba      = model.predict_proba(features)[0]
    label      = "Positive" if pred == 1 else "Negative"
    conf       = proba[pred] * 100
    return label, conf, cleaned

st.title("🎯 AI-Driven Sentiment Analysis")
st.markdown("**Brand & Product Intelligence System** — BCA Capstone | DIT University | CAN304")
st.divider()

col1, col2, col3 = st.columns(3)
col1.metric("Model", "Hybrid LR + TF-IDF + VADER")
col2.metric("Accuracy", "80.20%")
col3.metric("Dataset", "Sentiment140 — 100k")
st.divider()

st.subheader("📝 Single Text Analysis")
user_input = st.text_area("Enter text:", height=100,
                           placeholder="e.g. nothing great about this product...")
if st.button("Analyze", type="primary"):
    if user_input.strip():
        label, conf, cleaned = predict(user_input)
        c1, c2, c3 = st.columns(3)
        c1.metric("Sentiment", f"{'🟢' if label == 'Positive' else '🔴'} {label}")
        c2.metric("Confidence", f"{conf:.1f}%")
        c3.metric("Words", len(user_input.split()))
        with st.expander("Preprocessed text"):
            st.write(f"**Original:** {user_input}")
            st.write(f"**Cleaned:** {cleaned}")
    else:
        st.warning("Please enter some text!")

st.divider()

st.subheader("📊 Batch Analysis")
batch = st.text_area("Enter multiple texts (one per line):", height=150,
                      placeholder="This product is amazing!\nNothing good about this.\nNot bad at all.")
if st.button("Analyze All", type="secondary"):
    if batch.strip():
        lines = [l.strip() for l in batch.strip().split("\n") if l.strip()]
        results = []
        for line in lines:
            lbl, cf, _ = predict(line)
            results.append({
                "Text": line,
                "Sentiment": f"{'🟢' if lbl == 'Positive' else '🔴'} {lbl}",
                "Confidence": f"{cf:.1f}%"
            })
        df_r = pd.DataFrame(results)
        st.dataframe(df_r, use_container_width=True)
        pos = sum(1 for r in results if "Positive" in r["Sentiment"])
        st.markdown(f"**Summary:** 🟢 {pos} Positive  🔴 {len(results)-pos} Negative")
    else:
        st.warning("Please enter some texts!")

st.divider()
with st.expander("ℹ️ Model Info"):
    st.markdown("""
    **Model:** Hybrid (Logistic Regression + TF-IDF + VADER)  
    **Accuracy:** 80.20%  
    **Dataset:** Sentiment140 (100k sample)  
    **Key Feature:** Negation handling — not bad → Positive, nothing great → Negative  

    | Model | Accuracy |
    |-------|----------|
    | Hybrid (Our Model) | **80.20%** |
    | LR + TF-IDF | 79.56% |
    | NB + TF-IDF | 78.83% |
    | LR + BoW | 77.85% |
    | NB + BoW | 77.45% |
    | VADER | 67.31% |
    """)
