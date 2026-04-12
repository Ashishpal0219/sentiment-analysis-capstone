
import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="Sentiment Analysis — Brand & Product Intelligence",
    page_icon="🎯",
    layout="wide"
)

# ── Preprocessing ──
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
        elif tok in CLAUSE_BREAKS or tok in {'.', ',', '!', '?', ';'}:
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
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot",   text)
    text = re.sub(r"n't",   " not",     text)
    text = re.sub(r"'re",   " are",     text)
    text = re.sub(r"'ve",   " have",    text)
    text = re.sub(r"'ll",   " will",    text)
    text = re.sub(r"'d",    " would",   text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'#(\w+)', r'', text)
    text = re.sub(r'(.){2,}', r'', text)
    text = re.sub(r'[^a-zA-Z\s_]', '', text)
    return ' '.join(mark_negation(text.split())).strip()

def get_vader_features(texts):
    vader = SentimentIntensityAnalyzer()
    feats = []
    for t in texts:
        s = vader.polarity_scores(str(t))
        feats.append([s['pos'], s['neg'], s['compound']])
    return csr_matrix(np.array(feats))

# ── Train model (cached) ──
@st.cache_resource(show_spinner="Training model... (2-3 min, only once)")
def load_model():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/datasets/sentiment140/main/data/training.csv",
        encoding='latin-1', header=None,
        names=['sentiment','id','date','query','user','text'],
        nrows=100000
    )
    df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})
    df = df[['text','sentiment']].dropna()
    df['clean'] = df['text'].apply(clean_text)

    from sklearn.model_selection import train_test_split
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        df['text'], df['sentiment'], test_size=0.2,
        random_state=42, stratify=df['sentiment']
    )
    X_tr_clean = df.loc[X_tr_raw.index, 'clean']
    X_te_clean = df.loc[X_te_raw.index, 'clean']

    tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=100000,
                             min_df=2, max_df=0.95, sublinear_tf=True,
                             token_pattern=r'\w+')
    X_tr_tfidf = tfidf.fit_transform(X_tr_clean)
    X_te_tfidf = tfidf.transform(X_te_clean)

    vader_tr = get_vader_features(X_tr_raw)
    vader_te = get_vader_features(X_te_raw)

    X_tr = hstack([X_tr_tfidf, vader_tr])
    X_te = hstack([X_te_tfidf, vader_te])

    model = LogisticRegression(C=2.0, max_iter=1000, solver='saga', n_jobs=-1)
    model.fit(X_tr, y_tr)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_te, model.predict(X_te))
    return model, tfidf, acc

model, tfidf, acc = load_model()
vader = SentimentIntensityAnalyzer()

def predict(text):
    cleaned    = clean_text(text)
    tfidf_feat = tfidf.transform([cleaned])
    vader_feat = get_vader_features([text])
    features   = hstack([tfidf_feat, vader_feat])
    pred       = model.predict(features)[0]
    proba      = model.predict_proba(features)[0]
    label      = "Positive" if pred == 1 else "Negative"
    conf       = proba[pred] * 100
    return label, conf, cleaned

# ── UI ──
st.title("🎯 AI-Driven Sentiment Analysis")
st.markdown("**Brand & Product Intelligence System** — BCA Capstone | DIT University | CAN304")
st.divider()

col1, col2, col3 = st.columns(3)
col1.metric("Model", "Hybrid LR + TF-IDF + VADER")
col2.metric("Accuracy", f"{acc*100:.2f}%")
col3.metric("Dataset", "Sentiment140 — 100k")

st.divider()

# Single Analysis
st.subheader("📝 Single Text Analysis")
user_input = st.text_area("Enter text:", height=100,
                           placeholder="e.g. nothing great about this product...")
if st.button("Analyze", type="primary"):
    if user_input.strip():
        label, conf, cleaned = predict(user_input)
        c1, c2, c3 = st.columns(3)
        c1.metric("Sentiment", f"{'🟢' if label=='Positive' else '🔴'} {label}")
        c2.metric("Confidence", f"{conf:.1f}%")
        c3.metric("Words", len(user_input.split()))
        with st.expander("Preprocessed text"):
            st.write(f"**Original:** {user_input}")
            st.write(f"**Cleaned:** {cleaned}")
    else:
        st.warning("Please enter some text!")

st.divider()

# Batch Analysis
st.subheader("📊 Batch Analysis")
batch = st.text_area("Enter multiple texts (one per line):", height=150,
                      placeholder="This product is amazing!\nNothing good about this.\nNot bad at all.")
if st.button("Analyze All", type="secondary"):
    if batch.strip():
        lines = [l.strip() for l in batch.strip().split("\n") if l.strip()]
        results = []
        for line in lines:
            lbl, cf, _ = predict(line)
            results.append({"Text": line,
                            "Sentiment": f"{'🟢' if lbl=='Positive' else '🔴'} {lbl}",
                            "Confidence": f"{cf:.1f}%"})
        df_r = pd.DataFrame(results)
        st.dataframe(df_r, use_container_width=True)
        pos = sum(1 for r in results if "Positive" in r["Sentiment"])
        st.markdown(f"**Summary:** 🟢 {pos} Positive  🔴 {len(results)-pos} Negative")
    else:
        st.warning("Please enter some texts!")

st.divider()
with st.expander("ℹ️ Model Info"):
    st.markdown(f"""
    **Model:** Hybrid (Logistic Regression + TF-IDF + VADER)  
    **Accuracy:** {acc*100:.2f}%  
    **Dataset:** Sentiment140 (100k sample)  
    **Negation Handling:** not bad → Positive, nothing great → Negative  

    | Model | Accuracy |
    |-------|----------|
    | Hybrid (Our Model) | **{acc*100:.2f}%** |
    | LR + TF-IDF | 79.56% |
    | NB + TF-IDF | 78.83% |
    | LR + BoW | 77.85% |
    | NB + BoW | 77.45% |
    | VADER | 67.31% |
    """)
