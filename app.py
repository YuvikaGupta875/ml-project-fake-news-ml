import re
import numpy as np
import pandas as pd
import streamlit as st
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# NLTK setup (safe in Streamlit)
# -----------------------------
@st.cache_resource
def ensure_nltk():
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")

ensure_nltk()
STOP_WORDS = set(stopwords.words("english"))
PORT_STEM = PorterStemmer()

def stemming(content: str) -> str:
    content = re.sub(r"[^a-zA-Z]", " ", content)
    content = content.lower().split()
    content = [PORT_STEM.stem(w) for w in content if w not in STOP_WORDS]
    return " ".join(content)

# -----------------------------
# Synthetic dataset (Kaggle-like)
# -----------------------------
def make_synthetic_news_dataset(n_rows=60000, seed=42):
    rng = np.random.default_rng(seed)

    authors_real = ["Reuters Staff", "AP Newsroom", "BBC Reporter", "NYT Desk", "The Economist", "Bloomberg Writer"]
    authors_fake = ["TruthSeeker99", "FreedomEagle", "PatriotNews", "WakeUpWorld", "AnonSource", "ViralDaily"]

    real_topics = ["economy", "election", "health", "science", "climate", "sports", "technology", "education"]
    fake_topics = ["secret plot", "miracle cure", "shocking reveal", "hidden truth", "banned video", "they don't want you to know"]

    real_verbs = ["reports", "announces", "confirms", "explains", "states", "updates", "releases"]
    fake_verbs = ["exposes", "proves", "destroys", "reveals", "uncovers", "leaks", "shows"]

    real_title_templates = [
        "{org} {verb} new findings on {topic}",
        "{topic_cap} update: officials {verb} latest figures",
        "Study {verb} trends in {topic}",
        "Experts {verb} policy changes affecting {topic}",
        "{org} {verb} report related to {topic}"
    ]
    fake_title_templates = [
        "SHOCKING: {topic_cap} {verb} in leaked documents",
        "You won’t believe this: {topic} {verb} everything",
        "BREAKING!!! {topic_cap} {verb} the mainstream media",
        "VIRAL: {topic} {verb} - share before deleted",
        "EXPOSED: {topic_cap} {verb} secret agenda"
    ]

    orgs = ["Government", "Ministry", "Researchers", "Officials", "Agency", "Committee", "University"]

    filler_real = [
        "according to official sources", "in a statement on Tuesday", "based on peer-reviewed evidence",
        "as reported by multiple outlets", "after months of analysis", "with supporting data"
    ]
    filler_fake = [
        "they are hiding it", "this will get deleted", "share this now", "mainstream media won't tell you",
        "wake up everyone", "banned everywhere"
    ]

    ids = np.arange(1, n_rows + 1)
    labels = rng.integers(0, 2, size=n_rows)  # 0 real, 1 fake

    titles, authors, texts = [], [], []
    for y in labels:
        if y == 0:
            author = rng.choice(authors_real)
            topic = rng.choice(real_topics)
            verb = rng.choice(real_verbs)
            org = rng.choice(orgs)
            template = rng.choice(real_title_templates)
            title = template.format(org=org, verb=verb, topic=topic, topic_cap=topic.capitalize())
            text = (
                f"{title}. {rng.choice(filler_real)}. "
                f"The article discusses recent developments in {topic} and provides context, numbers, and quotes. "
                f"Background information is included to support the claims."
            )
        else:
            author = rng.choice(authors_fake)
            topic = rng.choice(fake_topics)
            verb = rng.choice(fake_verbs)
            template = rng.choice(fake_title_templates)
            title = template.format(verb=verb, topic=topic, topic_cap=topic.capitalize())
            text = (
                f"{title}. {rng.choice(filler_fake)}!!! "
                f"This post claims a {topic} and uses sensational language, vague sources, and emotional triggers. "
                f"It encourages rapid sharing without verification."
            )

        titles.append(title)
        authors.append(author)
        texts.append(text)

    return pd.DataFrame({
        "id": ids,
        "title": titles,
        "author": authors,
        "text": texts,
        "label": labels
    })

# -----------------------------
# Train + cache the model
# -----------------------------
@st.cache_resource
def train_model(n_rows=60000, seed=42):
    df = make_synthetic_news_dataset(n_rows=n_rows, seed=seed)
    df = df.fillna("")
    df["content"] = (df["author"] + " " + df["title"]).apply(stemming)

    X = df["content"].values
    y = df["label"].values

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, stratify=y, random_state=2
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    train_acc = accuracy_score(model.predict(X_train), y_train)
    test_acc = accuracy_score(model.predict(X_test), y_test)

    return model, vectorizer, train_acc, test_acc

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detector (TF-IDF + Logistic Regression)")
st.write("Enter a news *title + author* (or paste full text) and the model predicts **Real** or **Fake**.")

with st.sidebar:
    st.header("Training (Synthetic data)")
    n_rows = st.slider("Synthetic dataset size", min_value=10000, max_value=120000, value=60000, step=10000)
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)
    st.caption("Model trains on synthetic Kaggle-like columns: id, title, author, text, label.")

model, vectorizer, train_acc, test_acc = train_model(n_rows=int(n_rows), seed=int(seed))

st.info(f"Model trained ✅  |  Train accuracy: **{train_acc:.3f}**  |  Test accuracy: **{test_acc:.3f}**")

tab1, tab2 = st.tabs(["Predict using Title + Author", "Predict using Full Text"])

with tab1:
    author_in = st.text_input("Author", value="Reuters Staff")
    title_in = st.text_input("Title", value="Government announces new findings on climate")
    content = f"{author_in} {title_in}"
    if st.button("Predict (Title + Author)"):
        processed = stemming(content)
        vec = vectorizer.transform([processed])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]

        if pred == 0:
            st.success(f"✅ Prediction: REAL news (prob={proba[0]:.3f})")
        else:
            st.error(f"🚫 Prediction: FAKE news (prob={proba[1]:.3f})")

with tab2:
    text_in = st.text_area("Paste article text", height=180)
    if st.button("Predict (Full Text)"):
        if not text_in.strip():
            st.warning("Please paste some text first.")
        else:
            # keep same pipeline: we stem whatever user gives
            processed = stemming(text_in)
            vec = vectorizer.transform([processed])
            pred = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]

            if pred == 0:
                st.success(f"✅ Prediction: REAL news (prob={proba[0]:.3f})")
            else:
                st.error(f"🚫 Prediction: FAKE news (prob={proba[1]:.3f})")

st.caption("Note: This demo uses synthetic training data, so it won’t reflect real-world fake news patterns.")
