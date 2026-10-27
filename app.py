
%%writefile app.py
import streamlit as st
import pickle
import gzip
import re
import nltk
import distance
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
from scipy.sparse import hstack, csr_matrix

nltk.download("stopwords", quiet=True)

with gzip.open("model.pkl.gz", "rb") as f:
    model = pickle.load(f)

with open("cv.pkl", "rb") as f:
    cv = pickle.load(f)

STOP_WORDS = set(stopwords.words("english"))

def preprocess(q):
    q = str(q).lower().strip()
    q = q.replace("%", " percent").replace("$", " dollar ")
    q = q.replace("₹", " rupee ").replace("€", " euro ")
    q = q.replace("@", " at ").replace("[math]", "")
    q = re.sub(r"([0-9]+)000000000", r"\1b", q)
    q = re.sub(r"([0-9]+)000000", r"\1m", q)
    q = re.sub(r"([0-9]+)000", r"\1k", q)

    contractions = {
        "can't": "can not", "won't": "will not",
        "n't": " not", "i'm": "i am",
        "it's": "it is", "what's": "what is",
        "you're": "you are", "they're": "they are",
        "we're": "we are", "i've": "i have",
        "don't": "do not", "doesn't": "does not",
        "didn't": "did not", "isn't": "is not",
        "aren't": "are not", "wasn't": "was not",
        "weren't": "were not", "i'll": "i will",
        "you'll": "you will", "they'll": "they will",
        "we'll": "we will", "i'd": "i would",
        "shouldn't": "should not", "wouldn't": "would not"
    }

    for old, new in contractions.items():
        q = q.replace(old, new)

    q = BeautifulSoup(q, "html.parser").get_text()
    return re.sub(r"\W", " ", q).strip()


def make_features(q1, q2):
    a, b = q1.split(), q2.split()
    sa, sb = set(a), set(b)

    common = len(sa.intersection(sb))
    total = len(sa) + len(sb)
    word_share = round(common / total, 2) if total else 0

    # First 7 engineered features
    features = [
        len(q1), len(q2), len(a), len(b),
        common, total, word_share
    ]

    # 8 token features
    if not a or not b:
        features.extend([0.0] * 8)
    else:
        aw = set(x for x in a if x not in STOP_WORDS)
        bw = set(x for x in b if x not in STOP_WORDS)
        astop = set(x for x in a if x in STOP_WORDS)
        bstop = set(x for x in b if x in STOP_WORDS)
        cw = len(aw.intersection(bw))
        cs = len(astop.intersection(bstop))
        ct = len(sa.intersection(sb))
        safe = 0.0001

        features.extend([
            cw / (min(len(aw), len(bw)) + safe),
            cw / (max(len(aw), len(bw)) + safe),
            cs / (min(len(astop), len(bstop)) + safe),
            cs / (max(len(astop), len(bstop)) + safe),
            ct / (min(len(a), len(b)) + safe),
            ct / (max(len(a), len(b)) + safe),
            int(a[-1] == b[-1]),
            int(a[0] == b[0])
        ])

    # 3 length features
    if not a or not b:
        features.extend([0.0] * 3)
    else:
        substrings = list(distance.lcsubstrings(q1, q2))
        longest = len(substrings[0]) if substrings else 0
        features.extend([
            abs(len(a) - len(b)),
            (len(a) + len(b)) / 2,
            longest / (min(len(q1), len(q2)) + 1)
        ])

    # 4 fuzzy features
    features.extend([
        fuzz.QRatio(q1, q2),
        fuzz.partial_ratio(q1, q2),
        fuzz.token_sort_ratio(q1, q2),
        fuzz.token_set_ratio(q1, q2)
    ])

    return features


st.title("Duplicate Question Pair Detection")
st.write("Enter two questions to check whether they are duplicates.")

q1 = st.text_input("Enter First Question")
q2 = st.text_input("Enter Second Question")

if st.button("Check Duplicate"):
    if q1.strip() and q2.strip():
        p1 = preprocess(q1)
        p2 = preprocess(q2)

        extra = make_features(p1, p2)

        # Separate BOW vectors for each question
        v1 = cv.transform([p1])
        v2 = cv.transform([p2])

        # Expected order: 22 engineered + 3000 + 3000 BOW
        features = hstack([
            csr_matrix(np.array(extra).reshape(1, -1)),
            v1,
            v2
        ]).tocsr()

        if features.shape[1] != model.n_features_in_:
            st.error(
                f"Feature mismatch: app created {features.shape[1]}, "
                f"model expects {model.n_features_in_}. "
                "Training feature order must be checked."
            )
        else:
            prediction = model.predict(features)
            if prediction[0] == 1:
                st.success("Duplicate Questions")
            else:
                st.info("Not Duplicate Questions")
    else:
        st.warning("Please enter both questions.")

