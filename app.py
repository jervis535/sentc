
import os
import re

import joblib
import numpy as np
import streamlit as st
import tabm
import torch


# ---- Configuration ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CHOICES = ["TabM", "TabR", "TabPFN"]


# ---- TabPFN token ----
if "TABPFN_TOKEN" not in os.environ:
    try:
        os.environ["TABPFN_TOKEN"] = st.secrets["TABPFN_TOKEN"]
    except Exception:
        pass


# ---- Load shared preprocessing objects ----
@st.cache_resource
def load_shared():
    vectorizer = joblib.load("tfidf.pkl")
    svd = joblib.load("svd.pkl")
    le = joblib.load("label_encoder.pkl")

    return vectorizer, svd, le


# ---- Load TabM ----
@st.cache_resource
def load_tabm():

    ckpt = torch.load(
        "tabm.pth",
        map_location=DEVICE,
        weights_only=False
    )

    model = tabm.TabM(
        n_num_features=ckpt["n_num_features"],
        d_out=ckpt["n_classes"],
        n_blocks=3,
        d_block=256,
        dropout=0.1,
        k=ckpt["k"],
        arch_type="tabm",
        start_scaling_init="random-signs",
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model


# ---- Load TabR ----
@st.cache_resource
def load_tabr():
    return joblib.load("tabr.pkl")


# ---- Load TabPFN ----
@st.cache_resource
def load_tabpfn():
    return joblib.load("tabpfn.pkl")


# ---- Text preprocessing ----
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---- Prediction ----
def predict_proba(model_name: str, features: np.ndarray):

    if model_name == "TabM":

        model = load_tabm()

        x = torch.tensor(
            features,
            dtype=torch.float32,
            device=DEVICE
        )

        with torch.no_grad():
            out = model(x_num=x)
            probs = torch.softmax(out, dim=-1).mean(dim=1)

        return probs.cpu().numpy()[0]

    elif model_name == "TabR":
        model = load_tabr()

    elif model_name == "TabPFN":
        model = load_tabpfn()

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return np.asarray(model.predict_proba(features))[0]


# ---- Streamlit UI ----
st.set_page_config(
    page_title="Sentiment Classifier",
)

st.title("Sentiment Classifier")

st.write(
    'Enter a review and pick a model then click "predict"'
)

model_name = st.selectbox(
    "Model",
    MODEL_CHOICES
)

text = st.text_area(
    "Review text",
    height=150,
    placeholder="Type or paste a review here..."
)


if st.button("Predict", type="primary"):

    cleaned = clean_text(text)

    if not cleaned:

        st.warning(
            "Please enter some text "
            "(letters only count after cleaning)."
        )

    else:

        try:

            with st.spinner(f"Running {model_name}..."):

                vectorizer, svd, le = load_shared()

                features = svd.transform(
                    vectorizer.transform([cleaned])
                )

                probs = predict_proba(
                    model_name,
                    features
                )

            classes = [str(c) for c in le.classes_]

            pred = classes[int(np.argmax(probs))]

            st.subheader(
                f"Prediction: {pred.capitalize()}"
            )

        except Exception as e:

            st.error(f"Prediction failed: {e}")