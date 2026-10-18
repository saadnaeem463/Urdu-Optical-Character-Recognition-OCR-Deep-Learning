import html
import json
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from ocr_lib import CRNN, read_image

APP_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Urdu OCR Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load():
    with (APP_DIR / "chars.json").open(encoding="utf-8") as chars_file:
        chars = json.load(chars_file)
    idx2char = {i + 1: c for i, c in enumerate(chars)}
    model = CRNN(len(chars) + 1)
    model.load_state_dict(torch.load(APP_DIR / "crnn_best.pt", map_location="cpu"))
    model.eval()
    return model, idx2char

st.markdown(
    """
    <style>
    :root {
        --ink: #172b2a;
        --muted: #687c78;
        --green: #087e70;
        --green-dark: #075d55;
        --line: #dce9e5;
        --paper: #f4f8f6;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 8% 2%, rgba(166, 226, 210, .27), transparent 32rem),
            radial-gradient(ellipse at 94% 14%, rgba(205, 230, 218, .28), transparent 28rem),
            var(--paper);
        color: var(--ink);
    }
    .block-container {
        max-width: 1120px;
        padding: 2.1rem 2rem 4rem;
    }
    header[data-testid="stHeader"] { background: transparent; }
    .hero {
        position: relative;
        overflow: hidden;
        padding: clamp(1.7rem, 5vw, 3.5rem);
        border: 1px solid rgba(255,255,255,.72);
        border-radius: 28px;
        background: linear-gradient(118deg, #103c38 0%, #0a665c 57%, #14877a 100%);
        box-shadow: 0 22px 55px rgba(18, 74, 66, .15);
        color: #fff;
    }
    .hero:after {
        position: absolute;
        top: -110px;
        right: -64px;
        width: 330px;
        height: 330px;
        border: 1px solid rgba(255,255,255,.13);
        border-radius: 50%;
        box-shadow: 0 0 0 34px rgba(255,255,255,.035), 0 0 0 72px rgba(255,255,255,.025);
        content: "";
        pointer-events: none;
    }
    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: .55rem;
        margin-bottom: 1.15rem;
        color: #c7eee3;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .13em;
        text-transform: uppercase;
    }
    .hero-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #6be0b8;
        box-shadow: 0 0 14px rgba(107,224,184,.75);
    }
    .hero h1 {
        max-width: 700px;
        margin: 0;
        color: #fff;
        font-size: clamp(2.25rem, 6vw, 4.2rem);
        font-weight: 750;
        letter-spacing: -.055em;
        line-height: 1.02;
    }
    .hero p {
        max-width: 610px;
        margin: 1.15rem 0 0;
        color: rgba(239,255,250,.82);
        font-size: clamp(.98rem, 2vw, 1.12rem);
        line-height: 1.7;
    }
    .section-heading {
        margin: 2.15rem 0 .35rem;
        color: var(--ink);
        font-size: 1.3rem;
        font-weight: 750;
        letter-spacing: -.025em;
    }
    .section-copy {
        margin: 0 0 1rem;
        color: var(--muted);
        font-size: .96rem;
        line-height: 1.6;
    }
    [data-testid="stFileUploader"] {
        padding: .65rem;
        border: 1px solid var(--line);
        border-radius: 20px;
        background: rgba(255,255,255,.78);
        box-shadow: 0 10px 30px rgba(31, 83, 72, .055);
    }
    [data-testid="stFileUploaderDropzone"] {
        min-height: 145px;
        border: 1.5px dashed #a9c9bf;
        border-radius: 14px;
        background: #f8fcfa;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--green);
        background: #f0faf6;
    }
    [data-testid="stFileUploaderDropzone"] button {
        border: 1px solid #c8e1d8;
        border-radius: 10px;
        color: var(--green-dark);
        background: #fff;
        font-weight: 650;
    }
    [data-testid="stFileUploaderDropzone"] small { color: var(--muted); }
    [data-testid="stImage"] img {
        border: 1px solid var(--line);
        border-radius: 16px;
        background: #fff;
    }
    [data-testid="stCheckbox"] label {
        color: var(--ink);
        font-weight: 600;
    }
    .info-card {
        height: 100%;
        padding: 1.15rem 1.25rem;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: rgba(255,255,255,.78);
    }
    .info-label {
        margin-bottom: .35rem;
        color: var(--muted);
        font-size: .75rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
    }
    .info-value {
        color: var(--ink);
        font-size: 1rem;
        font-weight: 700;
        overflow-wrap: anywhere;
    }
    .result-card {
        margin: .7rem 0;
        padding: 1rem 1.2rem;
        border: 1px solid #d6e8e1;
        border-left: 4px solid #0a8878;
        border-radius: 14px;
        background: #fff;
        box-shadow: 0 8px 24px rgba(31, 83, 72, .05);
    }
    .result-label {
        margin-bottom: .55rem;
        color: var(--muted);
        font-family: sans-serif;
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
    }
    .urdu-text {
        color: #163c37;
        direction: rtl;
        font-family: "Noto Naskh Arabic", "Noto Sans Arabic", "Segoe UI", sans-serif;
        font-size: clamp(1.45rem, 3.4vw, 2rem);
        line-height: 1.8;
        overflow-wrap: anywhere;
        text-align: right;
        unicode-bidi: plaintext;
    }
    .empty-state {
        margin-top: 1.25rem;
        padding: 2.4rem 1.5rem;
        border: 1px dashed #c5dcd3;
        border-radius: 20px;
        background: rgba(255,255,255,.58);
        color: var(--muted);
        text-align: center;
    }
    .empty-state strong {
        display: block;
        margin-bottom: .4rem;
        color: var(--ink);
        font-size: 1.05rem;
    }
    [data-testid="stAlert"] { border-radius: 14px; }
    @media (max-width: 640px) {
        .block-container { padding: 1rem 1rem 2.5rem; }
        .hero { border-radius: 21px; }
        .hero:after { right: -210px; }
        .section-heading { margin-top: 1.7rem; }
        [data-testid="stFileUploader"] { padding: .45rem; }
        [data-testid="stFileUploaderDropzone"] { min-height: 125px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
      <div class="hero-kicker"><span class="hero-dot"></span> Urdu OCR · CRNN + CTC</div>
      <h1>Read Urdu,<br>effortlessly.</h1>
      <p>Turn a photo or scan of printed Urdu into clean, readable text. Upload an image to get started.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-heading">Start with an image</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">Best results come from clear, high-contrast printed text. Single lines and multi-line pages are both supported.</p>',
    unsafe_allow_html=True,
)
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
    help="PNG or JPG · Printed Urdu works best; handwriting is not supported.",
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="empty-state">
          <strong>Your result will appear here</strong>
          Upload a PNG or JPG to preview it and recognize the printed Urdu text.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    try:
        with Image.open(uploaded_file) as opened_image:
            image = opened_image.copy()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        st.error(f"We couldn't open that image. Please try a valid PNG or JPG. ({error})")
    else:
        st.markdown('<div class="section-heading">Review &amp; recognize</div>', unsafe_allow_html=True)
        preview_column, settings_column = st.columns([1.35, 1], gap="large")
        with preview_column:
            st.image(image, caption="Uploaded image", use_container_width=True)
        with settings_column:
            st.markdown(
                f"""
                <div class="info-card">
                  <div class="info-label">Image details</div>
                  <div class="info-value">{image.width:,} × {image.height:,} px</div>
                  <div class="info-label" style="margin-top:1rem">File</div>
                  <div class="info-value">{html.escape(uploaded_file.name)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            split_lines = st.checkbox("Automatically split multi-line pages", value=True)
            st.caption("Turn this off when your image contains just one text line.")

        st.markdown('<div class="section-heading">Recognized text</div>', unsafe_allow_html=True)
        with st.spinner("Reading the Urdu text…"):
            model, idx2char = load()
            lines = read_image(image, model, idx2char, "cpu", split=split_lines)

        if not lines:
            st.info("No usable text lines were found. Try a clearer image with larger printed text.")
        else:
            for index, text in enumerate(lines, start=1):
                safe_text = html.escape(text) if text else "No text detected on this line."
                st.markdown(
                    f"""
                    <div class="result-card">
                      <div class="result-label">Line {index}</div>
                      <div class="urdu-text" dir="rtl">{safe_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.caption("Note: handwriting is not supported by this model.")
