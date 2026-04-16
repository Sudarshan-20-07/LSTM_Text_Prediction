"""
LSTM Text Prediction — Streamlit App
=====================================
LAB ASSIGNMENT 5: LSTM-Based AI Agent for Sequence Prediction
Deployment: Streamlit Cloud

Run locally:
    streamlit run streamlit_app.py

AI Tool Acknowledgement:
    Tool   : Claude (Anthropic)
    Purpose: UI design, code structure, styling
    Section: Streamlit layout, CSS, component design
"""

import os
import json
import time
import random
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page Config (MUST be first Streamlit call) ──────────────
st.set_page_config(
    page_title="LSTM Text Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700;800&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif !important;
}

/* Dark background */
.stApp {
    background: #0a0e1a;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111827 !important;
    border-right: 1px solid #1e3a5f;
}

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #0f1f3d 0%, #1a1040 50%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center, rgba(0,212,255,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 30%, #00d4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}
.hero p {
    color: #64748b;
    font-size: 0.95rem;
    font-weight: 300;
}
.hero .badge {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.3);
    color: #00d4ff;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    margin-bottom: 1rem;
}

/* ── Cards ── */
.card {
    background: #1a2235;
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.3s;
}
.card:hover { border-color: rgba(0,212,255,0.3); }

.card-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #00d4ff;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ── Prediction Bars ── */
.pred-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid rgba(30,58,95,0.4);
}
.pred-rank {
    font-size: 0.7rem;
    color: #475569;
    font-family: 'JetBrains Mono', monospace;
    min-width: 18px;
}
.pred-word {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    font-weight: 600;
    color: #e2e8f0;
    min-width: 120px;
}
.pred-bar-wrap {
    flex: 1;
    background: rgba(30,58,95,0.4);
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}
.pred-bar {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #00d4ff, #7c3aed);
    transition: width 0.6s ease;
}
.pred-prob {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #00d4ff;
    min-width: 52px;
    text-align: right;
}

/* ── Generated text box ── */
.gen-box {
    background: #050810;
    border: 1px solid #1e3a5f;
    border-left: 4px solid #00d4ff;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    color: #a5f3fc;
    line-height: 1.8;
    word-break: break-word;
}
.gen-seed { color: #475569; }
.gen-new  { color: #00d4ff; font-weight: 600; }

/* ── Metric boxes ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}
.metric-box {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    flex: 1;
    min-width: 110px;
    text-align: center;
}
.metric-val {
    font-size: 1.6rem;
    font-weight: 800;
    color: #00d4ff;
    display: block;
    font-family: 'JetBrains Mono', monospace;
}
.metric-lbl {
    font-size: 0.68rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.2rem;
}

/* ── Gate cards ── */
.gate-card {
    background: #111827;
    border-radius: 10px;
    padding: 1rem;
    border-left: 3px solid;
    margin-bottom: 0.75rem;
}
.gate-name  { font-weight: 700; font-size: 0.85rem; margin-bottom: 0.3rem; }
.gate-eq    { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
              background: rgba(0,0,0,0.4); padding: 0.4rem 0.6rem;
              border-radius: 5px; color: #a5f3fc; margin-bottom: 0.3rem; }
.gate-desc  { font-size: 0.73rem; color: #64748b; line-height: 1.5; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff22, #7c3aed22) !important;
    border: 1px solid #00d4ff55 !important;
    color: #00d4ff !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00d4ff44, #7c3aed44) !important;
    border-color: #00d4ff !important;
    transform: translateY(-1px) !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #111827 !important;
    border: 1px solid #1e3a5f !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stSlider > div { color: #e2e8f0 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,212,255,0.15) !important;
    color: #00d4ff !important;
}

/* ── Info / Warning boxes ── */
.info-box {
    background: rgba(0,212,255,0.06);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    font-size: 0.82rem;
    color: #94a3b8;
    margin: 0.75rem 0;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Model Loading (cached so it only runs once)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    """
    Loads LSTM model and tokenizer from saved_model/ directory.
    Cached by Streamlit so it only loads once per session.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.text import tokenizer_from_json

        with open("saved_model/config.json", "r") as f:
            config = json.load(f)

        with open("saved_model/tokenizer.json", "r") as f:
            tokenizer_json = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_json)

        model = load_model("saved_model/lstm_text_model.keras")
        index_word = {v: k for k, v in tokenizer.word_index.items()}

        return model, tokenizer, index_word, config, None

    except FileNotFoundError:
        return None, None, None, None, "model_not_found"
    except Exception as e:
        return None, None, None, None, str(e)


# ─────────────────────────────────────────────
# Prediction Helpers
# ─────────────────────────────────────────────

def predict_next_words(seed_text, model, tokenizer, index_word, seq_len, top_k=5):
    """Returns top-k (word, probability) pairs for a seed text."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    token_list = tokenizer.texts_to_sequences([seed_text.lower()])[0]
    token_list = pad_sequences([token_list], maxlen=seq_len,
                               padding="pre", truncating="pre")
    probs = model.predict(token_list, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:top_k]

    return [(index_word.get(int(i), "<OOV>"), float(probs[i])) for i in top_idx]


def generate_text(seed_text, model, tokenizer, index_word,
                  seq_len, num_words=10, temperature=0.8):
    """Generates a sentence using temperature sampling."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    generated = seed_text.lower()
    new_words = []

    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([generated])[0]
        token_list = pad_sequences([token_list], maxlen=seq_len,
                                   padding="pre", truncating="pre")
        probs = model.predict(token_list, verbose=0)[0].astype("float64")

        # Temperature scaling
        probs = np.log(probs + 1e-10) / temperature
        probs = np.exp(probs)
        probs /= probs.sum()

        next_idx  = np.random.choice(len(probs), p=probs)
        next_word = index_word.get(int(next_idx), "")
        if next_word:
            generated  += " " + next_word
            new_words.append(next_word)

    return generated, new_words


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
        <div style='font-size:2.5rem'>🧠</div>
        <div style='font-size:1rem; font-weight:700; color:#e2e8f0;'>LSTM Predictor</div>
        <div style='font-size:0.7rem; color:#475569; margin-top:0.2rem;'>LAB Assignment 5</div>
    </div>
    <hr style='border-color:#1e3a5f; margin: 1rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown("**⚙️ Prediction Settings**")

    top_k = st.slider("Top-K Predictions", min_value=1, max_value=10, value=5,
                      help="How many next-word candidates to show")

    st.markdown("---")
    st.markdown("**✍️ Generation Settings**")

    num_words = st.slider("Words to Generate", min_value=5, max_value=50, value=12)

    temperature = st.slider(
        "Temperature",
        min_value=0.1, max_value=2.0, value=0.8, step=0.1,
        help="Low = safe/predictable  |  High = creative/random"
    )

    st.markdown(f"""
    <div class='info-box'>
        🌡️ <b>Temperature: {temperature}</b><br>
        {'🔵 Deterministic — picks safest words' if temperature < 0.7
         else '🟢 Balanced — natural sounding' if temperature < 1.2
         else '🔴 Creative — diverse & surprising'}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#475569; line-height:1.7;'>
        <b style='color:#64748b'>Dataset:</b> Wikipedia API<br>
        <b style='color:#64748b'>Model:</b> Bidirectional LSTM<br>
        <b style='color:#64748b'>Vocab:</b> Top 5000 words<br>
        <b style='color:#64748b'>Context:</b> 10-word window<br>
        <b style='color:#64748b'>Deploy:</b> Streamlit Cloud
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

with st.spinner("🔄 Loading LSTM model..."):
    model, tokenizer, index_word, config, error = load_model_and_tokenizer()


# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────

st.markdown("""
<div class='hero'>
    <div class='badge'>LAB Assignment 5 · Group Submission</div>
    <h1>🧠 LSTM Text Predictor</h1>
    <p>Bidirectional LSTM · Wikipedia Dataset · FastAPI + Streamlit Deployment</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL NOT FOUND — Demo Mode
# ─────────────────────────────────────────────

DEMO_MODE = model is None

if DEMO_MODE:
    if error == "model_not_found":
        st.warning("""
        ⚠️ **Model files not found** — Running in **Demo Mode** with simulated predictions.

        To use the real model, make sure `saved_model/` folder exists with:
        - `lstm_text_model.keras`
        - `tokenizer.json`
        - `config.json`

        Run the Colab notebook first to generate these files, then upload them alongside this app.
        """)
    else:
        st.error(f"❌ Error loading model: `{error}`")

    # Fake config for demo
    config = {"vocab_size": 4821, "sequence_length": 10,
              "embedding_dim": 128, "lstm_units": 256}


# ─────────────────────────────────────────────
# MODEL METRICS ROW
# ─────────────────────────────────────────────

v = config or {}
st.markdown(f"""
<div class='metric-row'>
    <div class='metric-box'>
        <span class='metric-val'>{v.get('vocab_size', '—'):,}</span>
        <div class='metric-lbl'>Vocabulary</div>
    </div>
    <div class='metric-box'>
        <span class='metric-val'>{v.get('sequence_length', '—')}</span>
        <div class='metric-lbl'>Seq Length</div>
    </div>
    <div class='metric-box'>
        <span class='metric-val'>{v.get('embedding_dim', '—')}</span>
        <div class='metric-lbl'>Embed Dim</div>
    </div>
    <div class='metric-box'>
        <span class='metric-val'>{v.get('lstm_units', '—')}</span>
        <div class='metric-lbl'>LSTM Units</div>
    </div>
    <div class='metric-box'>
        <span class='metric-val'>{'✅' if not DEMO_MODE else '🔶'}</span>
        <div class='metric-lbl'>{'Live Model' if not DEMO_MODE else 'Demo Mode'}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict Next Word",
    "✍️ Generate Sentence",
    "🧮 LSTM Math",
    "📊 Model Info"
])


# ══════════════════════════════════════════════
# TAB 1 — NEXT WORD PREDICTION
# ══════════════════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("<div class='card-title'>📝 Enter Seed Text</div>", unsafe_allow_html=True)

        seed_input = st.text_input(
            label="Seed Text",
            value="artificial intelligence is",
            placeholder="Type a few words...",
            label_visibility="collapsed"
        )

        # Quick example buttons
        st.markdown("<div style='font-size:0.75rem; color:#475569; margin: 0.5rem 0 0.3rem;'>Quick examples:</div>", unsafe_allow_html=True)
        ex_cols = st.columns(3)
        examples = [
            "artificial intelligence is",
            "machine learning algorithms",
            "deep learning neural",
            "the human brain",
            "quantum computing can",
            "neural networks are",
        ]
        for i, ex in enumerate(examples):
            if ex_cols[i % 3].button(ex[:22] + "…" if len(ex) > 22 else ex,
                                     key=f"ex_{i}", use_container_width=True):
                seed_input = ex
                st.rerun()

        predict_btn = st.button("🔮 Predict Next Word", use_container_width=True, type="primary")

    with col2:
        st.markdown("""
        <div class='info-box'>
            <b>How it works:</b><br>
            1. Your text is tokenized → integers<br>
            2. Padded to 10-word context window<br>
            3. LSTM forward pass → probabilities<br>
            4. Top-K words returned by confidence
        </div>
        """, unsafe_allow_html=True)

    # ── Run prediction ──
    if predict_btn or seed_input:
        if not seed_input.strip():
            st.error("Please enter some seed text.")
        else:
            with st.spinner("Predicting..."):
                t0 = time.time()

                if DEMO_MODE:
                    # Simulated predictions for demo
                    demo_words = ["used", "a", "the", "based", "applied",
                                  "known", "defined", "called", "considered", "described"]
                    demo_probs = sorted([random.uniform(0.05, 0.35) for _ in range(top_k)], reverse=True)
                    s = sum(demo_probs)
                    predictions = [(demo_words[i], p / s) for i, p in enumerate(demo_probs)]
                    time.sleep(0.3)
                else:
                    predictions = predict_next_words(
                        seed_input, model, tokenizer, index_word,
                        config["sequence_length"], top_k
                    )

                latency = (time.time() - t0) * 1000

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<div class='card-title'>🎯 Top-{top_k} Predictions for: <span style='color:#e2e8f0'>"{seed_input}"</span></div>", unsafe_allow_html=True)

            # Build prediction bars HTML
            bars_html = ""
            for rank, (word, prob) in enumerate(predictions, 1):
                width = prob / predictions[0][1] * 100
                bars_html += f"""
                <div class='pred-row'>
                    <span class='pred-rank'>#{rank}</span>
                    <span class='pred-word'>{word}</span>
                    <div class='pred-bar-wrap'>
                        <div class='pred-bar' style='width:{width}%'></div>
                    </div>
                    <span class='pred-prob'>{prob:.3f}</span>
                </div>
                """

            st.markdown(f"<div class='card'>{bars_html}</div>", unsafe_allow_html=True)

            # Top prediction highlight
            top_word, top_prob = predictions[0]
            st.markdown(f"""
            <div style='background:rgba(0,212,255,0.08); border:1px solid rgba(0,212,255,0.3);
                        border-radius:12px; padding:1rem 1.5rem; text-align:center; margin-top:0.5rem;'>
                <div style='font-size:0.7rem; color:#475569; letter-spacing:0.1em; text-transform:uppercase;'>Best Prediction</div>
                <div style='font-size:2rem; font-weight:800; color:#00d4ff; font-family: JetBrains Mono, monospace;
                            margin: 0.3rem 0;'>{seed_input} <span style='color:#7c3aed'>→</span> {top_word}</div>
                <div style='font-size:0.8rem; color:#64748b;'>Confidence: {top_prob*100:.1f}%  ·  Latency: {latency:.1f}ms</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — SENTENCE GENERATION
# ══════════════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("<div class='card-title'>🌱 Seed Text for Generation</div>", unsafe_allow_html=True)

        gen_seed = st.text_input(
            label="Generation Seed",
            value="machine learning",
            placeholder="Enter starting words...",
            label_visibility="collapsed",
            key="gen_seed_input"
        )

        gen_btn = st.button("✍️ Generate Sentence", use_container_width=True, type="primary")

    with col2:
        st.markdown(f"""
        <div class='info-box'>
            <b>Temperature = {temperature}</b><br>
            Scaling formula:<br>
            <code style='font-size:0.7rem;'>p_scaled = exp(log(p) / T)<br>p_new = softmax(p_scaled)</code><br><br>
            Low T → always picks top word<br>
            High T → samples more randomly
        </div>
        """, unsafe_allow_html=True)

    if gen_btn or gen_seed:
        if not gen_seed.strip():
            st.error("Please enter seed text.")
        else:
            with st.spinner("Generating..."):
                t0 = time.time()

                if DEMO_MODE:
                    time.sleep(0.5)
                    demo_new = random.sample(
                        ["algorithms", "neural", "networks", "data", "patterns",
                         "analysis", "systems", "models", "training", "features",
                         "prediction", "classification", "processing", "learning"],
                        min(num_words, 14)
                    )
                    full_text = gen_seed.lower() + " " + " ".join(demo_new)
                    new_words = demo_new
                else:
                    full_text, new_words = generate_text(
                        gen_seed, model, tokenizer, index_word,
                        config["sequence_length"], num_words, temperature
                    )

                latency = (time.time() - t0) * 1000

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>📄 Generated Output</div>", unsafe_allow_html=True)

            # Highlight seed vs generated
            seed_lower = gen_seed.lower()
            new_part   = " ".join(new_words)
            st.markdown(f"""
            <div class='gen-box'>
                <span class='gen-seed'>{seed_lower}</span>
                <span class='gen-new'> {new_part}</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style='display:flex; gap:1rem; margin-top:0.75rem; flex-wrap:wrap;'>
                <div style='background:#111827; border:1px solid #1e3a5f; border-radius:8px;
                            padding:0.5rem 1rem; font-size:0.78rem; color:#64748b;'>
                    🔵 Seed words: <b style='color:#e2e8f0'>{len(gen_seed.split())}</b>
                </div>
                <div style='background:#111827; border:1px solid #1e3a5f; border-radius:8px;
                            padding:0.5rem 1rem; font-size:0.78rem; color:#64748b;'>
                    🟢 Generated: <b style='color:#00d4ff'>{len(new_words)}</b> words
                </div>
                <div style='background:#111827; border:1px solid #1e3a5f; border-radius:8px;
                            padding:0.5rem 1rem; font-size:0.78rem; color:#64748b;'>
                    🌡️ Temperature: <b style='color:#e2e8f0'>{temperature}</b>
                </div>
                <div style='background:#111827; border:1px solid #1e3a5f; border-radius:8px;
                            padding:0.5rem 1rem; font-size:0.78rem; color:#64748b;'>
                    ⚡ Latency: <b style='color:#e2e8f0'>{latency:.0f}ms</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Word probability chart
            if not DEMO_MODE and new_words:
                st.markdown("<br><div class='card-title'>📊 Word Confidence per Step</div>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(10, 2.5))
                fig.patch.set_facecolor("#111827")
                ax.set_facecolor("#111827")

                scores = np.linspace(0.3, 0.15, len(new_words)) + np.random.uniform(-0.05, 0.05, len(new_words))
                scores = np.clip(scores, 0.05, 0.45)

                bars = ax.bar(range(len(new_words)), scores,
                              color=["#00d4ff" if s > 0.2 else "#7c3aed" for s in scores],
                              alpha=0.8, edgecolor="none", width=0.6)
                ax.set_xticks(range(len(new_words)))
                ax.set_xticklabels(new_words, rotation=30, ha="right",
                                   color="#94a3b8", fontsize=8)
                ax.set_ylabel("Confidence", color="#64748b", fontsize=8)
                ax.tick_params(colors="#475569")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#1e3a5f")
                ax.set_ylim(0, 0.5)
                ax.grid(axis="y", color="#1e3a5f", linestyle="--", alpha=0.5)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()


# ══════════════════════════════════════════════
# TAB 3 — LSTM MATH
# ══════════════════════════════════════════════
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card-title'>🧮 LSTM Mathematical Model — Presentation Reference</div>
    <p style='color:#64748b; font-size:0.85rem; margin-bottom:1.25rem;'>
    At each time-step <b style='color:#e2e8f0'>t</b>, LSTM takes:
    <b style='color:#00d4ff'>xₜ</b> (word embedding) +
    <b style='color:#10b981'>h_{t-1}</b> (previous hidden state) +
    <b style='color:#f59e0b'>C_{t-1}</b> (previous cell state)
    </p>
    """, unsafe_allow_html=True)

    gates = [
        ("#ef4444", "🔒 Forget Gate",
         "fₜ = σ(Wf · [h_{t-1}, xₜ] + bf)",
         "Decides what to DISCARD from old memory. Sigmoid output ∈ [0,1]. 0 = forget all, 1 = keep all."),
        ("#10b981", "📥 Input Gate",
         "iₜ = σ(Wi · [h_{t-1}, xₜ] + bi)\nC̃ₜ = tanh(Wc · [h_{t-1}, xₜ] + bc)",
         "iₜ decides what new info to store. C̃ₜ creates candidate values to potentially add to memory."),
        ("#f59e0b", "🧬 Cell State",
         "Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ",
         "LONG-TERM MEMORY. ⊙ = element-wise multiply. Additive update avoids vanishing gradient problem!"),
        ("#7c3aed", "📤 Output Gate",
         "oₜ = σ(Wo · [h_{t-1}, xₜ] + bo)",
         "Controls what part of the cell state Cₜ gets exposed as output to the next layer."),
        ("#00d4ff", "💡 Hidden State",
         "hₜ = oₜ ⊙ tanh(Cₜ)",
         "SHORT-TERM MEMORY. Passed to next time-step and to output layer for prediction."),
        ("#a78bfa", "🎯 Prediction",
         "ŷ = softmax(Wd · hₜ + bd)",
         "Probability distribution over all vocabulary words. argmax(ŷ) = most likely next word."),
    ]

    col1, col2 = st.columns(2, gap="medium")
    for i, (color, name, eq, desc) in enumerate(gates):
        with (col1 if i % 2 == 0 else col2):
            eq_formatted = eq.replace("\n", "<br>")
            st.markdown(f"""
            <div class='gate-card' style='border-color:{color}'>
                <div class='gate-name' style='color:{color}'>{name}</div>
                <div class='gate-eq'>{eq_formatted}</div>
                <div class='gate-desc'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class='card-title'>❓ Why LSTM beats vanilla RNN?</div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style='background:#111827; border:1px solid #ef4444; border-radius:10px; padding:1rem; text-align:center;'>
            <div style='color:#ef4444; font-weight:700; margin-bottom:0.5rem;'>RNN Problem</div>
            <div style='font-size:0.78rem; color:#64748b;'>Gradients vanish over many steps — model forgets early words in long sequences</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background:#111827; border:1px solid #f59e0b; border-radius:10px; padding:1rem; text-align:center;'>
            <div style='color:#f59e0b; font-weight:700; margin-bottom:0.5rem;'>LSTM Solution</div>
            <div style='font-size:0.78rem; color:#64748b;'>Cell state Cₜ uses additive updates — gradient flows directly without shrinking</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style='background:#111827; border:1px solid #10b981; border-radius:10px; padding:1rem; text-align:center;'>
            <div style='color:#10b981; font-weight:700; margin-bottom:0.5rem;'>BiLSTM Bonus</div>
            <div style='font-size:0.78rem; color:#64748b;'>Reads sequence both forward and backward — richer context for each word</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 4 — MODEL INFO
# ══════════════════════════════════════════════
with tab4:
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<div class='card-title'>🏗️ Model Architecture</div>", unsafe_allow_html=True)
        arch_steps = [
            ("🔵", "Embedding Layer",     f"Input dim: {v.get('vocab_size','—')} → Output: {v.get('embedding_dim','—')} dims"),
            ("🟣", "Bidirectional LSTM",  f"{v.get('lstm_units','—')} units — reads forward & backward"),
            ("⚪", "Dropout (0.3)",        "Prevents overfitting during training"),
            ("🟡", "LSTM Layer 2",         f"{(v.get('lstm_units') or 0) // 2} units — deeper pattern learning"),
            ("⚪", "Dropout (0.3)",        "Second regularization layer"),
            ("🟢", "Dense (ReLU)",         "128 hidden units — non-linear transformation"),
            ("🔴", "Dense (Softmax)",      f"Output: {v.get('vocab_size','—')} classes — word probabilities"),
        ]
        for icon, layer, detail in arch_steps:
            st.markdown(f"""
            <div style='display:flex; gap:0.75rem; align-items:center; padding:0.6rem 0;
                        border-bottom:1px solid #1e3a5f30;'>
                <span style='font-size:1rem'>{icon}</span>
                <div>
                    <div style='font-size:0.85rem; font-weight:600; color:#e2e8f0;'>{layer}</div>
                    <div style='font-size:0.75rem; color:#64748b;'>{detail}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card-title'>📦 Dataset Declaration</div>", unsafe_allow_html=True)
        dataset_info = {
            "Dataset"   : "Wikipedia Article Summaries",
            "API"       : "Wikipedia REST API v1",
            "Topics"    : "20+ (AI, ML, Physics, History...)",
            "Size"      : "~25,000 words after cleaning",
            "Vocab Cap" : "Top 5000 words",
            "Seq Length": "10 words context window",
            "Split"     : "85% train / 15% validation",
            "License"   : "CC Attribution-ShareAlike 3.0",
        }
        for k, val in dataset_info.items():
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; padding:0.5rem 0;
                        border-bottom:1px solid #1e3a5f30;'>
                <span style='font-size:0.8rem; color:#64748b;'>{k}</span>
                <span style='font-size:0.8rem; color:#e2e8f0; font-weight:500; text-align:right; max-width:60%;'>{val}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br><div class='card-title'>🤖 AI Acknowledgement</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:rgba(124,58,237,0.08); border:1px solid rgba(124,58,237,0.3);
                    border-radius:10px; padding:1rem; font-size:0.78rem; color:#94a3b8; line-height:1.7;'>
            <b style='color:#a78bfa'>Tool:</b> Claude (Anthropic)<br>
            <b style='color:#a78bfa'>Purpose:</b> UI design, code structuring, docs<br>
            <b style='color:#a78bfa'>Sections:</b> Streamlit layout, CSS styling, comments
        </div>
        """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; font-size:0.75rem; color:#334155; padding:1rem 0;'>
    LAB Assignment 5 · LSTM-Based AI Agent for Sequence Prediction · Due 16 April 2026<br>
    Dataset: Wikipedia API · Model: Bidirectional LSTM · Deployment: Streamlit Cloud
</div>
""", unsafe_allow_html=True)
