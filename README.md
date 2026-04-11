#  LSTM-Based Text Prediction System

> **LAB ASSIGNMENT 5** — LSTM-Based AI Agent for Sequence Prediction  
> **Task**: Text Prediction (Next Word)  
> **Deployment**: FastAPI  
**Name**  **PRN**
Sudarshan Khatal 202402070016
Prathamesh Gavhane 202402070024
Suraj Konda 202402070025
Gangotari Kompalwar 202402070019 

---

##  Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Declaration](#dataset-declaration)
3. [LSTM Mathematical Model](#lstm-mathematical-model)
4. [Project Structure](#project-structure)
5. [Setup & Installation](#setup--installation)
6. [Running the Model (Colab)](#running-the-model-colab)
7. [FastAPI Deployment](#fastapi-deployment)
8. [API Endpoints](#api-endpoints)
9. [Testing](#testing)
10. [Sample Outputs](#sample-outputs)
11. [AI Tool Acknowledgement](#ai-tool-acknowledgement)
12. [Group Contributions](#group-contributions)

---

##  Project Overview

This project implements a **complete end-to-end AI system** for **next word prediction** using LSTM (Long Short-Term Memory) neural networks, deployed via a **FastAPI REST API**.

### System Architecture

```
Wikipedia API ──► Data Preprocessing ──► LSTM Model Training
                                               │
                                               ▼
User Input ──► FastAPI /predict ──► Tokenize & Pad ──► LSTM Forward Pass ──► Top-K Words
```

### Key Features
- **Wikipedia API** dataset collection (20+ topics)
- **Bidirectional LSTM** with Embedding layer
- **Temperature sampling** for diverse text generation
- **FastAPI** deployment with Swagger UI
- **Top-K predictions** with confidence scores

---

##  Dataset Declaration

| Field              | Details                                                     |
|--------------------|-------------------------------------------------------------|
| **Dataset Name**   | Wikipedia Article Summaries                                 |
| **API Used**       | Wikipedia REST API                                          |
| **Source URL**     | `https://en.wikipedia.org/api/rest_v1/page/summary/{topic}`|
| **Topics Covered** | AI, ML, Deep Learning, NLP, Computer Science, Physics, etc.|
| **Approx. Size**   | ~25,000 words after cleaning                                |
| **License**        | Creative Commons Attribution-ShareAlike 3.0                 |

### Preprocessing Steps
1. **Fetch** article summaries via Wikipedia REST API
2. **Lowercase** all text
3. **Remove** URLs, citation markers `[1]`, `[2]`, etc.
4. **Remove** non-alphabetic characters
5. **Collapse** multiple whitespace into single spaces
6. **Tokenize** using Keras `Tokenizer` (vocab cap: 5000)
7. **Generate** overlapping input-output sequence pairs (window=10)
8. **Pad** sequences to fixed length
9. **Split** into train (85%) / validation (15%)

---

##  LSTM Mathematical Model

This section is **mandatory for presentation**.

### Gates & State Equations

At each time-step **t**, LSTM receives:
- **xₜ** → current input (word embedding vector)
- **h_{t-1}** → previous hidden state (short-term memory)
- **C_{t-1}** → previous cell state (long-term memory)

---

### 1️ Forget Gate
```
fₜ = σ(Wf · [h_{t-1}, xₜ] + bf)
```
- **Purpose**: Decides what to *discard* from the previous cell state
- **Activation**: Sigmoid σ → output in range [0, 1]
- `0` = completely forget, `1` = completely retain
- Example: When predicting next word in a new sentence, forget the previous sentence's context

---

### 2️ Input Gate + Candidate
```
iₜ = σ(Wi · [h_{t-1}, xₜ] + bi)       ← Input Gate
C̃ₜ = tanh(Wc · [h_{t-1}, xₜ] + bc)   ← Candidate Cell State
```
- **iₜ Purpose**: Decides *what new info* to store
- **C̃ₜ Purpose**: Creates candidate values to potentially add
- **Activation**: Sigmoid for gate, tanh for candidate values

---

### 3️ Cell State Update
```
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ
```
- **⊙** = element-wise (Hadamard) product
- **Purpose**: Long-term memory — selectively forget old + selectively add new
- **Key insight**: Additive update prevents vanishing gradient!

---

### 4️ Output Gate
```
oₜ = σ(Wo · [h_{t-1}, xₜ] + bo)
```
- **Purpose**: Controls what part of Cₜ gets exposed as output

---

### 5️ Hidden State
```
hₜ = oₜ ⊙ tanh(Cₜ)
```
- **Purpose**: Short-term memory, passed to next time-step and output layer

---

### 6️ Final Prediction
```
ŷ = softmax(Wd · hₜ + bd)
```
- **Output**: Probability distribution over entire vocabulary
- **Prediction**: argmax(ŷ) = most likely next word

---

### Why LSTM Beats Vanilla RNN

| Problem               | RNN         | LSTM                        |
|-----------------------|-------------|------------------------------|
| Vanishing Gradient    | ❌ Severe   | ✅ Solved via additive Cₜ   |
| Long-range dependency | ❌ Fails     | ✅ Cell state persists       |
| Selective memory      | ❌ No gates  | ✅ 3 learnable gates         |

---

##  Project Structure

```
lstm-text-prediction/
├── lstm_text_prediction_colab.py   # Main Colab notebook code
├── app.py                          # FastAPI deployment
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── saved_model/
│   ├── lstm_text_model.keras       # Trained model weights
│   ├── tokenizer.json              # Fitted tokenizer
│   └── config.json                 # Model configuration
└── training_history.png            # Loss/accuracy plot
```

---

##  Setup & Installation

### Prerequisites
- Python 3.9+
- pip

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

##  Running the Model (Colab)

1. Upload `lstm_text_prediction_colab.py` to Google Colab
2. Run all cells sequentially
3. The script will:
   - Fetch data from Wikipedia API
   - Preprocess and tokenize text
   - Train LSTM model
   - Show predictions and generated sentences
   - Save model to `saved_model/`

**Google Colab Link**: [Add your Colab link here]

---

##  FastAPI Deployment

### Step 1: Ensure saved model exists
```bash
# After running Colab notebook, download saved_model/ folder
ls saved_model/
# lstm_text_model.keras  tokenizer.json  config.json
```

### Step 2: Start the API server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Step 3: Open Swagger UI
Navigate to: **http://localhost:8000/docs**

---

##  API Endpoints

### `GET /health`
Health check — verify API is running.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "vocab_size": 4821,
  "sequence_length": 10,
  "load_time": "2026-04-10 14:32:11",
  "timestamp": "2026-04-10 14:45:00"
}
```

---

### `POST /predict`
Predict top-K next words for a seed text.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"seed_text": "artificial intelligence is", "top_k": 5}'
```

**Response:**
```json
{
  "seed_text": "artificial intelligence is",
  "predictions": [
    {"word": "used",    "probability": 0.2341, "rank": 1},
    {"word": "a",       "probability": 0.1823, "rank": 2},
    {"word": "the",     "probability": 0.1204, "rank": 3},
    {"word": "based",   "probability": 0.0934, "rank": 4},
    {"word": "applied", "probability": 0.0712, "rank": 5}
  ],
  "top_word": "used",
  "top_prob": 0.2341,
  "latency_ms": 42.3
}
```

---

### `POST /generate`
Generate a multi-word sentence from seed text.

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"seed_text": "machine learning", "num_words": 10, "temperature": 0.8}'
```

**Response:**
```json
{
  "seed_text": "machine learning",
  "generated_text": "machine learning algorithms are used to process large amounts of data efficiently",
  "num_words": 10,
  "temperature": 0.8,
  "latency_ms": 187.5
}
```

---

### `GET /model-info`
Returns model metadata and architecture details.

---

##  Testing

### Using Swagger UI (Recommended)
1. Open `http://localhost:8000/docs`
2. Click on an endpoint → "Try it out"
3. Enter request body → "Execute"
4. View response and status code

### Using Postman
- Import the API URL: `http://localhost:8000`
- Create POST request to `/predict`
- Set body type to `raw → JSON`
- Enter: `{"seed_text": "deep learning", "top_k": 5}`

### Using cURL
```bash
# Health check
curl http://localhost:8000/health

# Predict next word
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"seed_text": "neural networks are", "top_k": 3}'

# Generate sentence
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"seed_text": "quantum computing", "num_words": 8, "temperature": 0.7}'
```

---

##  Sample Outputs

### Training Results
```
Epoch 25/30
Training Accuracy  : 78.4%
Validation Accuracy: 62.1%
Validation Loss    : 2.3847
```

### Prediction Examples

| Seed Text                  | Top Prediction | Probability |
|----------------------------|----------------|-------------|
| "artificial intelligence"  | "is"           | 0.2341      |
| "machine learning"         | "algorithms"   | 0.1923      |
| "deep learning neural"     | "networks"     | 0.3102      |
| "the human brain"          | "is"           | 0.2841      |
| "quantum computing can"    | "be"           | 0.2654      |

### Generated Sentences

**Seed**: `"deep learning neural"`  
**Output**: `"deep learning neural networks are used to analyze complex patterns in data"`

**Seed**: `"artificial intelligence is"`  
**Output**: `"artificial intelligence is used in many fields including medicine and robotics"`

---

##  AI Tool Acknowledgement

As required by the assignment's academic integrity policy:

| Tool | Purpose | Sections Used |
|------|---------|---------------|
| **Claude (Anthropic)** | Code structuring, docstring writing, README generation | Code comments, API endpoint documentation, README |

All model architecture decisions, training logic, and mathematical formulations were understood and verified by the group members.

---

##  Group Contributions

| Member | PRN No. | Contributions |
|--------|----------|---------------|
| Member 1 | 202402070016 | Dataset collection, Wikipedia API integration, preprocessing |
| Member 2 | 202402070024 | LSTM model design, training, hyperparameter tuning |
| Member 3 | 202402070025 | FastAPI deployment, endpoint design, testing |
| Member 4 | 202402070019 | Documentation, README, presentation preparation |

> **Note**: All members participated in explaining the LSTM mathematical model during presentation.

---

##  References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [TensorFlow/Keras Docs](https://www.tensorflow.org/api_docs)
- [Keras LSTM Text Generation](https://keras.io/examples/nlp/text_generation/)
- [Wikipedia REST API](https://en.wikipedia.org/api/rest_v1/)
- [Anthropic Agent AI Course](https://www.anthropic.com/education)

---

**GitHub Repository**: []  
**Colab Notebook**: [https://colab.research.google.com/drive/1_pDIWbHxMPFCGYwPFQ1LYq-CoY3g99IQ?usp=sharing]
