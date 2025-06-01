# 🧠 ASL (American Sign Language) Detection with Python & MediaPipe

This project is a real-time ASL letter detector using a webcam and hand tracking via [MediaPipe](https://google.github.io/mediapipe/). It captures hand gestures, converts them into letters (A-Z), and builds full words using smoothing + spell correction.

---

## 🏗️ Features

- Real-time letter detection using hand landmarks
- Word building from letter predictions
- Fuzzy spelling correction using FuzzyWuzzy + TextBlob
- Easily extensible to numbers or custom gestures
- Training pipeline with label encoding

---

## 🖥️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/jainilDesai/ASL_Alphabet_Detector.git
cd ASL_Alphabet_Detector
```

### 2. Create Python Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# OR
.venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📸 Collecting Data

Use `collect_data.py` to record hand landmarks for each letter.

### Step-by-Step:

1. Open `collect_data.py`
2. Change the `LABEL` variable to the character you're recording:
   ```python
   LABEL = 'A'  # Or 'B', 'C', ..., '9'
   ```
3. Run:
   ```bash
   python collect_data.py
   ```

🔁 Repeat for each character/class.

---

## 🧠 Training the Model

### Option A: Basic

```bash
python train_model.py
```

### Option B: Improved (Recommended)

```bash
python train_model_v2.py
```

This version:

- Trains a better model with validation and early stopping
- Saves to `models/asl_model_v2.h5`
- Saves class labels to `models/labels.npy`

---

## 🤖 Predict in Real-Time

Once trained, run:

```bash
python predict.py
```

**Controls:**

- Webcam opens
- Predictions display live
- Press `Q` to quit

Output shows:

- Predicted letter + confidence
- Current word being formed
- Final sentence (auto-corrected)

---

## 📁 Project Structure

```
├── collect_data.py
├── train_model.py
├── train_model_v2.py
├── predict.py
├── models/
│   ├── asl_model.h5
│   ├── asl_model_v2.h5
│   └── labels.npy
├── data/
│   └── A/
│       ├── 0.npy
│       └── ...
├── requirements.txt
└── README.md
```

---

## 🚀 Future Improvements

- Add support for digits and custom signs
- Multi-hand detection
- Text-to-speech integration
- Deploy as a web app (Flask, Streamlit)
- Add GUI or visual feedback (bounding boxes, timers, etc.)

---

## 🧠 Credits & Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) – Hand detection
- [TensorFlow](https://www.tensorflow.org/) – Neural network training
- [TextBlob](https://textblob.readthedocs.io/en/dev/) – Spell correction
- [FuzzyWuzzy](https://github.com/seatgeek/fuzzywuzzy) – Fuzzy string matching

---

## 👨‍💻 Author

**Jainil Desai** – [GitHub](https://github.com/jainilDesai)  
Built with love, late nights, and a lot of Ctrl+Z ❤️
