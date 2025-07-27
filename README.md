<div align="center">

# 🧠 EmotionSense  
### A Hybrid Transformer-LSTM Based Emotion Predictor Built for 2025 Core Inductions  
**By Vihaan Kanwar**  
_“Understand the unsaid.”_


[![PyTorch](https://img.shields.io/badge/PyTorch-Enabled-red)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/)
[![Model Version](https://img.shields.io/badge/Version-1.0-green)](#)

</div>

---

## 🚀 Overview

**EmotionSense** is a  hybrid emotion prediction AI model designed for **natural language emotional understanding**, integrating:

- ⚙️ Transformer encoders (`RoBERTa-large`)
- 🧬 Sentence embeddings (`all-MiniLM-L6-v2`)
- 🧠 LSTM layers with attention pooling
- 🔗 Contextual keyword pattern matching
- 🔁 LLM-assisted ensemble predictions

It intelligently fuses deep learning and rule-based NLP signals to **detect 6 primary emotions**:
**Sadness, Joy, Love, Anger, Fear, Surprise.**

---

## 🧰 Technologies Used

| Layer             | Tech Stack                                     |
|------------------|------------------------------------------------|
|  Core Model     | PyTorch, HuggingFace Transformers              |
|  Sentence Embs | Sentence-Transformers (MiniLM-L6-v2)           |
|  Data Handling | NumPy, JSON, Regex NLP, Torch AMP              |
|  LLM Ensemble  | DistilRoBERTa Emotion Classifier via 🤗 Hub     |



## 🛠️ Run Locally

> 💡 Make sure you have Python 3.10+ and a working GPU (recommended) with CUDA support.

### 📦 Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/emotionsense.git
cd emotionsense
```


### 💾 Step 2: Install Libraries
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 transformers==4.39.3 sentence-transformers==2.2.2 numpy==1.26.4 flask==2.3.3 flask-cors==4.0.0 scikit-learn==1.3.2 flask==2.3.3 flask-cors==4.0.0 
```


### 🏃‍♂️ Step 3: Run train_model.py

 **💡 This will train the model and save it as "emotion_predictor_model.pth"**


### 🔄️ Step 4: Update the path in "api.py"

**In "api.py" update the path of emotion_predictor_model.pth at line 15**

### Final Step: Run the "api.py" and "index.html"

**After doing all steps, run the api.py. If the api.py starts successfuly, run the index.html!**

### 💡 Note:
> If you update the port in "api.py", please also update it in "index.html" to avoid errors

---
**This project is fully tested and workes perfect and error free**
