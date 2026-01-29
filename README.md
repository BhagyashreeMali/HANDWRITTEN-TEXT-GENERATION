# ✍️ Handwritten Text Generation using Character-Level RNN

This project implements a **character-level Recurrent Neural Network (RNN)**
to generate **handwritten-like text** by learning writing patterns from text data.

---

## 📂 Project Structure
```bash
Handwritten-Text-Generation/
├── data/
│   └── handwriting_text.txt
├── model/
│   └── char_rnn.py
├── train.py
├── generate.py
├── requirements.txt
└── README.md
```
---

## 🚀 How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 2️⃣ Train the model
```bash
python train.py
```
### 3️⃣ Generate handwritten-like text
```bash
python generate.py
```
### 📝 Output

The model generates text that mimics human handwritten writing style
by learning character-level patterns such as spacing, punctuation,
and natural flow of writing.

### 🧠 Dataset

The dataset consists of handwritten-style text inspired by trending
research papers, stored in:
```bash
data/handwriting_text.txt
```
### Example dataset content:
Handwriting is a beautiful form of expression.
Every stroke carries emotion and intent.
The flow of letters creates a unique personal style.

### 🛠 Technologies Used
	•	Python
	•	PyTorch
	•	Character-Level RNN

  ### 📌 Notes
	•	Model files (*.pth) and cache files are ignored using .gitignore
	•	This project generates text, not handwriting images



