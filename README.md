# spam-detection-heuristics-ml
Spam detection using heuristic rules and machine learning (Naïve Bayes, SVM) on TREC 2007 dataset.

# 📄 README: Assignment 1 - Spam Detection with Rule-Based & Machine Learning Methods

### 🔍 Course: UIMLSEC4032E01  
### 🏫 University of Isfahan  
### 🧪 Dataset: TREC 2007 Email Corpus

---

## 📌 Overview

This assignment focuses on identifying spam emails using two major approaches:

- **Rule-Based Heuristics** (manually defined indicators)
- **Machine Learning** using **Naive Bayes Classifier** and alternatives

You will evaluate and compare different methods and their effectiveness based on accuracy, false positive rate (FPR), and other metrics.

---

## ✅ Part 1.a - Rule-Based Spam Detection using Simple Heuristics

### 🎯 Objective:
Develop a basic spam filter using handcrafted rules based on:
- Suspicious words (blacklisted terms)
- Missing subject line
- Unreliable senders (e.g., `unknown`, `noreply`, `mailer-daemon`)

### 🧠 Method:
1. Preprocess emails: tokenization, stopword removal, stemming
2. Build word frequency sets for spam and ham emails
3. Generate a **blacklist** based on thresholds (25% occurrence in ham emails)
4. Classify test emails based on the defined rules
5. Evaluate performance with a **confusion matrix** and accuracy metrics

### 💾 Files:
- `email_read_util.py` → Handles email parsing and preprocessing
- `HW1.py` → Contains the training, blacklist generation, and evaluation logic

---

## ✅ Part 1.b - Effect of Threshold Adjustment on Blacklist

### 🎯 Objective:
Analyze how changing the ham threshold from `25% → 20%` impacts classification performance.

### 🔄 Key Change:
```python
HAM_THRESHOLD = 0.20
```

### 📊 Observation:
- A lower threshold adds more words to the blacklist.
- May increase **True Positives** but also risks raising **False Positives**.
- This experiment helps in tuning the **Precision–Recall** tradeoff.

---

## ✅ Part 1.c - Improving Detection with N-Grams

### 🎯 Objective:
Enhance detection by considering **consecutive word patterns** (bigrams) such as:
- `click here`
- `free money`

### 🧠 Features Added:
- **Unigrams + Bigrams** from email content
- Special tokens:
  - `__NOSUBJECT__` if the email lacks a subject
  - `__UNTRUSTED_SENDER__` if sender is not in the trusted list

### ✨ Outcome:
- Better semantic understanding of emails
- More robust detection of spam phrases
- Improved classification accuracy and lower false positives

---

## ✅ Part 2.a - Spam Detection using Naive Bayes Classifier

### 🎯 Objective:
Use a **Multinomial Naive Bayes** classifier to automate spam classification using tokenized email content.

### ⚙️ Workflow:
1. Preprocess each email with `email_read_util.py`
2. Convert tokens into text strings
3. Vectorize using `CountVectorizer` from `scikit-learn`
4. Train & evaluate using **5-fold Cross-Validation**
5. Report:
   - Accuracy
   - Classification report (precision, recall, F1)
   - False Positive Rate (FPR)

---

## ✅ Part 2.b - Replace Naive Bayes with Another Classifier

### 🎯 Objective:
Substitute the Naive Bayes classifier with another model (e.g., Logistic Regression, Decision Tree, or SVM) and compare results.

### 📌 Example: SVM Classifier
- Files used: `cross_svm.py`, `sample_svm.py`
- Evaluation uses same strategy as 2.a

---

## 📂 Project Structure

```
📁 1.a
├── blacklist.pkl
├── email_read_util.py
├── HW1.py
└── __pycache__/

📁 1.b
├── blacklist.pkl
├── email_read_util.py
├── HW1.py
└── __pycache__/

📁 1.c
├── blacklist.pkl
├── email_read_util.py
├── HW1.py
└── __pycache__/

📁 2.a
├── cross/
│   ├── cross_mnb.py
│   ├── email_read_util.py
│   └── __pycache__/
├── sample/
│   ├── sample_mnb.py
│   ├── email_read_util.py
│   └── __pycache__/

📁 2.b
├── cross/
│   ├── cross_svm.py
│   ├── email_read_util.py
│   └── __pycache__/
├── sample/
│   ├── sample_svm.py
│   ├── email_read_util.py
│   └── __pycache__/
```

---

## 🧪 How to Run

1. Make sure NLTK data is available:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

2. Adjust paths to `DATA_DIR` and `LABELS_FILE` accordingly.

3. Run each script depending on the section you want to test.

---
