# 🎫 Support Ticket Classification System

## 📌 Project Overview
This project is a simple Machine Learning-based system that classifies support tickets into different categories and assigns a priority level.

It helps organizations automatically manage and prioritize customer issues.

---

## 🚀 Features
- Classifies support tickets into:
  - Billing
  - Technical
  - Account
- Assigns priority levels:
  - High
  - Medium
  - Low
- Uses Natural Language Processing (NLP)
- Simple and beginner-friendly implementation

---

## 🧠 Technologies Used
- Python
- Pandas
- NLTK
- Scikit-learn

---

## ⚙️ How It Works
1. Input: User enters a support issue (text)
2. Text preprocessing:
   - Lowercasing
   - Removing punctuation
   - Removing stopwords
3. Feature extraction using CountVectorizer
4. Model training using Naive Bayes
5. Output:
   - Predicted category
   - Priority level

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install pandas nltk scikit-learn
