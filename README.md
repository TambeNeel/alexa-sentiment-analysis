# 📊 Amazon Alexa Reviews – Sentiment Analysis 

An **end-to-end Machine Learning application** that predicts the sentiment (Positive / Negative) of Amazon Alexa product reviews.

The project covers:
- **Data preprocessing & EDA**
- **Model building & evaluation**
- **Class imbalance handling with Random Oversampling**
- **Flask backend API**
- **Interactive frontend (Tailwind CSS + Alpine.js)**
- **Real-time text analysis & batch CSV processing**

---

## 🚀 Project Overview

Amazon Alexa is one of the most popular voice assistants. Understanding customer feedback is crucial for improving user experience.  
This project uses Natural Language Processing (NLP) techniques to classify reviews as **positive** or **negative**.

---

## 🛠️ Tech Stack

**Backend:**
- Python 3
- Flask (API development)
- Scikit-learn (TF-IDF, Logistic Regression)
- imbalanced-learn (RandomOverSampler)
- Matplotlib (CSV prediction visualization)

**Frontend:**
- Tailwind CSS (Responsive UI)
- Alpine.js (Reactivity & Interactivity)
- HTML5 + JavaScript

---

## 📊 Dataset

- **Source:** Amazon Alexa reviews dataset
- **Format:** TSV file with columns:
  - `rating` (1–5 stars)
  - `verified_reviews` (text)
  - `feedback` (1 = positive, 0 = negative)

---

## 📈 Model Experiments

During model building, I experimented with multiple algorithms and preprocessing strategies:

| Model | Vectorization | Class Imbalance Handling | Accuracy | ROC-AUC | Notes |
|-------|--------------|--------------------------|----------|---------|-------|
| **Logistic Regression** | TF-IDF | None | ~93% | 0.94 | Good baseline, weaker minority recall |
| **Logistic Regression** | TF-IDF | **Random Oversampling** | **~94%** | **0.95** | ✅ Best trade-off between accuracy & recall |
| XGBoost Classifier | TF-IDF | None | ~93% | 0.94 | Slightly slower, no major gain |
| Random Forest Classifier | TF-IDF | None | ~92% | 0.92 | Lower recall, heavier model |
| SVM (Linear) | TF-IDF | None | ~93% | 0.94 | Similar to logistic regression, slower training |

**Final Choice:**  
`TF-IDF → RandomOverSampler → Logistic Regression`  
- Highest accuracy and ROC-AUC
- Significant improvement in minority class recall
- Fast to train and lightweight for deployment

---

## ⚙️ Workflow

### 1️⃣ Data Preprocessing & EDA
- Removed missing values
- Text cleaning
- Exploratory Data Analysis (EDA) to understand review distribution

### 2️⃣ Handling Class Imbalance
- The dataset had **more positive reviews** than negative ones
- Applied **Random Oversampling** **only on training data** to balance classes

### 3️⃣ Model Building
- Built multiple pipelines
- Chose Logistic Regression with Random Oversampling based on evaluation metrics

### 4️⃣ Evaluation
- Accuracy: ~94%
- Minority class recall improved significantly
- ROC-AUC: ~0.95

---

## 🌐 Web App Features

- **Paste text** → Instant sentiment prediction
- **Upload CSV** → Batch predictions + Pie chart of sentiment distribution
- **Download results** → Get CSV file with predictions
- **Light/Dark mode**
- **Responsive UI** → Works on mobile and desktop

---

## 💻 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/TambeNeel/alexa-sentiment-analysis.git
cd alexa-sentiment-analysis

