# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
import nltk
import string

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Download stopwords (run once)
nltk.download('stopwords')

# ==============================
# 2. Create Dataset
# ==============================
data = {
    "text": [
        "My payment failed",
        "App is crashing frequently",
        "Need to reset my password",
        "Refund not received yet",
        "Website is very slow",
        "Unable to login to my account",
        "Payment deducted but order not placed",
        "App not opening",
        "Forgot my password",
        "Transaction failed again"
    ],
    "category": [
        "billing",
        "technical",
        "account",
        "billing",
        "technical",
        "account",
        "billing",
        "technical",
        "account",
        "billing"
    ]
}

df = pd.DataFrame(data)

# ==============================
# 3. Text Cleaning Function
# ==============================
def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

df["cleaned"] = df["text"].apply(clean_text)

# ==============================
# 4. Convert Text to Numbers
# ==============================
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["cleaned"])

y = df["category"]

# ==============================
# 5. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ==============================
# 6. Train Model
# ==============================
model = MultinomialNB()
model.fit(X_train, y_train)

# ==============================
# 7. Priority Function
# ==============================
def get_priority(text):
    text = text.lower()
    if "urgent" in text or "immediately" in text:
        return "high"
    elif "slow" in text or "delay" in text:
        return "medium"
    else:
        return "low"

# ==============================
# 8. Test with User Input
# ==============================
user_input = "payment not working urgently"

clean_input = clean_text(user_input)
vector_input = vectorizer.transform([clean_input])

predicted_category = model.predict(vector_input)[0]
priority = get_priority(user_input)

# ==============================
# 9. Output
# ==============================
print("\n--- RESULT ---")
print("Issue:", user_input)
print("Predicted Category:", predicted_category)
print("Priority Level:", priority)