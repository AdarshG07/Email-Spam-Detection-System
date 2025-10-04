import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Step 1: Load Local Dataset
df = pd.read_csv("Email_Spam.csv")   
print("Dataset Loaded:")
print(df.head())

# Step 2: Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

df['cleaned_message'] = df['message'].apply(clean_text)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_message'], df['label_num'], test_size=0.2, random_state=42
)

# Step 4: Feature Extraction
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Model Training
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 6: Evaluation
y_pred = model.predict(X_test_tfidf)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Save Model
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("Model and Vectorizer saved successfully!")
