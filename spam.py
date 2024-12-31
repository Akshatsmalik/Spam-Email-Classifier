import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load the dataset
def load_data():
    data = pd.read_csv("email_classification.csv")  # Replace with the correct path
    return data

# Preprocess and train the model

def train_model(data):
    # Split data into features and labels
    X = data['email']  # Replace with the actual column name for email text
    y = data['label']  # Replace with the actual column name for labels

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Vectorize the text data
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train a Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    return model, vectorizer, accuracy

# Load data and train model
data = load_data()
model, vectorizer, accuracy = train_model(data)

# Streamlit App
st.title("Spam Detector")
st.write(f"Model Accuracy: {accuracy:.2f}")

# Input box for email
email_input = st.text_area("Enter the email content here:")

# Submit button
if st.button("Submit"):
    if email_input.strip() == "":
        st.warning("Please enter an email.")
    else:
        # Transform input using the vectorizer
        input_features = vectorizer.transform([email_input])

        # Predict using the trained model
        prediction = model.predict(input_features)

        # Display result
        if prediction[0] == 'ham':  # Replace with the actual encoding of your labels
            st.success("This email is classified as: Ham")
        else:
            st.error("This email is classified as: Spam")

