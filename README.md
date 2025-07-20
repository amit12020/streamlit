import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load dataset and train model
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Page configuration
st.set_page_config(page_title="Iris Flower Predictor", layout="centered")

# App title
st.title("🌸 Iris Flower Prediction App")
st.markdown("""
This app predicts the **species of Iris flowers** using user input features and a trained Random Forest model.
""")

# Sidebar for user input
st.sidebar.header("Input Features")
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length (cm)', float(X['sepal length (cm)'].min()), float(X['sepal length (cm)'].max()), float(X['sepal length (cm)'].mean()))
    sepal_width = st.sidebar.slider('Sepal width (cm)', float(X['sepal width (cm)'].min()), float(X['sepal width (cm)'].max()), float(X['sepal width (cm)'].mean()))
    petal_length = st.sidebar.slider('Petal length (cm)', float(X['petal length (cm)'].min()), float(X['petal length (cm)'].max()), float(X['petal length (cm)'].mean()))
    petal_width = st.sidebar.slider('Petal width (cm)', float(X['petal width (cm)'].min()), float(X['petal width (cm)'].max()), float(X['petal width (cm)'].mean()))
    
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display user input
st.subheader("User Input Features")
st.write(input_df)

# Make prediction
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)

# Display prediction
st.subheader("Prediction")
st.write(f"Predicted Iris species: **{iris.target_names[prediction]}**")

# Show prediction probabilities
st.subheader("Prediction Probabilities")
prob_df = pd.DataFrame(prediction_proba, columns=iris.target_names)
st.bar_chart(prob_df.T)

# Feature importance visualization
st.subheader("Feature Importance")
fig, ax = plt.subplots()
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.sort_values().plot(kind='barh', ax=ax)
ax.set_title("Random Forest Feature Importances")
st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit")

