# ML Training and Evaluation

This repository contains machine learning projects focused on model training, evaluation, and deployment. It includes a deep learning script for digit classification and an interactive web application for real estate price prediction.

## 🛠 Setup & Run Commands

Open your terminal or command prompt. First, install all the required dependencies, and then run the specific script or application directly from the same terminal:

```bash
# 1. Install all required dependencies
pip install tensorflow numpy matplotlib pandas seaborn scikit-learn streamlit

# 2. Run the Deep Learning Digit Classifier
# This will download the MNIST dataset, train models with ReLU and Sigmoid activations, and display performance graphs.
python digit_classifier_keras.py

# 3. Run the House Price Predictor Web App
# This launches an interactive Streamlit application in your browser.
streamlit run house_price_predictor_app.py
