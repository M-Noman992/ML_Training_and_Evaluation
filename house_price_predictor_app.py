import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
data = {
    'Bedrooms': [2, 3, 3, 4, 4, 5, 6, 2, 3, 5],
    'Size (sq ft)': [850, 900, 1200, 1500, 1600, 1800, 2100, 700, 950, 1900],
    'Age (years)': [5, 10, 15, 20, 7, 8, 3, 25, 30, 4],
    'Price ($)': [300000, 350000, 400000, 500000, 480000, 600000, 700000, 250000, 300000, 650000]
}
df = pd.DataFrame(data)
X = df[['Bedrooms', 'Size (sq ft)', 'Age (years)']]
y = df['Price ($)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.title("House Price Prediction")
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
size = st.number_input("Size (sq ft)", min_value=500, max_value=5000, value=1000)
age = st.number_input("Age of the House (years)", min_value=1, max_value=100, value=10)
user_input = np.array([[bedrooms, size, age]])
predicted_price = model.predict(user_input)
st.write(f"Predicted House Price: ${predicted_price[0]:,.2f}")
st.subheader("Feature vs Price Scatter Plots")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.scatterplot(x='Bedrooms', y='Price ($)', data=df, ax=axes[0])
axes[0].set_title("Bedrooms vs Price")
sns.scatterplot(x='Size (sq ft)', y='Price ($)', data=df, ax=axes[1])
axes[1].set_title("Size (sq ft) vs Price")
sns.scatterplot(x='Age (years)', y='Price ($)', data=df, ax=axes[2])
axes[2].set_title("Age (years) vs Price")
st.pyplot(fig)
st.subheader("Predicted vs Actual Prices")
fig2, ax = plt.subplots()
ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Predicted vs Actual House Prices")
st.pyplot(fig2)