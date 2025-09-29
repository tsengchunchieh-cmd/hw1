import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# =========================================================
# Streamlit App: Simple Linear Regression (CRISP-DM)
# =========================================================

def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="CRISP-DM Linear Regression", layout="wide")
    st.title("📊 Simple Linear Regression - CRISP-DM Demo")

    # Sidebar controls
    st.sidebar.header("🔧 Data Generation Parameters")
    a = st.sidebar.slider("True slope (a)", 0.0, 10.0, 3.0, 0.1)
    b = st.sidebar.slider("True intercept (b)", -10.0, 10.0, 5.0, 0.1)
    noise_level = st.sidebar.slider("Noise standard deviation", 0.0, 10.0, 3.0, 0.1)
    n_points = st.sidebar.slider("Number of data points", 10, 200, 50, 5)

    # CRISP-DM Phases
    business_understanding()
    X, y = data_understanding(a, b, noise_level, n_points)
    X_train, X_test, y_train, y_test = data_preparation(X, y)
    model = modeling(X_train, y_train)
    evaluation(model, X_test, y_test)
    deployment(model)

def business_understanding():
    """
    Phase 1: Business Understanding
    """
    with st.expander("🔎 1. Business Understanding", expanded=True):
        st.write("""
        **Goal:** Predict a continuous target (y) from a single feature (x) using linear regression.
        **Method:** Simple linear regression (`y = ax + b`).
        **Example:** Predicting exam scores from study hours.
        """)

def data_understanding(a, b, noise_level, n_points):
    """
    Phase 2: Data Understanding
    """
    with st.expander("📂 2. Data Understanding", expanded=True):
        st.write("""
        **Data Generation:** We create a synthetic dataset based on your parameters.
        - `X` values are evenly spaced.
        - `y` values are calculated with `y = ax + b + noise`.
        - The data is then visualized.
        """)
        np.random.seed(42)
        X = np.linspace(0, 10, n_points).reshape(-1, 1)
        noise = np.random.normal(0, noise_level, n_points)
        y = a * X.flatten() + b + noise

        df = pd.DataFrame({"X": X.flatten(), "y": y})
        st.write("Sample of the generated dataset:")
        st.dataframe(df.head())

        st.write("Visualizing the data:")
        fig, ax = plt.subplots()
        ax.scatter(df["X"], df["y"], color="blue", label="Data")
        ax.set_xlabel("X")
        ax.set_ylabel("y")
        ax.set_title("Scatter Plot of Data")
        ax.legend()
        st.pyplot(fig)
        return X, y

def data_preparation(X, y):
    """
    Phase 3: Data Preparation
    """
    with st.expander("🛠️ 3. Data Preparation", expanded=True):
        st.write("""
        **Goal:** Split data for training and testing to prevent overfitting.
        - **Training set (80%):** Used to train the model.
        - **Test set (20%):** Used to evaluate the model on unseen data.
        """)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write(f"**Training samples:** {len(X_train)}")
        st.write(f"**Test samples:** {len(X_test)}")
        return X_train, X_test, y_train, y_test

def modeling(X_train, y_train):
    """
    Phase 4: Modeling
    """
    with st.expander("🤖 4. Modeling", expanded=True):
        st.write("""
        **Goal:** Train a linear regression model.
        **Action:** Fit the model on the training data to learn the best slope and intercept.
        """)
        model = LinearRegression()
        model.fit(X_train, y_train)
        st.write(f"Fitted Model: **y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}**")
        return model

def evaluation(model, X_test, y_test):
    """
    Phase 5: Evaluation
    """
    with st.expander("📏 5. Evaluation", expanded=True):
        st.write("""
        **Goal:** Evaluate the model's performance on the test data.
        **Metrics:**
        - **Mean Squared Error (MSE):** Lower is better.
        - **R² Score:** Closer to 1.0 is better.
        """)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")

        fig, ax = plt.subplots()
        ax.scatter(X_test, y_test, color="blue", label="Actual")
        ax.plot(X_test, y_pred, color="red", linewidth=2, label="Predicted")
        ax.set_xlabel("X")
        ax.set_ylabel("y")
        ax.set_title("Evaluation: Actual vs Predicted")
        ax.legend()
        st.pyplot(fig)

def deployment(model):
    """
    Phase 6: Deployment
    """
    with st.expander("🚀 6. Deployment", expanded=True):
        st.write("""
        **Goal:** Use the trained model for predictions on new data.
        **Action:** Enter a new 'X' value to get a prediction for 'y'.
        """)
        new_x = st.number_input("Enter a new X value for prediction", min_value=0.0, max_value=20.0, value=7.0, step=0.5)
        new_pred = model.predict(np.array([[new_x]]))
        st.success(f"Predicted y for x = {new_x:.2f} → {new_pred[0]:.2f}")

if __name__ == "__main__":
    main()
