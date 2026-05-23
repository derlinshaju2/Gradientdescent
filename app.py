import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="Gradient Descent Visualizer")
st.title("📉 Gradient Descent from Scratch")

# --- Sidebar Configuration ---
st.sidebar.header("Hyperparameters")
iterations = st.sidebar.slider("Number of Iterations", 10, 500, 100)
lr = st.sidebar.slider("Learning Rate (α)", 0.001, 0.1, 0.01, format="%.3f")

# --- Data ---
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

# Initialize parameters
m, b = 0.0, 0.0
n = len(x)

# Create placeholders for live updates
plot_placeholder = st.empty()
metrics_placeholder = st.empty()
progress_bar = st.progress(0)

# --- Gradient Descent Loop ---
for i in range(iterations):
    yp = m * x + b
    
    # Calculate Gradients (Partial Derivatives)
    dm = (-2/n) * sum(x * (y - yp))
    db = (-2/n) * sum(y - yp)
    
    # Update Weights
    m = m - lr * dm
    b = b - lr * db
    
    # Calculate Mean Squared Error
    mse = (1/n) * sum((y - yp)**2)
    
    # Update UI every few iterations to show progress
    if i % 2 == 0 or i == iterations - 1:
        # Plotting
        fig, ax = plt.subplots()
        ax.scatter(x, y, color='red', label='Original Data')
        ax.plot(x, m*x + b, color='blue', label=f'Model: y = {m:.2f}x + {b:.2f}')
        ax.set_ylim(0, 15)
        ax.legend()
        plot_placeholder.pyplot(fig)
        plt.close(fig)
        
        # Metrics
        metrics_placeholder.columns(3)[0].metric("Slope (m)", f"{m:.4f}")
        metrics_placeholder.columns(3)[1].metric("Intercept (b)", f"{b:.4f}")
        metrics_placeholder.columns(3)[2].metric("MSE (Loss)", f"{mse:.4f}")
        
        progress_bar.progress((i + 1) / iterations)
        time.sleep(0.05) # Adds a small delay for the animation effect

st.success("Training Complete!")
