# 📉 Gradient Descent Visualizer

A hands-on tool to understand how machine learning models "learn." This app implements the Gradient Descent algorithm from scratch without using high-level libraries like Scikit-Learn for the math.

** https://car-mpg-predictor-fztvga2962bzmcvx9eqhkm.streamlit.app/ **

## 💡 How it Works
The app minimizes the **Mean Squared Error (MSE)** by calculating partial derivatives for the slope ($m$) and intercept ($b$):

- **Partial Derivative for $m$:** $\frac{\partial J}{\partial m} = \frac{-2}{n} \sum x(y - y_p)$
- **Partial Derivative for $b$:** $\frac{\partial J}{\partial b} = \frac{-2}{n} \sum (y - y_p)$

## 🚀 Features
- **Live Animation:** Watch the regression line "drift" into the correct position.
- **Adjustable Hyperparameters:** Change the learning rate to see how it affects convergence (or causes divergence!).
- **Real-time Metrics:** Track the Loss (MSE) dropping with every iteration.

