# 🧠 Using AI in Financial Asset Selection  
### Deep Learning Applications for Designing Momentum-Based Investment Strategies  

**Author:** Manuel Díaz-Meco Terrés  
**Supervisor:** Fernando Berzal Galiano  
**University of Granada – Faculty of Sciences**  
**Double Degree in Computer Engineering and Mathematics**  

---

## 📘 Project Overview  

This project explores the application of **Artificial Intelligence models** to financial markets, aiming to enhance **momentum-based investment strategies**.  

Using historical market data from 1990 onwards, several **predictive models** — both classical and deep learning — are implemented, trained, and compared to evaluate their performance in predicting asset returns and constructing more profitable investment portfolios.  

A complementary **web application** has also been developed, allowing interactive visualization of model results and portfolio simulations.

> ⚠️ **Important:**  
> This project is built upon a proprietary framework developed by my thesis supervisor.  
> Therefore, it **cannot be executed or reused by third parties** without access to that original codebase.

---

## 🧩 Models Implemented  

The project includes and compares the following predictive models:

- **Linear and Exponential Regression**  
- **Support Vector Regression (SVR)**  
- **Random Forest Regressor**  
- **Recurrent Neural Networks (RNN)**  
- **Convolutional Neural Networks (CNN)**  
- **Transformers for Time Series**

Each model is evaluated using both **statistical metrics** (MSE, MAPE, R²) and **financial metrics** (Sharpe Ratio, cumulative return, drawdown, etc.).

---

## 💻 Web Application  

The web interface provides an interactive way to explore model outputs and portfolio results.

### Main Features  

- **Stock Visualization:** displays all available stocks with trained models.  
- **Models per Stock:** shows every trained model associated with a selected stock.  
- **Predictions & Metrics:** visualizes model predictions, quantitative metrics, and hyperparameters.  
- **Portfolio Simulation:** allows users to configure and run investment simulations.  
- **Comparative Analysis:** compares simulated portfolios against the S&P500 and the Clenow momentum strategy.

---

## 📈 Results  

The comparative study revealed that **Deep Learning models**, especially **RNNs** and **Transformers**, captured the temporal dependencies and nonlinear relationships in financial time series more effectively than classical approaches.  

Although models like **SVR** and **Random Forest** provided solid baselines, neural architectures achieved better generalization and higher predictive accuracy.  
Momentum-based portfolios constructed using these models obtained superior **Sharpe ratios** and **cumulative returns** compared to both the **Clenow strategy** and the **S&P500 benchmark**.

---

## 🧭 Future Work  

- Integrate macroeconomic and sentiment-based indicators.  
- Explore **transfer learning** and **reinforcement learning** for dynamic investment strategies.  
- Extend the web app with real-time data visualization.  
- Deploy the system through containerization and cloud services.

---

## 📚 References  

Key literature and sources consulted include:  

- Andreas Clenow — *Stocks on the Move*  
- Fernando Berzal — *Neuronal Networks & Deep Learning*  
- Jegadeesh & Titman (1993) — *Returns to Buying Winners and Selling Losers*  

---

## 🪪 License  

This project was developed for **academic purposes only** as part of a Bachelor's Thesis.  
It **cannot be executed or reused** without access to the base framework provided by the thesis supervisor.  

© 2025 Manuel Díaz-Meco Terrés. All rights reserved.


