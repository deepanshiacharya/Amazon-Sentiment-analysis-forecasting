# **Sentiment Analysis of Amazon Product Reviews Using Time-Series Modeling**

![image](https://github.com/user-attachments/assets/b94acfa3-efb3-417b-b3e2-d0cccd3a685c)


## **Abstract**

Sentiment analysis plays a crucial role in understanding user opinions and improving business strategies. While traditional sentiment analysis provides insights into static reviews, this project focuses on understanding the **temporal evolution** of sentiment, which is influenced by external factors like product updates, competitor activity, and market trends. By applying **BiLSTM with Attention** and **time-series modeling** (including **Hidden Markov Models (HMM)** and **LSTMs**), the project aims to track sentiment shifts over time and detect anomalies in user feedback for e-commerce platforms like Amazon. These insights can help businesses anticipate changes, respond to emerging trends, and refine product strategies.

## **Problem Statement**

In **e-commerce platforms**, user sentiment within product categories evolves over time due to several factors such as product innovation, competitive actions, and external events. Traditional sentiment analysis methods typically treat reviews as independent, static observations and fail to capture these temporal shifts.

A key challenge is identifying when sentiment **regimes** change within a category (e.g., from positive to negative sentiment) and detecting **anomalies** (e.g., sudden spikes or drops in sentiment) that may indicate disruptions like product failures, viral trends, or external influences.

The goal of this project is to use **time-series modeling techniques** to capture the dynamic nature of sentiment over time, detect shifts in sentiment regimes, and flag unusual sentiment patterns.

## **Approach**

### **BiLSTM + Attention for Sentiment Analysis:**

- **BiLSTM (Bidirectional Long Short-Term Memory)** is used to capture both past and future context in sentiment sequences. This allows for better understanding of sentiment changes, as the model considers both previous and upcoming review data.
- **Attention Mechanism** is added on top of BiLSTM to focus on important words or phrases in the reviews that contribute the most to sentiment changes, improving model interpretability and accuracy.

### **Time-Series Modeling:**

- To model sentiment as a **time series**, we track sentiment scores over time and apply **CNN** + **LSTM** for forecasting future Average sentiment scores.

### **Hidden Markov Models (HMM) for Regime Detection:**

- **Regime Detection** involves detecting shifts in sentiment regimes within a category. This can be achieved using **Hidden Markov Models (HMM)**, where sentiment scores are assumed to follow an underlying hidden regime. The HMM helps identify when sentiment transitions between different states (e.g., positive, neutral, or negative).
- The model uses **Markov processes** with transition probabilities between different sentiment regimes and applies algorithms like the **Viterbi algorithm** or **Bayesian Change Point Detection (BOCPD)** to estimate the most probable sequence of sentiment regimes over time.

### **Anomaly Detection in Sentiment Trends:**

- **Anomalies** in sentiment trends correspond to sudden shifts in customer opinion. For this, a robust approach such as **Exponential Weighted Moving Average (EWMA)** is used to detect abnormal sentiment shifts.
- If the sentiment score deviates significantly from the expected value (predicted sentiment), it is flagged as an anomaly. This can be further refined using **LSTM-based autoencoders** trained on sentiment sequences for detecting outlier behavior.

### **Sentiment Evolution Analysis:**

- The time-series models are applied to a sequence of sentiment scores over time, providing a clear understanding of how sentiment evolves within a product category. These models also help detect anomalies that could point to significant market events or disruptions.

