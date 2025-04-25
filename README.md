# 🔥 AI-Driven Micro-Wildfire Prediction and Evacuation Planning Using Multi-Modal Data Fusion

![GitHub last commit](https://img.shields.io/github/last-commit/RheaGhosal/AI_Driven_Micro_Wildfire_Prediction_and_Evacuation_Planning_Using_Multi_Modal_Data_Fusion)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

> **Author**: Rhea Ghosal  
> **Grade**: 10 | AI Researcher and Student Innovator  
> **Paper**: [Read the Full Research Paper (PDF)](./AI_Driven_Micro_Wildfire_Prediction_and_Evacuation_Planning_Using_Multi_Modal_Data_Fusion.pdf)

---

## 📘 Overview

Micro-wildfires pose critical threats to ecosystems and communities, yet are often missed by large-scale fire monitoring systems. This project proposes an AI-based framework combining **Convolutional Neural Networks (CNNs)**, **LSTMs**, and **Random Forests** using fused data from satellite imagery, weather systems, and sensor networks.

Key innovations include:
- **92% accuracy** using CNNs on spatial data
- GAN-based **bias mitigation** improving Equalized Odds from 0.21 → 0.05
- Edge deployment with **TensorFlow Lite** for IoT-based fire prevention

---

## 🧠 AI & ML Models Used

- ✅ **Random Forest**: Interpretable, robust to missing data  
- ⚡ **XGBoost**: Efficient and high-performing decision trees  
- 🧠 **CNN**: Spatial feature extraction from satellite imagery  
- ⏱ **LSTM**: Sequential prediction of wildfire progression  

---

## 🌍 Dataset Sources (Used in Full Research)

- **MODIS Active Fire (NASA FIRMS)**
- **NOAA Weather Data**: Temp, humidity, wind speed
- **Ground Sensor Data**: Smoke, gas, thermal readings

> ⚠️ _Note: The GitHub notebook uses **synthetic sample data** for illustration only. The research paper is based on real, publicly available data._

---

## 💻 Notebook Contents

[➡️ Open the Notebook](https://github.com/RheaGhosal/AI_Driven_Micro_Wildfire_Prediction_and_Evacuation_Planning_Using_Multi_Modal_Data_Fusion/blob/main/Micro-WildfirePredictionAI.ipynb)

```text
Micro-WildfirePredictionAI.ipynb
├── Data Preprocessing & Feature Engineering
├── Model Training (Random Forest, LSTM)
├── Fairness Evaluation (AIF360)
├── Performance Metrics (Accuracy, AUC, ROC)
└── Edge Deployment (TensorFlow Lite Conversion)
