# 🔥 AI-Driven Micro-Wildfire Prediction and Evacuation Planning Using Multi-Modal Data Fusion

DOI: 10.36227/techrxiv.174961871.12313030/v1
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15660249.svg)](https://doi.org/10.5281/zenodo.15660249) 


![GitHub last commit](https://img.shields.io/github/last-commit/RheaGhosal/AI_Driven_Micro_Wildfire_Prediction_and_Evacuation_Planning_Using_Multi_Modal_Data_Fusion)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

> **Author**: Rhea Ghosal  
             Independent Researcher | IEEE Student Member
> **Paper**: [Read the Full Research Paper (PDF)][(https://www.techrxiv.org/users/932476/articles/1303271-ai-driven-micro-wildfire-prediction-and-evacuation-planning-using-multi-modal-data-fusion]

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



## 📊 Key Results

| Model        | Accuracy | AUC   | Best Use Case              |
|--------------|----------|-------|-----------------------------|
| CNN          | 92%      | 0.94  | Satellite-based imagery     |
| LSTM         | 89%      | 0.91  | Sequential time series      |
| Random Forest| 87%      | 0.89  | Feature importance + small data |

✅ Bias Mitigation:
- Equalized Odds ↓ from 0.21 → **0.05**
- Demographic Parity ↓ from 0.18 → **0.02**

---

## ⚖️ Fairness and Equity

This project emphasizes equitable technology for wildfire detection by:
- Implementing **bias-aware models** (AIF360, GANs)
- Enhancing accuracy in **underrepresented regions**
- Measuring fairness using **Equalized Odds** and **Statistical Parity**

---

## 🚀 Edge AI & Deployment

- Converted LSTM model to **TensorFlow Lite**
- Supports deployment on **IoT sensor devices**
- Suitable for rural, disconnected, or wildfire-prone zones

---

## 🌱 Future Enhancements

- Integrate **Vision Transformers (ViT)** for better spatial features
- Use **drone-based LiDAR and thermal sensors**
- Deploy via **Federated Learning** for privacy and scale

---
## 📎 Citation (Preprint)

If referencing this research paper, please cite:

```bibtex
@misc{ghosal2025wildfire,
  author       = {Rhea Ghosal},
  title        = {{AI-Driven Micro-Wildfire Prediction and Evacuation Planning Using Multi-Modal Data Fusion}},
  year         = {2025},
  publisher    = {TechRxiv},
  note         = {Preprint},
  doi          = {10.36227/techrxiv.174961871.12313030/v1},
  url          = {https://doi.org/10.36227//techrxiv.174961871.12313030/v1}
}
 
---


## Citation (Code)

To reference the codebase specifically, use:

```bibtex
@software{ghosal2025code,
  author       = {Rhea Ghosal},
  title        = {{AI-Driven Micro-Wildfire Prediction and Evacuation Planning Using Multi-Modal Data Fusion (Code Repository)}},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15660249},
  url          = {https://doi.org/10.5281/zenodo.15660249}
}



