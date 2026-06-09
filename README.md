<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=Porto%20Taxi%20Traffic%20Prediction&fontSize=28&fontColor=fff&animation=twinkling&desc=ML-Powered%20Trajectory%20Forecasting&descSize=16&descAlignY=75" width="100%"/>

<div align="center">

# Porto Taxi Traffic Prediction

[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

</div>

---

## Overview

Uses the **Porto Taxi Dataset** (ECML/PKDD 2015 Challenge) to build a machine learning model that predicts taxi trip destinations and travel times based on partial trajectory data.

---

## Problem Statement

Given the first K GPS points of a taxi trip, predict:
- Final destination coordinates
- Estimated travel time

---

## Approach

1. **EDA** - Exploratory Data Analysis of trip patterns
2. **Feature Engineering** - Extract temporal, spatial, and route features
3. **Modeling** - Random Forest, Gradient Boosting, Neural Networks
4. **Evaluation** - RMSE on destination prediction

---

## Dataset

- **Source**: [Kaggle - ECML/PKDD 2015](https://www.kaggle.com/c/pkdd-15-taxi-trip-time-prediction-ii)
- **Size**: 1.7M taxi trips in Porto, Portugal
- **Features**: Taxi ID, Call Type, Timestamp, GPS Polyline

---

## Getting Started

```bash
git clone https://github.com/Divgagan/Porto_taxi_Traffic_trediction_.git
cd Porto_taxi_Traffic_trediction_
pip install -r requirements.txt
jupyter notebook
```

---

**Author:** [Gagan Diwakar](https://portfolio-gagan-nu.vercel.app/) | [LinkedIn](https://www.linkedin.com/in/gagan-diwakar-772134293/)

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>