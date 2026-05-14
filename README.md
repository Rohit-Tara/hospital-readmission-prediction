# Hospital Readmission Prediction — Distributed ML Pipeline
 
A distributed machine learning pipeline built with PySpark to predict hospital readmission risk using a large-scale clinical dataset of approximately 100,000 patient records. The project combines global predictive modelling with K-Means clustering to investigate whether patient subgroup-based models improve prediction performance over a single global model.
 
---
 
## Overview
 
Hospital readmission is a key quality metric in healthcare. This project applies distributed computing and machine learning techniques to predict whether a diabetic patient will be readmitted within 30 days, after 30 days, or not at all — using structured clinical data including medication counts, diagnoses, hospital stay length, and lab results.
 
---
 
## What This Project Does
 
**1. Distributed Model Development**
- Loads and preprocesses the diabetic patient dataset using PySpark
- Handles missing values, encodes categorical targets, and removes high-missingness columns (threshold: 30%)
- Assembles numeric features into a PySpark ML-compatible vector
- Trains a global Random Forest classifier (100 trees) on an 80/20 train/test split
- Evaluates global model performance using accuracy, precision, recall, and F1-score
**2. Clustering Analysis**
- Scales features using StandardScaler before clustering
- Applies the Elbow method and Silhouette scoring (via Yellowbrick) to determine the optimal number of clusters
- Fits a K-Means model in PySpark and assigns cluster labels to all patient records
- Applies PCA for 2D dimensionality reduction and visualises cluster structure using Seaborn and Matplotlib
**3. Cluster-Based Classification**
- Trains a separate Random Forest classifier for each patient cluster
- Evaluates per-cluster accuracy and F1-score
- Computes size-weighted average performance across clusters
- Compares cluster-based model performance against the global model to assess whether subgroup modelling improves prediction
---
 
## Tech Stack
 
- Python 3
- PySpark (MLlib — RandomForestClassifier, KMeans, VectorAssembler, StandardScaler, PCA)
- scikit-learn (KMeans for Elbow/Silhouette analysis)
- Yellowbrick (KElbowVisualizer)
- pandas, NumPy
- Matplotlib, Seaborn
- Google Colab (development environment)
---
 
## Dataset
 
The project uses the [Diabetes 130-US Hospitals dataset](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008) from the UCI Machine Learning Repository. It contains approximately 100,000 patient records with clinical features including:
 
- Hospital stay duration
- Number of medications and diagnoses
- Lab procedure counts
- Readmission outcome (NO / less than 30 days / more than 30 days)
**Note:** The dataset is not included in this repository. Download it from the UCI repository and update the file path in the notebook accordingly.
 
---
 
## How to Run
 
1. Clone the repository
2. Download the dataset from UCI and place it in your working directory
3. Open `Part3_DistributedAnalysis.ipynb` in Google Colab or a PySpark-enabled environment
4. Update the dataset file path in the data loading cell
5. Run all cells in order
**Requirements:**
```
pyspark
pandas
numpy
scikit-learn
yellowbrick
matplotlib
seaborn
```
 
---
 
## Key Results
 
- Global Random Forest model trained and evaluated on ~100,000 patient records
- K-Means clustering applied to identify clinically distinct patient subgroups
- Cluster-based models compared against the global baseline using weighted accuracy and F1-score
- Findings communicated through visualisations and structured summary outputs
---
 
## Author
 
Rohit Tara — MSc Bioinformatics, University of Leicester
github.com/Rohit-Tara
