Machine Learning Nanodegree Capstone Project
Allstate Claim Severity Prediction

📌 Project Overview
This project is based on the Kaggle competition “Allstate Claims Severity”, which focuses on predicting the severity of insurance claims using supervised machine learning regression techniques.
The objective is to predict a continuous target variable (loss) based on a high-dimensional dataset consisting of both categorical and numerical features. Accurate claim severity prediction enables insurance companies to improve risk assessment, pricing strategies, and financial planning.

🧠 Problem Statement
Given a set of claim-related features, predict the expected loss amount for each insurance claim.
This is formulated as a regression problem.

🛠️ Tech Stack
Python 3
Google Colab (cloud-based execution)
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
XGBoost
Keras
SciPy

📊 Dataset
The dataset is sourced from the Kaggle competition:
Allstate Claims Severity
https://www.kaggle.com/competitions/allstate-claims-severity
Dataset Characteristics
Total features: 131
Continuous features: 14 (cont1 – cont14)
Categorical features: 116 (cat1 – cat116)
Target variable: loss
Identifier: id
The training dataset contains labeled loss values, while the test dataset does not include the target variable.

🔄 Data Preprocessing
To handle the complexity and dimensionality of the dataset, the following preprocessing steps were applied:
Handling high-cardinality categorical features
Dimensionality reduction using PCA
Log transformation of the target variable to normalize distribution
Feature scaling and preparation for model training

📈 Exploratory Data Analysis
Feature correlations were explored using Seaborn heatmaps
Relationships between numerical variables were analyzed using scatter matrix plots
Distribution patterns of the target variable were examined to guide transformations

🤖 Models Implemented
The following supervised regression models were trained and evaluated:
Decision Tree Regressor
XGBoost Regressor
Deep Neural Network (Keras)
Each model was trained using 90% of the training data, with 10% held out for validation.

📏 Model Evaluation
Evaluation Metric: Mean Absolute Error (MAE)
MAE was used to measure the average difference between predicted and actual loss values
The final model was selected based on lowest validation MAE
Note: Kaggle test data does not contain true loss values, so MAE cannot be computed on the test set.

🏆 Results
MAE scores from all three models were compared
The model with the best MAE performance was selected as the final solution
Results demonstrate the effectiveness of tree-based and ensemble models for structured insurance data

📁 Project Structure
├── notebooks/          # Jupyter notebooks for EDA and model training
├── artifacts/          # Saved models and preprocessing artifacts
├── raw_code/           # Experimental and trial implementations
├── requirements.txt    # Python dependencies
└── README.md

💡 Business Relevance
Accurate claim severity prediction helps insurers:
Improve underwriting decisions
Optimize reserve allocation
Reduce financial risk
Enhance pricing strategies

📚 References
https://riskandinsurance.com/georgia-pacific/
https://www.investopedia.com/terms/a/average-severity.asp
https://www.casact.org/pubs/forum/05spforum/05spf215.pdf

👤 Author
Paras Saini
MSc Data Analytics
Machine Learning & Data Science Enthusiast
