# Creditcard_approval-prediction_using_ML
#### An interactive ML web app that predicts whether a credit card application is likely to be approved based on applicant details. Built using Python, scikit-learn, and Streamlit, the project demonstrates data preprocessing, model training, and real-time prediction through a sleek user interface.
**Live App**: Click here to try it out https://ccapprovalpredpy-wkas8ez5taaqwrkuhmpgez.streamlit.app/
![image alt](https://github.com/Architaa-P/creditcard_approval-prediction_using_ML/blob/main/Screenshot%202025-04-30%20201643.png?raw=true)
![image alt](https://github.com/Architaa-P/creditcard_approval-prediction_using_ML/blob/main/Screenshot%202025-04-30%20201728.png?raw=true)
## Overview
This machine learning project aims to automate and simulate a bank's credit card approval process. Using historical data and classification algorithms, it can predict the approval status of a new application in real-time.

## Key Features

- Clean and interactive user interface (Streamlit)
- Data preprocessing (missing values, encoding, scaling)
- Model training and evaluation (Random Forest Classifier)
- Performance visualization with Seaborn/Matplotlib
- SMOTE for class imbalance handling
- Live deployment on Streamlit Cloud

## Tech Stack
- **Frontend/UI**: Streamlit
- **Machine Learning**: scikit-learn, imbalanced-learn
- **Visualization**: Matplotlib, Seaborn
- **Data Handling**: pandas, numpy

## Getting Started Locally
```bash
# Clone the repository
git clone https://github.com/yourusername/credit-card-approval.git
cd credit-card-approval

# Install required packages
pip install -r requirements.txt

# Launch the app
streamlit run cc_approval_pred.py
