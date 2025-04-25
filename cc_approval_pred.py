import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import requests

# Load datasets
train_original = pd.read_csv(r"https://raw.githubusercontent.com/semasuka/Credit-card-approval-prediction-classification/refs/heads/main/dataset/train.csv")
test_original = pd.read_csv(r"https://raw.githubusercontent.com/semasuka/Credit-card-approval-prediction-classification/refs/heads/main/dataset/test.csv")
full_data = pd.concat([train_original, test_original], axis=0)
full_data = full_data.sample(frac=1).reset_index(drop=True)

def data_split(df, test_size):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

train_original, test_original = data_split(full_data, 0.2)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Streamlit UI Configuration
st.set_page_config(
    page_title="Credit Card Approval", 
    page_icon="💳", 
    layout="wide"
)

st.markdown("""
    <style>
        .big-font { font-size:20px !important; font-weight: bold; }
        .stButton > button { background-color: #4CAF50; color: white; padding: 12px 24px; border-radius: 5px; font-size: 16px; }
        .stButton > button:hover { background-color: #45a049; }
        .stTextInput > div > div > input { background-color: #f5f5f5; }
        .stProgress > div > div > div > div { background-color: #4CAF50; }
        .step-up { margin-top: 30px; }
        div[data-testid="stExpander"] div[role="button"] p { font-size: 1.1rem; font-weight: 600; }
        div.stTabs button { font-size: 16px; font-weight: bold; }
        div.stTabs button[aria-selected="true"] { background-color: #e6f3ff; border-radius: 5px 5px 0px 0px; }
    </style>
""", unsafe_allow_html=True)

# Create main tabs
tab1, tab2, tab3 = st.tabs(["📝 Application", "📊 Results", "ℹ️ Information"])

with tab1:
    # Title section
    st.title("Credit Card Approval Prediction")
    st.markdown("Complete the form below to check if you qualify for a credit card based on your details.")
    
    # Form sections
    col1, col2 = st.columns(2)
    
    # Left Panel - Primary User Inputs
    with col1:
        st.subheader("📋 Applicant Information")
        
        gender = st.radio(
            "Gender", 
            ["Male", "Female"], 
            horizontal=True,
            help="Select your gender as it appears on official documents"
        )
        
        age = st.slider(
            "Age", 
            18, 70, 30,
            help="Select your current age in years"
        )
        
        income = st.number_input(
            "Monthly Income (USD)", 
            min_value=0, 
            max_value=100000,
            value=3000, 
            step=100,
            help="Enter your average monthly income before taxes"
        )
        
        employment_length = st.slider(
            "Years of Employment", 
            0, 30, 5,
            help="How many years have you been with your current employer?"
        )
        
        owns_car = st.radio(
            "Do you own a car?", 
            ["Yes", "No"], 
            horizontal=True,
            help="Indicate whether you currently own a vehicle"
        )
        
        owns_property = st.radio(
            "Do you own a property?", 
            ["Yes", "No"], 
            horizontal=True,
            help="Indicate whether you own real estate property"
        )
    
    # Right Panel - Additional Info
    with col2:
        st.subheader("📌 Additional Information")
        
        marital_status = st.selectbox(
            "Marital Status", 
            ["Single", "Married", "Divorced", "Widowed"],
            help="Select your current marital status"
        )
        
        education = st.selectbox(
            "Education Level", 
            ["High School", "Undergraduate", "Postgraduate"],
            help="Select your highest completed level of education"
        )
        
        employment_status = st.selectbox(
            "Employment Status", 
            ["Working", "Self-Employed", "Unemployed", "Retired"],
            help="Select your current employment situation"
        )
        
        has_phone = st.radio(
            "Do you have a phone?", 
            ["Yes", "No"], 
            horizontal=True,
            help="Indicate whether you have a telephone number"
        )
        
        has_email = st.radio(
            "Do you have an email?", 
            ["Yes", "No"], 
            horizontal=True,
            help="Indicate whether you have an email address"
        )
        
        family_size = st.slider(
            "Family Size", 
            1, 10, 1,
            help="Total number of family members in your household"
        )
    
    # Action buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        predict_bt = st.button("Predict Credit Card Approval", use_container_width=True)
    with col2:
        reset_bt = st.button("Reset Form", use_container_width=True)
        if reset_bt:
            st.rerun()

# Machine learning code
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_outliers=["Family member count", "Income", "Employment length"]):
        self.feat_with_outliers = feat_with_outliers
    def fit(self, df): return self
    def transform(self, df):
        print("OutlierRemover: Before\n", df.head())
        if set(self.feat_with_outliers).issubset(df.columns):
            Q1 = df[self.feat_with_outliers].quantile(0.25)
            Q3 = df[self.feat_with_outliers].quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df[self.feat_with_outliers] < (Q1 - 3 * IQR)) | (df[self.feat_with_outliers] > (Q3 + 3 * IQR))).any(axis=1)]
            print("OutlierRemover: After\n", df.head())
            return df
        else: return df

class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_to_drop=["Has a mobile phone", "Children count", "Job title", "Account age", "ID"]): # Added "ID" here
        self.feature_to_drop = feature_to_drop
    def fit(self, df): return self
    def transform(self, df):
        print("DropFeatures: Before\n", df.head())
        if set(self.feature_to_drop).issubset(df.columns):
            df = df.drop(self.feature_to_drop, axis=1)
            print("DropFeatures: After\n", df.head())
            return df
        else: return df

class TimeConversionHandler(BaseEstimator, TransformerMixin):
    def fit(self, df): 
        return self

    def transform(self, df):
        print("🔹 TimeConversionHandler: Before\n", df[["Employment length", "Age"]].head())

        if "Employment length" in df.columns:
            df["Employment length"] = df["Employment length"].abs().fillna(0)  # Convert negatives & NaN to positive values

        if "Age" in df.columns:
            df["Age"] = df["Age"].abs().fillna(df["Age"].median())  # Fill NaN values with the median age

        print("🔹 TimeConversionHandler: After\n", df[["Employment length", "Age"]].head())

        return df



class SkewnessHandler(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_skewness=["Income", "Age"]): 
        self.feat_with_skewness = feat_with_skewness

    def fit(self, df): 
        return self

    def transform(self, df):
        print("🔹 SkewnessHandler: Before Transformation\n", df[self.feat_with_skewness].head())

        if set(self.feat_with_skewness).issubset(df.columns):
            df[self.feat_with_skewness] = df[self.feat_with_skewness].astype(float)

            # ✅ Replace NaN with the median to prevent transformation errors
            df[self.feat_with_skewness] = df[self.feat_with_skewness].fillna(df[self.feat_with_skewness].median())

            # ✅ Ensure no negative values before log transformation
            df[self.feat_with_skewness] = df[self.feat_with_skewness].applymap(lambda x: max(x, 1e-9))

            # ✅ Apply log transformation safely
            df[self.feat_with_skewness] = np.log1p(df[self.feat_with_skewness])

            print("🔹 SkewnessHandler: After Transformation\n", df[self.feat_with_skewness].head())

        return df




class BinningNumToYN(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_num_enc=["Has a work phone", "Has a phone", "Has an email"]): self.feat_with_num_enc = feat_with_num_enc
    def fit(self, df): return self
    def transform(self, df):
        print("BinningNumToYN: Before\n", df.head())
        if set(self.feat_with_num_enc).issubset(df.columns):
            for ft in self.feat_with_num_enc: df[ft] = df[ft].astype(object).map({1: "Y", 0: "N"})
            print("BinningNumToYN: After\n", df.head())
            return df
        else: return df

class OneHotWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, one_hot_enc_ft=["Gender", "Marital status", "Dwelling", "Employment status", "Has a car", "Has a property", "Has a work phone", "Has a phone", "Has an email"]): self.one_hot_enc_ft = one_hot_enc_ft
    def fit(self, df): return self
    def transform(self, df):
        print("OneHotWithFeatNames: Before\n", df.head())
        if set(self.one_hot_enc_ft).issubset(df.columns):
            one_hot_enc = OneHotEncoder()
            one_hot_enc.fit(df[self.one_hot_enc_ft])
            feat_names_one_hot_enc = one_hot_enc.get_feature_names_out(self.one_hot_enc_ft)
            one_hot_enc_df = pd.DataFrame(one_hot_enc.transform(df[self.one_hot_enc_ft]).toarray(), columns=feat_names_one_hot_enc, index=df.index)
            rest_of_features = [ft for ft in df.columns if ft not in self.one_hot_enc_ft]
            df_concat = pd.concat([one_hot_enc_df, df[rest_of_features]], axis=1)
            print("OneHotWithFeatNames: After\n", df_concat.head())
            return df_concat
        else: return df

class OrdinalFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_enc_ft=["Education level"]): self.ordinal_enc_ft = ordinal_enc_ft
    def fit(self, df): return self
    def transform(self, df):
        print("OrdinalFeatNames: Before\n", df.head())
        if "Education level" in df.columns:
            ordinal_enc = OrdinalEncoder()
            df.loc[:, self.ordinal_enc_ft] = ordinal_enc.fit_transform(df[self.ordinal_enc_ft])
            print("OrdinalFeatNames: After\n", df.head())
            return df
        else: return df

class MinMaxWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, min_max_scaler_ft=["Age", "Income", "Employment length"]): 
        self.min_max_scaler_ft = min_max_scaler_ft

    def fit(self, df): 
        return self

    def transform(self, df):
        print("🔹 MinMaxWithFeatNames: Before Scaling\n", df[self.min_max_scaler_ft].head())

        if set(self.min_max_scaler_ft).issubset(df.columns):
            # ✅ Fill NaN values with median to avoid MinMaxScaler errors
            df[self.min_max_scaler_ft] = df[self.min_max_scaler_ft].fillna(df[self.min_max_scaler_ft].median())

            # ✅ Convert to float before scaling
            df[self.min_max_scaler_ft] = df[self.min_max_scaler_ft].astype(float)

            # ✅ Apply MinMaxScaler
            min_max_enc = MinMaxScaler()
            df.loc[:, self.min_max_scaler_ft] = min_max_enc.fit_transform(df[self.min_max_scaler_ft])

            print("🔹 MinMaxWithFeatNames: After Scaling\n", df[self.min_max_scaler_ft].head())
        
        return df



        
class ChangeToNumTarget(BaseEstimator, TransformerMixin):
    def fit(self, df): 
        return self

    def transform(self, df):
        print("ChangeToNumTarget: Before\n", df.head())
        if "Is high risk" in df.columns:
            df["Is high risk"] = pd.to_numeric(df["Is high risk"])
            print("ChangeToNumTarget: After\n", df.head())
            return df
        else:
            return df

class OversampleSMOTE(BaseEstimator, TransformerMixin):
    def fit(self, df): 
        return self
    
    def transform(self, df):
        print("🔹 Before SMOTE class distribution:\n", df["Is high risk"].value_counts())

        if "Is high risk" in df.columns:
            smote = SMOTE(random_state=42)
            X_bal, y_bal = smote.fit_resample(df.drop(columns=["Is high risk"]), df["Is high risk"])
            df_resampled = pd.concat([pd.DataFrame(X_bal, columns=df.columns[:-1]), pd.Series(y_bal, name="Is high risk")], axis=1)

            print("🔹 After SMOTE class distribution:\n", df_resampled["Is high risk"].value_counts())
            return df_resampled
        return df


def full_pipeline(df):
    pipeline = Pipeline(
        [
            ("outlier_remover", OutlierRemover()),
            ("feature_dropper", DropFeatures()), # ID is dropped here
            ("time_conversion_handler", TimeConversionHandler()),
            ("skewness_handler", SkewnessHandler()),
            ("binning_num_to_yn", BinningNumToYN()),
            ("one_hot_with_feat_names", OneHotWithFeatNames()),
            ("ordinal_feat_names", OrdinalFeatNames()),
            ("min_max_with_feat_names", MinMaxWithFeatNames()),
            ("change_to_num_target", ChangeToNumTarget()),
            ("oversample_smote", OversampleSMOTE()),
        ]
    )
    return pipeline.fit_transform(df)


# List of all the input variables
profile_to_predict = {
    "ID": 0,
    "Gender": 1 if gender == "Male" else 0,  # Convert gender selection
    "Has a car": 1 if owns_car == "Yes" else 0,
    "Has a property": 1 if owns_property == "Yes" else 0,
    "Children count": 0,  # Dropped
    "Income": income,  # Use the new variable from the updated form
    "Employment status": employment_status,  # Directly use from new dropdown
    "Education level": education,
    "Marital status": marital_status,
    "Dwelling": "Unknown",  # If dwelling type is missing, set default
    "Age": age,
    "Employment length": employment_length,
    "Has a mobile phone": 1,  # Dropped
    "Has a work phone": 1 if has_phone == "Yes" else 0,
    "Has a phone": 1 if has_phone == "Yes" else 0,
    "Has an email": 1 if has_email == "Yes" else 0,
    "Job title": "to_be_dropped",  # Dropped
    "Family member count": 1,  # Set a default or add a new input field
    "Account age": 0.00,  # Dropped
    "Is high risk": 0,  # Placeholder

}

profile_to_predict_df = pd.DataFrame([profile_to_predict])

# ✅ Check target distribution before preprocessing
print("Before preprocessing:")
print(train_original["Is high risk"].value_counts())  # Ensure both 0 and 1 exist

# ✅ Apply preprocessing pipeline
# ✅ Manually Apply Preprocessing
train_prep = OutlierRemover().fit_transform(train_original.copy())
train_prep = DropFeatures().fit_transform(train_prep)
train_prep = TimeConversionHandler().fit_transform(train_prep)
train_prep = SkewnessHandler().fit_transform(train_prep)
train_prep = BinningNumToYN().fit_transform(train_prep)
train_prep = OneHotWithFeatNames().fit_transform(train_prep)
train_prep = OrdinalFeatNames().fit_transform(train_prep)
train_prep = MinMaxWithFeatNames().fit_transform(train_prep)
train_prep = ChangeToNumTarget().fit_transform(train_prep)  # Ensure this class exists
# ✅ Apply SMOTE for oversampling
train_prep = OversampleSMOTE().fit_transform(train_prep)  # Apply SMOTE only ONCE

# ✅ Final check on target distribution before training
print("Final y distribution before training:")
print(train_prep["Is high risk"].value_counts())

# ✅ Split features and target variable
X = train_prep.drop("Is high risk", axis=1)
y = train_prep["Is high risk"]

# ✅ Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)  # 🔹 Model is trained BEFORE computing feature importance

def plot_feature_importance(model, X):
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
    ax.set_title('Top 10 Important Features for Credit Approval')
    ax.set_xlabel('Relative Importance')
    ax.set_ylabel('Feature')
    return fig

# ✅ Compute feature importance AFTER training

feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by="Importance", ascending=False)

# ✅ Display top 20 most important features
print("🔹 Feature Importances:\n", feature_importances.head(20))

# ✅ Store feature names for later use
feature_names = X.columns.tolist()


# ✅ Prediction Process
if predict_bt:
    profile_to_predict = {
        "ID": 0,
        "Gender": 1 if gender == "Male" else 0,
        "Has a car": 1 if owns_car == "Yes" else 0,
        "Has a property": 1 if owns_property == "Yes" else 0,
        "Children count": 0,
        "Income": income,
        "Employment status": employment_status,
        "Education level": education,
        "Marital status": marital_status,
        "Dwelling": "Unknown",
        "Age": age,
        "Employment length": employment_length,
        "Has a mobile phone": 1,
        "Has a work phone": 1 if has_phone == "Yes" else 0,
        "Has a phone": 1 if has_phone == "Yes" else 0,
        "Has an email": 1 if has_email == "Yes" else 0,
        "Job title": "to_be_dropped",
        "Family member count": family_size,
        "Account age": 0.00,
        "Is high risk": 0,
    }
    
    profile_to_predict_df = pd.DataFrame([profile_to_predict])
    
    # Apply preprocessing and predict
    with tab2:
        with st.spinner("Analyzing your application..."):
            progress_bar = st.progress(0)
            for percent_complete in range(101):
                time.sleep(0.01)  # Simulate processing time
                progress_bar.progress(percent_complete)
            
            # Mock prediction for this example - replace with your real prediction code
            prediction = 0 if income > 2500 and age > 25 else 1
            confidence = [0.75, 0.25] if prediction == 0 else [0.35, 0.65]
            
            progress_bar.empty()  # Remove progress bar when done
            
            # Store prediction in history
            new_prediction = {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'age': age,
                'income': income,
                'marital_status': marital_status,
                'education': education,
                'result': "Approved" if prediction == 0 else "Declined"
            }
            st.session_state.prediction_history.append(new_prediction)
            
            # Display result
            st.subheader("📋 Application Result")
            if prediction == 0:
                confidence_score = confidence[0] * 100
                st.success(f"🎉 Congratulations! You are approved for a credit card. (Confidence: {confidence_score:.1f}%)")
                st.balloons()
                
                # Add explanation
                with st.expander("What factors contributed to your approval?"):
                    st.write("The most important factors in your approval were:")
                    st.write("- Your stable employment history")
                    st.write("- Your income level")
                    st.write("- Your education level")
                    
                    # Show customized advice
                    st.subheader("Recommended Credit Card Options")
                    st.info("""
                    Based on your profile, you might qualify for:
                    1. **Rewards Card** - Earn points on everyday purchases
                    2. **Cash Back Card** - Get 2% back on all transactions
                    3. **Low Interest Card** - Great for carrying balances
                    """)
            else:
                confidence_score = confidence[1] * 100
                st.error(f"🚫 Unfortunately, your application has been declined. (Confidence: {confidence_score:.1f}%)")
                
                # Add suggestions
                with st.expander("How can you improve your chances?"):
                    st.write("Consider these steps to improve your approval chances:")
                    st.write("- Increase your income stability")
                    st.write("- Reduce existing debt")
                    st.write("- Build a credit history with smaller credit lines")
            
            # Show prediction history
            with st.expander("Prediction History"):
                if st.session_state.prediction_history:
                    history_df = pd.DataFrame(st.session_state.prediction_history)
                    st.dataframe(history_df)
                else:
                    st.write("No prediction history yet.")
            
            # Feature importance visualization (mock)
            st.subheader("🔍 Model Insights")
            feature_names = ["Income", "Age", "Employment_length", "Has_property", "Education_level", 
                             "Has_email", "Marital_status", "Gender", "Has_phone", "Family_size"]
            importances = [0.25, 0.18, 0.15, 0.12, 0.10, 0.07, 0.05, 0.04, 0.03, 0.01]
            
            # Create a simple feature importance DataFrame
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
            ax.set_title('Top 10 Important Features for Credit Approval')
            ax.set_xlabel('Relative Importance')
            ax.set_ylabel('Feature')
            st.pyplot(fig)

# Information tab content
with tab3:
    st.subheader("Understanding Credit Approval")
    st.write("""
    ### Key Factors in Credit Card Approval
    
    Credit card issuers evaluate applications based on several key factors:
    
    1. **Income Level**: Higher and more stable income improves approval chances
    2. **Credit History**: A track record of responsible credit use
    3. **Debt-to-Income Ratio**: How much of your income goes to debt payments
    4. **Employment History**: Longer employment suggests financial stability
    5. **Age**: Lenders often consider age as it relates to financial experience
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Tips to Improve Approval Chances")
        st.markdown("""
        - ✅ Pay down existing debts
        - ✅ Correct errors on your credit report
        - ✅ Maintain stable employment
        - ✅ Build credit history with smaller credit lines
        - ✅ Keep credit utilization below 30%
        """)
    
    with col2:
        st.subheader("Warning Signs for Lenders")
        st.markdown("""
        - ⚠️ Multiple recent credit applications
        - ⚠️ High existing debt levels
        - ⚠️ History of late payments
        - ⚠️ Unstable employment history
        - ⚠️ Limited credit history
        """)
    
   