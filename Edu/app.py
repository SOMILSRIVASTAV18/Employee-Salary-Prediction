import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---

MODEL_PATH = 'model.pkl'

CATEGORICAL_COLS = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'gender', 'native-country'
]
NUMERICAL_COLS = [
    'age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week'
]
TARGET_COL = 'income' # The target column for prediction

# --- Load Model (Placeholder) ---

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    st.sidebar.success("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Error: Model file '{MODEL_PATH}' not found. Please train and save your model.")
    model = None
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# --- Data Preprocessing Function (Must match training preprocessing) ---
def preprocess_data(df, model_feature_names_in_order):
    """
    Applies preprocessing steps to the input DataFrame.
    Handles missing values (replacing '?'), encoding categorical features,
    and critically, aligning columns with the trained model.
    Row dropping (e.g., 'Without-pay') should be handled *before* calling this function.

    Args:
        df (pd.DataFrame): The raw input DataFrame from the user, potentially already filtered.
        model_feature_names_in_order (list): A list of feature names (columns) that the
                                             trained model expects, in the exact order.
    Returns:
        pd.DataFrame: The processed DataFrame ready for prediction.
    """
    df_processed = df.copy()

    # 1. Handle missing values (replace '?' with 'Others')
    for col in ['workclass', 'occupation', 'native-country']:
        if col in df_processed.columns: # Check if column exists in the uploaded data
            df_processed[col] = df_processed[col].replace('?', 'Others')

    # 2. Explicitly drop 'fnlwgt' if it exists, as it was dropped during training
    if 'fnlwgt' in df_processed.columns:
        df_processed = df_processed.drop('fnlwgt', axis=1)

    # 3. Encode categorical features using one-hot encoding
    current_categorical_cols = [col for col in CATEGORICAL_COLS if col in df_processed.columns and col != 'fnlwgt'] # Exclude fnlwgt from categorical check if it was dropped
    df_processed = pd.get_dummies(df_processed, columns=current_categorical_cols, drop_first=True)

    # 4. Ensure numerical columns (that are still present) are numeric
    # Filter NUMERICAL_COLS to only include those still in df_processed after 'fnlwgt' drop
    current_numerical_cols_present = [col for col in NUMERICAL_COLS if col in df_processed.columns and col != 'fnlwgt']
    for col in current_numerical_cols_present:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)

    # 5. CRITICAL STEP: Align columns with the model's expected features
    df_processed = df_processed.reindex(columns=model_feature_names_in_order, fill_value=0)

    return df_processed

# --- Streamlit Application ---
st.set_page_config(page_title="Employee Salary Prediction", layout="wide")

st.title("📊 Employee Salary/Income Prediction App")
st.markdown("""
    This application allows you to predict employee income levels based on various attributes.
    Upload a CSV file containing employee data, and the model will predict whether their
    income is <=50K or >50K.
""")

st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    st.subheader("1. Uploaded Data Preview")
    try:
        batch_data_raw = pd.read_csv(uploaded_file)
        st.write(batch_data_raw.head())
        st.write(f"Shape of uploaded data: {batch_data_raw.shape[0]} rows, {batch_data_raw.shape[1]} columns")

        if model is not None:
            model_feature_names = None
            if hasattr(model, 'feature_names_in_'):
                model_feature_names = list(model.feature_names_in_) # Convert to list for reindex
            else:
                st.error("Could not retrieve feature names from the loaded model. Ensure your model supports 'feature_names_in_'.")
                st.info("Prediction might fail if column order/names don't match training data.")

            if model_feature_names is not None:
                st.subheader("2. Preprocessing Data...")
                try:
                    # Apply row-dropping filters to batch_data_raw *before* sending to preprocess_data
                    # This ensures the raw data used for results_df matches the processed data length
                    filtered_batch_data_for_processing = batch_data_raw.copy()
                    
                    if 'workclass' in filtered_batch_data_for_processing.columns:
                        filtered_batch_data_for_processing = filtered_batch_data_for_processing[
                            (filtered_batch_data_for_processing['workclass'] != 'Without-pay') &
                            (filtered_batch_data_for_processing['workclass'] != 'Never-worked')
                        ]

                    # Drop the target column from the filtered data before processing
                    data_to_process = filtered_batch_data_for_processing.drop(columns=[TARGET_COL], errors='ignore')

                    processed_data = preprocess_data(data_to_process, model_feature_names)
                    st.write("Data preprocessed successfully. Ready for prediction.")

                    st.subheader("3. Making Predictions...")
                    batch_preds = model.predict(processed_data)

                    predicted_labels = pd.Series(batch_preds).map({0: '<=50K', 1: '>50K'})

                    st.subheader("✅ 4. Predictions Complete!")
                    # Use the filtered_batch_data_for_processing as the base for results_df
                    results_df = filtered_batch_data_for_processing.copy()
                    results_df['Predicted_Income'] = predicted_labels.values # Now lengths should match

                    st.write(results_df.head())
                    st.write(f"Total predictions made: {len(batch_preds)}")

                    st.subheader("Download Results")
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions CSV",
                        data=csv,
                        file_name='predicted_income_results.csv',
                        mime='text/csv',
                    )

                except Exception as e:
                    st.error(f"An error occurred during preprocessing or prediction: {e}")
                    st.info("Please ensure your uploaded CSV has the expected columns and format.")
            else:
                st.warning("Cannot proceed with prediction: Model feature names not available.")
        else:
            st.warning("Model not loaded. Cannot perform predictions.")

    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
        st.info("Please ensure it's a valid CSV and check its encoding.")

else:
    st.info("Please upload a CSV file to get started with predictions.")

st.sidebar.markdown("---")
st.sidebar.info("Developed for Capstone Project")
