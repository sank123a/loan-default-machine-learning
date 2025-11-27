📌 Loan Default Prediction using Machine Learning & Streamlit

Predicting loan default risk is a critical task for banks and financial institutions. This project provides an end-to-end Machine Learning pipeline that processes applicant data and predicts the probability of loan default. It includes a fully interactive Streamlit web application for real-time predictions.

⭐ Features

End-to-end ML workflow

Streamlit UI for real-time predictions

Random Forest classification pipeline

OneHotEncoding for categorical variables

Joblib-based model saving for fast inference

Clean folder structure

Easy deployment on Streamlit Cloud / Render

🔁 Project Flow
1. Data Collection

Loan dataset with applicant details stored in:

data/loan_data.csv

2. Data Preprocessing

Handle missing values

Encode categorical variables

Convert target to binary (0 = Non-default, 1 = Default)

Train-test split

3. Feature Engineering

Identify numeric + categorical columns

Apply OneHotEncoder

Combine transformations using ColumnTransformer

4. Model Training

Random Forest Classifier

Wrapped inside a scikit-learn Pipeline

Performance evaluation (Accuracy, Precision, Recall, F1 Score)

5. Model Saving

Serialized using:

models/loan_default_model.pkl

6. Streamlit App

Interactive UI for:

Applicant input

Default probability calculation

Displaying prediction result with risk score

Run using:

streamlit run app.py

📁 Project Structure
loan-default-prediction/
│
├── data/
│   └── loan_data.csv
│
├── models/
│   └── loan_default_model.pkl
│
├── train_model.py
├── app.py
└── requirements.txt

🔧 Installation & Setup
1. Clone Repository
git clone https://github.com/your-username/loan-default-prediction.git
cd loan-default-prediction

2. Create Virtual Environment
python -m venv venv

3. Activate Environment

Windows:

venv\Scripts\activate


Mac/Linux:

source venv/bin/activate

4. Install Dependencies
pip install -r requirements.txt

5. Train Model
python train_model.py

6. Run Streamlit App
streamlit run app.py

📸 Screenshots

Replace the links with your actual uploaded image URLs after generating screenshots.

Home Page

Input Form

Prediction Output

🎞 Demo GIF

(Replace after recording a GIF)

📊 Technologies Used

Python

Pandas

Scikit-learn

Streamlit

NumPy

Joblib

🚀 Deployment Options

You can deploy this Streamlit app on:

Streamlit Cloud

Render

Hugging Face Spaces

Docker

📄 License

This project is free to use under the MIT License.

🙌 Author

Sanket Kailas Kharpade

linkedin Profile : https://www.linkedin.com/in/sanket-kharpade-a2355922b 
