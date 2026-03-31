# train_model.py
# ============================================
# PART 1: IMPORT LIBRARIES
# ============================================

import pandas as pd          # For handling data (like Excel in Python)
import numpy as np           # For mathematical operations
from sklearn.model_selection import train_test_split  # To split data for training/testing
from sklearn.linear_model import LogisticRegression   # The ML algorithm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # To check performance
import joblib                # To save the trained model
import warnings
warnings.filterwarnings('ignore')  # Ignore warning messages

# WHAT THIS PART DOES:
# Libraries are like toolboxes. Each import gives us specific tools:
# - pandas: Read CSV files, create tables
# - sklearn: Machine learning tools
# - joblib: Save our trained model to use later

# ============================================
# PART 2: LOAD THE DATASET
# ============================================

print("="*50)
print("STEP 1: LOADING DATASET")
print("="*50)

# URL where the dataset is stored online
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

# Column names for our dataset
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# Read the CSV file from URL
df = pd.read_csv(url, names=columns)

# Show basic info
print(f"✅ Dataset loaded!")
print(f"📊 Total patients: {len(df)}")
print(f"📋 Features: {len(columns)-1}")
print(f"🎯 Target: Outcome (0=No Diabetes, 1=Diabetes)")

# Show first 5 rows to verify
print("\n👀 First 5 rows of data:")
print(df.head())

# WHAT THIS PART DOES:
# - Downloads the famous Pima Indians Diabetes dataset
# - Each row is one patient
# - Each column is a health measurement
# - 'Outcome' is what we want to predict (1=diabetes, 0=no diabetes)

# ============================================
# PART 3: DATA CLEANING
# ============================================

print("\n" + "="*50)
print("STEP 2: CLEANING DATA")
print("="*50)

# Check for zeros in medical measurements (shouldn't be zero!)
print("\n🔍 Checking for invalid zeros...")
invalid_zeros = {
    'Glucose': len(df[df['Glucose'] == 0]),
    'BloodPressure': len(df[df['BloodPressure'] == 0]),
    'BMI': len(df[df['BMI'] == 0])
}

for col, count in invalid_zeros.items():
    print(f"   {col}: {count} patients have value 0 (invalid)")

# Remove rows with invalid zeros
df_clean = df[(df['Glucose'] != 0) & (df['BloodPressure'] != 0) & (df['BMI'] != 0)]
print(f"\n✅ Removed {len(df) - len(df_clean)} invalid rows")
print(f"✅ Clean data has {len(df_clean)} patients")

# WHAT THIS PART DOES:
# - Medical measurements can't be zero (you can't have zero glucose!)
# - We remove these invalid rows so model learns from real data only

# ============================================
# PART 4: PREPARE FEATURES AND TARGET
# ============================================

print("\n" + "="*50)
print("STEP 3: PREPARING DATA FOR MODEL")
print("="*50)

# X = all features (everything except Outcome)
X = df_clean.drop('Outcome', axis=1)

# y = target variable (just Outcome)
y = df_clean['Outcome']

print(f"✅ Features shape: {X.shape} (patients × measurements)")
print(f"✅ Target shape: {y.shape}")

# WHAT THIS PART DOES:
# - Split data into:
#   X: The inputs (all 8 health measurements)
#   y: What we want to predict (diabetes or not)

# ============================================
# PART 5: SPLIT INTO TRAINING AND TESTING
# ============================================

print("\n" + "="*50)
print("STEP 4: SPLITTING DATA")
print("="*50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"🎯 Training set: {len(X_train)} patients (80%)")
print(f"🎯 Testing set: {len(X_test)} patients (20%)")

# WHAT THIS PART DOES:
# - 80% of data to TEACH the model (training)
# - 20% of data to TEST the model (testing)
# - random_state=42 ensures same split every time

# ============================================
# PART 6: TRAIN THE MODEL
# ============================================

print("\n" + "="*50)
print("STEP 5: TRAINING MODEL")
print("="*50)

# Create the model
model = LogisticRegression(max_iter=1000, random_state=42)

# Train the model
model.fit(X_train, y_train)

print("✅ Model training complete!")

# WHAT THIS PART DOES:
# - LogisticRegression = algorithm that finds patterns
# - max_iter=1000 = try harder to find best patterns
# - model.fit() = the actual learning happens here!

# ============================================
# PART 7: EVALUATE MODEL
# ============================================

print("\n" + "="*50)
print("STEP 6: CHECKING MODEL ACCURACY")
print("="*50)

# Predict on training data
train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
print(f"📈 Training Accuracy: {train_acc:.2%}")

# Predict on test data
test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)
print(f"📊 Testing Accuracy: {test_acc:.2%}")

# Detailed performance report
print("\n📋 Detailed Performance:")
print(classification_report(y_test, test_pred, 
                          target_names=['No Diabetes', 'Diabetes']))

# WHAT THIS PART DOES:
# - Checks how well our model learned
# - Training accuracy: How well it learned training data
# - Testing accuracy: How well it predicts NEW data (most important!)

# ============================================
# PART 8: SAVE THE MODEL
# ============================================

print("\n" + "="*50)
print("STEP 7: SAVING MODEL")
print("="*50)

joblib.dump(model, 'diabetes_model.pkl')
print("✅ Model saved as 'diabetes_model.pkl'")

# WHAT THIS PART DOES:
# - Saves the trained model to a file
# - We can load this file later WITHOUT retraining

# ============================================
# PART 9: TEST WITH SAMPLE
# ============================================

print("\n" + "="*50)
print("STEP 8: TESTING WITH SAMPLE PATIENT")
print("="*50)

# Example patient (high risk)
sample = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]
prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0]

print("Patient: Pregnancies=6, Glucose=148, BP=72, Age=50")
print(f"🎯 Prediction: {'NO Diabetes' if prediction==0 else 'YES Diabetes'}")
print(f"📊 Risk probability: {probability[1]:.1%}")

print("\n" + "="*50)
print("✅ TRAINING COMPLETE!")
print("="*50)