# 🩺 Diabetes Prediction Web App

A machine learning web application that predicts diabetes risk based on patient health metrics.

## ✨ Features
- Real-time predictions with probability scores
- User-friendly web interface
- Health recommendations based on results
- 77% accuracy using Logistic Regression

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt

2. Train the model
   python train_model.py(run this in cmd prompt )

3. Run the app
   streamlit run app.py(run this in cmd prompt)

4. Open browser
   Go to http://localhost:8501

📊 Dataset
Source: Pima Indians Diabetes Dataset
Samples: 768 patients
Features: 8 health metrics
Target: Diabetes (0=No, 1=Yes)

🛠️ Tech Stack
Python 3.8+
Scikit-learn (Machine Learning)
Streamlit (Web App)
Pandas, NumPy (Data Processing)
Joblib (Model Persistence)

📁 Project Structure
diabetes-prediction/
├── app.py              # Web application
├── train_model.py      # Model training script
├── diabetes_model.pkl  # Trained model
├── requirements.txt    # Dependencies
└── README.md          # Documentation


📝 Sample Input
Feature	         Value
Pregnancies	       6
Glucose	          148
Blood Pressure	  72
Skin Thickness	  35
Insulin	           0
BMI	              33.6
Diabetes Pedigree 0.627
Age	               50

Result: 🔴 High Risk of Diabetes (78.5%) 

📄 License
MIT License

🙏 Acknowledgments
UCI Machine Learning Repository
Scikit-learn
Streamlit    



---

## 🚀 **How to Use**

1. **Create a folder** on your Desktop called `diabetes-prediction`

2. **Create these 4 files** in the folder:
   - `train_model.py` (copy the first code block)
   - `app.py` (copy the second code block)
   - `requirements.txt` (copy the third code block)
   - `README.md` (copy the fourth code block)

3. **Open Command Prompt** and navigate to the folder:
   ```cmd
   cd C:\Users\spandana-vandana\OneDrive\Desktop\diabetes-prediction
4.install dependencies:
cmd:
pip install -r requirements.txt


5.Train the model:
cmd:
python train_model.py


6.Run the app:
cmd:
streamlit run app.py
Open browser at http://localhost:8501
