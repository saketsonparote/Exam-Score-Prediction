# 🎓 Exam Score Predictor

An end-to-end Machine Learning web application that predicts a student's exam score based on their study habits and lifestyle inputs — built with **XGBoost** and deployed using **Streamlit**.

🔗 **Live App:** [https://exam-score-prediction-xg.streamlit.app/](https://exam-score-prediction-xg.streamlit.app/)

---

## 📌 Project Overview

This project uses a tuned XGBoost regression model trained on student performance data to predict exam scores in real time. The app takes 6 key inputs — study hours, class attendance, sleep hours, sleep quality, study method, and facility rating — and returns a predicted score along with a grade, personalised feedback, and improvement tips.

---

## 🗂️ Project Structure

```
📁 Exam Score Prediction
│
├── main.py                     # Streamlit UI (main application)
├── model.ipynb                 # EDA + Model training notebook
├── Exam_Score_Prediction.csv   # Dataset
├── xgb_tuned_model.pkl         # Trained & tuned XGBoost model
├── label_encoders.pkl          # Label encoders for categorical features
└── requirements.txt            # Python dependencies
```

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Algorithm | XGBoost Regressor (Tuned) |
| Task | Regression (Exam Score Prediction) |
| Target Variable | `exam_score` (0–100) |
| Encoding | LabelEncoder (scikit-learn) |

### Features Used

| Feature | Type | Description |
|---|---|---|
| `study_hours` | Numerical | Average daily study hours (0–8) |
| `class_attendance` | Numerical | Class attendance percentage (40–100%) |
| `sleep_hours` | Numerical | Average sleep per night (4–10 hrs) |
| `sleep_quality` | Categorical | poor / average / good |
| `study_method` | Categorical | self-study / group study / online videos / coaching / mixed |
| `facility_rating` | Categorical | low / medium / high |

> `student_id` column is excluded as it carries no predictive value.

---

## 🚀 Getting Started (Run Locally)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/exam-score-prediction.git
cd exam-score-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run main.py
```

The app will open at `http://localhost:8501`

---

## 📦 Requirements

```
pandas
scikit-learn
xgboost
streamlit
```

---

## 🖥️ App Features

- 🎛️ **Interactive sliders** for numerical inputs (study hours, attendance, sleep)
- 📋 **Dropdowns** for categorical inputs (sleep quality, study method, facility rating)
- 🔮 **Instant prediction** with a single click
- 🏆 **Grade badge** — A+ to F with colour coding
- 📊 **Input summary tiles** — clean overview after prediction
- 💡 **Personalised tips** — actionable suggestions based on weak areas
- 🌙 **Dark gradient UI** — modern, polished design

---

## 📊 Dataset

- **File:** `Exam_Score_Prediction.csv`
- **Columns:** `student_id`, `age`, `gender`, `course`, `study_hours`, `class_attendance`, `internet_access`, `sleep_hours`, `sleep_quality`, `study_method`, `facility_rating`, `exam_difficulty`, `exam_score`
- The notebook (`model.ipynb`) covers full EDA, feature encoding, model training, and hyperparameter tuning.

---

## 🌐 Deployment

The app is deployed on **Streamlit Community Cloud**.

To deploy your own version:
1. Push the project to a GitHub repository
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your repo and set `main.py` as the entry point
4. Deploy 🚀

---

## 🤝 Author

Made with 💜 using Python, XGBoost, and Streamlit.
