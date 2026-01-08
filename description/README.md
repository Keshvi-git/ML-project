# Cardiovascular Disease Prediction - Flask Web Application

A simple Flask web application for predicting cardiovascular disease using machine learning with comprehensive performance metrics visualization.

## 🎯 Features

- **Two-Page Design:**
  - **Home Page**: Interactive prediction form with detailed instructions and helpful tooltips
  - **Metrics Page**: Comprehensive performance analysis with 7 different graphs

- **7 Performance Metric Graphs:**
  - Confusion Matrix
  - ROC Curve
  - Feature Importance
  - Performance Metrics Comparison
  - Precision-Recall Curve
  - Prediction Probability Distribution
  - Classification Report Heatmap

- **User-Friendly Features:**
  - Clear instructions for each form field
  - Auto-calculation for BMI and Pulse Pressure
  - Beautiful, peaceful color scheme (Teal/Green theme)
  - Responsive design that works on all devices
  - Smooth animations and transitions

- **Simple and Easy to Understand Code**

## 📋 Requirements

- Python 3.7+
- Required packages listed in `requirements.txt`

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (First Time Only)

```bash
python train_model.py
```

This will create `model.pkl` file.

### 3. Run the Flask Application

```bash
python app.py
```

The application will start at `http://localhost:5000`

Open your browser and visit: `http://localhost:5000`

## 📁 Project Structure

```
.
├── app.py                      # Main Flask application
├── train_model.py              # Script to train and save model
├── cardio.csv                  # Cleaned dataset
├── model.pkl                   # Trained model (created after training)
├── requirements.txt            # Python dependencies
├── templates/
│   ├── home.html              # Home page with prediction form
│   └── metrics.html           # Performance metrics page with graphs
└── README.md                   # This file
```

## 🌐 Deployment to Free Hosting Sites

### Option 1: Render.com (Recommended - Free)

1. Create account at [render.com](https://render.com)
2. Create a new "Web Service"
3. Connect your GitHub repository
4. Settings:
   - **Build Command:** `pip install -r requirements.txt && python train_model.py`
   - **Start Command:** `python app.py`
   - **Environment:** Python 3
5. Add environment variable if needed: `PORT=5000`
6. Deploy!

### Option 2: PythonAnywhere (Free)

1. Create account at [pythonanywhere.com](https://www.pythonanywhere.com)
2. Upload all files via Files tab
3. Go to Web tab and create new web app
4. Edit WSGI file to point to your app
5. Reload web app

### Option 3: Heroku (Free Tier Available)

1. Install Heroku CLI
2. Login: `heroku login`
3. Create app: `heroku create your-app-name`
4. Create `Procfile` with: `web: python app.py`
5. Deploy: `git push heroku main`

### Option 4: Railway.app (Free Credits)

1. Create account at [railway.app](https://railway.app)
2. New Project → Deploy from GitHub
3. Connect repository
4. Railway auto-detects Python and runs the app

## 📊 Model Information

- **Algorithm:** Gradient Boosting Classifier
- **Dataset:** Cardiovascular Disease Dataset (cleaned)
- **Features:** 13 features including age, height, weight, blood pressure, cholesterol, etc.
- **Target:** Binary classification (0 = No Disease, 1 = Has Disease)

## 🎨 Technologies Used

- **Backend:** Flask (Python)
- **Frontend:** HTML, CSS, JavaScript
- **ML Libraries:** scikit-learn, pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly

## 📝 Notes

- The model is trained on `cardio.csv` dataset
- Make sure `cardio.csv` file is in the same directory
- Model will auto-train on first run if `model.pkl` doesn't exist
- All graphs are generated dynamically when you visit the metrics page
- The homepage focuses on the prediction form with clear instructions
- BMI and Pulse Pressure are auto-calculated when you enter the required fields

## 🔧 Troubleshooting

- If graphs don't load, check that `cardio.csv` exists
- If prediction fails, ensure all form fields are filled correctly
- For deployment issues, check that `PORT` environment variable is set correctly


