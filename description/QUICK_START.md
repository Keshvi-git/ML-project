# Quick Start Guide

## Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 2: Train Model (First Time)
```bash
python train_model.py
```

## Step 3: Run Flask App
```bash
python app.py
```

## Step 4: Open Browser
Visit: http://localhost:5000

## 📊 What You'll See

1. **Performance Metrics Summary Cards** - Accuracy, Precision, Recall, F1 Score, ROC-AUC
2. **7 Different Graphs:**
   - Confusion Matrix
   - ROC Curve
   - Feature Importance
   - Performance Metrics Comparison
   - Precision-Recall Curve
   - Prediction Probability Distribution
   - Classification Report

3. **Prediction Form** - Enter patient data to predict cardiovascular disease

## 📁 Files Created

- `app.py` - Main Flask application with all routes
- `train_model.py` - Script to train the model
- `templates/index.html` - Beautiful frontend interface
- `requirements.txt` - All dependencies
- `README.md` - Detailed documentation
- `Procfile` - For Heroku deployment

## 🌐 To Deploy

See README.md for deployment instructions to:
- Render.com (Recommended)
- PythonAnywhere
- Heroku
- Railway.app

