# How to Run and View Output

## ✅ Step-by-Step Instructions

### 1. Open Terminal/Command Prompt
Navigate to your project folder:
```bash
cd D:\MLDL\Project
```

### 2. Install Dependencies (Already Done!)
```bash
pip install -r requirements.txt
```

### 3. Train Model (Already Done!)
The model is already trained and saved as `model.pkl`

### 4. Run Flask Application
```bash
python app.py
```

You should see output like:
```
 * Running on http://127.0.0.1:5000
 * Running on http://0.0.0.0:5000
```

### 5. Open Your Web Browser
Open any web browser (Chrome, Firefox, Edge, etc.) and visit:

**http://localhost:5000**

or

**http://127.0.0.1:5000**

## 📊 What You'll See

### On the Homepage:

1. **Performance Metrics Cards** (at the top):
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - ROC-AUC Score

2. **7 Performance Graphs**:
   - **Confusion Matrix** - Shows correct vs incorrect predictions
   - **ROC Curve** - Shows model's ability to distinguish classes
   - **Feature Importance** - Which features matter most
   - **Performance Metrics Comparison** - Bar chart comparing all metrics
   - **Precision-Recall Curve** - Balance between precision and recall
   - **Prediction Probability Distribution** - How predictions are distributed
   - **Classification Report** - Detailed performance breakdown

3. **Prediction Form** (at the bottom):
   - Fill in patient information
   - Click "Predict" button
   - See the prediction result (Has Disease or No Disease)
   - See probability percentages

## 🎯 Making Predictions

Fill in the form with patient data:
- **Gender**: 0 (Female) or 1 (Male)
- **Height**: in cm (e.g., 165)
- **Weight**: in kg (e.g., 70.5)
- **Systolic BP (ap_hi)**: Blood pressure (e.g., 120)
- **Diastolic BP (ap_lo)**: Blood pressure (e.g., 80)
- **Cholesterol**: 1 (normal), 2 (above normal), 3 (high)
- **Glucose**: 1 (normal), 2 (above normal), 3 (high)
- **Smoke**: 0 (No) or 1 (Yes)
- **Alcohol**: 0 (No) or 1 (Yes)
- **Active**: 0 (No) or 1 (Yes)
- **Age (years)**: e.g., 45
- **Pulse Pressure**: ap_hi - ap_lo (e.g., 40)
- **BMI**: Weight / (Height/100)² (e.g., 25.5)

Click **"Predict"** to see the result!

## 🛑 To Stop the Server

Press `Ctrl + C` in the terminal where Flask is running

## ❓ Troubleshooting

### If graphs don't load:
- Make sure `cardio.csv` file exists in the project folder
- Check browser console for errors (F12)

### If you see "Port already in use":
- Close any other Flask apps running on port 5000
- Or change port in `app.py` last line: `port=5001`

### If prediction doesn't work:
- Make sure all form fields are filled
- Check that values are within valid ranges

---

**🎉 Enjoy exploring your ML model's performance metrics!**

