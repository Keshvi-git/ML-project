"""
Flask Application for Cardiovascular Disease Prediction
Includes all performance metrics graphs
"""

import os
import base64
from io import BytesIO
import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, roc_curve, auc, classification_report)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve

app = Flask(__name__)

# Load or train model
def get_model():
    """Load model if exists, otherwise train and save it"""
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        # Train model
        df = pd.read_csv('cardio.csv')
        X = df.drop('cardio', axis=1)
        y = df['cardio']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        return model

# Load data for graphs
def load_data():
    """Load data for visualization"""
    df = pd.read_csv('cardio.csv')
    X = df.drop('cardio', axis=1)
    y = df['cardio']
    return X, y

def get_model_predictions():
    """Get model and make fresh predictions on full dataset for metrics"""
    model = get_model()
    X, y = load_data()
    
    # Convert to numpy arrays to avoid feature name validation issues
    X_array = X.values if hasattr(X, 'values') else np.array(X)
    
    # Make predictions on the FULL dataset (all actual data)
    y_pred = model.predict(X_array)
    y_prob = model.predict_proba(X_array)[:, 1]
    
    return model, X, y, y_pred, y_prob

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    img = BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/')
def index():
    """Home page with prediction form"""
    return render_template('home.html')

@app.route('/metrics')
def metrics():
    """Performance metrics page with all graphs"""
    return render_template('metrics.html')

@app.route('/api/graphs')
def get_graphs():
    """Generate all performance metric graphs using fresh predictions"""
    try:
        # Get fresh predictions (generated in real-time, not cached)
        model, X_test, y_test, y_pred, y_prob = get_model_predictions()
        
        graphs = {}
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
        ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold', pad=15)
        graphs['confusion_matrix'] = fig_to_base64(fig)
        plt.close(fig)
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', linewidth=2.5, color='#4CAF50')
        ax.plot([0, 1], [0, 1], 'r--', label='Random Classifier', linewidth=2, alpha=0.7)
        ax.fill_between(fpr, tpr, alpha=0.3, color='#81C784')
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title('ROC Curve', fontsize=12, fontweight='bold', pad=15)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        graphs['roc_curve'] = fig_to_base64(fig)
        plt.close(fig)
        
        # 3. Feature Importance
        feature_importance = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(4, 3))
        colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
        sns.barplot(data=feature_importance, x='Importance', y='Feature', ax=ax, palette='viridis')
        ax.set_title('Feature Importance', fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel('Importance Score', fontsize=10)
        ax.set_ylabel('Features', fontsize=10)
        graphs['feature_importance'] = fig_to_base64(fig)
        plt.close(fig)
        
        # 4. Metrics Bar Chart
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        values = [accuracy, precision, recall, f1]
        
        fig, ax = plt.subplots(figsize=(4, 3))
        colors = ['#64B5F6', '#81C784', '#FFB74D', '#BA68C8']
        bars = ax.bar(metrics, values, color=colors, edgecolor='white', linewidth=1.5)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold', pad=15)
        ax.set_ylim(0, 1)
        for i, (bar, v) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        graphs['metrics_bar'] = fig_to_base64(fig)
        plt.close(fig)
        
        # 5. Precision-Recall Curve (using probabilities)
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
        
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(recall_vals, precision_vals, linewidth=2.5, color='#42A5F5')
        ax.fill_between(recall_vals, precision_vals, alpha=0.3, color='#90CAF9')
        ax.set_xlabel('Recall', fontsize=10)
        ax.set_ylabel('Precision', fontsize=10)
        ax.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        graphs['pr_curve'] = fig_to_base64(fig)
        plt.close(fig)
        
        # 6. Prediction Distribution
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(y_prob[y_test == 0], bins=40, alpha=0.6, label='No Disease', color='#66BB6A', edgecolor='white', linewidth=1)
        ax.hist(y_prob[y_test == 1], bins=40, alpha=0.6, label='Has Disease', color='#EF5350', edgecolor='white', linewidth=1)
        ax.set_xlabel('Predicted Probability', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title('Prediction Probability Distribution', fontsize=12, fontweight='bold', pad=15)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        graphs['pred_dist'] = fig_to_base64(fig)
        plt.close(fig)
        
        # 7. Classification Report Heatmap
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).iloc[:-1, :].T
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(report_df, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax, 
                    cbar_kws={'label': 'Score'}, linewidths=0.5, linecolor='white', annot_kws={'size': 8})
        ax.set_title('Classification Report', fontsize=12, fontweight='bold', pad=15)
        graphs['class_report'] = fig_to_base64(fig)
        plt.close(fig)
    
        # Return metrics values
        metrics_data = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'roc_auc': float(roc_auc)
        }
        
        return jsonify({'graphs': graphs, 'metrics': metrics_data})
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({'error': error_msg, 'traceback': traceback.format_exc()}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        model = get_model()
        X, _ = load_data()
        feature_names = list(X.columns)
        
        # Get input data
        data = request.json
        features = []
        
        for feature in feature_names:
            value = float(data.get(feature, 0))
            features.append(value)
        
        # Make prediction
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]
        
        result = {
            'prediction': int(prediction),
            'probability_no_disease': float(probability[0]),
            'probability_disease': float(probability[1]),
            'message': 'Has Cardiovascular Disease' if prediction == 1 else 'No Cardiovascular Disease'
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)


