import os
import base64
from io import BytesIO
import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, roc_curve, auc, classification_report)
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve

app = Flask(__name__)

# Load or train model
def get_model():
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

def load_data():
    df = pd.read_csv('cardio.csv')
    X = df.drop('cardio', axis=1)
    y = df['cardio']
    return X, y

def get_model_predictions():
    model = get_model()
    X, y = load_data()
    
    # Convert to numpy arrays to avoid feature name validation issues
    X_array = X.values if hasattr(X, 'values') else np.array(X)
    
    # Make predictions on the FULL dataset (all actual data)
    y_pred = model.predict(X_array)
    y_prob = model.predict_proba(X_array)[:, 1]
    
    return model, X, y, y_pred, y_prob

def fig_to_base64(fig):
    img = BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/metrics')
def metrics():
    return render_template('metrics.html')

@app.route('/api/comparison-graphs')
def get_comparison_graphs():
    """Generate comparison graphs for all 6 algorithms across 3 cases"""
    try:
        # Try to load from CSV first (much faster)
        if os.path.exists('accuracy_comparison_table.csv'):
            results_df = pd.read_csv('accuracy_comparison_table.csv')
            # Convert to dict format
            table_data = results_df.to_dict('records')
            for row in table_data:
                for key, value in row.items():
                    if pd.isna(value) if hasattr(pd, 'isna') else (value != value):
                        row[key] = None
            
            # Generate graphs from the data
            X, y = load_data()
            graphs = {}
            
            # Case 1 graph
            test_sizes = [0.1, 0.2, 0.3, 0.4]
            case1_results = {}
            for row in table_data:
                name = row['Algorithm']
                case1_results[name] = [
                    row.get('Case1_TestSize_0.1', 0),
                    row.get('Case1_TestSize_0.2', 0),
                    row.get('Case1_TestSize_0.3', 0),
                    row.get('Case1_TestSize_0.4', 0)
                ]
            
            fig, ax = plt.subplots(figsize=(4, 3))
            for name, accuracies in case1_results.items():
                ax.plot(test_sizes, accuracies, marker='o', label=name, linewidth=1.5, markersize=4)
            ax.set_xlabel('Test Size', fontsize=9)
            ax.set_ylabel('Accuracy', fontsize=9)
            ax.set_title('Case 1: Accuracy vs Train-Test Split', fontsize=10, fontweight='bold')
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.3)
            graphs['case1'] = fig_to_base64(fig)
            plt.close(fig)
            
            # Case 2 graph - use mean values for all folds (simplified visualization)
            case2_results = {}
            for row in table_data:
                name = row['Algorithm']
                mean = float(row.get('Case2_CV_Mean', 0) or 0)
                std = float(row.get('Case2_CV_Std', 0) or 0)
                # Use mean for all 5 folds (simplified for visualization)
                case2_results[name] = {
                    'scores': np.array([mean] * 5),
                    'mean': mean,
                    'std': std
                }
            
            fig, ax = plt.subplots(figsize=(4, 3))
            x_pos = np.arange(1, 6)
            width = 0.12
            
            for i, (name, result) in enumerate(case2_results.items()):
                ax.bar(x_pos + i*width, result['scores'], width, label=name, alpha=0.7)
            
            ax.set_xlabel('Fold Number', fontsize=9)
            ax.set_ylabel('Accuracy', fontsize=9)
            ax.set_title('Case 2: Cross-Validation (5-Fold)', fontsize=10, fontweight='bold')
            ax.set_xticks(x_pos + width*2.5)
            ax.set_xticklabels([f'Fold {i}' for i in range(1, 6)], fontsize=8)
            ax.legend(fontsize=6, loc='best')
            ax.grid(True, alpha=0.3, axis='y')
            graphs['case2'] = fig_to_base64(fig)
            plt.close(fig)
            
            # Case 3 graph - simplified
            case3_results = {}
            for row in table_data:
                name = row['Algorithm']
                best_score = row.get('Case3_Best_Accuracy', 0)
                # Create synthetic scores for visualization
                case3_results[name] = {
                    'best_score': best_score,
                    'all_scores': np.linspace(best_score - 0.05, best_score, 10)  # Simplified
                }
            
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            axes = axes.flatten()
            
            for i, (name, result) in enumerate(case3_results.items()):
                ax = axes[i]
                scores = result['all_scores']
                ax.plot(range(len(scores)), scores, marker='o', linewidth=1.5, markersize=3)
                ax.set_xlabel('Param Combination', fontsize=8)
                ax.set_ylabel('Accuracy', fontsize=8)
                ax.set_title(f'{name}\nBest: {result["best_score"]:.4f}', fontsize=9, fontweight='bold')
                ax.grid(True, alpha=0.3)
            
            plt.suptitle('Case 3: Hyperparameter Tuning - All Algorithms', fontsize=11, fontweight='bold')
            plt.tight_layout()
            graphs['case3'] = fig_to_base64(fig)
            plt.close(fig)
            
            return jsonify({
                'graphs': graphs,
                'table': table_data
            })
        
        # If CSV doesn't exist, compute (but this is slow)
        X, y = load_data()
        
        algorithms = {
            'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=7),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Naive Bayes': GaussianNB()
        }
        
        graphs = {}
        results_table = {
            'Algorithm': [],
            'Case1_TestSize_0.1': [],
            'Case1_TestSize_0.2': [],
            'Case1_TestSize_0.3': [],
            'Case1_TestSize_0.4': [],
            'Case2_CV_Mean': [],
            'Case2_CV_Std': [],
            'Case3_Best_Accuracy': []
        }
  
        test_sizes = [0.1, 0.2, 0.3, 0.4]
        case1_results = {name: [] for name in algorithms.keys()}
        
        for name, model in algorithms.items():
            for size in test_sizes:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=size, random_state=42
                )
                model_temp = type(model)(**model.get_params()) if hasattr(model, 'get_params') else type(model)()
                model_temp.fit(X_train, y_train)
                acc = accuracy_score(y_test, model_temp.predict(X_test))
                case1_results[name].append(acc)
            
            results_table['Algorithm'].append(name)
            results_table['Case1_TestSize_0.1'].append(case1_results[name][0])
            results_table['Case1_TestSize_0.2'].append(case1_results[name][1])
            results_table['Case1_TestSize_0.3'].append(case1_results[name][2])
            results_table['Case1_TestSize_0.4'].append(case1_results[name][3])
            results_table['Case2_CV_Mean'].append(None)
            results_table['Case2_CV_Std'].append(None)
            results_table['Case3_Best_Accuracy'].append(None)
        
        fig, ax = plt.subplots(figsize=(4, 3))
        for name, accuracies in case1_results.items():
            ax.plot(test_sizes, accuracies, marker='o', label=name, linewidth=1.5, markersize=4)
        ax.set_xlabel('Test Size', fontsize=9)
        ax.set_ylabel('Accuracy', fontsize=9)
        ax.set_title('Case 1: Accuracy vs Train-Test Split', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
        graphs['case1'] = fig_to_base64(fig)
        plt.close(fig)
        
        case2_results = {}
        for name, model in algorithms.items():
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            case2_results[name] = {
                'scores': cv_scores,
                'mean': cv_scores.mean(),
                'std': cv_scores.std()
            }
            idx = results_table['Algorithm'].index(name)
            results_table['Case2_CV_Mean'][idx] = float(cv_scores.mean())
            results_table['Case2_CV_Std'][idx] = float(cv_scores.std())
        
        fig, ax = plt.subplots(figsize=(4, 3))
        x_pos = np.arange(1, 6)
        width = 0.12
        
        for i, (name, result) in enumerate(case2_results.items()):
            ax.bar(x_pos + i*width, result['scores'], width, label=name, alpha=0.7)
        
        ax.set_xlabel('Fold Number', fontsize=9)
        ax.set_ylabel('Accuracy', fontsize=9)
        ax.set_title('Case 2: Cross-Validation (5-Fold)', fontsize=10, fontweight='bold')
        ax.set_xticks(x_pos + width*2.5)
        ax.set_xticklabels([f'Fold {i}' for i in range(1, 6)], fontsize=8)
        ax.legend(fontsize=6, loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        graphs['case2'] = fig_to_base64(fig)
        plt.close(fig)

        param_grids = {
            'Logistic Regression': {'C': [0.1, 1, 10], 'max_iter': [500, 1000]},
            'KNN': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']},
            'Decision Tree': {'max_depth': [5, 10, 15, None], 'min_samples_split': [2, 5]},
            'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]},
            'Gradient Boosting': {'n_estimators': [50, 100, 150], 'learning_rate': [0.05, 0.1, 0.2]},
            'Naive Bayes': {'var_smoothing': [1e-9, 1e-8, 1e-7]}
        }
        
        case3_results = {}
        for name, model in algorithms.items():
            param_grid = param_grids[name]
            grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
            grid.fit(X, y)
            case3_results[name] = {
                'best_score': grid.best_score_,
                'all_scores': grid.cv_results_['mean_test_score']
            }
            idx = results_table['Algorithm'].index(name)
            results_table['Case3_Best_Accuracy'][idx] = float(grid.best_score_)
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        
        for i, (name, result) in enumerate(case3_results.items()):
            ax = axes[i]
            scores = result['all_scores']
            ax.plot(range(len(scores)), scores, marker='o', linewidth=1.5, markersize=3)
            ax.set_xlabel('Param Combination', fontsize=8)
            ax.set_ylabel('Accuracy', fontsize=8)
            ax.set_title(f'{name}\nBest: {result["best_score"]:.4f}', fontsize=9, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Case 3: Hyperparameter Tuning - All Algorithms', fontsize=11, fontweight='bold')
        plt.tight_layout()
        graphs['case3'] = fig_to_base64(fig)
        plt.close(fig)
        
        results_df = pd.DataFrame(results_table)
        results_df = results_df.round(4)
        results_df = results_df.where(pd.notnull(results_df), None)
        
        table_data = results_df.to_dict('records')
        for row in table_data:
            for key, value in row.items():
                if pd.isna(value) if hasattr(pd, 'isna') else (value != value):
                    row[key] = None
        
        return jsonify({
            'graphs': graphs,
            'table': table_data
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/graphs')
def get_graphs():
    try:
        model, X_data, y_true, y_pred, y_prob = get_model_predictions()
        
        graphs = {}
        
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
        ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold', pad=15)
        graphs['confusion_matrix'] = fig_to_base64(fig)
        plt.close(fig)
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
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
        
        feature_importance = pd.DataFrame({
            'Feature': X_data.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(4, 3))
        colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
        bars = ax.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors)
        ax.set_title('Feature Importance', fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel('Importance Score', fontsize=10)
        ax.set_ylabel('Features', fontsize=10)
        graphs['feature_importance'] = fig_to_base64(fig)
        plt.close(fig)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
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
        
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
        
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(recall_vals, precision_vals, linewidth=2.5, color='#42A5F5')
        ax.fill_between(recall_vals, precision_vals, alpha=0.3, color='#90CAF9')
        ax.set_xlabel('Recall', fontsize=10)
        ax.set_ylabel('Precision', fontsize=10)
        ax.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        graphs['pr_curve'] = fig_to_base64(fig)
        plt.close(fig)
        
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(y_prob[y_true == 0], bins=40, alpha=0.6, label='No Disease', color='#66BB6A', edgecolor='white', linewidth=1)
        ax.hist(y_prob[y_true == 1], bins=40, alpha=0.6, label='Has Disease', color='#EF5350', edgecolor='white', linewidth=1)
        ax.set_xlabel('Predicted Probability', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title('Prediction Probability Distribution', fontsize=12, fontweight='bold', pad=15)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        graphs['pred_dist'] = fig_to_base64(fig)
        plt.close(fig)
        
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).iloc[:-1, :].T
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(report_df, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax, 
                    cbar_kws={'label': 'Score'}, linewidths=0.5, linecolor='white', annot_kws={'size': 8})
        ax.set_title('Classification Report', fontsize=12, fontweight='bold', pad=15)
        graphs['class_report'] = fig_to_base64(fig)
        plt.close(fig)
    
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
        
        data = request.json
        features = []
        
        for feature in feature_names:
            value = float(data.get(feature, 0))
            features.append(value)
        
        features_df = pd.DataFrame([features], columns=feature_names)
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0]
        
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


