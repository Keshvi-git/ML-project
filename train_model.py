import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle

print("Loading data...")
df = pd.read_csv('cardio.csv')
X = df.drop('cardio', axis=1)
y = df['cardio']

algorithms = {
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Naive Bayes': GaussianNB()
}

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

print("\n=== CASE 1: Testing different train-test split sizes ===\n")
test_sizes = [0.1, 0.2, 0.3, 0.4]
case1_results = {name: [] for name in algorithms.keys()}

for name, model in algorithms.items():
    print(f"Testing {name}...")
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

print("\n=== CASE 2: Cross-validation (5-Fold) ===\n")
case2_results = {}

for name, model in algorithms.items():
    print(f"Testing {name} with cross-validation...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    case2_results[name] = {
        'scores': cv_scores,
        'mean': cv_scores.mean(),
        'std': cv_scores.std()
    }
    
    # Store in table
    idx = results_table['Algorithm'].index(name)
    results_table['Case2_CV_Mean'][idx] = cv_scores.mean()
    results_table['Case2_CV_Std'][idx] = cv_scores.std()

print("\n=== CASE 3: Hyperparameter tuning ===\n")

param_grids = {
    'Logistic Regression': {
        'C': [0.1, 1, 10],
        'max_iter': [500, 1000, 2000]
    },
    'KNN': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    },
    'Decision Tree': {
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None]
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.05, 0.1, 0.2]
    },
    'Naive Bayes': {
        'var_smoothing': [1e-9, 1e-8, 1e-7]
    }
}

case3_results = {}

for name, model in algorithms.items():
    print(f"Tuning hyperparameters for {name}...")
    param_grid = param_grids[name]
    
    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X, y)
    
    case3_results[name] = {
        'best_score': grid.best_score_,
        'best_params': grid.best_params_,
        'all_scores': grid.cv_results_['mean_test_score']
    }
    
    # Store in table
    idx = results_table['Algorithm'].index(name)
    results_table['Case3_Best_Accuracy'][idx] = grid.best_score_
    
    print(f"  Best accuracy: {grid.best_score_:.4f}")
    print(f"  Best params: {grid.best_params_}")


print("\n=== Creating Comparison Table ===\n")
results_df = pd.DataFrame(results_table)
results_df = results_df.round(4)

results_df.to_csv('accuracy_comparison_table.csv', index=False)
print("Comparison table saved as 'accuracy_comparison_table.csv'")

print("\n" + "="*100)
print("ACCURACY COMPARISON TABLE - ALL ALGORITHMS")
print("="*100)
print(results_df.to_string(index=False))
print("="*100)

print("\n")
print(f"{'Algorithm':<25} {'Case 1 (0.1)':<12} {'Case 1 (0.2)':<12} {'Case 1 (0.3)':<12} {'Case 1 (0.4)':<12} {'Case 2 CV Mean':<15} {'Case 2 CV Std':<15} {'Case 3 Best':<12}")
print("-" * 100)
for _, row in results_df.iterrows():
    print(f"{row['Algorithm']:<25} {row['Case1_TestSize_0.1']:<12.4f} {row['Case1_TestSize_0.2']:<12.4f} {row['Case1_TestSize_0.3']:<12.4f} {row['Case1_TestSize_0.4']:<12.4f} {row['Case2_CV_Mean']:<15.4f} {row['Case2_CV_Std']:<15.4f} {row['Case3_Best_Accuracy']:<12.4f}")
print("-" * 100)

print("\n=== Saving Best Model ===")
best_model_name = results_df.loc[results_df['Case3_Best_Accuracy'].idxmax(), 'Algorithm']
best_model = algorithms[best_model_name]
best_params = case3_results[best_model_name]['best_params']

final_model = type(best_model)(**best_params)
final_model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

print(f"Best model: {best_model_name}")
print(f"Best accuracy: {case3_results[best_model_name]['best_score']:.4f}")
print("Model saved as 'model.pkl'")

print("\n✓ All analysis complete!")
