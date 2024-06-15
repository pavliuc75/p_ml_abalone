#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import binom

def categorize_rings(rings):
    if rings <= 7:
        return 'young'
    elif 8 <= rings <= 15:
        return 'medium'
    else:
        return 'old'

file_path = r'C:\Users\jaimu\OneDrive\Desktop\abalone.csv'
data = pd.read_csv(file_path)
data['Rings_Category'] = data['Rings'].apply(categorize_rings)

# Separate features (X) and target variable (y)
X = data.drop(['Rings', 'Rings_Category'], axis=1)  # Features
y = data['Rings_Category']  # Target variable

X_encoded = pd.get_dummies(X, columns=['Sex'], drop_first=True)

#grid search
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']),
        ('cat', OneHotEncoder(), ['Sex'])
    ])

# Define logistic regression pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression(C=100, max_iter=1000))])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for grid search
param_grid = {'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]}

# Perform grid search using 5-fold cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameter and best score
best_C = grid_search.best_params_['classifier__C']
best_score = grid_search.best_score_

print("Best C (Regularization Parameter):", best_C)
print("Best Accuracy:", best_score)

import matplotlib.pyplot as plt

# Extract grid search results
results = grid_search.cv_results_
Cs = param_grid['classifier__C']
scores = results['mean_test_score']

# Plot
plt.figure(figsize=(10, 6))
plt.plot(Cs, scores, marker='o', linestyle='-')
plt.xscale('log')
plt.xlabel('C (Regularization Parameter)')
plt.ylabel('Mean Accuracy')
plt.title('Grid Search Results')
plt.grid(True)
plt.show()

###
outer_kfold = KFold(n_splits=5, shuffle=True, random_state=42)
lambda_values = np.linspace(0.005, 0.015, 5) 
b_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

log_reg_error_rates = []
log_reg_params = []
nb_error_rates = []
nb_params = []
baseline_error_rates = []
contingency_lr_baseline = []
contingency_nb_baseline = []

for fold, (train_idx, test_idx) in enumerate(outer_kfold.split(X_encoded, y), 1):
    if fold > 5:
        break
    
    X_train_outer, X_test_outer = X_encoded.iloc[train_idx], X_encoded.iloc[test_idx]
    y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]
    
    fold_log_reg_params = []
    fold_log_reg_error_rates = []
    
    fold_nb_params = []
    fold_nb_error_rates = []
    
    for lambda_val in lambda_values:
        logistic_reg = LogisticRegression(C=1/lambda_val, max_iter=1000)
        logistic_reg.fit(X_train_outer, y_train_outer)
        y_pred_log_reg = logistic_reg.predict(X_test_outer)
        log_reg_error_rate = 1 - accuracy_score(y_test_outer, y_pred_log_reg)
        fold_log_reg_params.append(lambda_val)
        fold_log_reg_error_rates.append(log_reg_error_rate)
    
        for b in b_values:
            nb = GaussianNB(var_smoothing=b)
            nb.fit(X_train_outer, y_train_outer)
            y_pred_nb = nb.predict(X_test_outer)
            nb_error_rate = 1 - accuracy_score(y_test_outer, y_pred_nb)
            fold_nb_params.append(b)
            fold_nb_error_rates.append(nb_error_rate)
    
  
    best_lambda_index = np.argmin(fold_log_reg_error_rates)
    best_lambda = fold_log_reg_params[best_lambda_index]
    log_reg_params.append(best_lambda)
    log_reg_error_rates.append(fold_log_reg_error_rates[best_lambda_index])
    best_b_index = np.argmin(fold_nb_error_rates)
    best_b = fold_nb_params[best_b_index]
    nb_params.append(best_b)
    nb_error_rates.append(fold_nb_error_rates[best_b_index])

    majority_class = y_train_outer.value_counts().idxmax()
    y_pred_baseline = np.full_like(y_test_outer, fill_value=majority_class)
    baseline_error_rate = 1 - accuracy_score(y_test_outer, y_pred_baseline)
    baseline_error_rates.append(baseline_error_rate)

    lr = LogisticRegression(C=1/best_lambda, max_iter=1000)
    lr.fit(X_train_outer, y_train_outer)
    nb = GaussianNB(var_smoothing=best_b)
    nb.fit(X_train_outer, y_train_outer)
    
    y_pred_lr = lr.predict(X_test_outer)
    y_pred_nb = nb.predict(X_test_outer)
    cm_lr_baseline = confusion_matrix(y_test_outer, y_pred_lr)
    cm_nb_baseline = confusion_matrix(y_test_outer, y_pred_nb)
    
    contingency_lr_baseline.append([
        cm_lr_baseline[0][1], cm_lr_baseline[1][0] 
    ])
    
   
    contingency_nb_baseline.append([
        cm_nb_baseline[0][1], cm_nb_baseline[1][0]  # Entries for NB incorrect and Baseline correct
    ])

#McNemar Test
mcnemar_lr_baseline = mcnemar(np.array(contingency_lr_baseline))
mcnemar_nb_baseline = mcnemar(np.array(contingency_nb_baseline))

contingency_lr_nb = []
for i in range(5):
    X_train, X_test = X_encoded.iloc[train_idx], X_encoded.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    lr = LogisticRegression(C=1/log_reg_params[i], max_iter=1000)
    lr.fit(X_train, y_train)
    nb = GaussianNB(var_smoothing=nb_params[i])
    nb.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_pred_nb = nb.predict(X_test)
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    cm_nb = confusion_matrix(y_test, y_pred_nb)
    
    contingency_lr_nb.append([
        cm_lr[0][1], cm_lr[1][0], 
        cm_nb[0][1], cm_nb[1][0]])

mcnemar_lr_nb = mcnemar(np.array(contingency_lr_nb))

# Confidence level
alpha = 0.05
conf_int_lr_baseline = binom.interval(1 - alpha, contingency_lr_baseline[0][0] + contingency_lr_baseline[0][1], 0.5, loc=contingency_lr_baseline[0][1])
conf_int_nb_baseline = binom.interval(1 - alpha, contingency_nb_baseline[0][0] + contingency_nb_baseline[0][1], 0.5, loc=contingency_nb_baseline[0][1])
conf_int_lr_nb = binom.interval(1 - alpha, contingency_lr_nb[0][0] + contingency_lr_nb[0][1], 0.5, loc=contingency_lr_nb[0][1])

#output DataFrame
output_df = pd.DataFrame({
    'Outer fold': range(1, 6),
    'b (Naive Bayes)': nb_params,
    'Naive Bayes Error': np.round(np.array(nb_error_rates) * 100, 1),
    'λ (Logistic Regression)': log_reg_params,
    'Logistic Regression Error': np.round(np.array(log_reg_error_rates) * 100, 1),
    'Baseline Error': np.round(np.array(baseline_error_rates) * 100, 1)
})

print("Output DataFrame:")
print(output_df)

#McNemar's test results
print("\nMcNemar's test results for Logistic Regression vs Baseline:")
print("Statistic:", mcnemar_lr_baseline.statistic)
print("p-value:", "{:.3e}".format(mcnemar_lr_baseline.pvalue))
print("95% Confidence Interval:", conf_int_lr_baseline)

print("\nMcNemar's test results for Naive Bayes vs Baseline:")
print("Statistic:", mcnemar_nb_baseline.statistic)
print("p-value:", "{:.3e}".format(mcnemar_nb_baseline.pvalue))
print("95% Confidence Interval:", conf_int_nb_baseline)

print("\nMcNemar's test results for Logistic Regression vs Naive Bayes:")
print("Statistic:", mcnemar_lr_nb.statistic)
print("p-value:", "{:.3e}".format(mcnemar_lr_nb.pvalue))
print("95% Confidence Interval:", conf_int_lr_nb)


# %%
# logistic regression (lambda = 0.005)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.5, random_state=42)

#lambda = 0.005 (C = 1/0.005 = 200)
logistic_reg_lambda_0005 = LogisticRegression(C=200, max_iter=1000)
logistic_reg_lambda_0005.fit(X_train, y_train)
y_pred_lambda_0005 = logistic_reg_lambda_0005.predict(X_test)
accuracy_lambda_0005 = accuracy_score(y_test, y_pred_lambda_0005)
print("Accuracy with lambda=0.005 (C=200):", accuracy_lambda_0005)

#confusion matrix
conf_matrix_lambda_0005 = confusion_matrix(y_test, y_pred_lambda_0005)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix_lambda_0005, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
classes = np.unique(y)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

