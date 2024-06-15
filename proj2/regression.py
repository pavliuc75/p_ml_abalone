import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from dtuimldmtools import feature_selector_lr, train_neural_net
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import torch
import scipy.stats as st


data = pd.read_csv('abalone.csv')
data = pd.get_dummies(data, columns=['Sex'])
data = data.astype(float)

X = data.drop(['Rings'], axis=1)
y = data['Rings']
y_ann = y.to_numpy().reshape(-1, 1)

# ------------------------------------------------
# apply one-of-K for Sex column and normalize data
cols_to_standardize = X.columns.difference(['Sex_F', 'Sex_I', 'Sex_M'])
X_normalized = X.copy()
X_normalized[cols_to_standardize] = (X[cols_to_standardize] - X[cols_to_standardize].mean()) / X[
    cols_to_standardize].std()
X = X_normalized
X_ann = X.values

# ------------------------------------------------
# feature selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
selected_features, features_record, loss_record = feature_selector_lr(X_train.values, y_train.values,
                                                                      5,
                                                                      display='')
selected_features = np.array(
    [1, 3, 4, 5, 6, 8])  # the feature_selector_lr gives sometimes different outputs, so we stick to this one
m = lm.LinearRegression().fit(X_train.values[:, selected_features], y_train.values)
y_pred = m.predict(X_test.iloc[:, selected_features].values)
mse = mean_squared_error(y_test, y_pred)
print(mse)  # for reference: gen err for linear regression = 4.92
X_ann = X_ann[:, selected_features]

# ------------------------------------------------
# use l2 regularization
lambdas = np.concatenate((np.linspace(0, 3, 50), np.linspace(3, 100, 50)))
train_errors = []
test_errors = []
weights = []

k_fold = KFold(n_splits=10, shuffle=True, random_state=42)
for lambda_ in lambdas:
    train_error_fold = []
    test_error_fold = []
    fold_weights = []

    for train_indices, test_indices in k_fold.split(X):
        X_train_cv = X.values[train_indices][:, selected_features]
        y_train_cv = y.values[train_indices]
        X_test_cv = X.values[test_indices][:, selected_features]
        y_test_cv = y.values[test_indices]

        lr_model = Ridge(alpha=lambda_)
        lr_model.fit(X_train_cv, y_train_cv)
        fold_weights.append(lr_model.coef_)

        y_train_pred = lr_model.predict(X_train_cv)
        y_test_pred = lr_model.predict(X_test_cv)

        train_error_fold.append(mean_squared_error(y_train_cv, y_train_pred))
        test_error_fold.append(mean_squared_error(y_test_cv, y_test_pred))

    train_errors.append(np.mean(train_error_fold))
    test_errors.append(np.mean(test_error_fold))
    avg_weights = np.mean(fold_weights, axis=0)
    weights.append(avg_weights)

# --------------------------------------
# results
index = np.argmin(test_errors)
opt_lambda = lambdas[index]
mse_of_model_with_opt_lambda = test_errors[index]
weights_of_model_with_opt_lambda = weights[index]
print(index)
print(opt_lambda)
print(mse_of_model_with_opt_lambda)
print(weights_of_model_with_opt_lambda)
print(X.columns[selected_features])

plt.figure(figsize=(10, 6))
plt.semilogx(lambdas, train_errors, label='Train MSE')
plt.semilogx(lambdas, test_errors, label='Test MSE')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs. Regularization Strength')
plt.legend()
plt.grid(True)
plt.show()

feature_labels = [X.columns[i] for i in selected_features]
plt.figure(figsize=(10, 5))
for i in range(len(selected_features)):
    plt.plot(lambdas, np.array(weights)[:, i], label=feature_labels[i])

plt.xlabel('Lambda')
plt.ylabel('Weights')
plt.title('Change in Weights with Regularization Strength')
plt.xscale('log')  # To make the x-axis logarithmic
plt.legend(loc='best')  # Show legend to make lines identifiable
plt.grid(True)
plt.show()

# regression Part b ============================================================
# ============================================================
# ============================================================
# ============================================================
# ============================================================

lambdas1 = np.linspace(0, 3,
                       30)  # as seen previously, the linear regression performs best for lambda close to 1, so this time we test 20 lambda values around 1

hs = np.random.randint(1,6,size=20)
loss_fn = torch.nn.MSELoss()

indexes_of_picked_lambdas_for_lr = []
indexes_of_picked_hs_for_ann = []

outer_gen_errors_for_lr = []
outer_gen_errors_for_ann = []
outer_gen_errors_for_baseline = []

outer_folds = KFold(n_splits=10, shuffle=True, random_state=42)

for outer_train_indices, outer_test_indices in outer_folds.split(X):
    X_outer_train = X.values[outer_train_indices][:, selected_features]
    y_outer_train = y.values[outer_train_indices]
    X_outer_test = X.values[outer_test_indices][:, selected_features]
    y_outer_test = y.values[outer_test_indices]

    X_train_ann_outer = torch.Tensor(X_ann[outer_train_indices, :])
    y_train_ann_outer = torch.Tensor(y_ann[outer_train_indices])
    X_test_ann_outer = torch.Tensor(X_ann[outer_test_indices, :])
    y_test_ann_outer = torch.Tensor(y_ann[outer_test_indices])

    inner_gen_errors_for_lr = np.empty((10, len(lambdas1)))
    inner_gen_errors_for_ann = np.empty((10, len(hs)))
    inner_gen_errors_for_baseline = np.empty((10, 1))

    inner_folds = KFold(n_splits=10, shuffle=True, random_state=42)

    for split_index, (inner_train_indices, inner_test_indices) in enumerate(inner_folds.split(X_outer_train)):
        X_inner_train = X_outer_train[inner_train_indices]
        y_inner_train = y_outer_train[inner_train_indices]
        X_inner_test = X_outer_train[inner_test_indices]
        y_inner_test = y_outer_train[inner_test_indices]

        X_train_ann_inner = torch.Tensor(X_ann[inner_train_indices, :])
        y_train_ann_inner = torch.Tensor(y_ann[inner_train_indices])
        X_test_ann_inner = torch.Tensor(X_ann[inner_test_indices, :])
        y_test_ann_inner = torch.Tensor(y_ann[inner_test_indices])

        for lambda_index, lambda_ in enumerate(lambdas1):
            lr_model = Ridge(alpha=lambda_)
            lr_model.fit(X_inner_train, y_inner_train)
            y_inner_pred = lr_model.predict(X_inner_test)
            gen_er_lr = mean_squared_error(y_inner_test, y_inner_pred)
            inner_gen_errors_for_lr[split_index, lambda_index] = gen_er_lr

        for h_index, h in enumerate(hs):
            ann_model = lambda: torch.nn.Sequential(
                torch.nn.Linear(selected_features.size, h),  # M features to n_hidden_units (h)
                torch.nn.Tanh(),
                torch.nn.Linear(h, 1),  # n_hidden_units to 1 output neuron
                # no final tranfer function, i.e. "linear output"
            )

            net, final_loss, learning_curve = train_neural_net(
                ann_model,
                loss_fn,
                X=X_train_ann_inner,
                y=y_train_ann_inner,
                n_replicates=1,
                max_iter=10000,
            )

            y_ann_test_est = net(X_test_ann_inner).detach().numpy()
            gen_er_ann = mean_squared_error(y_test_ann_inner, y_ann_test_est)
            inner_gen_errors_for_ann[split_index, h_index] = gen_er_ann

    mean_of_training_y = np.mean(y_outer_train)
    y_outer_pred_baseline = (np.ones(len(y_outer_test)) * mean_of_training_y).T  # single-column matrix
    outer_gen_err_baseline = mean_squared_error(y_outer_test, y_outer_pred_baseline)
    outer_gen_errors_for_baseline.append(outer_gen_err_baseline)

    mean_inner_gen_errors_for_lr = inner_gen_errors_for_lr.mean(axis=0)  # means for each lr model
    index_of_picked_lambda_for_lr = np.argmin(mean_inner_gen_errors_for_lr)
    lr_model = Ridge(alpha=lambdas1[index_of_picked_lambda_for_lr])  # make best lr model (the best lambda)
    lr_model.fit(X_outer_train, y_outer_train)  # train it
    y_outer_pred_lr = lr_model.predict(X_outer_test)
    outer_gen_err_lr = mean_squared_error(y_outer_test, y_outer_pred_lr)
    outer_gen_errors_for_lr.append(outer_gen_err_lr)
    indexes_of_picked_lambdas_for_lr.append(index_of_picked_lambda_for_lr)

    mean_inner_gen_errors_for_ann = inner_gen_errors_for_ann.mean(axis=1)
    index_of_picked_hs_for_ann = np.argmin(mean_inner_gen_errors_for_ann)
    ann_model = lambda: torch.nn.Sequential(
        torch.nn.Linear(selected_features.size, index_of_picked_hs_for_ann),  # M features to n_hidden_units (h)
        torch.nn.Tanh(),
        torch.nn.Linear(index_of_picked_hs_for_ann, 1),  # n_hidden_units to 1 output neuron
        # no final tranfer function, i.e. "linear output"
    )
    net, final_loss, learning_curve = train_neural_net(
        ann_model,
        loss_fn,
        X=X_train_ann_outer,
        y=y_train_ann_outer,
        n_replicates=1,
        max_iter=10000,
    )
    y_ann_outer_est = net(X_test_ann_outer).detach().numpy()
    outer_gen_er_ann = mean_squared_error(y_test_ann_outer, y_ann_outer_est)
    outer_gen_errors_for_ann.append(outer_gen_er_ann)
    indexes_of_picked_hs_for_ann.append(index_of_picked_hs_for_ann)

# print the table
table_col_names = ['Outer fold index', 'LR lambda', 'LR err', 'ANN h', 'ANN err', 'Baseline err']
table_data = np.full((10, 6), -1)
table_data = table_data.astype(float)

indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).T
table_data[:, 0] = indices
table_data[:, 1] = lambdas1[indexes_of_picked_lambdas_for_lr]
table_data[:, 2] = outer_gen_errors_for_lr
table_data[:, 3] = hs[indexes_of_picked_hs_for_ann]
table_data[:, 4] = outer_gen_errors_for_ann
table_data[:, 5] = outer_gen_errors_for_baseline

print(table_col_names)
print(table_data)

# comparison ============================================================
# ============================================================
# ============================================================
# ============================================================
# ============================================================

X_train_, X_test_, y_train_, y_test_ = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model_ = Ridge(alpha=opt_lambda)  # 1.1020 (as found in regression part a)
lr_model_.fit(X_train_.values[:, selected_features], y_train_.values)
yhat_lr = lr_model_.predict(X_test_.values[:, selected_features])

ann_model_ = lambda: torch.nn.Sequential(
                torch.nn.Linear(selected_features.size, 4),
                torch.nn.Tanh(),
                torch.nn.Linear(4, 1),  # n_hidden_units to 1 output neuron
                # no final tranfer function, i.e. "linear output"
            )

net, final_loss, learning_curve = train_neural_net(
                ann_model_,
                loss_fn,
                X=torch.Tensor(X_train_.values[:, selected_features]),
                y=torch.Tensor(y_train_.values),
                n_replicates=1,
                max_iter=10000,
            )

yhat_ann = net(torch.Tensor(X_test_.values[:, selected_features])).detach().numpy()

mean_of_training_y_ = np.mean(y_train_)
yhat_baseline = (np.ones(len(y_test_.values)) * mean_of_training_y_).T

z_lr = np.abs(y_test_.values - yhat_lr) ** 2
z_ann = np.abs(y_test_.values - yhat_ann.flatten()) ** 2
z_baseline = np.abs(y_test_.values - yhat_baseline) ** 2

alpha = 0.05

CI_lr = st.t.interval(
    1 - alpha, df=len(z_lr) - 1, loc=np.mean(z_lr), scale=st.sem(z_lr)
)

CI_baseline = st.t.interval(
    1 - alpha, df=len(z_baseline) - 1, loc=np.mean(z_baseline), scale=st.sem(z_baseline)
)

CI_ann = st.t.interval(
    1 - alpha, df=len(z_ann) - 1, loc=np.mean(z_ann), scale=st.sem(z_ann)
)

print("CI of lr: ", CI_lr)
print("CI of ANN: ", CI_ann)
print("CI of baseline: ", CI_baseline)

z_lr_vs_baseline = z_lr - z_baseline
CI_lr_vs_baseline = st.t.interval(
    1 - alpha, len(z_lr_vs_baseline) - 1, loc=np.mean(z_lr_vs_baseline), scale=st.sem(z_lr_vs_baseline))
p_lr_vs_baseline = 2 * st.t.cdf(-np.abs(np.mean(z_lr_vs_baseline)) / st.sem(z_lr_vs_baseline),
                                df=len(z_lr_vs_baseline) - 1)

z_ann_vs_baseline = z_ann - z_baseline
CI_ann_vs_baseline = st.t.interval(
    1 - alpha, len(z_ann_vs_baseline) - 1, loc=np.mean(z_ann_vs_baseline), scale=st.sem(z_ann_vs_baseline))
p_ann_vs_baseline = 2 * st.t.cdf(-np.abs(np.mean(z_ann_vs_baseline)) / st.sem(z_ann_vs_baseline),
                                 df=len(z_ann_vs_baseline) - 1)

z_ann_vs_lr  = z_ann - z_lr
CI_ann_vs_lr = st.t.interval(
    1 - alpha, len(z_ann_vs_lr) - 1, loc=np.mean(z_ann_vs_lr), scale=st.sem(z_ann_vs_lr))
p_ann_vs_lr = 2 * st.t.cdf(-np.abs(np.mean(z_ann_vs_lr)) / st.sem(z_ann_vs_lr),
                                 df=len(z_ann_vs_lr) - 1)

print("CI lr_vs_baseline", CI_lr_vs_baseline)
print("p-value lr_vs_baseline", p_lr_vs_baseline)

print("CI ann_vs_baseline", CI_ann_vs_baseline)
print("p-value ann_vs_baseline", p_ann_vs_baseline)

print("CI ann_vs_lr", CI_ann_vs_lr)
print("p-value ann_vs_lr", p_ann_vs_lr)