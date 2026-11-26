"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
import math
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    # TODO: Replace every NaN with that column's nan-aware mean...
    N, F = X.shape
    for i in range (F):
        tolnan = su = 0
        for j in range (N):
            if math.isnan(X[j][i]):
                tolnan += 1
            else:
                su += X[j][i]
        for j in range (N):
            if math.isnan(X[j][i]):
                if N != tolnan:
                    X[j][i] = su/(N - tolnan)
                else:
                    X[j][i] = 0
    return X

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    q1 = np.quantile(X, 0.25, axis=0)
    q3 = np.quantile(X, 0.75, axis=0)
    iqr = q3 - q1 
    lower = q1 - k*iqr 
    upper = q3 + k*iqr 
    return lower, upper

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    N, F = X.shape
    out = np.array(X, dtype=float, copy=True)
    for i in range (N):
        for j in range (F):
            out[i][j] = max(out[i][j], lower[j])
            out[i][j] = min(out[i][j], upper[j])
    return out

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    return numerator/(denominator + eps)

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    col_n = col.reshape(-1, 1)
    return np.concatenate((X, col_n), axis=1)

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    s = set(labels)
    s = sorted(s)
    dic = dict()
    for i in range (len(s)):
        dic[s[i]] = i 
    ans = []
    cur = [0.0 for _ in range (len(s))]
    for i in labels:
        curr = cur.copy()
        curr[dic[i]] = 1.0
        ans.append(curr)
    return np.asarray(ans, dtype=float)

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    N, F = X.shape
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    for i in range (F):
        if std[i] == 0:
            std[i] = 1.0
    return mean, std

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    X_n = (X - mean)/std 
    return X_n

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    N, F = X.shape
    bias = np.ones(N, dtype=float)
    bias = bias.reshape(-1, 1)
    return np.concatenate((bias, X), axis=1)

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    return indices

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    n_samples = len(indices)
    train_end = int(n_samples*train_ratio)
    val_end = int(n_samples*(train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    return train_idx, val_idx, test_idx

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    X_sub, y_sub = X[indices], y[indices]
    return X_sub, y_sub

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    U, S, Vt = np.linalg.svd(X)
    sigma = np.zeros(X.shape)
    np.fill_diagonal(sigma, S)
    m, n = sigma.shape
    for i in range (m):
        for j in range (n):
            if sigma[i][j] != 0:
                sigma[i][j] = 1.0/sigma[i][j]
    A_plus = Vt.T @ np.transpose(sigma) @ U.T
    x_plus = A_plus @ y 
    # theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    # return theta
    return x_plus

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    return X @ theta

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    return np.mean(np.abs(y_pred - y_true))

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    return np.sqrt(np.mean((y_pred - y_true) ** 2))

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    SS_res = np.sum((y_pred - y_true)**2)   
    SS_tot = np.sum((y_true - np.mean(y_true))**2)
    if SS_tot == 0:
        return 0.0
    return 1 - SS_res/SS_tot

# Step 18 - residual_summary (not yet solved)
# TODO: implement

# Step 19 - prepare_cleaned_features (not yet solved)
# TODO: implement

# Step 20 - assemble_feature_matrix (not yet solved)
# TODO: implement

# Step 21 - make_train_val_test (not yet solved)
# TODO: implement

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

