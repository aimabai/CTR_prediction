import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, log_loss


# -------------------------------------------------------
# Smoothed Target Encoding
# Converts high-cardinality IDs into stable CTR estimates
# Uses KFold to avoid leakage
# -------------------------------------------------------
def smoothed_target_encode(train_df, test_df, col, target_col, smoothing=100):

    global_mean = train_df[target_col].mean()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    train_df[col + '_te'] = np.nan

    # out-of-fold encoding for training data
    for tr_idx, val_idx in kf.split(train_df):
        stats = train_df.iloc[tr_idx].groupby(col)[target_col].agg(['count', 'mean'])
        smooth = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
        train_df.loc[train_df.index[val_idx], col + '_te'] = train_df.loc[train_df.index[val_idx], col].map(smooth)

    # full-data encoding for test
    stats = train_df.groupby(col)[target_col].agg(['count', 'mean'])
    smooth = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)

    test_df[col + '_te'] = test_df[col].map(smooth).fillna(global_mean)
    train_df[col + '_te'] = train_df[col + '_te'].fillna(global_mean)

    return train_df, test_df


def main():

    # -------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    target = "clicked"


    # -------------------------------------------------------
    # 2. Remove constant columns (no information)
    # -------------------------------------------------------
    dead_cols = ['C9', 'C10', 'C12', 'C18', 'C25']
    train.drop(columns=dead_cols, inplace=True)
    test.drop(columns=dead_cols, inplace=True)


    # -------------------------------------------------------
    # 3. Time based features
    # -------------------------------------------------------
    for df in [train, test]:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        df['hour'] = df['timestamp'].dt.hour

        # cyclic encoding helps tree models understand time continuity
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # simple behavior proxies
        df['C1_activity'] = df.groupby('C1')['C1'].transform('count')   # user activity
        df['C4_popularity'] = df.groupby('C4')['C4'].transform('count') # ad popularity


    # -------------------------------------------------------
    # 4. Target encode the two problematic high-cardinality IDs
    # (otherwise CatBoost memorizes them)
    # -------------------------------------------------------
    for col in ['C1', 'C4']:
        train, test = smoothed_target_encode(train, test, col, target, smoothing=100)


    # -------------------------------------------------------
    # 5. Define categorical features
    # Remaining C columns handled natively by CatBoost
    # -------------------------------------------------------
    cat_features = [f"C{i}" for i in range(1, 27) if f"C{i}" not in dead_cols + ['C1', 'C4']]

    for col in cat_features:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)


    # -------------------------------------------------------
    # 6. Time-based split (simulate real production scenario)
    # -------------------------------------------------------
    train = train.sort_values("timestamp")
    split_idx = int(len(train) * 0.8)

    train_df = train.iloc[:split_idx]
    valid_df = train.iloc[split_idx:]


    # -------------------------------------------------------
    # 7. Final feature list
    # -------------------------------------------------------
    num_cols = [f'N{i}' for i in range(1, 10)]
    engineered = ['hour_sin', 'hour_cos', 'C1_activity', 'C4_popularity', 'C1_te', 'C4_te']

    features = cat_features + num_cols + engineered

    X_train, y_train = train_df[features], train_df[target]
    X_valid, y_valid = valid_df[features], valid_df[target]
    X_test, y_test = test[features], test[target]


    # -------------------------------------------------------
    # 8. CatBoost model
    # Moderate depth + strong regularization → better generalization
    # -------------------------------------------------------
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=10,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        early_stopping_rounds=150,
        verbose=100
    )

    model.fit(
        X_train, y_train,
        cat_features=cat_features,
        eval_set=(X_valid, y_valid),
        use_best_model=True
    )


    # -------------------------------------------------------
    # 9. Evaluation
    # -------------------------------------------------------
    val_probs = model.predict_proba(X_valid)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    print(f"\nVALID AUC: {roc_auc_score(y_valid, val_probs):.4f}")
    print(f"VALID LogLoss: {log_loss(y_valid, val_probs):.4f}")

    print(f"TEST AUC: {roc_auc_score(y_test, test_probs):.4f}")
    print(f"TEST LogLoss: {log_loss(y_test, test_probs):.4f}")


if __name__ == "__main__":
    main()
