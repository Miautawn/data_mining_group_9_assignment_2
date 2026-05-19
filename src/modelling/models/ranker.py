import lightgbm as lgb
import pandas as pd
from scipy.stats import randint, uniform

from src.data_preparation.evaluation import mean_ndcg_at_k


def tune_hyperparameters(lgb_train, lgb_val, n_iter=16):
    """Runs a random search over LTR hyperparameter distributions."""
    param_distributions = {
        "n_estimators": randint(1000, 2000),
        "learning_rate": uniform(0.01, 0.11),
        "min_child_samples": randint(1000, 3000),
        "num_leaves": randint(24, 120),
        "min_data_in_leaf": randint(20, 150),
        "feature_fraction": uniform(0.6, 0.4),
        "bagging_fraction": uniform(0.7, 0.3),
        "bagging_freq": randint(1, 6),
    }

    best_score, best_params = -1.0, None

    for idx in range(n_iter):
        run_params = {k: v.rvs() for k, v in param_distributions.items()}
        base_params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [5],
            "boosting_type": "gbdt",
            "feature_pre_filter": False,
            "n_jobs": -1,
            "max_depth": -1,
            "verbose": -1,
        }
        base_params.update(run_params)

        print(f"▶️ Iteration {idx + 1}/{n_iter}...")
        model = lgb.train(
            params=base_params,
            train_set=lgb_train,
            num_boost_round=400,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        current_score = model.best_score["valid_0"]["ndcg@5"]
        if current_score > best_score:
            best_score = current_score
            best_params = base_params

    print(f"🏁 Tuning finished. Best NDCG@5: {best_score:.5f}")
    return best_params


def train_model(params, train_data, valid_data):
    """Trains the LambdaRank model with standard callbacks."""
    return lgb.train(
        params=params,
        train_set=train_data,
        num_boost_round=1000,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=10),
        ],
    )


def compute_ndcg_at_k(y_true: pd.DataFrame, y_score, k=5):
    """Computes NDCG@k for a single query."""

    y_true["score"] = y_score

    return mean_ndcg_at_k(
        frame=y_true, prediction_col="score", relevance_col="relevance", k=k
    )
