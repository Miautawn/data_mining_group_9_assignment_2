import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class ContentBasedRanker(BaseEstimator, RegressorMixin):
    """Custom scikit-learn estimator for Content-Based matching."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.content_features = [
            "star_user_pref_abs_diff",
            "price_user_pref_abs_diff",
            "prop_booking_rate_smooth",
            "prop_review_score",
            "promotion_flag",
        ]
        self.model_ = Ridge(alpha=self.alpha)

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "Input X must be a pandas DataFrame containing feature names."
            )

        X_content = X[self.content_features].copy()
        X_arr, y_arr = check_X_y(X_content, y, accept_sparse=False, y_numeric=True)

        self.model_.fit(X_arr, y_arr)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "Input X must be a pandas DataFrame containing feature names."
            )

        X_content = X[self.content_features].copy()
        X_arr = check_array(X_content, accept_sparse=False)
        return self.model_.predict(X_arr)

    def generate_ranking(self, y, predictions, test_mode=False):
        df_result = pd.DataFrame(
            {"srch_id": y["srch_id"], "prop_id": y["prop_id"], "score": predictions}
        )

        if not test_mode:
            df_result["relevance"] = y["relevance"]

        return df_result.sort_values(by=["srch_id", "score"], ascending=[True, False])
