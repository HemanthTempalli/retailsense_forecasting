"""
RetailSense — Model Wrappers
==============================
Consistent interface for all forecasting models.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge
from typing import Optional


class LGBMForecaster:
    """LightGBM wrapper for point and quantile forecasting."""

    def __init__(self, quantile: Optional[float] = None, **kwargs):
        self.quantile = quantile
        if quantile is not None:
            base = dict(objective='quantile', alpha=quantile,
                        n_estimators=300, learning_rate=0.05,
                        num_leaves=63, n_jobs=-1, random_state=42, verbose=-1)
        else:
            base = dict(n_estimators=400, learning_rate=0.05,
                        num_leaves=63, n_jobs=-1, random_state=42, verbose=-1)
        base.update(kwargs)
        self.model = lgb.LGBMRegressor(**base)

    def fit(self, X, y, X_val=None, y_val=None):
        callbacks = [lgb.log_evaluation(period=-1)]
        if X_val is not None:
            callbacks.append(lgb.early_stopping(50, verbose=False))
            self.model.fit(X, y, eval_set=[(X_val, y_val)], callbacks=callbacks)
        else:
            self.model.fit(X, y, callbacks=callbacks)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_sales(self, X):
        """Returns predictions in original (€) scale."""
        return np.expm1(self.predict(X))


class XGBForecaster:
    """XGBoost wrapper."""

    def __init__(self, **kwargs):
        base = dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                    random_state=42, verbosity=0, tree_method='hist')
        base.update(kwargs)
        self.model = xgb.XGBRegressor(**base)

    def fit(self, X, y, X_val=None, y_val=None):
        if X_val is not None:
            self.model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_sales(self, X):
        return np.expm1(self.predict(X))


class StackingEnsemble:
    """Two-layer stacking: XGB + LGB → Ridge meta-learner."""

    def __init__(self, xgb_model, lgb_model):
        self.xgb  = xgb_model
        self.lgb  = lgb_model
        self.meta = Ridge(alpha=1.0)

    def fit_meta(self, X_train, y_train):
        stack = np.column_stack([
            self.xgb.predict(X_train),
            self.lgb.predict(X_train),
        ])
        self.meta.fit(stack, y_train)
        return self

    def predict(self, X):
        stack = np.column_stack([
            self.xgb.predict(X),
            self.lgb.predict(X),
        ])
        return self.meta.predict(stack)

    def predict_sales(self, X):
        return np.expm1(self.predict(X))

    @property
    def weights(self):
        return {'xgb': self.meta.coef_[0], 'lgb': self.meta.coef_[1]}
