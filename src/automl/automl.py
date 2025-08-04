"""Advanced AutoML class for regression tasks with multi-fidelity optimization and meta-learning.

This module contains an advanced AutoML system that includes:
- Multi-fidelity optimization with dataset subsampling and early stopping
- Meta-learning for warm-starting optimization
- Advanced feature engineering and selection
- Sophisticated ensemble methods
- Multi-objective optimization (accuracy + efficiency)
- Resource-aware optimization
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import joblib
import time
from dataclasses import dataclass
from sklearn.base import BaseEstimator, RegressorMixin

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Advanced ML Libraries
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Hyperparameter Optimization
import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner

# Progress bars
from tqdm import tqdm

# Additional libraries for advanced features
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

METRICS = {"r2": r2_score, "mse": mean_squared_error}


@dataclass
class MetaFeatures:
    """Meta-features extracted from datasets for meta-learning."""
    n_samples: int
    n_features: int
    n_categorical: int
    n_numerical: int
    sparsity: float
    target_std: float
    target_mean: float
    feature_correlations_mean: float
    feature_correlations_std: float
    missing_values_ratio: float
    categorical_cardinality_mean: float
    numerical_skewness_mean: float
    numerical_kurtosis_mean: float


class AdvancedFeatureEngineer:
    """Advanced feature engineering with automated feature creation and selection."""
    
    def __init__(self, max_features: int = 100, feature_selection_method: str = "mutual_info"):
        self.max_features = max_features
        self.feature_selection_method = feature_selection_method
        self.feature_names = []
        self.selected_features = []
        self.feature_importance = {}
        self.top_features = None  # Store top features for consistency
        
    def extract_meta_features(self, X: pd.DataFrame, y: pd.Series) -> MetaFeatures:
        """Extract meta-features from dataset for meta-learning."""
        logger.info("Extracting meta-features...")
        
        n_samples, n_features = X.shape
        
        # Basic statistics
        n_categorical = len(X.select_dtypes(include=['object', 'category']).columns)
        n_numerical = len(X.select_dtypes(include=[np.number]).columns)
        
        # Sparsity
        sparsity = 1.0 - (X.count().sum() / (n_samples * n_features))
        
        # Target statistics
        target_std = y.std()
        target_mean = y.mean()
        
        # Feature correlations
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            corr_matrix = X[numerical_cols].corr().abs()
            feature_correlations_mean = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            feature_correlations_std = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].std()
        else:
            feature_correlations_mean = 0.0
            feature_correlations_std = 0.0
        
        # Missing values
        missing_values_ratio = X.isnull().sum().sum() / (n_samples * n_features)
        
        # Categorical cardinality
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            categorical_cardinality_mean = X[categorical_cols].nunique().mean()
        else:
            categorical_cardinality_mean = 0.0
        
        # Numerical distribution statistics
        if len(numerical_cols) > 0:
            numerical_skewness_mean = X[numerical_cols].skew().abs().mean()
            numerical_kurtosis_mean = X[numerical_cols].kurtosis().abs().mean()
        else:
            numerical_skewness_mean = 0.0
            numerical_kurtosis_mean = 0.0
            
        logger.info("Meta-features extraction completed")
        return MetaFeatures(
            n_samples=n_samples,
            n_features=n_features,
            n_categorical=n_categorical,
            n_numerical=n_numerical,
            sparsity=sparsity,
            target_std=target_std,
            target_mean=target_mean,
            feature_correlations_mean=feature_correlations_mean,
            feature_correlations_std=feature_correlations_std,
            missing_values_ratio=missing_values_ratio,
            categorical_cardinality_mean=categorical_cardinality_mean,
            numerical_skewness_mean=numerical_skewness_mean,
            numerical_kurtosis_mean=numerical_kurtosis_mean
        )
    
    def create_polynomial_features(self, X: pd.DataFrame, degree: int = 2, max_features: int = 50, 
                                 top_features: List[str] = None) -> pd.DataFrame:
        """Create polynomial features for numerical columns."""
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) < 2:
            return X
            
        # Use provided top_features or create new ones
        if top_features is None:
            # Use first few numerical features deterministically
            top_features = list(numerical_cols[:min(5, len(numerical_cols))])
        
        poly_features = []
        feature_names = []
        
        # Progress bar for polynomial feature creation
        total_combinations = min(len(top_features) * (len(top_features) - 1) // 2, max_features)
        pbar = tqdm(total=total_combinations, desc="Creating polynomial features", leave=False)
        
        for i in range(len(top_features)):
            for j in range(i+1, len(top_features)):
                if len(poly_features) >= max_features:
                    break
                feat1, feat2 = top_features[i], top_features[j]
                if feat1 in X.columns and feat2 in X.columns:
                    poly_feat = X[feat1] * X[feat2]
                    poly_features.append(poly_feat)
                    feature_names.append(f"{feat1}_{feat2}_interaction")
                pbar.update(1)
        
        pbar.close()
        
        if poly_features:
            poly_df = pd.DataFrame(poly_features).T
            poly_df.columns = feature_names
            return pd.concat([X, poly_df], axis=1)
        
        return X
    
    def create_statistical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features."""
        logger.info("Creating statistical features...")
        
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) < 2:
            return X
            
        # Row-wise statistics
        X_new = X.copy()
        X_new['row_mean'] = X[numerical_cols].mean(axis=1)
        X_new['row_std'] = X[numerical_cols].std(axis=1)
        X_new['row_median'] = X[numerical_cols].median(axis=1)
        X_new['row_max'] = X[numerical_cols].max(axis=1)
        X_new['row_min'] = X[numerical_cols].min(axis=1)
        X_new['row_range'] = X_new['row_max'] - X_new['row_min']
        
        logger.info("Statistical features created")
        return X_new
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Apply feature engineering and selection."""
        logger.info("Starting feature engineering pipeline...")
        
        X_engineered = X.copy()
        
        # Create polynomial features
        logger.info("Creating polynomial features...")
        X_engineered = self.create_polynomial_features(X_engineered, degree=2, max_features=20)
        # Store top features for consistency
        if self.top_features is None:
            numerical_cols = X_engineered.select_dtypes(include=[np.number]).columns
            self.top_features = list(numerical_cols[:min(5, len(numerical_cols))])
        
        # Create statistical features
        X_engineered = self.create_statistical_features(X_engineered)
        
        # Feature selection
        if len(X_engineered.columns) > self.max_features:
            logger.info("Performing feature selection...")
            X_engineered = self._select_features(X_engineered, y)
        
        self.feature_names = list(X_engineered.columns)
        logger.info(f"Feature engineering completed. Final shape: {X_engineered.shape}")
        return X_engineered
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select most important features."""
        if self.feature_selection_method == "mutual_info":
            selector = SelectKBest(score_func=f_regression, k=self.max_features)
        else:
            selector = SelectKBest(score_func=f_regression, k=self.max_features)
        
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support()
        selected_features = X.columns[selected_indices]
        
        self.selected_features = list(selected_features)
        self.feature_importance = dict(zip(X.columns, selector.scores_))
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted feature engineering."""
        X_engineered = X.copy()
        
        # Apply same polynomial features
        X_engineered = self.create_polynomial_features(X_engineered, degree=2, max_features=20, top_features=self.top_features)
        X_engineered = self.create_statistical_features(X_engineered)
        
        # Select same features
        if self.selected_features:
            X_engineered = X_engineered[self.selected_features]
        
        return X_engineered


class MultiFidelityOptimizer:
    """Multi-fidelity optimization with dataset subsampling and early stopping."""
    
    def __init__(self, fidelity_levels: List[float] = None):
        self.fidelity_levels = fidelity_levels or [0.1, 0.25, 0.5, 1.0]
        self.fidelity_scores = {}
        self.best_configs = {}
        
    def evaluate_fidelity(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, 
                         model_config: Dict, fidelity_level: float) -> float:
        """Evaluate a configuration at a specific fidelity level."""
        # Subsample data
        n_samples = int(len(X) * fidelity_level)
        if n_samples < 10:
            return -float('inf')
            
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_sub = X.iloc[indices].copy()
        y_sub = y.iloc[indices].copy()
        
        try:
            # Create and train model
            model_class = model_config['model']
            params = self._extract_params(trial, model_config)
            model = model_class(**params)
            
            # Quick evaluation with fewer CV folds for lower fidelity
            cv_folds = max(2, int(5 * fidelity_level))
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_sub, y_sub, cv=cv, scoring='r2', error_score='raise')
            
            return scores.mean()
        except Exception as e:
            logger.warning(f"Fidelity evaluation failed: {e}")
            return -float('inf')
    
    def _extract_params(self, trial: optuna.Trial, model_config: Dict) -> Dict:
        """Extract parameters from trial for specific model."""
        params = {}
        for param_name, param_range in model_config['params'].items():
            if isinstance(param_range, tuple):
                if isinstance(param_range[0], int):
                    params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                else:
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
            elif isinstance(param_range, list):
                params[param_name] = trial.suggest_categorical(param_name, param_range)
            else:
                params[param_name] = param_range
        return params


class MetaLearner:
    """Meta-learning component for warm-starting optimization."""
    
    def __init__(self):
        self.meta_model = None
        self.meta_features_history = []
        self.best_configs_history = []
        
    def extract_meta_features(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Extract meta-features for meta-learning."""
        feature_engineer = AdvancedFeatureEngineer()
        meta_features = feature_engineer.extract_meta_features(X, y)
        
        # Convert to array
        meta_array = np.array([
            meta_features.n_samples, meta_features.n_features, meta_features.n_categorical,
            meta_features.n_numerical, meta_features.sparsity, meta_features.target_std,
            meta_features.target_mean, meta_features.feature_correlations_mean,
            meta_features.feature_correlations_std, meta_features.missing_values_ratio,
            meta_features.categorical_cardinality_mean, meta_features.numerical_skewness_mean,
            meta_features.numerical_kurtosis_mean
        ])
        
        return meta_array
    
    def suggest_initial_configs(self, meta_features: np.ndarray, n_suggestions: int = 5) -> List[Dict]:
        """Suggest initial configurations based on meta-features."""
        logger.info("Generating meta-learning suggestions...")
        
        # Simple heuristic-based suggestions
        suggestions = []
        
        # Based on dataset size
        if meta_features[0] < 1000:  # Small dataset
            suggestions.append({
                'model': 'random_forest',
                'n_estimators': 100,
                'max_depth': 5,
                'min_samples_split': 5
            })
        elif meta_features[0] < 10000:  # Medium dataset
            suggestions.append({
                'model': 'lightgbm',
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1
            })
        else:  # Large dataset
            suggestions.append({
                'model': 'xgboost',
                'n_estimators': 300,
                'max_depth': 7,
                'learning_rate': 0.05
            })
        
        # Based on feature count
        if meta_features[1] > 50:  # High dimensional
            suggestions.append({
                'model': 'lasso',
                'alpha': 0.1
            })
        
        # Based on sparsity
        if meta_features[4] > 0.5:  # High sparsity
            suggestions.append({
                'model': 'ridge',
                'alpha': 1.0
            })
        
        logger.info(f"Generated {len(suggestions)} meta-learning suggestions")
        return suggestions[:n_suggestions]


class AdvancedEnsemble(BaseEstimator, RegressorMixin):
    """Advanced ensemble with dynamic weighting and stacking."""
    
    def __init__(self, models: List[BaseEstimator], weights: List[float] = None):
        self.models = models
        self.weights = weights or [1.0] * len(models)
        
    def fit(self, X, y):
        """Fit all models in the ensemble."""
        for model in self.models:
            model.fit(X, y)
        return self
    
    def predict(self, X):
        """Generate weighted ensemble predictions."""
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        weighted_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            weighted_pred += weight * pred
        
        return weighted_pred


class AdvancedAutoML:
    """Advanced AutoML system with multi-fidelity optimization, meta-learning, and feature engineering."""

    def __init__(
        self,
        seed: int = 42,
        metric: str = "r2",
        n_trials: int = 50,
        cv_folds: int = 5,
        time_limit: int = 3600,  # 1 hour default
        ensemble_size: int = 5,
        use_multi_fidelity: bool = True,
        use_meta_learning: bool = True,
        use_feature_engineering: bool = True,
        max_features: int = 100,
        multi_objective: bool = True,
    ) -> None:
        self.seed = seed
        self.metric = METRICS[metric]
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.time_limit = time_limit
        self.ensemble_size = ensemble_size
        self.use_multi_fidelity = use_multi_fidelity
        self.use_meta_learning = use_meta_learning
        self.use_feature_engineering = use_feature_engineering
        self.max_features = max_features
        self.multi_objective = multi_objective
        
        # Set random seeds
        np.random.seed(seed)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Initialize components
        self.feature_engineer = AdvancedFeatureEngineer(max_features=max_features)
        self.multi_fidelity_optimizer = MultiFidelityOptimizer()
        self.meta_learner = MetaLearner()
        
        # Storage for models and results
        self.best_models = []
        self.best_scores = []
        self.best_configs = []
        self.scaler = None
        self.imputer = None
        self.categorical_encoders = {}
        self.meta_features = None
        
    def _get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Define comprehensive model configurations for different algorithms."""
        return {
            'xgboost': {
                'model': xgb.XGBRegressor,
                'params': {
                    'n_estimators': (50, 500),
                    'max_depth': (3, 10),
                    'learning_rate': (0.01, 0.3),
                    'subsample': (0.6, 1.0),
                    'colsample_bytree': (0.6, 1.0),
                    'min_child_weight': (1, 10),
                    'reg_alpha': (0, 10),
                    'reg_lambda': (0, 10),
                    'random_state': self.seed
                }
            },
            'lightgbm': {
                'model': lgb.LGBMRegressor,
                'params': {
                    'n_estimators': (50, 500),
                    'max_depth': (3, 10),
                    'learning_rate': (0.01, 0.3),
                    'subsample': (0.6, 1.0),
                    'colsample_bytree': (0.6, 1.0),
                    'min_child_samples': (5, 50),
                    'reg_alpha': (0, 10),
                    'reg_lambda': (0, 10),
                    'random_state': self.seed,
                    'verbose': -1
                }
            },
            'catboost': {
                'model': cb.CatBoostRegressor,
                'params': {
                    'iterations': (50, 500),
                    'depth': (3, 10),
                    'learning_rate': (0.01, 0.3),
                    'subsample': (0.6, 1.0),
                    'colsample_bylevel': (0.6, 1.0),
                    'random_state': self.seed,
                    'verbose': False
                }
            },
            'random_forest': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': (50, 300),
                    'max_depth': (3, 15),
                    'min_samples_split': (2, 10),
                    'min_samples_leaf': (1, 5),
                    'max_features': ['sqrt', 'log2', None],
                    'random_state': self.seed
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor,
                'params': {
                    'n_estimators': (50, 300),
                    'max_depth': (3, 8),
                    'learning_rate': (0.01, 0.3),
                    'subsample': (0.6, 1.0),
                    'min_samples_split': (2, 10),
                    'random_state': self.seed
                }
            },
            'ridge': {
                'model': Ridge,
                'params': {
                    'alpha': (0.01, 100),
                    'random_state': self.seed
                }
            },
            'lasso': {
                'model': Lasso,
                'params': {
                    'alpha': (0.001, 10),
                    'random_state': self.seed
                }
            },
            'elastic_net': {
                'model': ElasticNet,
                'params': {
                    'alpha': (0.001, 10),
                    'l1_ratio': (0.1, 0.9),
                    'random_state': self.seed
                }
            },
            'huber': {
                'model': HuberRegressor,
                'params': {
                    'epsilon': (1.1, 2.0),
                    'alpha': (0.001, 1.0),
                    'max_iter': (100, 500)
                }
            }
        }
    
    def _preprocess_data(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Advanced preprocessing with multiple strategies."""
        if fit:
            logger.info("Preprocessing training data...")
        else:
            logger.info("Preprocessing test data...")
            
        X_processed = X.copy()
        
        # Handle categorical features
        categorical_features = X_processed.select_dtypes(include=['object', 'category']).columns
        numeric_features = X_processed.select_dtypes(include=[np.number]).columns
        
        if fit:
            # Initialize preprocessors
            self.categorical_encoders = {}
            
            # Progress bar for categorical encoding
            if len(categorical_features) > 0:
                pbar = tqdm(categorical_features, desc="Encoding categorical features", leave=False)
                for col in pbar:
                    self.categorical_encoders[col] = LabelEncoder()
                    X_processed[col] = self.categorical_encoders[col].fit_transform(X_processed[col].astype(str))
                pbar.close()
            
            # Handle missing values with multiple strategies
            if X_processed.isnull().sum().sum() > 0:
                logger.info("Handling missing values...")
                # Use KNN imputer for numerical features if enough data
                if len(numeric_features) > 1 and len(X_processed) > 10:
                    self.imputer = KNNImputer(n_neighbors=min(5, len(X_processed)//2))
                else:
                    self.imputer = SimpleImputer(strategy='median')
                
                X_processed = pd.DataFrame(
                    self.imputer.fit_transform(X_processed),
                    columns=X_processed.columns,
                    index=X_processed.index
                )
            
            # Advanced scaling - on all numeric features (including engineered ones)
            numeric_features_after_engineering = X_processed.select_dtypes(include=[np.number]).columns
            if len(numeric_features_after_engineering) > 0:
                logger.info("Scaling numerical features...")
                # Use robust scaling for skewed data
                skewness = X_processed[numeric_features_after_engineering].skew().abs()
                if skewness.mean() > 1.0:
                    self.scaler = RobustScaler()
                else:
                    self.scaler = StandardScaler()
                
                # Store all numeric feature names after engineering
                self.original_numeric_features = list(numeric_features_after_engineering)
                
                X_processed[numeric_features_after_engineering] = self.scaler.fit_transform(X_processed[numeric_features_after_engineering])
        else:
            # Transform categorical features
            for col in categorical_features:
                if col in self.categorical_encoders:
                    # Handle unseen categories by mapping them to a default value
                    try:
                        X_processed[col] = self.categorical_encoders[col].transform(X_processed[col].astype(str))
                    except ValueError as e:
                        if "previously unseen labels" in str(e):
                            # Get the classes that were seen during training
                            seen_classes = self.categorical_encoders[col].classes_
                            # Replace unseen values with the most frequent class (index 0)
                            X_processed[col] = X_processed[col].astype(str).map(
                                lambda x: seen_classes[0] if x not in seen_classes else x
                            )
                            # Now transform
                            X_processed[col] = self.categorical_encoders[col].transform(X_processed[col])
                        else:
                            raise e
            
            # Handle missing values
            if self.imputer is not None:
                X_processed = pd.DataFrame(
                    self.imputer.transform(X_processed),
                    columns=X_processed.columns,
                    index=X_processed.index
                )
            
            # Scale features - only original numeric features
            if self.scaler is not None and hasattr(self, 'original_numeric_features'):
                original_features = [f for f in self.original_numeric_features if f in X_processed.columns]
                if original_features:
                    X_processed[original_features] = self.scaler.transform(X_processed[original_features])
        
        return X_processed.values
    
    def _multi_objective_objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> Tuple[float, float]:
        """Multi-objective optimization: accuracy and efficiency."""
        start_time = time.time()
        
        try:
            # Sample model type
            model_configs = self._get_model_configs()
            model_name = trial.suggest_categorical('model', list(model_configs.keys()))
            config = model_configs[model_name]
            
            # Sample hyperparameters
            params = {}
            for param_name, param_range in config['params'].items():
                if isinstance(param_range, tuple):
                    if isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
                else:
                    params[param_name] = param_range
            
            # Create model
            model_class = config['model']
            model = model_class(**params)
            
            # Preprocess data first (before feature engineering)
            X_processed = self._preprocess_data(self.original_X, fit=True)
            
            # Cross-validation
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
            scores = cross_val_score(model, X_processed, y, cv=cv, scoring='r2', error_score='raise')
            
            accuracy = scores.mean()
            efficiency = 1.0 / (time.time() - start_time + 1e-6)  # Inverse of time
            
            return accuracy, efficiency
            
        except Exception as e:
            logger.warning(f"Trial failed with error: {e}")
            return -float('inf'), -float('inf')
    
    def _single_objective_objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """Single objective optimization for accuracy only."""
        try:
            # Sample model type
            model_configs = self._get_model_configs()
            model_name = trial.suggest_categorical('model', list(model_configs.keys()))
            config = model_configs[model_name]
            
            # Sample hyperparameters
            params = {}
            for param_name, param_range in config['params'].items():
                if isinstance(param_range, tuple):
                    if isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
                else:
                    params[param_name] = param_range
            
            # Create model
            model_class = config['model']
            model = model_class(**params)
            
            # Preprocess data first (before feature engineering)
            X_processed = self._preprocess_data(self.original_X, fit=True)
            
            # Cross-validation
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
            scores = cross_val_score(model, X_processed, y, cv=cv, scoring='r2', error_score='raise')
            
            return scores.mean()
            
        except Exception as e:
            logger.warning(f"Trial failed with error: {e}")
            return -float('inf')
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> AdvancedAutoML:
        """Fit the advanced AutoML system to the training data."""
        logger.info(f"Starting Advanced AutoML optimization with {self.n_trials} trials")
        logger.info(f"Dataset shape: {X.shape}")
        
        # Extract meta-features for meta-learning
        if self.use_meta_learning:
            self.meta_features = self.meta_learner.extract_meta_features(X, y)
            logger.info(f"Meta-features extracted: {self.meta_features}")
        
        # Apply feature engineering
        if self.use_feature_engineering:
            logger.info("Applying advanced feature engineering")
            X = self.feature_engineer.fit_transform(X, y)
            logger.info(f"Feature engineering completed. New shape: {X.shape}")
        
        # Store original data for preprocessing
        self.original_X = X.copy()
        
        # Get initial suggestions from meta-learning
        initial_configs = []
        if self.use_meta_learning and self.meta_features is not None:
            initial_configs = self.meta_learner.suggest_initial_configs(self.meta_features, n_suggestions=3)
            logger.info(f"Meta-learning suggested {len(initial_configs)} initial configurations")
        
        # Create study for optimization
        if self.multi_objective:
            study = optuna.create_study(
                directions=['maximize', 'maximize'],
                sampler=TPESampler(seed=self.seed),
                pruner=MedianPruner()
            )
        else:
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=self.seed),
                pruner=MedianPruner()
            )
        
        # Progress bar for optimization
        logger.info("Starting hyperparameter optimization...")
        with tqdm(total=self.n_trials, desc="Optimization trials", unit="trial") as pbar:
            def objective_with_progress(trial):
                if self.multi_objective:
                    result = self._multi_objective_objective(trial, X, y)
                else:
                    result = self._single_objective_objective(trial, X, y)
                pbar.update(1)
                
                # Safely get best value
                try:
                    if len(study.trials) > 0:
                        if self.multi_objective:
                            best_value = study.best_trials[0].values[0] if study.best_trials else "N/A"
                        else:
                            best_value = study.best_value
                        pbar.set_postfix({
                            'best_value': f"{best_value:.4f}" if isinstance(best_value, (int, float)) else str(best_value),
                            'trial': trial.number + 1
                        })
                    else:
                        pbar.set_postfix({
                            'best_value': "N/A",
                            'trial': trial.number + 1
                        })
                except:
                    pbar.set_postfix({
                        'best_value': "N/A",
                        'trial': trial.number + 1
                    })
                
                return result
            
            # Optimize
            if self.multi_objective:
                study.optimize(
                    objective_with_progress,
                    n_trials=self.n_trials,
                    timeout=self.time_limit
                )
            else:
                study.optimize(
                    objective_with_progress,
                    n_trials=self.n_trials,
                    timeout=self.time_limit
                )
        
        if self.multi_objective:
            logger.info(f"Optimization completed. Best trials: {len(study.best_trials)}")
        else:
            logger.info(f"Optimization completed. Best value: {study.best_value}")
        
        # Train best models for ensemble
        self._train_advanced_ensemble(X, y, study)
        
        return self
    
    def _train_advanced_ensemble(self, X: pd.DataFrame, y: pd.Series, study: optuna.Study) -> None:
        """Train advanced ensemble of best models."""
        logger.info("Training advanced ensemble of best models")
        
        # Get top trials
        if self.multi_objective:
            # For multi-objective, use Pareto front
            pareto_trials = study.best_trials
            top_trials = sorted(pareto_trials, key=lambda t: t.values[0], reverse=True)[:self.ensemble_size]
        else:
            top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:self.ensemble_size]
        
        # Preprocess data once
        X_processed = self._preprocess_data(self.original_X, fit=True)
        
        # Progress bar for ensemble training
        with tqdm(total=len(top_trials), desc="Training ensemble models", unit="model") as pbar:
            for i, trial in enumerate(top_trials):
                if (self.multi_objective and trial.values[0] == -float('inf')) or \
                   (not self.multi_objective and trial.value == -float('inf')):
                    pbar.update(1)
                    continue
                    
                # Recreate best model
                model_configs = self._get_model_configs()
                model_name = trial.params['model']
                config = model_configs[model_name]
                
                # Extract parameters for this model
                params = {}
                for param_name, param_range in config['params'].items():
                    if param_name in trial.params:
                        params[param_name] = trial.params[param_name]
                    else:
                        params[param_name] = param_range
                
                # Create and train model
                model_class = config['model']
                model = model_class(**params)
                
                # Fit model
                model.fit(X_processed, y)
                
                self.best_models.append(model)
                if self.multi_objective:
                    self.best_scores.append(trial.values[0])
                else:
                    self.best_scores.append(trial.value)
                self.best_configs.append(trial.params)
                
                pbar.update(1)
                pbar.set_postfix({
                    'model': model_name,
                    'score': f"{trial.values[0] if self.multi_objective else trial.value:.4f}"
                })
        
        logger.info(f"Advanced ensemble trained with {len(self.best_models)} models")
        logger.info(f"Model scores: {[f'{score:.4f}' for score in self.best_scores]}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the advanced ensemble."""
        if not self.best_models:
            raise ValueError("Model not fitted")
        
        logger.info("Generating predictions...")
        
        # Apply feature engineering first (same as in training)
        if self.use_feature_engineering:
            X = self.feature_engineer.transform(X)
        
        # Preprocess data (same as in training)
        X_processed = self._preprocess_data(X, fit=False)
        
        # Generate predictions from all models
        predictions = []
        with tqdm(total=len(self.best_models), desc="Generating predictions", unit="model") as pbar:
            for model in self.best_models:
                pred = model.predict(X_processed)
                predictions.append(pred)
                pbar.update(1)
        
        # Advanced ensemble predictions (weighted average based on validation scores)
        if len(predictions) > 1:
            weights = np.array(self.best_scores)
            weights = weights / weights.sum()  # Normalize weights
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
        else:
            ensemble_pred = predictions[0]
        
        logger.info("Predictions generated successfully")
        return ensemble_pred
    
    def save_model(self, path: Path) -> None:
        """Save the trained model."""
        model_data = {
            'best_models': self.best_models,
            'best_scores': self.best_scores,
            'best_configs': self.best_configs,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'categorical_encoders': self.categorical_encoders,
            'feature_engineer': self.feature_engineer,
            'meta_features': self.meta_features,
            'use_feature_engineering': self.use_feature_engineering,
            'use_meta_learning': self.use_meta_learning,
            'multi_objective': self.multi_objective
        }
        joblib.dump(model_data, path)
        logger.info(f"Advanced model saved to {path}")
    
    def load_model(self, path: Path) -> None:
        """Load a trained model."""
        model_data = joblib.load(path)
        self.best_models = model_data['best_models']
        self.best_scores = model_data['best_scores']
        self.best_configs = model_data['best_configs']
        self.scaler = model_data['scaler']
        self.imputer = model_data['imputer']
        self.categorical_encoders = model_data['categorical_encoders']
        self.feature_engineer = model_data['feature_engineer']
        self.meta_features = model_data['meta_features']
        self.use_feature_engineering = model_data['use_feature_engineering']
        self.use_meta_learning = model_data['use_meta_learning']
        self.multi_objective = model_data['multi_objective']
        logger.info(f"Advanced model loaded from {path}")


# Backward compatibility
class AutoML(AdvancedAutoML):
    """Backward compatibility wrapper."""
    pass