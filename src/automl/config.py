"""Configuration file for Advanced AutoML system."""

from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class AutoMLConfig:
    """Configuration class for Advanced AutoML system."""
    
    # Basic settings
    seed: int = 42
    metric: str = "r2"
    n_trials: int = 100
    cv_folds: int = 5
    time_limit: int = 7200  # 2 hours default
    ensemble_size: int = 5
    
    # Advanced features
    use_multi_fidelity: bool = True
    use_meta_learning: bool = True
    use_feature_engineering: bool = True
    multi_objective: bool = True
    
    # Feature engineering settings
    max_features: int = 150
    feature_selection_method: str = "mutual_info"
    polynomial_degree: int = 2
    max_polynomial_features: int = 30
    
    # Multi-fidelity settings
    fidelity_levels: List[float] = None
    
    # Meta-learning settings
    meta_learning_suggestions: int = 5
    
    # Optimization settings
    sampler_type: str = "tpe"  # "tpe", "cmaes"
    pruner_type: str = "median"  # "median", "hyperband"
    
    def __post_init__(self):
        if self.fidelity_levels is None:
            self.fidelity_levels = [0.1, 0.25, 0.5, 1.0]


# Predefined configurations for different scenarios
QUICK_CONFIG = AutoMLConfig(
    n_trials=20,
    time_limit=1800,  # 30 minutes
    ensemble_size=3,
    use_multi_fidelity=False,
    use_meta_learning=False,
    multi_objective=False
)

BALANCED_CONFIG = AutoMLConfig(
    n_trials=50,
    time_limit=3600,  # 1 hour
    ensemble_size=5,
    use_multi_fidelity=True,
    use_meta_learning=True,
    multi_objective=False  # Disabled for testing
)

FULL_CONFIG = AutoMLConfig(
    n_trials=200,
    time_limit=21600,  # 6 hours
    ensemble_size=7,
    use_multi_fidelity=True,
    use_meta_learning=True,
    multi_objective=True,
    max_features=200
)

EXAM_CONFIG = AutoMLConfig(
    n_trials=150,
    time_limit=86400,  # 24 hours
    ensemble_size=5,
    use_multi_fidelity=True,
    use_meta_learning=True,
    multi_objective=True,
    max_features=150
)


# Model-specific hyperparameter spaces
MODEL_CONFIGS = {
    'xgboost': {
        'n_estimators': (50, 1000),
        'max_depth': (3, 12),
        'learning_rate': (0.01, 0.3),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0),
        'min_child_weight': (1, 15),
        'reg_alpha': (0, 15),
        'reg_lambda': (0, 15),
        'gamma': (0, 5)
    },
    'lightgbm': {
        'n_estimators': (50, 1000),
        'max_depth': (3, 12),
        'learning_rate': (0.01, 0.3),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0),
        'min_child_samples': (5, 100),
        'reg_alpha': (0, 15),
        'reg_lambda': (0, 15),
        'num_leaves': (10, 300)
    },
    'catboost': {
        'iterations': (50, 1000),
        'depth': (3, 12),
        'learning_rate': (0.01, 0.3),
        'subsample': (0.6, 1.0),
        'colsample_bylevel': (0.6, 1.0),
        'l2_leaf_reg': (1, 10),
        'border_count': [32, 64, 128, 254]
    },
    'random_forest': {
        'n_estimators': (50, 500),
        'max_depth': (3, 20),
        'min_samples_split': (2, 15),
        'min_samples_leaf': (1, 10),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    },
    'gradient_boosting': {
        'n_estimators': (50, 500),
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3),
        'subsample': (0.6, 1.0),
        'min_samples_split': (2, 15),
        'min_samples_leaf': (1, 10)
    },
    'ridge': {
        'alpha': (0.001, 1000),
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    },
    'lasso': {
        'alpha': (0.0001, 10),
        'max_iter': (100, 2000),
        'tol': (1e-5, 1e-3)
    },
    'elastic_net': {
        'alpha': (0.0001, 10),
        'l1_ratio': (0.1, 0.9),
        'max_iter': (100, 2000)
    },
    'huber': {
        'epsilon': (1.1, 2.0),
        'alpha': (0.0001, 1.0),
        'max_iter': (100, 1000)
    }
}


# Feature engineering configurations
FEATURE_ENGINEERING_CONFIGS = {
    'basic': {
        'use_polynomial_features': False,
        'use_statistical_features': False,
        'use_interaction_features': False
    },
    'standard': {
        'use_polynomial_features': True,
        'use_statistical_features': True,
        'use_interaction_features': False,
        'polynomial_degree': 2,
        'max_polynomial_features': 20
    },
    'advanced': {
        'use_polynomial_features': True,
        'use_statistical_features': True,
        'use_interaction_features': True,
        'polynomial_degree': 3,
        'max_polynomial_features': 50,
        'use_ratio_features': True,
        'use_log_features': True
    }
}


# Multi-fidelity configurations
MULTI_FIDELITY_CONFIGS = {
    'conservative': {
        'fidelity_levels': [0.1, 0.25, 0.5, 1.0],
        'trials_per_fidelity': [10, 15, 20, 30],
        'early_stopping_patience': 5
    },
    'aggressive': {
        'fidelity_levels': [0.05, 0.1, 0.25, 0.5, 1.0],
        'trials_per_fidelity': [5, 10, 15, 25, 40],
        'early_stopping_patience': 3
    },
    'balanced': {
        'fidelity_levels': [0.1, 0.3, 0.6, 1.0],
        'trials_per_fidelity': [15, 20, 25, 35],
        'early_stopping_patience': 4
    }
}


# Meta-learning configurations
META_LEARNING_CONFIGS = {
    'basic': {
        'use_dataset_similarity': True,
        'use_performance_history': False,
        'use_feature_importance': False,
        'n_suggestions': 3
    },
    'advanced': {
        'use_dataset_similarity': True,
        'use_performance_history': True,
        'use_feature_importance': True,
        'n_suggestions': 5,
        'similarity_threshold': 0.7
    }
} 