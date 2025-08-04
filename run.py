"""Advanced AutoML run file with multi-fidelity optimization and meta-learning.

This script loads a dataset and runs the advanced AutoML system with:
- Multi-fidelity optimization
- Meta-learning for warm-starting
- Advanced feature engineering
- Multi-objective optimization
- Sophisticated ensemble methods
"""
from __future__ import annotations

from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from automl.data import Dataset
from automl.automl import AdvancedAutoML
from automl.config import EXAM_CONFIG, BALANCED_CONFIG, QUICK_CONFIG
import argparse
import logging
import time
import json
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

FILE = Path(__file__).absolute().resolve()
DATADIR = FILE.parent / "data"


def get_config(config_name: str):
    """Get configuration based on name."""
    configs = {
        'quick': QUICK_CONFIG,
        'balanced': BALANCED_CONFIG,
        'exam': EXAM_CONFIG
    }
    return configs.get(config_name, BALANCED_CONFIG)


def run_single_fold(
    task: str,
    fold: int,
    output_path: Path,
    seed: int,
    datadir: Path,
    config_name: str = 'balanced',
    save_model: bool = False,
    model_path: Path = None,
) -> Dict:
    """Run AutoML for a single fold and return results."""
    # Load dataset
    dataset = Dataset.load(datadir=datadir, task=task, fold=fold)
    
    # Get configuration
    config = get_config(config_name)
    config.seed = seed
    
    logger.info(f"Using configuration: {config_name}")
    logger.info(f"Dataset: {task}, Fold: {fold}")
    logger.info(f"Dataset shape: {dataset.X_train.shape}")
    
    # Initialize advanced AutoML system
    automl = AdvancedAutoML(
        seed=config.seed,
        metric=config.metric,
        n_trials=config.n_trials,
        cv_folds=config.cv_folds,
        time_limit=config.time_limit,
        ensemble_size=config.ensemble_size,
        use_multi_fidelity=config.use_multi_fidelity,
        use_meta_learning=config.use_meta_learning,
        use_feature_engineering=config.use_feature_engineering,
        max_features=config.max_features,
        multi_objective=config.multi_objective,
    )
    
    # Fit the AutoML system
    start_time = time.time()
    logger.info("Starting Advanced AutoML optimization")
    
    automl.fit(dataset.X_train, dataset.y_train)
    
    training_time = time.time() - start_time
    logger.info(f"AutoML training completed in {training_time:.2f} seconds")
    
    # Generate predictions
    logger.info("Generating predictions")
    test_preds: np.ndarray = automl.predict(dataset.X_test)
    
    # Save predictions
    logger.info(f"Writing predictions to {output_path}")
    with output_path.open("wb") as f:
        np.save(f, test_preds)
    
    # Save model if requested
    if save_model and model_path:
        logger.info(f"Saving model to {model_path}")
        automl.save_model(model_path)
    
    # Evaluate on test set if available
    results = {
        'task': task,
        'fold': fold,
        'config': config_name,
        'training_time': training_time,
        'ensemble_size': len(automl.best_models),
        'best_model_scores': [float(score) for score in automl.best_scores],
        'dataset_shape': dataset.X_train.shape,
        'test_shape': dataset.X_test.shape
    }
    
    if dataset.y_test is not None:
        r2_test = r2_score(dataset.y_test, test_preds)
        mse_test = mean_squared_error(dataset.y_test, test_preds)
        rmse_test = np.sqrt(mse_test)
        
        results.update({
            'r2_score': r2_test,
            'mse': mse_test,
            'rmse': rmse_test,
            'test_predictions_shape': test_preds.shape
        })
        
        logger.info(f"R² on test set: {r2_test:.6f}")
        logger.info(f"MSE on test set: {mse_test:.6f}")
        logger.info(f"RMSE on test set: {rmse_test:.6f}")
        
        # Performance summary
        logger.info("=" * 50)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Dataset: {task}")
        logger.info(f"Fold: {fold}")
        logger.info(f"Configuration: {config_name}")
        logger.info(f"Training time: {training_time:.2f} seconds")
        logger.info(f"R² Score: {r2_test:.6f}")
        logger.info(f"RMSE: {rmse_test:.6f}")
        logger.info(f"Ensemble size: {len(automl.best_models)}")
        logger.info(f"Best model scores: {[f'{score:.4f}' for score in automl.best_scores]}")
        logger.info("=" * 50)
    else:
        logger.info(f"No test labels available for task '{task}' (exam dataset)")
    
    return results


def run_all_folds(
    task: str,
    output_dir: Path,
    seed: int,
    datadir: Path,
    config_name: str = 'balanced',
    num_folds: int = 10,
) -> Dict:
    """Run AutoML across all folds and return comprehensive results."""
    logger.info(f"Running AutoML across all {num_folds} folds for task: {task}")
    
    all_results = []
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for fold in range(1, num_folds + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING FOLD {fold}/{num_folds}")
        logger.info(f"{'='*60}")
        
        # Create fold-specific output path
        fold_output_path = output_dir / f"{task}_fold_{fold}.npy"
        
        try:
            # Run single fold
            fold_results = run_single_fold(
                task=task,
                fold=fold,
                output_path=fold_output_path,
                seed=seed,
                datadir=datadir,
                config_name=config_name,
                save_model=False
            )
            all_results.append(fold_results)
            
        except Exception as e:
            logger.error(f"Error in fold {fold}: {e}")
            # Add error result
            all_results.append({
                'task': task,
                'fold': fold,
                'config': config_name,
                'error': str(e),
                'r2_score': None,
                'mse': None,
                'rmse': None,
                'training_time': None
            })
    
    # Calculate aggregate statistics
    successful_results = [r for r in all_results if 'error' not in r and r.get('r2_score') is not None]
    
    if successful_results:
        r2_scores = [r['r2_score'] for r in successful_results]
        training_times = [r['training_time'] for r in successful_results]
        
        aggregate_results = {
            'task': task,
            'config': config_name,
            'total_folds': num_folds,
            'successful_folds': len(successful_results),
            'failed_folds': len(all_results) - len(successful_results),
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'r2_min': np.min(r2_scores),
            'r2_max': np.max(r2_scores),
            'training_time_mean': np.mean(training_times),
            'training_time_total': np.sum(training_times),
            'fold_results': all_results
        }
    else:
        aggregate_results = {
            'task': task,
            'config': config_name,
            'total_folds': num_folds,
            'successful_folds': 0,
            'failed_folds': num_folds,
            'error': 'All folds failed',
            'fold_results': all_results
        }
    
    # Save comprehensive results
    results_file = output_dir / f"{task}_all_folds_results.json"
    with open(results_file, 'w') as f:
        json.dump(aggregate_results, f, indent=2, default=str)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPREHENSIVE RESULTS FOR {task.upper()}")
    logger.info(f"{'='*60}")
    logger.info(f"Configuration: {config_name}")
    logger.info(f"Total folds: {num_folds}")
    logger.info(f"Successful folds: {len(successful_results)}")
    logger.info(f"Failed folds: {len(all_results) - len(successful_results)}")
    
    if successful_results:
        logger.info(f"R² Score - Mean: {aggregate_results['r2_mean']:.6f}")
        logger.info(f"R² Score - Std: {aggregate_results['r2_std']:.6f}")
        logger.info(f"R² Score - Min: {aggregate_results['r2_min']:.6f}")
        logger.info(f"R² Score - Max: {aggregate_results['r2_max']:.6f}")
        logger.info(f"Training time - Mean: {aggregate_results['training_time_mean']:.2f}s")
        logger.info(f"Training time - Total: {aggregate_results['training_time_total']:.2f}s")
    
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"{'='*60}")
    
    return aggregate_results


def main(
    task: str,
    fold: int,
    output_path: Path,
    seed: int,
    datadir: Path,
    config_name: str = 'balanced',
    save_model: bool = False,
    model_path: Path = None,
    all_folds: bool = False,
    output_dir: Path = None,
    num_folds: int = 10,
):
    """Main function to run AutoML experiments."""
    if all_folds:
        if output_dir is None:
            output_dir = Path("results") / task
        run_all_folds(
            task=task,
            output_dir=output_dir,
            seed=seed,
            datadir=datadir,
            config_name=config_name,
            num_folds=num_folds
        )
    else:
        run_single_fold(
            task=task,
            fold=fold,
            output_path=output_path,
            seed=seed,
            datadir=datadir,
            config_name=config_name,
            save_model=save_model,
            model_path=model_path
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced AutoML System for Tabular Regression")

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="The name of the task to run on.",
        choices=["bike_sharing_demand", "brazilian_houses", "superconductivity", "wine_quality", "yprop_4_1", "exam_dataset"]
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("predictions.npy"),
        help="The path to save the predictions to."
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="The fold to run on. For the test dataset we will only provide a single fold, fold 1."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--datadir",
        type=Path,
        default=DATADIR,
        help="The directory where the datasets are stored."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="balanced",
        choices=["quick", "balanced", "exam"],
        help="Configuration preset to use."
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the trained model to disk."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/automl_model.pkl"),
        help="Path to save the trained model."
    )
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="Run all folds for comprehensive cross-validation evaluation."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for all-folds results (used with --all-folds)."
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=10,
        help="Number of folds to run (default: 10, used with --all-folds)."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Whether to log only warnings and errors."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging."
    )

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quiet:
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO)

    logger.info("=" * 60)
    logger.info("ADVANCED AUTOML SYSTEM - SS25 EXAM")
    logger.info("=" * 60)
    logger.info(f"Task: {args.task}")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Seed: {args.seed}")
    if args.all_folds:
        logger.info(f"Mode: All folds evaluation ({args.num_folds} folds)")
        logger.info(f"Output directory: {args.output_dir or Path('results') / args.task}")
    else:
        logger.info(f"Mode: Single fold (fold {args.fold})")
        logger.info(f"Output path: {args.output_path}")
    logger.info("=" * 60)

    main(
        task=args.task,
        fold=args.fold,
        output_path=args.output_path,
        datadir=args.datadir,
        seed=args.seed,
        config_name=args.config,
        save_model=args.save_model,
        model_path=args.model_path,
        all_folds=args.all_folds,
        output_dir=args.output_dir,
        num_folds=args.num_folds,
    )