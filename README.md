# Advanced AutoML System - SS25 Tabular Data Exam

## Project Overview

This repository contains an advanced AutoML system developed for the SS25 Machine Learning Lab final exam, focusing on tabular data regression tasks. The system implements sophisticated AutoML concepts including multi-fidelity optimization, meta-learning, advanced feature engineering, and ensemble methods.

## Key Features

- **Multi-fidelity Optimization**: Dataset subsampling (10%, 25%, 50%, 100%) for efficient training
- **Meta-learning**: 13 meta-features for warm-starting hyperparameter optimization
- **Advanced Feature Engineering**: Polynomial and statistical feature creation
- **Sophisticated Ensemble**: Weighted ensemble of 5 diverse models
- **Multi-objective Optimization**: Balances accuracy and computational efficiency
- **Resource-aware Design**: Optimized for 24-hour time budget (achieved in 22 minutes)

## Performance

- **Exam Dataset R² Score**: 0.9275 (92.75%)
- **Baseline R² Score**: 0.9290 (92.90%)
- **Training Time**: 22 minutes (well within 24-hour limit)
- **Test Predictions**: 1378 samples generated successfully

## Quick Start

### Prerequisites

- Python 3.10 or higher
- 8GB RAM minimum (16GB recommended)
- 2GB free storage space

### Installation

```bash
# Create and activate virtual environment
python3 -m venv automl-tabular-env
source automl-tabular-env/bin/activate  # On Windows: automl-tabular-env\Scripts\activate

# Install dependencies
pip install -e .
```

### Download Exam Dataset

```bash
python download-exam-dataset.py
```

### Generate Test Predictions

```bash
python run.py --task exam_dataset --seed 42 --save-model --output-path data/exam_dataset/predictions.npy --config exam
```

## Detailed Instructions

For comprehensive setup and usage instructions, see [run_instructions.md](run_instructions.md).

## Project Structure

```
├── src/automl/           # Core AutoML implementation
│   ├── automl.py        # Advanced AutoML system
│   ├── config.py        # Configuration management
│   └── data.py          # Data loading utilities
├── data/                # Dataset storage
│   └── exam_dataset/    # Exam dataset and predictions
├── results/             # Performance results for all datasets
├── run.py              # Main execution script
├── run_instructions.md # Detailed setup and usage guide
├── PERFORMANCE_SUMMARY.md # Comprehensive performance analysis
└── final_poster_tabular_frankhutterfanclub-1.pdf # Final poster
```

## Team Information

- Member 1: 6002473
- Member 2: 5150885

## License

BSD License - see [LICENSE](LICENSE) file for details.

## Contact

For questions about this implementation, refer to the detailed documentation in [run_instructions.md](run_instructions.md) and [PERFORMANCE_SUMMARY.md](PERFORMANCE_SUMMARY.md).
