# Module 1 — Advanced Machine Learning Techniques

FlexiSAF Generative AI & Data Science Internship, Intermediate pathway.

For this deliverable I had to train models showing I understand at least two of the advanced ML techniques listed in the curriculum. I went with **Ensemble Learning** and **Hyperparameter Optimization**. I considered doing Transfer Learning too, but I couldn't get PyTorch or TensorFlow installed in the environment I was working in (network restrictions on the model downloads), so I stuck with two techniques I could actually run end to end and verify, rather than submit something untested.

Both scripts use datasets that come bundled with scikit-learn, so there's nothing extra to download, just install the requirements and run.

## 1. Ensemble Learning

`ensemble_learning/ensemble_learning.py`

I used the Wine recognition dataset (178 samples, 3 wine classes) to compare a single decision tree against two ensemble approaches:

- **Random Forest** (bagging) — trains a bunch of trees on random subsets of the data and features, then lets them vote. This cuts down variance compared to one tree.
- **Gradient Boosting** (boosting) — trains trees one after another, where each new tree is trying to fix the mistakes of the ones before it. This cuts down bias.

Results on the test set:

| Model | Test accuracy | 5-fold CV accuracy |
| --- | --- | --- |
| Single Decision Tree | 0.9556 | 0.8654 ± 0.0440 |
| Random Forest | 1.0000 | 0.9665 ± 0.0207 |
| Gradient Boosting | 0.9778 | 0.9386 ± 0.0321 |

Both ensembles beat the single tree, and the cross-validation numbers back that up so it's not just a lucky train/test split. Full output (feature importances, classification report, confusion matrix) is in `evidence/ensemble_learning_output.txt`.

Run it:
```bash
pip install -r requirements.txt
cd ensemble_learning
python ensemble_learning.py
```

## 2. Hyperparameter Optimization

`hyperparameter_optimization/hyperparameter_optimization.py`

Here I used the breast cancer diagnostic dataset (569 samples, binary classification) and ran `GridSearchCV` to tune an SVM's `C`, `gamma`, and `kernel` instead of just guessing values. Scaling and the model are wrapped in a `Pipeline` so the cross-validation folds don't leak into the scaling step.

| | Accuracy |
| --- | --- |
| Baseline (default settings) | 0.9790 |
| Tuned (best of 32 combos × 5-fold CV = 160 fits) | 0.9790 |

Best combo found: `C=10, gamma=0.001, kernel='rbf'`.

The accuracy didn't move here since the default was already strong on this dataset, but that's not really the point. The point is going from "I guessed these settings" to "I systematically searched and can show you exactly why these are the best settings," which is in the evidence file as the top 5 combos tried. Full output: `evidence/hyperparameter_optimization_output.txt`.

Run it:
```bash
pip install -r requirements.txt
cd hyperparameter_optimization
python hyperparameter_optimization.py
```

## Folder layout

```
flexisaf-module1/
├── README.md
├── requirements.txt
├── ensemble_learning/
│   └── ensemble_learning.py
├── hyperparameter_optimization/
│   └── hyperparameter_optimization.py
└── evidence/
    ├── ensemble_learning_output.txt
    └── hyperparameter_optimization_output.txt
```
