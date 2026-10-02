"""
FlexiSAF Generative AI & Data Science Internship
Module 1: Advanced Machine Learning Techniques
Technique 2 of 2: HYPERPARAMETER OPTIMIZATION

Goal
----
Demonstrate understanding of hyperparameter optimization by systematically
searching for the best hyperparameters of a Support Vector Machine (SVM)
classifier, rather than guessing values by hand.

Method
------
GridSearchCV: exhaustively tries every combination of hyperparameters in a
defined grid, using k-fold cross-validation on the training set to score
each combination, then refits the best combination on the full training set.

Dataset
-------
Breast cancer diagnostic dataset (built into scikit-learn, no download
required): 569 samples, 30 numeric features, binary classification
(malignant / benign).

Run
---
    python hyperparameter_optimization.py
"""

import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    print("=" * 70)
    print("MODULE 1 - HYPERPARAMETER OPTIMIZATION (GridSearchCV on an SVM)")
    print("=" * 70)

    # 1. Load data
    data = load_breast_cancer()
    X, y = data.data, data.target
    class_names = data.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"\nDataset: Breast cancer diagnostic | {X.shape[0]} samples, "
          f"{X.shape[1]} features, binary classification")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # SVMs are distance-based, so features must be scaled first.
    # A Pipeline keeps scaling + model together so cross-validation folds
    # never leak information from the validation fold into scaling.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC()),
    ])

    # 2. Baseline model with scikit-learn's untuned defaults
    pipeline.fit(X_train, y_train)
    baseline_pred = pipeline.predict(X_test)
    baseline_acc = accuracy_score(y_test, baseline_pred)
    print(f"\nBaseline SVM (default hyperparameters) test accuracy: "
          f"{baseline_acc:.4f}")
    print(f"Default hyperparameters used: {pipeline.named_steps['svm'].get_params()}")

    # 3. Define the search space
    #    C: regularization strength (how much misclassification is tolerated)
    #    gamma: kernel coefficient (how far a single training point's
    #           influence reaches)
    #    kernel: the function used to map data into higher dimensions
    param_grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", 0.01, 0.001, 0.0001],
        "svm__kernel": ["rbf", "linear"],
    }
    n_combinations = (len(param_grid["svm__C"]) *
                       len(param_grid["svm__gamma"]) *
                       len(param_grid["svm__kernel"]))
    print(f"\nSearch space: {n_combinations} hyperparameter combinations, "
          f"5-fold cross-validation each "
          f"({n_combinations * 5} total model fits)")

    # 4. Run the grid search
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1
    )
    start = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"Grid search completed in {elapsed:.2f} seconds")

    print(f"\nBest cross-validation accuracy: {grid_search.best_score_:.4f}")
    print(f"Best hyperparameters found: {grid_search.best_params_}")

    # 5. Evaluate the tuned model on the held-out test set
    best_model = grid_search.best_estimator_
    tuned_pred = best_model.predict(X_test)
    tuned_acc = accuracy_score(y_test, tuned_pred)

    print(f"\n--- Before vs after tuning (test set) ---")
    print(f"Baseline (default) accuracy : {baseline_acc:.4f}")
    print(f"Tuned (GridSearchCV) accuracy: {tuned_acc:.4f}")
    print(f"Improvement                 : {tuned_acc - baseline_acc:+.4f}")

    print("\n--- Classification report: tuned model ---")
    print(classification_report(y_test, tuned_pred, target_names=class_names))
    print("--- Confusion matrix: tuned model ---")
    print(confusion_matrix(y_test, tuned_pred))

    # 6. Show the top 5 combinations tried, for transparency into the search
    results = grid_search.cv_results_
    order = np.argsort(results["mean_test_score"])[::-1][:5]
    print("\n--- Top 5 hyperparameter combinations tried ---")
    for rank, i in enumerate(order, start=1):
        print(f"{rank}. score={results['mean_test_score'][i]:.4f}  "
              f"params={results['params'][i]}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(
        "GridSearchCV replaces manual guesswork with a systematic, "
        "cross-validated search across the hyperparameter space, finding a "
        "combination of C, gamma and kernel that matches or improves on the "
        "untuned baseline while giving a defensible, reproducible reason for "
        "the final choice of hyperparameters."
    )


if __name__ == "__main__":
    main()
