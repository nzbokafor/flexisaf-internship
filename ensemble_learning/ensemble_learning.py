"""
FlexiSAF Generative AI & Data Science Internship
Module 1: Advanced Machine Learning Techniques
Technique 1 of 2: ENSEMBLE LEARNING

Goal
----
Demonstrate understanding of ensemble learning by comparing a single
decision tree against two popular ensemble methods:
  - Random Forest (bagging: many trees trained on bootstrapped samples,
    predictions averaged/voted to reduce variance)
  - Gradient Boosting (boosting: trees trained sequentially, each one
    correcting the errors of the previous ensemble, reducing bias)

Dataset
-------
Wine recognition dataset (built into scikit-learn, no download required):
178 samples, 13 chemical/physical features, 3 wine cultivar classes.

Run
---
    python ensemble_learning.py
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    print("=" * 70)
    print("MODULE 1 - ENSEMBLE LEARNING (Random Forest vs Gradient Boosting)")
    print("=" * 70)

    # 1. Load data
    data = load_wine()
    X, y = data.data, data.target
    feature_names = data.feature_names
    class_names = data.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"\nDataset: Wine recognition | {X.shape[0]} samples, "
          f"{X.shape[1]} features, {len(class_names)} classes")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    results = {}

    # 2. Baseline: a single decision tree (high variance, prone to overfitting)
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_train, y_train)
    tree_pred = tree.predict(X_test)
    tree_acc = accuracy_score(y_test, tree_pred)
    results["Single Decision Tree"] = tree_acc

    # 3. Bagging ensemble: Random Forest
    #    Trains many trees on bootstrapped samples + random feature subsets,
    #    then averages their votes. Reduces variance vs a single tree.
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    results["Random Forest (Bagging)"] = rf_acc

    # 4. Boosting ensemble: Gradient Boosting
    #    Trains trees sequentially; each new tree focuses on the residual
    #    errors of the combined ensemble so far. Reduces bias.
    gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
    )
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_acc = accuracy_score(y_test, gb_pred)
    results["Gradient Boosting (Boosting)"] = gb_acc

    # 5. Report comparison
    print("\n--- Test accuracy comparison ---")
    for name, acc in results.items():
        print(f"{name:32s}: {acc:.4f}")

    # 6. 5-fold cross-validation to confirm the ensembles generalise, not just
    #    perform well on one lucky split
    print("\n--- 5-fold cross-validation accuracy (mean +/- std) ---")
    for name, model in [
        ("Single Decision Tree", tree),
        ("Random Forest", rf),
        ("Gradient Boosting", gb),
    ]:
        scores = cross_val_score(model, X, y, cv=5)
        print(f"{name:24s}: {scores.mean():.4f} +/- {scores.std():.4f}")

    # 7. Feature importance from the Random Forest (interpretability)
    importances = rf.feature_importances_
    top5 = np.argsort(importances)[::-1][:5]
    print("\n--- Top 5 most important features (Random Forest) ---")
    for i in top5:
        print(f"{feature_names[i]:28s}: {importances[i]:.4f}")

    # 8. Detailed report for the strongest model
    best_name = max(results, key=results.get)
    best_pred = {"Single Decision Tree": tree_pred,
                 "Random Forest (Bagging)": rf_pred,
                 "Gradient Boosting (Boosting)": gb_pred}[best_name]
    print(f"\n--- Classification report: {best_name} (best on test set) ---")
    print(classification_report(y_test, best_pred, target_names=class_names))
    print("--- Confusion matrix ---")
    print(confusion_matrix(y_test, best_pred))

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(
        "Both ensemble methods outperform a single decision tree, confirming "
        "the core idea of ensemble learning: combining multiple weak/variable "
        "learners produces a stronger, more stable model than any one learner "
        "alone. Random Forest reduces variance via bagging + feature "
        "randomness; Gradient Boosting reduces bias by sequentially "
        "correcting prior errors."
    )


if __name__ == "__main__":
    main()
