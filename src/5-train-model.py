# src/5-train-model.py
"""
Step-by-step training script for multiple types of models.
Saves:
 - model/*.pkl (one .pkl per model)
 - model/scaler.pkl (feature scaler)
 - model/evaluation/knn_cv_results.csv
 - model/evaluation/all_model_scores.csv (test accuracies)
 - model/evaluation/model_summary.json (best model summary)
"""

import json
import pickle
import time
import traceback
from pathlib import Path
from datetime import datetime
import argparse
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from config import load_config

console = Console()
cfg = load_config()

# Constants
MODEL_DIR = Path(cfg["MODEL_DIR"])
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_DIR = MODEL_DIR / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_FILE = cfg["FEATURES_FILE"]
RANDOM_SEED = cfg.get("RANDOM_SEED", 42)

KNN_CV_RESULTS = EVAL_DIR / "knn_cv_results.csv"
ALL_SCORES_CSV = EVAL_DIR / "all_model_scores.csv"
MODEL_SUMMARY_JSON = EVAL_DIR / "model_summary.json"
SCALER_PKL = MODEL_DIR / "scaler.pkl"

# Model-agnostic parameters (these parameters can be used across
# all models, unlike model-specific hyperparameters defined later)

TEST_SIZE = 0.30 # Train/test split size

# Helper Functions

def parse_args():
    parser = argparse.ArgumentParser(description="Train one or all models.")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model to train: all, knn, logistic, random_forest, gradient_boosting, ada_boost, svc"
    )
    return parser.parse_args()

def should_train(target, selected):
    return selected == "all" or selected == target

def save_model(model_obj, path: Path, model_name: str):
    try:
        with open(path, 'wb') as f:
            pickle.dump(model_obj, f)
        console.print(f"[green]Saved {model_name} to {path}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to save {model_name} to {path}: {e}[/red]")
        console.print(traceback.format_exc())

def to_repo_relative(path: Path) -> str:
    """Convert absolute path to repo-relative (relative to project root)."""
    try:
        return str(path.relative_to(Path(__file__).parent.parent))
    except Exception:
        # fallback: just return the name
        return str(path.name)

def print_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix"):
    """
    Print a confusion matrix in the terminal using rich tables.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names (speaker IDs)
        title: Table title
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Create rich table
    table = Table(title=title, show_header=True, header_style="bold magenta")
    
    # Add header row (first column is "True/Pred", then predicted labels)
    table.add_column("True \\ Pred", style="cyan", justify="right")
    for label in labels:
        table.add_column(str(label), justify="center", style="yellow")
    
    # Add data rows
    for i, label in enumerate(labels):
        row = [str(label)]
        for j in range(len(labels)):
            # Highlight diagonal (correct predictions) in green
            if i == j:
                row.append(f"[bold green]{cm[i][j]}[/bold green]")
            else:
                # Show misclassifications in red if non-zero
                if cm[i][j] > 0:
                    row.append(f"[red]{cm[i][j]}[/red]")
                else:
                    row.append(str(cm[i][j]))
        table.add_row(*row)
    
    console.print()
    console.print(table)
    console.print()
    
    # Print summary statistics
    total_samples = cm.sum()
    correct_predictions = np.trace(cm)
    console.print(f"[cyan]Total samples: {total_samples}[/cyan]")
    console.print(f"[green]Correct predictions: {correct_predictions} ({correct_predictions/total_samples*100:.2f}%)[/green]")
    console.print(f"[red]Incorrect predictions: {total_samples - correct_predictions} ({(total_samples - correct_predictions)/total_samples*100:.2f}%)[/red]")
    console.print()

    

# Model Training Steps
def main(selected_model="all"):
    console.rule("[bold green]Multi-model training[/bold green]")

    # Pre-training setup
    if not FEATURES_FILE.exists():
        console.print(f"[red]Error: features file not found: {FEATURES_FILE}[/red]")
        return

    df = pd.read_csv(FEATURES_FILE)
    if df.empty:
        console.print("[red]Error: features CSV is empty[/red]")
        return
    if "speaker_id" not in df.columns:
        console.print("[red]Error: 'speaker_id' column not found in features CSV[/red]")
        return

    # Prepare features and labels
    X = df.drop(['speaker_id', 'clip_filename', 'duration'], axis=1, errors='ignore').values
    y = df['speaker_id'].values
    console.print(f"[cyan]Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features[/cyan]")

    # Get unique labels for confusion matrix
    unique_labels = sorted(np.unique(y))
    console.print(f"[cyan]Number of classes: {len(unique_labels)}[/cyan]")

    # Scale X features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler
    with open(SCALER_PKL, 'wb') as f:
        pickle.dump(scaler, f)
    console.print(f"[green]Saved scaler to {SCALER_PKL}[/green]")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    console.print(f"[cyan]Train/test split: train={len(X_train)}, test={len(X_test)}[/cyan]")

    # Container for final test results and predictions
    all_model_scores = []
    model_predictions = {}  # Store predictions for each model

    if should_train("knn", selected_model):
        # ------------------------------------------
        # KNN CV sweep (before final KNN training)
        # ------------------------------------------
        try:
            console.rule("[bold magenta]KNN CV sweep (k=1..20) / folds 5,10[/bold magenta]")
            knn_results = []
            k_values = range(1, 20)
            fold_values = [5, 10]
            for folds in fold_values:
                console.print(f"[magenta]{folds}-fold CV for k=1..20[/magenta]")
                for k in k_values:
                    try:
                        knn_tmp = KNeighborsClassifier(n_neighbors=k)
                        scores = cross_val_score(knn_tmp, X_scaled, y, cv=folds, scoring='accuracy', n_jobs=-1)
                        mean_acc = float(scores.mean())
                        knn_results.append({"k": k, "folds": folds, "mean_accuracy": mean_acc})
                        console.print(f"k={k:2d} | mean acc: {mean_acc*100:.2f}%")
                    except Exception as e:
                        console.print(f"[red]KNN CV error (k={k}, folds={folds}): {e}[/red]")
                        knn_results.append({"k": k, "folds": folds, "mean_accuracy": 0.0, "error": str(e)})
            pd.DataFrame(knn_results).to_csv(KNN_CV_RESULTS, index=False)
            console.print(f"[green]Saved KNN CV results to {KNN_CV_RESULTS}[/green]")
        
        except Exception as e:
            console.print(f"[red]Error during KNN CV sweep: {e}[/red]")
            console.print(traceback.format_exc())

    # ----------------------------------------------------
    # Train final KNN
    # ----------------------------------------------------
    try:
        # KNN hyperparameters
        KNN_FINAL_K = 9

        console.rule(f"[bold cyan]Training final KNN (k={KNN_FINAL_K})[/bold cyan]")
        
        knn_model = KNeighborsClassifier(n_neighbors=KNN_FINAL_K)
        
        # CV for KNN
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        scores = cross_val_score(knn_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        cv_mean = float(scores.mean())
        cv_std = float(scores.std())
        console.print(f"[yellow]KNN (k={KNN_FINAL_K}) CV acc: {cv_mean*100:.2f}% (+/- {cv_std*100:.2f}%) [/yellow]")
        
        # Fit on full training set
        start = time.time()
        knn_model.fit(X_train, y_train)
        dur = time.time() - start
        y_pred = knn_model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        console.print(f"[yellow]KNN (k={KNN_FINAL_K}) test acc: {acc*100:.2f}% (train time: {dur:.2f}s)[/yellow]")
        console.print(classification_report(y_test, y_pred))
        
        # Print confusion matrix
        print_confusion_matrix(
            y_test,
            y_pred,
            labels=unique_labels,
            title=f"KNN (k={KNN_FINAL_K}) Confusion Matrix - Test Accuracy: {acc*100:.2f}%"
        )
        
        knn_path = MODEL_DIR / f"knn_k{KNN_FINAL_K}.pkl"
        save_model(knn_model, knn_path, "KNN")
        
        model_name = f"knn_k{KNN_FINAL_K}"
        all_model_scores.append({"model": model_name, "model_path": to_repo_relative(knn_path), "test_accuracy": acc, "cv_mean": cv_mean})
        model_predictions[model_name] = y_pred
    except Exception as e:
        console.print(f"[red]Error training KNN: {e}[/red]")
        console.print(traceback.format_exc())
        all_model_scores.append({"model": f"knn_k{KNN_FINAL_K}", "model_path": None, "test_accuracy": 0.0, "error": str(e)})

    if should_train("logistic", selected_model):
        # ----------------------------------------------------
        # Logistic Regression 
        # ----------------------------------------------------
        try:
            console.rule("[bold cyan]Training Logistic Regression[/bold cyan]")
            
            # Logistic Regression hyperparameters
            lr_C = 1.0
            lr_max_iter = 400

            logreg = LogisticRegression(C=lr_C, max_iter=lr_max_iter, solver="lbfgs", multi_class="auto", random_state=RANDOM_SEED)
            # CV for logistic regression
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
            start = time.time()
            scores = cross_val_score(logreg, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
            cv_mean = float(scores.mean())
            cv_std = float(scores.std())
            console.print(f"[yellow]LogReg CV acc: {cv_mean*100:.2f}% (+/- {cv_std*100:.2f}%) [/yellow]")

            # Fit on full training set
            logreg.fit(X_train, y_train)
            dur = time.time() - start
            y_pred = logreg.predict(X_test)
            acc = float(accuracy_score(y_test, y_pred))
            console.print(f"[yellow]LogReg test acc: {acc*100:.2f}% (time: {dur:.2f}s)[/yellow]")
            console.print(classification_report(y_test, y_pred))
            
            # Print confusion matrix
            print_confusion_matrix(
                y_test,
                y_pred,
                labels=unique_labels,
                title=f"Logistic Regression Confusion Matrix - Test Accuracy: {acc*100:.2f}%"
            )
            
            lr_path = MODEL_DIR / "logistic_regression.pkl"
            save_model(logreg, lr_path, "LogisticRegression")
            
            model_name = "logistic_regression"
            all_model_scores.append({"model": model_name, "model_path": to_repo_relative(lr_path), "test_accuracy": acc, "cv_mean": cv_mean})
            model_predictions[model_name] = y_pred
        except Exception as e:
            console.print(f"[red]Error training Logistic Regression: {e}[/red]")
            console.print(traceback.format_exc())
            all_model_scores.append({"model": "logistic_regression", "model_path": None, "test_accuracy": 0.0, "error": str(e)})

    if should_train("random_forest", selected_model):
        # ------------------------------------------
        # Random Forest
        # ------------------------------------------
        try:
            console.rule("[bold cyan]Training Random Forest[/bold cyan]")
            
            # Random Forest hyperparameters
            rf_n_estimators = 200
            rf_max_depth = None  # set to int for limited depth
            rf = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, n_jobs=-1, random_state=RANDOM_SEED)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
            start = time.time()
            scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
            cv_mean = float(scores.mean())
            cv_std = float(scores.std())
            console.print(f"[yellow]RandomForest CV acc: {cv_mean*100:.2f}% (+/- {cv_std*100:.2f}%) [/yellow]")

            rf.fit(X_train, y_train)
            dur = time.time() - start
            y_pred = rf.predict(X_test)
            acc = float(accuracy_score(y_test, y_pred))
            console.print(f"[yellow]RandomForest test acc: {acc*100:.2f}% (time: {dur:.2f}s)[/yellow]")
            console.print(classification_report(y_test, y_pred))
            
            # Print confusion matrix
            print_confusion_matrix(
                y_test,
                y_pred,
                labels=unique_labels,
                title=f"Random Forest Confusion Matrix - Test Accuracy: {acc*100:.2f}%"
            )
            
            rf_path = MODEL_DIR / "random_forest.pkl"
            save_model(rf, rf_path, "RandomForest")
            
            model_name = "random_forest"
            all_model_scores.append({"model": model_name, "model_path": to_repo_relative(rf_path), "test_accuracy": acc, "cv_mean": cv_mean})
            model_predictions[model_name] = y_pred
        except Exception as e:
            console.print(f"[red]Error training RandomForest: {e}[/red]")
            console.print(traceback.format_exc())
            all_model_scores.append({"model": "random_forest", "model_path": None, "test_accuracy": 0.0, "error": str(e)})

    if should_train("gradient_boosting", selected_model):
        # ------------------------------------------
        # Gradient Boosting
        # ------------------------------------------
        try:
            console.rule("[bold cyan]Training Gradient Boosting[/bold cyan]")
            
            # Gradient Boosting hyperparameters
            gb_n_estimators = 200
            gb = GradientBoostingClassifier(n_estimators=gb_n_estimators, random_state=RANDOM_SEED)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
            start = time.time()
            scores = cross_val_score(gb, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
            cv_mean = float(scores.mean())
            cv_std = float(scores.std())
            console.print(f"[yellow]GradientBoosting CV acc: {cv_mean*100:.2f}% (+/- {cv_std*100:.2f}%) [/yellow]")

            gb.fit(X_train, y_train)
            dur = time.time() - start
            y_pred = gb.predict(X_test)
            acc = float(accuracy_score(y_test, y_pred))
            console.print(f"[yellow]GradientBoosting test acc: {acc*100:.2f}% (time: {dur:.2f}s)[/yellow]")
            console.print(classification_report(y_test, y_pred))
            
            # Print confusion matrix
            print_confusion_matrix(
                y_test,
                y_pred,
                labels=unique_labels,
                title=f"Gradient Boosting Confusion Matrix - Test Accuracy: {acc*100:.2f}%"
            )
            
            gb_path = MODEL_DIR / "gradient_boosting.pkl"
            save_model(gb, gb_path, "GradientBoosting")
            
            model_name = "gradient_boosting"
            all_model_scores.append({"model": model_name, "model_path": to_repo_relative(gb_path), "test_accuracy": acc, "cv_mean": cv_mean})
            model_predictions[model_name] = y_pred
        except Exception as e:
            console.print(f"[red]Error training GradientBoosting: {e}[/red]")
            console.print(traceback.format_exc())
            all_model_scores.append({"model": "gradient_boosting", "model_path": None, "test_accuracy": 0.0, "error": str(e)})

    if should_train("svc", selected_model):
        # ------------------------------------------
        # SVC
        # ------------------------------------------
        try:
            console.rule("[bold cyan]Training SVC[/bold cyan]")
            
            # SVC hyperparameters
            svc_kernel = "rbf"
            svc = SVC(probability=True, kernel=svc_kernel, gamma="scale", random_state=RANDOM_SEED)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
            start = time.time()
            scores = cross_val_score(svc, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
            cv_mean = float(scores.mean())
            cv_std = float(scores.std())
            console.print(f"[yellow]SVC CV acc: {cv_mean*100:.2f}% (+/- {cv_std*100:.2f}%) [/yellow]")

            svc.fit(X_train, y_train)
            dur = time.time() - start
            y_pred = svc.predict(X_test)
            acc = float(accuracy_score(y_test, y_pred))
            console.print(f"[yellow]SVC test acc: {acc*100:.2f}% (time: {dur:.2f}s)[/yellow]")
            console.print(classification_report(y_test, y_pred))
            
            # Print confusion matrix
            print_confusion_matrix(
                y_test,
                y_pred,
                labels=unique_labels,
                title=f"SVC Confusion Matrix - Test Accuracy: {acc*100:.2f}%"
            )
            
            svc_path = MODEL_DIR / "svc.pkl"
            save_model(svc, svc_path, "SVC")
            
            model_name = "svc"
            all_model_scores.append({"model": model_name, "model_path": to_repo_relative(svc_path), "test_accuracy": acc, "cv_mean": cv_mean})
            model_predictions[model_name] = y_pred
        except Exception as e:
            console.print(f"[red]Error training SVC: {e}[/red]")
            console.print(traceback.format_exc())
            all_model_scores.append({"model": "svc", "model_path": None, "test_accuracy": 0.0, "error": str(e)})

    # ------------------------------------------
    # Print summary table of all models
    # ------------------------------------------
    if all_model_scores:
        console.rule("[bold cyan]Model Performance Summary[/bold cyan]")
        
        # Create a rich table
        table = Table(show_header=True, header_style="bold magenta", title="All Models Test Accuracy")
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Model", style="cyan", width=25)
        table.add_column("Test Accuracy", justify="right", style="yellow", width=15)
        table.add_column("CV Mean", justify="right", style="green", width=12)
        
        df_scores = pd.DataFrame(all_model_scores).sort_values(by="test_accuracy", ascending=False)
        
        for idx, row in df_scores.iterrows():
            rank = f"#{df_scores.index.get_loc(idx) + 1}"
            model_name = row['model']
            test_acc = f"{row['test_accuracy']*100:.2f}%"
            cv_mean = f"{row.get('cv_mean', 0)*100:.2f}%" if pd.notna(row.get('cv_mean')) else "N/A"
            
            # Highlight best model
            if df_scores.index.get_loc(idx) == 0:
                table.add_row(rank, f"[bold]{model_name}[/bold]", f"[bold]{test_acc}[/bold]", f"[bold]{cv_mean}[/bold]")
            else:
                table.add_row(rank, model_name, test_acc, cv_mean)
        
        console.print(table)
        console.print()

    # ------------------------------------------
    # Final aggregation & summary
    # ------------------------------------------

    try:   
        console.rule("[bold cyan]Done Model Training[/bold cyan]")

        df_scores = pd.DataFrame(all_model_scores).sort_values(by="test_accuracy", ascending=False)
        df_scores.to_csv(ALL_SCORES_CSV, index=False)
        console.print(f"[green]Saved all model scores to {ALL_SCORES_CSV}[/green]")

        if df_scores.empty:
            console.print("[red]No model scores to summarize[/red]")
            return

        best = df_scores.iloc[0].to_dict()
        best_name = best.get("model")
        best_path = best.get("model_path")
        best_acc = float(best.get("test_accuracy", 0.0))
        timestamp = datetime.utcnow().isoformat() + "Z"

        summary = {
            "best_model_name": best_name,
            "best_model_path": best_path,
            "best_model_test_accuracy": best_acc,
            "timestamp_utc": timestamp,
            "num_samples": int(X.shape[0]),
            "num_features": int(X.shape[1]),
            "num_classes": len(unique_labels),
            "models_evaluated": df_scores['model'].tolist()
        }
        with open(MODEL_SUMMARY_JSON, 'w') as f:
            json.dump(summary, f, indent=2)
        console.print(f"[green]Saved model summary to {MODEL_SUMMARY_JSON}[/green]")
        console.print(f"[bold green]Best model: {best_name} ({best_acc*100:.2f}% test acc)[/bold green]")
    except Exception as e:
        console.print(f"[red]Error writing final summary: {e}[/red]")
        console.print(traceback.format_exc())



if __name__ == "__main__":
    args = parse_args()
    main(selected_model=args.model)
