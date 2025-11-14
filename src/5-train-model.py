# src/5-train-model.py
import pandas as pd
from rich.console import Console
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from config import load_config

console = Console()
cfg = load_config()

# Ensure data folder exists for saving CSV
(Path(__file__).parent.parent / "data").mkdir(parents=True, exist_ok=True)

def main():
    """Main function to train speaker recognition model using KNN."""
    
    if not cfg["FEATURES_FILE"].exists():
        console.print(f'[red]error: features file not found: {cfg["FEATURES_FILE"]}[/red]')
        return
    
    console.rule("[bold green]starting model training[/bold green]")
    
    # load feature manifest (CSV)
    console.print(f'[cyan]loading feature manifest from {cfg["FEATURES_FILE"]}[/cyan]')
    feature_manifest = pd.read_csv(cfg["FEATURES_FILE"])

    if feature_manifest.empty:
        console.print(f"[red]error: feature manifest is empty[/red]")
        return
    
    # --- Step 1: Train final model with best k=6 ---
    console.rule("[bold cyan]training final KNN model with k=6[/bold cyan]")
    knn_model, scaler = train_knn_model(feature_manifest, k=6)

    if knn_model is None:
        console.print(f"[red]error: model training failed[/red]")
        return

    # save trained model and scaler
    cfg["MODEL_DIR"].mkdir(parents=True, exist_ok=True)
    model_path = cfg["MODEL_DIR"] / "lisa_knn.pkl"
    scaler_path = cfg["MODEL_DIR"] / "scaler.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(knn_model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    console.print(f"[green]model training complete! saved KNN model to {model_path}[/green]")

    # --- Step 2: Cross-validation analysis for multiple k/folds ---
    console.rule("[bold magenta]cross-validation analysis for multiple k/folds[/bold magenta]")
    run_knn_cross_validation(feature_manifest, cfg["KNN_RESULTS_FILE"])

def train_knn_model(feature_manifest, test_size=0.25, k=6):
    """Train a KNN speaker recognition model using best k.
    
    Returns:
        knn_model: trained KNeighborsClassifier
        scaler: StandardScaler used to scale features
    """
    X = feature_manifest.drop(['speaker_id', 'clip_filename', 'duration'], axis=1).values
    y = feature_manifest['speaker_id'].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=test_size,
        random_state=cfg["RANDOM_SEED"],
        stratify=y
    )

    # Train KNN
    knn_model = KNeighborsClassifier(n_neighbors=k)
    console.print(f"[cyan]training KNN model with k={k}...[/cyan]")
    knn_model.fit(X_train, y_train)

    # Evaluate
    y_pred = knn_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    console.print(f"[yellow]Validation Accuracy: {acc*100:.2f}%[/yellow]")
    console.print("[yellow]Classification report:[/yellow]")
    console.print(classification_report(y_test, y_pred))

    return knn_model, scaler


def run_knn_cross_validation(feature_manifest, save_path):
    """Run cross-validation for multiple k values and fold counts and save results."""
    X = feature_manifest.drop(['speaker_id', 'clip_filename', 'duration'], axis=1).values
    y = feature_manifest['speaker_id'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k_values = range(1, 16)  # evaluate k = 1..15
    fold_values = [3, 5, 10]

    results = []

    for folds in fold_values:
        console.print(f"\n[bold magenta]{folds}-fold CV[/bold magenta]")
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X_scaled, y, cv=folds, scoring='accuracy')
            mean_acc = scores.mean()
            console.print(f"k={k:2d} | mean accuracy: {mean_acc*100:.2f}%")
            results.append({
                "k": k,
                "folds": folds,
                "mean_accuracy": f"{mean_acc*100:.2f}%"
            })

    # Save results to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(save_path, index=False)
    console.print(f"\n[green]Cross-validation results saved to {save_path}[/green]")


if __name__ == "__main__":
    main()
