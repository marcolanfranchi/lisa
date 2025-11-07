# src/5-train-model.py
import pandas as pd
from rich.console import Console
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from config import load_config

console = Console()

cfg = load_config()

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
    
    # train KNN model
    console.rule("[bold cyan]training speaker recognition model (KNN)[/bold cyan]")
    knn_model, scaler = train_knn_model(feature_manifest)
    
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


def train_knn_model(feature_manifest, test_size=0.25, k=5):
    """Train a KNN speaker recognition model.
    
    Args:
        feature_manifest (pd.DataFrame): DataFrame with MFCC features + speaker_id
        test_size (float): fraction of data for validation
        k (int): number of neighbors
    
    Returns:
        knn_model: trained KNeighborsClassifier
        scaler: StandardScaler used to scale features
    """
    # Extract features (X) and labels (y)
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


if __name__ == "__main__":
    main()
