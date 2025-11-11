# src/5-train-model.py
import pandas as pd
from rich.console import Console
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
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
    
    # train KNN model with best k
    console.rule("[bold cyan]training speaker recognition model (KNN)[/bold cyan]")
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

    # evaluate with 10-fold cross-validation
    #evaluate_knn_cv(feature_manifest, k=6, folds=10)
    #BY TRIAL AN ERROR, I FOUND THAT K=6 GIVES THE BEST RESULTS FOR THIS DATASET
    '''
    training KNN model with k=6...
    Validation Accuracy: 99.11%
    Classification report:
              precision    recall  f1-score   support

     georgii       1.00      0.96      0.98        56
       kolya       1.00      1.00      1.00        56
       marco       0.98      1.00      0.99        56
        vova       0.98      1.00      0.99        56

    accuracy                               0.99       224
    macro avg          0.99      0.99      0.99       224
    weighted avg       0.99      0.99      0.99       224

    
    '''


def train_knn_model(feature_manifest, test_size=0.25, k=4):
    """Train a KNN speaker recognition model using best k.
    
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


def evaluate_knn_cv(feature_manifest, k=4, folds=10):
    """Perform k-fold cross-validation with KNN and print mean accuracy."""
    console.rule(f"[bold blue]{folds}-fold cross-validation with k={k}[/bold blue]")

    X = feature_manifest.drop(['speaker_id', 'clip_filename', 'duration'], axis=1).values
    y = feature_manifest['speaker_id'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled, y, cv=folds, scoring='accuracy')
    mean_acc = scores.mean()

    console.print(f"[green]{folds}-fold mean accuracy: {mean_acc*100:.2f}%[/green]")


if __name__ == "__main__":
    main()
