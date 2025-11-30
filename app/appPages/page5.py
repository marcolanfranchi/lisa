import streamlit as st
import pandas as pd
import json
import plotly.express as px
from pathlib import Path
import plotly.graph_objects as go
from appPages.components import section_header, blank_lines
from config import load_config

cfg = load_config()
MODEL_DIR = Path(cfg["MODEL_DIR"])
EVAL_DIR = MODEL_DIR / "evaluation"


def page5():
    """Display the page for step 5-train-model.py."""
    # ==============================================================================
    # Header
    # ==============================================================================
    section_header(
        "Model Training & Evaluation",
        "This page visualizes the results from training multiple speaker identification models. "
        "Models are evaluated using cross-validation and test set accuracy. The pipeline trains "
        "KNN, Logistic Regression, Random Forest, Gradient Boosting, and SVC models, with "
        "hyperparameter tuning for KNN."
    )

    # ==============================================================================
    # Load Model Summary
    # ==============================================================================
    model_summary_path = EVAL_DIR / "model_summary.json"
    all_scores_path = EVAL_DIR / "all_model_scores.csv"
    knn_cv_path = EVAL_DIR / "knn_cv_results.csv"

    if not model_summary_path.exists():
        st.warning(
            f"No model summary found at `{model_summary_path}`. "
            "Please run 5-train-model.py first."
        )
        return

    try:
        with open(model_summary_path, 'r') as f:
            summary = json.load(f)
    except Exception as e:
        st.error(f"Error loading model summary: {str(e)}")
        return

    # ==============================================================================
    # Best Model Summary
    # ==============================================================================
    blank_lines(2)
    st.markdown("#### Best Model Summary")

    best_model = summary.get("best_model_name", "N/A")
    best_acc = summary.get("best_model_test_accuracy", 0.0)
    num_samples = summary.get("num_samples", 0)
    num_features = summary.get("num_features", 0)
    timestamp = summary.get("timestamp_utc", "N/A")

    with st.container(border=False):
        metricHeight = 120
        cols = st.columns(4)
        with cols[0]:
            with st.container(border=True, height=metricHeight):
                st.markdown(":small[Best Model:]")
                st.markdown(f"#### {best_model}")
        with cols[1]:
            with st.container(border=True, height=metricHeight):
                st.markdown(":small[Test Accuracy:]")
                st.markdown(f"#### {best_acc*100:.2f}%")
        with cols[2]:
            with st.container(border=True, height=metricHeight):
                st.markdown(":small[Training Samples:]")
                st.markdown(f"#### {num_samples}")
        with cols[3]:
            with st.container(border=True, height=metricHeight):
                st.markdown(":small[Features Used:]")
                st.markdown(f"#### {num_features}")

    st.caption(f"Training completed: {timestamp}")

    # ==============================================================================
    # Model Comparison
    # ==============================================================================
    blank_lines(2)
    st.markdown("#### Model Performance Comparison")

    if all_scores_path.exists():
        try:
            df_scores = pd.read_csv(all_scores_path)
            
            if not df_scores.empty:
                # Sort by test accuracy
                df_scores = df_scores.sort_values('test_accuracy', ascending=False)
                
                # Create comparison bar chart
                fig = px.bar(
                    df_scores,
                    x='model',
                    y='test_accuracy',
                    color='model',
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                    title='Test Accuracy by Model',
                    labels={'test_accuracy': 'Test Accuracy', 'model': 'Model'},
                    text=df_scores['test_accuracy'].apply(lambda x: f'{x*100:.2f}%')
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(
                    showlegend=False,
                    height=500,
                    yaxis_tickformat='.0%',
                    yaxis_range=[0, min(1.1, df_scores['test_accuracy'].max() * 1.1)]
                )
                st.plotly_chart(fig, use_container_width=True)

                # Test vs CV Accuracy comparison
                if 'cv_mean' in df_scores.columns:
                    blank_lines(2)
                    st.markdown("#### Cross-Validation vs Test Performance")
                    
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(
                        name='CV Mean Accuracy',
                        x=df_scores['model'],
                        y=df_scores['cv_mean'],
                        marker_color='lightblue'
                    ))
                    fig2.add_trace(go.Bar(
                        name='Test Accuracy',
                        x=df_scores['model'],
                        y=df_scores['test_accuracy'],
                        marker_color='darkblue'
                    ))
                    
                    fig2.update_layout(
                        barmode='group',
                        title='Cross-Validation vs Test Accuracy',
                        yaxis_title='Accuracy',
                        yaxis_tickformat='.0%',
                        height=500
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    st.caption(
                        "CV (Cross-Validation) scores show model performance during training, "
                        "while test scores show performance on unseen data. Similar scores "
                        "indicate good generalization."
                    )

                # Detailed scores table
                blank_lines(2)
                st.markdown("#### Detailed Model Scores")
                
                display_df = df_scores.copy()
                display_df['test_accuracy'] = display_df['test_accuracy'].apply(lambda x: f'{x*100:.2f}%')
                if 'cv_mean' in display_df.columns:
                    display_df['cv_mean'] = display_df['cv_mean'].apply(lambda x: f'{x*100:.2f}%')
                
                st.dataframe(
                    display_df[['model', 'test_accuracy', 'cv_mean']],
                    hide_index=True,
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Error loading model scores: {str(e)}")
    else:
        st.info("Model scores file not found.")

    # ==============================================================================
    # KNN Hyperparameter Tuning
    # ==============================================================================
    blank_lines(2)
    st.markdown("#### KNN Hyperparameter Tuning")

    if knn_cv_path.exists():
        try:
            knn_cv = pd.read_csv(knn_cv_path)
            
            if not knn_cv.empty:
                st.caption(
                    "KNN performance was evaluated across different k values (number of neighbors) "
                    "and fold counts to find the optimal configuration."
                )
                
                # Create line plot for each fold count
                fig_knn = px.line(
                    knn_cv,
                    x='k',
                    y='mean_accuracy',
                    color='folds',
                    markers=True,
                    title='KNN Cross-Validation: Effect of k and Fold Count',
                    labels={
                        'k': 'Number of Neighbors (k)',
                        'mean_accuracy': 'Mean CV Accuracy',
                        'folds': 'CV Folds'
                    }
                )
                fig_knn.update_layout(
                    height=500,
                    yaxis_tickformat='.0%',
                    hovermode='x unified'
                )
                st.plotly_chart(fig_knn, use_container_width=True)

                # Find best k for each fold count
                best_configs = knn_cv.loc[knn_cv.groupby('folds')['mean_accuracy'].idxmax()]
                
                st.markdown("**Best k values by fold count:**")
                for _, row in best_configs.iterrows():
                    st.write(
                        f"- **{int(row['folds'])}-fold CV**: k={int(row['k'])} "
                        f"({row['mean_accuracy']*100:.2f}% accuracy)"
                    )

                # Heatmap of k vs folds
                blank_lines(1)
                pivot_knn = knn_cv.pivot(index='k', columns='folds', values='mean_accuracy')
                fig_heat = px.imshow(
                    pivot_knn,
                    labels=dict(x="CV Folds", y="k (Neighbors)", color="Accuracy"),
                    title="KNN Accuracy Heatmap: k vs Fold Count",
                    color_continuous_scale="Blues",
                    aspect="auto"
                )
                fig_heat.update_layout(height=400)
                st.plotly_chart(fig_heat, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading KNN CV results: {str(e)}")
    else:
        st.info("KNN cross-validation results not found.")

    # ==============================================================================
    # Model Architecture Insights
    # ==============================================================================
    blank_lines(2)
    st.markdown("#### Model Overviews")

    with st.expander("About the Trained Models", expanded=False):
        st.markdown("""
        **K-Nearest Neighbors (KNN)**
        - Non-parametric algorithm that classifies based on similarity to k closest training samples
        - Optimal k found through cross-validation sweep
        - Best for: Small to medium datasets with clear boundaries
        
        **Logistic Regression**
        - Linear model that finds decision boundaries between speaker classes
        - Fast training and prediction
        - Best for: Linearly separable features
        
        **Random Forest**
        - Ensemble of decision trees with voting mechanism
        - Handles non-linear relationships well
        - Best for: Complex patterns, resistant to overfitting
        
        **Gradient Boosting**
        - Sequential ensemble that corrects previous model errors
        - Often achieves highest accuracy
        - Best for: Maximum performance, handles feature interactions
        
        **Support Vector Classifier (SVC)**
        - Finds optimal hyperplane separating speaker classes
        - RBF kernel for non-linear boundaries
        - Best for: High-dimensional feature spaces
        """)

    # ==============================================================================
    # Training Configuration
    # ==============================================================================
    blank_lines(2)
    st.markdown("#### Training Configuration")

    config_cols = st.columns(2)
    with config_cols[0]:
        st.markdown("**Data Split**")
        st.write(f"- Training: 70% ({int(num_samples * 0.7)} samples)")
        st.write(f"- Testing: 30% ({int(num_samples * 0.3)} samples)")
        st.write(f"- Random seed: {cfg['RANDOM_SEED']}")
    
    with config_cols[1]:
        st.markdown("**Feature Processing**")
        st.write(f"- Features: {num_features} MFCC-based")
        st.write(f"- Scaling: StandardScaler (mean=0, std=1)")
        st.write(f"- Sample rate: {cfg['SAMPLE_RATE']} Hz")

    st.markdown("""
    **Model files saved:**
    - `models/[model_name].pkl` - Trained model files
    - `models/scaler.pkl` - Feature scaler for preprocessing
    - `data/model_summary.json` - Best model metadata
    - `data/all_model_scores.csv` - Complete performance metrics
    """)


# Run standalone for debugging
if __name__ == "__main__":
    st.set_page_config(page_title="Model Training", layout="wide")
    page5()