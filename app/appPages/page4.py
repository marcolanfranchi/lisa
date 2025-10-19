import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')
from appPages.components import section_header, data_metrics

def page4():
    """Display the page for step 4-filter-and-balance.py."""

    section_header("Feature Engineering", "Extracted vocal features from audio clips for analysis and model training.")
#     data_metrics(from_step=4)

#     # Load features data
#     df = load_features_data()
    
#     if df is None:
#         st.warning("No features data found. Please ensure features.csv exists in data/generated/")
#         return
    
#     # Data overview
#     st.write("")
#     st.markdown("#### Dataset Overview")
    
#     speakers = df['speaker_id'].unique()
#     total_samples = len(df)
#     total_features = len([col for col in df.columns if col not in ['speaker_id', 'clip_filename', 'duration']])
    
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("Total Samples", total_samples)
#     with col2:
#         st.metric("Speakers", len(speakers))
#     with col3:
#         st.metric("Features", total_features)
#     with col4:
#         st.metric("Samples per Speaker", f"{df['speaker_id'].value_counts().iloc[0]}")
    
#     # Feature groups
#     feature_groups, feature_columns = get_feature_groups(df)
    
#     # Feature group analysis
#     st.write("")
#     st.markdown("#### Feature Group Analysis")
    
#     selected_group = st.selectbox(
#         "Select feature group for distribution analysis:",
#         list(feature_groups.keys()),
#         format_func=lambda x: f"{x.upper()} ({len(feature_groups[x])} features)"
#     )
    
#     if feature_groups[selected_group]:
#         dist_fig = create_feature_distribution_plot(df, selected_group, feature_groups[selected_group])
#         if dist_fig:
#             st.plotly_chart(dist_fig)
    
#     # PCA Analysis
#     st.write("")
#     st.markdown("#### Principal Component Analysis")
    
#     pca_scatter, pca_variance, variance_ratios = create_pca_analysis(df, feature_columns)
    
#     col1, col2 = st.columns(2)
#     with col1:
#         st.plotly_chart(pca_scatter)
#     with col2:
#         st.plotly_chart(pca_variance)
    
#     st.write(f"First 3 PCs explain {sum(variance_ratios[:3]):.2%} of total variance")
    
#     # t-SNE Analysis
#     st.write("")
#     st.markdown("#### t-SNE Visualization")
#     st.write("Non-linear dimensionality reduction to visualize speaker clustering")
    
#     if st.button("Generate t-SNE Plot"):
#         with st.spinner("Computing t-SNE (this may take a moment)..."):
#             tsne_fig = create_tsne_visualization(df, feature_columns)
#             st.plotly_chart(tsne_fig)
    
#     # Feature Importance
#     st.write("")
#     st.markdown("#### Feature Importance Analysis")
    
#     importance_fig, importance_df = create_feature_importance_analysis(df, feature_columns)
#     st.plotly_chart(importance_fig)
    
#     # Show top features
#     st.write("**Top 10 Most Discriminative Features:**")
#     top_10_features = importance_df.head(10)[['feature', 'importance']]
#     st.dataframe(top_10_features, hide_index=True)
    
#     # Correlation Analysis
#     st.write("")
#     st.markdown("#### Feature Correlation Analysis")
    
#     corr_fig = create_correlation_heatmap(df, feature_columns)
#     st.plotly_chart(corr_fig)
    
#     # Speaker Comparison
#     st.write("")
#     st.markdown("#### Speaker Characteristics Comparison")
    
#     radar_fig = create_speaker_comparison_radar(df, feature_groups)
#     st.plotly_chart(radar_fig)
    
#     # Statistical Summary
#     st.write("")
#     st.markdown("#### Statistical Summary by Speaker")
    
#     summary_stats = []
#     for speaker in speakers:
#         speaker_data = df[df['speaker_id'] == speaker]
#         stats = {
#             'Speaker': speaker,
#             'Sample Count': len(speaker_data),
#             'Avg MFCC_0': speaker_data['mfcc_0_mean'].mean() if 'mfcc_0_mean' in df.columns else 'N/A',
#             'Avg F0': speaker_data['f0_mean'].mean() if 'f0_mean' in df.columns else 'N/A',
#             'Avg RMS': speaker_data['rms_mean'].mean() if 'rms_mean' in df.columns else 'N/A',
#             'Avg Spectral Centroid': speaker_data['spectral_centroid_mean'].mean() if 'spectral_centroid_mean' in df.columns else 'N/A'
#         }
#         summary_stats.append(stats)
    
#     summary_df = pd.DataFrame(summary_stats)
#     st.dataframe(summary_df, hide_index=True)
    
#     # Model Training Readiness Check
#     st.write("")
#     st.markdown("#### Model Training Readiness")
    
#     # Check for issues
#     issues = []
#     if df.isnull().any().any():
#         issues.append("Missing values detected in features")
    
#     if len(speakers) < 2:
#         issues.append("Need at least 2 speakers for classification")
    
#     # Check class balance
#     class_counts = df['speaker_id'].value_counts()
#     if class_counts.max() / class_counts.min() > 3:
#         issues.append("Significant class imbalance detected")
    
#     if not issues:
#         st.success("Data appears ready for model training!")
#         st.info(f"Ready to train on {total_samples} samples with {total_features} features across {len(speakers)} speakers")
#     else:
#         for issue in issues:
#             st.warning(issue)

# def load_features_data(features_path="data/generated/vocal_features.csv"):
#     """Load features CSV and prepare for analysis"""
#     try:
#         df = pd.read_csv(features_path)
#         return df
#     except Exception as e:
#         st.error(f"Error loading features: {str(e)}")
#         return None

# def get_feature_groups(df):
#     """Group features by type for organized analysis"""
#     feature_columns = [col for col in df.columns if col not in ['speaker_id', 'clip_filename', 'duration']]
    
#     groups = {
#         'mfcc': [col for col in feature_columns if col.startswith('mfcc_')],
#         'spectral': [col for col in feature_columns if any(x in col for x in ['spectral_', 'zcr_', 'rms_'])],
#         'prosodic': [col for col in feature_columns if any(x in col for x in ['f0_', 'voicing_rate'])],
#         'mel': [col for col in feature_columns if col.startswith('mel_')],
#         'chroma': [col for col in feature_columns if col.startswith('chroma_')],
#         'temporal': [col for col in feature_columns if any(x in col for x in ['tempo', 'beat_', 'onset_', 'silence_'])]
#     }
    
#     return groups, feature_columns

# def create_feature_distribution_plot(df, feature_group, group_features):
#     """Create distribution plots for a feature group"""
#     if not group_features:
#         return None
    
#     # Select subset of features to avoid overcrowding
#     display_features = group_features[:8] if len(group_features) > 8 else group_features
    
#     fig = make_subplots(
#         rows=2, cols=4,
#         subplot_titles=display_features,
#         vertical_spacing=0.15,
#         horizontal_spacing=0.1
#     )
    
#     speakers = df['speaker_id'].unique()
#     colors = px.colors.qualitative.Set3[:len(speakers)]
    
#     for i, feature in enumerate(display_features):
#         row = (i // 4) + 1
#         col = (i % 4) + 1
        
#         for j, speaker in enumerate(speakers):
#             speaker_data = df[df['speaker_id'] == speaker][feature]
            
#             fig.add_trace(
#                 go.Histogram(
#                     x=speaker_data,
#                     name=f"{speaker}",
#                     opacity=0.7,
#                     legendgroup=speaker,
#                     showlegend=(i == 0),  # Only show legend for first subplot
#                     marker_color=colors[j]
#                 ),
#                 row=row, col=col
#             )
    
#     fig.update_layout(
#         title=f"{feature_group.upper()} Features Distribution by Speaker",
#         height=600,
#         barmode='overlay'
#     )
    
#     return fig

# def create_pca_analysis(df, feature_columns):
#     """Create PCA visualization and analysis"""
#     # Prepare data
#     X = df[feature_columns].fillna(0)
#     y = df['speaker_id']
    
#     # Standardize features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     # PCA
#     pca = PCA()
#     X_pca = pca.fit_transform(X_scaled)
    
#     # Create PCA scatter plot
#     pca_df = pd.DataFrame({
#         'PC1': X_pca[:, 0],
#         'PC2': X_pca[:, 1], 
#         'PC3': X_pca[:, 2],
#         'speaker_id': y
#     })
    
#     fig_scatter = px.scatter(
#         pca_df, x='PC1', y='PC2', color='speaker_id',
#         title="PCA Analysis - First Two Principal Components",
#         labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
#                 'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)'}
#     )
    
#     # Explained variance plot
#     cumsum_var = np.cumsum(pca.explained_variance_ratio_)
#     fig_variance = go.Figure()
    
#     fig_variance.add_trace(go.Scatter(
#         x=list(range(1, len(cumsum_var[:20]) + 1)),
#         y=cumsum_var[:20],
#         mode='lines+markers',
#         name='Cumulative Variance'
#     ))
    
#     fig_variance.update_layout(
#         title="PCA Explained Variance (First 20 Components)",
#         xaxis_title="Principal Component",
#         yaxis_title="Cumulative Explained Variance Ratio",
#         height=400
#     )
    
#     return fig_scatter, fig_variance, pca.explained_variance_ratio_

# def create_tsne_visualization(df, feature_columns):
#     """Create t-SNE visualization"""
#     # Prepare data
#     X = df[feature_columns].fillna(0)
#     y = df['speaker_id']
    
#     # Standardize features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     # t-SNE (use subset if too many samples)
#     if len(X_scaled) > 1000:
#         # Sample data for t-SNE to avoid long computation
#         sample_indices = np.random.choice(len(X_scaled), 1000, replace=False)
#         X_sample = X_scaled[sample_indices]
#         y_sample = y.iloc[sample_indices]
#     else:
#         X_sample = X_scaled
#         y_sample = y
    
#     tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_sample)-1))
#     X_tsne = tsne.fit_transform(X_sample)
    
#     tsne_df = pd.DataFrame({
#         'TSNE1': X_tsne[:, 0],
#         'TSNE2': X_tsne[:, 1],
#         'speaker_id': y_sample
#     })
    
#     fig = px.scatter(
#         tsne_df, x='TSNE1', y='TSNE2', color='speaker_id',
#         title="t-SNE Visualization of Feature Space"
#     )
    
#     return fig

# def create_feature_importance_analysis(df, feature_columns):
#     """Calculate and visualize feature importance using mutual information"""
#     # Prepare data
#     X = df[feature_columns].fillna(0)
#     y = df['speaker_id']
    
#     # Calculate mutual information
#     mi_scores = mutual_info_classif(X, y, random_state=42)
    
#     # Create feature importance dataframe
#     importance_df = pd.DataFrame({
#         'feature': feature_columns,
#         'importance': mi_scores
#     }).sort_values('importance', ascending=False)
    
#     # Top 20 features
#     top_features = importance_df.head(20)
    
#     fig = px.bar(
#         top_features, x='importance', y='feature',
#         title="Top 20 Most Important Features (Mutual Information)",
#         orientation='h'
#     )
#     fig.update_layout(height=600)
    
#     return fig, importance_df

# def create_correlation_heatmap(df, feature_columns, top_n=20):
#     """Create correlation heatmap for top features"""
#     # Select top features based on variance
#     feature_vars = df[feature_columns].var().sort_values(ascending=False)
#     top_features = feature_vars.head(top_n).index.tolist()
    
#     # Calculate correlation matrix
#     corr_matrix = df[top_features].corr()
    
#     fig = go.Figure(data=go.Heatmap(
#         z=corr_matrix.values,
#         x=corr_matrix.columns,
#         y=corr_matrix.columns,
#         colorscale='RdBu',
#         zmid=0
#     ))
    
#     fig.update_layout(
#         title=f"Feature Correlation Heatmap (Top {top_n} by Variance)",
#         height=600
#     )
    
#     return fig

# def create_speaker_comparison_radar(df, feature_groups):
#     """Create radar chart comparing speaker characteristics"""
#     speakers = df['speaker_id'].unique()
    
#     # Calculate mean values for each feature group per speaker
#     radar_data = []
#     group_names = []
    
#     for group_name, features in feature_groups.items():
#         if features:  # Only include non-empty groups
#             group_names.append(group_name)
#             for speaker in speakers:
#                 speaker_data = df[df['speaker_id'] == speaker]
#                 mean_value = speaker_data[features].mean().mean()  # Mean of means
#                 radar_data.append({
#                     'speaker': speaker,
#                     'feature_group': group_name,
#                     'value': mean_value
#                 })
    
#     radar_df = pd.DataFrame(radar_data)
    
#     fig = go.Figure()
    
#     for speaker in speakers:
#         speaker_data = radar_df[radar_df['speaker'] == speaker]
        
#         fig.add_trace(go.Scatterpolar(
#             r=speaker_data['value'],
#             theta=speaker_data['feature_group'],
#             fill='toself',
#             name=speaker,
#             opacity=0.7
#         ))
    
#     fig.update_layout(
#         polar=dict(
#             radialaxis=dict(visible=True)
#         ),
#         title="Speaker Characteristics Comparison (Feature Group Averages)",
#         height=500
#     )
    
#     return fig


