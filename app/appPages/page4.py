import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from appPages.components import section_header, blank_lines
from utils import SPEAKER_COLOURS
from config import load_config

cfg = load_config()

def page4():
    """Display the page for step 4-extract-features.py."""
    # ==============================================================================
    # Header
    # ==============================================================================
    section_header(
        "Feature Extraction Overview",
        "This stage computes MFCC-based features for each balanced clip, including means and standard deviations for both MFCCs and their deltas. "
        "These features form the foundation for the speaker identification model trained in the next step. "
        "You can use these plots to identify which MFCC features differ most between speakers. "
        "Highly correlated features can be removed or reduced using PCA during model training."
    )

    # ==============================================================================
    # Load Features
    # ==============================================================================
    if not cfg["FEATURES_FILE"].exists():
        st.warning(f'No features file found at `{cfg["FEATURES_FILE"]}`. Please run 4-extract-features.py.')


        return

    try:
        df = pd.read_csv(cfg["FEATURES_FILE"])
    except Exception as e:
        st.error(f"Error loading features: {str(e)}")
        return

    if df.empty:
        st.warning("Feature file is empty. Please re-run 4-extract-features.py.")
        return

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    speaker_col = "speaker_id" if "speaker_id" in df.columns else None

    # ==============================================================================
    # Summary Metrics
    # ==============================================================================
    blank_lines(2)
    st.markdown("#### Extracted Feature Summary")

    num_samples = len(df)
    num_features = len(numeric_cols)
    num_speakers = df[speaker_col].nunique() if speaker_col else 0

    with st.container(border=False):
        metricHeight = 120
        cols = st.columns(3)
        with cols[0]:
            with st.container(border=True, height=metricHeight):
                st.markdown(":small[Samples:]")
                st.markdown(f"#### {num_samples}")
        with cols[1]:
            with st.container(border=True, height=metricHeight):
                st.markdown(":small[Speakers:]")
                st.markdown(f"#### {num_speakers}")
        with cols[2]:
            with st.container(border=True, height=metricHeight):
                st.markdown(":small[Feature columns:]")
                st.markdown(f"#### {num_features}")

    # ==============================================================================
    # Feature Distribution by Speaker
    # ==============================================================================
    blank_lines(2)
    st.markdown("#### Example Feature Distributions")

    # Pick one representative feature from each MFCC type
    feature_options = [col for col in numeric_cols if "mfcc_" in col]
    if feature_options:
        selected_feature = st.selectbox("Select a feature to visualize:", feature_options)
        if speaker_col:
            fig = px.box(
                df,
                x=speaker_col,
                y=selected_feature,
                color=speaker_col,
                color_discrete_map=SPEAKER_COLOURS,
                points=False,
                title=f"{selected_feature} distribution by speaker"
            )
            fig.update_layout(
                showlegend=False,
                height=400,
                margin=dict(l=40, r=40, t=60, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Speaker column not found in feature file.")
    else:
        st.info("No MFCC features found in dataset.")

    # ==============================================================================
    # Feature Correlation
    # ==============================================================================
    blank_lines(2)
    st.markdown("#### Feature Correlation Matrix")

    corr = df[numeric_cols].corr().round(2)
    corr_fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="gray",
        title="Correlation Matrix of Extracted Features"
    )
    corr_fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=40, r=40, t=60, b=40),
        height=600
    )
    st.plotly_chart(corr_fig, use_container_width=True)

    # ==============================================================================
    # Feature Relationship (Scatter)
    # ==============================================================================
    blank_lines(2)
    st.markdown("#### Feature Relationships")

    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        x_feature = col1.selectbox("X-axis:", numeric_cols, index=0)
        y_feature = col2.selectbox("Y-axis:", numeric_cols, index=1)

        if speaker_col:
            scatter_fig = px.scatter(
                df,
                x=x_feature,
                y=y_feature,
                color=speaker_col,
                color_discrete_map=SPEAKER_COLOURS,
                opacity=0.8,
                title=f"{x_feature} vs. {y_feature}"
            )
            scatter_fig.update_layout(
                showlegend=False,
                height=500,
                margin=dict(l=40, r=40, t=60, b=40),
            )
            st.plotly_chart(scatter_fig, use_container_width=True)
        else:
            st.info("Speaker column not found in feature file.")
    else:
        st.info("Not enough numeric features to generate scatter plot.")

    # ==============================================================================
    # Data Preview
    # ==============================================================================
    blank_lines(2)
    st.markdown("#### Feature Data Preview")

    st.dataframe(df.head(20), hide_index=True, use_container_width=True)

# Run standalone for debugging
if __name__ == "__main__":
    st.set_page_config(page_title="Feature Extraction", layout="wide")
    page4()
