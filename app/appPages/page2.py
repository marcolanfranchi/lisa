from appPages.components import section_header, data_metrics
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import librosa
from streamlit_advanced_audio import audix, WaveSurferOptions

def page2():
    """Display the page for step 2-split-clips.py."""
    section_header("Audio Splitting", "Data splitting phase. Splits the cleaned audio recordings into smaller clips and stores them in `data/generated/processed_clips`.")
    
#     data_metrics(from_step=2)

#     # Load clips data
#     clips_data, manifest_df = load_clips_data()
    
#     if not clips_data:
#         st.warning("No clip data found. Please ensure clips exist in data/generated/processed_clips/")
#         return
    
#     # Current data overview
#     st.write("")
#     st.markdown("#### Current Data Status")
    
#     total_clips = sum(len(clips) for clips in clips_data.values())
#     speakers = list(clips_data.keys())
#     clips_per_speaker = [len(clips_data[speaker]) for speaker in speakers]
    
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("Total Clips", total_clips)
#     with col2:
#         st.metric("Speakers", len(speakers))
#     with col3:
#         st.metric("Min per Speaker", min(clips_per_speaker))
#     with col4:
#         st.metric("Max per Speaker", max(clips_per_speaker))
    
#     # RMS distribution for threshold selection
#     st.write("")
#     st.markdown("#### Step 1: Analyze RMS Distribution")
#     st.write("Examine the energy distribution to choose an appropriate filtering threshold")
    
#     rms_fig = create_rms_distribution_chart(clips_data)
#     if rms_fig:
#         st.plotly_chart(rms_fig)
    
#     # Interactive threshold setting
#     st.write("")
#     st.markdown("#### Step 2: Set Filtering Threshold")
    
#     # Calculate reasonable threshold range
#     all_rms = []
#     for clips in clips_data.values():
#         all_rms.extend([clip['rms_mean'] for clip in clips.values()])
    
#     min_rms = min(all_rms)
#     max_rms = max(all_rms)
#     median_rms = np.median(all_rms)
    
#     # Threshold slider
#     rms_threshold = st.slider(
#         "RMS Energy Threshold (clips below this will be filtered out):",
#         min_value=float(min_rms),
#         max_value=float(max_rms),
#         value=min_rms + (median_rms - min_rms) * 0.3,  # Start at reasonable default
#         step=(max_rms - min_rms) / 1000,
#         format="%.4f"
#     )
    
#     # Show threshold impact
#     threshold_fig = create_threshold_analysis_chart(clips_data, rms_threshold)
#     if threshold_fig:
#         st.plotly_chart(threshold_fig)
    
#     # Calculate and display filtering impact
#     st.write("")
#     st.markdown("#### Step 3: Filtering Impact Analysis")
    
#     impact_df = calculate_filtering_impact(clips_data, rms_threshold)
    
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         balance_fig = create_balance_analysis_chart(impact_df)
#         if balance_fig:
#             st.plotly_chart(balance_fig)
    
#     with col2:
#         st.write("**Filtering Results:**")
#         for _, row in impact_df.iterrows():
#             st.write(f"**{row['Speaker']}**")
#             st.write(f"  • Original: {row['Original Clips']} clips")
#             st.write(f"  • After filter: {row['Kept Clips']} clips")
#             st.write(f"  • Retention: {row['Retention Rate']:.1f}%")
#             st.write("")
    
#     # Balance target calculation
#     min_kept = impact_df['Kept Clips'].min()
#     total_after_balance = min_kept * len(speakers)
#     total_removed = total_clips - total_after_balance
    
#     st.info(f"**Final balanced dataset:** {total_after_balance} clips ({min_kept} per speaker)")
#     st.info(f"**Total clips removed:** {total_removed} ({(total_removed/total_clips)*100:.1f}% of original)")
    
#     # Sample clip inspection
#     st.write("")
#     st.markdown("#### Step 4: Inspect Threshold Boundary")
#     st.write("Listen to clips near the threshold to verify filtering quality")
    
#     # Find clips near the threshold
#     boundary_clips = []
#     threshold_range = (max_rms - min_rms) * 0.05  # 5% range around threshold
    
#     for speaker, clips in clips_data.items():
#         for clip_name, data in clips.items():
#             if abs(data['rms_mean'] - rms_threshold) <= threshold_range:
#                 boundary_clips.append({
#                     'speaker': speaker,
#                     'clip_name': clip_name,
#                     'rms': data['rms_mean'],
#                     'status': 'Keep' if data['rms_mean'] >= rms_threshold else 'Filter',
#                     'data': data
#                 })
    
#     # Sort by RMS value
#     boundary_clips.sort(key=lambda x: x['rms'])
    
#     if boundary_clips:
#         selected_boundary = st.selectbox(
#             "Select boundary clip to inspect:",
#             range(len(boundary_clips)),
#             format_func=lambda i: f"{boundary_clips[i]['speaker']} - {boundary_clips[i]['clip_name'][:15]}... (RMS: {boundary_clips[i]['rms']:.4f}) [{boundary_clips[i]['status']}]"
#         )
        
#         if selected_boundary is not None:
#             clip_info = boundary_clips[selected_boundary]
#             clip_data = clip_info['data']
            
#             col1, col2, col3, col4 = st.columns(4)
#             with col1:
#                 st.write(f":blue-badge[{clip_info['speaker']}]")
#             with col2:
#                 st.write(f":orange-badge[RMS: {clip_info['rms']:.4f}]")
#             with col3:
#                 color = "green" if clip_info['status'] == 'Keep' else "red"
#                 st.write(f":{color}-badge[{clip_info['status']}]")
#             with col4:
#                 st.write(f":gray-badge[Duration: {clip_data['duration']:.2f}s]")
            
#             # Audio player
#             options = WaveSurferOptions(
#                 wave_color="#2E8B57" if clip_info['status'] == 'Keep' else "#CD5C5C",
#                 progress_color="#1f77b4",
#                 height=80,
#                 bar_width=2,
#                 bar_gap=1,
#                 normalize=True
#             )
            
#             try:
#                 result = audix(clip_data['path'], wavesurfer_options=options)
#             except Exception as e:
#                 st.error(f"Error loading audio player: {str(e)}")
#                 st.audio(clip_data['path'])
    
#     else:
#         st.info("No clips found near the threshold boundary")
    
#     # Summary recommendations
#     st.write("")
#     st.markdown("#### Filtering Recommendations")
    
#     # Check for data balance issues
#     retention_rates = impact_df['Retention Rate'].values
#     if max(retention_rates) - min(retention_rates) > 30:
#         st.warning(f"Large variation in retention rates ({min(retention_rates):.1f}% - {max(retention_rates):.1f}%). Consider adjusting threshold or recording quality.")
    
#     if min_kept < 10:
#         st.warning(f"Very few clips remaining after balancing ({min_kept} per speaker). Consider lowering threshold or recording more data.")
#     elif min_kept > 100:
#         st.success(f"Good amount of data after balancing ({min_kept} clips per speaker).")
    
#     # Export threshold for next step
#     st.write("")
#     if st.button("Export Threshold for 3-filter-and-balance.py"):
#         st.code(f"RECOMMENDED_RMS_THRESHOLD = {rms_threshold:.6f}", language="python")
#         st.success("Threshold ready for use in filtering script!")

    


# def load_clips_data(data_path="data/generated/processed_clips", manifest_path="data/generated/manifest.csv"):
#     """Load clip files and manifest data"""
#     clips_data = {}
#     manifest_df = None
    
#     # Load manifest if exists
#     if os.path.exists(manifest_path):
#         try:
#             manifest_df = pd.read_csv(manifest_path)
#         except Exception as e:
#             st.error(f"Error loading manifest: {str(e)}")
    
#     if not os.path.exists(data_path):
#         return clips_data, manifest_df
    
#     # Load clip files and analyze
#     for speaker_folder in os.listdir(data_path):
#         speaker_path = os.path.join(data_path, speaker_folder)
#         if os.path.isdir(speaker_path):
#             clips_data[speaker_folder] = {}
#             clip_files = [f for f in os.listdir(speaker_path) if f.endswith('.wav')]
            
#             for clip_file in clip_files:
#                 file_path = os.path.join(speaker_path, clip_file)
#                 try:
#                     y, sr = librosa.load(file_path, sr=None)
#                     duration = len(y) / sr
                    
#                     # Calculate silence metrics
#                     rms = librosa.feature.rms(y=y)[0]
#                     rms_mean = np.mean(rms)
                    
#                     # Zero crossing rate (indicates speech activity)
#                     zcr = librosa.feature.zero_crossing_rate(y)[0]
#                     zcr_mean = np.mean(zcr)
                    
#                     clips_data[speaker_folder][clip_file] = {
#                         'path': file_path,
#                         'duration': duration,
#                         'sample_rate': sr,
#                         'rms_mean': rms_mean,
#                         'zcr_mean': zcr_mean,
#                         'audio_data': y
#                     }
#                 except Exception as e:
#                     st.error(f"Error loading {file_path}: {str(e)}")
    
#     return clips_data, manifest_df

# def calculate_filtering_impact(clips_data, rms_threshold):
#     """Calculate the impact of applying RMS filtering threshold"""
#     impact_data = []
    
#     for speaker, clips in clips_data.items():
#         total_clips = len(clips)
#         kept_clips = sum(1 for clip in clips.values() if clip['rms_mean'] >= rms_threshold)
#         filtered_clips = total_clips - kept_clips
        
#         impact_data.append({
#             'Speaker': speaker,
#             'Original Clips': total_clips,
#             'Kept Clips': kept_clips,
#             'Filtered Out': filtered_clips,
#             'Retention Rate': (kept_clips / total_clips * 100) if total_clips > 0 else 0
#         })
    
#     return pd.DataFrame(impact_data)

# def create_threshold_analysis_chart(clips_data, rms_threshold):
#     """Create visualization showing effect of RMS threshold"""
#     if not clips_data:
#         return None
    
#     # Prepare data with threshold classification
#     plot_data = []
#     for speaker, clips in clips_data.items():
#         for clip_name, data in clips.items():
#             plot_data.append({
#                 'Speaker': speaker,
#                 'Clip': clip_name,
#                 'RMS Energy': data['rms_mean'],
#                 'Status': 'Keep' if data['rms_mean'] >= rms_threshold else 'Filter Out'
#             })
    
#     df = pd.DataFrame(plot_data)
    
#     # Create scatter plot with threshold line
#     fig = px.scatter(df, x='Speaker', y='RMS Energy', color='Status',
#                     title=f"Clip Filtering Preview (RMS Threshold: {rms_threshold:.3f})",
#                     color_discrete_map={'Keep': '#2E8B57', 'Filter Out': '#CD5C5C'})
    
#     fig.add_hline(y=rms_threshold, line_dash="dash", line_color="black",
#                  annotation_text=f"Threshold: {rms_threshold:.3f}")
    
#     fig.update_layout(height=500)
    
#     return fig

# def create_balance_analysis_chart(impact_df):
#     """Create chart showing data balance after filtering"""
#     if impact_df.empty:
#         return None
    
#     fig = go.Figure()
    
#     # Bar chart showing kept clips per speaker
#     fig.add_trace(go.Bar(
#         name='Clips After Filtering',
#         x=impact_df['Speaker'],
#         y=impact_df['Kept Clips'],
#         marker_color='#2E8B57',
#         text=impact_df['Kept Clips'],
#         textposition='auto'
#     ))
    
#     # Add line showing minimum for balancing
#     min_clips = impact_df['Kept Clips'].min()
#     fig.add_hline(y=min_clips, line_dash="dash", line_color="orange",
#                  annotation_text=f"Balance Target: {min_clips} clips per speaker")
    
#     fig.update_layout(
#         title='Data Balance After Filtering',
#         xaxis_title='Speaker',
#         yaxis_title='Number of Clips',
#         showlegend=False
#     )
    
#     return fig

# def create_rms_distribution_chart(clips_data):
#     """Create histogram showing RMS distribution across all clips"""
#     if not clips_data:
#         return None
    
#     plot_data = []
#     for speaker, clips in clips_data.items():
#         for clip_name, data in clips.items():
#             plot_data.append({
#                 'Speaker': speaker,
#                 'RMS Energy': data['rms_mean']
#             })
    
#     df = pd.DataFrame(plot_data)
    
#     fig = px.histogram(df, x='RMS Energy', color='Speaker',
#                       title="RMS Energy Distribution - Choose Filtering Threshold",
#                       marginal="rug", nbins=50)

