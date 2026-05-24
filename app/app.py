# app/app.py
import os
import sys
import shutil
from pathlib import Path

# Add project root path to sys.path to avoid ModuleNotFoundError
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import pydicom

# Import core processing logic
from src.pre_processing import extract_zip_dicom
from src.data_processing import COCATransformer
from models.model import UNetModel

# Import clinical Agatston scoring module
from src.agatston_score import calculate_agatston_for_volume

# Define temporary storage and model weight paths
TEMP_STORAGE_DIR = root_path / "app" / "temp_storage"
WEIGHT_PATH = root_path / "models" / "weights" / "best_model.pt"

# Initialize U-Net Model
@st.cache_resource
def init_model(weight_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetModel(in_channels=3, out_channels=1, base_channels=16)
    if weight_file.exists():
        model.load_state_dict(torch.load(str(weight_file), map_location=device))
        model.to(device)
        model.eval()
        return model, device, True
    return model, device, False

model, device, is_loaded = init_model(WEIGHT_PATH)
transformer = COCATransformer()

# Configure Streamlit Page Layout
st.set_page_config(page_title="Coronary Artery Calcification AI System", layout="wide")

st.title("🫀 Coronary Artery Calcification Segmentation & Automated Agatston Scoring")
st.markdown("---")

# Sidebar Configuration
with st.sidebar:
    st.header("📂 Data Upload")
    uploaded_file = st.file_uploader("Select Patient Zip containing DICOM series (.zip)", type=["zip"])
    
    if not is_loaded:
        st.error("⚠️ 'best_model.pt' weights not found! Running in structural simulation mode.")
    else:
#        st.success(f"⚡ : {str(device).upper()}")
        st.success(f"⚡ Model loaded successfully on {str(device).upper()}")

# --- STATE MANAGEMENT ---
if "current_file_name" not in st.session_state:
    st.session_state["current_file_name"] = None

if uploaded_file is not None and st.session_state["current_file_name"] != uploaded_file.name:
    st.session_state["processed"] = False
    st.session_state["current_file_name"] = uploaded_file.name
elif uploaded_file is None:
    st.session_state["processed"] = False
    st.session_state["current_file_name"] = None

# Main Pipeline Execution
if uploaded_file is not None and not st.session_state.get('processed', False):
    if st.sidebar.button("Run AI Diagnostics Pipeline", type="primary"):
        with st.spinner("Extracting DICOMs, preprocessing volume, and running neural network inference..."):
            
            # Extract ZIP file
            extract_dir = extract_zip_dicom(uploaded_file, str(TEMP_STORAGE_DIR))
            
            # Load and sort DICOM folder by Z-axis geometry
            dicom_files, pixel_spacing = transformer.load_and_sort_dicom_folder(str(extract_dir))
            
            if len(dicom_files) == 0:
                st.error("Error: No valid .dcm slices found in the uploaded zip file.")
                st.stop()
                
            # Process volume using 2.5D anatomical stacking
            images_25d_list, hu_volume_list = transformer.prepare_25d_volume(dicom_files)
            
            # Cast to NumPy arrays for compatibility with model inference
            images_25d = np.array(images_25d_list, dtype=np.float32)
            hu_volume = np.array(hu_volume_list, dtype=np.float32)
            
            # Model prediction
            pred_mask_volume, raw_hu_volume = model.predict_volume(images_25d, hu_volume, device)
            
            # Calculate Agatston Score and Cardiovascular Risk Category
            total_score, risk_label, risk_color, df_slices = calculate_agatston_for_volume(
                pred_mask_volume, raw_hu_volume, pixel_spacing
            )
            
            # Cache results in Session State
            st.session_state['pred_mask_volume'] = pred_mask_volume
            st.session_state['hu_volume'] = raw_hu_volume
            st.session_state['total_score'] = total_score
            st.session_state['risk_label'] = risk_label
            st.session_state['risk_color'] = risk_color
            st.session_state['df_slices'] = df_slices
            st.session_state['processed'] = True
            
            st.rerun()

# --- DIAGNOSTIC OUTPUT DISPLAY ---
if st.session_state.get('processed', False):
    
    # Section 1: Quantitative Clinical Metrics Summary
    st.header("📊 Coronary Artery Calcium (CAC) Summary")
    card_col1, card_col2 = st.columns(2)
    
    with card_col1:
        st.metric(label="Total Coronary Artery Calcium (Agatston Score)", value=f"{st.session_state['total_score']:.2f}")
        
    with card_col2:
        st.markdown("**Cardiovascular Event Risk Stratification:**")
        st.markdown(
            f"<h3 style='color:{st.session_state['risk_color']}; margin-top:0px; font-weight:bold;'>"
            f"{st.session_state['risk_label']}</h3>", 
            unsafe_allow_html=True
        )
        
    st.markdown("---")
    
    # Section 2: Dual Column Layout (Visualization vs. Per-Slice Statistics)
    layout_col1, layout_col2 = st.columns([5, 3])
    
    pred_mask_volume = st.session_state['pred_mask_volume']
    hu_volume = st.session_state['hu_volume']
    num_slices = pred_mask_volume.shape[0]
    df_slices = st.session_state['df_slices']

    with layout_col2:
        st.subheader("📋 Calcium Distribution Per Slice")
        
        if not df_slices.empty:
            # Rename columns to English for the UI
            df_slices_en = df_slices.rename(columns={
                "Slice Index": "Slice Index",
                "Agatston Score": "Agatston Score"
            })
            
            st.markdown("*💡 Click on a row in the table below to jump directly to that specific slice:*")
            
            # FIXED: Changed selection_mode from "single" to "single-row" to prevent StreamlitAPIException
            selected_row = st.dataframe(
                df_slices_en, 
                use_container_width=True, 
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row"
            )
            
            # Determine default index based on user click interaction
            default_slice_idx = int(num_slices / 2)
            if selected_row and len(selected_row.get("selection", {}).get("rows", [])) > 0:
                clicked_row_idx = selected_row["selection"]["rows"][0]
                default_slice_idx = int(df_slices_en.iloc[clicked_row_idx]["Slice Index"])
        else:
            st.info("No clinically significant calcification detected on any slice.")
            default_slice_idx = int(num_slices / 2)
            
    with layout_col1:
        st.subheader("🖼️ Segmentation Mask Visualization")
        
        # The slider can scan ALL slices, but its position updates automatically if a row is selected above
        slice_idx = st.slider(
            "Select Axial CT Slice Index (Allows scanning full volume)", 
            0, num_slices - 1, default_slice_idx
        )
        
        current_mask = pred_mask_volume[slice_idx]
        current_hu = hu_volume[slice_idx]
        
        # Soft tissue / Bone windowing [-160, 240]
        windowed_view = np.clip(current_hu, -160, 240)
        
        # Generate Matplotlib plots with English descriptions
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
        
        # Frame 1: Original CT Scan
        axes[0].imshow(windowed_view, cmap='gray')
        axes[0].set_title(f"Original CT (Slice {slice_idx})", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Frame 2: AI Predicted Binary Mask
        axes[1].imshow(current_mask, cmap='gray')
        axes[1].set_title("AI Predicted Mask", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Frame 3: Red Overlay & Neon Green Boundary Lines
        axes[2].imshow(windowed_view, cmap='gray')
        
        color_mask = np.zeros((current_mask.shape[0], current_mask.shape[1], 3), dtype=np.uint8)
        color_mask[current_mask == 1] = [255, 0, 0]  # Pure Red for calcified lesions
        
        alpha_mask = np.where(current_mask == 1, 0.70, 0.0)
        axes[2].imshow(color_mask, alpha=alpha_mask)
        
        if np.any(current_mask == 1):
            axes[2].contour(current_mask, colors='#00FF00', levels=[0.5], linewidths=1.5)
            
        axes[2].set_title("Lesion Overlay (Red + Neon Green Border)", fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        st.pyplot(fig)