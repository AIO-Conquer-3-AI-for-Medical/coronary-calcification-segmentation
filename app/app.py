# app/app.py
import os
import sys
import shutil
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from src.pre_processing import extract_zip_dicom
from src.data_processing import COCATransformer
from models.model import UNetModel
from src.agatston_score import calculate_agatston_score, xml_plist_to_mask_volume

WEIGHT_PATH = root_path / "models" / "weights" / "best_model.pt"
BASE_TEMP_DIR = root_path / "app" / "temp_storage"
BASE_TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Hậu xử lý khởi động: Xóa sạch các thư mục rác tồn đọng từ các lần chạy app trước đó
@st.cache_resource
def clean_all_legacy_temp_storage():
    if BASE_TEMP_DIR.exists():
        import shutil
        shutil.rmtree(BASE_TEMP_DIR, ignore_errors=True)
    BASE_TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Chạy hàm dọn dẹp hệ thống một lần duy nhất khi khởi động ứng dụng
clean_all_legacy_temp_storage()

# Initialize independent session storage to prevent cross-session IO conflicts
if "session_id" not in st.session_state:
    import uuid
    st.session_state["session_id"] = str(uuid.uuid4())
USER_TEMP_DIR = BASE_TEMP_DIR / st.session_state["session_id"]

# Initialize analysis execution state
if "run_analysis" not in st.session_state:
    st.session_state["run_analysis"] = False

st.set_page_config(layout="wide", page_title="End-to-End Coronary Artery Calcium Quantification System")
st.title("🫀 Automated Coronary Artery Calcium Segmentation & Agatston Scoring")

# --- SIDEBAR CONFIGURATION & UPLOAD LAYER ---
st.sidebar.header("⚙️ System Configuration")
score_threshold = st.sidebar.slider("AI Binary Segmentation Threshold", 0.1, 0.9, 0.60, 0.05)
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"💻 Active Compute Device: **{device.upper()}**")

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Patient Data Ingestion")
uploaded_zip = st.sidebar.file_uploader("1. Upload DICOM Series ZIP File:", type=["zip"])
uploaded_xml = st.sidebar.file_uploader("2. Upload Expert Ground Truth XML (Optional):", type=["xml"])

st.sidebar.markdown("---")
# Trigger analysis execution upon button press
if st.sidebar.button("🚀 Run AI Analysis", use_container_width=True):
    if uploaded_zip is not None:
        # 🛡️ FIX [Errno 39]: Force-wipes any remnant folders from previous runs in this session
        if USER_TEMP_DIR.exists():
            shutil.rmtree(USER_TEMP_DIR, ignore_errors=True)
        USER_TEMP_DIR.mkdir(parents=True, exist_ok=True)
        
        st.session_state["run_analysis"] = True
    else:
        st.sidebar.error("❌ Please upload a valid DICOM ZIP file first!")

# --- DATA PROCESSING & INFERENCE ENGINE ---
hu_volume = None
images_25d_stack = None
sorted_slices = None
pixel_spacing = [0.5, 0.5]

# Load Segmentation Model (Cached Resource Pattern)
@st.cache_resource
def load_segmentation_model(weight_path, target_device):
    model = UNetModel(in_channels=3, out_channels=1)
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=target_device))
    model.to(target_device)
    return model

model = load_segmentation_model(WEIGHT_PATH, device)

# Execute pipeline only if zip is uploaded and analysis state is active
if uploaded_zip is not None and st.session_state["run_analysis"]:
    with st.spinner("⏳ Extracting and pre-processing CT volume data..."):
        try:
            # 1. Extract raw DICOM series into sandbox temp directory
            dicom_extracted_path = extract_zip_dicom(uploaded_zip, USER_TEMP_DIR)
            
            # 2. Parse metadata and geometrically align slices along the anatomical Z-axis
            transformer = COCATransformer()
            sorted_slices, pixel_spacing = transformer.load_and_sort_dicom_folder(dicom_extracted_path)
            
            # 3. Reconstruct Hounsfield Unit (HU) volume and construct sliding 2.5D multi-channel stacks
            images_25d_stack, raw_hu_volume = transformer.transform_patient_volume(sorted_slices)
            
            # Array integrity enforcement: Squeeze dimensions to strict 2D matrices (Height, Width)
            standardized_hu = []
            for slice_img in raw_hu_volume:
                if isinstance(slice_img, torch.Tensor):
                    slice_img = slice_img.cpu().numpy()
                slice_flat = np.squeeze(slice_img)
                if slice_flat.shape != (512, 512):
                    slice_flat = cv2.resize(slice_flat, (512, 512), interpolation=cv2.INTER_LINEAR)
                standardized_hu.append(slice_flat)
            hu_volume = np.stack(standardized_hu, axis=0).astype(np.float32)
            
            st.success(f"✅ Successfully loaded {len(sorted_slices)} DICOM slices. Pixel Spacing: {pixel_spacing[0]:.3f} x {pixel_spacing[1]:.3f} mm².")
        except Exception as e:
            st.error(f"❌ Error processing DICOM image series: {str(e)}")
            st.stop()

    if hu_volume is not None and images_25d_stack is not None:
        # --- MODEL INFERENCE ---
        with st.spinner("🤖 Deep Learning model executing coronary calcification segmentation..."):
            try:
                raw_pred_volume, _ = model.predict_volume(images_25d_stack, hu_volume, device, threshold=score_threshold)
                
                # Standardize prediction masks geometry
                standardized_preds = []
                for mask in raw_pred_volume:
                    if isinstance(mask, torch.Tensor):
                        mask = mask.cpu().numpy()
                    mask_flat = np.squeeze(mask)
                    if mask_flat.shape != (512, 512):
                        mask_flat = cv2.resize(mask_flat, (512, 512), interpolation=cv2.INTER_NEAREST)
                    standardized_preds.append(mask_flat)
                pred_mask_volume = np.stack(standardized_preds, axis=0).astype(np.uint8)
            except Exception as e:
                st.error(f"❌ Inference processing error: {str(e)}")
                st.stop()

        # --- GROUND TRUTH SYNC & MASK GENERATION ---
        gt_mask_volume = np.zeros_like(hu_volume, dtype=np.uint8)
        gt_available = False

        if uploaded_xml is not None:
            with st.spinner("📑 Parsing expert annotations and executing semantic Z-matching..."):
                try:
                    temp_xml_path = USER_TEMP_DIR / uploaded_xml.name
                    with open(temp_xml_path, "wb") as f:
                        f.write(uploaded_xml.getbuffer())
                    
                    raw_gt_volume = xml_plist_to_mask_volume(str(temp_xml_path), sorted_slices, hu_volume)
                    
                    # Standardize ground truth masks geometry
                    standardized_gt = []
                    for gt_slice in raw_gt_volume:
                        if isinstance(gt_slice, torch.Tensor):
                            gt_slice = gt_slice.cpu().numpy()
                        gt_flat = np.squeeze(gt_slice)
                        if gt_flat.shape != (512, 512):
                            gt_flat = cv2.resize(gt_flat, (512, 512), interpolation=cv2.INTER_NEAREST)
                        standardized_gt.append(gt_flat)
                    gt_mask_volume = np.stack(standardized_gt, axis=0).astype(np.uint8)
                    
                    if np.any(gt_mask_volume > 0):
                        gt_available = True
                        st.success("🎯 Semantic Z-matching aligned Expert Ground Truth annotations successfully!")
                    else:
                        st.warning("⚠️ Valid XML loaded, but no Region of Interest (ROI) polygons matched the Z-axis slice range.")
                except Exception as e:
                    st.error(f"❌ XML parsing or matching error: {str(e)}")

        # --- CLINICAL QUANTIFICATION (AGATSTON SCORING) ---
        ai_score, ai_details_df = calculate_agatston_score(pred_mask_volume, hu_volume, pixel_spacing)
        
        if gt_available:
            gt_score, gt_details_df = calculate_agatston_score(gt_mask_volume, hu_volume, pixel_spacing)
            
            # Metrics Dashboard Panel
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric(label="Total Agatston Score (AI Model)", value=f"{ai_score:.2f}")
            m_col2.metric(label="Total Agatston Score (Expert GT)", value=f"{gt_score:.2f}")
            m_col3.metric(label="Absolute Deviation (MAE)", value=f"{abs(ai_score - gt_score):.2f}")
        else:
            st.metric(label="Total Agatston Score (AI Model)", value=f"{ai_score:.2f}")

        # --- CALCIFIED SLICES ANALYSIS VISUALIZER ---
        st.markdown("---")
        st.header("🔍 Calcified Slice-by-Slice Comparative Visualization")

        ai_slices = set(ai_details_df["slice_idx"].unique()) if not ai_details_df.empty else set()
        gt_slices = set(gt_details_df["slice_idx"].unique()) if (gt_available and not gt_details_df.empty) else set()
        calcified_slices = sorted(list(ai_slices.union(gt_slices)))

        if not calcified_slices:
            st.info("🎈 Excellent! No coronary artery calcification lesions detected across the entire volume by either AI or XML annotations.")
        else:
            st.write(f"Detected a total of **{len(calcified_slices)}** slices containing valid calcification plaques.")

            # 3-COLUMN TABLE: SLICE ID, AGATSTON AI, AGATSTON GT
            table_data = []
            for idx in calcified_slices:
                s_ai_score = 0.0
                s_gt_score = 0.0
                if not ai_details_df.empty and idx in ai_details_df["slice_idx"].values:
                    s_ai_score = ai_details_df[ai_details_df["slice_idx"] == idx]["agatston_score"].values[0]
                if gt_available and not gt_details_df.empty and idx in gt_details_df["slice_idx"].values:
                    s_gt_score = gt_details_df[gt_details_df["slice_idx"] == idx]["agatston_score"].values[0]
                table_data.append({
                    "Slice ID": int(idx),
                    "Agatston Score (AI)": f"{s_ai_score:.2f}",
                    "Agatston Score (Expert GT)": f"{s_gt_score:.2f}" if gt_available else "N/A"
                })
            
            st.markdown("#### 🗂️ Slice-Level Quantification Summary Table")
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

            st.markdown("---")

            # SLICE SELECTION NAVIGATOR
            st.markdown("#### 🎚️ Select Target Slice for Detailed Examination")
            selected_slice = st.slider(
                "Navigate through slices:",
                min_value=0,
                max_value=len(hu_volume) - 1,
                value=int(min(calcified_slices)) if calcified_slices else 0,
                step=1
            )

            # Extract data matrices for the selected slice
            img_slice = hu_volume[selected_slice]
            mask_ai = pred_mask_volume[selected_slice]
            mask_gt = gt_mask_volume[selected_slice] if gt_available else None

            # Apply windowing to optimize contrast for coronary artery visualization
            vmin, vmax = -200, 600
            img_clipped = np.clip(img_slice, vmin, vmax)

            # Metrics for current slice
            ai_info = ai_details_df[ai_details_df["slice_idx"] == selected_slice]
            gt_info = gt_details_df[gt_details_df["slice_idx"] == selected_slice] if gt_available else pd.DataFrame()

            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.subheader(f"📊 Quantitative Metrics (Slice {selected_slice})")
                metrics_data = {
                    "Evaluation Metric": ["Agatston Score", "Peak Density (Max HU)"],
                    "AI Model Prediction": [
                        f"{ai_info['agatston_score'].values[0]:.2f}" if not ai_info.empty else "0.00",
                        f"{ai_info['max_hu'].values[0]:.1f} HU" if not ai_info.empty else "N/A"
                    ]
                }
                if gt_available:
                    metrics_data["Expert Ground Truth"] = [
                        f"{gt_info['agatston_score'].values[0]:.2f}" if not gt_info.empty else "0.00",
                        f"{gt_info['max_hu'].values[0]:.1f} HU" if not gt_info.empty else "N/A"
                    ]
                st.table(pd.DataFrame(metrics_data))

            with info_col2:
                st.subheader("💡 Applied Clinical Standards")
                st.caption("- **Density Cutoff:** Lesions are only quantified if attenuation density hits the clinical calcium threshold: **≥ 130 HU**.")
                st.caption("- **Spatial Extent:** Connected component clusters must possess an isolated cross-sectional area **≥ 1 mm²** to filter noise artifacts.")

# --- PLOT NATIVE IMAGES AND SEGMENTATION MASKS (MATPLOTLIB OVERLAY) ---
            img_slice = hu_volume[selected_slice]
            mask_ai = pred_mask_volume[selected_slice]
            mask_gt = gt_mask_volume[selected_slice] if gt_available else None

            # Configure windowing (soft tissue/bone) to optimize contrast for coronary CT images
            vmin = -200
            max = 600
            img_clipped = np.clip(img_slice, vmin, vmax)

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.patch.set_facecolor('#0e1117') # Match Streamlit dark theme background color

            # Subplot 1: Native coronary CT image
            axes[0].imshow(img_clipped, cmap="gray", vmin=vmin, vmax=vmax)
            axes[0].set_title("Native CT Image (HU Windowed)", color="white", fontsize=12)
            axes[0].axis("off")

            # Subplot 2: AI predicted segmentation mask only
            mask_ai_colored = np.zeros((mask_ai.shape[0], mask_ai.shape[1], 3))
            mask_ai_colored[mask_ai > 0] = [1.0, 0.0, 0.0]  # Red
            axes[1].imshow(mask_ai_colored)
            axes[1].set_title(f"AI Predicted Segmentation (Slice {selected_slice})", color="orange", fontsize=12)
            axes[1].axis("off")

            # Subplot 3: Blend AI (red), GT (green), and overlap (yellow) masks overlay on native CT image
            axes[2].imshow(img_clipped, cmap="gray", vmin=vmin, vmax=vmax)
            # Create color array for overlay
            rgba_overlay = np.zeros((mask_ai.shape[0], mask_ai.shape[1], 4))
            
            if gt_available and mask_gt is not None:
                # AI-only region (Red)
                only_ai = (mask_ai > 0) & (mask_gt == 0)
                rgba_overlay[only_ai] = [1.0, 0.0, 0.0, 0.6]  # Red
                
                # GT-only region (Green)
                only_gt = (mask_ai == 0) & (mask_gt > 0)
                rgba_overlay[only_gt] = [0.0, 1.0, 0.0, 0.6]  # Green
                
                # Overlap region between AI and GT (Yellow)
                overlap = (mask_ai > 0) & (mask_gt > 0)
                rgba_overlay[overlap] = [1.0, 1.0, 0.0, 0.8]  # Yellow, Alpha = 0.8
                
                axes[2].imshow(rgba_overlay)
                axes[2].set_title(f"Overlap Masks (Slice {selected_slice}): Red=AI only, Green=GT only, Yellow=Overlap", color="lightblue", fontsize=11)
            else:
                # If Ground Truth is unavailable, show AI predicted overlay only
                rgba_overlay[mask_ai > 0] = [1.0, 0.0, 0.0, 0.6]  # Red
                axes[2].imshow(rgba_overlay)
                axes[2].set_title(f"AI Predicted Mask Overlay (Slice {selected_slice})", color="orange", fontsize=12)
            axes[2].axis("off")

            plt.tight_layout()
            st.pyplot(fig)

            plt.close(fig)
            del fig, axes

            # Detailed Logs Viewers
            st.markdown("#### 📝 Diagnostic Logs Checklist")
            tab_ai, tab_gt = st.tabs(["AI Model Predicted Lesions", "Expert Ground Truth Lesions (XML)"])
            with tab_ai:
                st.dataframe(ai_details_df, use_container_width=True)
            with tab_gt:
                if gt_available:
                    st.dataframe(gt_details_df, use_container_width=True)
                else:
                    st.caption("No expert validation XML file uploaded for this case.")

else:
    # Standby state view
    st.info("💡 Ready for execution. Please load the patient data in the sidebar and click 'Run AI Analysis'.")

# --- CONTEXT PERSISTENCE MEMORY CLEANUP ---
if USER_TEMP_DIR.exists() and uploaded_zip is None:
    shutil.rmtree(USER_TEMP_DIR)