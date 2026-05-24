import os
import numpy as np
import time
# from pipeline import CalciumScoringPipeeles

# Configuration & Paths
BASE_DIR = r"G:\My Drive\AIVIETNAM 2026\Conquer\CAC - AIO Conquer\cocacoronarycalciumandchestcts-2\Gated_release_final"
ROOT_DICOM = os.path.join(BASE_DIR, "patient")
ROOT_XML = os.path.join(BASE_DIR, "calcium_xml")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_npy")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    pipeline = CalciumScoringPipeline()
    
    # Sort patients numerically for consistent processing
    all_patients = sorted([p for p in os.listdir(ROOT_DICOM) if os.path.isdir(os.path.join(ROOT_DICOM, p))],
                          key=lambda x: int(x) if x.isdigit() else 999)

    BATCH_SIZE = 10
    print(f"🚀 Total patients to process: {len(all_patients)}")

    for i in range(0, len(all_patients), BATCH_SIZE):
        current_batch = all_patients[i : i + BATCH_SIZE]
        print(f"\n--- Processing Batch {i//BATCH_SIZE + 1} ({len(current_batch)} patients) ---")

        for patient_id in current_batch:
            # Update paths to check inside subdirectories for resuming
            out_x = os.path.join(DIR_IMAGES, f"{patient_id}_X.npy")
            out_y = os.path.join(DIR_MASKS, f"{patient_id}_Y.npy")
            xml_path = os.path.join(ROOT_XML, f"{patient_id}.xml")
            
            if os.path.exists(out_x):
                print(f"   ⏩ {patient_id}: Already processed, skipping.")
                continue

            if os.path.exists(xml_path):
                try:
                    dicom_path = os.path.join(ROOT_DICOM, patient_id)
                    X, Y = pipeline.run(dicom_path, xml_path)
                    
                    np.save(out_x, X)
                    np.save(os.path.join(OUTPUT_DIR, f"{patient_id}_Y.npy"), Y)
                    print(f"   ✅ {patient_id}: Success (Samples: {len(X)})")
                except Exception as e:
                    print(f"   ❌ {patient_id}: Error -> {e}")
            else:
                print(f"   ⚠️ {patient_id}: XML not found.")
        
        # Sleep to prevent network drive or memory congestion
        time.sleep(2)

if __name__ == "__main__":
    main()
