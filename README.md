# Automated Preprocessing Pipeline for Coronary Artery Calcium (CAC) Scoring

This repository provides a Python-based pipeline designed to transform raw medical imaging data (**DICOM**) and clinician annotations (**XML**) from the **Stanford COCA Dataset** into a deep learning-ready format (**2.5D NumPy arrays**).

## 🌟 Key Features
- **Hounsfield Unit (HU) Conversion**: Automatically rescales raw pixel values to standardized HU.
- **Dynamic Windowing**: Optimized range of **[-100, 500] HU** to enhance visibility of bone and calcification.
- **2.5D Spatial Stacking**: Generates 3-channel image tensors (Previous, Current, Next slice).
- **Automated Batch Processing**: Efficiently handles large patient cohorts with built-in support for resuming interrupted tasks.

## 📁 Project Structure
The repository is organized according to the team's standard structure. All processing scripts are located in the `src/` directory.

```text
coronary-calcification-segmentation/
├── app/                 # Application & UI logic
├── data/                # Local data storage (Raw DICOM & Processed NPY)
├── models/              # Model architectures and saved weights
├── src/                 # Core Source Code
│   ├── agatston_score.py
│   ├── data_processing.py
│   ├── post_processing.py
│   └── pre_processing.py
├── .gitignore
├── README.md
└── requirements.txt

### 3. Directory Layout (Output)
When the pipeline finishes, the `data/` folder will be populated as follows:
- `data/processed_npy/images/`: Contains the `.npy` files for the CT scans.
- `data/processed_npy/masks/`: Contains the corresponding binary ground-truth labels.

>  **Note for Reviewers:** To maintain repository speed, the actual `.npy` files are stored externally. You can verify the output quality by downloading the sample batch here: [**Google Drive Link**](https://drive.google.com/drive/folders/18Ka7Q3hTVjM2Zf0BZsQCM8fnXsPz982x?usp=sharing)