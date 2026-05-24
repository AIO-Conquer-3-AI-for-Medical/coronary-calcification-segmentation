import os
import pydicom
import numpy as np
import cv2
import xml.etree.ElementTree as ET

class CalciumScoringPipeline:
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size

    def _get_hu_image(self, path):
        """Reads DICOM, converts to Hounsfield Units (HU), and extracts Z-coordinate."""
        try:
            ds = pydicom.dcmread(path, force=True)
            if 'PixelData' not in ds: 
                return None
            
            # Convert raw pixel values to Hounsfield Units (HU)
            image = ds.pixel_array.astype(np.float32)
            image = image * ds.RescaleSlope + ds.RescaleIntercept
            
            # Windowing: Focus on bone/calcification range [0, 500] and normalize to [0, 1]
            image = np.clip((image - 0) / 500, 0, 1)
            
            return image, float(ds.ImagePositionPatient[2]), str(ds.SOPInstanceUID)
        except Exception:
            return None

    def _parse_xml(self, xml_path):
        """Parses XML to extract calcium coordinates for each image UID."""
        try:
            tree = ET.parse(xml_path)
            labels = {}
            for image_node in tree.findall(".//Image"):
                uid = image_node.find("UID").text
                points = []
                for roi in image_node.findall(".//ROI"):
                    pts = [[float(pt.text) for pt in p.findall("float")] 
                           for p in roi.findall(".//Point")]
                    if pts: 
                        points.append(np.array(pts, dtype=np.int32))
                if points: 
                    labels[uid] = points
            return labels
        except Exception:
            return {}

    def run(self, dicom_dir, xml_path):
        """Main Pipeline: Converts DICOM directory to 2.5D Numpy Arrays."""
        files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir)]
        slices = [self._get_hu_image(f) for f in files if f.endswith('.dcm')]
        
        # Sort slices by Z-coordinate
        slices = sorted([s for s in slices if s], key=lambda x: x[1])
        
        images, uids = [s[0] for s in slices], [s[2] for s in slices]
        labels = self._parse_xml(xml_path)
        
        X, Y = [], []
        # Generate 2.5D data (3 consecutive slices)
        for i in range(1, len(images) - 1):
            stack = np.stack([images[i-1], images[i], images[i+1]], axis=0)
            X.append(stack)
            
            mask = np.zeros(self.target_size, dtype=np.uint8)
            if uids[i] in labels:
                cv2.fillPoly(mask, labels[uids[i]], 1)
            Y.append(mask[np.newaxis, ...]) 

        return np.array(X), np.array(Y)