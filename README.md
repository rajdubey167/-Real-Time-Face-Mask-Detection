# Real-Time Face Mask Detection

This project is a real-time face mask detection system built using Python, OpenCV, and TensorFlow. The system is designed to accurately classify and localize faces with or without masks, ensuring high reliability and efficient performance on edge devices.

---

## Features
- **Real-Time Detection**: Detects masked and unmasked faces in real-time.
- **High Accuracy**: Uses deep learning models trained on custom and pre-trained datasets.
- **Optimized for Edge Devices**: Low-latency pipelines for smooth performance on resource-constrained devices.
- **Adaptable**: Robust detection across diverse environments and conditions.

---

## Project Workflow
1. **Image Preprocessing**:
   - Utilized OpenCV for face detection and preprocessing steps, including resizing and normalization.

2. **Model Training**:
   - Leveraged TensorFlow for training deep learning models.
   - Integrated transfer learning with pre-trained models to boost performance.

3. **Object Detection**:
   - Implemented algorithms to classify and localize masked and unmasked faces.

4. **Optimization**:
   - Enhanced detection pipelines for real-time inference with low latency.

5. **Deployment**:
   - Deployed the system on edge devices to ensure reliable and efficient processing.

---

## Prerequisites
- Python 3.8 or later
- TensorFlow 2.x
- OpenCV 4.x
- NumPy
- Jupyter Notebook or any preferred IDE for development

---

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/real-time-face-mask-detection.git
   ```

2. Navigate to the project directory:
   ```bash
   cd real-time-face-mask-detection
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
1. **Prepare Dataset**:
   - Place images in directories (e.g., `data/masked` and `data/unmasked`) for training and testing.

2. **Train Model**:
   - Train the deep learning model by running:
     ```bash
     python train.py
     ```

3. **Run Real-Time Detection**:
   - Launch the detection system with a connected camera:
     ```bash
     python detect_mask.py
     ```

4. **Deploy on Edge Devices**:
   - Export the model for edge deployment and integrate it with your target device.

---

## Results
- Achieved an accuracy of **XX%** on the test dataset.
- Real-time detection latency of **XX ms** on edge devices.
- Reliable performance in diverse lighting and environmental conditions.

---

## References
- [OpenCV Documentation](https://docs.opencv.org/)
- [TensorFlow Object Detection API](https://www.tensorflow.org/lite/models/object_detection/overview)
- [Face Mask Dataset](https://www.kaggle.com/)

---

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests to improve the system.

---
