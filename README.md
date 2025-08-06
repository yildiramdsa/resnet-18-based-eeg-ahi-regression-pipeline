# Sleep-Apnea Severity Estimation from EEG Spectrograms via ResNet-18

**Tech Stack:**
![ResNet18](https://img.shields.io/badge/ResNet18-D00000?logo=pytorch&logoColor=white)
![ResNet50](https://img.shields.io/badge/ResNet50-EE4C2C?logo=pytorch&logoColor=white)
![YOLOv5](https://img.shields.io/badge/YOLOv5-FF9900?logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-0072C6?logo=pytorch&logoColor=white)

An end‑to‑end deep learning pipeline that regresses continuous AHI from EEG spectrograms using a ResNet‑18 backbone. With subject-stratified validation, it achieves an RMSE of 6.8 events/hour and a Pearson correlation coefficient of 0.76, delivering accurate severity rankings across the entire AHI range. The workflow combines robust preprocessing, transfer learning optimization, and scalable inference for real-time sleep apnea screening.

**Live Demo:** [Sleep-Apnea Severity Estimation from EEG Spectrograms via ResNet-18](https://resnet-18-based-eeg-ahi-regression-pipeline.streamlit.app)
