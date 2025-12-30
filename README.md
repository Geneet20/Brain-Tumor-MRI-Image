# 📄 Brain Tumor MRI Image Classification

A comprehensive deep learning project for classifying brain MRI images into multiple tumor categories using custom CNN and transfer learning approaches.

## 🎯 Project Overview

This project develops an AI-powered solution for medical imaging analysis, specifically for brain tumor classification from MRI scans. It combines custom CNN architectures with state-of-the-art transfer learning models and provides an interactive web interface for real-time predictions.

## 🔬 Skills & Technologies

- **Deep Learning**: CNN, Transfer Learning
- **Frameworks**: TensorFlow/Keras
- **Languages**: Python
- **Deployment**: Streamlit
- **Domain**: Medical Imaging, Computer Vision

## 📂 Project Structure

```
Brain Tumor MRI Image Classification/
├── data/                      # Dataset directory (add your MRI images here)
│   ├── train/
│   ├── validation/
│   └── test/
├── models/                    # Saved trained models (.h5 files)
├── notebooks/                 # Jupyter notebooks for exploration and analysis
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
├── scripts/                   # Python scripts for training and evaluation
│   ├── data_preprocessing.py
│   ├── custom_cnn.py
│   ├── transfer_learning.py
│   ├── train.py
│   └── evaluate.py
├── app/                       # Streamlit web application
│   └── streamlit_app.py
├── results/                   # Training results, plots, and metrics
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- GPU support (optional, but recommended for faster training)

### Installation

1. Clone this repository or download the project files

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset from [Tumour(updated)](https://www.kaggle.com/datasets) and place it in the `data/` directory

## 📊 Dataset Structure

Organize your dataset in the following structure:
```
data/
├── train/
│   ├── class1/
│   ├── class2/
│   └── class3/
├── validation/
│   ├── class1/
│   ├── class2/
│   └── class3/
└── test/
    ├── class1/
    ├── class2/
    └── class3/
```

## 💻 Usage

### 1. Data Exploration
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 2. Train Models
```bash
# Train custom CNN
python scripts/train.py --model custom_cnn

# Train transfer learning models
python scripts/train.py --model resnet50
python scripts/train.py --model mobilenet
python scripts/train.py --model inceptionv3
python scripts/train.py --model efficientnetb0
```

### 3. Evaluate Models
```bash
python scripts/evaluate.py --model_path models/best_model.h5
```

### 4. Run Streamlit Application
```bash
streamlit run app/streamlit_app.py
```

## 🏗️ Model Architecture

### Custom CNN
- Multiple convolutional layers with ReLU activation
- Max pooling for spatial dimension reduction
- Batch normalization for training stability
- Dropout layers for regularization
- Dense layers for classification

### Transfer Learning Models
- **ResNet50**: Deep residual learning network
- **MobileNet**: Lightweight model for efficient inference
- **InceptionV3**: Multi-scale feature extraction
- **EfficientNetB0**: Balanced accuracy and efficiency

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Training/Validation Loss and Accuracy Curves

## 🌐 Business Applications

1. **AI-Assisted Medical Diagnosis**: Support radiologists with rapid tumor classification
2. **Early Detection & Patient Triage**: Automatic flagging of high-risk cases
3. **Research & Clinical Trials**: Patient dataset segmentation by tumor type
4. **Second-Opinion AI Systems**: Remote diagnostic support for underserved regions

## 📝 Project Deliverables

- ✅ Trained models (custom CNN and pretrained models)
- ✅ Interactive Streamlit application
- ✅ Complete training and evaluation scripts
- ✅ Comprehensive documentation
- ✅ Model comparison analysis
- ✅ Clean, modular, well-commented code

## 🏷️ Technical Tags

Deep Learning, Image Classification, Medical Imaging, Brain MRI Analysis, CNN, Transfer Learning, TensorFlow, Keras, Data Augmentation, Data Preprocessing, Model Evaluation, Streamlit Deployment, Healthcare AI, Computer Vision, AI in Radiology

## 👤 Author

Project developed as part of medical imaging AI research

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- Dataset source: Kaggle Tumor Dataset
- Pretrained models: ImageNet weights
- Framework: TensorFlow/Keras team
