# Brain Tumor MRI Classification - Quick Start Guide

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- 8GB+ RAM recommended
- GPU with CUDA support (optional but recommended)

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
# Navigate to project directory
cd "Brain Tumor MRI Image Classification"

# Install required packages
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download the brain tumor MRI dataset and organize it in the following structure:

```
data/
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
├── validation/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── test/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

**Note:** Replace the class folder names with your actual tumor types.

## 📊 Step-by-Step Workflow

### Step 1: Data Exploration

Explore your dataset using the data exploration notebook:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

This will help you understand:
- Class distribution
- Image properties
- Data imbalance
- Sample visualizations

### Step 2: Train Models

#### Option A: Train a Single Model

```bash
# Train Custom CNN
python scripts/train.py --model custom_cnn --epochs 50 --batch_size 32

# Train ResNet50
python scripts/train.py --model resnet50 --epochs 30 --batch_size 16

# Train MobileNet
python scripts/train.py --model mobilenet --epochs 30 --batch_size 32

# Train InceptionV3
python scripts/train.py --model inceptionv3 --epochs 30 --batch_size 16

# Train EfficientNetB0
python scripts/train.py --model efficientnetb0 --epochs 30 --batch_size 32
```

#### Option B: Train All Models

```bash
python scripts/train_all_models.py
```

**Training Parameters:**
- `--model`: Model architecture to train
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--data_dir`: Path to data directory (default: ../data)
- `--no_class_weights`: Disable class weights for imbalanced data

### Step 3: Evaluate Models

Evaluate a single model:

```bash
python scripts/evaluate.py --model_path models/custom_cnn_best.h5 --model_name custom_cnn
```

### Step 4: Compare Models

After training multiple models, compare their performance:

```python
# Run in Python or Jupyter
from scripts.evaluate import compare_models
import json

# Load all evaluation results
results = {}
for model in ['custom_cnn', 'resnet50', 'mobilenet', 'inceptionv3', 'efficientnetb0']:
    with open(f'results/{model}_evaluation.json', 'r') as f:
        results[model] = json.load(f)

# Compare
comparison_df = compare_models(results)
```

### Step 5: Deploy Streamlit App

Launch the web application:

```bash
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 🎯 Common Commands

### Test Data Pipeline

```bash
python scripts/data_preprocessing.py
```

### Test Model Architectures

```bash
# Test custom CNN
python scripts/custom_cnn.py

# Test transfer learning models
python scripts/transfer_learning.py
```

### Check GPU Availability

```bash
python scripts/utils.py
```

## 📈 Monitoring Training

### TensorBoard

Monitor training in real-time with TensorBoard:

```bash
tensorboard --logdir results/logs/tensorboard
```

Open http://localhost:6006 in your browser

### Training Logs

Training logs are saved as CSV files in `results/logs/`:
- `{model_name}_{timestamp}_log.csv`

## 🔍 Troubleshooting

### Out of Memory (OOM) Errors

1. Reduce batch size:
   ```bash
   python scripts/train.py --model resnet50 --batch_size 8
   ```

2. Use a lighter model:
   ```bash
   python scripts/train.py --model lightweight_cnn
   ```

### Slow Training

1. Ensure GPU is being used:
   ```python
   import tensorflow as tf
   print("GPU Available:", tf.config.list_physical_devices('GPU'))
   ```

2. Reduce image size (edit in scripts):
   ```python
   img_size = (128, 128)  # Instead of (224, 224)
   ```

### Dataset Not Found

Ensure your data directory structure matches the expected format:
```bash
python scripts/data_preprocessing.py
```

## 📊 Understanding Results

### Training History Plot
- Shows accuracy and loss curves for training and validation
- Located in: `results/{model_name}_training_history.png`

### Confusion Matrix
- Shows true vs predicted classifications
- Located in: `results/{model_name}_confusion_matrix.png`

### Model Comparison
- Compares all models across metrics
- Located in: `results/model_comparison.png` and `model_comparison.csv`

## 🎓 Tips for Better Results

1. **Data Augmentation**: Already implemented in the pipeline
   - Rotation, shifting, zoom, brightness adjustments

2. **Class Weights**: Automatically calculated for imbalanced datasets
   - Use `--no_class_weights` to disable

3. **Early Stopping**: Prevents overfitting
   - Training stops if validation loss doesn't improve for 10 epochs

4. **Model Checkpointing**: Saves best model
   - Based on validation accuracy

5. **Learning Rate Scheduling**: Reduces learning rate when stuck
   - Patience of 5 epochs

## 📝 Model Selection Guide

| Model | Best For | Speed | Accuracy | Parameters |
|-------|----------|-------|----------|------------|
| **Custom CNN** | Learning & experimentation | Fast | Good | Low |
| **Lightweight CNN** | Quick prototyping | Very Fast | Fair | Very Low |
| **ResNet50** | High accuracy | Medium | Excellent | High |
| **MobileNet** | Mobile deployment | Fast | Very Good | Low |
| **InceptionV3** | Complex patterns | Slow | Excellent | Very High |
| **EfficientNetB0** | Best balance | Medium | Excellent | Medium |

## 🚀 Production Deployment

### Save Best Model
Models are automatically saved in `models/` directory:
- `{model_name}_best.h5` - Best model during training
- `{model_name}_final.h5` - Final model after training

### Using Saved Model
```python
from tensorflow import keras

# Load model
model = keras.models.load_model('models/resnet50_best.h5')

# Make prediction
prediction = model.predict(preprocessed_image)
```

## 📚 Additional Resources

- **TensorFlow Documentation**: https://www.tensorflow.org/
- **Keras API**: https://keras.io/
- **Streamlit Docs**: https://docs.streamlit.io/
- **Medical Imaging AI**: Research papers and articles

## ⚠️ Important Notes

1. **Medical Disclaimer**: This is an educational project. Do NOT use for actual medical diagnosis without proper validation and regulatory approval.

2. **Data Privacy**: Ensure compliance with healthcare data regulations (HIPAA, GDPR, etc.) when working with medical images.

3. **Model Validation**: Always validate models on diverse, representative datasets before any real-world application.

## 💬 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review error messages in console
3. Verify dataset structure and file formats
4. Check TensorFlow/GPU compatibility

## 📄 License

This project is for educational and research purposes. Ensure proper licensing for any datasets used.
