# 🎯 Brain Tumor MRI Classification - Command Reference

Quick reference for all commands and scripts in this project.

---

## 📦 Installation

```bash
# Install all dependencies
pip install -r requirements.txt

# Install specific packages
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn streamlit opencv-python pillow
```

---

## 📊 Data Exploration

### Jupyter Notebook
```bash
# Launch data exploration notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Test Data Pipeline
```bash
# Test preprocessing pipeline
python scripts/data_preprocessing.py
```

---

## 🏋️ Model Training

### Train Single Model

```bash
# Custom CNN (from scratch)
python scripts/train.py --model custom_cnn --epochs 50 --batch_size 32 --lr 0.001

# Lightweight CNN (faster training)
python scripts/train.py --model lightweight_cnn --epochs 30 --batch_size 32

# ResNet50 (transfer learning)
python scripts/train.py --model resnet50 --epochs 30 --batch_size 16 --lr 0.0001

# MobileNet (mobile-optimized)
python scripts/train.py --model mobilenet --epochs 30 --batch_size 32 --lr 0.0001

# InceptionV3 (high accuracy)
python scripts/train.py --model inceptionv3 --epochs 30 --batch_size 16 --lr 0.0001

# EfficientNetB0 (balanced performance)
python scripts/train.py --model efficientnetb0 --epochs 30 --batch_size 32 --lr 0.0001
```

### Train All Models
```bash
# Train all models sequentially
python scripts/train_all_models.py

# Train specific models only
python scripts/train_all_models.py --models custom_cnn resnet50 mobilenet

# Training only (skip evaluation)
python scripts/train_all_models.py --train_only

# Evaluation only (skip training)
python scripts/train_all_models.py --eval_only
```

### Training Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--model` | Model architecture | custom_cnn | resnet50 |
| `--data_dir` | Data directory path | ../data | ./my_data |
| `--img_size` | Image size (square) | 224 | 299 |
| `--batch_size` | Batch size | 32 | 16 |
| `--epochs` | Number of epochs | 50 | 30 |
| `--lr` | Learning rate | 0.001 | 0.0001 |
| `--no_class_weights` | Disable class weights | False | (flag) |

---

## 📈 Model Evaluation

### Evaluate Single Model
```bash
# Evaluate custom CNN
python scripts/evaluate.py \
  --model_path models/custom_cnn_best.h5 \
  --model_name custom_cnn \
  --data_dir ../data

# Evaluate ResNet50
python scripts/evaluate.py \
  --model_path models/resnet50_best.h5 \
  --model_name resnet50
```

### Compare All Models
```python
# In Python or Jupyter
from scripts.evaluate import compare_models
import json

# Load results
results = {}
models = ['custom_cnn', 'resnet50', 'mobilenet', 'inceptionv3', 'efficientnetb0']

for model in models:
    with open(f'results/{model}_evaluation.json', 'r') as f:
        results[model] = json.load(f)

# Compare
df = compare_models(results)
print(df)
```

---

## 🌐 Streamlit Web Application

### Launch App
```bash
# Start Streamlit server
streamlit run app/streamlit_app.py

# Specify port
streamlit run app/streamlit_app.py --server.port 8080

# Open in browser automatically
streamlit run app/streamlit_app.py --server.headless false
```

### Access App
- **Local**: http://localhost:8501
- **Network**: http://YOUR_IP:8501

---

## 📊 Monitoring & Logging

### TensorBoard
```bash
# Launch TensorBoard
tensorboard --logdir results/logs/tensorboard

# Specify port
tensorboard --logdir results/logs/tensorboard --port 6006

# Access: http://localhost:6006
```

### View Training Logs
```bash
# View CSV logs
cat results/logs/custom_cnn_*_log.csv

# In Python
import pandas as pd
df = pd.read_csv('results/logs/custom_cnn_20231228_120000_log.csv')
print(df.head())
```

---

## 🧪 Testing & Utilities

### Test Model Architectures
```bash
# Test custom CNN
python scripts/custom_cnn.py

# Test transfer learning models
python scripts/transfer_learning.py

# Test utilities
python scripts/utils.py
```

### Check GPU
```python
# In Python
import tensorflow as tf
print("GPUs:", tf.config.list_physical_devices('GPU'))
print("GPU Name:", tf.test.gpu_device_name())
```

```bash
# Or run utility
python scripts/utils.py
```

---

## 📓 Jupyter Notebooks

```bash
# Launch Jupyter
jupyter notebook

# Launch specific notebook
jupyter notebook notebooks/01_data_exploration.ipynb
jupyter notebook notebooks/02_model_training.ipynb

# Launch JupyterLab (alternative)
jupyter lab
```

---

## 🐍 Python API Usage

### Load and Use Trained Model

```python
from tensorflow import keras
import numpy as np
from PIL import Image

# Load model
model = keras.models.load_model('models/resnet50_best.h5')

# Load and preprocess image
img = Image.open('path/to/mri_image.jpg')
img = img.resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class]

print(f"Class: {predicted_class}, Confidence: {confidence:.2%}")
```

### Custom Training Script

```python
from scripts.train import train_model

# Train with custom parameters
model, history, config = train_model(
    model_type='resnet50',
    data_dir='../data',
    img_size=(224, 224),
    batch_size=16,
    epochs=30,
    learning_rate=0.0001,
    use_class_weights=True
)
```

---

## 📁 File Management

### Model Files
```bash
# List all models
ls -lh models/

# Best models (saved during training)
models/custom_cnn_best.h5
models/resnet50_best.h5
models/mobilenet_best.h5

# Final models (after training completion)
models/custom_cnn_final.h5
```

### Results Files
```bash
# Training plots
results/{model_name}_training_history.png

# Confusion matrices
results/{model_name}_confusion_matrix.png

# Model comparison
results/model_comparison.png
results/model_comparison.csv

# Evaluation results
results/{model_name}_evaluation.json

# Configuration
results/{model_name}_config.json

# Training logs
results/logs/{model_name}_{timestamp}_log.csv
```

---

## 🔍 Common Workflows

### Complete Training Pipeline
```bash
# 1. Explore data
jupyter notebook notebooks/01_data_exploration.ipynb

# 2. Train models
python scripts/train_all_models.py

# 3. View in TensorBoard (optional)
tensorboard --logdir results/logs/tensorboard

# 4. Deploy app
streamlit run app/streamlit_app.py
```

### Quick Single Model Training
```bash
# Train
python scripts/train.py --model custom_cnn --epochs 30

# Evaluate
python scripts/evaluate.py \
  --model_path models/custom_cnn_best.h5 \
  --model_name custom_cnn

# Deploy
streamlit run app/streamlit_app.py
```

### Transfer Learning Workflow
```bash
# Train multiple transfer learning models
for model in resnet50 mobilenet inceptionv3 efficientnetb0; do
    python scripts/train.py --model $model --epochs 30 --batch_size 16
done

# Compare results
python -c "from scripts.evaluate import compare_models; compare_models(...)"
```

---

## 🛠️ Troubleshooting Commands

### Memory Issues
```bash
# Reduce batch size
python scripts/train.py --model resnet50 --batch_size 8

# Use lighter model
python scripts/train.py --model mobilenet --batch_size 32
```

### Check Installation
```python
import tensorflow as tf
import streamlit as st
import cv2
import PIL

print(f"TensorFlow: {tf.__version__}")
print(f"Streamlit: {st.__version__}")
print("All packages installed!")
```

### Verify Dataset
```bash
# Check directory structure
tree data/

# Count images per class
find data/train -type f -name "*.jpg" | wc -l
find data/validation -type f -name "*.jpg" | wc -l
find data/test -type f -name "*.jpg" | wc -l
```

---

## 📊 Batch Operations

### Evaluate All Models
```bash
# Bash script
for model in custom_cnn resnet50 mobilenet inceptionv3 efficientnetb0; do
    python scripts/evaluate.py \
      --model_path models/${model}_best.h5 \
      --model_name $model
done
```

### Export All Results
```bash
# Create results archive
zip -r results_$(date +%Y%m%d).zip results/
```

---

## 🎓 Learning & Experimentation

### Modify Hyperparameters
```bash
# Experiment with learning rates
python scripts/train.py --model custom_cnn --lr 0.01
python scripts/train.py --model custom_cnn --lr 0.001
python scripts/train.py --model custom_cnn --lr 0.0001

# Experiment with batch sizes
python scripts/train.py --model custom_cnn --batch_size 8
python scripts/train.py --model custom_cnn --batch_size 16
python scripts/train.py --model custom_cnn --batch_size 32
python scripts/train.py --model custom_cnn --batch_size 64
```

### Custom Model Architecture
Edit `scripts/custom_cnn.py` and modify the `create_custom_cnn()` function.

---

## 📚 Additional Resources

### Documentation
- [README.md](README.md) - Project overview
- [QUICKSTART.md](QUICKSTART.md) - Setup guide
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Complete summary

### Online Resources
- TensorFlow: https://tensorflow.org/
- Keras: https://keras.io/
- Streamlit: https://streamlit.io/

---

## 💡 Pro Tips

1. **Always start with data exploration**: Understand your dataset before training
2. **Use TensorBoard**: Monitor training in real-time
3. **Start with lighter models**: Test pipeline with fast-training models first
4. **Save regularly**: Models are auto-saved, but backup important results
5. **Compare models**: Train multiple architectures to find the best
6. **Monitor GPU usage**: Ensure GPU is being utilized during training
7. **Use class weights**: Essential for imbalanced medical datasets

---

*For detailed explanations, see [QUICKSTART.md](QUICKSTART.md)*
