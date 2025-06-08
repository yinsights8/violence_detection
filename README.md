# violence_detection
A comparative study on deep learning models (CNN, LSTM, ViT) for violence detection using traditional training, transfer learning, and meta-learning. Achieved 98.82% accuracy with Vision Transformer.


# Violence Detection Using Deep Learning Architectures

This project presents a comprehensive performance comparison of state-of-the-art deep learning models—CNN, LSTM, and Vision Transformer (ViT)—applied to the critical task of violence detection in images. Developed as part of the CS826 coursework, the study investigates the impact of traditional training, transfer learning, and meta-learning paradigms across different architectures.

## 📌 Project Objectives

- Compare baseline CNN and LSTM models for violence classification.
- Implement transfer learning using ResNet50 and EfficientNetB0.
- Apply Model-Agnostic Meta-Learning (MAML) to CNN and LSTM.
- Evaluate a Vision Transformer (ViT) on the same task.
- Analyze training efficiency, performance, and real-world applicability.

## 🧠 Deep Learning Models

| Learning Paradigm     | Architecture         | Accuracy  | Key Strength                          |
|-----------------------|----------------------|-----------|----------------------------------------|
| Base Model            | LSTM                 | 98.74%    | High accuracy, good recall             |
| Base Model            | CNN                  | 97.12%    | Strong baseline model                  |
| Transfer Learning     | CNN (ResNet50)       | 80.43%    | Moderate accuracy                      |
| Transfer Learning     | LSTM (EfficientNet)  | 52.80%    | High recall, biased towards positives  |
| Meta Learning         | CNN (MAML)           | 95.00%    | Fast adaptation, high generalization   |
| Meta Learning         | LSTM (MAML)          | 51.25%    | Strong bias, needs tuning              |
| Transformer           | Vision Transformer   | 98.82%    | Best performance & fastest convergence |

## 🗂 Dataset

- Total: 2,934 images (balanced)
  - Train: 1,584
  - Validation: 680
  - Test: 670
- Categories: `Violent`, `Non-Violent`
- Preprocessed to 224×224 resolution with normalization.

## 📈 Evaluation Metrics

- Accuracy
- Precision & Recall
- F1-Score
- Confusion Matrix
- ROC-AUC
- Training Efficiency (Epochs, Convergence Time)

## 🚀 How to Run

1. Clone the repository and open the notebook in Google Colab.
2. Mount Google Drive and upload the dataset into the appropriate folder structure.
3. Run each cell sequentially:
    - Data preparation
    - Model selection (choose CNN, LSTM, ViT, etc.)
    - Training and evaluation

## 🔧 Tools and Technologies

- Python
- TensorFlow / Keras
- PyTorch
- Torchvision
- Vision Transformer (ViT)
- Scikit-learn
- Google Colab

## 👨‍💻 Authors

- Yash Pramodrao Dhakade  
- Muhammad Panji Muryandi  
- Alvee Morsele Kabir  
- Nuzhat Tarannum Ibrahimy  

## 📝 Future Work

- Improve meta-learning performance with LSTM.
- Extend to real-world video violence detection.
- Integrate facial/body pose detection for early prediction of violence.

## 📌 License

This project is for educational purposes and part of the CS826 module at the University of Strathclyde.
