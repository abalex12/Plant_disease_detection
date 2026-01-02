# Tomato Leaf Disease Detection with Vision Transformer

A deep learning-based system for automated detection and classification of tomato leaf diseases using Vision Transformer (ViT) architecture. This project leverages transfer learning and state-of-the-art computer vision techniques to assist in early disease diagnosis for improved crop management.

## Overview

Plant diseases pose significant challenges to agricultural productivity and food security. This system provides an automated approach to identifying tomato leaf diseases from digital images, enabling rapid diagnosis and timely intervention. The implementation utilizes a pre-trained Vision Transformer model fine-tuned on plant disease imagery to achieve robust classification performance.

## Key Features

- Pre-trained Vision Transformer (ViT) backbone for feature extraction
- Custom classification head for disease-specific predictions
- Comprehensive data augmentation pipeline to improve model generalization
- Differential learning rates for optimized fine-tuning
- Detailed evaluation metrics including confusion matrices and classification reports
- Model checkpointing to preserve best-performing weights

## Requirements

### Dependencies

- Python 3.7 or higher
- PyTorch
- torchvision
- transformers (Hugging Face)
- scikit-learn
- matplotlib
- seaborn
- numpy

### Installation

Install all required packages using pip:
```bash
pip install torch torchvision transformers scikit-learn matplotlib seaborn numpy
```

For GPU acceleration, ensure CUDA-compatible PyTorch is installed according to your system configuration.

## Dataset

This project uses the New Plant Diseases Dataset (Augmented), which contains labeled images of healthy and diseased plant leaves across multiple categories.  

## Methodology

### Data Preprocessing

**Training Pipeline:**
- Image resizing to 224×224 pixels (ViT input requirement)
- Random horizontal flipping for geometric augmentation
- Random rotation (±10 degrees) to simulate natural variations
- Color jitter adjustments (brightness, contrast, saturation, hue)
- Normalization using ImageNet statistics (mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])

**Validation Pipeline:**
- Image resizing to 224×224 pixels
- Normalization using ImageNet statistics

### Model Architecture

The architecture combines a pre-trained Vision Transformer with a custom classification layer:

**Base Model:**
- Vision Transformer: `google/vit-base-patch16-224`
- Pre-trained on ImageNet-21k
- Hidden dimension: 768
- Patch size: 16×16 pixels

**Classification Head:**
- Fully connected layer mapping 768 features to output classes
- Supports multi-class disease classification

### Training Configuration

- Loss Function: Cross-Entropy Loss
- Optimizer: Adam with differential learning rates
  - ViT backbone: Lower learning rate for fine-tuning
  - Classification head: Higher learning rate for faster adaptation
- Training Epochs: 20
- Model Selection: Best validation accuracy checkpoint

The differential learning rate strategy preserves pre-trained features while allowing the classification head to adapt rapidly to the specific disease detection task.


## Evaluation Metrics

The model performance is assessed using multiple metrics:

- **Accuracy**: Overall classification correctness
- **Precision**: Positive predictive value per class
- **Recall**: Sensitivity or true positive rate per class
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions vs. actual labels

## Results Visualization

Training progress and model performance are visualized through:

- Training and validation loss curves across epochs
- Training and validation accuracy progression
- Confusion matrix heatmap for multi-class performance analysis
- Per-class precision, recall, and F1-score comparison

## Model Performance

The fine-tuned Vision Transformer achieves competitive accuracy in distinguishing between healthy and diseased tomato leaves across multiple disease categories. Detailed metrics are generated during evaluation and saved for analysis.


## Future Enhancements

- Expand dataset to include additional crop types and diseases
- Implement ensemble methods for improved robustness
- Deploy model as a mobile or web application for field use
- Integrate real-time monitoring capabilities
- Add explainability features (attention visualization, Grad-CAM)

## Contributing

Contributions are welcome to improve model performance, expand disease coverage, or enhance documentation. Please ensure code follows established conventions and includes appropriate tests.

## License

This project is available for research and educational purposes. Please refer to the dataset license for usage restrictions on the training data.

## Acknowledgments

This project builds upon the Vision Transformer architecture developed by Google Research and utilizes the Hugging Face Transformers library for model implementation. The dataset is provided by the plant pathology research community.

## References

- Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- Hughes, D. P., & Salathé, M. (2015). "An open access repository of images on plant health to enable the development of mobile disease diagnostics"

---

**Note**: Ensure proper attribution when using this system in research publications or commercial applications. Model performance may vary depending on image quality, lighting conditions, and disease presentation.
