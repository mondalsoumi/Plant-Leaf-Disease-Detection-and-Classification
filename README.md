# Leaf Disease Detection System

A machine learning-based web application for detecting plant leaf diseases and providing treatment recommendations.

## Features

- **Disease Detection**: Upload images of plant leaves to detect various diseases
- **Multiple Disease Support**: Detects 10 different plant diseases including:
  - Bacterial Spot
  - Early Blight
  - Late Blight
  - Leaf Mold
  - Septoria Leaf Spot
  - Spider Mites
  - Target Spot
  - Tomato Yellow Leaf Curl Virus
  - Two-Spotted Spider Mite
  - Healthy plants

- **Detailed Information**: For each detected disease, provides:
  - Disease description
  - Prevention steps
  - Treatment options
  - Key symptoms

- **Disease Comparison**: Compare different diseases side by side
- **Detection History**: Save and track your detection history
- **Camera Integration**: Capture images directly from your device

## Technologies Used

- **Backend**: Flask, TensorFlow, Keras
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Machine Learning**: Convolutional Neural Network (CNN)

## Prerequisites

Before installing the application, make sure you have:
- Python 3.8 or higher
- pip (Python package installer)
- Git

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Shivansh3636/Leaf-Disease-Detection-.git
   cd Leaf-Disease-Detection-
   ```

2. Download Required Model Files:
   - Download the pre-trained model files from [Google Drive](https://drive.google.com/drive/folders/1mWfkgC_Fv5h2oau3eKWz34mYCL2Fu84z?usp=sharing)
   - Place the following files in the root directory:
     - `leaf_disease_model.h5` (437MB)
     - `best_model.h5` (437MB)

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up the database:
   ```
   python auth.py
   ```
   This will create the necessary database for user authentication.

5. Run the application:
   ```
   python main.py
   ```

6. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Important Notes

### Large Files and Resources

#### 1. Model Files
The pre-trained model files can be downloaded from:
- [Google Drive - Models Folder](https://drive.google.com/drive/folders/1mWfkgC_Fv5h2oau3eKWz34mYCL2Fu84z?usp=sharing)
  - `leaf_disease_model.h5` (437MB)
  - `best_model.h5` (437MB)

#### 2. Dataset
The training and testing datasets are available at:
- [Google Drive - Dataset Folder](https://drive.google.com/drive/folders/1uCYCZ61obZEBcUDQeAf9vHfNgXfBqJtT?usp=sharing)
  - Training data: `train_data.npy` (115MB)
  - Testing data: `test_data.npy` (115MB)
  - Labels: `test_labels.npy` (528B)
  - Confusion matrix: `confusion_matrix.npy` (928B)
  - Training history: `training_history.json` (287B)
  - Model metrics: `metrics.json` (53B)
  - Performance plots: `training_history.png` (51KB), `confusion_matrix.png` (74KB)

#### 3. Documentation
Project documentation is available in multiple formats:
- [Google Drive - Documentation](https://drive.google.com/drive/folders/2-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx)
  - Project Vision Document
  - Architecture Diagrams
  - Use Case Diagrams
- [GitHub Wiki](https://github.com/Shivansh3636/Leaf-Disease-Detection-/wiki)
  - Technical Documentation
  - API Reference
  - Development Guide
- [ReadTheDocs](https://leaf-disease-detection.readthedocs.io)
  - User Guide
  - Installation Guide
  - Troubleshooting Guide

### File Structure
```
project/
├── main.py              # Main application file
├── auth.py             # Authentication module
├── model.py            # Model architecture
├── requirements.txt    # Python dependencies
├── templates/          # HTML templates
├── static/            # Static files (CSS, JS, images)
├── uploads/           # User uploaded images
└── dataset/           # Training dataset
```

## Dataset Structure

Place your dataset in the following structure:
```
dataset/
    train/
        healthy/
            image1.jpg
            image2.jpg
            ...
        diseased/
            image1.jpg
            image2.jpg
            ...
    val/
        healthy/
            image1.jpg
            image2.jpg
            ...
        diseased/
            image1.jpg
            image2.jpg
            ...
```

## Usage

1. Upload a leaf image or use the camera to capture one
2. Click "Analyze" to detect the disease
3. View detailed information about the detected disease
4. Save the detection to history
5. Use the comparison tool to compare different diseases

## Troubleshooting

Common issues and solutions:
1. **Model not found**: Make sure you've downloaded and placed the model files in the correct location
2. **Database errors**: Run `python auth.py` to recreate the database
3. **Memory issues**: The model requires at least 4GB of RAM to run properly

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Shivansh3636 