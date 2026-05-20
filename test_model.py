import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import argparse

# Class names
CLASS_NAMES = [
    'Bacterial Spot', 
    'Early Blight', 
    'Healthy', 
    'Late Blight', 
    'Leaf Mold',
    'Septoria Leaf Spot',
    'Spider Mites',
    'Target Spot',
    'Tomato Yellow Leaf Curl Virus',
    'Two-Spotted Spider Mite'
]

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """Load and preprocess a single image"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_image(model, img_path):
    """Make prediction on a single image"""
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class] * 100
    
    # Get top 3 predictions
    top3_indices = np.argsort(predictions[0])[-3:][::-1]
    top3_predictions = [
        (CLASS_NAMES[idx], predictions[0][idx] * 100)
        for idx in top3_indices
    ]
    
    return {
        'predicted_class': CLASS_NAMES[predicted_class],
        'confidence': confidence,
        'top3_predictions': top3_predictions,
        'all_probabilities': {CLASS_NAMES[i]: predictions[0][i] * 100 for i in range(len(CLASS_NAMES))}
    }

def visualize_prediction(img_path, prediction_results):
    """Visualize the image and prediction results"""
    img = image.load_img(img_path, target_size=(224, 224))
    
    plt.figure(figsize=(12, 6))
    
    # Plot image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Predicted: {prediction_results['predicted_class']}\nConfidence: {prediction_results['confidence']:.2f}%")
    plt.axis('off')
    
    # Plot probabilities
    plt.subplot(1, 2, 2)
    classes = list(prediction_results['all_probabilities'].keys())
    probabilities = list(prediction_results['all_probabilities'].values())
    
    # Sort by probability
    sorted_indices = np.argsort(probabilities)
    classes = [classes[i] for i in sorted_indices]
    probabilities = [probabilities[i] for i in sorted_indices]
    
    plt.barh(range(len(classes)), probabilities)
    plt.yticks(range(len(classes)), classes)
    plt.xlabel('Probability (%)')
    plt.title('All Class Probabilities')
    
    plt.tight_layout()
    plt.savefig(f"prediction_{os.path.basename(img_path)}")
    plt.close()

def test_model_on_directory(model_path, test_dir, output_file):
    """Test model on all images in a directory"""
    model = load_model(model_path)
    results = []
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        
        print(f"\nTesting on {class_name} images:")
        for img_name in os.listdir(class_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, img_name)
                prediction = predict_image(model, img_path)
                
                results.append({
                    'image': img_path,
                    'true_class': class_name,
                    'predicted_class': prediction['predicted_class'],
                    'confidence': prediction['confidence'],
                    'top3_predictions': prediction['top3_predictions']
                })
                
                print(f"  {img_name}: {prediction['predicted_class']} ({prediction['confidence']:.2f}%)")
                
                # Visualize prediction
                visualize_prediction(img_path, prediction)
    
    # Write results to file
    with open(output_file, 'w') as f:
        f.write("Image,True Class,Predicted Class,Confidence,Top 3 Predictions\n")
        for result in results:
            top3_str = "; ".join([f"{c}: {p:.2f}%" for c, p in result['top3_predictions']])
            f.write(f"{result['image']},{result['true_class']},{result['predicted_class']},{result['confidence']:.2f},{top3_str}\n")
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test leaf disease detection model')
    parser.add_argument('--model', type=str, default='leaf_disease_model.h5', help='Path to the model file')
    parser.add_argument('--test_dir', type=str, default='test_images', help='Directory containing test images')
    parser.add_argument('--output', type=str, default='test_results.csv', help='Output file for test results')
    
    args = parser.parse_args()
    
    test_model_on_directory(args.model, args.test_dir, args.output) 