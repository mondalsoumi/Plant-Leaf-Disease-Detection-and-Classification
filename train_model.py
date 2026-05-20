import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
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

def create_model():
    """Create an improved model architecture"""
    # Use EfficientNetB0 as base model
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])
    
    return model

def create_data_generators():
    """Create data generators with augmentation for training"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.2
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    return train_datagen, val_datagen

def analyze_class_distribution(data_dir):
    """Analyze the distribution of classes in the dataset"""
    class_counts = {}
    for class_name in CLASS_NAMES:
        class_path = os.path.join(data_dir, class_name)
        if os.path.exists(class_path):
            class_counts[class_name] = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        else:
            class_counts[class_name] = 0
    
    # Plot class distribution
    plt.figure(figsize=(12, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Disease Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.close()
    
    return class_counts

def evaluate_model(model, validation_generator):
    """Evaluate model and generate confusion matrix"""
    # Get predictions
    predictions = model.predict(validation_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = validation_generator.classes
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    return cm

def train_model(data_dir):
    """Train the model with improved data handling"""
    # Analyze class distribution
    class_counts = analyze_class_distribution(data_dir)
    print("\nClass Distribution:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} images")
    
    # Calculate class weights to handle imbalance
    total_samples = sum(class_counts.values())
    class_weights = {}
    for class_name, count in class_counts.items():
        if count > 0:
            class_weights[list(CLASS_NAMES).index(class_name)] = total_samples / (len(CLASS_NAMES) * count)
        else:
            class_weights[list(CLASS_NAMES).index(class_name)] = 1.0
    
    print("\nClass Weights:")
    for class_name, weight in class_weights.items():
        print(f"{CLASS_NAMES[class_name]}: {weight:.2f}")
    
    # Create data generators
    train_datagen, val_datagen = create_data_generators()
    
    # Load and preprocess training data
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    # Verify class mapping
    print("\nClass Mapping:")
    for i, class_name in enumerate(train_generator.class_indices.keys()):
        print(f"{i}: {class_name}")
    
    # Create and compile model
    model = create_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train the model with class weights
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # Evaluate model before fine-tuning
    print("\nEvaluating model before fine-tuning:")
    evaluate_model(model, validation_generator)
    
    # Fine-tune the model
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Freeze all layers except the last few
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tune for a few more epochs
    history_fine = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # Evaluate model after fine-tuning
    print("\nEvaluating model after fine-tuning:")
    evaluate_model(model, validation_generator)
    
    # Plot training history
    plot_training_history(history, history_fine)
    
    return model

def plot_training_history(history, history_fine):
    """Plot training and validation accuracy/loss"""
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

if __name__ == '__main__':
    # Specify your dataset directory
    data_dir = 'dataset'  # Update this path to your dataset directory
    model = train_model(data_dir)
    
    # Save the final model
    model.save('leaf_disease_model.h5') 