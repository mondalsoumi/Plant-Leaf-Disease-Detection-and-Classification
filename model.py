import os
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from werkzeug.utils import secure_filename

# --------------------- CONFIG ---------------------
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'leaf_disease_model.h5'
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
]  # Updated with more disease types

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------------------- MODEL ---------------------
def create_and_train_model():
    print("Training model...")
    train_dir = 'dataset/train'
    val_dir = 'dataset/val'

    img_height, img_width = 224, 224  # Increased image size for better feature detection
    batch_size = 32
    epochs = 20  # Increased epochs for better training

    # Enhanced data augmentation for better generalization
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1.0/255)

    train_data = train_datagen.flow_from_directory(
        train_dir, 
        target_size=(img_height, img_width), 
        batch_size=batch_size, 
        class_mode='categorical'
    )
    val_data = val_datagen.flow_from_directory(
        val_dir, 
        target_size=(img_height, img_width), 
        batch_size=batch_size, 
        class_mode='categorical'
    )

    # Enhanced model architecture
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(train_data.num_classes, activation='softmax')
    ])

    # Compile with better learning rate
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train with early stopping
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
    ]
    
    model.fit(
        train_data, 
        validation_data=val_data, 
        epochs=epochs,
        callbacks=callbacks
    )
    model.save(MODEL_PATH)
    print("Model saved to", MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    create_and_train_model()

model = load_model(MODEL_PATH)

# --------------------- UTILS ---------------------
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


# --------------------- API ENDPOINTS ---------------------
@app.route('/')
def index():
    return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Leaf Disease Detector</title>
            <style>
                body { font-family: Arial; text-align: center; padding: 50px; }
                input[type="file"] { margin: 20px; }
            </style>
        </head>
        <body>
            <h1>Leaf Disease Detection</h1>
            <input type="file" id="imageInput">
            <button onclick="uploadImage()">Detect Disease</button>
            <p id="result"></p>

            <script>
                async function uploadImage() {
                    const input = document.getElementById('imageInput');
                    if (input.files.length === 0) {
                        alert("Please upload a leaf image.");
                        return;
                    }

                    const formData = new FormData();
                    formData.append('image', input.files[0]);

                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();
                    document.getElementById('result').textContent = `Prediction: ${result.prediction}`;
                }
            </script>
        </body>
        </html>
    """)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(img_path)

    processed = preprocess_image(img_path)
    prediction = model.predict(processed)[0]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    return jsonify({'prediction': predicted_class})

# --------------------- RUN ---------------------
if __name__ == '__main__':
    app.run(debug=True)
