from tensorflow.keras.models import load_model
from keras.saving import save_model
import os

# Load your existing .h5 model
model = load_model("leaf_disease_model.h5")

# Make sure target folder exists
os.makedirs("model", exist_ok=True)

# Save the model in .keras format
save_model(model, "model/leaf_model.keras")
