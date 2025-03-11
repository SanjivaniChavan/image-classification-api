import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(
    filename="app.log",        # Log file
    level=logging.INFO,        # Log level
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load the trained model
try:
    model = load_model("model/image_classifier.keras")
    logging.info("✅ Model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load model: {e}")
    raise RuntimeError("Model loading failed. Please check the model path or file format.")

# Class labels for prediction output
CLASS_LABELS = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                 "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Check file type
        if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload a .png, .jpg, or .jpeg file.")

        # Read and preprocess the image
        contents = await file.read()
        img = image.load_img(io.BytesIO(contents), target_size=(32, 32))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_label = CLASS_LABELS[np.argmax(prediction)]

        # Log successful prediction
        logging.info(f"✅ Prediction successful: {predicted_label}")

        return {"prediction": predicted_label}

    except HTTPException as e:
        logging.warning(f"⚠️ {e.detail}")
        raise e

    except Exception as e:
        logging.error(f"❌ Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during prediction.")
