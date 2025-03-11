from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import uvicorn
import io
from PIL import Image

app = FastAPI()

# Load the trained model
model = load_model("model/image_classifier.keras")
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("L").resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=[0, -1])

    prediction = np.argmax(model.predict(img_array))
    return {"predicted_class": int(prediction)}
    



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)