from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import uvicorn
import asyncio

app = FastAPI()

# Load your trained model
model = load_model('ct1.h5')  # Replace with your model file path

# Define class names
class_names = {
    0: "Benign",
    1: "Malignant",
    2: "Normal"
}

img_size = 256  # Ensure this matches your model's input size

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read and decode the image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # Preprocess the image (resize, normalize, etc.) as done in your training
        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0  
        img = img.reshape(1, img_size, img_size, 1)

        # Make a prediction
        prediction = model.predict(img)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names.get(predicted_class_index, "Unknown")

        return JSONResponse(content={"predicted_class": predicted_class_name})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
 
