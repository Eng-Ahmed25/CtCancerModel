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

# Function to run the server
async def run_server():
    config = uvicorn.Config(app, host="0.0.0.0", port=8880)
    server = uvicorn.Server(config)
    await server.serve()

# Check if running in an interactive environment
def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

# Run the server
if __name__ == "__main__":
    if is_interactive():
        # For interactive environments (e.g., Jupyter Notebook)
        loop = asyncio.get_event_loop()
        loop.create_task(run_server())
        print("Server is running on http://0.0.0.0:8880")
    else:
        # For standalone scripts
        asyncio.run(run_server())