{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "9454df04-16bf-48bf-aad7-1f20a9423b89",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Server is running on http://0.0.0.0:8880\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:     Started server process [13116]\n",
      "INFO:     Waiting for application startup.\n"
     ]
    }
   ],
   "source": [
    "from fastapi import FastAPI, File, UploadFile\n",
    "from fastapi.responses import JSONResponse\n",
    "import numpy as np\n",
    "from tensorflow.keras.models import load_model\n",
    "import cv2\n",
    "import uvicorn\n",
    "import asyncio\n",
    "\n",
    "app = FastAPI()\n",
    "\n",
    "# Load your trained model\n",
    "model = load_model('ct1.h5')  # Replace with your model file path\n",
    "\n",
    "# Define class names\n",
    "class_names = {\n",
    "    0: \"Benign\",\n",
    "    1: \"Malignant\",\n",
    "    2: \"Normal\"\n",
    "}\n",
    "\n",
    "img_size = 256  # Ensure this matches your model's input size\n",
    "\n",
    "@app.post(\"/predict/\")\n",
    "async def predict_image(file: UploadFile = File(...)):\n",
    "    try:\n",
    "        # Read and decode the image\n",
    "        contents = await file.read()\n",
    "        nparr = np.frombuffer(contents, np.uint8)\n",
    "        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)\n",
    "        \n",
    "        # Preprocess the image (resize, normalize, etc.) as done in your training\n",
    "        img = cv2.resize(img, (img_size, img_size))\n",
    "        img = img / 255.0  \n",
    "        img = img.reshape(1, img_size, img_size, 1)\n",
    "\n",
    "        # Make a prediction\n",
    "        prediction = model.predict(img)\n",
    "        predicted_class_index = np.argmax(prediction)\n",
    "        predicted_class_name = class_names.get(predicted_class_index, \"Unknown\")\n",
    "\n",
    "        return JSONResponse(content={\"predicted_class\": predicted_class_name})\n",
    "    except Exception as e:\n",
    "        return JSONResponse(content={\"error\": str(e)}, status_code=500)\n",
    "\n",
    "# Function to run the server\n",
    "async def run_server():\n",
    "    config = uvicorn.Config(app, host=\"0.0.0.0\", port=8880)\n",
    "    server = uvicorn.Server(config)\n",
    "    await server.serve()\n",
    "\n",
    "# Check if running in an interactive environment\n",
    "def is_interactive():\n",
    "    import __main__ as main\n",
    "    return not hasattr(main, '__file__')\n",
    "\n",
    "# Run the server\n",
    "if __name__ == \"__main__\":\n",
    "    if is_interactive():\n",
    "        # For interactive environments (e.g., Jupyter Notebook)\n",
    "        loop = asyncio.get_event_loop()\n",
    "        loop.create_task(run_server())\n",
    "        print(\"Server is running on http://0.0.0.0:8880\")\n",
    "    else:\n",
    "        # For standalone scripts\n",
    "        asyncio.run(run_server())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "83228605-ce0c-4625-b4dd-745bca531f14",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base]",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
