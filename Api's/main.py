from operator import imod
from typing import Any

import uvicorn
import numpy as np
from io import BytesIO
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import os
from fastapi.middleware.cors import CORSMiddleware

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = FastAPI()


origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = tf.keras.models.load_model("../Our_models/2")
#beta_model = tf.keras.models.load_model("../Our_models/2")



class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy']

@app.get("/ping")  #entering point of api
async def ping():
    return ("hello")

def read_file_as_image(data) -> np.ndarray: #data take bytes as input
  image_as_np_array = np.array(Image.open(BytesIO(data)))
  return image_as_np_array

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image_as_np_array = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image_as_np_array,  0)

    prediction = model.predict(image_batch)

    predicted_class = class_names[np.argmax(prediction[0])]
    confidence= np.max(prediction[0])     #confidence of model
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }
if __name__ == "__main__":
    uvicorn.run(app, host= 'localhost', port=8000)



