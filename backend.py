from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import uvicorn
import cv2

app= FastAPI()
load_model = tf.keras.models.load_model
model_new = load_model('mnist.h5')
@app.get("/")
def index():
    return {"ok": True}

@app.post("/predict")
async def predict(img: UploadFile = File(...)):
    contents=await img.read()
    nparr = np.fromstring(contents, np.uint8)
    image_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_grey = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_grey, (28, 28))
    img_normalized = img_resized.astype('float32') / 255.0
    pred = model_new.predict(img_normalized.reshape(1, 28, 28))
    return {"result": float(np.argmax(pred[0])), "percent": pred[0].tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)