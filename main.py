from fastapi import FastAPI, UploadFile, File
from app import predict
from fastapi.responses import HTMLResponse 
import os
import shutil

app = FastAPI()

@app.get('/', response_class=HTMLResponse)
async def root():
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.post('/predict')
async def make_prediction(photo: UploadFile = File(...)):
    with open(os.path.join("prediction_data", photo.filename), "wb") as buffer:
        shutil.copyfileobj(photo.file, buffer)
    
    result = await predict(photo.filename)

    return {"prediction": f"{result}"}

