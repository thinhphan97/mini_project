from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ValidationError, validator
from detect import detect
from PIL import Image
import cv2
import numpy as np
from TwoClassClassifier import predict


import base64

app = FastAPI()

templates = Jinja2Templates(directory='templates')

def results_to_json(name, probabilities, labels = [""], img = ""):
    return {
            "name" : name,
            "probabilities": probabilities,
            "image" : img,
            "labels" : labels
            }
        

@app.get("/")
def home(request: Request):
    """
    Return html template render for home page form
    """
    return templates.TemplateResponse('index.html', {"request": request})

class YOLORequest(BaseModel):
    """
    Class used for pypandic validation
    """
    img_size: int

    @validator('img_size')
    def validate_img_size(cls, v):
        assert v%32 == 0 and v>0, f'Invalid inference size. Must be multiple of 32 and greater than 0.'
        return v

@app.post("/")
async def detect_from_web(request: Request,
                file: UploadFile = File(...),
                img_size: int = Form(640)
                ):
    """
    Requires an image file upload and Optional image size parameter.
    Intended for human (non-api) users.
    Return: HTML template render showing bbox data and base64 encode image.
    """

    try:
        yr = YOLORequest(img_size=img_size)
    except ValidationError as e:
        return JSONResponse(content=e.errors(), status_code=422)
    
    filestr = await file.read()
    #convert string data to numpy array
    npimg = np.frombuffer(filestr, np.uint8) #change fromstring --> frombuffer
    # convert numpy array to image
    opencvImage = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    img = Image.fromarray(npimg).convert('RGB')
    name, probabilities = predict(model_name='efficientnet_b2_pruned',
                                     weights_path="model_step1.pth", image = img)
    probabilities = float(probabilities[0][0])
    probabilities = format(probabilities, '.2f')

    if name == 'normal' :
        #base64 encoding for displaying image on webpage
        print('name = ', name, '\t', 'probabilities = ', probabilities)
        retval, buffer_img= cv2.imencode('.png', opencvImage)

        data = base64.b64encode(buffer_img).decode('utf-8')
        json_result = results_to_json(name, probabilities)
        return templates.TemplateResponse('result.html', {"request": request, "imagebase64" : data,
                                           "name" : name, "probability" : probabilities})
    else :
        image_pred, names = detect(opencvImage, weights="best.pt", img_size=640)
        #base64 encoding for displaying image on webpage
        for i in names:
            print("-->", i)
        retval, buffer_img= cv2.imencode('.png', image_pred)
        data = base64.b64encode(buffer_img).decode('utf-8')
        json_result = results_to_json(name, float(probabilities[0][0]), labels=names, img = data)
        return templates.TemplateResponse('result.html', {"request": request, "imagebase64" : data,
                                           "name" : name, "probability" : probabilities})

@app.post("/detect")
async def detect_from_api(request: Request,
                file: UploadFile = File(...),
                img_size: int = Form(640)
                ):
    """
    Requires an image file upload and Optional image size parameter.
    Intended for API users.
    Return: JSON results of running YOLOv5 on the uploaded image.
    """

    try:
        yr = YOLORequest(img_size=img_size)
    except ValidationError as e:
        return JSONResponse(content=e.errors(), status_code=422)
    
    filestr = await file.read()
    #convert string data to numpy array
    npimg = np.frombuffer(filestr, np.uint8) #change fromstring --> frombuffer
    # convert numpy array to image
    opencvImage = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    img = Image.fromarray(npimg).convert('RGB')
    name, probabilities = predict(model_name='efficientnet_b2_pruned',
                                     weights_path="model_step1.pth", image = img)
    probabilities = float(probabilities[0][0])
    probabilities = format(probabilities, '.2f')
    if name == 'normal' :
        #base64 encoding for displaying image on webpage
        print('name = ', name, '\t', 'probabilities = %.2f ', probabilities)
        retval, buffer_img= cv2.imencode('.png', opencvImage)

        data = base64.b64encode(buffer_img).decode('utf-8')
        json_result = results_to_json(name, probabilities)
        return json_result
    else :
        image_pred, names = detect(opencvImage, weights="best.pt", img_size=640)
        #base64 encoding for displaying image on webpage
        for i in names:
            print("-->", i)
        retval, buffer_img= cv2.imencode('.png', image_pred)
        data = base64.b64encode(buffer_img).decode('utf-8')
        json_result = results_to_json(name, probabilities, labels=names, img = data)
        print(json_result)
        return json_result

if __name__ == '__main__':
	import uvicorn
	app_str = 'server:app'
	uvicorn.run(app_str, host='localhost', port=8000, reload=True, workers=1)