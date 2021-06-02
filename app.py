from PIL import Image
from flask import Flask, render_template, request
import numpy as np
import cv2
import os
import base64
from TwoClassClassifier import predict

from efficientnet_pytorch import EfficientNet


app = Flask(__name__)
app.config['TESTING'] = True


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            print('No selected file')
            return render_template('wrong_image.html')
        else:
            #read image file string data
            filestr = file.read()
            #convert string data to numpy array
            npimg = np.fromstring(filestr, np.uint8)
            # convert numpy array to image

            # img = cv2.imdecode(npimg,cv2.IMREAD_COLOR) # Using cv2 library
            img = Image.fromarray(npimg).convert('RGB') # Using PIL library
            probabilities = 0

            name, probabilities = predict(model_name='efficientnet_b2_pruned', weights_path="model_step1.pth", image = img)
            cv2.imwrite('static/leaf_photo/leaf_image.png',img)
            #base64 encoding for displaying image on webpage
            retval, buffer_img= cv2.imencode('.jpg', img)
            data = base64.b64encode(buffer_img).decode('utf-8')
            return render_template('result.html', imagebase64=data, name=name, probabilities=probabilities)
            

if __name__ == '__main__':
    app.run(debug=False, port=os.getenv('PORT',5000))
