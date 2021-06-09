from PIL import Image
from flask import Flask, render_template, request
import numpy as np
import base64
from TwoClassClassifier import predict, img_to_base64_str
import base64
from io import BytesIO
import cv2
from yolov5.utils.general import convert_from_image_to_cv2
from detect import detect

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
            npimg = np.frombuffer(filestr, np.uint8) #change fromstring --> frombuffer
            # convert numpy array to image
            opencvImage = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
            img = Image.fromarray(npimg).convert('RGB') # Using PIL library because input of EffNet need a PIL format
            name, probabilities = predict(model_name='efficientnet_b2_pruned', weights_path="model_step1.pth", image = img)

            if name == 'normal' :
                #base64 encoding for displaying image on webpage
                print('name = ', name, '\t', 'probabilities = ', probabilities)
                retval, buffer_img= cv2.imencode('.png', opencvImage)

                data = base64.b64encode(buffer_img).decode('utf-8')
                return render_template('result.html', imagebase64=data, name=name, probability=probabilities)
            else :
                image_pred, names = detect(opencvImage, weights="best.pt", img_size=640, augment=True)
                #base64 encoding for displaying image on webpage
                for i in names:
                    print("-->", i)
                retval, buffer_img= cv2.imencode('.png', image_pred)

                data = base64.b64encode(buffer_img).decode('utf-8')
                return render_template('result.html', imagebase64=data, name=name, probability=probabilities)
            
            

if __name__ == '__main__':
    app.run(debug=True, port=5000)