from PIL import Image
from flask import Flask, render_template, request
import numpy as np
import base64
from TwoClassClassifier import predict, img_to_base64_str
import base64
from io import BytesIO


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
            print('Image have been uploaded!')
            name, probabilities = predict(model_name='efficientnet_b2_pruned', weights_path="model_step1.pth", image = img)

            #base64 encoding for displaying image on webpage
            
            # img = img_to_base64_str(img)
            print('name = ', name, '\t', 'probabilities = ', probabilities)
            # retval, buffer_img= cv2.imencode('.jpg', img)
            # data = base64.b64encode(img).decode('utf-8')
            return render_template('result.html', imagebase64=img, name=name, probability=probabilities)
            

if __name__ == '__main__':
    app.run(debug=True, port=5000)
