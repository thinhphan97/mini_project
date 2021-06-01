from flask import Flask, render_template, request
import numpy
import cv2
import os
import base64

#LEAF_FOLDER = os.path.join('static', 'leaf_photo')

app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = LEAF_FOLDER
app.config['TESTING'] = True

# from Inference import get_plant_disease, background_removal, object_detection

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
            npimg = numpy.fromstring(filestr, numpy.uint8)
            # convert numpy array to image
            img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
            valid=0
            img, valid = object_detection(image_bytes=img,val=valid)
            #cv2.imwrite('static/leaf_photo/leaf_image.png',img)
            if (valid==1):
                retval, buffer_img= cv2.imencode('.jpg', img)
                data = base64.b64encode(buffer_img).decode('utf-8')
                return render_template('result.html', leaf_image=data)
            else:
                return render_template('wrong_image.html')
            #base64 encoding for displaying image on webpage
            retval, buffer_img= cv2.imencode('.jpg', img)
            data = base64.b64encode(buffer_img).decode('utf-8')
            

if __name__ == '__main__':
    app.run(debug=True,port=os.getenv('PORT',5000))
