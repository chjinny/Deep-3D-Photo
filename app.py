# /app.py
#  nohup python3 -u app.py &
from flask import Flask, render_template, request,redirect,url_for
from model_code.model import DepthEstimate
import tensorflow as tf 
import numpy as np
import cv2
from model_code.utils import load_images,predict
import datetime
import tensorflow as tf
tf.__version__

#Flask 객체 인스턴스 생성
app = Flask(__name__)
model=DepthEstimate()
model.load_weights('./model/cp-0010.ckpt')

@app.route('/') # 접속하는 url
def hello():
    return render_template('apply.html')
        
@app.route('/upload_done',methods=('GET', 'POST')) 
def upload_done():
    if request.method == "GET":
        return render_template('render.html', 
    url_img="static/img/img.jpg",
    url_depth="static/img/depth.jpg"
    )

    timestamp = datetime.datetime.today().strftime('%Y%m%d%H%M%S%f')
    uploaded_file=request.files["file"]
    if uploaded_file:
        with tf.device("/gpu:0"):
            uploaded_file.save("static/img/{}_img.jpg".format(timestamp))

            #input_img=cv2.imread("static/img/{}.jpg".format(1))
            input_img=load_images(["static/img/{}_img.jpg".format(timestamp)] )
        
            sero=input_img.shape[1]
            garo=input_img.shape[2]
            
            input_img=input_img.reshape(sero,garo,3)
            input_img=cv2.resize(input_img,(640,480))
            input_img=input_img.reshape(1,480,640,3)
            
            outputs = predict(model, input_img)
            
            outputs=outputs.reshape(240,320,1) #(1,240,320,1)상태에서는 resize가 안되니까
            outputs=cv2.resize(outputs,(garo,sero))
            outputs=255- outputs*255
            outputs = (outputs-np.min(outputs))/np.max(outputs) * 255
            
            cv2.imwrite("static/img/{}_depth.jpg".format(timestamp),outputs)

    return render_template('render.html', 
    url_img="static/img/{}_img.jpg".format(timestamp),
    url_depth="static/img/{}_depth.jpg".format(timestamp)
    )


if __name__=="__main__":
    app.run(host="0.0.0.0", port="5000", debug=False)
    