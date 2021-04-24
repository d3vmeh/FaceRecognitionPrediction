from flask import Flask, render_template, request, redirect
import os
import numpy as np
import cv2
from keras.models import load_model

facecascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
app = Flask(__name__)

model = load_model(r"model/faces.h5")

choices = ["Curry", "Brady"]

@app.route("/",methods=["GET","POST"])
def index():
    if request.method == "GET":
        return render_template("index.html",msg="",img="")
    else:
        
        givenimage = request.files["img"].read()
        imagearray = np.fromstring(givenimage,np.uint8)
        imagearray = cv2.imdecode(imagearray,cv2.IMREAD_COLOR)
        imagearraygray = cv2.cvtColor(imagearray,cv2.COLOR_BGR2GRAY)
        faces = facecascade.detectMultiScale(imagearraygray,1.3,5)

        
        
      
        if faces != ():
            for x,y,w,h in faces:
                
                roi = imagearraygray[y:y+h,x:x+w]
            
                roi = cv2.resize(roi,(50,50))


                predictionarray = np.array(roi)
                predictionarray = predictionarray.reshape(1,50,50,1)
                predictions = model.predict(predictionarray)
                
                cv2.rectangle(imagearray, (x,y), (x+w,y+h), (255,0,0))
                
                cv2.putText(imagearray,choices[np.argmax(predictions[0])], (x,y),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
                
                

            
            imagearray = cv2.resize(imagearray,(int(imagearray.shape[1]*1.5),int(imagearray.shape[0]*1.5)))
            cv2.imwrite(r"static/img.png", imagearray)

        else:
            return render_template("index.html",msg="No faces found",img="")

        return render_template("index.html",msg="",img="static/img.png")

if __name__=="__main__":
    app.run()

