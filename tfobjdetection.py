from imageai.Detection import ObjectDetection
import os
from twilio.twiml.messaging_response import MessagingResponse
from flask import Flask, request, redirect, send_from_directory
import requests

curr_dir = os.getcwd()
print(curr_dir)
app = Flask(__name__)
@app.route("/sms", methods=['GET', 'POST'])
def sms():
    resp = MessagingResponse()
    
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join(
        curr_dir, "resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()
    if request.values['NumMedia'] != '0':
        filename = request.values['MessageSid'] + '.jpg'
        respStr = ''
        with open(filename, 'wb') as f:
            image_url = request.values['MediaUrl0']
            f.write(requests.get(image_url).content)
            detections = detector.detectObjectsFromImage(input_image=filename, output_image_path= filename)
            for eachObject in detections:
                perc = eachObject["percentage_probability"] 
                respStr += (str(eachObject["name"] +
                                " : ") + str(perc) + "%\n")
                print(eachObject["name"], " : ", eachObject["percentage_probability"], "%")
        msg = resp.message(respStr)
        msg.media('https://lizzie.ngrok.io/output/{}'.format(filename))
    else:
        resp.message("Try sending a picture message.")
    return str(resp)


@app.route('/output/<filename>', methods=['GET', 'POST'])
def uploaded_file(filename):
    return send_from_directory(curr_dir, filename)

if __name__ == "__main__":
    app.run(debug=True)
