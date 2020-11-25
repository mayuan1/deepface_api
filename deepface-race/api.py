from flask import Flask, jsonify, request, make_response

import argparse
import uuid
import json
import time
from tqdm import tqdm

import tensorflow as tf
import numpy as np

# from deepface import DeepFace
# from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace, DeepID
# from deepface.basemodels.DlibResNet import DlibResNet

#from src.Race import loadModel
import src.Race as Race
import src.Age as Age
import src.Emotion as Emotion
#from src.Emotion import loadModel
from src.commons.functions import preprocess_face

app = Flask(__name__)

race_model = Race.loadModel()
race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']

age_model = Age.loadModel()

emotion_model = Emotion.loadModel()
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

@app.route('/analyze/<name>', methods=['POST'])
def analyze(name):

    tic = time.time()
    req = request.get_json()
    trx_id = uuid.uuid4()

    resp_obj = jsonify({'success': False})

    if 'image' in req:
        base64_img = req['image']
        if 'image_type' in req:
            image_type = req['image_type']
        else:
            image_type = 'base64'
    else:
        return jsonify({'success': False, 'error': 'you must pass an img object in the request'}), 205
    
    print("Analyzing image...")

    try:
        if name == "race":
            resp_obj = predict_race(base64_img, image_type)
        elif name == "age":
            resp_obj = predict_age(base64_img, image_type)
        elif name == "emotion":
            resp_obj = predict_emotion(base64_img, image_type)

        toc = time.time()

        resp_obj["trx_id"] = trx_id
        resp_obj["seconds"] = toc-tic

        return resp_obj, 200

    except Exception as ex:
        print(ex)
        return {'error': str(ex)}, 200


def predict_race(img, img_type, race_probs = 0):
    img_224 = preprocess_face(img = img, img_type = img_type, target_size = (224, 224), grayscale = False, 
        enforce_detection = True, detector_backend = 'opencv') #just emotion model expects grayscale images
    print("img_224 finish!")
    race_predictions = race_model.predict(img_224)[0,:]
    print("prediction finish!")
    
    sum_of_predictions = race_predictions.sum()

    if race_probs == 0:
        resp_obj = dict.fromkeys(race_labels, False)
        resp_obj[race_labels[np.argmax(race_predictions)]] = True
    
    else:

        resp_obj = "{"
        for i in range(0, len(race_labels)):
            race_label = race_labels[i]
            race_prediction = 100 * race_predictions[i] / sum_of_predictions

            if i > 0: resp_obj += ", "

            resp_obj += "\"%s\": %s" % (race_label, race_prediction)
        resp_obj += "}"

        resp_obj = json.loads(resp_obj)

    return resp_obj


def predict_age(img, img_type, age_probs = 0):
    img_224 = preprocess_face(img = img, img_type = img_type, target_size = (224, 224), grayscale = False, 
        enforce_detection = True, detector_backend = 'opencv') #just emotion model expects grayscale images
    print("img_224 finish!")
    age_predictions = age_model.predict(img_224)[0,:]
    print("prediction finish!")
    
    apparent_age = Age.findApparentAge(age_predictions)

    resp_obj = {str(apparent_age): True}

    #resp_obj = json.loads(resp_obj)

    return resp_obj

def predict_emotion(img, img_type, emotion_probs = 0):
    img_48 = preprocess_face(img=img, img_type = img_type, target_size=(48, 48),
                          grayscale=True, enforce_detection=True, detector_backend='opencv')
 #just emotion model expects grayscale images
    print("img_48 finish!")
    emotion_predictions = emotion_model.predict(img_48)[0,:]
    print("prediction finish!")
    
    sum_of_predictions = emotion_predictions.sum()

    if emotion_probs == 0:
        resp_obj = dict.fromkeys(emotion_labels, False)
        resp_obj[emotion_labels[np.argmax(emotion_predictions)]] = True
    
    else:

        resp_obj = "{"
        for i in range(0, len(emotion_labels)):
            emotion_label = emotion_labels[i]
            emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions

            if i > 0: resp_obj += ", "

            resp_obj += "\"%s\": %s" % (emotion_label, emotion_prediction)
        resp_obj += "}"

    resp_obj = json.loads(resp_obj)

    return resp_obj


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'-p', '--port',
		type=int,
		default=5000,
		help='Port of serving api')
	args = parser.parse_args()
	app.run(host='0.0.0.0', port=args.port)