import os 
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import pickle as pkl
from werkzeug.utils import secure_filename
from keras import backend as K

graph = tf.get_default_graph()
K.set_image_dim_ordering('th')
app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

validation_list = ['visibile angle of car in the image is damaged', 'visibile angle of car in the image is not damaged']
location_list = ['front', 'rear', 'sides']
severity_list = ['minor', 'moderate', 'severe']


with open('car_damage_validation.pickle', 'rb') as dam:
    model_validation = pkl.load(dam)
with open('car_damage_location.pickle', 'rb') as loc:
    model_location = pkl.load(loc)
with open('car_damage_severity.pickle', 'rb') as sev:
    model_severity = pkl.load(sev)

def preprocess(file, size):
    x = x = load_img(file, target_size = size)
    x = img_to_array(x)
    x = x.reshape((1,) + x.shape)/255
    return x
    
def predict(file):
    x1 = preprocess(file, size = (256, 256))
    x2 = preprocess(file, size = (224, 224))
    with graph.as_default():
        pred1 = model_validation.predict_classes(x1)
    with graph.as_default():
        pred2 = model_location.predict_classes(x2)
    with graph.as_default():
        pred3 = model_severity.predict_classes(x2)
    print(pred1, pred2, pred3)
    pred1 = pred1[0][0]
    pred2 = pred2[0]
    pred3 = pred3[0]
    validation = validation_list[pred1]
    location = location_list[pred2]
    severity = severity_list[pred3]
    if validation == validation_list[1]:
        list = ['visibile angle of car in the image is not damaged']
    else:
        list = [validation, location, severity]
    return list

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
    target = os.path.join(APP_ROOT, 'static/')
    if not os.path.isdir(target):
        os.mkdir(target)
    filename = file.filename
    destination = "/".join([target, filename])
    file.save(destination)
    label = predict(file)
    if len(label) == 1:
        label1 = label[0]
        label2 = 'NA'
        label3 = 'NA'
    else:
        label1 = label[0]
        label2 = label[1]
        label3 = label[2]
    
    return render_template("template.html", label1 = label1, label2 = label2, label3 = label3, image_name = filename)
    
if __name__ == "__main__":
    app.run(debug = True)	    
    
    
    
    