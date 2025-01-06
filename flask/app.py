from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pandas as pd 
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__, static_folder='./static/', template_folder='./templates/')

model = load_model("D:/GOKUL/project 1/braintumor.keras")

upload_folder = './uploads'
os.makedirs(upload_folder, exist_ok=True)

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']

        filepath = os.path.join(upload_folder, f.filename)
        f.save(filepath)
        image = Image.open(upload_folder+'\\'+f.filename)
        img = image.resize((150, 150))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        preds = model.predict(img_array)
        class_labels = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

        predictions_df = pd.DataFrame(preds,columns=[class_labels])
        predicted_class = class_labels[np.argmax(preds)]
        print(predictions_df)
        print("\nPredicted class: "+ predicted_class)
        
        result = {
            "predicted_class": predicted_class,
            "predictions": preds.tolist()[0]
        }
        print(preds.tolist())
    return jsonify(result)

if __name__ == '__main__':
    app.run()