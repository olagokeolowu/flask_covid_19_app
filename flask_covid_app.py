import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from flask import Flask, render_template, request, url_for

# Create the flask app
app = Flask(__name__)

# load the trained model
model = load_model("covid.hdf5")

def predict_image(image):
    # preprocess the image
    img = np.array(image)
    img = tf.image.resize(img, (64, 64))
    # normalize the image
    img = img / 255.0
    img = np.expand_dims(img, axis = 0)

    

    # make predictions
    prediction = model.predict(img)

    # Define class labels
    labels = ['Covid', 'Viral Pneumonia', 'Normal']

  
    # Get the predicted label
    predicted_label_index = np.argmax(prediction)
    predicted_label = labels[predicted_label_index]

    return predicted_label

@app.route("/predict/", methods = ["GET", "POST"])
def predict():
    image = None
    predicted_label = None

    if request.method == "POST":
        uploaded_file = request.files.get("file")

        # check if the file is uploaded
        if uploaded_file is not None:

            # display the image
            image = Image.open(uploaded_file)
            image = image.convert("RGB")
            image.save("static/uploaded_image.png")
            predicted_label = predict_image(image)

    return render_template("index.html", image=url_for("static", filename="uploaded_image.png"), label=predicted_label)
# Define the "result" route for displaying the result
@app.route("/result")
def result():
    return render_template("result.html", predicted_label=predicted_label)

if __name__ == "__main__":
    app.run(debug = True)