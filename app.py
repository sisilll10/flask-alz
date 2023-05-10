from flask import Flask, render_template, request

from keras.models import load_model

from tensorflow.keras.preprocessing import image
import numpy as np

from PIL import Image

app = Flask(__name__)



@app.route("/", methods=['GET'])
def hello_world():
    return render_template('index.html')



@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "static/images/temp.png"
    imagefile.save(image_path)

    model = load_model('./model_alz.h5')

    def alzPrediction(path, _model):
        classes_dir = ['MildDementia', 'ModerateDementia', 'NonDementia', 'VeryMildDementia']
        # Loading Image
        img = image.load_img(path, target_size=(224,224))
        # Normalizing Image
        norm_img = image.img_to_array(img)/255
        # Converting Image to Numpy Array
        input_arr_img = np.array([norm_img])
        # Getting Predictions
        pred = np.argmax(_model.predict(input_arr_img))
        # Printing Model Prediction
        pred_score = _model.predict(input_arr_img)

        # Normalisasi skor
        total_skor = sum(pred_score[0])
        skor_normal = [s/total_skor for s in pred_score[0]]

        # Konversi skor ke persentase
        skor_persen = [round(s * 100, 2) for s in skor_normal]
        nilai_terbesar = max(skor_persen)


        print(classes_dir)
        print(pred_score * 100)
        print(classes_dir[pred])
        

        return [classes_dir, nilai_terbesar, classes_dir[pred]]
    
    path = image_path
    result = alzPrediction(path,model)

    return render_template('index.html', label = result[0], skor = result[1], prediction = result[2]);


if __name__ == "__main__":
    app.run(debug=True)

