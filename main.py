import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image

import numpy as np
import cv2
import tensorflow as tf 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#https://drive.google.com/file/d/16nbIQg7pDVUL1njBY_56-X5E1_YxsRnc/view?usp=sharing
import subprocess 

# モデルをダウンロードするコマンドを実行する
command = '''
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=16nbIQg7pDVUL1njBY_56-X5E1_YxsRnc" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)" 
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=16nbIQg7pDVUL1njBY_56-X5E1_YxsRnc" -o Unet1_12.h5
'''    
subprocess.run(command, shell=True)
#classes = ["0","1","2","3","4","5","6","7","8","9"]
image_size = 255

UPLOAD_FOLDER = "./static/uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

unet = load_model('./Unet1_12.h5')#学習済みモデルをロード


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            #受け取った画像を読み込み、np形式に変換
            #img = image.load_img(filepath, grayscale=True, target_size=(image_size,image_size))

            I = cv2.imread(filepath)/255.0
            

            input_mask = tf.image.resize(I, (224, 224), method='nearest')
            input_mask = tf.expand_dims(
                                        input_mask,
                                        axis = 0,)
            pred_mask = unet.predict(input_mask)

            def create_mask(pred_mask):
                g = list(range(224))
                pred_mask = tf.argmax(pred_mask, axis=-1)
                pred_mask = pred_mask[..., tf.newaxis]
                return pred_mask[0]

            def display(display_list):
                plt.figure(figsize=(15, 15))

                title = ['Input Image', 'True Mask', 'Predicted Mask']

                for i in range(len(display_list)):
                    plt.subplot(1, len(display_list), i+1)
                    plt.title(title[i])
                    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
                    plt.axis('off')
                plt.savefig("./static/mask_img/mask.jpeg")

            
            display([create_mask(pred_mask)])
            #img = image.img_to_array(img)
            #data = np.array([img])
            #変換したデータをモデルに渡して予測する
            

            return render_template("index.html",filepath1="./static/mask_img/mask.jpeg",filepath2=filepath)

    return render_template("index.html",filepath1="",filepath2="")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)