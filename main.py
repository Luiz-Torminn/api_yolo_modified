import argparse

from utils.video_inference import VideoInference

import os
import cv2
from werkzeug.utils import send_from_directory
from flask import Flask, render_template, request, Response, redirect, url_for

from ultralytics import YOLO

app = Flask(__name__)


#%%
MODEL_PATH = [f'{os.getcwd()}/model/{m}' for m in os.listdir('model')][0]
video_inference = VideoInference()


#%%
@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            files = request.files['file']
            basepath = os.path.dirname(__file__)

            if len(files) > 1:
                filepaths = [os.path.join(basepath, 'uploads', fp) for fp.filename in files]
                
            else: 
                filepath = os.path.join(basepath, 'uploads', files.filename)
            
            print("File was uploaded to: ", filepath)
            
            files.save(filepath)
            
            model = YOLO(MODEL_PATH)
            
            if files.filename.endswith('.jpg'):
                img = cv2.imread(filepath) if len(files) == 1 else [cv2. for path in filepaths cv2.imread(path)]
                detections = model(img, save=True) # FINISH THE MULTIPLE IMAGE-FILES LOGIC...
                
                return redirect(url_for('download_file', filename=os.path.basename(filepath)))
            
            elif files.filename.endswith('.mp4'):
                video_path = filepath
                print(video_path)
                processed_video_path = video_inference(model, video_path)
                
                print("\nFinished video processing...\n")
                
                return redirect(url_for('download_file', filename=os.path.basename(processed_video_path)))
            
    return render_template('index.html')


# function to display the detected objects video on html page
@app.route("/download/<filename>")
def download_file(filename):
    environ = request.environ
    return send_from_directory(directory="runs/detect", path=filename, environ=environ)

@app.route("/video_feed")
def video_feed():
    print("function called")
    
    return render_template('index.html')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov9 models")
    parser.add_argument("--port", default=9000, type=int, help="port number")
    args = parser.parse_args()
    model = YOLO(MODEL_PATH)
    app.run(host="0.0.0.0", port=args.port) 
