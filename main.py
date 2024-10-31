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
            files = request.files.getlist('file')  # Use getlist to handle multiple files
            basepath = os.path.dirname(__file__)
            
            # Generate file paths for each uploaded file
            filepaths = [os.path.join(basepath, 'uploads', fp.filename) for fp in files]
            for fp, filepath in zip(files, filepaths):
                fp.save(filepath)  # Save each file to its respective path
            
            model = YOLO(MODEL_PATH)
            
            if any(fp.filename.endswith('.jpg') for fp in files):
                imgs = [cv2.imread(path) for path in filepaths]
                detections = [model(img, save=True) for img in imgs]
                
                # Get the paths to the saved images
                image_paths = [os.path.join('./runs/detect', img) for img in os.listdir('./runs/detect')]
                return redirect(url_for('display_images'))     
            
            elif any(fp.filename.endswith('.mp4') for fp in files):
                video_path = filepaths[0]  # Assuming the first file is a video if it's present
                processed_video_path = video_inference(model, video_path)
                
                print("\nFinished video processing...\n")
                
                return redirect(url_for('download_file', filename=os.path.basename(processed_video_path)))
            
    return render_template('index.html')


# function to display the detected objects video on html page
@app.route("/download/<filename>")
def download_file(filename):
    environ = request.environ
    filename = Path(filename).name

    return send_from_directory(directory="runs/detect", path=filename, environ=environ)

@app.route("/display")
def display_images(image_paths):
    return render_template('display.html', imgs = image_paths)
    

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
