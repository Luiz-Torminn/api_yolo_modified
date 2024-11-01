from flask import Flask, render_template, request, Response, redirect, url_for, session
import argparse
from utils.video_inference import VideoInference
import os
import shutil
import cv2
from werkzeug.utils import send_from_directory
from ultralytics import YOLO

from pathlib import Path

app = Flask(__name__)
app.secret_key = os.urandom(24)


#%%
MODEL_PATH = [f'{os.getcwd()}/model/{m}' for m in os.listdir('model')][0]
video_inference = VideoInference()


#%%
@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/", methods=["GET", "POST"])
def predict_img():
    if os.path.exists('./runs'):
        shutil.rmtree('./runs')

    if request.method == "POST":
        if 'file' in request.files:
            files = request.files.getlist('file')  # Use getlist to handle multiple files
            basepath = os.path.dirname(__file__)

            # Generate file paths for each uploaded file
            filepaths = [os.path.join(basepath, 'uploads', fp.filename) for fp in files]
            for fp, filepath in zip(files, filepaths):
                fp.save(filepath)  # Save each file to its respective path
            
            model = YOLO(MODEL_PATH)

            if any(fp.filename.endswith(('.jpg', '.png')) for fp in files):

                for filepath in filepaths:
                    img = cv2.imread(filepath)
                    detection = model(img, save=True, name='predict')
                
                # Get paths to the saved images
                # image_paths = [img for img in os.listdir('./runs/detect/predict') if img.endswith(('.jpg', '.png'))]
                image_paths = []
                for dirname, _, imgs in os.walk('./runs/detect'):
                    for img in imgs: 
                        if img.endswith(('jpg', 'png')):
                            relative_path = os.path.relpath(os.path.join(dirname, img), 'runs/detect')
                            image_paths.append(relative_path)

                # Store image paths in session and redirect to display route
                session['image_paths'] = image_paths

                return redirect(url_for('display_images'))     
            
            elif any(fp.filename.endswith('.mp4') for fp in files):
                video_path = filepaths[0]  # Assuming the first file is a video if it's present
                processed_video_path = video_inference(model, video_path)
                
                print("\nFinished video processing...\n")
                
                return redirect(url_for('download_file', filename=os.path.basename(processed_video_path)))
            
    return render_template('index.html')


# function to display the detected objects video on html page
@app.route("/download/<path:filename>")
def download_file(filename):
    environ = request.environ

    if filename.endswith(('.jpg', '.png')):
        return send_from_directory(directory="runs/detect", path=filename, environ=environ)
    else:
        filename = Path(filename).name
        return send_from_directory(directory="runs/detect", path=filename, environ=environ)

@app.route("/display")
def display_images():
    image_paths = session.get('image_paths', [])  # Retrieve image paths from session

    return render_template('display.html', imgs=image_paths)
    

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
