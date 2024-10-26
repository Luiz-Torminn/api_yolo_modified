import argparse

import os
import cv2
from pathlib import Path
from werkzeug.utils import send_from_directory
from flask import Flask, render_template, request, Response, redirect, url_for

from ultralytics import YOLO

app = Flask(__name__)

codecs = ['XVID', 'mp4v', 'avc1']

def video_inference(video_path: Path, model) -> Path:
    video_path = Path(video_path)
    
    output_dir = f"{os.getcwd()}/runs/detect"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{video_path.stem}_processed.mp4"  # Change to .avi
    
    capture = cv2.VideoCapture(str(video_path))
    
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)  # Use float for fps
    
    
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        
        if not video_writer.isOpened():
            print(f"Error: Could not open VideoWriter with output path: {output_path}")
            continue
        
        else: 
            break
    
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        
        result = model.predict(frame)
        plotted_result = result[0].plot()
        video_writer.write(plotted_result)
    
    capture.release()
    video_writer.release()
    
    return output_path

#%%
MODEL_PATH = [f'{os.getcwd()}/model/{m}' for m in os.listdir('model')][0]
print(MODEL_PATH)

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
            filepath = os.path.join(basepath, 'uploads', files.filename)
            
            print("File was uploaded to: ", filepath)
            
            files.save(filepath)
            
            model = YOLO(MODEL_PATH)
            
            if files.filename.endswith('.jpg'):
                img = cv2.imread(filepath)
                detections = model(img, save=True)
                
                return hello_world()
            
            elif files.filename.endswith('.mp4'):
                video_path = filepath
                print(video_path)
                processed_video_path = video_inference(video_path, model)
                video_inference(video_path, model)
                
                print("\nFinished video processing...\n")
                
                return redirect(url_for('download_file', filename=os.path.basename(processed_video_path)))
            
    # folder_path = 'runs/detect'
    # subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    # latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    # image_path = folder_path + '/' + latest_subfolder + '/' + files.filename
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