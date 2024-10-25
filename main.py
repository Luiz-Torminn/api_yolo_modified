import argparse

import os
import cv2
import time
from pathlib import Path
from werkzeug.utils import send_from_directory
from flask import Flask, render_template, request, Response, redirect, url_for

from ultralytics import YOLO

app = Flask(__name__)
    

def video_inference(video_path: Path, model) -> Path:
    video_path = Path(video_path)
    
    # Define the output path for the processed video
    output_dir = f"{os.getcwd()}/runs/detect"
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    output_path = f"{output_dir}/{video_path.stem}_processed{video_path.suffix}"
    
    # Open the input video
    capture = cv2.VideoCapture(str(video_path))
    
    # Get video properties (width, height, fps)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    
    # Define the codec and create the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))
    
    print(f"Processing video: {video_path}")
    
    while capture.isOpened():
        ret, frame = capture.read()
        
        if not ret:
            break
        
        # Perform YOLO prediction on the frame
        result = model.predict(frame)
        
        # Get the frame with the detections drawn on it
        plotted_result = result[0].plot()
        
        # Write the processed frame to the output video
        video_writer.write(plotted_result)
    
    # Release the video capture and writer objects to finalize the output video
    capture.release()
    video_writer.release()
    
    print(f"\nVideo processing completed. Output saved to: {output_path}")
    
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
    return send_from_directory(directory="runs/detect", filename=filename)

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