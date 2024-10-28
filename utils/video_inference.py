import os
import cv2
from pathlib import Path

class VideoInference():
    def __init__(self):
        self.codecs = ['XVID', 'mp4v', 'avc1']
    
    def __call__(self, model, video_path: Path) -> Path:
        return self.video_inference(model, video_path)

    def video_inference(self, model, video_path: Path) -> Path:
        video_path = Path(video_path)

        output_dir = f"{os.getcwd()}/runs/detect"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/{video_path.stem}_processed.mp4"  # Change to .avi

        capture = cv2.VideoCapture(str(video_path))

        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)  # Use float for fps


        for codec in self.codecs:
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

    