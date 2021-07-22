import os

import argparse
from flask import Flask, render_template, Response

from src.video_dense_checker import VideoDenseChecker
from src.counter import Counter
from src.calc_fps import FPSCalculator

def parse_args():
    parser = argparse.ArgumentParser(description='Dense Check Parameters')
    
    parser.add_argument('--model', default='vgg', type=str, help='set the model architecture')
    parser.add_argument('--data-dir', default='src/data', type=str, help='set the directory where weights and movies put')
    parser.add_argument('--weight-path', default='ucf_vgg_best_model.pth', type=str, help='set the model architecture')

    parser.add_argument('--particle-num', default=1000, type=int, help='set the number of particle')

    parser.add_argument('--use-movie', action='store_false', help='if use an existing movie, set this option')
    parser.add_argument('--media-path', default='shinjuku1.mp4', type=str, help='if use an existing movie, set the file name of movie')
    
    args = parser.parse_args()
    return args

app = Flask(__name__)
PORT = 8000
args = parse_args()

@app.route("/")
def index():
    return render_template("index.html")    

def gen(checker):
    while True:
        frame = checker.get_frame()
        if frame is not None:
            yield (b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame.tobytes() + b"\r\n")
        else:
            print("frame is none")

@app.route("/video_feed")
def video_feed():
    fps_calculator = FPSCalculator()
    counter = Counter(model=args.model, model_path=os.path.join(args.data_dir, args.weight_path))
    checker = VideoDenseChecker(counter, fps_calculator, use_camera=args.use_movie, media_path=os.path.join(args.data_dir, args.media_path))

    return Response(gen(checker),
            mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=PORT, threaded=True)