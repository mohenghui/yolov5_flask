from importlib import import_module
import os
import cv2
from flask import Flask, render_template, Response, request, redirect, url_for

if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from camera import Camera


app = Flask(__name__)

NAME = ""
FILE_FLAG = False
CAMERA_FLAG=False

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def video_gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def camera_gen():
    vid = cv2.VideoCapture(1)
    while True:
        return_value, frame = vid.read()
        image = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

@app.route('/video_start')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    if FILE_FLAG:
        return Response(video_gen(Camera(NAME, True if NAME.endswith(('.jpg', '.png', '.jpeg', '.bmp')) else False)),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    elif CAMERA_FLAG:
        return Response(video_gen(Camera("1",False)),#选择你的摄像头ID
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(mimetype='multipart/x-mixed-replace; boundary=frame')
        # pass


@app.route('/video', methods=['POST'])
def upload():
    f = request.files['file']
    basepath = os.path.dirname(__file__)  # 当前文件所在路径
    upload_path = './static/uploads'
    if not os.path.exists(upload_path):
        os.mkdir(upload_path)
    upload_file_path = os.path.join(basepath, upload_path, (f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
    f.save(upload_file_path)
    global NAME, FILE_FLAG,CAMERA_FLAG
    NAME = upload_file_path
    FILE_FLAG = True
    CAMERA_FLAG = False
    return redirect(url_for('index'))

@app.route('/camera', methods=['POST'])
def camera_get():
    global CAMERA_FLAG,FILE_FLAG
    CAMERA_FLAG = True
    FILE_FLAG=False
    # return redirect(url_for('index'))
    return redirect('/')



if __name__ == '__main__':
    app.run(host='127.0.0.1', threaded=True, port=5001)
