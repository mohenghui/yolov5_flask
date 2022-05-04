from importlib import import_module
import os
from flask_script import Manager
from flask import Flask, render_template, Response, request, redirect, url_for

# import camera driver
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from camera import Camera

app = Flask(__name__)

manager = Manager(app)
NAME = ""
FILE_FLAG = False


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_start')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    if FILE_FLAG:
        return Response(gen(Camera(NAME, True if NAME.endswith(('.jpg', '.png', '.jpeg', '.bmp')) else False)),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(mimetype='multipart/x-mixed-replace; boundary=frame')
        # pass


@app.route('/', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path='static\\uploads'
        if not os.path.exists(upload_path):os.mkdir(upload_path)
        upload_file_path = os.path.join(basepath, upload_path, (f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_file_path)
        global NAME, FILE_FLAG
        NAME = upload_file_path
        FILE_FLAG = True
    return redirect(url_for('index'))


@manager.command
def dev():
    from livereload import Server
    live_server = Server(app.wsgi_app)
    live_server.watch("**/*.*")
    live_server.serve(open_url=True)


if __name__ == '__main__':
    app.run(host='127.0.0.1', threaded=True, port=5001)
