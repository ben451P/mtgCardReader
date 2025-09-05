from flask import Flask, render_template, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import io
from PIL import Image
from flask import Flask, render_template, Response
import cv2
import numpy as np
from backend.background_remover import BackgroundRemover

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = "secret_key"

camera = cv2.VideoCapture(0)

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

    stored_cards = db.relationship("Database", backref="user", lazy=True)
    
    def __repr__(self):
        return f'<User {self.username}>'
    
class Database(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    storage = db.Column(db.JSON, nullable=False,default=lambda: {})

    owner = db.Column(db.Integer, db.ForeignKey("user.id"))

with app.app_context():
    db.create_all()


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        remover = BackgroundRemover(pil_img)
        remover.remove_bg_main()
        result = remover.return_result()

        

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        print(type(image))

        print(f"Image loaded: {image.format}, {image.size}, {image.mode}")

        return f"Successfully received image: {image.format}, {image.size}, {image.mode}"
    except Exception as e:
        return f"Error processing image: {e}", 500


if __name__ == '__main__':
    app.run(debug=True)