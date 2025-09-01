from flask import Flask, render_template, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import io
from PIL import Image

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = "secret_key"


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

@app.route('/')
def index():
    return render_template('index.html')

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