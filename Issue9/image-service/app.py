from flask import Flask, request, jsonify
import os
import hashlib
from models import db, Image
from config import Config
from confluent_kafka import Producer
import uuid

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

producer = Producer({'bootstrap.servers': app.config['KAFKA_BROKER']})

UPLOAD_FOLDER = '/mnt/data/images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return 'No file part', 400
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400
        if file:
            filename = file.filename
            file_content = file.read()
            file_hash = hashlib.md5(file_content).hexdigest()
            file_size = len(file_content)
            ext = filename.rsplit('.', 1)[1].lower()
            saved_filename = f"{file_hash}-{file_size}.{ext}"
            file_path = os.path.join(UPLOAD_FOLDER, saved_filename)
            
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            image = Image(id=str(uuid.uuid4()), hash=file_hash, filename=saved_filename, size=file_size, path=file_path)
            db.session.add(image)
            db.session.commit()

            producer.produce('image-uploads', key=image.id, value=image.id)
            producer.flush()

            return jsonify({'id': image.id}), 201
    except Exception as e:
        print(f"Error: {e}", flush=True)
        db.session.rollback()  # 오류가 발생하면 트랜잭션 롤백
        return 'Internal Server Error', 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
