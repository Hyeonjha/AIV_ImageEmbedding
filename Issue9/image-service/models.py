from flask_sqlalchemy import SQLAlchemy
import uuid

db = SQLAlchemy()

class Image(db.Model):
    id = db.Column(db.String, primary_key=True, default=lambda: str(uuid.uuid4()))
    hash = db.Column(db.String, nullable=False)
    filename = db.Column(db.String, nullable=False)
    size = db.Column(db.Integer, nullable=False)
    path = db.Column(db.String, nullable=False)
