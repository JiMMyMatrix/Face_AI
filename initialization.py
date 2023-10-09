import os
import numpy as np
import sqlite3
import onnxruntime as ort
from retinaface import RetinaFace
from database import create_db, adapt_array, convert_array

detector = RetinaFace(quality='normal')

onnx_path = 'model/arcfaceresnet100-11-int8.onnx'
sess = ort.InferenceSession(onnx_path)

# create db
db_path = 'database/database.db'
file_path = 'Figure'
sqlite3.register_adapter(np.array, adapt_array)
sqlite3.register_converter("ARRAY", convert_array)

if not os.path.exists(db_path):
    create_db(db_path, file_path, detector, sess)