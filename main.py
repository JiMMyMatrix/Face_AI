import onnxruntime as ort
import socket
from retinaface import RetinaFace
from recognition import recognize_image, recognize_video, recognize_cam, close_all


if __name__ == "__main__":
    detector = RetinaFace(quality='normal')
    onnx_path = 'model/arcfaceresnet100-11-int8.onnx'
    sess = ort.InferenceSession(onnx_path)
    db_path = 'database\database.db'


    recognize_cam(detector, sess, db_path)
    # recognize_video(detector, sess, db_path)
    # close_all()