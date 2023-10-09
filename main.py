import onnxruntime as ort
from retinaface import RetinaFace
from recognition import recognize_image, recognize_video

if __name__ == "__main__":
    detector = RetinaFace(quality='normal')
    onnx_path = 'model/arcfaceresnet100-11-int8.onnx'
    sess = ort.InferenceSession(onnx_path)
    db_path = 'database\database.db'

    while(True):
        # recognize_image(detector, sess, db_path)
        recognize_video(detector, sess, db_path)
