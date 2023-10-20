import onnxruntime as ort
import socket
import pyttsx3
from retinaface import RetinaFace
from recognition import recognize_image, recognize_video, recognize_cam, close_all, recognize_test




if __name__ == "__main__":
    detector = RetinaFace(quality='high')
    onnx_path = 'model/arcfaceresnet100-11-int8.onnx'
    sess = ort.InferenceSession(onnx_path)
    db_path = 'database\database.db'
    

    # engine = pyttsx3.init()

    # rate = engine.getProperty('rate')
    # engine.setProperty('rate', 125)

    # volume = engine.getProperty('volume')
    # engine.setProperty('volume', 2.0)

    # engine.say("I will speak this text")

    # engine.runAndWait()
    # engine.stop()

    # recognize_cam(detector, sess, db_path)
    recognize_test(detector, sess, db_path)