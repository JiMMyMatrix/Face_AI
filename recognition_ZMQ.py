from tkinter import filedialog
import cv2
import matplotlib.pyplot as plt
from comparsion import compare_face
from detection import face_detect, face_detect_bgr, face_detect_cam
from feature_extraction import feature_extract
import winsound
from IPython.display import clear_output
import socket
import pickle
import struct ## new
import imagezmq
import traceback
import time
import simplejpeg

def init_TCP_conn():
    HOST='192.168.20.21'
    PORT=10100
    VIDEO_NAME='WebCAM_Ouput.mp4'

    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    print('Socket created')

    s.bind((HOST,PORT))
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')

    conn,addr=s.accept()
    #video record
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_NAME, fourcc, 30.0, (640,  480))
    return s, out, conn

def receive_WebCAM(s, conn):
    data = b""
    payload_size = struct.calcsize(">L")
    print("payload_size: {}".format(payload_size))
    while len(data) < payload_size:
        data += conn.recv(4096)
        if not data:
            cv2.destroyAllWindows()
            conn,addr=s.accept()
            continue
    # receive image row data form client socket
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]
    # unpack image using pickle 
    frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    cv2.imshow('server',frame)
    cv2.waitKey(1)
    return frame

def close_all(err,sock,out):
    print(err)
    out.release()
    sock.close()
    cv2.destroyAllWindows()

def recognize_cam(detector, sess, db_path):
    print("Recognizing camera image...")
    image_hub = imagezmq.ImageHub(open_port='tcp://192.168.3.2:5555',REQ_REP=False)
    frame_count = 1
    time1 = 0
    # s,out,conn=init_TCP_conn()
    # cam = cv2.VideoCapture(0)
    # cam.set(3,640)
    # cam.set(4,480)

    while True:
        try:
            time1 = time.time() if frame_count == 1 else time1
            name, image = image_hub.recv_jpg()
            # 解码
            image = simplejpeg.decode_jpeg(image, colorspace='BGR')
            # out.write(frame)
            # cv2.imshow('server',frame)
            # cv2.waitKey(1)

            # ret, frame = cam.read()
            # if len(frame)>0:
            img_rgb, detections = face_detect_cam(image, detector)
            position, landmarks, embeddings = feature_extract(img_rgb, detections, sess)
            threshold = 1

            img_out = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

            for i, embedding in enumerate(embeddings):
                name, distance, total_result = compare_face(embedding, threshold, db_path)
                
                # Draw a rectangle around the recognized face and put the name of the person
                cv2.rectangle(img_out, (position[i][0], position[i][1]), (position[i][2], position[i][3]), (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img_out, name + ', ' + str(distance), (position[i][0] + 10, position[i][1] - 10), font, 0.8,
                            (0, 255, 0), 2)

                print(total_result)

                cv2.imshow('Image', img_out)
            # else:
            #     break

            k = cv2.waitKey(10) & 0xff
            if k == 27:
                image_hub.close() 
                cv2.destroyAllWindows()
                # close_all('', s, out)
                print('Closing the Socket and video...')
                break
        except Exception as err:
            print("Error msg: ",err)
            print('Closing the Socket and video...')
            image_hub.close() 
            cv2.destroyAllWindows()
            # close_all('', s, out)
            break
    print('End while loop!!')

    
    


def recognize_image(detector, sess, db_path):
    """
    Recognize faces in an image selected by the user using a file dialog.

    :return: None
    """

    print('Recognizing image...')

    # Open a file dialog to select the image file
    img_path = filedialog.askopenfilename()

    # Detect faces in the image
    img_rgb, detections = face_detect(img_path, detector)

    # Extract features from the detected faces
    position, landmarks, embeddings = feature_extract(img_rgb, detections, sess)

    # Set a threshold for face recognition
    threshold = 1

    # Compare the embeddings of the detected faces with the faces in the database
    for i, embedding in enumerate(embeddings):
        name, distance, total_result = compare_face(embedding, threshold, db_path)

        # Draw a rectangle around the recognized face and put the name of the person
        cv2.rectangle(img_rgb, (position[i][0], position[i][1]), (position[i][2], position[i][3]), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_rgb, name + ', ' + str(distance), (position[i][0] + 10, position[i][1] - 10), font, 0.8,
                    (0, 255, 0), 2)

        print(total_result)

    # Display the image with recognized faces
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb / 255)
    _ = plt.axis('off')
    plt.show()


def recognize_video(detector, sess, db_path):
    """
    Recognize faces in a video selected by the user using a file dialog.

    :return: None
    """

    print('Recognizing video...')

    # Open a file dialog to select the video file
    video_path = filedialog.askopenfilename()

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frames per second of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    found = False
    count = 0

    while True and not found:  # read every frame to play video
        ret, frame = cap.read()  # return two values(boolean, next frame)
        if ret:
            # Detect faces in the frame
            img_rgb, detections = face_detect_bgr(frame, detector)

            if count == fps // 3:
                count = 0

                # Extract features from the detected faces
                position, landmarks, embeddings = feature_extract(img_rgb, detections, sess)

                threshold = 1
                for i, embedding in enumerate(embeddings):
                    name, distance, total_result = compare_face(embedding, threshold, db_path)
                    if distance < threshold:
                        # Draw a rectangle around the recognized face and put the name of the person
                        cv2.rectangle(img_rgb, (position[i][0], position[i][1]), (position[i][2], position[i][3]),
                                      (0, 255, 0), 2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(img_rgb, name + ', ' + str(distance), (position[i][0] + 10, position[i][1] - 10),
                                    font, 0.8, (0, 255, 0), 2)
                        print(name, distance)
                        print('Found the person in the video!')

                        # Show the image with the recognized face
                        plt.figure(figsize=(10, 10))
                        plt.imshow(img_rgb / 255)
                        _ = plt.axis('off')
                        winsound.Beep(600, 1000)
                        plt.show()

                        found = True
                        break

            # Resize the frame and show the video
            frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
            cv2.imshow('video', frame)

            count += 1

        else:
            print('No found the person in the video!')
            break

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()