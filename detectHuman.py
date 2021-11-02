import cv2
import numpy as np


protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
#detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
#detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
DSIZE_FRAME=(1000,600)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

def detect(frame):
    frame= cv2.resize(frame,DSIZE_FRAME)
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

    detector.setInput(blob)
    person_detections = detector.forward()
    rects = []     
    for i in np.arange(0, person_detections.shape[2]):
        confidence = person_detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(person_detections[0, 0, i, 1])
            if CLASSES[idx] != "person":
                continue
            person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            rects.append(person_box)

    boundingboxes = np.array(rects)     # box hien tai
    boundingboxes = boundingboxes.astype(int)
   
    objects = boundingboxes
    count=0
    for bbox in objects:
        x1, y1, x2, y2 = bbox
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        count=count+1
        text = "ID: {}".format(count)
        cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        
    opc_txt = "Count: {}".format(len(boundingboxes))
    cv2.putText(frame, opc_txt, (3, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    cv2.imshow('app',frame)
    return frame



def detectByCamera(writer):   
    video = cv2.VideoCapture(0)
    while True:
        check, frame = video.read()
        frame = detect(frame)
        if writer is not None:
            writer.write(frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    video.release()
    writer.release()

def detectByPathVideo(path, writer):

    video = cv2.VideoCapture(path)
    check, frame = video.read()
    if check == False:
        print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
        return
    while video.isOpened():
        #check is True if reading was successful 
        check, frame =  video.read()
        if check:
            frame = detect(frame)
            if writer is not None:
                writer.write(frame)
            key = cv2.waitKey(1)
            if key== ord('q'):
                break
        else:
            break
    cv2.destroyAllWindows()
    video.release()
    writer.release()

def detectByPathImage(path, output_path):
    image = cv2.imread(path)
    result_image = detect(image)
    if output_path is not None:
        cv2.imwrite(output_path, result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    out = cv2.VideoWriter("video/outputtest2.mp4", fourcc, 5.0, DSIZE_FRAME)
    detectByPathVideo('video/testvideo2.mp4',out)
    
    
