import cv2
import numpy as np
import dlib

def extract_face(imgGray,detector,predictor):
    faces = detector(imgGray)
    print(len(faces))
    box = []
    all_faces_Points = []
    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        box.append([x1,y1,x2,y2])
        landmarks = predictor(imgGray, face)
        myPoints = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            myPoints.append([x, y])

        all_faces_Points.append(myPoints)

    return box,all_faces_Points

def detect_face_landmark(img,detector,predictor):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    box,landmarks = extract_face(imgGray,detector,predictor)
    # Visualize the output
    for item in box:
        x1, y1, x2, y2 =item
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    for item in landmarks:
        for n,subitem in enumerate(item):
            x, y = subitem
            cv2.circle(img, (x, y), 5, (50,50,255),cv2.FILLED)
            #cv2.putText(img,str(n),(x,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,0,255),1)

    return img



def main():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    image_source = False
    if image_source is True:
        path = 'images/04.png'
        image = cv2.imread(path)
        detect_face_landmark(image, detector, predictor)

    else:
        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        frameWidth, frameHeight = 900, 1200  # according to show series
        out = cv2.VideoWriter('output4.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15,
                              (frameHeight, frameWidth))

        cap = cv2.VideoCapture(0)

        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            img = detect_face_landmark(frame,detector,predictor)
            img =cv2.resize(img,(1200,900))
            out.write(img)
            cv2.imshow('image', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        out.release()
        cv2.destroyAllWindows()



if __name__ =='__main__':
    main()



