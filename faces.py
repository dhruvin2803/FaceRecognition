import cv2
import pickle

faces_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
# cascade for the eyes and the smile
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')
flag = False
flag2 = True
recognizer = cv2.face.LBPHFaceRecognizer_create()  # adding face recognizer from faces_train
recognizer.read("trainner.yml")  # bring in the trained data file
# to label the recognized image we are loading the pickle file we created
labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faces_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3)

    for (x, y, w, h) in faces:
        # print(x,y,w,h)
        # commented upper line to just see the clear output if it recognize me or not
        roi_gray = gray[y:y + h, x:x + w]  # for the gray frame as we trained our recognizer
        # for the gray
        roi_color = frame[y:y + h, x:x + w]
        id_, conf = recognizer.predict(roi_gray)
        if 40 <= conf <= 90 and flag == False:
            print(id_)
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
            flag = True
            break
        else:
            if (flag2):
                print("welcome shreeji")
                flag2 = False
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255, 255, 255)
            stroke = 2
            name = "Shreeji"
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
        imgItem = "my-img.png"
        cv2.imwrite(imgItem, roi_gray)

        color = (0, 255, 0)  # BGR
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for( ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
        # smile = smile_cascade.detectMultiScale(roi_gray)
        # for( ex, ey, ew, eh) in smile:
        #     cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
