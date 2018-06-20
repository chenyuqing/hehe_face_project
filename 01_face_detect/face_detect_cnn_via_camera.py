
import dlib
import cv2

cap = cv2.VideoCapture(0)

cnn_face_detector = dlib.cnn_face_detection_model_v1('../model/mmod_human_face_detector.dat')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = cnn_face_detector(gray, 1)

    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for i, d in enumerate(faces):
        x = d.rect.left()
        y = d.rect.top()
        w = d.rect.right() - x
        h = d.rect.bottom() - y
        print('confidence : {}'.format(d.confidence))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


