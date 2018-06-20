
import dlib
import cv2
import numpy as np

# model to detect face
cnn_face_detector = dlib.cnn_face_detection_model_v1('../model/mmod_human_face_detector.dat')
# model to predict the landmarks
landmark_predictor = dlib.shape_predictor('../model/shape_predictor_68_face_landmarks.dat')

# Function for creating landmark coordinate list
def land2coords(landmarks, dtype="int"):
    # initialize the list of tuples
    # (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (a, b)-coordinates
    for i in range(0, 68):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

    # return the list of (a, b)-coordinates
    return coords

def main():
    img_path = '../two_people.jpg'
    # read the image
    image = cv2.imread(img_path)

    faces = cnn_face_detector(image, 1)

    for (enum, face) in enumerate(faces):
        # let's first draw a rectangle on the face portion of image
        x = face.rect.left()
        y = face.rect.top()
        w = face.rect.right() - x
        h = face.rect.bottom() - y
        # Drawing Rectangle on face part
        cv2.rectangle(image, (x, y), (x + w, y + h), (120, 160, 230), 2)

        # Now when we have our ROI(face area) let's
        # predict and draw landmarks
        landmarks = landmark_predictor(image, face.rect)
        # converting co-ordinates to NumPy array
        landmarks = land2coords(landmarks)
        for (a, b) in landmarks:
            # Drawing points on face
            cv2.circle(image, (a, b), 3, (255, 0, 0), -1)

        # Writing face number on image
        cv2.putText(image, "Face :{}".format(enum + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 128), 2)

    cv2.imwrite('./tp_landmarks.jpg', image)

    # display on the screen
    cv2.imshow('img', image)
    cv2.waitKey()
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()

import time
if __name__ == '__main__':
    start = time.time()
    main()
    print(time.time() - start)