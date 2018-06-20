# -*- coding:utf-8 -*-

from face_landmark_estimation import face_detector_hog, face_landmarks_detector
import cv2
from PIL import Image, ImageDraw
import numpy as np


def discern(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cap = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

    faceRects = cap.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50))
    if len(faceRects):
        for faceRect in faceRects:
            x, y, w, h = faceRect
            cv2.rectangle(img, (x, y), (x + h, y + w), (0, 255, 0), 2)  # 框出人脸

    cv2.imshow("Image", img)

def digit_markup(image):
    b, l, t, r = face_detector_hog(image)
    if [b, l, t, r] == [0, 0, 0, 0]:
        print('No face found')
        # exit(-1)
    print('face location : t {}, r {}, b {}, l {}'.format(t, r, b, l))
    # Find all facial features in all the faces in the image
    face_landmarks_list = face_landmarks_detector(image, [(t, r, b, l)])

    for face_landmarks in face_landmarks_list:
        pil_image = Image.fromarray(image)
        d = ImageDraw.Draw(pil_image, 'RGBA')
        print("face_landmarks['left_eyebrow'] : {}".format(face_landmarks['left_eyebrow']))
        # Make the eyebrows into a nightmare
        d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
        d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
        d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
        d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

        # sparkle the nose
        d.line(face_landmarks['nose_bridge'], fill=(120, 89, 78), width=5)
        d.polygon(face_landmarks['nose_tip'], fill=(120, 89, 78))

        # Gloss the lips
        d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
        d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
        d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
        d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

        # Sparkle the eyes
        d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
        d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

        # Apply some eyeliner
        d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
        d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)

        img = np.array(pil_image)
        cv2.imshow('Image', img)


if __name__ == '__main__':
    process_this_frame = True
    # capture the # 0 camera
    cap = cv2.VideoCapture(0)
    while (1):  # display image frame by frame
        ret, frame = cap.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        if process_this_frame:
            digit_markup(small_frame)
            # discern(small_frame)
        process_this_frame = not process_this_frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()  # release the camera
    cv2.destroyAllWindows()  # release the windows resource