
import cv2
import dlib
import argparse
import numpy as np
import time
from PIL import Image, ImageDraw

def argument_parse():
    # handle command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help='path to image file')
    ap.add_argument('-w', '--weights', default='./mmod_human_face_detector.dat',
                    help='path to weights file')
    args = ap.parse_args()
    return args

### 1. Read image
def read_img(path):
    # load input image
    image = cv2.imread(path)
    # print(type(image))

    if image is None:
        print("Could not read input image")
        exit()
    return image


### 2. Face detector
def face_detector_hog(image):
    b, l, t, r = 0, 0, 0, 0
    # initialize hog + svm based face detector
    hog_face_detector = dlib.get_frontal_face_detector()

    # apply face detection (hog)
    faces_hog = hog_face_detector(image, 1)

    if len(faces_hog) == 0:
        return b, l, t, r
    else:
        # loop over detected faces
        for face in faces_hog:
            l = face.left()
            t = face.top()
            r = face.right()
            b = face.bottom()
    return b, l, t, r

### 3. face landmarks estimation
def _css_to_rect(css):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object

    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])


def _raw_face_landmarks(face_image, face_locations=None, model="large"):
    pose_predictor_5_point = dlib.shape_predictor('../model/shape_predictor_5_face_landmarks.dat')
    pose_predictor_68_point = dlib.shape_predictor('../model/shape_predictor_68_face_landmarks.dat')

    face_locations = [_css_to_rect(face_location) for face_location in face_locations]

    pose_predictor = pose_predictor_68_point

    if model == "small":
        pose_predictor = pose_predictor_5_point

    return [pose_predictor(face_image, face_location) for face_location in face_locations]


def face_landmarks_detector(face_image, face_locations=None):
    """
    Given an image, returns a dict of face feature locations (eyes, nose, etc) for each face in the image

    :param face_image: image to search
    :param face_locations: Optionally provide a list of face locations to check.
    :return: A list of dicts of face feature locations (eyes, nose, etc)
    """
    landmarks = _raw_face_landmarks(face_image, face_locations)
    landmarks_as_tuples = [[(p.x, p.y) for p in landmark.parts()] for landmark in landmarks]

    # For a definition of each point index, see https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png
    return [{
        "chin": points[0:17],
        "left_eyebrow": points[17:22],
        "right_eyebrow": points[22:27],
        "nose_bridge": points[27:31],
        "nose_tip": points[31:36],
        "left_eye": points[36:42],
        "right_eye": points[42:48],
        "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
        "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [points[64]]
    } for points in landmarks_as_tuples]

if __name__ == '__main__':
    args = argument_parse()
    image = read_img(args.image)

    b, l, t, r = face_detector_hog(image)
    if [b, l, t, r] == [0, 0, 0, 0]:
        print('No face found')
        exit(-1)
    print('face location : t {}, r {}, b {}, l {}'.format(t, r, b, l))
    # Find all facial features in all the faces in the image
    face_landmarks_list = face_landmarks_detector(image, [(t, r, b, l)])

    for face_landmarks in face_landmarks_list:
        pil_image = Image.fromarray(image)
        d = ImageDraw.Draw(pil_image, 'RGBA')

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

        pil_image.show()