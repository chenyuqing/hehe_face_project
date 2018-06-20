
from scipy import misc
import dlib
import cv2
import os

cnn_face_detector = dlib.cnn_face_detection_model_v1('../model/mmod_human_face_detector.dat')

def get_batch_image(file_path):
    return os.listdir(file_path)

def crop_one_image(image, name, width_size, height_size):
    # Our operations on the frame come here
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = cnn_face_detector(gray, 1)

    # print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for i, d in enumerate(faces):
        x = d.rect.left()
        y = d.rect.top()
        w = d.rect.right() - x
        h = d.rect.bottom() - y
        # print('confidence : {}'.format(d.confidence))
        # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cropped = image[y:y+h, x:x+w]
        try:
            scaled = misc.imresize(cropped, (width_size, height_size), interp='bilinear')
            misc.imsave(name, scaled)
        except (IOError, ValueError, IndexError) as e:
            errorMessage = '{}: {}'.format(os.path.join(file_path, f), e)
            print(errorMessage)

        # scaled = cv2.resize(cropped, (width_size, height_size))



        # cv2.imwrite(name, image[y:y+h,x:x+w])
    # Display the resulting frame
    # cv2.imshow('frame', image[y:y+h,x:x+w])
    # cv2.waitKey()
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()

if __name__ == '__main__':
    file_path = '/home/honghaier1688/workspaces/facenet/data/train/lfw_5590'
    fl = get_batch_image(file_path)
    print(fl)
    for f in fl:
        name = file_path + '_align_face_182/' + f.split('.')[0] + '_face.jpg'
        print(name)
        # image = cv2.imread(os.path.join(file_path, f))
        try:
            image = misc.imread(os.path.join(file_path, f))
        except (IOError, ValueError, IndexError) as e:
            errorMessage = '{}: {}'.format(os.path.join(file_path, f), e)
            print(errorMessage)
        crop_one_image(image, name, 182, 182)
