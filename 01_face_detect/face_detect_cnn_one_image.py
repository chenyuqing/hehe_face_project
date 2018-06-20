
import dlib
import cv2

cnn_face_detector = dlib.cnn_face_detection_model_v1('../model/mmod_human_face_detector.dat')

def main():
    # img_path = '../zdd.jpg'
    img_path = '/home/honghaier1688/workspaces/facenet/data/train/lfw_5590/Aishwarya_Rai_0001.jpg'
    image = cv2.imread(img_path)
    # Our operations on the frame come here
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # cv2.imwrite('./sample.jpg', image)
    # Display the resulting frame
    cv2.imshow('frame', image)
    cv2.waitKey()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

if __name__ == '__main__':
    import time
    start = time.time()
    main()
    print(time.time() - start)
