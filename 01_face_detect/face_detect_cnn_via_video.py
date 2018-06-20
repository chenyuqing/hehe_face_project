
import dlib
import cv2
print(cv2.__version__)

cnn_face_detector = dlib.cnn_face_detection_model_v1('../model/mmod_human_face_detector.dat')

# Open the input movie file
vid_name = 'Utolaba_tanweiwei.mp4'
# vid_name = '22.mp4'
vid = cv2.VideoCapture('../'+vid_name)


# get the fps, width and height of video

length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
fps = vid.get(cv2.CAP_PROP_FPS)
width, height = int(vid.get(3)), int(vid.get(4))
print(width, height)

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# output_movie = cv2.VideoWriter('output_cnn'+vid_name+'.avi', fourcc, fps, (height, width))
output_movie = cv2.VideoWriter('output_cnn'+vid_name+'.avi', fourcc, fps, (width, height))

frame_number = 0

while True:
    ret, frame = vid.read()

    frame_number += 1
    # Quit when the input video file ends
    if not ret:
        break

    # rotate the image
    # M = cv2.getRotationMatrix2D((width / 2, height / 2), 270, scale=1.0)
    # frame = cv2.warpAffine(frame, M, (height, width))

    # grayscale conversion of image because it is computationally efficient
    # to perform operations on single channeled (grayscale) image
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecting faces
    # face_boundaries = face_detector(frame_gray, 0)
    face_boundaries = cnn_face_detector(frame_gray, 1)


    for (enum, face) in enumerate(face_boundaries):
        # let's first draw a rectangle on the face portion of image
        ## cnn
        x = face.rect.left()
        y = face.rect.top()
        w = face.rect.right() - x
        h = face.rect.bottom() - y

        # Drawing Rectangle on face part
        cv2.rectangle(frame, (x, y), (x + w, y + h), (120, 160, 230), 2)

        # Writing face number on image
        cv2.putText(frame, "Face :{}".format(enum + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 128), 2)

    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

    cv2.imshow("frame", frame)

    #  Stop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break;

# All done!
vid.release()
output_movie.release()
cv2.destroyAllWindows()
