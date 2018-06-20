
from mtcnn.mtcnn import MTCNN
import cv2

# image = cv2.imread('../tanweiwei.jpg')
detector = MTCNN()

# Open the input movie file
vid_name = 'dongdong.mp4'
vid = cv2.VideoCapture('../'+vid_name)
length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('output_mtcnn'+vid_name+'.avi', fourcc, 24.97, (640, 360))

frame_number = 0

while True:
    ret, frame = vid.read()

    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break

    # grayscale conversion of image because it is computationally efficient
    # to perform operations on single channeled (grayscale) image
    # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecting faces
    result = detector.detect_faces(frame)

    for i in range(len(result)):
        # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
        bounding_box = result[i]['box']
        keypoints = result[i]['keypoints']

        cv2.rectangle(frame,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                      (0, 155, 255),
                      2)

        cv2.circle(frame, (keypoints['left_eye']), 2, (0, 155, 255), 2)
        cv2.circle(frame, (keypoints['right_eye']), 2, (0, 155, 255), 2)
        cv2.circle(frame, (keypoints['nose']), 2, (0, 155, 255), 2)
        cv2.circle(frame, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
        cv2.circle(frame, (keypoints['mouth_right']), 2, (0, 155, 255), 2)

        # Writing face number on image
        cv2.putText(frame, "Face :{}".format(i + 1), (bounding_box[0] - 10, bounding_box[1] - 10),
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