import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)
pTime = 0
frame_idx = 0

while True:
    frame_idx += 1
    status, frame = cap.read()  # Get frame
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert img to RGB
    results = pose.process(imgRGB)  # Process img
    print('results : {}'.format(results.pose_landmarks))
    print(type(results.pose_landmarks))
    if results.pose_landmarks:
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = frame.shape
            print('Id {}, Lm {}'.format(id, lm))
            cx, cy = int(lm.x * w), int(lm.y * h)
            print('in results x: {} y: {}'.format(cx, cy))
            cv2.putText(frame, "{}:{}".format(id, (cx, cy)), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow("Image", frame)
        cv2.imwrite('testing/img_{}.jpg'.format(frame_idx), frame)
        cv2.waitKey(1)