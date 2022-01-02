import cv2 as cv
import argparse
import tests


parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--thr', default=0.4, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

args = parser.parse_args()

BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

BODY_PARTS_LIST = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'RHip', 'RKnee',
                   'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye', 'REar', 'LEar', 'Background']


def get_ankle_angle(point_list):
    if point_list[BODY_PARTS['LAnkle']]:
        if point_list[BODY_PARTS['LKnee']]:
            ankle_point = point_list[BODY_PARTS['LAnkle']]
            parrallel_point = [ankle_point[0] + 1, ankle_point[1]]
            knee_point = point_list[BODY_PARTS['LKnee']]
            ankle_angle = tests.calculate_angle(parrallel_point, ankle_point, knee_point)
            return int(ankle_angle) if ankle_angle < 95 else int(180 - ankle_angle)


def get_knee_angle(point_list):
    if point_list[BODY_PARTS['LAnkle']]:
        if point_list[BODY_PARTS['LKnee']]:
            if point_list[BODY_PARTS['LHip']]:
                knee_angle = tests.calculate_angle(point_list[BODY_PARTS['LAnkle']], point_list[BODY_PARTS['LKnee']],
                                                   point_list[BODY_PARTS['LHip']])
                return int(knee_angle)


def get_hip_angle(point_list):
    if point_list[BODY_PARTS['LKnee']]:
        if point_list[BODY_PARTS['LHip']]:
            if point_list[BODY_PARTS['Neck']]:
                hip_angle = tests.calculate_angle(point_list[BODY_PARTS['LKnee']], point_list[BODY_PARTS['LHip']],
                                                  point_list[BODY_PARTS['Neck']])
                return int(hip_angle)

inWidth = args.width
inHeight = args.height
net = cv.dnn.readNetFromTensorflow("models/graph_opt.pb")
cap = cv.VideoCapture(args.input if args.input else 0)
while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
    assert (len(BODY_PARTS) == out.shape[1])
    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > args.thr else None)
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert (partFrom in BODY_PARTS)
        assert (partTo in BODY_PARTS)
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv.imshow('OpenPose using OpenCV', frame)
 #   output.write(frame)
#output.release()
cap.release()
cv.destroyAllWindows()
