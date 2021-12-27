import cv2 as cv
import argparse
from FullBody import Body, BODY_PARTS_LIST
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--thr', default=0.18, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')
parser.add_argument('--save', default=False, type=bool, help='Save the video output')


BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]


args = parser.parse_args()

inWidth = args.width
inHeight = args.height
run_dir = utils.create_run_dir()
user_body = Body(run_dir)

net = cv.dnn.readNetFromTensorflow("models/graph_opt.pb")
cap = cv.VideoCapture(args.input if args.input else 0)
fourcc = None
output = None
if args.save:
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    output = cv.VideoWriter('output.mp4', fourcc, 10.0, (640, 480))

frame_counter = 0
standing_line_points = None


def draw_points(frame, points):
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)


def draw_standing_line(frame, standing_line):
    if standing_line:
        cv.line(frame, standing_line[0], standing_line[1], (0, 0, 255), 3)


def show_img(frame, save_frame=True):
    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv.imshow('OpenPose using OpenCV', frame)
    if save_frame:
        file_name = f'{run_dir}/img_samples/img_frame={frame_counter}.jpg'
        cv.imwrite(file_name, frame)
    if output:
        output.write(frame)
        output.release()

while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    if not standing_line_points:
        standing_line_points = utils.get_standing_line(frameWidth, frameHeight, 0.15, 0.3)
        user_body.set_standing_line(standing_line_points, (frameWidth, frameHeight))
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), mean=(0, 0, 0), swapRB=False, crop=False))
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
    if not user_body.is_ready():
        print(user_body.calibrate_class(points, frame_counter))
        if user_body.is_ready():
            points = user_body.get_network_point_format()
            draw_standing_line(frame, standing_line_points)
            draw_points(frame, points)
            show_img(frame)
            user_body.output_points()
            utils.print_points(points, BODY_PARTS_LIST)
            utils.plot_points(points, BODY_PARTS_LIST, 'Vision point format frame={}'.format(frame_counter), run_dir,
                              'vision_plots_id_{}'.format(frame_counter), 'vision_plots')
        else:
            draw_standing_line(frame, standing_line_points)
            show_img(frame, save_frame=False)
    else:
        print(user_body.check_body_points(points, frame_counter))
        if user_body.got_valid_points():
            points = user_body.get_network_point_format()
            utils.plot_points(points, BODY_PARTS_LIST, 'Vision point format frame={}'.format(frame_counter), run_dir,
                              'vision_plots_id_{}'.format(frame_counter), 'vision_plots')
            draw_standing_line(frame, standing_line_points)
            draw_points(frame, points)
            show_img(frame)
    frame_counter += 1
cap.release()
cv.destroyAllWindows()

