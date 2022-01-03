import pickle

import cv2 as cv
import argparse
from FullBody import Body, BODY_PARTS_LIST
import utils
import mediapipe as mp


parser = argparse.ArgumentParser()
parser.add_argument('--input', default=1, help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--display_width', default=1280, type=int, help='Resize input to specific width.')
parser.add_argument('--display_height', default=920, type=int, help='Resize input to specific height.')
parser.add_argument('--save', default=False, type=bool, help='Save the video output')
parser.add_argument('--data_set_mode', default=False, type=bool, help='Mark true to create a dataset.')
parser.add_argument('--output_data', default=False, type=bool, help='Mark true to create a run_dir.')


INSTRUCTIONS_COLOR = (0, 0, 0)
COUNTER_COLOR = (0, 0, 0)
EXIT_COLOR = (0, 0, 255)
ON_LINE_COLOR = (255, 0, 0)
OFF_LINE_COLOR = (0, 0, 255)
last_squat_predict = None


args = parser.parse_args()
data_set_mode = args.data_set_mode
output_data = args.output_data
model_knn_2_d = 'models/model_number_0_2d_knn.pickle'
model_net_3_d = 'models/model_number_0_3d_net.pickle'
if output_data:
    run_dir = utils.create_run_dir()
else:
    run_dir = None
if data_set_mode:
    data_types = ['good', 'high_waste', 'knee_collapse', 'lifting_heels']
    type_index = 0
    sample_count = {'good': 30, 'high_waste': 10, 'knee_collapse': 10, 'lifting_heels': 10}
    user_body = Body(run_dir, data_types[0])
else:
    user_body = Body(run_dir)
display_width = args.display_width
display_height = args.display_height
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
loaded_file = open(f'{model_knn_2_d}', 'rb')
model_knn = pickle.load(loaded_file)
loaded_file_2 = open(f'{model_net_3_d}', 'rb')
model_net = pickle.load(loaded_file_2)


cap = cv.VideoCapture(args.input if args.input else 1)
output = None
if args.save:
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    output = cv.VideoWriter('training_example.mp4', fourcc, 15.0, (display_width, display_height))
frame_counter = 0
standing_line_points = None
calibrate_dur = 5
fps = 15
calibrate_frames = calibrate_dur * fps
squat_count = 0


def draw_standing_line(frame, standing_line):
    if standing_line:
        if user_body.on_standing_line():
            cv.line(frame, standing_line[0], standing_line[1], ON_LINE_COLOR, 2)
        else:
            cv.line(frame, standing_line[0], standing_line[1], OFF_LINE_COLOR, 2)


def insert_instructions(frame, txt):
    cv.putText(frame, txt, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, INSTRUCTIONS_COLOR,  2)


def insert_squat_count(frame):
    global squat_count
    height = frame.shape[0]
    cv.putText(frame, "Squat count: {}".format(squat_count), (10, height - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5,
               COUNTER_COLOR, 1)


def show_img(frame, save_frame=True):
    global frame_counter, data_types, type_index
    width, height = frame.shape[1], frame.shape[0]
    cv.putText(frame, 'To quit press q', (width - 150, height - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, EXIT_COLOR, 2)
    if data_set_mode:
        cv.putText(frame, 'Perform type: {} - {} To go'.format(data_types[type_index],
                                                               sample_count[data_types[type_index]] - squat_count),
                   (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, INSTRUCTIONS_COLOR, 2)
    resized = cv.resize(frame, (display_width, display_height))
    cv.imshow('OpenPose using OpenCV', resized)
    if save_frame:
        file_name = f'{run_dir}/img_samples/img_frame={frame_counter}.jpg'
        cv.imwrite(file_name, frame)
    if output:
        output.write(resized)


def get_points(frame):
    points = None
    global standing_line_points
    imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)  # Process img
    if results.pose_landmarks:
        points = list()
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            points.append((cx, cy))
        if not standing_line_points:
            standing_line_points = utils.get_standing_line(w, h, 0.07, 0.35)
            user_body.set_standing_line(standing_line_points, (w, h))
    return points, results


def calibrate_mode(frame, points, results):
    global calibrate_frames, frame_counter
    user_body.calibrate_class(points, frame_counter)
    draw_standing_line(frame, standing_line_points)
    if user_body.is_ready() and calibrate_frames != 0:
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        insert_instructions(frame, 'Calibrating.. stand still. frames left to start: {}'.format(calibrate_frames))
        calibrate_frames -= 1
    elif not user_body.is_ready():
        calibrate_frames = calibrate_dur * fps
        insert_instructions(frame, 'Calibrating.. place your heels on the red line. frames left to start: '
                            '{}'.format(calibrate_frames))
    elif user_body.is_ready() and calibrate_frames == 0:
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        insert_instructions(frame, 'Finished calibrating! start squatting.')
        user_body.calibrate_mode = False
        user_body.sample_squat_mode = True
    show_img(frame, save_frame=False)


def sample_squat(frame, points, results):
    global frame_counter
    instruction = user_body.check_body_points(points, frame_counter)
    if user_body.got_valid_points():
        if not user_body.init_squat_ref():
            insert_instructions(frame, 'Now lets sample a reference squat. Start a squat')
            mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        else:
            insert_instructions(frame, 'Finished sampling. You can start squatting!')
            user_body.sample_squat_mode = False
    else:
        insert_instructions(frame, instruction)
    draw_standing_line(frame, standing_line_points)
    show_img(frame, save_frame=False)


def run(frame, points, results):
    global frame_counter, squat_count, last_squat_predict
    instruction = user_body.check_body_points(points, frame_counter)
    draw_standing_line(frame, standing_line_points)
    if user_body.got_valid_points():
        if last_squat_predict:
            insert_instructions(frame, 'Squat {}: net: {} knn {}'.format(squat_count, last_squat_predict[0],
                                                                         last_squat_predict[1]))
        else:
            insert_instructions(frame, 'Squat!')
        insert_squat_count(frame)
        if user_body.squat():
            if not data_set_mode:
                last_squat_predict = get_squat_predict()
            squat_count += 1
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        if data_set_mode:
            show_img(frame, save_frame=False)
        else:
            show_img(frame, save_frame=True)
    else:
        insert_instructions(frame, instruction)
        insert_squat_count(frame)
        show_img(frame, save_frame=False)


def get_squat_predict():
    global model_knn, model_net
    centered, three_d = user_body.get_squat()
    three_d = utils.convert_list_to_np(three_d)
    centered = utils.convert_list_to_np(centered)
    indices_3d = utils.find_slicing_indices(int(model_net.get_dim() / 45), utils.find_min_y_index(three_d), len(three_d))
    indices_2d = utils.find_slicing_indices(int(model_knn.get_dim() / 30), utils.find_min_y_index(centered), len(centered))
    data_knn = utils.convert_list_to_np([centered[indices_2d]])
    data_net = utils.convert_list_to_np([three_d[indices_3d]])
    predict_knn = model_knn.predict(data_knn.reshape(len(data_knn), -1))
    # predict_net = model_net.predict(data_net)
    predict_net = [None]
    return predict_net[0], predict_knn[0]


def data_set_creator(frame, points, results):
    global data_types, type_index, squat_count, sample_count, cap
    if squat_count == sample_count[data_types[type_index]]:
        cap.release()
        cv.destroyAllWindows()
        print('Saving info for type {}'.format(data_types[type_index]))
        user_body.output_data_set_points()
        type_index += 1
        if type_index >= len(data_types):
            return False
        else:
            user_body.training_dir = data_types[type_index]
            user_body.calibrate_mode = True
            squat_count = 0
            cap = cv.VideoCapture(args.input if args.input else 1)
            return True
    run(frame, points, results)
    return True


while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    frame_counter += 1
    if not hasFrame:
        cv.waitKey()
        break
    points, results = get_points(frame)
    if not points:
        continue
    if user_body.calibrate_mode:
        calibrate_mode(frame, points, results)
    elif user_body.sample_squat_mode:
        sample_squat(frame, points, results)
    else:
        if data_set_mode:
            if not data_set_creator(frame, points, results):
                break
            else:
                continue
        else:
            run(frame, points, results)
if output:
    output.release()
cap.release()
cv.destroyAllWindows()
if not data_set_mode and run_dir:
    print('Outputting data...')
    user_body.output_points()

