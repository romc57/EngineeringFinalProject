import pickle

import cv2 as cv
import argparse
from FullBody import Body, BODY_PARTS_LIST
import utils
import mediapipe as mp
from model_network import *


parser = argparse.ArgumentParser()
parser.add_argument('--input', default=0, help='Path to image or video. Skip to capture frames from camera')
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
predict_mistake = None


args = parser.parse_args()  # Load arguments
data_set_mode = args.data_set_mode  # Create mode for a data set
output_data = args.output_data  # Flag to indicate if data should be outputed
model_knn_2_d = 'models/model_number_0_2d_knn.pickle'  # Trained KNN 2d model
model_net_3_d = 'models/model_number_0_3d_net.pickle'  # Trained net 3d model
model_knn_multi_2_d = 'models/model_number_multi_0_3d_knn.pickle'  # Trained knn for multi classification
if output_data:
    run_dir = utils.create_run_dir()
else:
    run_dir = None
if data_set_mode:
    data_types = ['good', 'high_waste', 'knee_collapse', 'lifting_heels']
    type_index = 0
    sample_count = {'good': 30, 'high_waste': 10, 'knee_collapse': 10, 'lifting_heels': 10}  # How many samples should the system capture
    user_body = Body(run_dir, data_types[0])  # Create a Body object
else:
    user_body = Body(run_dir)
display_width = args.display_width
display_height = args.display_height
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
model_net, model_knn, model_multi = utils.load_models([model_net_3_d, model_knn_2_d, model_knn_multi_2_d])


cap = cv.VideoCapture(args.input if args.input else 0)
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
    """
    Draw the standing line on frame
    :param frame: Frame to draw on
    :param standing_line: Standing line coordinates
    """
    if standing_line:
        if user_body.on_standing_line():
            cv.line(frame, standing_line[0], standing_line[1], ON_LINE_COLOR, 2)
        else:
            cv.line(frame, standing_line[0], standing_line[1], OFF_LINE_COLOR, 2)


def insert_instructions(frame, txt):
    """
    Insert a user instruction on the top of the frame
    :param frame: Frame to insert on
    :param txt: Instruction to insert
    """
    cv.putText(frame, txt, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, INSTRUCTIONS_COLOR,  2)


def insert_squat_count(frame):
    """
    Insert the squat count on the frame
    :param frame: Frame to insert on
    """
    global squat_count
    height = frame.shape[0]
    cv.putText(frame, "Squat count: {}".format(squat_count), (10, height - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5,
               COUNTER_COLOR, 1)


def show_img(frame, save_frame=True):
    """
    Present the frame to the user and save the frame if got save_frame
    :param frame: Frame to present
    :param save_frame: Flag to save the frame
    """
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
    """
    Get the body points from the CV Network
    :param frame: Frame to detect
    :return: The points of the body parts
    """
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
    """
    Runs on calibrate mode is responsible to calibrate the Body object
    :param frame: Current frame detected
    :param points: Points detected
    :param results: Reguarding the points detected
    """
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
    """
    Runs the sample squat stage, initializes the squat on the Body class
    :param frame: Frame detected
    :param points: Points detected
    """
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


def get_multi_predict():
    pass


def run(frame, points, results):
    global frame_counter, squat_count, last_squat_predict, predict_mistake
    instruction = user_body.check_body_points(points, frame_counter)
    draw_standing_line(frame, standing_line_points)
    if user_body.got_valid_points():
        if last_squat_predict is not None:
            if predict_mistake is not None and last_squat_predict == 0:
                insert_instructions(frame, 'Squat {}: knn {}'.format(squat_count, utils.MULTI_LABELS[predict_mistake]))
            else:
                insert_instructions(frame, 'Squat {}: knn {}'.format(squat_count, 'Good Job!'))
        else:
            insert_instructions(frame, 'Squat!')
        insert_squat_count(frame)
        if user_body.squat():
            centered, three_d = user_body.get_squat()
            if not data_set_mode:
                last_squat_predict = get_knn_squat_predict(model_knn, centered)
                if last_squat_predict == 0:
                    predict_mistake = get_knn_squat_predict(model_multi, centered)
                else:
                    predict_mistake = None
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


def get_knn_squat_predict(knn, centered):
    centered = utils.convert_list_to_np(centered)
    indices_2d = utils.find_slicing_indices(int(knn.get_dim() / 30), utils.find_min_y_index(centered)[0],
                                            len(centered))
    data_knn = centered[indices_2d]
    predict_knn = knn.predict(data_knn.reshape(1, knn.get_dim()))
    return predict_knn[0]


def get_squat_predict():
    global model_knn, model_net
    centered, three_d = user_body.get_squat()
    three_d = utils.convert_list_to_np(three_d)
    centered = utils.convert_list_to_np(centered)
    indices_3d = utils.find_slicing_indices(int(model_net.get_dim() / 45), utils.find_min_y_index(three_d)[0],
                                      len(three_d))
    indices_2d = utils.find_slicing_indices(int(model_knn.get_dim() / 30), utils.find_min_y_index(centered)[0],
                                      len(centered))
    data_knn = centered[indices_2d]
    data_net = three_d[indices_3d]
    predict_knn = model_knn.predict(data_knn.reshape(1, model_knn.get_dim()))
    data_test_manager = DataManager(torch.tensor([data_net]), torch.tensor([1]))
    data_net_it = data_test_manager.get_data_iterator()
    for data in data_net_it:
        input, label = data
        predict_net = model_net.predict(input.float())
    # predict_net = [None]
    return predict_net[0].item(), predict_knn[0]


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
            cap = cv.VideoCapture(args.input if args.input else 0)
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

