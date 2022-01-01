import cv2 as cv
import argparse
from FullBody import Body, BODY_PARTS_LIST
import utils
import mediapipe as mp

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--display_width', default=1280, type=int, help='Resize input to specific width.')
parser.add_argument('--display_height', default=920, type=int, help='Resize input to specific height.')
parser.add_argument('--save', default=False, type=bool, help='Save the video output')

INSTRUCTIONS_COLOR = (0, 0, 0)
COUNTER_COLOR = (0, 0, 0)
EXIT_COLOR = (0, 0, 255)
ON_LINE_COLOR = (255, 0, 0)
OFF_LINE_COLOR = (0, 0, 255)


args = parser.parse_args()
display_width = args.display_width
display_height = args.display_height
run_dir = utils.create_run_dir()
user_body = Body(run_dir)
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
vision_points = list()

cap = cv.VideoCapture(args.input if args.input else 1)
output = None
if args.save:
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    output = cv.VideoWriter('output.mp4', fourcc, 15.0, (display_width, display_height))
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
    global frame_counter
    width, height = frame.shape[1], frame.shape[0]
    cv.putText(frame, 'To quit press q', (width - 150, height - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, EXIT_COLOR, 2)
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
    global frame_counter, squat_count
    instruction = user_body.check_body_points(points, frame_counter)
    draw_standing_line(frame, standing_line_points)
    if user_body.got_valid_points():
        insert_instructions(frame, 'Squat!')
        insert_squat_count(frame)
        if user_body.squat():
            squat_count += 1
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        show_img(frame, save_frame=True)
    else:
        insert_instructions(frame, instruction)
        insert_squat_count(frame)
        show_img(frame, save_frame=False)


while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
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
        run(frame, points, results)
    frame_counter += 1
if output:
    output.release()
cap.release()
cv.destroyAllWindows()
print('Outputing data...')
user_body.output_points()

