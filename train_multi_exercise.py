import cv2
import mediapipe as mp
import os
import utils
from model_network import *

WIDTH = 1200
HEIGHT = 800

def results_to_coordinates(results):
    points = list()
    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_world_landmarks.landmark):
            cx, cy, cz = lm.x, lm.y, lm.z
            points.append((cx, cy, cz))
    return points


def create_data_set(root_directory: str, folder_name: str, run_directory: str):
    path = os.path.join(root_directory)
    for folder in os.listdir(path):
        root = os.path.join(root_directory, folder)
        id_video = 0
        for video in os.listdir(root):
            points = get_key_points(os.path.join(root, video))
            utils.write_to_txt_points(points, run_directory, f"{id_video}_{folder}.txt",
                                      f'{folder} {folder_name}')
            id_video += 1


def get_key_points(video_path):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    all_results = list()
    # For webcam input:
    cap = cv2.VideoCapture(video_path)
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            image = cv2.resize(image, (WIDTH, HEIGHT))
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            all_results.append(results)
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                output = list()
                for results in all_results:
                    # mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
                    point = results_to_coordinates(results)
                    if point:
                        output.append(point)
                break
        cap.release()
        return output


if __name__ == '__main__':
    video_path = r"D:\projects_files\final project"
    # get_key_points(video_path)
    # create_data_set(video_path, 'data', r"D:\projects_files")
    train_set, train_tags = utils.get_data_set( r"D:",  r"projects_files", [], ["push ups data", "squats data"], False)
    train_set = train_set[:1] + train_set[2:]
    train_tags = train_tags[:1] + train_tags[2:]
    normalize_data_set = utils.normalize_data_len(train_set)
    data_knn_multi = np.array(normalize_data_set)
    train_x, train_y, test_x, test_y = split_data_train_test(data_knn_multi, train_tags, 0.5)
    dim = train_x.shape[1] * train_x.shape[2] * train_x.shape[3]

    train_x_knn = train_x.reshape(len(train_x), -1)
    test_x_knn = test_x.reshape(len(test_x), -1)
    for i in range(1, 2):
        model = SimpleKNN(i, dim)
        print(f'KNN with {i} neighbors:')
        y_hat, test_y = evaluate_knn(train_x_knn, train_y, test_x_knn, test_y, model, 'accuracy')
        # plot_confusion_matrix(y_hat, test_y)
