import cv2
import mediapipe as mp
import numpy as np
import main
import os
import utils


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


def normalize_data(data_set: list) -> list:
    pass


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
            image = cv2.resize(image, (1200, 800))
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
    create_data_set(video_path, 'data', r"D:\projects_files")
