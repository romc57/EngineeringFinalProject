import cv2
import mediapipe as mp
import numpy as np
import main


def create_data_set(root_directory: str, classes: dict, target_path: str):
    # TODO : run on directory and save frames. normalize and tag. save txt in target
    pass


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
                for results in all_results:
                    mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
                    print(results.pose_world_landmarks)
                break
        cap.release()


if __name__ == '__main__':
    video_path = r"D:\projects_files\final project\push ups\pexels-kampus-production-8171374.mp4"
    get_key_points(video_path)

