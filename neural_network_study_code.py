import numpy as np
from pose_estimation import *
from data_initialization import *
from model_network import *

VIDEO = r'C:\Users\Roy\PycharmProjects\IML\FInalProject\pexels-mart-production-8836896.mp4'

if __name__ == '__main__':
    mat, frames = pose_estimation_video(VIDEO, 25)
    # mat, frames, i = recognize_start_of_movement(mat, frames, 'squat')
    # movement_video_streaming(frames, mat)
    # mat, frames, j = recognize_end_of_movement(mat, frames, 'squat')
    mats, frames_lst = recognize_movement(mat, frames, 'squat')
    for i in range(len(frames_lst)):
        print(i)
        movement_video_streaming(frames_lst[i], mat[i])
