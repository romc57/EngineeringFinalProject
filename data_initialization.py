import numpy as np
from pose_estimation import *

LEFT_KNEE = 11
RIGHT_KNEE = 11


def recognize_start_of_movement(mat_of_elements, frames_lst, exercise):
    """Will cut the lines of the matrix before movement for each exercise"""
    for i in range(1,len(mat_of_elements)):
        if exercise == 'squat':
            if mat_of_elements[i-1][LEFT_KNEE][0] > mat_of_elements[i][LEFT_KNEE][0] + 10 or mat_of_elements[i-1][RIGHT_KNEE][0]\
                    > mat_of_elements[i][RIGHT_KNEE][0] + 10:
                return mat_of_elements[i - 1::], frames_lst[i - 1::]




def build_matrix_for_nn(mat, tag, desired_shape, none_handle):
    """Build matrix for the model for classification, include flag or swithc case on how to handle None."""
    pass


def create_data_set(directory, tags, none_handle):
    """Multi video, expect a path includes videos array of tags who are matched the video, or dictionary,
    none handle build matrix for nn"""
    pass