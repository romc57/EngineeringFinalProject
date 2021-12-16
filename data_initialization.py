import numpy as np
from pose_estimation import *

LEFT_KNEE = 11
RIGHT_KNEE = 11
LEFT_HIP = 11
RIGHT_HIP = 8


# Todo : To center the photo and move the data to 3d.
# Todo : Take care of Nones!

def recognize_start_of_movement(mat_of_elements, frames_lst, exercise):
    """
    The next method Will cut the lines of the matrix before movement for each exercise and return the index where
     the decision was made.
    :param mat_of_elements: list of lists of body parts location correspond to the dictionary in pose_estimation.py
    :param frames_lst: the corresponds frames to the matrix.
    :param exercise: the type of exercise in the video.
    :return: mat_of_elements from the right point, frames corresponds, index.
    """
    for i in range(1, len(mat_of_elements)):
        if exercise == 'squat':
            if mat_of_elements[i - 1][LEFT_HIP][0] > mat_of_elements[i][LEFT_HIP][0] + 20 or \
                    mat_of_elements[i - 1][RIGHT_HIP][0] > mat_of_elements[i][RIGHT_HIP][0] + 20:
                return mat_of_elements[i - 1::], frames_lst[i - 1::], i


def recognize_end_of_movement(mat_of_elements, frames_lst, exercise):
    """

    :param mat_of_elements:
    :param frames_lst:
    :param exercise:
    :return:
    """
    max_y = 1000
    max_index = 0
    for i in range(4, len(mat_of_elements)):
        if exercise == 'squat':
            if mat_of_elements[i][LEFT_HIP] is not None and max_y > mat_of_elements[i][LEFT_HIP][0]:
                max_y = mat_of_elements[i][LEFT_HIP][0]
                max_index = i
            elif mat_of_elements[i][RIGHT_HIP] is not None and max_y > mat_of_elements[i][RIGHT_HIP][0]:
                max_y = mat_of_elements[i][RIGHT_HIP][0]
                max_index = i
    return mat_of_elements[:max_index - 1], frames_lst[:max_index - 1], max_index


def recognize_movement(mat_of_elements, frames, exercise):
    """Recognize full movement and return an array that each entrance is list of frames of a full movement"""
    movements_lst = list()
    frames_lst = list()
    index_1 = 0
    index_2 = 0
    while index_2 < len(mat_of_elements):
        a, b, curr_index = recognize_start_of_movement(mat_of_elements[index_2::], frames[index_2::], exercise)
        index_1 += curr_index
        a, b, curr_index = recognize_end_of_movement(mat_of_elements[index_1::], frames[index_1::], exercise)
        index_2 += curr_index
        movements_lst.append(mat_of_elements[index_1:index_2])
        frames_lst.append(frames[index_1:index_2])
    return movements_lst , frames_lst


def build_matrix_for_nn(mat, tag, desired_shape, none_handle):
    """Build matrix for the model for classification, include flag or swithc case on how to handle None."""
    pass


def create_data_set(directory, tags, none_handle):
    """Multi video, expect a path includes videos array of tags who are matched the video, or dictionary,
    none handle build matrix for nn"""
    pass
