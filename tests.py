import math


BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}


def get_length(p1, p2):
    return math.sqrt(math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2))


def calculate_angle(bottom_dot, middle_dot, top_dot):
    a_len = get_length(top_dot, bottom_dot)
    b_len = get_length(middle_dot, top_dot)
    c_len = get_length(middle_dot, bottom_dot)
    alpha_rad = math.acos((math.pow(b_len, 2) + math.pow(c_len, 2) - math.pow(a_len, 2)) / (2 * b_len * c_len))
    alpha_deg = alpha_rad * 180 / math.pi
    return alpha_deg


if __name__ == '__main__':
    calculate_angle([2, 0], [0, 0], [2, 2])
