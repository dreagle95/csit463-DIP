import cv2


def normalize(potentials, dims):
    # dim = (28, 28)
    norm_signs = []
    for sign in potentials:
        norm_sign = cv2.resize(sign, dims)
        norm_signs.append(norm_sign)
    return norm_signs