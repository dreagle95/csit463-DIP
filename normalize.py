import cv2
import os

def normalize(potentials):
    dim = (28, 28)
    norm_signs = []
    for sign in potentials:
        norm_sign = cv2.resize(sign, dim)
        norm_signs.append(norm_sign)
    return norm_signs