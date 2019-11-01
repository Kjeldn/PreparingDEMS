import numpy as np

def calculate_distance_from_pixels(width: bool, n_pixels: int, flown_height):
    return ((flown_height * np.tan(np.arcsin(14/35)) * 2) / 3648 if width else 5472) * n_pixels;


def calculate_flown_height_from_pixels(width:bool, n_pixels: int, dist):
    return (dist /n_pixels) * (3648 if width else 5472) / (np.tan(np.arcsin(14/35)) * 2)