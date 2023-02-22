import math

import cv2
import cv2 as cv
import numpy as np
import time

from Segment import Segment

def convertBGR2HLS(image):
    image_HLS = np.zeros(image.shape, dtype='uint8')

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            B = image[row][col][0] / 255
            G = image[row][col][1] / 255
            R = image[row][col][2] / 255

            V_max = max(B, G, R)
            V_min = min(B, G, R)
            L = (V_max + V_min)/2
            S = 0
            if L < 0.5:
                S = (V_max-V_min)/(V_max+V_min)
            else:
                S = (V_max-V_min) / (2-(V_max+V_min))

            V_diff = V_max-V_min
            H = 0
            if V_max == R:
                H = 60*(G-B)/V_diff
            elif V_max == G:
                H = 120 + (60*(B-R)/V_diff)
            elif V_max == B:
                H = 240 + (60*(R-G)/V_diff)
            elif R == G and G == B:
                H = 0

            if H < 0:
                H = H + 360

            L = cap_value_to_8bit(255*L)
            S = cap_value_to_8bit(255*S)
            H = H / 2
            image_HLS[row][col] = (np.rint([H, L, S])).astype(int)

    hls_cv = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    return image_HLS

def convertHLS2BGR(image):
    image_BGR = np.zeros(image.shape, dtype='uint8')

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            H = image[row][col][0]
            L = image[row][col][1] / 255
            S = image[row][col][2] / 255

            C = (1-abs(2*L-1))*S
            H_prim = H / 30 # we divide by 30 not 60 because hue is stored as 0-180 value, not 0-360
            X = C * (1-abs(H_prim % 2 - 1))

            R = G = B = 0
            if 0 <= H_prim < 1:
                R = C
                G = X
            elif 1 <= H_prim < 2:
                R = X
                G = C
            elif 2 <= H_prim < 3:
                G = C
                B = X
            elif 3 <= H_prim < 4:
                G = X
                B = C
            elif 4 <= H_prim < 5:
                R = X
                B = C
            elif 5 <= H_prim < 6:
                R = C
                B = X

            m = L - C/2
            image_BGR[row][col] = (np.rint([(B+m)*255, (G+m)*255, (R+m)*255])).astype(int)

    bgr_cv = cv2.cvtColor(image, cv2.COLOR_HLS2BGR)
    return image_BGR


def convertBGR2YUV(image):
    image_YUV = np.zeros(image.shape, dtype='uint8')

    YUV_lut = np.array([[0.114, 0.587, 0.299], [0.436, -0.289, -0.147], [-0.100, -0.515, 0.615]])

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            B = image[row][col][0]
            R = image[row][col][2]
            Y = cap_value_to_8bit(np.matmul(YUV_lut[0], image[row][col]))
            U = cap_value_to_8bit((B - Y) * 0.492 + 128)
            V = cap_value_to_8bit((R - Y) * 0.877 + 128)

            image_YUV[row][col] = (np.rint([Y, U, V])).astype(int)

    return image_YUV


def convertYUV2BGR(image):
    image_BGR = np.zeros(image.shape, dtype='uint8')

    BGR_lut = np.array([[1, 2.03211, 0], [1, -0.39465, -0.58060], [1, 0, 1.13983]])

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            Y = image[row][col][0]
            U = image[row][col][1]
            V = image[row][col][2]

            B = cap_value_to_8bit(np.matmul(BGR_lut[0], [Y, U - 128, V - 128]))
            G = cap_value_to_8bit(np.matmul(BGR_lut[1], [Y, U - 128, V - 128]))
            R = cap_value_to_8bit(np.matmul(BGR_lut[2], [Y, U - 128, V - 128]))
            image_BGR[row][col] = (np.rint([B, G, R])).astype(int)

    return image_BGR

def cap_value_to_8bit(value):
    return max(0, min(value, 255))

def equalize_histogram(image_channel):
    equalized_image = np.zeros(image_channel.shape, dtype='uint8')

    num_of_pixels = image_channel.shape[0] * image_channel.shape[1]
    num_of_pixels_for_value = np.zeros(256, dtype=int)
    equalizing_LUT = np.zeros(256, dtype=int)

    for row in range(image_channel.shape[0]):
        for col in range(image_channel.shape[1]):
            pixel_value = image_channel[row][col]
            num_of_pixels_for_value[pixel_value] += 1

    probabilites_sum = 0
    for i in range(256):
        probabilites_sum += num_of_pixels_for_value[i]
        equalizing_LUT[i] = (np.rint(probabilites_sum*255/num_of_pixels)).astype(int)

    for row in range(image_channel.shape[0]):
        for col in range(image_channel.shape[1]):
            equalized_image[row][col] = equalizing_LUT[image_channel[row][col]]

    return equalized_image


def check_if_color_in_threshold(pixel):
    MIN_VALUE = [16, 40, 150]
    MAX_VALUE = [30, 240, 255]

    if pixel[0] >= MIN_VALUE[0] and pixel[0] <= MAX_VALUE[0] and \
            pixel[1] >= MIN_VALUE[1] and pixel[1] <= MAX_VALUE[1] and \
            pixel[2] >= MIN_VALUE[2] and pixel[2] <= MAX_VALUE[2]:
        return True
    else:
        return False

def threshold_image(image):
    thresholded_image = np.zeros(image.shape, dtype='uint8')

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if (check_if_color_in_threshold(image[row][col])):
                thresholded_image[row][col][:] = 255

    return thresholded_image

def is_pixel_white(pixel):
    return True if pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255 else False

def erosion(binary_image, mask_size):
    assert mask_size % 2 == 1, f"Odd number expected, got: {mask_size}"
    image_after_erosion = binary_image.copy()

    offset = math.floor(mask_size/2)
    #first and last {offset} rows/columns are skipped
    for row in range(offset, binary_image.shape[0] - offset):
        for col in range(offset, binary_image.shape[1] - offset):
            pixel_neighbourhood = np.zeros((mask_size, mask_size, 3))

            for r in range(-offset, offset+1):
                for c in range(-offset, offset+1):
                    pixel_neighbourhood[r+1][c+1] = binary_image[row+r][col+c]

            image_after_erosion[row][col] = np.amin(pixel_neighbourhood)

    return image_after_erosion

def dilation(binary_image, mask_size):
    assert mask_size % 2 == 1, f"Odd number expected, got: {mask_size}"
    image_after_dilation = binary_image.copy()

    offset = math.floor(mask_size/2)
    #first and last {offset} rows/columns are skipped
    for row in range(offset, binary_image.shape[0] - offset):
        for col in range(offset, binary_image.shape[1] - offset):
            pixel_neighbourhood = np.zeros((mask_size, mask_size, 3))

            for r in range(-offset, offset+1):
                for c in range(-offset, offset+1):
                    pixel_neighbourhood[r+1][c+1] = binary_image[row+r][col+c]

            image_after_dilation[row][col] = np.amax(pixel_neighbourhood)

    return image_after_dilation


def flood_fill(pixel, segment_color, image, segments_list):
    colored_image = image.copy()
    segment_pixels = []
    pixels_queue = []
    pixels_queue.append(pixel)

    while len(pixels_queue) > 0:
        current_pixel = pixels_queue[0]
        pixels_queue.pop(0)

        colored_image[current_pixel][:] = segment_color
        segment_pixels.append(current_pixel)

        if current_pixel[0] != 0 and current_pixel[0] != image.shape[0]-1 and current_pixel[1] != 0 and current_pixel[1] != image.shape[1]-1:
            left_pixel = (current_pixel[0], current_pixel[1]-1)
            right_pixel = (current_pixel[0], current_pixel[1]+1)
            top_pixel = (current_pixel[0]-1, current_pixel[1])
            bottom_pixel = (current_pixel[0]+1, current_pixel[1])

            if left_pixel not in pixels_queue and is_pixel_white(colored_image[left_pixel]):
                pixels_queue.append(left_pixel)

            if right_pixel not in pixels_queue and is_pixel_white(colored_image[right_pixel]):
                pixels_queue.append(right_pixel)

            if top_pixel not in pixels_queue and is_pixel_white(colored_image[top_pixel]):
                pixels_queue.append(top_pixel)

            if bottom_pixel not in pixels_queue and is_pixel_white(colored_image[bottom_pixel]):
                pixels_queue.append(bottom_pixel)

    segments_list.append(Segment(segment_color, segment_pixels, colored_image))

    return colored_image

def get_random_color():
    return list(np.random.randint(255, size=3))

def segmentate_image(image, segments_list):
    segmented_image = image.copy()

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if is_pixel_white(segmented_image[row][col]):
                segmented_image = flood_fill((row, col), get_random_color(), segmented_image, segments_list)

    return segmented_image

def filter_substantial_segments(segments, image):
    for segment in reversed(segments):
        if segment.calculate_area() < 300:
            for pixel in segment.pixels:
                image[pixel][0] = 0
                image[pixel][1] = 0
                image[pixel][2] = 0

            segments.remove(segment)


def search_for_logos(segments):
    segments_with_logos = []

    for segment in segments:
        width = segment.x_max-segment.x_min
        height = segment.y_max-segment.y_min
        if (segment.M1 >= 0.4 and segment.M1 <= 0.6 and
            segment.M2 >= 0.005 and segment.M2 <= 0.06 and
            segment.M7 >= 0.018 and segment.M7 <= 0.1 and
            segment.W3 >= 1.8 and segment.W3 <= 4 and
            segment.W7 >= 0.75 and segment.W7 <= 1) or \
            (width/height < 1.33 and height/width < 1.33 and
            0.1 <= segment.M1 <= 1 and
            0.00001 <= segment.M2 <= 0.3 and
            0.01 <= segment.M7 <= 0.2 and
            1 <= segment.W3 <= 4 and
            0.75 <= segment.W7 <= 1):
            segments_with_logos.append(segment)

    return segments_with_logos

def mark_logos_on_image(segments, image):
    image_with_logos = image.copy()
    for segment in segments:
        image_with_logos = segment.mark_logo(image_with_logos)

    return image_with_logos


# def resize_image(image, resize_width, resize_height):
#
#     original_height = image.shape[0]
#     original_width = image.shape[1]
#     channels = len(image[0][0])
#
#     B = image[:, :, 0]
#     G = image[:, :, 1]
#     R = image[:, :, 2]
#
#     resized_image = np.zeros((resize_width, resize_height, channels), dtype=np.uint8)
#
#     x_scale = original_width / resize_width
#     y_scale = original_height / resize_height
#
#     resize_idx = np.zeros((resize_width, resize_height))
#     resize_index_x = np.ceil(np.arange(0, original_width, x_scale)).astype(int)
#     resize_index_y = np.ceil(np.arange(0, original_height, y_scale)).astype(int)
#     resize_index_x[np.where(resize_index_x == original_width)] -= 1
#     resize_index_y[np.where(resize_index_y == original_height)] -= 1
#
#     resized_image[:, :, 0] = B[resize_index_x, :][:, resize_index_y]
#     resized_image[:, :, 1] = G[resize_index_x, :][:, resize_index_y]
#     resized_image[:, :, 2] = R[resize_index_x, :][:, resize_index_y]
#
#     return resized_image


if __name__ == '__main__':
    start_time = time.time()
    image_BGR = cv.imread('resources/mc5.jpg')
    image_BGR = cv2.resize(image_BGR, (1200, 900)) #resize_image(image_BGR, 1200, 900)#
    print(time.time() - start_time)

    image_YUV = convertBGR2YUV(image_BGR)
    print(time.time() - start_time)
    image_YUV[:, :, 0] = equalize_histogram(image_YUV[:, :, 0])
    print(time.time() - start_time)
    image_equalized = convertYUV2BGR(image_YUV)
    print("equalized")
    print(time.time() - start_time)
    cv2.imshow('YUV equalized', image_equalized)

    image_HLS = convertBGR2HLS(image_equalized)
    cv2.imshow('HLS', convertHLS2BGR(image_HLS))

    binary_image = threshold_image(image_HLS)
    cv2.imshow('Binary', binary_image)
    binary_image = dilation(binary_image, 3)
    binary_image = erosion(binary_image, 3)

    segments_list = []
    segmented_image = segmentate_image(binary_image, segments_list)
    cv2.imshow('Threshold', segmented_image)

    filter_substantial_segments(segments_list, segmented_image)
    cv2.imshow('Segmented', segmented_image)

    seg_before_logos = segments_list.copy()
    segments_list = search_for_logos(segments_list)
    image_with_logos = mark_logos_on_image(segments_list, image_BGR)

    cv2.imshow('Detected logos', image_with_logos)
    end_time = time.time()
    print(end_time - start_time)

    cv2.waitKey(0)
    cv2.destroyAllWindows()