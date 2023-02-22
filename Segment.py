import math
import numpy as np


class Segment:
    def __init__(self, color, pixels, image):
        self.color = color
        self.pixels = pixels
        self.segment_image = []
        self.x_min = math.inf
        self.x_max = 0
        self.y_min = math.inf
        self.y_max = 0
        self.area = 0
        self.perimeter = 0
        self.W3 = 0
        self.W7 = 0
        self.M1 = 0
        self.M2 = 0
        self.M3 = 0
        self.M7 = 0
        self.create_segment_image(image)
        self.calculate_area()
        self.calculate_perimeter()
        self.calculate_characteristics()


    def calculate_area(self):
        self.area = len(self.pixels)
        return self.area

    def calculate_perimeter(self):
        sum_of_vertical_border_pixels = 0
        sum_of_horizontal_border_pixels = 0

        for row in range(self.segment_image.shape[0]):
            for col in range(self.segment_image.shape[1] - 1):
                if (not np.array_equal(self.segment_image[row, col], self.segment_image[row, col+1]) and
                        ( np.array_equal(self.segment_image[row, col], self.color) or np.array_equal(self.segment_image[row, col+1], self.color) )):
                    sum_of_vertical_border_pixels += 1


        for col in range(self.segment_image.shape[1]):
            for row in range(self.segment_image.shape[0] - 1):
                if (not np.array_equal(self.segment_image[row, col], self.segment_image[row+1, col]) and
                        ( np.array_equal(self.segment_image[row, col], self.color) or np.array_equal(self.segment_image[row+1, col], self.color) )):
                    sum_of_horizontal_border_pixels += 1


        self.perimeter = sum_of_horizontal_border_pixels + sum_of_vertical_border_pixels
        return self.perimeter

    def search_for_segment_boundaries(self):
        for row, col in self.pixels:
            if col < self.x_min:
                self.x_min = col
            if col > self.x_max:
                self.x_max = col
            if row < self.y_min:
                self.y_min = row
            if row > self.y_max:
                self.y_max = row

    def create_segment_image(self, image):
        self.search_for_segment_boundaries()

        x_start = 0
        x_end = image.shape[1] - 1
        y_start = 0
        y_end = image.shape[0] - 1

        if self.x_min > 0:
            x_start = self.x_min - 1
        if self.y_min > 0:
            y_start = self.y_min - 1
        if self.x_max < image.shape[1] - 1:
            # +2 because end parameter of array slicing is taking y_end-1 as last column
            x_end = self.x_max + 2
        if self.y_max < image.shape[0] - 1:
            y_end = self.y_max + 2

        self.segment_image = image[y_start:y_end, x_start:x_end]

    def count_W3(self):
        self.W3 = self.perimeter / (2 * math.sqrt(math.pi * self.area)) - 1

    def count_distance(self, x1, x2, y1, y2):
        return np.sqrt(np.power(x2 - x1, 2) + np.power(y2 - y1, 2))

    def count_W7(self):
        weight_center_x = self.count_moment(0, 1)/self.area     #j - in pdf
        weight_center_y = self.count_moment(1, 0)/self.area     #i - in pdf

        distance_for_min = self.count_distance(
            self.x_min, weight_center_x, self.y_min, weight_center_y)
        distance_for_max = self.count_distance(
            self.x_max, weight_center_x, self.y_max, weight_center_y)

        r_min = distance_for_max if distance_for_min >= distance_for_max else distance_for_min
        r_max = distance_for_max if distance_for_min <= distance_for_max else distance_for_min

        self.W7 = r_min/r_max

    def count_moment(self, p, q):
        moment = 0

        for pixel in self.pixels:
            moment += math.pow(pixel[0], p) * math.pow(pixel[1], q)

        return moment

    def calculate_characteristics(self):
        self.count_W3()
        self.count_W7()
        m00 = self.count_moment(0, 0)
        m10 = self.count_moment(1, 0)
        m01 = self.count_moment(0, 1)
        m11 = self.count_moment(1, 1)
        m12 = self.count_moment(1, 2)
        m21 = self.count_moment(2, 1)
        m02 = self.count_moment(0, 2)
        m20 = self.count_moment(2, 0)
        m03 = self.count_moment(0, 3)
        m30 = self.count_moment(3, 0)
        weight_center_x = self.count_moment(0, 1)/self.area     #j - in pdf
        weight_center_y = self.count_moment(1, 0)/self.area     #i - in pdf

        M11 = m11 - (m10 * m01) / m00
        M20 = m20 - math.pow(m10, 2) / m00
        M02 = m02 - math.pow(m01, 2) / m00
        M21 = m21 - 2*m11*weight_center_y - m20*weight_center_x + 2*m01*math.pow(weight_center_y, 2)
        M12 = m12 - 2*m11*weight_center_x - m02*weight_center_y + 2*m10*math.pow(weight_center_x, 2)
        M30 = m30 - 3*m20*weight_center_y - 2*m10*math.pow(weight_center_y, 2)
        M03 = m03 - 3*m02*weight_center_x - 2*m01*math.pow(weight_center_x, 2)

        self.M1 = (M20 + M02) / math.pow(m00, 2)
        self.M2 = (math.pow(M20 - M02, 2) + 4*math.pow(M11, 2)) / math.pow(m00, 4)
        self.M3 = (math.pow(M30-3*M12, 2) + math.pow(3*M21-M03, 2)) / math.pow(m00, 5)
        self.M7 = (M20 * M02 - math.pow(M11, 2)) / math.pow(m00, 4)

    def mark_logo(self, image):
        FRAME_COLOR = [0, 255, 0]
        x_start = self.x_min
        x_end = self.x_max
        y_start = self.y_min
        y_end = self.y_max

        if self.x_min > 10:
            x_start = self.x_min - 10
        if self.y_min > 10:
            y_start = self.y_min - 10
        if self.x_max < image.shape[1] - 10:
            # +2 because end parameter of array slicing is taking y_end-1 as last column
            x_end = self.x_max + 11
        if self.y_max < image.shape[0] - 10:
            y_end = self.y_max + 11

        for col in range(x_start, x_end):
            image[y_start][col] = FRAME_COLOR
            image[y_start+1][col] = FRAME_COLOR
            image[y_end-1][col] = FRAME_COLOR
            image[y_end][col] = FRAME_COLOR

        for row in range(y_start, y_end):
            image[row][x_start] = FRAME_COLOR
            image[row][x_start+1] = FRAME_COLOR
            image[row][x_end-1] = FRAME_COLOR
            image[row][x_end] = FRAME_COLOR

        return image