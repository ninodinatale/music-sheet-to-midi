import logging
import sys
from collections import Counter
from copy import deepcopy
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from midiutil.MidiFile import MIDIFile
from numpy import ndarray

from best_match import match
from exceptions import ValidationError
from image_processing import prepare_image
from models import Staff, Notation, Bar, BoundingBox, NotationType, Color

clef_paths = {
    "treble": [
        "resources/template/clef/treble_1.jpg",
        "resources/template/clef/treble_2.jpg"
    ],
    "bass": [
        "resources/template/clef/bass_1.jpg"
    ]
}

accidental_paths = {
    "sharp": [
        "resources/template/sharp-line.png",
        "resources/template/sharp-space.png"
    ],
    "flat": [
        "resources/template/flat-line.png",
        "resources/template/flat-space.png"
    ]
}

note_paths = {
    "quarter": [
        "resources/template/note/quarter.png",
        "resources/template/note/solid-note.png"
    ],
    "half": [
        "resources/template/note/half-space.png",
        "resources/template/note/half-note-line.png",
        "resources/template/note/half-line.png",
        "resources/template/note/half-note-space.png"
    ],
    "whole": [
        "resources/template/note/whole-space.png",
        "resources/template/note/whole-note-line.png",
        "resources/template/note/whole-line.png",
        "resources/template/note/whole-note-space.png"
    ]
}
rest_paths = {
    "eighth": ["resources/template/rest/eighth_rest.jpg"],
    "quarter": ["resources/template/rest/quarter_rest.jpg"],
    "half": ["resources/template/rest/half_rest_1.jpg",
             "resources/template/rest/half_rest_2.jpg"],
    "whole": ["resources/template/rest/whole_rest.jpg"]
}

flag_paths = ["resources/template/flag/eighth_flag_1.jpg",
              "resources/template/flag/eighth_flag_2.jpg",
              "resources/template/flag/eighth_flag_3.jpg",
              "resources/template/flag/eighth_flag_4.jpg",
              "resources/template/flag/eighth_flag_5.jpg",
              "resources/template/flag/eighth_flag_6.jpg"]

barline_paths = ["resources/template/barline/barline_1.jpg",
                 "resources/template/barline/barline_2.jpg",
                 "resources/template/barline/barline_3.jpg",
                 "resources/template/barline/barline_4.jpg"]

# -------------------------------------------------------------------------------
# Template Images
# -------------------------------------------------------------------------------

# Clefs
clef_imgs = {
    "treble": [cv2.imread(clef_file, 0) for clef_file in clef_paths["treble"]],
    "bass": [cv2.imread(clef_file, 0) for clef_file in clef_paths["bass"]]
}

# Time Signatures
time_imgs = {
    "common": [cv2.imread(time, 0) for time in
               ["resources/template/time/common.jpg"]],
    "44": [cv2.imread(time, 0) for time in ["resources/template/time/44.jpg"]],
    "34": [cv2.imread(time, 0) for time in ["resources/template/time/34.jpg"]],
    "24": [cv2.imread(time, 0) for time in ["resources/template/time/24.jpg"]],
    "68": [cv2.imread(time, 0) for time in ["resources/template/time/68.jpg"]]
}

# Accidentals
sharp_imgs = [cv2.imread(sharp_files, 0) for sharp_files in
              accidental_paths["sharp"]]
flat_imgs = [cv2.imread(flat_file, 0) for flat_file in accidental_paths["flat"]]

# Notes
quarter_note_imgs = [cv2.imread(quarter, 0) for quarter in
                     note_paths["quarter"]]
half_note_imgs = [cv2.imread(half, 0) for half in note_paths["half"]]
whole_note_imgs = [cv2.imread(whole, 0) for whole in note_paths['whole']]

# Rests
eighth_rest_imgs = [cv2.imread(eighth, 0) for eighth in rest_paths["eighth"]]
quarter_rest_imgs = [cv2.imread(quarter, 0) for quarter in
                     rest_paths["quarter"]]
half_rest_imgs = [cv2.imread(half, 0) for half in rest_paths["half"]]
whole_rest_imgs = [cv2.imread(whole, 0) for whole in rest_paths['whole']]

# Eighth Flag
eighth_flag_imgs = [cv2.imread(flag, 0) for flag in flag_paths]

# Bar line
bar_imgs = [cv2.imread(barline, 0) for barline in barline_paths]

# -------------------------------------------------------------------------------
# Template Thresholds
# -------------------------------------------------------------------------------

# Clefs
clef_lower, clef_upper, clef_thresh = 50, 150, 0.88

# Time
time_lower, time_upper, time_thresh = 50, 150, 0.85

# Accidentals
sharp_lower, sharp_upper, sharp_thresh = 50, 150, 0.70
flat_lower, flat_upper, flat_thresh = 50, 150, 0.77

# Notes
quarter_note_lower, quarter_note_upper, quarter_note_thresh = 50, 150, 0.70
half_note_lower, half_note_upper, half_note_thresh = 50, 150, 0.70
whole_note_lower, whole_note_upper, whole_note_thresh = 50, 150, 0.7011

# Rests
eighth_rest_lower, eighth_rest_upper, eighth_rest_thresh = 50, 150, 0.75  # Before was 0.7
quarter_rest_lower, quarter_rest_upper, quarter_rest_thresh = 50, 150, 0.70
half_rest_lower, half_rest_upper, half_rest_thresh = 50, 150, 0.80
whole_rest_lower, whole_rest_upper, whole_rest_thresh = 50, 150, 0.80

# Eighth Flag
eighth_flag_lower, eighth_flag_upper, eighth_flag_thresh = 50, 150, 0.8

# Bar line
bar_lower, bar_upper, bar_thresh = 50, 150, 0.85

# -------------------------------------------------------------------------------
# Mapping Functions
# -------------------------------------------------------------------------------

pitch_to_MIDI = {
    "C8": 108,
    "B7": 107,
    "Bb7": 106,
    "A#7": 106,
    "A7": 105,
    "Ab7": 104,
    "G#7": 104,
    "G7": 103,
    "Gb7": 102,
    "F#7": 102,
    "F7": 101,
    "E7": 100,
    "Eb7": 99,
    "D#7": 99,
    "D7": 98,
    "Db7": 97,
    "C#7": 97,
    "C7": 96,
    "B6": 95,
    "Bb6": 94,
    "A#6": 94,
    "A6": 93,
    "Ab6": 92,
    "G#6": 92,
    "G6": 91,
    "Gb6": 90,
    "F#6": 90,
    "F6": 89,
    "E6": 88,
    "Eb6": 87,
    "D#6": 87,
    "D6": 86,
    "Db6": 85,
    "C#6": 85,
    "C6": 84,
    "B5": 83,
    "Bb5": 82,
    "A#5": 82,
    "A5": 81,
    "Ab5": 80,
    "G#5": 80,
    "G5": 79,
    "Gb5": 78,
    "F#5": 78,
    "F5": 77,
    "E5": 76,
    "Eb5": 75,
    "D#5": 75,
    "D5": 74,
    "Db5": 73,
    "C#5": 73,
    "C5": 72,
    "B4": 71,
    "Bb4": 70,
    "A#4": 70,
    "A4": 69,
    "Ab4": 68,
    "G#4": 68,
    "G4": 67,
    "Gb4": 66,
    "F#4": 66,
    "F4": 65,
    "E4": 64,
    "Eb4": 63,
    "D#4": 63,
    "D4": 62,
    "Db4": 61,
    "C#4": 61,
    "C4": 60,
    "B3": 59,
    "Bb3": 58,
    "A#3": 58,
    "A3": 57,
    "Ab3": 56,
    "G#3": 56,
    "G3": 55,
    "Gb3": 54,
    "F#3": 54,
    "F3": 53,
    "E3": 52,
    "Eb3": 51,
    "D#3": 51,
    "D3": 50,
    "Db3": 49,
    "C#3": 49,
    "C3": 48,
    "B2": 47,
    "Bb2": 46,
    "A#2": 46,
    "A2": 45,
    "Ab2": 44,
    "G#2": 44,
    "G2": 43,
    "Gb2": 42,
    "F#2": 42,
    "F2": 41,
    "E2": 40,
    "Eb2": 39,
    "D#2": 39,
    "D2": 38,
    "Db2": 37,
    "C#2": 37,
    "C2": 36,
    "B1": 35,
    "Bb1": 34,
    "A#1": 34,
    "A1": 33,
    "Ab1": 32,
    "G#1": 32,
    "G1": 31,
    "Gb1": 30,
    "F#1": 30,
    "F1": 29,
    "E1": 28,
    "Eb1": 27,
    "D#1": 27,
    "D1": 26,
    "Db1": 25,
    "C#1": 25,
    "C1": 24,
    "B0": 23,
    "Bb0": 22,
    "A#0": 22,
    "A0": 21
}

MIDI_to_pitch = {
    108: "C8",
    107: "B7",
    106: "A#7",
    105: "A7",
    104: "G#7",
    103: "G7",
    102: "F#7",
    101: "F7",
    100: "E7",
    99: "D#7",
    98: "D7",
    97: "C#7",
    96: "C7",
    95: "B6",
    94: "A#6",
    93: "A6",
    92: "G#6",
    91: "G6",
    90: "F#6",
    89: "F6",
    88: "E6",
    87: "D#6",
    86: "D6",
    85: "C#6",
    84: "C6",
    83: "B5",
    82: "A#5",
    81: "A5",
    80: "G#5",
    79: "G5",
    78: "F#5",
    77: "F5",
    76: "E5",
    75: "D#5",
    74: "D5",
    73: "C#5",
    72: "C5",
    71: "B4",
    70: "A#4",
    69: "A4",
    68: "G#4",
    67: "G4",
    66: "F#4",
    65: "F4",
    64: "E4",
    63: "D#4",
    62: "D4",
    61: "C#4",
    60: "C4",
    59: "B3",
    58: "A#3",
    57: "A3",
    56: "G#3",
    55: "G3",
    54: "F#3",
    53: "F3",
    52: "E3",
    51: "D#3",
    50: "D3",
    49: "C#3",
    48: "C3",
    47: "B2",
    46: "A#2",
    45: "A2",
    44: "G#2",
    43: "G2",
    42: "F#2",
    41: "F2",
    40: "E2",
    39: "D#2",
    38: "D2",
    37: "C#2",
    36: "C2",
    35: "B1",
    34: "A#1",
    33: "A1",
    32: "G#1",
    31: "G1",
    30: "F#1",
    29: "F1",
    28: "E1",
    27: "D#1",
    26: "D1",
    25: "C#1",
    24: "C1",
    23: "B0",
    22: "A#0",
    21: "A0"
}

key_signature_changes = {
    "sharp": ["", "F", "FC", "FCG", "FCGD", "FCGDA", "FCGDAE", "FCGDAEB"],
    "flat": ["", "B", "BE", "BEA", "BEAD", "BEADG", "BEADGC", "BEADGCF"]
}


# -------------------------------------------------------------------------------
# General Functions
# -------------------------------------------------------------------------------

def deskew(img):
    skew_img = cv2.bitwise_not(img)  # Invert image

    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(skew_img > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)

    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

    # rotate the image to deskew it
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    return angle, rotated


def get_line_dimensions(img: ndarray) -> Tuple[int, int]:
    """
    Calculates and gets the line height and the line spacing of lines
    in a staff using run length encoding https://en.wikipedia.org/wiki/Run-length_encoding.
    :param img: The image where the staff and thus the lines are in.
    :return: The line height and their spacing as a tuple.
    """
    num_rows: int = img.shape[0]  # height
    num_cols: int = img.shape[1]  # width
    image_white_points: List[int] = []
    image_black_points: List[int] = []
    consecutive_points: List[int] = []

    for i in range(num_cols):
        column: List[Color] = img[:, i]
        rle_col = []
        colors = {
            Color.White.value: [],
            Color.Black.value: []
        }
        consecutive_px_of_same_value = 0
        current_color: Color = column[
            0]  # Either 0 (black) or 255 (white). Should be white initially
        for j in range(num_rows):
            if column[j] == current_color:
                consecutive_px_of_same_value += 1
            else:
                # add previous run length to rle encoding
                rle_col.append(consecutive_px_of_same_value)

                colors[current_color].append(consecutive_px_of_same_value)

                # alternate run type
                current_color = column[j]
                # increment consecutive_px_of_same_value for new value
                consecutive_px_of_same_value = 1

        # add final run length to encoding
        rle_col.append(consecutive_px_of_same_value)
        colors[current_color].append(consecutive_px_of_same_value)

        # Calculate sum of consecutive vertical runs
        sum_rle_col = [sum(rle_col[i: i + 2]) for i in range(len(rle_col))]

        # Add to column accumulation list
        image_white_points.extend(colors[Color.White.value])
        image_black_points.extend(colors[Color.Black.value])
        consecutive_points.extend(sum_rle_col)

    white_points = Counter(image_white_points)
    black_points = Counter(image_black_points)
    black_white_sum = Counter(consecutive_points)

    line_spacing = white_points.most_common(1)[0][0]
    line_height = black_points.most_common(1)[0][0]
    height_and_spacing_sum = black_white_sum.most_common(1)[0][0]

    if line_spacing + line_height != height_and_spacing_sum:
        raise ValidationError(
            message="Sum of line height and line spacing is not equal to the "
                    "most common sum of white and black points.")

    return line_height, line_spacing


def find_staff_line_rows(img, line_width, line_spacing):
    num_rows: int = img.shape[0]  # height
    num_cols: int = img.shape[1]  # width
    row_black_pixel_histogram = []

    # determine number of black pixels in each row
    for i in range(num_rows):
        row = img[i]
        num_black_pixels = 0
        for j in range(len(row)):
            if row[j] == 0:
                num_black_pixels += 1

        row_black_pixel_histogram.append(num_black_pixels)

    all_staff_row_indices = []
    num_staff_lines = 5
    threshold = 0.4
    staff_length = num_staff_lines * (line_width + line_spacing) - line_spacing
    iter_range = num_rows - staff_length + 1

    current_row = 0
    while current_row < iter_range:
        staff_lines = [row_black_pixel_histogram[j: j + line_width] for j in
                       range(current_row,
                             current_row + (num_staff_lines - 1) * (
                                     line_width + line_spacing) + 1,
                             line_width + line_spacing)]

        for line in staff_lines:
            if sum(line) / line_width < threshold * num_cols:
                current_row += 1
                break
        else:
            staff_row_indices = [list(range(j, j + line_width)) for j in
                                 range(current_row,
                                       current_row + (num_staff_lines - 1) * (
                                               line_width + line_spacing) + 1,
                                       line_width + line_spacing)]
            all_staff_row_indices.append(staff_row_indices)
            current_row = current_row + staff_length

    return all_staff_row_indices


def find_staff_line_columns(img, all_staff_line_vertical_indices, line_width):
    num_cols = img.shape[1]  # Image Width (number of columns)
    all_staff_extremes = []

    # Find start of staff for every staff in piece
    for i in range(len(all_staff_line_vertical_indices)):
        begin_list = []  # Stores possible beginning column indices for staff
        end_list = []  # Stores possible end column indices for staff
        begin = 0
        end = num_cols - 1

        # Find staff beginning
        for j in range(num_cols // 2):
            first_staff_rows_isolated = img[
                                        all_staff_line_vertical_indices[i][0][
                                            0]:
                                        all_staff_line_vertical_indices[i][4][
                                            line_width - 1], j]
            num_black_pixels = len(
                list(filter(lambda x: x == 0, first_staff_rows_isolated)))

            if num_black_pixels == 0:
                begin_list.append(j)

        # find maximum column that has no black pixels in staff window
        list.sort(begin_list, reverse=True)
        begin = begin_list[0]

        # find staff beginning
        for j in range(num_cols // 2, num_cols):
            first_staff_rows_isolated = img[
                                        all_staff_line_vertical_indices[i][0][
                                            0]:
                                        all_staff_line_vertical_indices[i][4][
                                            line_width - 1], j]
            num_black_pixels = len(
                list(filter(lambda x: x == 0, first_staff_rows_isolated)))

            if num_black_pixels == 0:
                end_list.append(j)

        list.sort(end_list)
        end = end_list[0]

        staff_extremes = (begin, end)
        all_staff_extremes.append(staff_extremes)

    return all_staff_extremes


def remove_staff_lines(img, all_staffline_vertical_indices):
    no_staff_img = deepcopy(img)
    for staff in all_staffline_vertical_indices:
        for line in staff:
            for row in line:
                # Remove top and bottom line to be sure
                no_staff_img[row - 1, :] = 255
                no_staff_img[row, :] = 255
                no_staff_img[row + 1, :] = 255

    return no_staff_img


def open_file(path):
    img = Image.open(path)
    img.show()


def locate_templates(img, templates, start, stop, threshold) -> List[
    List[BoundingBox]]:
    locations, scale = match(img, templates, start, stop, threshold)
    img_locations: List[List[BoundingBox]] = []
    for i in range(len(templates)):
        w, h = templates[i].shape[::-1]
        w *= scale
        h *= scale
        img_locations.append([BoundingBox(pt[0], pt[1], w, h) for pt in
                              zip(*locations[i][::-1])])
    return img_locations


def merge_boxes(boxes: List[BoundingBox], threshold) -> List[BoundingBox]:
    filtered_boxes: List[BoundingBox] = []
    while len(boxes) > 0:
        r: BoundingBox = boxes.pop(0)
        boxes.sort(key=lambda box: box.distance(r))
        merged = True
        while merged:
            merged = False
            i = 0
            for _ in range(len(boxes)):
                if r.overlap(boxes[i]) > threshold or boxes[i].overlap(
                        r) > threshold:
                    r = r.merge(boxes.pop(i))
                    merged = True
                elif boxes[i].distance(r) > r.width / 2 + boxes[i].width / 2:
                    break
                else:
                    i += 1
        filtered_boxes.append(r)
    return filtered_boxes


if __name__ == "__main__":

    # todo validate file name
    file_name = sys.argv[1:][0]
    img: ndarray = prepare_image(file_name)

    line_width, line_spacing = get_line_dimensions(img)

    logging.info("staff line width: %s", line_width)
    logging.info("staff line spacing: %s", line_spacing)

    all_staffline_vertical_indices = find_staff_line_rows(img, line_width,
                                                          line_spacing)
    logging.info("%s sets of staff lines found line spacing: %s",
                 len(all_staffline_vertical_indices))

    all_staffline_horizontal_indices = find_staff_line_columns(img,
                                                               all_staffline_vertical_indices,
                                                               line_width)
    staffs = []
    half_dist_between_staffs = (all_staffline_vertical_indices[1][0][0] -
                                all_staffline_vertical_indices[0][4][
                                    line_width - 1]) // 2

    for i in range(len(all_staffline_vertical_indices)):
        x = all_staffline_horizontal_indices[i][0]
        y = all_staffline_vertical_indices[i][0][0]
        width = all_staffline_horizontal_indices[i][1] - x
        height = all_staffline_vertical_indices[i][4][line_width - 1] - y
        staff_box = BoundingBox(x, y, width, height)

        staff_img = img[max(0, y - half_dist_between_staffs): min(
            y + height + half_dist_between_staffs, img.shape[0] - 1),
                    x:x + width]

        pixel = half_dist_between_staffs
        normalized_staff_line_vertical_indices = []

        for j in range(5):
            line = []
            for k in range(line_width):
                line.append(pixel)
                pixel += 1
            normalized_staff_line_vertical_indices.append(line)
            pixel += line_spacing + 1

        staff = Staff(normalized_staff_line_vertical_indices, staff_box,
                      line_width, line_spacing, staff_img)
        staffs.append(staff)

    staff_boxes_img = img.copy()
    staff_boxes_img = cv2.cvtColor(staff_boxes_img, cv2.COLOR_GRAY2RGB)
    red = (0, 0, 255)
    box_thickness = 2
    for staff in staffs:
        box = staff.box
        box.draw(staff_boxes_img, red, box_thickness)
        x = int(box.get_corner()[0] + (box.width // 2))
        y = int(box.get_corner()[1] + box.height + 35)
        cv2.putText(staff_boxes_img, "Staff", (x, y), cv2.FONT_HERSHEY_DUPLEX,
                    0.9, red)

    cv2.imwrite('output/staffs.jpg', staff_boxes_img)
    logging.info("Saved detected staffs to output/staffs.jpg")

    staff_imgs_color = []

    for i in range(len(staffs)):
        red = (0, 0, 255)
        box_thickness = 2
        staff_img = staffs[i].img
        staff_img_color = staff_img.copy()
        staff_img_color = cv2.cvtColor(staff_img_color, cv2.COLOR_GRAY2RGB)

        for clef in clef_imgs:
            logging.info("Locating templates for clef template %s for staff %s",
                         clef, i + 1)

            clef_boxes = locate_templates(staff_img, clef_imgs[clef],
                                          clef_lower, clef_upper, clef_thresh)
            clef_boxes = merge_boxes([j for i in clef_boxes for j in i], 0.5)

            if len(clef_boxes) == 1:
                logging.info("Found clef %s for staff %s", clef, i + 1)
                staffs[i].clef = clef

                clef_boxes_img = staffs[i].img
                clef_boxes_img = clef_boxes_img.copy()

                for boxes in clef_boxes:
                    boxes.draw(staff_img_color, red, box_thickness)
                    x = int(boxes.get_corner()[0] + (boxes.width // 2))
                    y = int(boxes.get_corner()[1] + boxes.height + 10)
                    cv2.putText(staff_img_color, "{} clef".format(clef), (x, y),
                                cv2.FONT_HERSHEY_DUPLEX, 0.9, red)
                break

        else:
            logging.warning("No clef found for staff %s", i + 1)

        for time in time_imgs:
            logging.info(
                "Locating templates for time signature %s for staff %s", time,
                i + 1)

            time_boxes = locate_templates(staff_img, time_imgs[time],
                                          time_lower, time_upper, time_thresh)
            time_boxes = merge_boxes([j for i in time_boxes for j in i], 0.5)

            if len(time_boxes) == 1:
                logging.info("Found time signature %s for staff %s", time,
                             i + 1)

                staffs[i].time_signature = time

                for boxes in time_boxes:
                    boxes.draw(staff_img_color, red, box_thickness)
                    x = int(boxes.get_corner()[0] - (boxes.width // 2))
                    y = int(boxes.get_corner()[1] + boxes.height + 20)
                    cv2.putText(staff_img_color, "{} time".format(time), (x, y),
                                cv2.FONT_HERSHEY_DUPLEX, 0.9, red)
                break

            elif len(time_boxes) == 0 and i > 0:
                # Take time signature of previous staff
                previousTime = staffs[i - 1].time_signature
                staffs[i].time_signature = previousTime

                logging.info(
                    "No time signature found for staff %s for staff %s "
                    "- using time signature from previous staff line: ", i + 1,
                    previousTime)
                break
        else:
            logging.warning("No time signature available for staff %s", i + 1)

        staff_imgs_color.append(staff_img_color)

    for i in range(len(staffs)):
        logging.info("Finding Elements on Staff ", i + 1)
        staff_elements = []
        staff_img = staffs[i].img
        staff_img_color = staff_imgs_color[i]
        red = (0, 0, 255)
        box_thickness = 2

        logging.info("Matching sharp accidental template...")
        sharp_boxes: List[List[BoundingBox]] = locate_templates(staff_img,
                                                                sharp_imgs,
                                                                sharp_lower,
                                                                sharp_upper,
                                                                sharp_thresh)
        sharp_boxes: List[BoundingBox] = merge_boxes(
            [j for i in sharp_boxes for j in i], 0.5)

        logging.info("Displaying Matching Results on staff", i + 1)
        for box in sharp_boxes:
            box.draw(staff_img_color, red, box_thickness)
            text = "sharp"
            font = cv2.FONT_HERSHEY_DUPLEX
            textsize = cv2.getTextSize(text, font, fontScale=0.7, thickness=1)[
                0]
            x = int(box.get_corner()[0] - (textsize[0] // 2))
            y = int(box.get_corner()[1] + box.height + 20)
            cv2.putText(staff_img_color, text, (x, y), font, fontScale=0.7,
                        color=red, thickness=1)
            sharp = Notation(NotationType.Sharp, 0, box)
            staff_elements.append(sharp)

        logging.info("Matching flat accidental template...")
        flat_boxes = locate_templates(staff_img, flat_imgs, flat_lower,
                                      flat_upper, flat_thresh)
        flat_boxes = merge_boxes([j for i in flat_boxes for j in i], 0.5)

        logging.info("Displaying Matching Results on staff", i + 1)
        for box in flat_boxes:
            box.draw(staff_img_color, red, box_thickness)
            text = "flat"
            font = cv2.FONT_HERSHEY_DUPLEX
            textsize = cv2.getTextSize(text, font, fontScale=0.7, thickness=1)[
                0]
            x = int(box.get_corner()[0] - (textsize[0] // 2))
            y = int(box.get_corner()[1] + box.height + 20)
            cv2.putText(staff_img_color, text, (x, y), font, fontScale=0.7,
                        color=red, thickness=1)
            flat = Notation(NotationType.Flat, 0, box)
            staff_elements.append(flat)

        logging.info("Matching quarter note template...")
        quarter_boxes = locate_templates(staff_img, quarter_note_imgs,
                                         quarter_note_lower, quarter_note_upper,
                                         quarter_note_thresh)
        quarter_boxes = merge_boxes([j for i in quarter_boxes for j in i], 0.5)

        logging.info("Displaying Matching Results on staff", i + 1)
        for box in quarter_boxes:
            box.draw(staff_img_color, red, box_thickness)
            text = "1/4 note"
            font = cv2.FONT_HERSHEY_DUPLEX
            textsize = cv2.getTextSize(text, font, fontScale=0.7, thickness=1)[
                0]
            x = int(box.get_corner()[0] - (textsize[0] // 2))
            y = int(box.get_corner()[1] + box.height + 20)
            cv2.putText(staff_img_color, text, (x, y), font, fontScale=0.7,
                        color=red, thickness=1)
            pitch = staffs[i].get_pitch(round(box.center[1]))
            quarter = Notation(NotationType.Note, 1, box, pitch)
            staff_elements.append(quarter)

        logging.info("Matching half note template...")
        half_boxes = locate_templates(staff_img, half_note_imgs,
                                      half_note_lower, half_note_upper,
                                      half_note_thresh)
        half_boxes = merge_boxes([j for i in half_boxes for j in i], 0.5)

        logging.info("Displaying Matching Results on staff", i + 1)
        for box in half_boxes:
            box.draw(staff_img_color, red, box_thickness)
            text = "1/2 note"
            font = cv2.FONT_HERSHEY_DUPLEX
            textsize = cv2.getTextSize(text, font, fontScale=0.7, thickness=1)[
                0]
            x = int(box.get_corner()[0] - (textsize[0] // 2))
            y = int(box.get_corner()[1] + box.height + 20)
            cv2.putText(staff_img_color, text, (x, y), font, fontScale=0.7,
                        color=red, thickness=1)
            pitch = staffs[i].get_pitch(round(box.center[1]))
            half = Notation(NotationType.Note, 2, box, pitch)
            staff_elements.append(half)

        logging.info("Matching whole note template...")
        whole_boxes = locate_templates(staff_img, whole_note_imgs,
                                       whole_note_lower, whole_note_upper,
                                       whole_note_thresh)
        whole_boxes = merge_boxes([j for i in whole_boxes for j in i], 0.5)

        logging.info("Displaying Matching Results on staff", i + 1)
        for box in whole_boxes:
            box.draw(staff_img_color, red, box_thickness)
            text = "1 note"
            font = cv2.FONT_HERSHEY_DUPLEX
            textsize = cv2.getTextSize(text, font, fontScale=0.7, thickness=1)[
                0]
            x = int(box.get_corner()[0] - (textsize[0] // 2))
            y = int(box.get_corner()[1] + box.height + 20)
            cv2.putText(staff_img_color, text, (x, y), font, fontScale=0.7,
                        color=red, thickness=1)
            pitch = staffs[i].get_pitch(round(box.center[1]))
            whole = Notation(NotationType.Note, 4, box, pitch)
            staff_elements.append(whole)

        logging.info("Matching eighth rest template...")
        eighth_boxes = locate_templates(staff_img, eighth_rest_imgs,
                                        eighth_rest_lower, eighth_rest_upper,
                                        eighth_rest_thresh)
        eighth_boxes = merge_boxes([j for i in eighth_boxes for j in i], 0.5)

        logging.info("Displaying Matching Results on staff", i + 1)
        for box in eighth_boxes:
            box.draw(staff_img_color, red, box_thickness)
            text = "1/8 rest"
            font = cv2.FONT_HERSHEY_DUPLEX
            textsize = cv2.getTextSize(text, font, fontScale=0.7, thickness=1)[
                0]
            x = int(box.get_corner()[0] - (textsize[0] // 2))
            y = int(box.get_corner()[1] + box.height + 20)
            cv2.putText(staff_img_color, text, (x, y), font, fontScale=0.7,
                        color=red, thickness=1)
            eighth = Notation(NotationType.Rest, 0.5, box)
            staff_elements.append(eighth)

        logging.info("Matching quarter rest template...")
        quarter_boxes = locate_templates(staff_img, quarter_rest_imgs,
                                         quarter_rest_lower, quarter_rest_upper,
                                         quarter_rest_thresh)
        quarter_boxes = merge_boxes([j for i in quarter_boxes for j in i], 0.5)

        logging.info("Displaying Matching Results on staff", i + 1)
        for box in quarter_boxes:
            box.draw(staff_img_color, red, box_thickness)
            text = "1/4 rest"
            font = cv2.FONT_HERSHEY_DUPLEX
            textsize = cv2.getTextSize(text, font, fontScale=0.7, thickness=1)[
                0]
            x = int(box.get_corner()[0] - (textsize[0] // 2))
            y = int(box.get_corner()[1] + box.height + 20)
            cv2.putText(staff_img_color, text, (x, y), font, fontScale=0.7,
                        color=red, thickness=1)
            quarter = Notation(NotationType.Rest, 1, box)
            staff_elements.append(quarter)

        logging.info("Matching half rest template...")
        half_boxes = locate_templates(staff_img, half_rest_imgs,
                                      half_rest_lower, half_rest_upper,
                                      half_rest_thresh)
        half_boxes = merge_boxes([j for i in half_boxes for j in i], 0.5)

        logging.info("Displaying Matching Results on staff", i + 1)
        for box in half_boxes:
            box.draw(staff_img_color, red, box_thickness)
            text = "1/2 rest"
            font = cv2.FONT_HERSHEY_DUPLEX
            textsize = cv2.getTextSize(text, font, fontScale=0.7, thickness=1)[
                0]
            x = int(box.get_corner()[0] - (textsize[0] // 2))
            y = int(box.get_corner()[1] + box.height + 20)
            cv2.putText(staff_img_color, text, (x, y), font, fontScale=0.7,
                        color=red, thickness=1)
            half = Notation(NotationType.Rest, 2, box)
            staff_elements.append(half)

        logging.info("Matching whole rest template...")
        whole_boxes = locate_templates(staff_img, whole_rest_imgs,
                                       whole_rest_lower, whole_rest_upper,
                                       whole_rest_thresh)
        whole_boxes = merge_boxes([j for i in whole_boxes for j in i], 0.5)

        logging.info("Displaying Matching Results on staff", i + 1)
        for box in whole_boxes:
            box.draw(staff_img_color, red, box_thickness)
            text = "1 rest"
            font = cv2.FONT_HERSHEY_DUPLEX
            textsize = cv2.getTextSize(text, font, fontScale=0.7, thickness=1)[
                0]
            x = int(box.get_corner()[0] - (textsize[0] // 2))
            y = int(box.get_corner()[1] + box.height + 20)
            cv2.putText(staff_img_color, text, (x, y), font, fontScale=0.7,
                        color=red, thickness=1)
            whole = Notation(NotationType.Rest, 4, box)
            staff_elements.append(whole)

        logging.info("Matching eighth flag template...")
        flag_boxes = locate_templates(staff_img, eighth_flag_imgs,
                                      eighth_flag_lower, eighth_flag_upper,
                                      eighth_flag_thresh)
        flag_boxes = merge_boxes([j for i in flag_boxes for j in i], 0.5)

        logging.info("Displaying Matching Results on staff", i + 1)

        for box in flag_boxes:
            box.draw(staff_img_color, red, box_thickness)
            text = "1/8 flag"
            font = cv2.FONT_HERSHEY_DUPLEX
            textsize = cv2.getTextSize(text, font, fontScale=0.7, thickness=1)[
                0]
            x = int(box.get_corner()[0] - (textsize[0] // 2))
            y = int(box.get_corner()[1] + box.height + 20)
            cv2.putText(staff_img_color, text, (x, y), font, fontScale=0.7,
                        color=red, thickness=1)
            flag = Notation(NotationType.EighthFlag, 0, box)
            staff_elements.append(flag)

        logging.info("Matching bar line template...")
        bar_boxes = locate_templates(staff_img, bar_imgs, bar_lower, bar_upper,
                                     bar_thresh)
        bar_boxes = merge_boxes([j for i in bar_boxes for j in i], 0.5)

        logging.info("Displaying Matching Results on staff", i + 1)
        for box in bar_boxes:
            box.draw(staff_img_color, red, box_thickness)
            text = "line"
            font = cv2.FONT_HERSHEY_DUPLEX
            textsize = cv2.getTextSize(text, font, fontScale=0.7, thickness=1)[
                0]
            x = int(box.get_corner()[0] - (textsize[0] // 2))
            y = int(box.get_corner()[1] + box.height + 20)
            cv2.putText(staff_img_color, text, (x, y), font, fontScale=0.7,
                        color=red, thickness=1)
            line = Notation(NotationType.Line, 0, box)
            staff_elements.append(line)

        logging.info("Saving detected elements in staff {} onto disk".format(
            i + 1))
        cv2.imwrite("output/staff_{}.jpg".format(i + 1),
                    staff_img_color)

        staff_elements.sort(
            key=lambda element: element.box.center)

        logging.info("Staff elements sorted in time")
        eighth_flag_indices = []
        for j in range(len(staff_elements)):

            if staff_elements[j].type == NotationType.EighthFlag:
                # Find all eighth flags
                eighth_flag_indices.append(j)

            if staff_elements[j].type == NotationType.Note:
                logging.info(staff_elements[j].pitch, end=", ")
            else:
                logging.info(staff_elements[j].type, end=", ")

        logging.info("\n")

        logging.info("Correcting for misclassified eighth notes")
        for j in eighth_flag_indices:

            distances = []
            distance = staff_elements[j].box.distance(
                staff_elements[j - 1].box)
            distances.append(distance)
            if j + 1 < len(staff_elements):
                distance = staff_elements[j].box.distance(
                    staff_elements[j + 1].box)
                distances.append(distance)

            if distances[1] and distances[0] > distances[1]:
                staff_elements[j + 1].duration = 0.5
            else:
                staff_elements[j - 1].duration = 0.5

            logging.info(
                "Notation {} was a eighth note misclassified as a quarter note".format(
                    j + 1))
            del staff_elements[j]

        for j in range(len(staff_elements)):
            if (j + 1 < len(staff_elements)
                    and staff_elements[j].type == NotationType.Note
                    and staff_elements[j + 1].type == NotationType.Note
                    and (staff_elements[j].duration == 1 or
                         staff_elements[j].duration == 0.5)
                    and staff_elements[j + 1].duration == 1):

                # Notes of interest
                note_1_center_x = staff_elements[j].box.center[0]
                note_2_center_x = staff_elements[j + 1].box.center[
                    0]

                # Regular number of black pixels in staff column
                num_black_pixels = 5 * staffs[i].line_width

                # Actual number of black pixels in mid column
                center_column = (note_2_center_x - note_1_center_x) // 2
                mid_col = staff_img[:, int(note_1_center_x + center_column)]
                num_black_pixels_mid = len(np.where(mid_col == 0)[0])

                if num_black_pixels_mid > num_black_pixels:
                    staff_elements[j].duration = 0.5
                    staff_elements[j + 1].duration = 0.5
                    logging.info(
                        "Notation {} and {} were eighth notes misclassified as quarter notes".format(
                            j + 1, j + 2))

        logging.info("Applying key signature note value changes")
        num_sharps = 0
        num_flats = 0
        j = 0
        while (staff_elements[j].duration == 0):
            accidental = staff_elements[j].type
            if (accidental == "sharp"):
                num_sharps += 1
                j += 1

            elif (accidental == "flat"):
                num_flats += 1
                j += 1

        if j != 0:
            max_accidental_offset_x = staff_elements[j].box.center[
                                          0] - staff_elements[
                                          j].box.width
            accidental_center_x = staff_elements[j - 1].box.center[
                0]
            accidental_type = staff_elements[j - 1].type

            if accidental_center_x > max_accidental_offset_x:
                logging.info("Last accidental belongs to first note")
                num_sharps = num_sharps - 1 if accidental_type == "sharp" else num_sharps
                num_flats = num_flats - 1 if accidental_type == "flat" else num_flats

            notes_to_modify = []
            if accidental_type == "sharp":
                logging.info("Key signature has {} sharp accidentals: ".format(
                    num_sharps))
                notes_to_modify = key_signature_changes[accidental_type][
                    num_sharps]
                staff_elements = staff_elements[num_sharps:]
            else:
                logging.info("Key signature has {} flat accidentals: ".format(
                    num_flats))
                notes_to_modify = key_signature_changes[accidental_type][
                    num_flats]
                staff_elements = staff_elements[num_flats:]

            logging.info("Corrected note values after key signature: ")
            for element in staff_elements:
                type = element.type
                note = element.pitch
                if type == "note" and note[0] in notes_to_modify:
                    new_note = MIDI_to_pitch[pitch_to_MIDI[
                                                 note] + 1] if accidental_type == "sharp" else \
                        MIDI_to_pitch[pitch_to_MIDI[note] - 1]
                    element.pitch = new_note

                if element.type == NotationType.Note:
                    logging.info(element.pitch, end=", ")
                else:
                    logging.info(element.type, end=", ")

            logging.info("\n")

        logging.info("Applying any accidental to neighboring note")
        element_indices_to_remove = []
        for j in range(len(staff_elements)):
            accidental_type = staff_elements[j].type

            if accidental_type == "flat" or accidental_type == "sharp":
                max_accidental_offset_x = \
                    staff_elements[j + 1].box.center[0] - \
                    staff_elements[j + 1].box.width
                accidental_center_x = staff_elements[j].box.center[
                    0]
                element_type = staff_elements[j + 1].type

                if (
                        accidental_center_x > max_accidental_offset_x and element_type == "note"):
                    logging.info("Notation has accidental associated with it")
                    note = staff_elements[j + 1].pitch
                    new_note = MIDI_to_pitch[pitch_to_MIDI[
                                                 note] + 1] if accidental_type == "sharp" else \
                        MIDI_to_pitch[pitch_to_MIDI[note] - 1]
                    staff_elements[j + 1].pitch = new_note
                    element_indices_to_remove.append(i)

        for j in element_indices_to_remove:
            del staff_elements[j]

        logging.info("Corrected note values after accidentals: ")
        for j in range(len(staff_elements)):
            if staff_elements[j].type == NotationType.Note:
                logging.info(staff_elements[j].pitch, end=", ")
            else:
                logging.info(staff_elements[j].type, end=", ")

        logging.info("\n")

        logging.info("Assembling current staff")
        bar = Bar()
        while len(staff_elements) > 0:
            element = staff_elements.pop(0)

            if element.type != NotationType.Line:
                bar.elements.append(element)
            else:
                staffs[i].add_bar(bar)
                bar = Bar()
        staffs[i].add_bar(bar)

    logging.info("Sequencing MIDI")
    midi = MIDIFile(1)
    track = 0
    time = 0
    channel = 0
    volume = 100

    midi.addTrackName(track, time, "Track")
    midi.addTempo(track, time, 110)

    for i in range(len(staffs)):
        logging.info("==== Staff {} ====".format(i + 1))
        bars = staffs[i].bars
        for j in range(len(bars)):
            logging.info("--- Bar {} ---".format(j + 1))
            elements = bars[j].elements
            for k in range(len(elements)):
                duration = elements[k].duration
                if elements[k].type == NotationType.Note:
                    pitch = pitch_to_MIDI[elements[k].pitch]
                    midi.addNote(track, channel, pitch, time, duration, volume)
                logging.info(elements[k].type)
                logging.info(elements[k].pitch)
                logging.info(elements[k].duration)
                logging.info("-----")
                time += duration

    logging.info("Writing MIDI to disk")
    binfile = open("output/output.mid", 'wb')
    midi.writeFile(binfile)
    binfile.close()
