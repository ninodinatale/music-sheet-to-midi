import enum
import math
from typing import List, Tuple

import cv2
from numpy import ndarray


class Color(enum.Enum):
    Black = 0
    White = 255


class BoundingBox(object):
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x: int = x
        self.y: int = y
        self.width: int = width
        self.height: int = height
        self.center: Tuple[
            float, float] = (self.x + self.width / 2, self.y + self.height / 2)
        self.area: int = self.width * self.height

    def overlap(self, other: 'BoundingBox') -> float:
        overlap_x: float = max(0,
                               min(self.x + self.width,
                                   other.x + other.width) - max(
                                   self.x,
                                   other.x))
        overlap_y: float = max(0,
                               min(self.y + self.height,
                                   other.y + other.height) - max(
                                   self.y,
                                   other.y))
        overlap_area: float = overlap_x * overlap_y
        return overlap_area / self.area

    def distance(self, other: 'BoundingBox') -> float:
        distance_x = self.center[0] - other.center[0]
        distance_y = self.center[1] - other.center[1]
        return math.sqrt(distance_x * distance_x + distance_y * distance_y)

    def merge(self, other: 'BoundingBox') -> 'BoundingBox':
        x: int = min(self.x, other.x)
        y: int = min(self.y, other.y)
        width: int = max(self.x + self.width, other.x + other.width) - x
        height: int = max(self.y + self.height, other.y + other.height) - y
        return BoundingBox(x, y, width, height)

    def draw(self, img, color: Tuple[int, int, int],
             thickness: int):
        cv2.rectangle(img,
                      (int(self.x), int(self.y)),
                      (int(self.x + self.width),
                       int(self.y + self.height)),
                      color,
                      thickness)

    def get_corner(self) -> Tuple[int, int]:
        return self.x, self.y


class NotationType(enum.Enum):
    Note = "note",
    Sharp = "sharp",
    Flat = "flat",
    Rest = "rest",
    EighthFlag = "eighth_flag"
    Line = "line"


class Notation(object):
    def __init__(self, type: NotationType, duration, box: BoundingBox,
                 pitch: int = -1):
        self.pitch: int = pitch
        self.duration: int = duration
        self.type: NotationType = type
        self.box: BoundingBox = box


class Bar(object):
    def __init__(self):
        self.elements: List[Notation] = []


class Clef(enum.Enum):
    Treble = "treble"
    Bass = "bass"


class TimeSignature(enum.Enum):
    Sig44 = "45"


class Instrument(enum.Enum):
    Unknown = -1


class Staff(object):
    def __init__(self, staff_matrix: List[List[int]], box: BoundingBox,
                 line_width: int, line_spacing: int,
                 staff_img: ndarray, clef: Clef = Clef.Treble,
                 time_signature=TimeSignature.Sig44,
                 instrument: Instrument = Instrument.Unknown):
        self.clef = clef
        self.time_signature = time_signature
        self.instrument = instrument
        self.line_one = staff_matrix[0]
        self.line_two = staff_matrix[1]
        self.line_three = staff_matrix[2]
        self.line_four = staff_matrix[3]
        self.line_five = staff_matrix[4]
        self.box = box
        self.img = staff_img
        self.bars = []
        self.line_width = line_width
        self.line_spacing = line_spacing

    def add_bar(self, bar):
        self.bars.append(bar)

    def get_pitch(self, note_center_y: int):
        clef_info = {
            Clef.Treble: [
                ("F5", "E5", "D5", "C5", "B4", "A4", "G4", "F4", "E4"),
                (5, 3), (4, 2)],
            Clef.Bass: [("A3", "G3", "F3", "E3", "D3", "C3", "B2", "A2", "G2"),
                        (3, 5), (2, 4)]
        }
        note_names = ["C", "D", "E", "F", "G", "A", "B"]

        # print("[get_pitch] Using {} clef".format(self.clef))

        # Check within staff first
        if note_center_y in self.line_one:
            return clef_info[self.clef][0][0]
        elif (note_center_y in list(
                range(self.line_one[-1] + 1, self.line_two[0]))):
            return clef_info[self.clef][0][1]
        elif note_center_y in self.line_two:
            return clef_info[self.clef][0][2]
        elif (note_center_y in list(
                range(self.line_two[-1] + 1, self.line_three[0]))):
            return clef_info[self.clef][0][3]
        elif note_center_y in self.line_three:
            return clef_info[self.clef][0][4]
        elif (note_center_y in list(
                range(self.line_three[-1] + 1, self.line_four[0]))):
            return clef_info[self.clef][0][5]
        elif note_center_y in self.line_four:
            return clef_info[self.clef][0][6]
        elif (note_center_y in list(
                range(self.line_four[-1] + 1, self.line_five[0]))):
            return clef_info[self.clef][0][7]
        elif note_center_y in self.line_five:
            return clef_info[self.clef][0][8]
        else:
            # print("[get_pitch] Note was not within staff")
            if note_center_y < self.line_one[0]:
                # print("[get_pitch] Note above staff ")
                # Check above staff
                line_below = self.line_one
                current_line = [pixel - self.line_spacing for pixel in
                                self.line_one]  # Go to next line above
                octave = clef_info[self.clef][1][
                    0]  # The octave number at line one
                note_index = clef_info[self.clef][1][
                    1]  # Line one's pitch has this index in note_names

                while current_line[0] > 0:
                    if note_center_y in current_line:
                        # Grab note two places above
                        octave = octave + 1 if (note_index + 2 >= 7) else octave
                        note_index = (note_index + 2) % 7
                        return note_names[note_index] + str(octave)
                    elif (note_center_y in range(current_line[-1] + 1,
                                                 line_below[0])):
                        # Grab note one place above
                        octave = octave + 1 if (note_index + 1 >= 7) else octave
                        note_index = (note_index + 1) % 7
                        return note_names[note_index] + str(octave)
                    else:
                        # Check next line above
                        octave = octave + 1 if (note_index + 2 >= 7) else octave
                        note_index = (note_index + 2) % 7
                        line_below = current_line.copy()
                        current_line = [pixel - self.line_spacing for pixel in
                                        current_line]

                assert False, "[ERROR] Note was above staff, but not found"
            elif note_center_y > self.line_five[-1]:
                # print("[get_pitch] Note below staff ")
                # Check below staff
                line_above = self.line_five
                current_line = [pixel + self.line_spacing for pixel in
                                self.line_five]  # Go to next line above
                octave = clef_info[self.clef][2][
                    0]  # The octave number at line five
                note_index = clef_info[self.clef][2][
                    1]  # Line five's pitch has this index in note_names

                while current_line[-1] < self.img.shape[0]:
                    if note_center_y in current_line:
                        # Grab note two places above
                        octave = octave - 1 if (note_index - 2 <= 7) else octave
                        note_index = (note_index - 2) % 7
                        return note_names[note_index] + str(octave)
                    elif (note_center_y in range(line_above[-1] + 1,
                                                 current_line[0])):
                        # Grab note one place above
                        octave = octave - 1 if (note_index - 1 >= 7) else octave
                        note_index = (note_index - 1) % 7
                        return note_names[note_index] + str(octave)
                    else:
                        # Check next line above
                        octave = octave - 1 if (note_index - 2 <= 7) else octave
                        note_index = (note_index - 2) % 7
                        line_above = current_line.copy()
                        current_line = [pixel + self.line_spacing for pixel in
                                        current_line]
                assert False, "[ERROR] Note was below staff, but not found"
            else:
                # Should not get here
                assert False, "[ERROR] Note was neither, within, above or below staff"
