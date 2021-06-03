# Template building parameters

TEMPLATE_DIMS = (715, 460, 3)

QRCODE_SHAPE = (130, 130)
QRCODE_BORDER_SIZE = 20
QRCODE_H_RANGE = (QRCODE_BORDER_SIZE, QRCODE_SHAPE[0] + QRCODE_BORDER_SIZE)
QRCODE_W_RANGE = (
    TEMPLATE_DIMS[1] - QRCODE_BORDER_SIZE - QRCODE_SHAPE[0],
    TEMPLATE_DIMS[1] - QRCODE_BORDER_SIZE,
)

MARKS_START_Y = QRCODE_SHAPE[0] + 2 * QRCODE_BORDER_SIZE
MARKS_DIMS = (20, 180)
MARKS_VERTICAL_SPACING_SIZE = 15
MARKS_MARGIN_SIZE = 20
MARKS_MIDDLE_COLUMN_SIZE = 60

MARK_OPTION_CHUNK_SIZE = 39
MARK_OPTION_DOT_SIZE = 26
MARK_TEMPLATE_SIZE = (20, 26)

N_QUESTIONS_PER_COLUMN = 15


## Grading parameters
QRCODE_RESIZE_VALUE = 2000
MIN_FILLED_DISTANCE = 0.25
ANSWERS = {
    1: ["A"],
    2: ["D", "E"],
    3: [],
    4: ["A"],
    5: [],
    6: [],
    7: ["A", "B"],
    8: ["D"],
    9: ["B", "C"],
    10: ["E"],
    11: ["B"],
    12: ["C", "D"],
    13: [],
    14: ["C"],
    15: ["B"],
    16: ["E"],
    17: ["B"],
    18: ["A"],
    19: ["D"],
    20: [],
    21: [],
    22: ["B"],
    23: ["E"],
    24: [],
    25: ["C"],
    26: ["A"],
    27: [],
    28: ["E"],
    29: ["E"],
    30: ["A"],
}
