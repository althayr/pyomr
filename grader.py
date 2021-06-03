import cv2
import config
import qrcode
import logging
import argparse
import traceback
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from pyzbar import pyzbar
from skimage.metrics import structural_similarity

logging.getLogger("pyzbar").setLevel("ERROR")

FILLED_ANSWER_IMAGE = cv2.imread(
    "./images/template_mark_options/filled_dot.png", cv2.IMREAD_GRAYSCALE
)
EMPTY_ANSWER_IMAGE = cv2.imread(
    "./images/template_mark_options/empty.png", cv2.IMREAD_GRAYSCALE
)
BLANK_ANSWER_IMAGE_MAP = {
    1: cv2.imread("./images/template_mark_options/blank_a.png", cv2.IMREAD_GRAYSCALE),
    2: cv2.imread("./images/template_mark_options/blank_b.png", cv2.IMREAD_GRAYSCALE),
    3: cv2.imread("./images/template_mark_options/blank_c.png", cv2.IMREAD_GRAYSCALE),
    4: cv2.imread("./images/template_mark_options/blank_d.png", cv2.IMREAD_GRAYSCALE),
    5: cv2.imread("./images/template_mark_options/blank_e.png", cv2.IMREAD_GRAYSCALE),
}
ANSWER_TO_LETTER_MAP = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}


def get_qrcode_info(img):
    resize_dim = config.QRCODE_RESIZE_VALUE
    h_ratio, w_ratio = resize_dim / img.shape[0], resize_dim / img.shape[1]
    img = cv2.resize(img, (resize_dim, resize_dim))
    barcodes = pyzbar.decode(img, symbols=[pyzbar.ZBarSymbol.QRCODE])
    if not barcodes:
        raise ValueError("QRCode not found!")
    qr_codes = [x for x in barcodes if x.type == "QRCODE"]
    if not qr_codes:
        raise ValueError("QRCode not found!")
    qr_code = barcodes[0]
    qr_coords = np.array(
        [[p.x / w_ratio, p.y / h_ratio] for p in qr_code.polygon], np.int32
    )
    return qr_code.data, qr_coords


def outline_qrcode(img, qr_coords):
    img2 = cv2.drawContours(img, [qr_coords], 0, (0, 255, 0), 5)
    plot_bgr(img2)


def four_point_transform(img, points):
    # Directly from https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    # unpack rectangle points: top left, top right, bottom right, bottom left
    #     hh, ww = img.shape[:2]
    ww, hh = config.TEMPLATE_DIMS[1], config.TEMPLATE_DIMS[0]
    #     ww, hh = size, size
    source = np.array(points, dtype="float32")
    # destination points which will be used to map the screen to a "scanned" view

    dst = np.array([[ww, hh], [0, hh], [0, 0], [ww, 0]], dtype="float32")

    # print(dst)
    # calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(points, dst)
    # warp the perspective to grab the screen
    return cv2.warpPerspective(
        img,
        M,
        (ww, hh),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def find_contours(gray):
    # To remove excess gray and make edges sharper against background
    # Improves contour detection
    _, gray = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(
        gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours, hierarchy


def calculate_contour_features(contour):
    moments = cv2.moments(contour)
    return cv2.HuMoments(moments)


def calculate_corner_features():
    corner_img = cv2.imread("./images/corner_contour_base.png")
    corner_img_gray = cv2.cvtColor(corner_img, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(
        corner_img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) != 2:
        raise RuntimeError(
            "Did not find the expected contours when looking for the corner"
        )

    corner_contour = next(
        ct for i, ct in enumerate(contours) if hierarchy[0][i][3] != -1
    )

    return calculate_contour_features(corner_contour)


def l2_distance(p1, p2):
    "Euclidean distance between the two Hu moment feature vectors"
    return np.linalg.norm(np.array(p1) - np.array(p2))


def find_corner_contours(contours):
    corner_features = calculate_corner_features()
    # Sort contours by similarity with our template corner
    sorted_contours = sorted(
        contours,
        key=lambda c: l2_distance(corner_features, calculate_contour_features(c)),
    )

    # We can use a area-based heuristic to find the four corners,
    # Premises:
    # - The first contour is always correct
    # - Other corner elements must be on the 10 most similar contours
    # - The following ones should not have a area and perimeter different by more than
    #   50% of the average area and perimeter of the previous found corners
    #   (area can differ by a good margin depending on perspective distortion)
    corner_contours = [sorted_contours[0]]
    avg_area = cv2.contourArea(sorted_contours[0])
    avg_perimeter = cv2.arcLength(sorted_contours[0], True)
    for i, cnt in enumerate(sorted_contours[1:10]):
        if len(corner_contours) == 4:
            break

        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        area_diff = abs((area - avg_area) / avg_area)
        perimeter_diff = abs((perimeter - avg_perimeter) / avg_perimeter)
        if not area_diff < 0.5:
            continue
        if not perimeter_diff < 0.5:
            continue

        corner_contours.append(cnt)
        avg_area = np.mean([cv2.contourArea(x) for x in corner_contours])
        avg_perimeter = np.mean([cv2.arcLength(x, True) for x in corner_contours])

    assert len(corner_contours) == 4
    return corner_contours


def get_centroid(contour):
    m = cv2.moments(contour)
    x = int(m["m10"] / m["m00"])
    y = int(m["m01"] / m["m00"])
    return (x, y)


def sort_rectangle_based_on_qrcode_distance(points, qrcode_coords):
    qrcode_centroid = get_centroid(qrcode_coords)
    distances = [l2_distance(point, qrcode_centroid) for point in points]
    sorted_points = [x for x, _ in sorted(zip(points, distances), key=lambda x: x[1])]
    tr, tl, br, bl = sorted_points
    return [br, bl, tl, tr]


def sort_corners_based_on_qrcode_distance(corners_contours, qrcode_coords):
    qrcode_centroid = get_centroid(qrcode_coords)
    centroids = [get_centroid(corner) for corner in corners_contours]
    centroid_distance = [
        l2_distance(centroid, qrcode_centroid) for centroid in centroids
    ]
    sorted_corners = [
        x
        for x, _ in sorted(zip(corners_contours, centroid_distance), key=lambda x: x[1])
    ]
    tr, tl, br, bl = sorted_corners
    return [br, bl, tl, tr]


def get_bounding_rect(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.int0(box)


def get_outmost_points(contours):
    all_points = np.concatenate(contours)
    return get_bounding_rect(all_points)


def sort_points_counter_clockwise(points):
    origin = np.mean(points, axis=0)

    def positive_angle(p):
        x, y = p - origin
        ang = np.arctan2(y, x)
        return 2 * np.pi + ang if ang < 0 else ang

    return sorted(points, key=positive_angle)


def get_point_quadrant(point, img_center_x, img_center_y):

    if point[0] <= img_center_x and point[1] <= img_center_y:
        return "tl"
    elif point[0] <= img_center_x and point[1] > img_center_y:
        return "bl"
    elif point[0] > img_center_x and point[1] > img_center_y:
        return "br"
    else:
        return "tr"


def find_sheet_corners_points(corner_contours, qr_coords, img):
    """Now that we have the contours, we have to find the hard corners position, even if it is slightly rotated"""

    edges = []

    # We have to use quadrant information in order to find out which corner of the bounding rectangle to pick
    corner_centroids = [get_centroid(c) for c in corner_contours]

    # Always return points on 'br', 'bl', 'tr', 'tl' orientation according to source image
    angle_sorted_centroids = sort_points_counter_clockwise(corner_centroids)
    template_center_x = int(
        (angle_sorted_centroids[0][0] + angle_sorted_centroids[1][0]) / 2
    )
    template_center_y = int(
        (angle_sorted_centroids[1][1] + angle_sorted_centroids[2][1]) / 2
    )
    # print(angle_sorted_centroids)
    # print(template_center_x, template_center_y)
    bounding_rect_side_to_pick = [
        get_point_quadrant(c, template_center_x, template_center_y)
        for c in corner_centroids
    ]

    # contours are expected to be on 'br', 'bl', 'tl', 'tr' orientation according to QR code
    for contour, str_rect_edge in zip(corner_contours, bounding_rect_side_to_pick):
        box = sort_points_counter_clockwise(get_bounding_rect(contour))
        br, bl, tl, tr = box
        rect_edge = eval(str_rect_edge)
        sorted_contours = np.array(
            [x for x in sorted(contour, key=lambda x: l2_distance(x[0], rect_edge))]
        )
        sorted_points = [
            x[0] for x in sorted(contour, key=lambda x: l2_distance(x[0], rect_edge))
        ]
        edge = sorted_contours[0][0]
        edges.append(edge)
        # print(edge, str_rect_edge, box)
    return edges


def to_gray(img):
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return gray


def is_option_marked(scanned, option, row):

    filled_template = FILLED_ANSWER_IMAGE.copy()
    # empty_template = EMPTY_ANSWER_IMAGE.copy()
    # not_marked_template = BLANK_ANSWER_IMAGE_MAP[option].copy()
    scanned = cv2.resize(scanned, config.MARK_TEMPLATE_SIZE[::-1])
    kernel = np.ones((3, 3), np.uint8)
    scanned = cv2.dilate(scanned, kernel, 1)
    filled_dist = structural_similarity(scanned, filled_template, full=False)
    # empty_dist = structural_similarity(scanned, empty_template, full=False)
    # blank_dist = structural_similarity(scanned, not_marked_template, full=False)

    if filled_dist > config.MIN_FILLED_DISTANCE:
        return True

    return False


def grade_chunk(mark_sheet, x_start, x_end, y_start, y_end, i):
    marked_alternatives = []
    for j in np.arange(1, 6):
        y_min = y_start
        y_max = y_end - config.MARKS_VERTICAL_SPACING_SIZE
        x_min = x_start + (j - 1) * config.MARK_OPTION_CHUNK_SIZE
        x_max = x_min + config.MARK_OPTION_DOT_SIZE
        patch = mark_sheet[y_min:y_max, x_min:x_max]
        answer = is_option_marked(patch, j, i)
        if answer:
            marked_alternatives.append(ANSWER_TO_LETTER_MAP[j])
        # mark_sheet = cv2.rectangle(mark_sheet, (x_min, y_min), (x_max, y_max), (0, 255, 255), 3)
    return marked_alternatives


def grade_mark_sheet(mark_sheet, n_questions):

    marked = {}

    # Inserting answers for column 1
    answer_chunk_height = config.MARKS_DIMS[0] + config.MARKS_VERTICAL_SPACING_SIZE
    insert_y_starts = config.MARKS_START_Y + np.arange(
        0, config.N_QUESTIONS_PER_COLUMN * answer_chunk_height, answer_chunk_height
    )
    insert_y_ends = (
        config.MARKS_START_Y
        + answer_chunk_height
        + np.arange(
            0, config.N_QUESTIONS_PER_COLUMN * answer_chunk_height, answer_chunk_height
        )
    )
    x_start, x_end = (
        config.MARKS_MARGIN_SIZE,
        config.MARKS_MARGIN_SIZE + config.MARKS_DIMS[1],
    )

    for question_number, (y_start, y_end) in enumerate(
        zip(insert_y_starts, insert_y_ends), 1
    ):
        if question_number > n_questions:
            break
        start = (x_start, y_start)
        end = (x_end, y_end)
        marked_answers = grade_chunk(
            mark_sheet, x_start, x_end, y_start, y_end, question_number
        )
        marked[question_number] = marked_answers
        mark_sheet = cv2.rectangle(mark_sheet, start, end, (0, 255, 255), 3)
        # print(f"{question_number} : {marked_answers}")

    # Inserting answers for column 2
    x_start = x_start + config.MARKS_DIMS[1] + config.MARKS_MIDDLE_COLUMN_SIZE
    x_end = x_end + config.MARKS_DIMS[1] + config.MARKS_MIDDLE_COLUMN_SIZE

    for question_number, (y_start, y_end) in enumerate(
        zip(insert_y_starts, insert_y_ends), question_number + 1
    ):
        if question_number > n_questions:
            break
        start = (x_start, y_start)
        marked_answers = grade_chunk(
            mark_sheet, x_start, x_end, y_start, y_end, question_number
        )
        marked[question_number] = marked_answers
        mark_sheet = cv2.rectangle(mark_sheet, start, end, (0, 255, 255), 3)
        # print(f"{question_number} : {marked_answers}")

    return marked


def evaluate_answers(marks, n_questions):
    errors = 0
    for i in range(1, 31):
        if i > n_questions:
            break
        if not set(marks[i]) == set(config.ANSWERS[i]):
            print(f"{i} : Marked {marks[i]} - Should be {config.ANSWERS[i]}")
            errors += 1
    print(
        f"Finished evaluation | {n_questions} questions | {n_questions-errors} corrects | {errors} errors"
    )


def run_pipeline(img_path, n_questions, debug=False):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image is invalid or doesn't exist")

    gray = to_gray(img)
    qr_text, qr_coords = get_qrcode_info(img)
    print(f"QRCode Text : '{qr_text.decode()}'")

    contours, _ = find_contours(gray)
    corner_contours = find_corner_contours(contours)
    corner_contours = sort_corners_based_on_qrcode_distance(corner_contours, qr_coords)

    outmost_corner_points = find_sheet_corners_points(corner_contours, qr_coords, img)
    outmost_corner_points = sort_rectangle_based_on_qrcode_distance(
        outmost_corner_points, qr_coords
    )
    # print(outmost)

    corner_centroids = list(map(get_centroid, corner_contours))
    # print(corner_centroids)

    transformed = four_point_transform(
        img.copy(), np.array(outmost_corner_points, dtype="float32")
    )
    answers = grade_mark_sheet(to_gray(transformed), n_questions)

    # Debug mode will output many of the intermediate images,
    # such as QR Code position, found contours and corners centroids
    if debug:
        # Display found QRCode + corner centroids
        img3 = img.copy()
        for circle in corner_centroids:
            img3 = cv2.circle(img3, circle, 10, (0, 255, 0), -1)
        for edge in outmost_corner_points:
            img3 = cv2.circle(img3, edge, 20, (255, 0, 0), -1)

        outline_qrcode(img3, qr_coords)

        # Display found contours
        img2 = img.copy()
        contours, _ = find_contours(img2)
        cv2.drawContours(img2, contours, -1, (0, 255, 0), 5)
        plot_bgr(img2)

        # Display deskewed image
        plot_bgr(transformed)

        print("Writing colored perspective transformed to 'perspective_color.png'")
        cv2.imwrite("perspective_color.png", transformed)

        print("Writing gray perspective transformed to 'perspective_gray.png'")
        cv2.imwrite("perspective_gray.png", to_gray(transformed))

    return answers


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--n-questions", type=int, default=30)
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()
    answers = run_pipeline(args.image, args.n_questions, args.debug)
    for question, itens in answers.items():
        print(f"{question} : {itens}")
