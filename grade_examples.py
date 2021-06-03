import cv2
import config
import argparse
import traceback
from grader import run_pipeline, evaluate_answers


if __name__ == "__main__":

    images = [
        "./images/answer_sheets/rotated_sideways_sheet.png",
        "./images/answer_sheets/simple_scanned_sheet.png",
        "./images/answer_sheets/flipped_scanned_sheet.png",
        "./images/answer_sheets/rotated_scanned_sheet.png",
        "./images/answer_sheets/camera_shot_sheet.png",
        # Grades correctly but answers are no the default ones
        "./images/answer_sheets/half_page_sheet.png",
        # Grades correctly but it's empty
        "./images/example_student_mark_sheet.png"
    ]
    for image in images:
        print(f"Evaluating '{image}'")
        try:
            answers = run_pipeline(image, 30)
            evaluate_answers(answers, 30)
        except Exception as e:
            traceback.print_exc()
            pass
        print("---")
