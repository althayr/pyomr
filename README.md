# pyOMR

You can use this library to grade a marked scans or pictures of the template answer sheet.

The implemented grader is robust to rotation, and also corrects perspective if the image comes from a phone camera.

## Installation

This repository is not import ready, so you should clone it first with `git clone`.

Then install all the python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Creating a template answer sheet

The default sheet allows up to 30 answers divided on two columns. You can tinker with the configuration on `config.py` to change the image size.

To generate a new template sheet, just run:

```python
python3 produce_answer_sheet.py \
        --qr_text "Althayr Santos de Nazaret, 11502414" \
        --dst "./example_student_answer_sheet.png"
```

## Grading a answer sheet image

It will print on console the assigned options for each question, starting on the first column up to `n_questions`.

For each question the grader supports assignment of one, more than one or no answers.

```python
python3 grader.py \
        --n-questions 20 \
        --image ./images/answer_sheets/camera_shot_sheet.png
```

## Grading all examples

Under `./images/answer_sheets/` there are many different images already assigned and ready to be graded.

You can evaluate all of them in bulk by running:

**Note:** The last two are expected to produce errors, the first one has answers different than the default ones under `config.py`, and the second one is a empty sheet.

```python
python3 grade_examples.py
```

Expected output:

```bash
(venv) ~/pyomr$ python3 grade_examples.py 
Evaluating './images/answer_sheets/rotated_sideways_sheet.png'
QRCode Text : 'Althayr Santos de Nazaret, 11502414'
Finished evaluation | 30 questions | 30 corrects | 0 errors
---
Evaluating './images/answer_sheets/simple_scanned_sheet.png'
QRCode Text : 'Althayr Santos de Nazaret, 11502414'
Finished evaluation | 30 questions | 30 corrects | 0 errors
---
Evaluating './images/answer_sheets/flipped_scanned_sheet.png'
QRCode Text : 'Althayr Santos de Nazaret, 11502414'
Finished evaluation | 30 questions | 30 corrects | 0 errors
---
Evaluating './images/answer_sheets/rotated_scanned_sheet.png'
QRCode Text : 'Althayr Santos de Nazaret, 11502414'
Finished evaluation | 30 questions | 30 corrects | 0 errors
---
Evaluating './images/answer_sheets/camera_shot_sheet.png'
QRCode Text : 'Althayr Santos de Nazaret, 11502414'
Finished evaluation | 30 questions | 30 corrects | 0 errors
---
Evaluating './images/answer_sheets/half_page_sheet.png'
QRCode Text : 'Althayr Santos de Nazaret, 11502414'
1 : Marked ['B'] - Should be ['A']
2 : Marked ['B'] - Should be ['D', 'E']
3 : Marked ['C'] - Should be []
4 : Marked ['D'] - Should be ['A']
5 : Marked ['E'] - Should be []
6 : Marked ['D'] - Should be []
7 : Marked ['D'] - Should be ['A', 'B']
8 : Marked [] - Should be ['D']
9 : Marked [] - Should be ['B', 'C']
10 : Marked [] - Should be ['E']
11 : Marked [] - Should be ['B']
12 : Marked [] - Should be ['C', 'D']
13 : Marked ['D'] - Should be []
14 : Marked ['E'] - Should be ['C']
15 : Marked ['A'] - Should be ['B']
16 : Marked [] - Should be ['E']
17 : Marked [] - Should be ['B']
18 : Marked [] - Should be ['A']
19 : Marked [] - Should be ['D']
22 : Marked [] - Should be ['B']
23 : Marked ['B'] - Should be ['E']
24 : Marked ['D'] - Should be []
25 : Marked ['B'] - Should be ['C']
26 : Marked [] - Should be ['A']
27 : Marked ['C'] - Should be []
28 : Marked [] - Should be ['E']
29 : Marked [] - Should be ['E']
30 : Marked [] - Should be ['A']
Finished evaluation | 30 questions | 2 corrects | 28 errors
---
Evaluating './images/answer_sheets/example_student_answer_sheet.png'
QRCode Text : 'Althayr Santos de Nazaret, 11502414'
1 : Marked [] - Should be ['A']
2 : Marked [] - Should be ['D', 'E']
4 : Marked [] - Should be ['A']
7 : Marked [] - Should be ['A', 'B']
8 : Marked [] - Should be ['D']
9 : Marked [] - Should be ['B', 'C']
10 : Marked [] - Should be ['E']
11 : Marked [] - Should be ['B']
12 : Marked [] - Should be ['C', 'D']
14 : Marked [] - Should be ['C']
15 : Marked [] - Should be ['B']
16 : Marked [] - Should be ['E']
17 : Marked [] - Should be ['B']
18 : Marked [] - Should be ['A']
19 : Marked [] - Should be ['D']
22 : Marked [] - Should be ['B']
23 : Marked [] - Should be ['E']
25 : Marked [] - Should be ['C']
26 : Marked [] - Should be ['A']
28 : Marked [] - Should be ['E']
29 : Marked [] - Should be ['E']
30 : Marked [] - Should be ['A']
Finished evaluation | 30 questions | 8 corrects | 22 errors
---
```