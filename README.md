# pyOMR

You can use this library to grade a marked scans or pictures of the template answer sheet. 

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
