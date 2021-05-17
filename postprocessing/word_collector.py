import typing as tp
from typing import List

import numpy as np
from cfuzzyset import cFuzzySet as FuzzySet

from classifier.predicted_letter import PredictedLetter
from classifier.predictor import predicted_class_to_letter

_fuzzy_set = FuzzySet(use_levenshtein=True)

with open("postprocessing/words.txt", "r") as words_file:
    for word in words_file.readlines():
        _fuzzy_set.add(word.strip())

# common mistakes
digits_to_letters = {
    "0": "o",
    "1": "l",
    "5": "s",
    "9": "q"
}


def de_digit(word: str):
    without_digits = ""
    for c in word:
        if not c.isdigit() or c not in digits_to_letters:
            without_digits += c
        else:
            without_digits += digits_to_letters[c]
    return without_digits


def correct_word(word: str, threshold=0.8) -> tp.Tuple[str, float]:
    # information about other candidates is not used
    result: tp.Tuple[str, float] = (word, 0.0)

    corrections = _fuzzy_set.get(word)
    if corrections is None:
        if all(letter.isdigit() for letter in word):
            return result[0], 1
        return result

    for score, matched_word in corrections:
        if len(matched_word) == len(word):
            if score >= threshold and score >= result[1]:
                result = matched_word, score
            if score == 1:
                break

    if word[0].isupper():
        result = result[0].capitalize(), result[1]

    return result


def beam_search(letters: List[PredictedLetter], k: int, buffer: int) -> List[str]:
    current_sequences: List[tp.Tuple[List[str], float]] = [
        (list(), 0.0)
    ]

    data = np.array([
        letter.probabilities for letter in letters
    ])

    for row in data:
        candidates = list()
        for sequence, score in current_sequences:
            for j, prob in enumerate(row):
                candidates.append(
                    (sequence + [predicted_class_to_letter(j)], score - np.log(prob))
                )

        candidates.sort(key=lambda v: v[1])
        current_sequences = candidates[:buffer]

    return ["".join(lst) for lst, _ in current_sequences[:k]]


def predicted_letters_to_str(letters: List[PredictedLetter]) -> str:
    return str(letter.probable_letter for letter in letters)


def letters_to_words(letters: List[PredictedLetter]) -> List[str]:
    words: List[str] = []

    letters_number = len(letters)
    current_word: List[PredictedLetter] = []

    average_width = sum(map(lambda x: x.extracted.width, letters)) / letters_number

    for i in range(letters_number):
        current_word += [letters[i]]

        if i < letters_number - 1:
            difference = letters[i + 1].extracted.coordinates[0] - letters[i].extracted.coordinates[0]
            difference -= letters[i].extracted.width

            # some heuristic
            if difference > average_width / 2:
                beams = beam_search(current_word, 5, 10)
                best_beam = max((correct_word(beam) for beam in beams), key=lambda b: b[1])

                print(beams)
                print(best_beam)
                words.append(best_beam[0])
                current_word = []
    if current_word:
        beams = beam_search(current_word, 5, 10)
        best_beam = max((correct_word(beam) for beam in beams), key=lambda b: b[1])
        print(beams)
        print(best_beam)
        words.append(best_beam[0])
    return words
