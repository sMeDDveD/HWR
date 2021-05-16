from typing import List

from cfuzzyset import cFuzzySet as FuzzySet

from classifier.predicted_letter import PredictedLetter

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


def correct_word(word: str, threshold=0.8):
    # information about other candidates is not used
    corrections = _fuzzy_set.get(word)
    if corrections is None:
        return word

    corrected_word = None
    for score, matched_word in corrections:
        if len(matched_word) == len(word):
            if score >= threshold:
                corrected_word = matched_word

    if corrected_word is None:
        without_digits = de_digit(word)
        if without_digits == word:
            corrected_word = word
        else:
            corrected_word = correct_word(without_digits)

    if word[0].isupper():
        corrected_word = corrected_word.capitalize()

    return corrected_word


def letters_to_words(letters: List[PredictedLetter]) -> List[str]:
    words: List[str] = []

    letters_number = len(letters)
    current_word = ""

    average_width = sum(map(lambda x: x.extracted.width, letters)) / letters_number

    for i in range(letters_number):
        current_word += letters[i].probable_letter

        if i < letters_number - 1:
            difference = letters[i + 1].extracted.coordinates[0] - letters[i].extracted.coordinates[0]
            difference -= letters[i].extracted.width

            # some heuristic
            if difference > average_width / 2:
                words.append(current_word)
                current_word = ""
    if current_word:
        words.append(current_word)
    return words
