from typing import List

from cfuzzyset import cFuzzySet as FuzzySet

from preprocessing.letter import ExtractedLetter

_fuzzy_set = FuzzySet(use_levenshtein=True)

with open("postprocessing\words.txt", "r") as words_file:
    for word in words_file:
        _fuzzy_set.add(word[:-1])

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
    corrections = _fuzzy_set.get(word)
    if corrections is None:
        return word

    corrected_word = None
    for correction in corrections:
        if len(correction[1]) == len(word):
            if correction[0] >= threshold:
                corrected_word = correction[1]

    if corrected_word is None:
        without_digits = de_digit(word)
        if without_digits == word:
            corrected_word = word
        else:
            corrected_word = correct_word(without_digits)

    if word[0].isupper():
        corrected_word = corrected_word[0].upper() + corrected_word[1:]

    return corrected_word


def letters_to_words(letters: List[ExtractedLetter]):
    words = []

    letters_number = len(letters)
    current_word = ""

    average_width = sum(map(lambda x: x.width, letters)) / letters_number

    for i in range(letters_number):
        current_word += letters[i].prediction

        print(letters[i].candidates)

        if i < letters_number - 1:
            difference = letters[i + 1].coordinates[0] - letters[i].coordinates[0]
            difference -= letters[i].width

            if difference > average_width / 2:
                words.append(current_word)
                current_word = ""
    if current_word:
        words.append(current_word)
    return words
