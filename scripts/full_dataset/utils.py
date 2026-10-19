import nltk

def soundex(word, size=4):
    """Return the Soundex code of a word: first letter + digits, padded with zeros."""
    # Digit for each letter A..Z (0 = ignored: vowels, H, W, Y)
    codes = "01230120022455012623010202"
    code = ""
    for letter in word.upper():
        if not "A" <= letter <= "Z":
            continue
        if not code:
            code = letter                                  # keep the first letter
        else:
            digit = codes[ord(letter) - ord("A")]
            if digit != "0" and code[-1] != digit:         # skip repeated digits
                code += digit
        if len(code) == size:
            break
    return code.ljust(size, "0")

def levenshtein(s1, s2):
    """ Returns Levenshtein distance of two given strings s1 and s2 """
    return nltk.edit_distance(s1, s2)