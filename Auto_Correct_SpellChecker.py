from spellchecker import SpellChecker


def Auto_Correct(word):
    
    mySpellChecker = SpellChecker(language='es')
    correction =  mySpellChecker.correction(word)
    
    if correction != None:
        return correction
    return "a"
