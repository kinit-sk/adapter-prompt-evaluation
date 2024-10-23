def convert_language(language):
    if language == 'english':
        return 'en'
    elif language == 'spanish':
        return 'es'
    elif language == 'turkish':
        return 'tr'
    elif language == 'dutch':
        return 'nl'
    elif language == 'bulgarian':
        return 'bg'
    elif language == 'arabic':
        return 'ar'
    elif language == 'czech':
        return 'cs'
    elif language == 'hungarian':
        return 'hu'
    elif language == 'polish':
        return 'pl'
    elif language == 'slovak':
        return 'sk'
    elif language == 'slovenian':
        return 'sl'
    elif language == 'croatian':
        return 'hr'
    elif language == 'serbian':
        return 'sr'
    elif language == 'russian':
        return 'ru'
    elif language == 'ukrainian':
        return 'uk'
    elif language == 'romanian':
        return 'ro'
    elif language == 'german':
        return 'de'
    elif language == 'italian':
        return 'it'
    elif language == 'french':
        return 'fr'
    elif language == 'portuguese':
        return 'pt'
    elif language == 'swedish':
        return 'sv'
    elif language == 'norwegian':
        return 'no'
    elif language == 'danish':
        return 'da'
    elif language == 'finnish':
        return 'fi'
    elif language == 'estonian':
        return 'et'
    elif language == 'latvian':
        return 'lv'
    elif language == 'lithuanian':
        return 'lt'
    elif language == 'greek':
        return 'el'
    elif language == 'hebrew':
        return 'he'
    elif language == 'hindi':
        return 'hi'
    elif language == 'chinese':
        return 'zh'
    elif language == 'telugu':
        return 'te'
    elif language == 'swahili':
        return 'sw'
    elif language == 'urdu':
        return 'ur'
    elif language == 'malayalam':
        return 'ml'
    else:
        raise ValueError(f'Invalid language: {language}')
