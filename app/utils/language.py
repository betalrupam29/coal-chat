def detect_language(text: str) -> str:

    bengali_count = 0
    hindi_count = 0

    for char in text:

        if "\u0980" <= char <= "\u09FF":
            bengali_count += 1

        elif "\u0900" <= char <= "\u097F":
            hindi_count += 1

    if bengali_count > 0:
        return "bn"

    if hindi_count > 0:
        return "hi"

    return "en"


def language_name(code: str) -> str:

    languages = {
        "en": "English",
        "hi": "Hindi",
        "bn": "Bengali"
    }

    return languages.get(code, "English")