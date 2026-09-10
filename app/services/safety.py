EMERGENCY_KEYWORDS = [

    # English
    "fire",
    "smoke",
    "gas leak",
    "gas leakage",
    "explosion",
    "collapse",
    "injury",
    "accident",
    "emergency",
    "trapped",
    "roof fall",

    # Hindi
    "आग",
    "धुआं",
    "गैस रिसाव",
    "विस्फोट",
    "ढहना",
    "दुर्घटना",
    "चोट",
    "आपातकाल",
    "फंसा",
    "छत गिरना",

    # Bengali
    "আগুন",
    "ধোঁয়া",
    "গ্যাস লিক",
    "গ্যাস লিকেজ",
    "বিস্ফোরণ",
    "ধস",
    "দুর্ঘটনা",
    "আঘাত",
    "জরুরি",
    "আটকে",
    "ছাদ ধস"
]


def is_emergency(message: str) -> bool:

    message = message.lower()

    for keyword in EMERGENCY_KEYWORDS:

        if keyword.lower() in message:
            return True

    return False