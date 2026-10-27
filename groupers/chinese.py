"""Chinese and Cantonese phrases: segmented words, with sentence-final and aspect particles attached to the word before."""

# Particles that never start a phrase: Mandarin (的, 了, 吗, …) and Cantonese (嘅, 咗, 㗎, …)
MANDARIN = "的了吗呢吧啊着过们地得嘛呀哦"
MANDARIN_TRADITIONAL = "嗎們過著"
CANTONESE = "嘅咗啦喇囉吖㗎喎啫咩噃"
ATTACHES_TO_PREVIOUS = set(MANDARIN + MANDARIN_TRADITIONAL + CANTONESE)


def tokens(text: str) -> list[tuple[str, bool]]:
    import rjieba  # installed with f5-tts
    return [(word, word not in ATTACHES_TO_PREVIOUS) for word in rjieba.cut(text)]
