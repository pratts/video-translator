"""Japanese phrases (bunsetsu): a content word plus the function words that follow it, e.g. してるんですけど, 日本の."""
from functools import cache

# UniDic parts of speech that attach to the word before them: particles (けど, の), auxiliary verbs (です, てる),
# suffixes and symbols
ATTACHES_TO_PREVIOUS = {"助詞", "助動詞", "接尾辞", "補助記号"}


@cache
def _tagger():
    import fugashi  # with unidic-lite
    return fugashi.Tagger()


def tokens(text: str) -> list[tuple[str, bool]]:
    result, after_prefix = [], False
    for token in _tagger()(text):
        pos1, pos2 = token.feature.pos1, token.feature.pos2
        # dependent verbs (し in お聞きします, いる in 思っている) belong to the verb before them
        attaches = pos1 in ATTACHES_TO_PREVIOUS or (pos1 == "動詞" and pos2 == "非自立可能")
        result.append((token.surface, not (attaches or after_prefix)))
        after_prefix = pos1 == "接頭辞"   # a prefix (お, ご) belongs to the word after it
    return result
