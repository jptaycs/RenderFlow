"""Prompt→keywords extraction shared by the Pexels photo and video adapters.

Pexels search wants 2-4 keywords, not the long photo-caption prompts the
pipeline builds — these helpers reduce the known prompt shapes
(_local_image_prompt, _thumbnail_prompt) to a short query, with a
stopword-filter fallback for anything else. If the prompt builders'
wording changes, update the regexes here (both adapters inherit the fix).
"""

from __future__ import annotations

import re

# Matches the prompt builders in pipeline/script.py and pipeline/assets.py.
SCENE_TOPIC_RE = re.compile(r"photograph about (.+?) —")
SCENE_EXCERPT_RE = re.compile(r"described here:\s*(.+?)\s+Keep ")
THUMBNAIL_TOPIC_RE = re.compile(r"dramatic photograph of\s*(.+?)[.,]")
# api.py::_vary_prompt appends this on manual regenerate — search junk.
VARIATION_MARKER = " Try this take: "

STOPWORDS = frozenset(
    """a an the and or but of in on at to for with from by about into over
    under is are was were be been being it its this that these those his her
    their there here as if then than so not no never only even still very
    while when where which who whom what how all any each every some most
    more much they them he she we you i had has have will would could should
    photograph photo image picture shot wide medium documentary b-roll
    realistic cinematic dramatic lighting composition frame text words
    letters typography face faces people person camera portrait visible
    absolutely never recognizably present directly""".split()
)

# Stock search returns plenty of face-forward shots; scene visuals must
# avoid them. Results whose alt text looks face-y go to the back of the
# candidate list.
FACEY_ALT_RE = re.compile(r"\b(face|portrait|selfie|headshot|smiling)\b", re.IGNORECASE)


def content_words(text: str, limit: int) -> list[str]:
    words: list[str] = []
    for raw in re.findall(r"[A-Za-z][A-Za-z'-]+", text):
        word = raw.lower()
        if word in STOPWORDS or word in words:
            continue
        words.append(word)
        if len(words) >= limit:
            break
    return words


def search_query(prompt: str) -> str:
    """Reduce a pipeline image prompt to a short stock-search query."""
    base = prompt.split(VARIATION_MARKER)[0]
    topic_match = SCENE_TOPIC_RE.search(base) or THUMBNAIL_TOPIC_RE.search(base)
    if topic_match:
        topic_words = content_words(topic_match.group(1), 3)
        excerpt = SCENE_EXCERPT_RE.search(base)
        extra = content_words(excerpt.group(1), 6) if excerpt else []
        words = topic_words + [w for w in extra if w not in topic_words]
        return " ".join(words[:6])
    return " ".join(content_words(base, 6))


def candidate_queries(query: str) -> list[str]:
    """The query, then progressively shorter fallbacks for empty results."""
    words = query.split()
    candidates: list[str] = []
    while words:
        candidates.append(" ".join(words))
        words = words[:-1] if len(words) > 2 else []
    return candidates or [query]
