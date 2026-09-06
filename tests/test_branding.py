"""Pure-function tests for pipeline/branding.py's outro-card line logic.

No Pillow/ffmpeg needed — these test outro_lines directly rather than
rendering an actual card image."""

from __future__ import annotations

from renderflow.pipeline.branding import CTA_LINE, outro_lines


def test_outro_lines_with_engagement_question_splits_headline_from_cta():
    # A real topic-specific question, exactly as assets._outro_line
    # concatenates it: "<question> Let us know in the comments! <subscribe>".
    message = (
        "If you were him, what would you do? Let us know in the comments! "
        "Please subscribe for more trivia like this."
    )
    lines = outro_lines(message)
    texts = [text for text, _, _ in lines]

    assert texts[0] == "If you were him, what would you do?"
    assert texts[1] == CTA_LINE
    # The CTA phrase must appear exactly once across all lines — the bug
    # this replaced duplicated it.
    assert sum(1 for t in texts if t == CTA_LINE) == 1
    # The trailing subscribe reminder isn't crammed into the headline.
    assert "subscribe" not in texts[0].lower()


def test_outro_lines_plain_fallback_has_no_bogus_cta():
    # No engagement question (LLM unset/failed) — assets._outro_line's
    # plain fallback has no "Let us know in the comments!" phrase, so the
    # card must not show a call-to-action about comments that was never
    # actually asked for.
    message = "Thanks for watching! Please subscribe for more trivia like this."
    lines = outro_lines(message)
    texts = [text for text, _, _ in lines]

    assert CTA_LINE not in texts
    assert texts[0] == "Thanks for watching"


def test_outro_lines_none_message_uses_the_same_plain_fallback():
    # Narration disabled, or an old project predating this feature.
    assert outro_lines(None) == outro_lines(
        "Thanks for watching! Please subscribe for more trivia like this."
    )
