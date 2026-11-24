"""Compatibility exports for quality checks and the tagger."""

from kitshot.render.quality_checks import (  # noqa: F401 - compatibility exports
    CONTENT_TAGS,
    FORBIDDEN_TAGS,
    HEAD_TOP_MAX,
    MAN_CONTENT_TAGS,
    MAN_FORBIDDEN_TAGS,
    MAN_REQUIRED_TAGS,
    PROBE_TAGS,
    REQUIRED_TAGS,
    TAG_THRESHOLD,
    CheckResult,
    Expectation,
    TagResult,
    best,
    check,
    expectation,
    head_top,
    man_expectation,
    score,
)
from kitshot.render.tagger import TAGGER_REPO, Eva02Tagger  # noqa: F401 - compatibility exports
