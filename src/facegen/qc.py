"""Compatibility exports for quality checks and the tagger."""

from facegen.render.quality_checks import (  # noqa: F401 - compatibility exports
    CONTENT_TAGS,
    FORBIDDEN_TAGS,
    HEAD_TOP_MAX,
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
    score,
)
from facegen.render.tagger import TAGGER_REPO, Eva02Tagger  # noqa: F401 - compatibility exports
