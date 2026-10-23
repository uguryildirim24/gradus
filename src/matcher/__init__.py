"""Rule-based labels for the job matcher."""

from .rules import format_completion, label, parse_completion
from .schema import CandidateCard, RequirementCard
from .variants import Variant, full_match, make_variants

__all__ = [
    "CandidateCard",
    "RequirementCard",
    "Variant",
    "format_completion",
    "full_match",
    "label",
    "make_variants",
    "parse_completion",
]
