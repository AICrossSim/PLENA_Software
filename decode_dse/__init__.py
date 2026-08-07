"""Decode-only precision, accuracy, and hardware co-design interfaces."""

from decode_dse.legality import (
    CrossStackCapability,
    PackedKVRuntimeTarget,
    StackValidity,
    constrain_stack_validity,
    evaluate_profile_legality,
    evaluate_stack_capability,
)
from decode_dse.profiles import DecodePrecisionProfile, enumerate_decode_profiles
from decode_dse.simulator_bridge import (
    CompilerTracePointDescriptor,
    CompilerTraceRequestBinder,
    DecodeMetrics,
    DecodeSimulator,
    Precision,
)

__all__ = [
    "CrossStackCapability",
    "CompilerTracePointDescriptor",
    "CompilerTraceRequestBinder",
    "DecodeMetrics",
    "DecodePrecisionProfile",
    "DecodeSimulator",
    "PackedKVRuntimeTarget",
    "Precision",
    "StackValidity",
    "constrain_stack_validity",
    "enumerate_decode_profiles",
    "evaluate_profile_legality",
    "evaluate_stack_capability",
]
