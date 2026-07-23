"""Software side of the decode DSE: decode-phase quantisation accuracy.

The prefill chip is unquantised; only the decode chip is quantised (MXINT/MXFP
weights + KV, low-precision activation compute). Accuracy is scored with every
token forced through the decode numerics (`force_runtime_phase("decode")`).
"""

from decode_dse.software.decode_quant import (
    DecodeQuantSpec,
    build_decode_pass_args,
    gptq_cache_key,
)

__all__ = ["DecodeQuantSpec", "build_decode_pass_args", "gptq_cache_key"]
