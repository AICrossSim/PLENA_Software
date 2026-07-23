"""Decode-chip DSE for disaggregated serving on PLENA.

Two stages, deliberately decoupled:

* ``decode_dse.software`` — measures decode-phase accuracy (perplexity + GSM8K
  + IFEval) of MXINT/MXFP decode quantisation while the prefill chip stays
  FP16, and writes the precision-vs-accuracy front to a CSV.
* ``decode_dse.hardware`` — an Optuna co-design search that maximises TPS and
  minimises TPOT on the PLENA decode-chip analytic model, per precision point.

The two meet at the CSV: hardware co-design reads the accuracy front and reports
the joint precision x hardware trade-off.
"""

from decode_dse.simulator_bridge import (
    DecodeMetrics,
    DecodeSimulator,
    Precision,
)

__all__ = ["DecodeSimulator", "DecodeMetrics", "Precision"]
