"""Hardware side of the decode DSE: Optuna co-design on the analytic model.

For each decode precision, search the PLENA decode-chip hardware knobs to
maximise TPS and minimise TPOT, with HBM-fit as a hard constraint. Every
candidate is scored in-process by the decode-chip analytic model (no proxy).
"""

from decode_dse.hardware.codesign_search import (
    CodesignSearch,
    HardwareSpace,
    search_precision,
)

__all__ = ["CodesignSearch", "HardwareSpace", "search_precision"]
