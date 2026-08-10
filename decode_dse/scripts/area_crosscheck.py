"""Cross-check frontier chip areas against an independent DC-fitted model.

The ranking area model is a structural census whose calibration domain tops
out well below the compiler-legal decode geometries, so frontier areas carry
an extrapolation flag. This script re-evaluates each frontier candidate on a
second, independently fitted area model (the area_new package from a
reference checkout) and reports the per-candidate relative spread. The
spread is a disclosure band for figures and tables — it does not replace
either model, and a large spread is a finding to report, not to hide.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

AREA_CROSSCHECK_SCHEMA = "decode-area-crosscheck/v1"


def _area_config(point: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "MLEN": int(point["mlen"]),
        "VLEN": int(point.get("vlen", point["mlen"])),
        "BLEN": int(point["blen"]),
        "HLEN": int(point.get("hlen", point["blen"])),
        "BLOCK_DIM": int(point["blen"]),
        "WEIGHT_WIDTH": str(point["weight_width"]),
        "KV_WIDTH": str(point["kv_width"]),
        "ACT_WIDTH": str(point["act_width"]),
        "FP_SETTING": str(point.get("fp_setting", "FP_E5M6")),
        "INT_DATA_WIDTH": int(point.get("int_data_width", 32)),
        "MATRIX_SRAM_DEPTH": int(point.get("matrix_sram_depth", 4096)),
        "VECTOR_SRAM_DEPTH": int(point.get("vector_sram_depth", 4096)),
        "INT_SRAM_DEPTH": int(point.get("int_sram_depth", 32)),
        "FP_SRAM_DEPTH": int(point.get("fp_sram_depth", 512)),
        "HBM_M_Prefetch_Amount": int(point.get("hbm_m_prefetch", 16)),
        "HBM_V_Prefetch_Amount": int(point.get("hbm_v_prefetch", 16)),
        "HBM_V_Writeback_Amount": int(point.get("hbm_v_writeback", 16)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--points",
        required=True,
        help="JSON file with a list of candidate points (geometry + widths)",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--simulator-root",
        required=True,
        help="checkout providing the ranking analytic_models.area package",
    )
    parser.add_argument(
        "--reference-root",
        required=True,
        help="checkout providing the independent analytic_models.area_new package",
    )
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(args.simulator_root).resolve()))
    sys.path.insert(0, str(Path(args.simulator_root).resolve() / "analytic_models"))
    import area as ranking_area

    reference_root = Path(args.reference_root).resolve()
    sys.path.insert(0, str(reference_root))
    from analytic_models import area_new as reference_area

    points = json.loads(Path(args.points).read_text(encoding="utf-8"))
    if not isinstance(points, list) or not points:
        raise ValueError("the cross-check needs a non-empty list of points")
    rows = []
    for ordinal, point in enumerate(points):
        config = _area_config(point)
        row: dict[str, Any] = {"ordinal": ordinal, "point": dict(point)}
        try:
            ranking = ranking_area.estimate_area(config)
            row["ranking_area_mm2"] = float(ranking["area"]) / 1e6
            row["ranking_model"] = str(ranking["area_model"])
            row["ranking_evidence_tier"] = str(ranking["evidence_tier"])
        except Exception as error:  # disclosed, not fatal
            row["ranking_error"] = f"{type(error).__name__}: {error}"
        try:
            reference = reference_area.estimate_area(config)
            row["reference_area_mm2"] = float(reference["area"]) / 1e6
            row["reference_model"] = str(
                reference.get("area_model", "area_new")
            )
        except Exception as error:
            row["reference_error"] = f"{type(error).__name__}: {error}"
        if "ranking_area_mm2" in row and "reference_area_mm2" in row:
            base = row["ranking_area_mm2"]
            row["relative_spread"] = (
                (row["reference_area_mm2"] - base) / base if base else None
            )
        rows.append(row)
    spreads = [
        abs(row["relative_spread"])
        for row in rows
        if row.get("relative_spread") is not None
    ]
    payload = {
        "schema_version": AREA_CROSSCHECK_SCHEMA,
        "points": rows,
        "compared_points": len(spreads),
        "max_abs_relative_spread": max(spreads) if spreads else None,
        "median_abs_relative_spread": (
            sorted(spreads)[len(spreads) // 2] if spreads else None
        ),
        "note": (
            "both models extrapolate beyond their DC calibration domains at "
            "compiler-legal decode geometry; the spread is the disclosure "
            "band quoted next to frontier areas"
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "compared_points": payload["compared_points"],
                "max_abs_relative_spread": payload["max_abs_relative_spread"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
