"""Database files: one CSV per strain path, with the column names of the legacy database
(``GetResultsUnitCell`` in ``legacy/fedoo_coupling/fedoo/libHomogen/Homog_path.py``).

Columns: ``Increment``, ``Time``, strain ``Exx Eyy Ezz Exy Exz Eyz`` (engineering shear),
stress ``Sxx Syy Szz Sxy Sxz Syz`` and the full tangent ``C11 ... C66`` (row i = stress
component, column j = strain component). The legacy ``Wm Wm_r Wm_ir Wm_d`` columns are not
written. The symmetric terms ``Cij_m`` used to train the tangent network are computed in the
training stage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from database_generation.loading import COMPONENTS
from database_generation.rve import PathResult

STRAIN_COLUMNS = list(COMPONENTS)
STRESS_COLUMNS = ["Sxx", "Syy", "Szz", "Sxy", "Sxz", "Syz"]
TANGENT_COLUMNS = [f"C{i + 1}{j + 1}" for i in range(6) for j in range(6)]


def path_file(database_dir: str | Path, path_index: int) -> Path:
    return Path(database_dir) / f"path_{path_index:05d}.csv"


def to_dataframe(result: PathResult) -> pd.DataFrame:
    n = result.strain.shape[0]
    data = np.hstack([
        np.arange(1, n + 1)[:, None], result.time[:, None], result.strain, result.stress,
        result.tangent.reshape(n, 36),
    ])
    return pd.DataFrame(
        data, columns=["Increment", "Time"] + STRAIN_COLUMNS + STRESS_COLUMNS + TANGENT_COLUMNS
    ).astype({"Increment": int})


def write_path(database_dir: str | Path, path_index: int, result: PathResult) -> Path:
    out = path_file(database_dir, path_index)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".csv.tmp")
    to_dataframe(result).to_csv(tmp, index=False, float_format="%.10g")
    tmp.replace(out)  # a path file exists only once it is complete
    return out


def read_path(database_dir: str | Path, path_index: int) -> pd.DataFrame:
    return pd.read_csv(path_file(database_dir, path_index))
