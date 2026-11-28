"""Paths shared by the database generation tests (a uniquely named module: the root pytest run
also collects fe_lstm/tests, whose conftest.py would shadow a plain ``from conftest import``)."""

from pathlib import Path

STAGE = Path(__file__).resolve().parents[1]
REPO = STAGE.parent
# The thesis RVE mesh (small, in git): lets the simulation tests run without microgen.
LEGACY_MESH = REPO / "legacy/database_generation/octet_surf.msh"
