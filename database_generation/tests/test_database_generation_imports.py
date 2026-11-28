"""Smoke test: the database generation stage and its dependencies import."""


def test_dependencies_import():
    import fedoo  # noqa: F401
    import simcoon  # noqa: F401


def test_package_imports():
    import database_generation

    assert database_generation.__version__
