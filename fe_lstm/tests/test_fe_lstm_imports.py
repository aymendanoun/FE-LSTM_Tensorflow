"""Smoke test: the environment and the package import correctly."""


def test_dependencies_import():
    import fedoo  # noqa: F401
    import simcoon  # noqa: F401
    import tensorflow  # noqa: F401


def test_package_imports():
    import fe_lstm
    from fe_lstm import fem, model  # noqa: F401

    assert fe_lstm.__version__
