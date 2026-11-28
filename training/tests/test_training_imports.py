"""Smoke test: the training stage and its dependencies import."""


def test_dependencies_import():
    import sklearn  # noqa: F401
    import tensorflow  # noqa: F401


def test_package_imports():
    import lstm_training

    assert lstm_training.__version__
