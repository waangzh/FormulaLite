import formulalite


def test_package_exposes_version() -> None:
    assert isinstance(formulalite.__version__, str)
    assert formulalite.__version__
