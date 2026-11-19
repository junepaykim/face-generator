"""FM27 Face Generator."""


def __getattr__(name: str) -> str:
    if name != "__version__":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .locations import PROJECT_ROOT

    # A source install can be updated in place before its wheel metadata is refreshed.
    if (PROJECT_ROOT / "pyproject.toml").is_file():
        from .setup.update import installed_version

        return installed_version(PROJECT_ROOT)

    from importlib.metadata import version

    return version("facegen")
