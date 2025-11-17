import warnings


def assert_experimental_flag_acknowledged(flag_name: str, enabled: bool) -> None:
    """
    Centralized helper to log a clear warning when an experimental flag is enabled.

    This does NOT change behavior; it simply makes it harder to enable
    experimental paths by accident.
    """
    if enabled:
        warnings.warn(
            f"[experimental] Flag '{flag_name}' is ENABLED. "
            "This path is not baseline and should be used intentionally.",
            RuntimeWarning,
        )
