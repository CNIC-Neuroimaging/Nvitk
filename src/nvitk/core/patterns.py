"""Reusable patterns (e.g. singleton metaclass)."""

# ──────────────────────────────────────────────────────────────────────────────
# Singleton
# ──────────────────────────────────────────────────────────────────────────────


class Singleton(type):
    """
    Metaclass that returns the same instance for each ``cls(*args, **kwargs)`` call.

    The instance is stored on ``cls._instances[cls]``. Used by :class:`~nvitk.core.logger.Logger`.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
