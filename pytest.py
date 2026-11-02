"""
Lightweight pytest compatibility module and runner for offline environments where pytest is not preinstalled.
Supports markers (@pytest.mark.mocked, @pytest.mark.synthetic, @pytest.mark.realmodel), pytest.skip, and pytest.mark.skipif.
"""
import sys
import unittest

class SkipException(Exception):
    def __init__(self, reason=""):
        super().__init__(reason)
        self.reason = reason

def skip(reason=""):
    raise SkipException(reason)

class _Mark:
    def __getattr__(self, name):
        def decorator(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                fn = args[0]
                if not hasattr(fn, "pytest_marks"):
                    fn.pytest_marks = set()
                fn.pytest_marks.add(name)
                return fn
            def inner(fn):
                if not hasattr(fn, "pytest_marks"):
                    fn.pytest_marks = set()
                fn.pytest_marks.add(name)
                if name == "skipif":
                    condition = args[0] if args else kwargs.get("condition", False)
                    reason = kwargs.get("reason", "Condition met")
                    fn._pytest_skipif = (condition, reason)
                return fn
            return inner
        return decorator

mark = _Mark()
