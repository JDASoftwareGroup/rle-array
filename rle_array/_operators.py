from typing import Any, Callable


def rev(op: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
    """
    Reverses given binary operator.
    """

    def f(a: Any, b: Any) -> Any:
        return op(b, a)

    setattr(f, "__name__", f"r{getattr(op, '__name__', '???')}")

    return f
