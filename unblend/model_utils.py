"""Small model-construction helpers with no serialization side effects."""

import functools
from collections.abc import Callable
from typing import Any


def capture_init(init: Callable) -> Callable:
    """
    Decorate a model constructor and retain its positional/keyword arguments.

    :param init: Constructor to wrap.
    :return: Wrapped constructor storing ``_init_args_kwargs`` on the instance.
    """

    @functools.wraps(init)
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Store the init args/kwargs, then call the wrapped constructor.

        :param args: Positional arguments forwarded to the constructor.
        :param kwargs: Keyword arguments forwarded to the constructor.
        """
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__
