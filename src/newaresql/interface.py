from typing import Literal, Self


class Test:
    def __init__(self, test: dict):
        self._test = test


class Connector:
    def __init__(
        self,
    ):
        self._plugin = None
        return

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return

    def list_tests(self): ...

    def get_data(self, test: Test | dict): ...

    def stream_data(self, test: Test | dict): ...



