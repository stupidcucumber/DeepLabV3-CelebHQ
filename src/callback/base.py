from typing import Any


class Callback:
    def epoch_start(self, data: Any) -> None:
        pass

    def epoch_end(self, data: Any) -> None:
        pass

    def batch_start(self, data: Any) -> None:
        pass

    def batch_end(self, data: Any) -> None:
        pass
    