from typing import Any


class Callback:
    def epoch_start(self, data: Any) -> None:
        pass

    def epoch_end(self, data: Any) -> None:
        pass

    def run(self, data: Any) -> None:
        self.epoch_start(data=data)
        self.epoch_end(data=data)