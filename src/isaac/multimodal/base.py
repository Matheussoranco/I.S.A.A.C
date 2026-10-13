from abc import ABC, abstractmethod
from typing import Any


class BaseProvider(ABC):
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    def verify_auth(self, key_name: str):
        if not self.api_key:
            raise RuntimeError(f"Missing API key for {key_name}")
        return True

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        pass
