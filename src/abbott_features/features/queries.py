"""Feature Queries."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import polars as pl


class FeatureQuery(ABC):
    """Base class for feature queries."""

    @abstractmethod
    def load_resources(self) -> dict[str, Any]:
        """Load resources required for the query."""
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> "pl.DataFrame":
        """Compute the features."""
        raise NotImplementedError
