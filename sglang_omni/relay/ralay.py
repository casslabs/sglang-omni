# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any


class Ralay(ABC):
    """
    Abstract base class for relay connectors.

    This class defines the interface that all relay implementations must follow.
    Subclasses should implement the abstract methods to provide specific
    relay functionality (e.g., NIXL-based, Mooncake-based, etc.).
    """

    @abstractmethod
    def put(self, descriptors: list[Any]) -> Any:
        """
        Put descriptors into the distributed store (synchronous).

        Parameters
        ----------
        descriptors : list[Any]
            List of Descriptor objects containing tensor data

        Returns
        -------
        Any
            Operation object with metadata() and wait_for_completion() methods
        """

    @abstractmethod
    def get(self, metadata: Any, descriptors: list[Any]) -> Any:
        """
        Get data from the distributed store using metadata and descriptors (synchronous).

        Parameters
        ----------
        metadata : Any
            Metadata from readable operation (returned by put)
        descriptors : list[Any]
            List of Descriptor objects for receiving data

        Returns
        -------
        Any
            Operation object with wait_for_completion() method
        """

    @abstractmethod
    async def put_async(self, descriptors: list[Any]) -> Any:
        """
        Put descriptors into the distributed store (asynchronous).

        Parameters
        ----------
        descriptors : list[Any]
            List of Descriptor objects containing tensor data

        Returns
        -------
        Any
            Operation object with metadata() and wait_for_completion() methods
        """

    @abstractmethod
    async def get_async(self, metadata: Any, descriptors: list[Any]) -> Any:
        """
        Get data from the distributed store using metadata and descriptors (asynchronous).

        Parameters
        ----------
        metadata : Any
            Metadata from readable operation (returned by put)
        descriptors : list[Any]
            List of Descriptor objects for receiving data

        Returns
        -------
        Any
            Operation object with wait_for_completion() method
        """

    @abstractmethod
    def cleanup(self, request_id: str) -> None:
        """
        Clean up resources for a request.

        Parameters
        ----------
        request_id : str
            Identifier for the request to clean up
        """

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """
        Get connector health status.

        Returns
        -------
        dict[str, Any]
            Dictionary containing health status and metrics
        """

    @abstractmethod
    def close(self) -> None:
        """
        Clean shutdown of the connector.
        """
