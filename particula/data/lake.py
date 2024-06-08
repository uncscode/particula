"""creates the lake class, a collection of streams"""


from dataclasses import dataclass, field
from typing import Dict, Iterator, Any, Tuple
from particula.data.stream import Stream


@dataclass
class Lake:
    """A class representing a lake which is a collection of streams.

    Attributes:
        streams (Dict[str, Stream]): A dictionary to hold streams with their
        names as keys.
    """
    streams: Dict[str, Stream] = field(default_factory=dict)

    def add_stream(self, stream: Stream, name: str) -> None:
        """Add a stream to the lake.

        Args:
        -----------
            stream (Stream): The stream object to be added.
            name (str): The name of the stream.

        Raises:
        -------
            ValueError: If the stream name is already in use or not a valid
            identifier.
        """
        if name in self.streams:
            raise ValueError(f"Stream name {name} already in use")
        if not name.isidentifier():
            raise ValueError(
                f"Stream name '{name}' is not a valid python identifier")
        self.streams[name] = stream

    def __getattr__(self, name: str) -> Any:
        """Allow accessing streams as an attributes.
        Raises:
            AttributeError: If the stream name is not in the lake.
        Example: lake.stream_name
        """
        try:
            return self.streams[name]
        except KeyError as exc:
            raise AttributeError(f"No stream named {name}") from exc

    def __dir__(self) -> list:
        """List available streams.
        Example: dir(lake)"""
        return list(self.streams.keys())

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the streams in the lake.
        Example: [stream.header for stream in lake]""
        """
        return iter(self.streams.items())

    def items(self) -> Iterator[Tuple[Any, Any]]:
        """Return an iterator over the key-value pairs."""
        return iter(self)

    def values(self) -> Iterator[Any]:
        """Return an iterator over the values."""
        return iter(self.streams.values())

    def keys(self) -> Iterator[Any]:
        """Return an iterator over the keys."""
        return iter(self.streams.keys())

    def __len__(self) -> int:
        """Return the number of streams in the lake.
        Example: len(lake)"""
        return len(self.streams)

    def __getitem__(self, key: str) -> Any:
        """Get a stream by name.
        Example: lake['stream_name']"""
        return self.streams[key]

    def __setitem__(self, key: str, value: Stream) -> None:
        """Set a stream by name.
        Example: lake['stream_name'] = new_stream"""
        # verify it is a stream object
        if not isinstance(value, Stream):
            raise ValueError(f"This is not a Stream object, {value}")
        self.streams[key] = value

    def __delitem__(self, key: str) -> None:
        """Remove a stream by name.
        Example: del lake['stream_name']"""
        if key in self.streams:
            del self.streams[key]
        else:
            raise ValueError(f"No stream named {key} to remove")

    def __repr__(self) -> str:
        """Return a string representation of the lake.
        Example: print(lake)"""
        return f"Lake with streams: {list(self.streams.keys())}"

    @property
    def summary(self) -> None:
        """Return a string summary iterating over each stream
            and print Stream.header.
        Example: lake.summary
        """
        for stream in self.streams:
            print(f"{stream} Headers:")
            print(f"      {self[stream].header}")
