"""Versioned JSON/numeric archive transport with atomic file replacement.

The format uses standard ZIP, JSON and NPY primitives; it never loads pickle,
imports a named class, or executes metadata. Scientific schemas live above this
transport. Payload hashes detect corruption, not authenticity. All payloads are
detached before writing. Loading enforces a declared uncompressed byte budget.
"""

import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np

_FORMAT = "particula-array-archive"


def _json_bytes(value):
    return json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _json_load(data):
    def invalid_constant(value):
        raise ValueError("nonfinite JSON constant")

    return json.loads(
        data, object_pairs_hook=_unique_object, parse_constant=invalid_constant
    )


def _numeric(array):
    if array.dtype.kind not in "biuf" or array.dtype.hasobject:
        raise ValueError(
            "archive arrays must have real numeric or boolean dtype"
        )
    if array.ndim > 8:
        raise ValueError("archive arrays support at most eight axes")


def write_archive(path, *, kind, metadata, arrays, blobs=None, compressed=True):
    """Atomically write detached arrays/bytes and JSON-compatible metadata.

    A failure before os.replace leaves an existing destination unchanged.
    Parent directories must already exist. Compression occurs on host outside
    simulation execution/capture. blobs stores immutable checkpoint bytes.
    """
    if kind not in ("analysis", "restart"):
        raise ValueError("archive kind must be analysis or restart")
    payloads, descriptors = {}, {}
    for index, (key, value) in enumerate(sorted(arrays.items())):
        if not isinstance(key, str) or not key:
            raise ValueError("array keys must be nonempty strings")
        array = np.array(value, copy=True, order="C")
        _numeric(array)
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        data = buffer.getvalue()
        name = f"arrays/{index}.npy"
        payloads[name] = data
        descriptors[key] = {
            "member": name,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    binary = {}
    for index, (key, value) in enumerate(sorted((blobs or {}).items())):
        if not isinstance(key, str) or not key or not isinstance(value, bytes):
            raise ValueError(
                "blob keys must be strings and values immutable bytes"
            )
        name = f"blobs/{index}.bin"
        payloads[name] = value
        binary[key] = {
            "member": name,
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    manifest = _json_bytes(
        {
            "format": _FORMAT,
            "version": 1,
            "kind": kind,
            "metadata": metadata,
            "arrays": descriptors,
            "blobs": binary,
        }
    )
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            with zipfile.ZipFile(
                stream,
                "w",
                compression=(
                    zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
                ),
            ) as archive:
                archive.writestr("manifest.json", manifest)
                for name, data in payloads.items():
                    archive.writestr(name, data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _decode_array(data, descriptor):
    """Validate NPY dimensions against actual bytes before allocating arrays."""
    stream = io.BytesIO(data)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError("unsupported NPY version")
    if dtype.hasobject or dtype.kind not in "biuf" or len(shape) > 8:
        raise ValueError("unsupported numeric array schema")
    if any(type(size) is not int or size < 0 for size in shape):
        raise ValueError("invalid numeric shape")
    if math.prod(shape) * dtype.itemsize != len(data) - stream.tell():
        raise ValueError("numeric dimensions disagree with payload length")
    if list(shape) != descriptor["shape"] or dtype.str != descriptor["dtype"]:
        raise ValueError("numeric dimensions/dtype disagree with manifest")
    return np.load(io.BytesIO(data), allow_pickle=False)


def read_archive(path, *, kind, max_bytes=128 * 1024 * 1024):
    """Read a matching archive kind within a hard uncompressed byte budget.

    Returns metadata, detached numeric arrays, immutable binary payloads. No
    filesystem extraction is performed. Unknown versions/kinds fail explicitly.
    """
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            entries = archive.infolist()
            names = {entry.filename for entry in entries}
            if (
                len(names) != len(entries)
                or sum(entry.file_size for entry in entries) > max_bytes
            ):
                raise ValueError(
                    "duplicate archive member or byte budget exceeded"
                )
            if "manifest.json" not in names or archive.getinfo(
                "manifest.json"
            ).file_size > min(max_bytes, 4 * 1024 * 1024):
                raise ValueError("missing or oversized archive manifest")
            manifest = _json_load(archive.read("manifest.json"))
            if (
                manifest["format"] != _FORMAT
                or type(manifest["version"]) is not int
                or manifest["version"] != 1
                or manifest["kind"] != kind
            ):
                raise ValueError(
                    "unsupported archive version or wrong archive kind"
                )
            return _read_payloads(archive, manifest)
    except (KeyError, TypeError, zipfile.BadZipFile, EOFError) as error:
        raise ValueError("malformed or corrupted numeric archive") from error


def _read_payloads(archive, manifest):
    """Validate every named payload without extracting files."""
    arrays: dict[str, np.ndarray] = {}
    blobs: dict[str, bytes] = {}
    expected = {"manifest.json"}
    for category, target in (("arrays", arrays), ("blobs", blobs)):
        for key, descriptor in manifest[category].items():
            name = descriptor["member"]
            if not isinstance(key, str) or not key or name in expected:
                raise ValueError("invalid or repeated payload reference")
            expected.add(name)
            data = archive.read(name)
            if hashlib.sha256(data).hexdigest() != descriptor["sha256"]:
                raise ValueError("payload digest mismatch")
            target[key] = (
                _decode_array(data, descriptor)
                if category == "arrays"
                else data
            )
    if set(archive.namelist()) != expected:
        raise ValueError("unreferenced archive members")
    return manifest["metadata"], arrays, blobs
