"""Read and write PoE2 net files, the engine's loadable evaluator parameters."""

from __future__ import annotations

import hashlib
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


class NetFileError(ValueError):
    """Raised when bytes are not a valid net file or parameters cannot be stored."""


MAGIC = b"POE2NET\0"
CONTAINER_VERSION = 1
PATTERN_GAIN_DESIGN = "pattern-gain-v1"
FRACTIONAL_BITS = 5
PLY_COUNT = 50
ORBIT_COUNT = 1716
GAIN_FEATURE_COUNT = 19
LINE_KNOTS = (0, 28, 49)
GAIN_KNOTS = (0, 12, 24, 36, 49)
LINE_KNOT_COUNT = len(LINE_KNOTS)
GAIN_KNOT_COUNT = len(GAIN_KNOTS)
# pattern-gain-v1 tables in file order: line knots (u8), gain knots (u8), intercepts (i32),
# line weights (i16, knot-major), gain weights (i16, knot-major).
TABLE_SIZES = (LINE_KNOT_COUNT, GAIN_KNOT_COUNT, PLY_COUNT,
               LINE_KNOT_COUNT * ORBIT_COUNT, GAIN_KNOT_COUNT * GAIN_FEATURE_COUNT)
DESIGN_BYTES = 32
FIXED_HEADER_BYTES = 56
TRAILER_BYTES = 32
MAXIMUM_METADATA_BYTES = 16384
MAXIMUM_INTERCEPT = 1 << 24
FILE_SUFFIX = ".poe2net"

_KEY = re.compile(r"[a-z][a-z0-9_]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class PatternGainParameters:
    """Integer parameters of one pattern-gain-v1 net, in file order."""

    line_knots: tuple[int, ...]
    gain_knots: tuple[int, ...]
    intercept: tuple[int, ...]
    line_weights: tuple[int, ...]
    gain_weights: tuple[int, ...]

    def tables(self) -> tuple[tuple[int, ...], ...]:
        return (self.line_knots, self.gain_knots, self.intercept,
                self.line_weights, self.gain_weights)


@dataclass(frozen=True)
class NetFile:
    """One decoded net file and its identity."""

    design: str
    parameters: PatternGainParameters
    metadata: dict[str, str]
    sha256: str

    @property
    def name(self) -> str:
        return self.metadata["name"]

    @property
    def net_id(self) -> str:
        """The 12-digit ID: of the converted header for legacy nets, else of the file."""
        return (self.metadata.get("header_sha256") or self.sha256)[:12]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NetFileError(message)


def _validate_parameters(parameters: PatternGainParameters) -> None:
    for values, size, label in zip(parameters.tables(), TABLE_SIZES, (
            "line knots", "gain knots", "intercepts", "line weights", "gain weights")):
        _require(len(values) == size, f"net {label} must contain {size} values")
        _require(all(isinstance(value, int) and not isinstance(value, bool) for value in values),
                 f"net {label} must be integers")
    for knots, expected, label in ((parameters.line_knots, LINE_KNOTS, "line"),
                                   (parameters.gain_knots, GAIN_KNOTS, "gain")):
        _require(tuple(knots) == expected,
                 f"net {label} knots must be {', '.join(map(str, expected))} in design "
                 f"{PATTERN_GAIN_DESIGN}")
    _require(all(-MAXIMUM_INTERCEPT <= value <= MAXIMUM_INTERCEPT
                 for value in parameters.intercept),
             "net intercepts must lie within +/-2^24")
    _require(all(-(1 << 15) <= value < (1 << 15)
                 for value in parameters.line_weights + parameters.gain_weights),
             "net weights must fit int16")


def encode_metadata(metadata: Mapping[str, str]) -> bytes:
    """Encode metadata as sorted "key=value" lines, the one canonical form."""
    _require("name" in metadata and bool(metadata["name"]), "net metadata needs a name")
    lines = []
    for key in sorted(metadata):
        value = metadata[key]
        _require(isinstance(value, str), f"net metadata {key} must be text")
        _require(_KEY.fullmatch(key) is not None, f"net metadata key is malformed: {key!r}")
        _require(not any(ord(character) < 0x20 or ord(character) == 0x7f
                         for character in value),
                 f"net metadata {key} holds a control character")
        lines.append(f"{key}={value}\n")
    encoded = "".join(lines).encode()
    _require(len(encoded) <= MAXIMUM_METADATA_BYTES, "net metadata is too large")
    if "header_sha256" in metadata:
        _require(_SHA256.fullmatch(metadata["header_sha256"]) is not None,
                 "net metadata header_sha256 must be a SHA-256 hex digest")
    return encoded


def encode_net(parameters: PatternGainParameters, metadata: Mapping[str, str]) -> bytes:
    """Return the canonical bytes of a pattern-gain-v1 net file."""
    _validate_parameters(parameters)
    metadata_bytes = encode_metadata(metadata)
    design = PATTERN_GAIN_DESIGN.encode()
    header = (MAGIC + struct.pack("<II", CONTAINER_VERSION,
                                  FIXED_HEADER_BYTES + 4 * len(TABLE_SIZES) + 4) +
              design + bytes(DESIGN_BYTES - len(design)) +
              struct.pack("<II", FRACTIONAL_BITS, len(TABLE_SIZES)) +
              struct.pack(f"<{len(TABLE_SIZES)}I", *TABLE_SIZES) +
              struct.pack("<I", len(metadata_bytes)))
    body = (header +
            struct.pack(f"<{LINE_KNOT_COUNT}B", *parameters.line_knots) +
            struct.pack(f"<{GAIN_KNOT_COUNT}B", *parameters.gain_knots) +
            struct.pack(f"<{PLY_COUNT}i", *parameters.intercept) +
            struct.pack(f"<{len(parameters.line_weights)}h", *parameters.line_weights) +
            struct.pack(f"<{len(parameters.gain_weights)}h", *parameters.gain_weights) +
            metadata_bytes)
    return body + hashlib.sha256(body).digest()


def decode_metadata(data: bytes) -> dict[str, str]:
    """Decode canonical metadata lines, rejecting any other encoding."""
    try:
        text = data.decode()
    except UnicodeDecodeError as error:
        raise NetFileError("net metadata is not UTF-8") from error
    _require(text == "" or text.endswith("\n"), "net metadata must end each line with a newline")
    metadata: dict[str, str] = {}
    previous = ""
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        _require(bool(separator), f"net metadata line has no '=': {line!r}")
        _require(key > previous, "net metadata keys must be unique and in ascending order")
        metadata[key] = value
        previous = key
    _require(encode_metadata(metadata) == data, "net metadata is malformed")
    return metadata


def decode_net(data: bytes) -> NetFile:
    """Decode and validate one net file, as the engine does."""
    _require(data[:len(MAGIC)] == MAGIC, "not a PoE2 net file")
    _require(len(data) >= FIXED_HEADER_BYTES + TRAILER_BYTES, "net file is truncated")
    version, header_bytes = struct.unpack_from("<II", data, 8)
    _require(version == CONTAINER_VERSION,
             f"net file uses container format {version}; this reader reads format "
             f"{CONTAINER_VERSION}")
    _require(len(data) >= header_bytes + TRAILER_BYTES, "net file is truncated")
    body = data[:-TRAILER_BYTES]
    _require(hashlib.sha256(body).digest() == data[-TRAILER_BYTES:],
             "net file is damaged: its integrity check does not match its contents")
    design_field = data[16:16 + DESIGN_BYTES]
    design = design_field.rstrip(b"\0").decode("ascii", errors="replace")
    _require(design == PATTERN_GAIN_DESIGN,
             f"net design {design!r} is not supported (supported: {PATTERN_GAIN_DESIGN})")
    fractional_bits, table_count = struct.unpack_from("<II", data, 48)
    _require(header_bytes == FIXED_HEADER_BYTES + 4 * table_count + 4,
             "net file header is malformed")
    sizes = struct.unpack_from(f"<{table_count}I", data, 56)
    (metadata_size,) = struct.unpack_from("<I", data, 56 + 4 * table_count)
    _require(fractional_bits == FRACTIONAL_BITS,
             f"net design {design} requires {FRACTIONAL_BITS} fractional bits")
    _require(tuple(sizes) == TABLE_SIZES, f"net table sizes do not match design {design}")
    offset = header_bytes
    tables = []
    for size, code, width in zip(TABLE_SIZES, "BBihh", (1, 1, 4, 2, 2)):
        _require(offset + size * width <= len(body), "net file is truncated")
        tables.append(tuple(struct.unpack_from(f"<{size}{code}", body, offset)))
        offset += size * width
    _require(offset + metadata_size == len(body),
             "net file size does not match its header")
    parameters = PatternGainParameters(*tables)
    _validate_parameters(parameters)
    metadata = decode_metadata(body[offset:])
    return NetFile(design=design, parameters=parameters, metadata=metadata,
                   sha256=hashlib.sha256(data).hexdigest())


def read_net(path: Path) -> NetFile:
    """Read and validate one net file."""
    try:
        data = path.read_bytes()
    except OSError as error:
        raise NetFileError(f"could not read net file {path}: {error}") from error
    try:
        return decode_net(data)
    except NetFileError as error:
        raise NetFileError(f"net file {path}: {error}") from error
    except struct.error as error:
        raise NetFileError(f"net file {path} is truncated") from error


_HEADER_ARRAY = re.compile(
    r"inline constexpr std::array<std::(?:u?int\d+_t), (\d+)> (k\w+)\{\{(.*?)\}\};", re.S)
_HEADER_STRING = re.compile(r'inline constexpr std::string_view (k\w+) = "([^"]*)";')


def parameters_from_header(text: str) -> tuple[PatternGainParameters, dict[str, str]]:
    """Parse a generated C++ model header into parameters and its provenance strings."""
    arrays = {}
    for size, name, values in _HEADER_ARRAY.findall(text):
        numbers = tuple(int(item) for item in values.replace("\n", " ").split(",")
                        if item.strip())
        _require(len(numbers) == int(size), f"header array {name} has the wrong length")
        arrays[name] = numbers
    strings = dict(_HEADER_STRING.findall(text))
    required = {"kLineKnots", "kGainKnots", "kIntercept", "kLineWeights", "kGainWeights"}
    _require(required <= arrays.keys(), "header lacks pattern-gain tables")
    _require(f"inline constexpr int kFractionalBits = {FRACTIONAL_BITS};" in text,
             "header uses a different fixed-point scale")
    parameters = PatternGainParameters(
        line_knots=arrays["kLineKnots"], gain_knots=arrays["kGainKnots"],
        intercept=arrays["kIntercept"], line_weights=arrays["kLineWeights"],
        gain_weights=arrays["kGainWeights"])
    _validate_parameters(parameters)
    return parameters, strings
