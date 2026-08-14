"""Locate package data, examples, documentation, and R reference files."""

from pathlib import Path

_VALID_COLLECTIONS = {"data", "examples", "docs", "reference_r", "validation", "gui"}


def get_resource_path(collection=None, *parts, must_exist=True):
    """Return a path to an installed or source-tree project resource.

    Parameters
    ----------
    collection : str or None
        One of ``data``, ``examples``, ``docs``, ``reference_r`` or
        ``validation``. With ``None``, return the common resource root.
    *parts : str
        Optional path components below the collection.
    must_exist : bool
        Raise ``FileNotFoundError`` when the requested resource is absent.
    """
    if collection is not None and collection not in _VALID_COLLECTIONS:
        raise ValueError(f"Unknown resource collection: {collection!r}")
    package_root = Path(__file__).resolve().parent
    suffix = (() if collection is None else (collection,)) + tuple(parts)
    candidate = package_root.joinpath(*suffix)
    if candidate.exists():
        return candidate
    if must_exist:
        raise FileNotFoundError(f"MixSIARPy resource was not found: {'/'.join(suffix)}")
    return candidate


def list_resources(collection):
    """List installed files in a resource collection recursively."""
    root = get_resource_path(collection)
    return sorted(path for path in root.rglob("*") if path.is_file())
