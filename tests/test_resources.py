from pathlib import Path

import pytest

from mixsiarpy import get_resource_path, list_resources


@pytest.mark.parametrize(
    "collection,filename",
    [
        ("data", "wolves_consumer.csv"),
        ("examples", "wolves.py"),
        ("docs", "VALIDATION.md"),
        ("reference_r", "DESCRIPTION"),
        ("validation", "README.md"),
    ],
)
def test_resource_collections(collection, filename):
    path = get_resource_path(collection, filename)
    assert path.is_file()
    assert path in list_resources(collection)


def test_unknown_resource_collection():
    with pytest.raises(ValueError, match="Unknown resource"):
        get_resource_path("not-a-collection")
