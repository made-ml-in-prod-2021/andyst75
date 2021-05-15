import os

from src.utils import make_path, dump_object, load_object
from ..constats import OUTPUT_PATH


def test_make_path():
    path = "test.csv"
    assert path == make_path(path)


def test_dump_and_load():
    path = os.path.join(OUTPUT_PATH, "dump.pkl")
    dump_object(path, path)
    loaded_path = load_object(path)
    assert path == loaded_path
    os.remove(path)
