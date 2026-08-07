import argparse
import asyncio

import pytest

import kvs_api_shim
from kvs_api_shim import KvsError, folder_type, set_many_verified, set_verified


class FakeClient:
    """Just enough of KvsClient for the write helpers."""

    def __init__(self, fail_sets=0, corrupt_key=None):
        self.store = {}
        self.fail_sets = fail_sets  # KvsErrors to raise from set() before succeeding
        self.corrupt_key = corrupt_key  # key whose readback never matches
        self.set_calls = 0

    async def set(self, folder, key, value):
        self.set_calls += 1
        if self.fail_sets > 0:
            self.fail_sets -= 1
            raise KvsError("device locked")
        stored = "corrupted" if key == self.corrupt_key else value
        self.store[(folder, key)] = stored

    async def get(self, folder, key):
        return self.store[(folder, key)]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def instant(_seconds):
        pass

    monkeypatch.setattr(kvs_api_shim.asyncio, "sleep", instant)


def test_set_verified_success():
    client = FakeClient()
    readback = asyncio.run(set_verified(client, "F", "exc", "4.53,nominal"))
    assert readback == "4.53,nominal"
    assert client.set_calls == 1


def test_set_verified_mismatch_returns_readback():
    client = FakeClient(corrupt_key="exc")
    readback = asyncio.run(set_verified(client, "F", "exc", "4.53,nominal"))
    assert readback == "corrupted"
    assert readback != "4.53,nominal"
    assert client.set_calls == 1  # a mismatch is not retried


def test_set_verified_retries_transient_error():
    client = FakeClient(fail_sets=2)
    readback = asyncio.run(set_verified(client, "F", "exc", "4.53", attempts=3))
    assert readback == "4.53"
    assert client.set_calls == 3


def test_set_verified_reraises_after_retries():
    client = FakeClient(fail_sets=10)
    with pytest.raises(KvsError):
        asyncio.run(set_verified(client, "F", "exc", "4.53", attempts=2))
    assert client.set_calls == 2


def test_set_verified_rejects_zero_attempts():
    with pytest.raises(ValueError):
        asyncio.run(set_verified(FakeClient(), "F", "exc", "4.53", attempts=0))


def test_set_many_verified():
    client = FakeClient(corrupt_key="bad")
    entries = {"a": "1", "bad": "2", "c": "3"}
    readbacks = asyncio.run(set_many_verified(client, "F", entries))
    assert readbacks == {"a": "1", "bad": "corrupted", "c": "3"}


def test_folder_type():
    assert folder_type("f") == "F"
    assert folder_type("U") == "U"
    with pytest.raises(argparse.ArgumentTypeError):
        folder_type("X")
