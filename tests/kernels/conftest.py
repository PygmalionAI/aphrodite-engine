import pytest

from aphrodite.common.utils import (create_kv_caches_with_random,
                                    create_kv_caches_with_random_flash)


@pytest.fixture()
def kv_cache_factory():
    return create_kv_caches_with_random


@pytest.fixture()
def kv_cache_factory_flashinfer():
    return create_kv_caches_with_random_flash
