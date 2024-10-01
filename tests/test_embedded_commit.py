import aphrodite


def test_embedded_commit_defined():
    assert aphrodite.__commit__ != "COMMIT_HASH_PLACEHOLDER"
    # 7 characters is the length of a short commit hash
    assert len(aphrodite.__commit__) >= 7
