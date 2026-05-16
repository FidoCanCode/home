"""Tests for the deep-freeze JSON helper (#1748)."""

import pytest
from frozendict import frozendict

from fido.frozen import deep_freeze, freeze_object


class TestDeepFreeze:
    def test_scalars_pass_through(self) -> None:
        assert deep_freeze("hi") == "hi"
        assert deep_freeze(42) == 42
        assert deep_freeze(3.14) == 3.14
        assert deep_freeze(True) is True
        assert deep_freeze(None) is None

    def test_dict_becomes_frozendict(self) -> None:
        result = deep_freeze({"a": 1, "b": 2})
        assert isinstance(result, frozendict)
        assert dict(result) == {"a": 1, "b": 2}

    def test_list_becomes_tuple(self) -> None:
        result = deep_freeze([1, 2, 3])
        assert result == (1, 2, 3)
        assert isinstance(result, tuple)

    def test_tuple_stays_tuple(self) -> None:
        result = deep_freeze((1, 2, 3))
        assert result == (1, 2, 3)
        assert isinstance(result, tuple)

    def test_nested_structures_recurse(self) -> None:
        result = deep_freeze(
            {"users": [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]}
        )
        assert isinstance(result, frozendict)
        users = result["users"]
        assert isinstance(users, tuple)
        assert isinstance(users[0], frozendict)
        assert users[0]["name"] == "alice"

    def test_strings_are_not_treated_as_sequences(self) -> None:
        # str matches Sequence in the typing hierarchy; deep_freeze
        # must NOT iterate it character-by-character.
        result = deep_freeze("hello")
        assert result == "hello"
        assert isinstance(result, str)

    def test_frozen_inputs_idempotent(self) -> None:
        original = deep_freeze({"a": [1, 2, {"b": "c"}]})
        assert isinstance(original, frozendict)
        again = deep_freeze(original)
        assert again == original


class TestFreezeObject:
    def test_returns_mapping_subclass(self) -> None:
        result = freeze_object({"a": 1})
        assert isinstance(result, frozendict)
        assert result["a"] == 1

    def test_mutation_raises(self) -> None:
        result = freeze_object({"a": 1})
        with pytest.raises(TypeError):
            result["a"] = 2  # type: ignore[index]

    def test_severs_link_to_input_dict(self) -> None:
        # The frozen view must not reflect later mutations to the
        # input dict — otherwise a caller holding the original
        # reference could mutate the cached state through the back
        # door.
        original = {"a": 1}
        result = freeze_object(original)
        original["a"] = 999
        assert result["a"] == 1
