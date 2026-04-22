from positive_claim_count import positive_claim_count
from positive_claim_diff import positive_claim_diff
from positive_claim_elements import positive_claim_elements
from positive_claim_has_2 import positive_claim_has_2
from positive_claim_inter import positive_claim_inter
from positive_claim_removed import positive_claim_removed
from positive_claim_set import positive_claim_set
from positive_claim_union import positive_claim_union
from positive_task_count import positive_task_count
from positive_task_elements import positive_task_elements
from positive_task_has_3 import positive_task_has_3
from positive_task_hit import positive_task_hit
from positive_task_map import positive_task_map
from positive_task_missing import positive_task_missing
from positive_task_removed import positive_task_removed
from string_label_elements import string_label_elements
from string_label_hit import string_label_hit
from string_label_map import string_label_map
from string_label_set import string_label_set
from string_label_set_elements import string_label_set_elements


def test_positive_maps_are_native_persistent_dicts() -> None:
    assert positive_task_map == {1: "plan", 3: "ci"}
    assert positive_task_hit == "plan"
    assert positive_task_missing is None
    assert positive_task_removed == {3: "ci"}
    assert positive_task_map == {1: "plan", 3: "ci"}
    assert positive_task_has_3 is True
    assert positive_task_count == 2
    assert positive_task_elements == [(1, "plan"), (3, "ci")]


def test_positive_sets_are_native_persistent_frozensets() -> None:
    assert positive_claim_set == frozenset({2, 5})
    assert positive_claim_union == frozenset({2, 5, 7})
    assert positive_claim_inter == frozenset({5})
    assert positive_claim_diff == frozenset({7})
    assert positive_claim_removed == frozenset({5})
    assert positive_claim_has_2 is True
    assert positive_claim_count == 2
    assert positive_claim_elements == [2, 5]


def test_string_maps_and_sets_have_sorted_views() -> None:
    assert string_label_map == [("alpha", 1), ("beta", 2)]
    assert string_label_hit == 1
    assert string_label_elements == [("alpha", 1), ("beta", 2)]
    labels = ["alpha", "beta"]
    assert string_label_set(labels) == labels
    assert string_label_set_elements(labels) == labels
