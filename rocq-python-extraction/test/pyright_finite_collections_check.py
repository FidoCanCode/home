from positive_claim_elements import positive_claim_elements
from positive_claim_removed import positive_claim_removed
from positive_claim_set import positive_claim_set
from positive_task_elements import positive_task_elements
from positive_task_hit import positive_task_hit
from positive_task_map import positive_task_map
from string_label_elements import string_label_elements
from string_label_hit import string_label_hit
from string_label_map import string_label_map
from string_label_set import string_label_set
from string_label_set_elements import string_label_set_elements

positive_map_check: dict[int, str] = positive_task_map
positive_hit_check: str | None = positive_task_hit
positive_elements_check: list[tuple[int, str]] = positive_task_elements
positive_set_check: frozenset[int] = positive_claim_set
positive_set_removed_check: frozenset[int] = positive_claim_removed
positive_set_elements_check: list[int] = positive_claim_elements

string_map_check: list[tuple[str, int]] = string_label_map
string_hit_check: int | None = string_label_hit
string_elements_check: list[tuple[str, int]] = string_label_elements
string_set_check: list[str] = string_label_set(["alpha", "beta"])
string_set_elements_check: list[str] = string_label_set_elements(["alpha", "beta"])
