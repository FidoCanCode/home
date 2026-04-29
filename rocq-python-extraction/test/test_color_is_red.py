import inspect

import datatypes
from datatypes import Blue, Color, Green, Red, color_is_red, color_tag_matches_filter


def test_color_is_red_round_trip() -> None:
    assert color_is_red(Red()) is True, "color_is_red(Red()): got " + repr(
        color_is_red(Red())
    )
    assert color_is_red(Green()) is False, "color_is_red(Green()): got " + repr(
        color_is_red(Green())
    )
    assert color_is_red(Blue()) is False, "color_is_red(Blue()): got " + repr(
        color_is_red(Blue())
    )
    assert isinstance(Red(), Color), (
        "Red() must be instance of Color (capitalized base class)"
    )


def test_color_is_red_lowers_to_direct_type_check() -> None:
    source = inspect.getsource(color_is_red)

    assert "match c:" not in source
    assert "isinstance(c, Red)" in source


def test_constructor_tag_predicate_helper_is_inlined() -> None:
    assert color_tag_matches_filter(True, Red()) is True
    assert color_tag_matches_filter(True, Green()) is False
    assert color_tag_matches_filter(False, Red()) is False
    assert color_tag_matches_filter(False, Green()) is True

    module_source = inspect.getsource(datatypes)
    filter_source = inspect.getsource(color_tag_matches_filter)

    assert "def color_is_red_tag(" not in module_source
    assert "color_is_red_tag(" not in filter_source
    assert "isinstance(c, Red)" in filter_source
    assert "not isinstance(c, Red)" in filter_source


def test_generated_dataclasses_are_final() -> None:
    module_source = inspect.getsource(datatypes)

    assert "    final,\n" in module_source
    assert "@final\n@dataclass(frozen=True)\nclass Red(Color):" in module_source
