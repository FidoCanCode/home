import inspect

from primitives import nat_count_down


def test_nat_count_down_round_trip() -> None:
    assert nat_count_down(0, 5) == 5
    assert nat_count_down(3, 4) == 7


def test_nat_count_down_uses_loop_for_tail_call() -> None:
    source = inspect.getsource(nat_count_down)

    assert nat_count_down(10_000, 0) == 10_000
    assert "while True:" in source
    assert "continue" in source
    assert "return nat_count_down(" not in source
