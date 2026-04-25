from typing import assert_type

from strings_bytes import ascii_A, byte_lf, github_key, payload_fragment, tail_or_empty

assert_type(github_key, str)
assert_type(payload_fragment, bytes)
assert_type(ascii_A, str)
assert_type(byte_lf, int)
assert_type(tail_or_empty("abc"), str)
