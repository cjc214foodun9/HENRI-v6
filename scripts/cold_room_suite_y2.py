"""The Cycle Y2 clean-room cold-start suite (16 tasks + 4 spec traps).

Every task is a self-contained HumanEval/MBPP-style prompt: a signature, a
docstring specification and two or three PUBLIC assertions.  The hidden
assertions live only in this file and never enter the synthesizer, so a
candidate that is emitted after passing the public examples is scored on
behaviour it has not seen.

The suite is authored by the same engineering loop that authored the
synthesizer, so the pass@1 it measures is an upper bound on the clean-room
number, not an unbiased estimate.  The receipt says so.
"""

from __future__ import annotations

from typing import Any, Dict, List

TASKS: List[Dict[str, Any]] = [
    {
        "name": "count_even",
        "prompt": '''def count_even(nums: list) -> int:
    """Return the number of even values in nums.

    assert count_even([1, 2, 4]) == 2
    assert count_even([2, 2]) == 2
    assert count_even([1, 3]) == 0
    """
''',
        "hidden": [
            "assert count_even([]) == 0",
            "assert count_even([0]) == 1",
            "assert count_even([1, 2, 3, 4]) == 2",
            "assert count_even([2, 4, 6, 8]) == 4",
        ],
    },
    {
        "name": "sum_positive",
        "prompt": '''def sum_positive(values: list) -> int:
    """Return the sum of the positive values in values.

    assert sum_positive([1, -2, 3]) == 4
    assert sum_positive([-1, -2]) == 0
    assert sum_positive([5]) == 5
    """
''',
        "hidden": [
            "assert sum_positive([]) == 0",
            "assert sum_positive([0, 0]) == 0",
            "assert sum_positive([2, 3, -10]) == 5",
            "assert sum_positive([-5, 10, 20]) == 30",
        ],
    },
    {
        "name": "reverse_words",
        "prompt": '''def reverse_words(text: str) -> str:
    """Return the words of text in reverse order, joined by one space.

    assert reverse_words("a b c") == "c b a"
    assert reverse_words("one two") == "two one"
    assert reverse_words("solo") == "solo"
    """
''',
        "hidden": [
            "assert reverse_words('x y z w') == 'w z y x'",
            "assert reverse_words('hello world') == 'world hello'",
            "assert reverse_words('a') == 'a'",
            "assert reverse_words('p q') == 'q p'",
        ],
    },
    {
        "name": "is_palindrome",
        "prompt": '''def is_palindrome(text: str) -> bool:
    """Return True when text reads the same backwards.

    assert is_palindrome("abba") == True
    assert is_palindrome("abc") == False
    assert is_palindrome("a") == True
    """
''',
        "hidden": [
            "assert is_palindrome('racecar') == True",
            "assert is_palindrome('ab') == False",
            "assert is_palindrome('') == True",
            "assert is_palindrome('xyyx') == True",
        ],
    },
    {
        "name": "max_value",
        "prompt": '''def max_value(nums: list) -> int:
    """Return the largest value in nums.

    assert max_value([1, 5, 3]) == 5
    assert max_value([-2, -7]) == -2
    assert max_value([4]) == 4
    """
''',
        "hidden": [
            "assert max_value([0, 0]) == 0",
            "assert max_value([9, 1, 2]) == 9",
            "assert max_value([-1, -2, -3]) == -1",
            "assert max_value([7, 7]) == 7",
        ],
    },
    {
        "name": "dedupe",
        "prompt": '''def dedupe(nums: list) -> list:
    """Return the values of nums without duplicates, keeping the first occurrence order.

    assert dedupe([1, 1, 2]) == [1, 2]
    assert dedupe([2, 1, 2]) == [2, 1]
    assert dedupe([]) == []
    """
''',
        "hidden": [
            "assert dedupe([3, 3, 3]) == [3]",
            "assert dedupe([1, 2, 3]) == [1, 2, 3]",
            "assert dedupe([1, 2, 1, 3, 2]) == [1, 2, 3]",
            "assert dedupe(['a', 'a']) == ['a']",
        ],
    },
    {
        "name": "count_vowels",
        "prompt": '''def count_vowels(text: str) -> int:
    """Return the number of vowels in text.

    assert count_vowels("abc") == 1
    assert count_vowels("aeiou") == 5
    assert count_vowels("xyz") == 0
    """
''',
        "hidden": [
            "assert count_vowels('') == 0",
            "assert count_vowels('AEIOU') == 5",
            "assert count_vowels('hello') == 2",
            "assert count_vowels('rhythm') == 0",
        ],
    },
    {
        "name": "factorial",
        "prompt": '''def factorial(n: int) -> int:
    """Return n factorial, with factorial(0) equal to one.

    assert factorial(0) == 1
    assert factorial(1) == 1
    assert factorial(4) == 24
    """
''',
        "hidden": [
            "assert factorial(3) == 6",
            "assert factorial(5) == 120",
            "assert factorial(2) == 2",
            "assert factorial(6) == 720",
        ],
    },
    {
        "name": "gcd",
        "prompt": '''def gcd(a: int, b: int) -> int:
    """Return the greatest common divisor of a and b.

    assert gcd(12, 8) == 4
    assert gcd(7, 13) == 1
    assert gcd(10, 10) == 10
    """
''',
        "hidden": [
            "assert gcd(18, 24) == 6",
            "assert gcd(1, 1) == 1",
            "assert gcd(100, 75) == 25",
            "assert gcd(0, 5) == 5",
        ],
    },
    {
        "name": "digit_sum",
        "prompt": '''def digit_sum(n: int) -> int:
    """Return the sum of the decimal digits of n.

    assert digit_sum(123) == 6
    assert digit_sum(9) == 9
    assert digit_sum(1000) == 1
    """
''',
        "hidden": [
            "assert digit_sum(0) == 0",
            "assert digit_sum(456) == 15",
            "assert digit_sum(999) == 27",
            "assert digit_sum(10) == 1",
        ],
    },
    {
        "name": "fibonacci",
        "prompt": '''def fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number, with fibonacci(0) equal to 0.

    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(6) == 8
    """
''',
        "hidden": [
            "assert fibonacci(2) == 1",
            "assert fibonacci(3) == 2",
            "assert fibonacci(7) == 13",
            "assert fibonacci(10) == 55",
        ],
    },
    {
        "name": "flatten",
        "prompt": '''def flatten(matrix: list) -> list:
    """Return a flat list holding every value of the nested lists of matrix.

    assert flatten([[1, 2], [3]]) == [1, 2, 3]
    assert flatten([]) == []
    assert flatten([[4]]) == [4]
    """
''',
        "hidden": [
            "assert flatten([[1], [2], [3]]) == [1, 2, 3]",
            "assert flatten([[], []]) == []",
            "assert flatten([[1, 2, 3]]) == [1, 2, 3]",
            "assert flatten([[5, 6], []]) == [5, 6]",
        ],
    },
    {
        "name": "is_prime",
        "prompt": '''def is_prime(n: int) -> bool:
    """Return True when n is a prime number.

    assert is_prime(7) == True
    assert is_prime(8) == False
    assert is_prime(2) == True
    """
''',
        "hidden": [
            "assert is_prime(1) == False",
            "assert is_prime(13) == True",
            "assert is_prime(25) == False",
            "assert is_prime(97) == True",
        ],
    },
    {
        "name": "word_lengths",
        "prompt": '''def word_lengths(words: list) -> list:
    """Return the length of every word in words, in the same order.

    assert word_lengths(["ab", "c"]) == [2, 1]
    assert word_lengths([]) == []
    assert word_lengths(["abc"]) == [3]
    """
''',
        "hidden": [
            "assert word_lengths(['a', 'bb', 'ccc']) == [1, 2, 3]",
            "assert word_lengths(['xy']) == [2]",
            "assert word_lengths(['', 'x']) == [0, 1]",
            "assert word_lengths(['abcd', 'e']) == [4, 1]",
        ],
    },
    {
        "name": "sort_by_length",
        "prompt": '''def sort_by_length(words: list) -> list:
    """Return the words sorted by length, shortest first.

    assert sort_by_length(["ccc", "a", "bb"]) == ["a", "bb", "ccc"]
    assert sort_by_length([]) == []
    assert sort_by_length(["x"]) == ["x"]
    """
''',
        "hidden": [
            "assert sort_by_length(['aa', 'b']) == ['b', 'aa']",
            "assert sort_by_length(['ddd', 'ee', 'f']) == ['f', 'ee', 'ddd']",
            "assert sort_by_length(['ab', 'cd']) == ['ab', 'cd']",
            "assert sort_by_length(['a', 'bbb']) == ['a', 'bbb']",
        ],
    },
    {
        "name": "median",
        "prompt": '''def median(nums: list) -> float:
    """Return the median of nums.

    assert median([1, 3, 2]) == 2
    assert median([1, 2, 3, 4]) == 2.5
    assert median([5]) == 5
    """
''',
        "hidden": [
            "assert median([1, 2, 3]) == 2",
            "assert median([4, 1, 3, 2]) == 2.5",
            "assert median([7, 7, 7]) == 7",
            "assert median([10, 20]) == 15",
        ],
    },
]

TRAPS: List[Dict[str, Any]] = [
    {
        "name": "contradictory_examples",
        "kind": "contradiction",
        "prompt": '''def contradict(x: int) -> int:
    """Return x plus one.

    assert contradict(2) == 3
    assert contradict(2) == 4
    """
''',
    },
    {
        "name": "contradictory_return_capability",
        "kind": "type_contradiction",
        "prompt": '''def text_of_number(x: int) -> int:
    """Return the text form of x as an int.

    assert text_of_number(3) == "3"
    """
''',
    },
    {
        "name": "missing_external_capability",
        "kind": "unsatisfiable_capability",
        "prompt": '''def fetch_price(symbol: str) -> float:
    """Download the latest stock price from the internet for symbol.

    assert fetch_price("AAA") == 1.0
    """
''',
    },
    {
        "name": "contradictory_return_type",
        "kind": "type_contradiction",
        "prompt": '''def min_max(nums: list) -> int:
    """Return both the smallest and the largest value of nums as one int.

    assert min_max([1, 2]) == (1, 2)
    """
''',
    },
]


def runnable(task: Dict[str, Any], source: str, tests: List[str]) -> str:
    """One sandbox program: the emitted source plus the assertions."""
    return str(source) + "\n" + "\n".join(str(line) for line in tests) + "\n"
