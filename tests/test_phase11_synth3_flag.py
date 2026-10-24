"""G-SYNTH3: pin the name_error ablation switch so an arm cannot go inert.

WHY THIS FILE EXISTS AND WHAT IT PREVENTS
The G-SYNTH3 gate compares two arms that differ in exactly one setting:
`HENRI_CEGIS_NAME_ERROR_REPAIR=1` (treatment) against `=0` (control), both at
`max_cegis_rounds=2`. If that switch were read once at import and cached, both
arms would execute identical code and the gate could never detect an inert
mechanism. The gate would then report a difference of zero and the reader could
not distinguish "the branch adds nothing" from "the ablation never ran".

These tests pin the live read. They are the engagement fixture for the gate.

DEFAULT MUST STAY ON
`name_error` repair shipped in `4ceabd3` with the branch active. If the default
flipped to off, that commit's receipt would stop describing its own revision.
So this is a DISABLE flag, not a default-OFF feature flag.
"""

import os

import pytest

from henri import cegis as cg

ENV = "HENRI_CEGIS_NAME_ERROR_REPAIR"

MISSING_IMPORT = "def f(x):\n    return sqrt(x)\n"
MISSING_IMPORT_TESTS = ["assert f(9) == 3.0"]


@pytest.fixture(autouse=True)
def _restore_env():
    """Every test starts and ends with a clean environment."""
    saved = os.environ.pop(ENV, None)
    yield
    if saved is None:
        os.environ.pop(ENV, None)
    else:
        os.environ[ENV] = saved


class TestSwitchDefault:
    def test_default_is_enabled(self):
        """Unset must mean ON. `4ceabd3` shipped the branch active."""
        assert ENV not in os.environ
        assert cg.name_error_repair_enabled() is True

    def test_explicit_one_is_enabled(self):
        os.environ[ENV] = "1"
        assert cg.name_error_repair_enabled() is True

    def test_explicit_zero_disables(self):
        os.environ[ENV] = "0"
        assert cg.name_error_repair_enabled() is False

    def test_empty_string_keeps_default_on(self):
        """An empty value is a deployment accident, not an instruction to off."""
        os.environ[ENV] = ""
        assert cg.name_error_repair_enabled() is True


class TestLiveRead:
    def test_switch_flips_within_one_process(self):
        """THE ENGAGEMENT TEST. A cached read fails exactly here.

        Both arms of the ablation run in one process. If the value were bound
        at import time, the second assertion could not pass.
        """
        os.environ[ENV] = "1"
        first = cg.name_error_repair_enabled()
        os.environ[ENV] = "0"
        second = cg.name_error_repair_enabled()
        os.environ[ENV] = "1"
        third = cg.name_error_repair_enabled()
        assert (first, second, third) == (True, False, True)


class TestDispatchObeysSwitch:
    """The flag must change EMITTED CANDIDATES, not just a boolean."""

    def _residual(self):
        r = cg.probe_failure(MISSING_IMPORT, MISSING_IMPORT_TESTS)
        assert r is not None and r.kind == "name_error"
        return r

    def test_enabled_emits_an_import_repair(self):
        os.environ[ENV] = "1"
        rows = cg.synthesise_repairs(MISSING_IMPORT, self._residual())
        assert [x["mechanism"] for x in rows] == ["import_sqrt"]
        assert cg.probe_failure(rows[0]["text"], MISSING_IMPORT_TESTS) is None

    def test_disabled_emits_nothing(self):
        """Zero candidates is the honest control: no targeted fix was offered."""
        os.environ[ENV] = "0"
        rows = cg.synthesise_repairs(MISSING_IMPORT, self._residual())
        assert rows == []

    def test_disabled_does_not_suppress_other_classes(self):
        """The ablation must be surgical.

        A flag that silently disabled every repair class would collapse the
        treatment arm and manufacture a large fake difference.

        MEASURED FIXTURE DEFECT: my first version used `return x + 100` against
        `assert f(2) == 3` and called it `numeric_offset`. Measured kind is
        `unclassified` (offset 97 exceeds the classifier's range), so the test
        failed on the fixture, not the mechanism. The fixture below measures as
        `numeric_offset` before it is used.
        """
        src = "def f(x):\n    return x + 1\n"
        tests = ["assert f(2) == 4"]
        r = cg.probe_failure(src, tests)
        assert r is not None and r.kind == "numeric_offset", r.kind if r else None

        os.environ[ENV] = "1"
        on = cg.synthesise_repairs(src, r)
        assert on, "numeric_offset must produce a candidate when enabled"

        os.environ[ENV] = "0"
        off = cg.synthesise_repairs(src, r)
        assert [x["mechanism"] for x in off] == [x["mechanism"] for x in on]
