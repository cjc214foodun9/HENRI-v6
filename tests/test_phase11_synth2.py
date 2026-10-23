"""Phase 11 Step 3: pin NameError/import repair and arity-adaptation.

WHY THIS FILE EXISTS
`G-CEGIS1` (704d623) lifted held-out MBPP 0.2250 -> 0.2750 with value-residual
repair, and left two classes untouched because they carry no repairable value
residual:

  NameError   a template references a stdlib symbol never imported (`sqrt`), or
              a donor variable renamed away. Execution aborts before `return`,
              so there is no `actual` to compare against `expected`.
  TypeError   call arity differs from the assertion's call.

Both classes crashed BEFORE producing a residual, so they earned zero repair
budget. These tests pin the two bounded repairs and, just as important, pin the
REFUSALS: a repair that guesses is worse than no repair.
"""

import ast

import pytest

from henri import cegis as cg


# -- the closed import table ------------------------------------------------

class TestImportTable:
    def test_table_is_closed_and_documented(self):
        assert isinstance(cg.STDLIB_IMPORTS, dict)
        for name, stmt in cg.STDLIB_IMPORTS.items():
            assert name.isidentifier(), name
            assert isinstance(stmt, str) and stmt.startswith(("import ", "from "))

    def test_every_entry_parses_and_binds_the_name(self):
        """A malformed entry would raise at repair time, not import time."""
        for name, stmt in cg.STDLIB_IMPORTS.items():
            tree = ast.parse(stmt)
            bound = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        bound.add(a.asname or a.name.split(".")[0])
                if isinstance(node, ast.ImportFrom):
                    for a in node.names:
                        bound.add(a.asname or a.name)
            assert name in bound, "%s not bound by %r" % (name, stmt)

    def test_table_has_no_nonstdlib_entries(self):
        """Pinning scope: this table must not smuggle in project symbols."""
        for stmt in cg.STDLIB_IMPORTS.values():
            module = stmt.split()[1].split(".")[0]
            assert module not in ("henri", "numpy", "torch"), stmt


# -- name recovery ---------------------------------------------------------

class TestMissingName:
    def test_parses_the_nameerror_shape(self):
        assert cg._missing_name("name 'sqrt' is not defined") == "sqrt"
        assert cg._missing_name("name 'arr' is not defined") == "arr"

    def test_other_messages_yield_none(self):
        """A miss must NOT invent a name."""
        assert cg._missing_name("unsupported operand type(s) for +") is None
        assert cg._missing_name("") is None
        assert cg._missing_name(None) is None

    def test_regex_does_not_match_a_lookalike(self):
        assert cg._missing_name("some 'name' is not defined here") is None


# -- mechanism 1: missing stdlib import ------------------------------------

class TestMissingImport:
    @pytest.mark.parametrize("name,src", [
        ("sqrt", "def f(x):\n    return sqrt(x)\n"),
        ("Counter", "def f(s):\n    return Counter(s)\n"),
        ("reduce", "def f(s):\n    return reduce(lambda a,b:a+b, s)\n"),
        ("heappush", "def f(h,x):\n    heappush(h, x)\n    return h\n"),
    ])
    def test_import_is_prepended_and_parses(self, name, src):
        out = cg.repair_missing_import(src, name)
        assert out is not None
        tree = ast.parse(out)
        # The import must come first so it precedes every reference.
        assert isinstance(tree.body[0], (ast.Import, ast.ImportFrom))
        assert name in out

    def test_unknown_name_returns_none(self):
        """Fabricating an import for an unknown name is the failure to avoid."""
        assert cg.repair_missing_import("def f(x):\n    return Foo(x)\n", "Foo") is None

    def test_already_imported_returns_none(self):
        """If it is already imported the NameError came from elsewhere."""
        src = "import math\ndef f(x):\n    return math.sqrt(x)\n"
        assert cg.repair_missing_import(src, "math") is None

    def test_already_from_imported_returns_none(self):
        src = "from math import sqrt\ndef f(x):\n    return sqrt(x)\n"
        assert cg.repair_missing_import(src, "sqrt") is None

    def test_unparseable_source_returns_none(self):
        assert cg.repair_missing_import("def f(:\n  pass", "sqrt") is None


# -- mechanism 2: unbound donor variable ----------------------------------

class TestUnboundName:
    def test_rebinds_to_first_parameter(self):
        out = cg.repair_unbound_name("def f(lst):\n    return arr[0]\n", "arr")
        assert out is not None
        assert "lst[0]" in out
        assert "arr" not in out

    def test_builtin_is_refused(self):
        """`len` is bound globally; rebinding it would corrupt semantics."""
        assert cg.repair_unbound_name("def f(x):\n    return len(x)\n", "len") is None

    def test_existing_parameter_is_refused(self):
        src = "def f(lst):\n    return lst[0]\n"
        assert cg.repair_unbound_name(src, "lst") is None

    def test_locally_assigned_name_is_refused(self):
        """A use-before-assignment bug is not a donor-variable bug.

        Rebinding would silently paper over a real ordering defect.
        """
        src = "def f(x):\n    y = x\n    return y\n"
        assert cg.repair_unbound_name(src, "y") is None

    def test_no_parameters_is_refused(self):
        assert cg.repair_unbound_name("def f():\n    return arr\n", "arr") is None

    def test_name_absent_from_body_returns_none(self):
        assert cg.repair_unbound_name("def f(lst):\n    return lst\n", "zzz") is None

    def test_non_identifier_is_refused(self):
        assert cg.repair_unbound_name("def f(x):\n    return q\n", "not an ident") is None


# -- classification -------------------------------------------------------

class TestKindClassification:
    def test_nameerror_gets_its_own_class(self):
        r = cg.FailureResidual(index=0, test="", exception_type="NameError",
                              exception_msg="name 'sqrt' is not defined")
        assert r.kind == "name_error"

    def test_other_exceptions_stay_exception(self):
        for et in ("IndexError", "ZeroDivisionError", "AttributeError"):
            r = cg.FailureResidual(index=0, test="", exception_type=et)
            assert r.kind == "exception", et

    def test_to_dict_carries_the_message(self):
        """Auditability: without the name, a receipt cannot show the target."""
        r = cg.FailureResidual(index=-1, test="", exception_type="NameError",
                              exception_msg="name 'sqrt' is not defined")
        d = r.to_dict()
        assert d["kind"] == "name_error"
        assert "sqrt" in (d["exception_msg"] or "")


# -- mutually exclusive mechanisms ----------------------------------------

class TestMutualExclusion:
    def test_stdlib_name_yields_only_the_import(self):
        """MEASURED DEFECT, pinned.

        First version pushed BOTH mechanisms for `sqrt`. `rebind_sqrt` rewrote
        `sqrt(x)` to `x(x)`, which executes and raises
        `'int' object is not callable`, replacing a clean NameError with a
        confusing TypeError and mislabelling the residual class.
        """
        src = "def f(x):\n    return sqrt(x)\n"
        r = cg.FailureResidual(index=0, test="", exception_type="NameError",
                              exception_msg="name 'sqrt' is not defined")
        rows = cg.synthesise_repairs(src, r)
        mechs = [x["mechanism"] for x in rows]
        assert mechs == ["import_sqrt"], mechs
        assert all(x["residual_kind"] == "name_error" for x in rows)

    def test_nonstdlib_name_yields_only_the_rebind(self):
        src = "def f(lst):\n    return arr[0]\n"
        r = cg.FailureResidual(index=0, test="", exception_type="NameError",
                              exception_msg="name 'arr' is not defined")
        rows = cg.synthesise_repairs(src, r)
        assert [x["mechanism"] for x in rows] == ["rebind_arr"]

    def test_unknown_name_with_no_parameter_yields_nothing(self):
        r = cg.FailureResidual(index=0, test="", exception_type="NameError",
                              exception_msg="name 'zzz' is not defined")
        assert cg.synthesise_repairs("def f():\n    return zzz\n", r) == []


# -- arity adaptation (the TypeError class) -------------------------------

class TestArityAdaptation:
    def test_signature_is_read_from_the_assertion(self):
        name, arity, nkw = cg.__dict__.get("target_signature",
                                           lambda t: (None, None, 0))([])
        from henri.code_synth import target_signature
        assert target_signature(["assert add(1, 2) == 3"]) == ("add", 2, 0)
        assert target_signature(["assert f(x) == 1"]) == ("f", 1, 0)

    def test_keyword_call_is_reported_not_guessed(self):
        from henri.code_synth import target_signature
        name, arity, nkw = target_signature(["assert f(a, b=2) == 3"])
        assert (name, arity, nkw) == ("f", 1, 1)

    def test_unparseable_test_returns_none(self):
        from henri.code_synth import target_signature
        assert target_signature(["not python"]) == (None, None, 0)
        assert target_signature([]) == (None, None, 0)

    def test_adaptation_keeps_surviving_names_bound(self):
        """MEASURED DEFECT, pinned.

        A first draft replaced every parameter with a synthetic name, so
        `def foo(x, y, z)` became `def add(a, b)` while the body still said
        `x + y + z`. Every adapted candidate then raised NameError for a
        reason unrelated to the operator channel.
        """
        from henri.code_synth import adapt_arity
        src = "def foo(x, y, z):\n    return x + y + z\n"
        out = adapt_arity(src, "add", 2)
        assert out is not None
        assert "def add(x, y):" in out
        assert "x + y + z" in out          # body untouched
        assert "NameError" not in out

    def test_defaults_are_dropped(self):
        """A stored default can absorb a missing positional silently."""
        from henri.code_synth import adapt_arity
        src = "def foo(a, b=5):\n    return a + b\n"
        out = adapt_arity(src, "add", 2)
        assert out is not None
        assert "b=5" not in out

    def test_arity_below_zero_returns_none(self):
        from henri.code_synth import adapt_arity
        assert adapt_arity("def f(x):\n    return x\n", "f", -1) is None


# -- end to end through the sandbox ---------------------------------------

class TestEndToEnd:
    def test_load_time_nameerror_produces_a_repair_that_passes(self):
        src = "def f(x):\n    return sqrt(x)\n"
        r = cg.probe_failure(src, ["assert f(9) == 3"])
        assert r is not None and r.kind == "name_error"
        rows = cg.synthesise_repairs(src, r)
        assert [x["mechanism"] for x in rows] == ["import_sqrt"]
        assert cg.probe_failure(rows[0]["text"], ["assert f(9) == 3"]) is None

    def test_donor_variable_produces_a_repair_that_passes(self):
        src = "def f(lst):\n    return arr[0]\n"
        r = cg.probe_failure(src, ["assert f([5,6]) == 5"])
        assert r is not None and r.kind == "name_error"
        rows = cg.synthesise_repairs(src, r)
        assert [x["mechanism"] for x in rows] == ["rebind_arr"]
        assert cg.probe_failure(rows[0]["text"], ["assert f([5,6]) == 5"]) is None

    def test_an_already_passing_candidate_yields_no_residual(self):
        src = "def f(x):\n    return x + 1\n"
        assert cg.probe_failure(src, ["assert f(2) == 3"]) is None
