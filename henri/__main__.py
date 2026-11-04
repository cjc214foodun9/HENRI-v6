"""Module entry point for `python -m henri <command>`.

The Harness plugin invokes this form, so it must exist. Without the file,
`python -m henri` fails with "a package cannot be directly executed" and all
eight tools stop working without an obvious error.
"""

from __future__ import annotations

import sys

from henri.cli import main

if __name__ == "__main__":
    sys.exit(main())
