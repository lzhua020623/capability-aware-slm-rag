"""Run frozen tests with Python 3.11's AST schema in this TEST process only.

Python 3.12 added an empty type_params field to ordinary function/class ASTs;
Python 3.13 changed dumping of empty/optional fields. Serialize all legacy
fields explicitly and omit only the new empty type_params field.
Nonempty type parameters are rejected rather than silently discarded. Nothing
in the frozen bundle or any model-inference process is patched or rewritten.
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

def canonical_dump(node, annotate_fields=True, include_attributes=False, *, indent=None):
    if not isinstance(node, ast.AST) or indent is not None:
        raise AssertionError("The frozen fingerprint uses an unindented AST dump")

    def render(value):
        if isinstance(value, ast.AST):
            fields = []; keywords = annotate_fields
            for name in value._fields:
                try:
                    child = getattr(value, name)
                except AttributeError:
                    keywords = True
                    continue
                if name == "type_params" and isinstance(value, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if child:
                        raise AssertionError("Nonempty type parameters cannot be compared using the frozen Python 3.11 AST schema")
                    continue
                if child is None and getattr(type(value), name, ...) is None:
                    keywords = True
                    continue
                fields.append((name + "=" if keywords else "") + render(child))
            if include_attributes:
                fields.extend(name + "=" + render(getattr(value, name)) for name in value._attributes
                              if hasattr(value, name) and not (getattr(value, name) is None
                                  and getattr(type(value), name, ...) is None))
            return type(value).__name__ + "(" + ", ".join(fields) + ")"
        if isinstance(value, list):
            return "[" + ", ".join(render(item) for item in value) + "]"
        return repr(value)

    return render(node)


def main():
    # The notebook launches this external adapter with cwd set to the verified
    # frozen bundle. Make that bundle's tests/imports take precedence.
    if not (Path.cwd() / "bundle_checksums.json").is_file():
        raise AssertionError("Run the test adapter from the verified frozen bundle")
    sys.path.insert(0, str(Path.cwd()))
    ast.dump = canonical_dump
    print("Frozen preflight AST schema: Python 3.11; only empty type_params omitted; test process only.", flush=True)
    unittest.main(module=None, argv=sys.argv)


if __name__ == "__main__":
    main()
