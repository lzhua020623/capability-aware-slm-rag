"""The compatibility runner must normalize representation, not hide source changes."""
import ast
import copy
import unittest

from scripts.run_answer_repair_portable_tests import canonical_dump


class PortableAstTests(unittest.TestCase):
    def test_empty_added_field_matches_legacy_without_modifying_nodes(self):
        old = ast.parse("def f(x):\n    return x + 1\n")
        expected = canonical_dump(old, include_attributes=False)
        new = copy.deepcopy(old); fn = new.body[0]
        if "type_params" not in fn._fields:
            fn._fields = (*fn._fields, "type_params")
        fn.type_params = []
        self.assertEqual(canonical_dump(new, include_attributes=False), expected)
        self.assertIn("type_params", new.body[0]._fields)
        new.body[0].body[0].value.right.value = 2
        self.assertNotEqual(canonical_dump(new, include_attributes=False), expected)

    def test_nonempty_type_parameters_are_never_hidden(self):
        node = ast.parse("def f(x):\n    return x\n")
        fn = node.body[0]
        if "type_params" not in fn._fields:
            fn._fields = (*fn._fields, "type_params")
        fn.type_params = [ast.Name(id="T", ctx=ast.Load())]
        with self.assertRaisesRegex(AssertionError, "Nonempty type parameters"):
            canonical_dump(node, include_attributes=False)


if __name__ == "__main__":
    unittest.main()
