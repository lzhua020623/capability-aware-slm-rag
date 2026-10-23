"""Rules are search paraphrases, never an answer or a Gold lookup."""

import unittest

from src.compensation.retrieval_enhancement import RuleQueryRewriter, filter_rewrites


class QueryRewritingTests(unittest.TestCase):
    def setUp(self):
        self.rewriter = RuleQueryRewriter()

    def test_simple_natural_questions_rules(self):
        cases = {
            "Where was Ada Lovelace born?": ["Ada Lovelace birthplace"],
            "When was Ada Lovelace born?": ["Ada Lovelace date of birth"],
            "Who wrote Hamlet?": ["Hamlet author"],
            "When was Acme founded?": ["Acme founding date"],
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertEqual(self.rewriter.rewrite(query, "natural_questions"), expected)
        self.assertEqual(self.rewriter.llm_calls_per_rewrite, 0)

    def test_entities_numbers_and_quoted_titles_are_preserved(self):
        self.assertEqual(self.rewriter.rewrite('Who wrote "1984"?', "nq"), ['"1984" author'])
        self.assertEqual(self.rewriter.rewrite("Where was Jean-Luc Picard born?", "nq"),
                         ["Jean-Luc Picard birthplace"])

    def test_unknown_or_complex_queries_are_skipped(self):
        queries = ["Explain how engines work.", "Who won the award?",
                   "Who wrote Hamlet in 1603?", "Who wrote Hamlet before 1984?",
                   "Who wrote Hamlet with another author?", "Who wrote Hamlet and Macbeth?",
                   "When was Acme, which is a company, founded?", "Who wrote Hamlet; explain why?",
                   "Where was Ada Lovelace born after 1800?", "Who wrote only Hamlet?"]
        for query in queries:
            with self.subTest(query=query):
                self.assertEqual(self.rewriter.rewrite(query, "nq"), [])

    def test_negation_is_not_removed_by_rules(self):
        for query in ["Who wrote a book without illustrations?", "Who wrote a book not published?",
                      "Who wrote a book that wasn't published?", "Who wrote No Country for Old Men?"]:
            with self.subTest(query=query):
                self.assertEqual(self.rewriter.rewrite(query, "nq"), [])

    def test_fever_and_unknown_datasets_are_not_rewritten(self):
        for dataset in ["fever", "other"]:
            self.assertEqual(self.rewriter.rewrite("Who wrote Hamlet?", dataset), [])

    def test_empty_duplicate_and_original_outputs_are_removed(self):
        query = "Where was Ada Lovelace born?"
        outputs = ["", " \t", query, query.upper(), "Ada Lovelace birthplace",
                   "  Ada\tLovelace birthplace  ", "ada lovelace BIRTHPLACE",
                   "Ada Lovelace birth location"]
        self.assertEqual(filter_rewrites(query, outputs),
                         ["Ada Lovelace birthplace", "Ada Lovelace birth location"])

    def test_at_most_two_unique_queries(self):
        self.assertEqual(filter_rewrites("Original", ["One", "Two", "Three"]), ["One", "Two"])

    def test_unicode_and_whitespace_duplicate_comparison(self):
        self.assertEqual(filter_rewrites("Who wrote Caf\u00e9?",
                                         ["Cafe\u0301 author", "Caf\u00e9\u00a0author"]), ["Caf\u00e9 author"])

    def test_numeric_constraints_cannot_be_added_or_removed(self):
        self.assertEqual(filter_rewrites("Who wrote 1984?",
                                         ["book author", "1985 author", "1984 1949 author", "1984 author"]),
                         ["1984 author"])

    def test_negation_cannot_be_added_or_removed(self):
        self.assertEqual(filter_rewrites("Which book was not published?",
                                         ["book published", "book never published", "book not published"]),
                         ["book not published"])
        self.assertEqual(filter_rewrites("Who wrote Hamlet?", ["Hamlet not author"]), [])

    def test_invalid_outputs_report_errors(self):
        for output in [None, "Hamlet author", {}, ["valid", None], ["One", "Two", 3]]:
            with self.subTest(output=output), self.assertRaises(TypeError):
                filter_rewrites("Who wrote Hamlet?", output)
        for query in [None, "", " \t"]:
            with self.subTest(query=query), self.assertRaises(ValueError):
                self.rewriter.rewrite(query, "nq")


if __name__ == "__main__":
    unittest.main()
