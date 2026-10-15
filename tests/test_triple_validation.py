import unittest

from evidenceloop.validation import clean_predicate, predicate_validation_error


class TripleValidationTests(unittest.TestCase):
    def test_clean_predicate_trims_and_folds_whitespace(self):
        self.assertEqual(clean_predicate("  place   of\tbirth  "), "place of birth")

    def test_predicate_validation_accepts_single_line_up_to_80_characters(self):
        self.assertEqual(predicate_validation_error("natural language relation"), "")
        self.assertEqual(predicate_validation_error("x" * 80), "")

    def test_predicate_validation_rejects_empty_multiline_and_overlong_values(self):
        self.assertIn("required", predicate_validation_error(" \t "))
        self.assertIn("single line", predicate_validation_error("line one\nline two"))
        self.assertIn("single line", predicate_validation_error("line one\rline two"))
        self.assertIn("exceeds 80 characters", predicate_validation_error("x" * 81))


if __name__ == "__main__":
    unittest.main()
