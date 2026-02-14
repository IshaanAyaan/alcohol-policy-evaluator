import unittest

from src.download.apis import _extract_tokens, _find_state_row_blocks


class TestApisParser(unittest.TestCase):
    def test_state_row_blocks(self):
        text = """
Alabama
AL
1/1/2025
$1.05
3 Citations

Alaska
AK
1/1/2025
$1.07
2 Citations
"""
        tokens = _extract_tokens(text)
        blocks = _find_state_row_blocks(tokens)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0][0], "Alabama")
        self.assertEqual(blocks[1][1], "AK")


if __name__ == "__main__":
    unittest.main()
