from django.test import SimpleTestCase

from wsd.templatetags.wsd_extras import pos_name


class PosNameTests(SimpleTestCase):
    def test_known_codes(self) -> None:
        self.assertEqual(pos_name("n"), "noun")
        self.assertEqual(pos_name("s"), "adjective")

    def test_unknown_or_missing_codes(self) -> None:
        self.assertEqual(pos_name("x"), "x")
        self.assertEqual(pos_name(None), "")
