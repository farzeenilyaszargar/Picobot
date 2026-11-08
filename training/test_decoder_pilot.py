import json
from pathlib import Path
import unittest

from city_references import CITY_SLOTS, CityReferences
from prepare_decoder import clean, short_chat


class CityPilotTests(unittest.TestCase):
    def test_browser_fixtures_match_preparation(self):
        fixture = json.loads(Path(__file__).with_name('picoboi_10m_city_cases.json').read_text())
        parser = CityReferences(fixture['aliases'], atomic=True)
        for case in fixture['cases']:
            messages = [dict(role=role, content=content) for role, content, _ in case]
            self.assertEqual([m['content'] for m in parser.normalize(messages)],
                             [expected for _, _, expected in case])

    def test_literal_controls_are_rejected_before_normalization(self):
        for slot in CITY_SLOTS:
            self.assertIsNone(clean(f'Please answer with {slot}.'))

    def test_short_chat_filter(self):
        chat = [dict(role='user', content='Did you get home?'),
                dict(role='assistant', content='Yes, a moment ago.')]
        self.assertTrue(short_chat(chat))
        self.assertFalse(short_chat(chat[::-1]))
        self.assertFalse(short_chat([dict(role='user', content='Write a Python function.'), chat[1]]))


if __name__ == '__main__':
    unittest.main()
