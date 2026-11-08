"""Small deterministic supplement for named-city references."""
from itertools import permutations


def city_dialogues():
    names = ('Bengaluru', 'Delhi', 'London', 'Mumbai', 'New York City', 'Paris, France', 'Tokyo', 'Toronto')
    for origin, destination in permutations(names, 2):
        for turns in (
            (f'I am going from {origin} to {destination}.',
             f'How long will you stay in {destination}?',
             'For three days. Where did I say I was starting?',
             f'You said you were starting in {origin}.'),
            (f'I will meet a friend in {origin}.',
             f'When will you meet your friend in {origin}?',
             f'On Friday, but I meant {destination}.',
             f'Got it. You will meet your friend in {destination} on Friday.'),
            (f'We leave {origin} on Monday and reach {destination} on Tuesday.',
             f'You will arrive in {destination} on Tuesday.',
             'Which city are we leaving on Monday?',
             f'You are leaving {origin} on Monday.'),
        ):
            yield [dict(role='user' if i % 2 == 0 else 'assistant', content=content)
                   for i, content in enumerate(turns)]
