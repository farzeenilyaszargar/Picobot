"""Generate bounded pilot calculator traces from exact Python rational arithmetic."""
from fractions import Fraction
import json
import random


def calculator_traces():
    rng = random.Random(100)
    # The preparer splits complete records before tokenization.
    for _ in range(300):
        a, b = rng.sample(range(2, 100), 2)
        symbol, verb, value = rng.choice([
            ('+', 'plus', Fraction(a + b)), ('-', 'minus', Fraction(a - b)),
            ('*', 'times', Fraction(a * b)), ('/', 'divided by', Fraction(a, b))])
        expression = f'{a} {symbol} {b}'
        result = str(value)
        yield [
            {'role': 'user', 'content': f'what is {a} {verb} {b}?'},
            {'role': 'assistant', 'content': json.dumps({'tool': 'calculator', 'expression': expression},
                                                        separators=(',', ':')), 'call': True},
            {'role': 'tool', 'content': json.dumps({'value': result}, separators=(',', ':'))},
            {'role': 'assistant', 'content': f'{expression} = {result}.'},
        ]
