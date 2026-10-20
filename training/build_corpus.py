#!/usr/bin/env python3
"""Controlled synthetic chat data, with an audited closed English vocabulary.

The data is composed from authored patterns, not independently sampled LLM prose.
Only Python's standard library is required. No network or paid model API is used.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parent
SPECIAL = ['<pad>', '<unk>', '<bos>', '<eos>', '<user>', '<assistant>', '<turn>']
PUNCTUATION = ['.', ',', '?', '!', ':']
WORD_RE = re.compile(r'[a-z]+')
TOKEN_RE = re.compile(r'[a-z]+|[.,?!:]')
SEED = 20261021

COLORS = 'red blue green yellow black white brown orange pink purple gray'.split()
OBJECTS = 'book ball cup bag box key phone coat hat pen pencil shirt toy bottle plate spoon fork letter card coin clock watch picture towel blanket pillow brush basket bowl mirror camera'.split()
SPOTS = ['on the table', 'under the table', 'on the desk', 'under the bed',
         'by the door', 'by the window', 'in the box', 'in the bag',
         'in the basket', 'behind the chair', 'next to the bed',
         'on the shelf', 'in the kitchen', 'in the room', 'on the chair',
         'near the door', 'inside the house', 'outside the house']
PEOPLE = [('a boy', 'the boy', 'he'), ('a girl', 'the girl', 'she'),
          ('a child', 'the child', 'they'), ('a teacher', 'the teacher', 'they'),
          ('a friend', 'the friend', 'they'), ('a student', 'the student', 'they'),
          ('a worker', 'the worker', 'they'), ('a man', 'the man', 'he'),
          ('a woman', 'the woman', 'she')]
SUBJECTS = [('i', 'am', False), ('you', 'are', False), ('we', 'are', False),
            ('they', 'are', False), ('he', 'is', True), ('she', 'is', True),
            ('my friend', 'is', True), ('my sister', 'is', True),
            ('my brother', 'is', True)]
NUMBERS = 'zero one two three four five six seven eight nine ten'.split()
COUNT_OBJECTS = [('book', 'books'), ('ball', 'balls'), ('cup', 'cups'),
                 ('key', 'keys'), ('pen', 'pens'), ('box', 'boxes'),
                 ('apple', 'apples'), ('egg', 'eggs'), ('coin', 'coins'),
                 ('bag', 'bags'), ('chair', 'chairs'), ('toy', 'toys')]

# base, third person, past, progressive, and valid verb complements.
ACTIONS = [
    ('read', 'reads', 'read', 'reading', ['a book', 'the new book', 'a short story', 'a letter']),
    ('write', 'writes', 'wrote', 'writing', ['a letter', 'a short story', 'a note', 'a question']),
    ('make', 'makes', 'made', 'making', ['some tea', 'a plan', 'some soup', 'a small cake']),
    ('wash', 'washes', 'washed', 'washing', ['a cup', 'the plate', 'a shirt', 'the bowl']),
    ('clean', 'cleans', 'cleaned', 'cleaning', ['the room', 'the kitchen', 'the table', 'the desk']),
    ('open', 'opens', 'opened', 'opening', ['the door', 'the window', 'a box', 'the book']),
    ('close', 'closes', 'closed', 'closing', ['the door', 'the window', 'the box', 'the book']),
    ('carry', 'carries', 'carried', 'carrying', ['a bag', 'a box', 'a chair', 'a basket']),
    ('visit', 'visits', 'visited', 'visiting', ['a friend', 'my family', 'the park', 'the school']),
    ('watch', 'watches', 'watched', 'watching', ['a film', 'the birds', 'a game', 'the rain']),
    ('play', 'plays', 'played', 'playing', ['a game', 'some music', 'with a toy', 'in the park']),
    ('cook', 'cooks', 'cooked', 'cooking', ['some rice', 'some soup', 'a meal', 'some fish']),
    ('eat', 'eats', 'ate', 'eating', ['some bread', 'an apple', 'some rice', 'a small cake']),
    ('drink', 'drinks', 'drank', 'drinking', ['some water', 'some milk', 'some tea', 'some juice']),
    ('buy', 'buys', 'bought', 'buying', ['some food', 'a book', 'a shirt', 'some fruit']),
    ('find', 'finds', 'found', 'finding', ['a key', 'a book', 'a new place', 'a quiet room']),
    ('draw', 'draws', 'drew', 'drawing', ['a picture', 'a house', 'a tree', 'a small bird']),
    ('take', 'takes', 'took', 'taking', ['a walk', 'a short break', 'a picture', 'the bus']),
    ('go', 'goes', 'went', 'going', ['to school', 'to the shop', 'to the park', 'home']),
    ('come', 'comes', 'came', 'coming', ['home', 'to school', 'to the park', 'to the shop']),
    ('sit', 'sits', 'sat', 'sitting', ['by the window', 'on the chair', 'by the door', 'at the table']),
    ('stand', 'stands', 'stood', 'standing', ['by the window', 'by the door', 'in the room', 'near the tree']),
    ('run', 'runs', 'ran', 'running', ['in the park', 'to the door', 'outside', 'to school']),
    ('walk', 'walks', 'walked', 'walking', ['to the shop', 'in the park', 'home', 'to school']),
    ('sleep', 'sleeps', 'slept', 'sleeping', ['at home', 'in the bed', 'in a quiet room', 'for a while']),
    ('work', 'works', 'worked', 'working', ['at home', 'in the office', 'in the garden', 'at the desk']),
    ('study', 'studies', 'studied', 'studying', ['at home', 'at school', 'in a quiet room', 'with a friend']),
    ('talk', 'talks', 'talked', 'talking', ['with a friend', 'with my family', 'about a book', 'about a plan']),
    ('listen', 'listens', 'listened', 'listening', ['to music', 'to a friend', 'to the rain', 'to the birds']),
    ('look', 'looks', 'looked', 'looking', ['at the sky', 'at a picture', 'at a book', 'at the tree']),
    ('help', 'helps', 'helped', 'helping', ['a friend', 'my family', 'a child', 'a student']),
    ('wait', 'waits', 'waited', 'waiting', ['for the bus', 'for a friend', 'by the door', 'at home']),
    ('ride', 'rides', 'rode', 'riding', ['a bike', 'the bus', 'the train', 'to school']),
    ('hold', 'holds', 'held', 'holding', ['a book', 'a cup', 'a bag', 'a small box']),
    ('give', 'gives', 'gave', 'giving', ['a book to a friend', 'a pen to a student', 'a toy to a child', 'some food to a friend']),
    ('keep', 'keeps', 'kept', 'keeping', ['a pen in a bag', 'a book on the shelf', 'a key in a box', 'a cup on the table']),
    ('bring', 'brings', 'brought', 'bringing', ['a coat', 'a bag', 'a book', 'some food']),
    ('choose', 'chooses', 'chose', 'choosing', ['a shirt', 'a book', 'a game', 'a new plan']),
    ('try', 'tries', 'tried', 'trying', ['a new game', 'a new food', 'a new way', 'a new plan']),
    ('start', 'starts', 'started', 'starting', ['a new book', 'a new game', 'a new plan', 'a short story']),
    ('finish', 'finishes', 'finished', 'finishing', ['a book', 'a game', 'a meal', 'a short story']),
    ('learn', 'learns', 'learned', 'learning', ['a new word', 'a new game', 'a new song', 'a new way to work']),
    ('call', 'calls', 'called', 'calling', ['a friend', 'my sister', 'my brother', 'my family']),
    ('ask', 'asks', 'asked', 'asking', ['a question', 'for help', 'about a book', 'about a plan']),
    ('answer', 'answers', 'answered', 'answering', ['a question', 'the phone', 'a letter', 'a friend']),
    ('change', 'changes', 'changed', 'changing', ['a plan', 'a shirt', 'a small part', 'the story']),
    ('move', 'moves', 'moved', 'moving', ['a chair', 'a box', 'a table', 'a small bed']),
    ('get', 'gets', 'got', 'getting', ['some rest', 'some food', 'a book', 'some water']),
    ('meet', 'meets', 'met', 'meeting', ['a friend', 'a teacher', 'a new person', 'my family']),
    ('build', 'builds', 'built', 'building', ['a small house', 'a box', 'a table', 'a small bed']),
    ('fix', 'fixes', 'fixed', 'fixing', ['a chair', 'a table', 'a door', 'a small box']),
]

MOODS = {
    'happy': (['my friend is here', 'i had a good day', 'the sun is out'], ['talk about the good part', 'tell me about your day']),
    'sad': (['my friend is away', 'i lost my book', 'it was a hard day'], ['talk about it', 'take a quiet break']),
    'tired': (['i had a long day', 'i did not sleep well', 'i worked all day'], ['take a short break', 'get some rest']),
    'worried': (['i have a test tomorrow', 'my work is not done', 'i do not know where to start'], ['make a small plan', 'talk about the next step']),
    'lonely': (['my friends are away', 'i am at home alone', 'i miss my family'], ['talk for a while', 'think of someone to call']),
    'calm': (['the house is quiet', 'i have time to rest', 'i had a nice walk'], ['talk about your day', 'enjoy the quiet']),
}

FOODS = {
    'tea': ['it is warm', 'i enjoy the taste', 'i like a warm drink'],
    'coffee': ['i enjoy the taste', 'i like a warm drink', 'it is part of my morning'],
    'rice': ['i enjoy the taste', 'it goes well with soup', 'it is easy to cook'],
    'bread': ['i enjoy the taste', 'it is part of my breakfast', 'it goes well with cheese'],
    'soup': ['it is warm', 'it is easy to eat', 'i enjoy the taste'],
    'milk': ['i enjoy the taste', 'it goes well with breakfast', 'i like a cold drink'],
    'juice': ['i enjoy the taste', 'i like a cold drink', 'it is part of my breakfast'],
    'fruit': ['i enjoy the taste', 'it is easy to eat', 'it is part of my lunch'],
    'cheese': ['i enjoy the taste', 'it goes well with bread', 'it is part of my lunch'],
    'fish': ['i enjoy the taste', 'it goes well with rice', 'it is part of my dinner'],
}

COMMON_WORDS = '''
the be to of and a in that have i it for not on with he as you do at this but his
by from they we say her she or an will my one all would there their what so up
out if about who get which go me when make can like time no just him know take
people into year your good some could them see other than then now look only
come its over think also back after use two how our work first well way even
new want because any these give day most us is are am was were been being has
had does did said went came made took got very much more many little few every
each both same own such here where why again around down off while before
through during between under above below next near away together still already
always often sometimes never really maybe almost enough too please thank thanks
yes okay hello hi hey goodbye sorry welcome sure right wrong true false need
help let try ask tell talk speak listen hear read write learn understand mean
remember forget believe hope feel seem become keep leave put bring begin start
stop finish end turn move live stay wait call meet find lose show play run walk
sit stand lie sleep wake eat drink cook clean wash open close hold carry send
receive buy sell pay cost spend save choose change build break fix cut grow
fall rise draw sing dance laugh smile cry enjoy love hate wish care share join
follow lead reach pass pull push return answer explain check count add pick
am are is was were been being having doing going coming getting making taking
giving looking thinking feeling trying working reading writing talking playing
walking running sleeping eating drinking sitting standing waiting watching
helping learning asking telling saying knowing seeing hearing living keeping
brings brought brings asks asked answers answered calls called carries carried
changes changed chooses chose cleans cleaned closes closed cooks cooked draws
drew drinks drank eats ate finishes finished fixes fixed gets goes grows grew
helps helped holds held keeps kept learns learned listens listened looks looked
makes meets met moves moved opens opened plays played reads rides rode riding
runs says sleeps slept speaks spoke starts started studies studied studying
takes talks talked tries tried visits visited visiting waits waited walks walked
washes washed watches watched writes wrote builds built buys bought finds found
breaks broke broken cuts sends sent feels felt knows knew leaves left sees saw
must should might may shall can could will would cannot do does did not
man woman boy girl child children baby person people friend friends family
mother father parent parents sister brother son daughter husband wife cousin
uncle aunt grandmother grandfather neighbor teacher student worker doctor nurse
home house room kitchen bathroom bedroom door window floor wall roof garden
school office shop store market bank hospital station library restaurant hotel
park street road town city country village world place space area side front
back top bottom middle inside outside left right north south east west
table chair bed desk shelf sofa box bag basket cup glass bottle plate bowl
spoon fork knife pan pot book paper page pen pencil letter note card picture
photo camera phone computer screen radio television clock watch light key coin
money price ticket map bus train car bike boat plane trip journey holiday
food water milk tea coffee juice bread rice soup meat fish chicken egg eggs
apple apples fruit banana orange lemon potato tomato onion carrot bean beans
cheese butter sugar salt cake dinner lunch breakfast meal taste drink ice cream
shirt coat hat shoe shoes dress pants sock socks clothes towel blanket pillow
brush soap mirror toy ball game music song film story word words sentence
question idea plan problem reason example part piece thing things kind type
number name color size shape line point circle square level step list order
life health body head face eye eyes ear ears nose mouth tooth teeth hair hand
hands arm arms leg legs foot feet heart mind voice sound air fire earth sky
sun moon star stars cloud rain snow wind weather tree leaf leaves flower plant
grass river lake sea beach mountain hill stone rock sand ground bird birds dog
cat horse cow sheep pig duck animal animals fish pet pets
morning afternoon evening night today tomorrow yesterday week weekend month
year hour minute second moment season spring summer autumn winter monday
tuesday wednesday thursday friday saturday sunday january february march april
may june july august september october november december
good bad big small large little long short high low old young new early late
hot cold warm cool wet dry full empty easy hard simple difficult strong weak
fast slow quiet loud nice kind mean happy sad angry afraid scared worried calm
tired hungry thirsty lonely ready busy free sure safe careful clear dark bright
clean dirty soft rough smooth heavy light rich poor cheap expensive close far
beautiful pretty ugly funny serious important useful possible different special
real personal public local basic main best better worse worst last whole half
red blue green yellow black white brown orange pink purple gray
zero one two three four five six seven eight nine ten eleven twelve thirteen
fourteen fifteen sixteen seventeen eighteen nineteen twenty hundred thousand
this that these those someone anyone everyone nobody something anything nothing
everything somewhere anywhere everywhere myself yourself himself herself itself
ourselves themselves yours mine ours theirs hers whose whom either neither
and or but because so although though until unless whether however instead
perhaps probably certainly exactly simply quickly slowly softly loudly kindly
carefully easily finally usually once twice soon later ago yet else rather
only even quite less least together apart across along behind beside beyond
within without against toward towards upon since for until from to into onto
about myself your his her its our their them us me him more most least
computer model chat browser tool tools memory test testing message messages
internet website save saved learn grammar sentence structure basic words
books balls cups keys pens boxes chairs toys shelves windows doors rooms
families questions stories plans games pictures letters shirts coats hats
found lost alone nice happening comes first last later before after rest
can could must might should would will may need needs needed want wants wanted
hope hopes hoped love loves loved like likes liked enjoy enjoys enjoyed
please tell me can you help thank you i am ready to try again
'''.split()


@dataclass
class Dialogue:
    family: str
    pattern: str
    group: str
    turns: list[str]

    def __post_init__(self):
        self.turns = [re.sub(r'\ba orange\b', 'an orange', t) for t in self.turns]

    @property
    def messages(self):
        return [dict(role='user' if i % 2 == 0 else 'assistant', content=t)
                for i, t in enumerate(self.turns)]


def pick_action(rng):
    action = rng.choice(ACTIONS)
    return action, rng.choice(action[4])


def address_user(text):
    return re.sub(r'\bi\b', 'you', text).replace('my ', 'your ')


def spatial(rng):
    obj = rng.choice(OBJECTS)
    color = rng.choice(COLORS)
    spots = [s for s in SPOTS if s.split()[-1] != obj]
    first, second = rng.sample(spots, 2)
    pattern = rng.randrange(3)
    opening = [f'the {color} {obj} is {first}. where is it?',
               f'i left a {color} {obj} {first}. where is the {obj}?',
               f'please remember this: the {color} {obj} is {first}. where is it?'][pattern]
    return Dialogue('spatial_memory', str(pattern), f'{obj}|{color}|{first}|{second}', [
        opening, f'the {obj} is {first}.',
        f'i have put it {second}. where is it now?', f'it is {second} now.',
        'what color is it?', f'it is {color}.'])


def possessions(rng):
    a, b = rng.sample(OBJECTS, 2)
    ca, cb = rng.sample(COLORS, 2)
    person = rng.choice(['me', 'my friend', 'my sister', 'my brother', 'my teacher'])
    other = 'you' if person == 'me' else person.replace('my ', 'your ')
    statement = f'i have a {ca} {a}. {person} has a {cb} {b}.' if person != 'me' else f'i have a {ca} {a} and a {cb} {b}.'
    pattern = 0
    return Dialogue('possession', str(pattern), f'{a}|{b}|{ca}|{cb}|{person}', [
        statement + f' what color is my {a}?', f'your {a} is {ca}.',
        f'and what color is the {b}?', f'the {b} is {cb}.',
        f'who has the {b}?', f'{other} {"have" if other == "you" else "has"} the {cb} {b}.'])


def plans(rng):
    chosen = []
    while len(chosen) < 3:
        action, obj = pick_action(rng)
        phrase = f'{action[0]} {obj}'
        if phrase not in chosen:
            chosen.append(phrase)
    a, b, c = chosen
    day = rng.choice(['today', 'tomorrow', 'this morning', 'this afternoon', 'this evening'])
    pattern = rng.randrange(3)
    opening = [f'i want to {a}, {b}, and {c} {day}. can you help me make a plan?',
               f'please put these things in order: {a}, {b}, and {c}.',
               f'can you make a simple plan for me {day}? i need to {a}, {b}, and {c}.'][pattern]
    return Dialogue('plans_and_order', str(pattern), f'{a}|{b}|{c}|{day}', [
        opening, f'first, {address_user(a)}. then, {address_user(b)}. after that, {address_user(c)}.',
        'what should i do first?', f'you can {address_user(a)} first.',
        'and what comes last?', f'you can {address_user(c)} last.'])


def description(rng):
    obj = rng.choice(OBJECTS)
    color = rng.choice(COLORS)
    size, opposite = rng.choice([('small', 'big'), ('big', 'small'), ('large', 'little'), ('little', 'large')])
    spot = rng.choice([s for s in SPOTS if s.split()[-1] != obj])
    return Dialogue('description_and_yes_no', '0', f'{obj}|{color}|{size}|{spot}', [
        f'i have a {size} {color} {obj} {spot}. can you say that in another way?',
        f'your {obj} is {size} and {color}. it is {spot}.',
        f'is it {opposite}?', f'no. you said it is {size}.',
        f'is it {color}?', f'yes. it is {color}.'])


def story(rng):
    person, definite, pronoun = rng.choice(PEOPLE)
    obj = rng.choice(OBJECTS)
    color = rng.choice(COLORS)
    spot = rng.choice([s for s in SPOTS if s.split()[-1] != obj])
    mood = rng.choice(['happy', 'calm', 'tired', 'sad'])
    ending = {'happy': 'it was a good day', 'calm': 'the room was quiet',
              'tired': 'it was time to rest', 'sad': 'the day had been hard'}[mood]
    return Dialogue('story_in_chat', '0', f'{person}|{obj}|{color}|{spot}|{mood}', [
        f'please tell me a short story about {person} and a {color} {obj}.',
        f'one day, {person} found a {color} {obj}. {definite} put it {spot}. {pronoun} felt {mood}. {ending}.',
        f'where did {definite} put the {obj}?', f'{pronoun} put it {spot}.',
        f'how did {definite} feel?', f'{pronoun} felt {mood}.'])


def tense(rng):
    action, obj = pick_action(rng)
    subject, be, singular = rng.choice(SUBJECTS)
    present = f'{subject} {action[1] if singular else action[0]} {obj}'
    pattern = rng.randrange(3)
    if pattern == 0:
        turns = [f'please say this about yesterday: {present}.', f'yesterday, {subject} {action[2]} {obj}.',
                 'and about tomorrow?', f'tomorrow, {subject} will {action[0]} {obj}.',
                 'how do i say it is happening now?', f'{subject} {be} {action[3]} {obj} now.']
    elif pattern == 1:
        turns = [f'can you help me make this about now: {present}?', f'{subject} {be} {action[3]} {obj} now.',
                 'now say it about yesterday.', f'yesterday, {subject} {action[2]} {obj}.']
    else:
        turns = [f'how do i say the opposite of this: {present}?',
                 f'{subject} {"does" if singular else "do"} not {action[0]} {obj}.',
                 'and if it was yesterday?', f'yesterday, {subject} did not {action[0]} {obj}.']
    return Dialogue('tense_and_negation', str(pattern), f'{subject}|{action[0]}|{obj}', turns)


def correction(rng):
    obj = rng.choice(OBJECTS)
    color = rng.choice(COLORS)
    subject, be, _ = rng.choice(SUBJECTS)
    wrong_be = 'is' if be != 'is' else 'are'
    adjective = rng.choice(['happy', 'sad', 'calm', 'tired', 'ready', 'busy', 'free', 'hungry'])
    return Dialogue('sentence_repair', '0', f'{obj}|{color}|{subject}|{adjective}', [
        f'please fix this sentence: {subject} {wrong_be} {adjective}.', f'{subject} {be} {adjective}.',
        f'and this one: the {color} {obj} are here.', f'the {color} {obj} is here.'])


def preferences(rng):
    a, b = rng.sample(list(FOODS), 2)
    def compatible_reason(food):
        return rng.choice([r for r in FOODS[food]
                           if not re.search(r'\b(morning|breakfast|lunch|dinner)\b', r)])
    reason_a, reason_b = compatible_reason(a), compatible_reason(b)
    time_a, time_b = rng.sample(['in the morning', 'in the afternoon', 'in the evening', 'at lunch', 'at dinner'], 2)
    return Dialogue('preferences', '0', f'{a}|{b}|{reason_a}|{reason_b}|{time_a}|{time_b}', [
        f'i like {a} {time_a} because {reason_a}. i like {b} {time_b} because {reason_b}.',
        f'you like {a} {time_a} and {b} {time_b}.',
        f'why do i like {a}?', f'you said you like {a} because {address_user(reason_a)}.',
        f'when do i like {b}?', f'you like {b} {time_b}.'])


def feelings(rng):
    mood = rng.choice(list(MOODS))
    causes, responses = MOODS[mood]
    cause, response = rng.choice(causes), rng.choice(responses)
    action, obj = pick_action(rng)
    when = rng.choice(['today', 'this morning', 'this evening'])
    greeting = rng.choice(['hello', 'hi', 'hey there', 'good morning', 'good evening'])
    return Dialogue('feelings_and_small_talk', '0', f'{mood}|{cause}|{response}|{action[0]}|{obj}|{when}|{greeting}', [
        f'{greeting}. i feel {mood} {when} because {cause}.',
        f'thank you for telling me. would you like to {response}?',
        f'i would like to {action[0]} {obj} instead.', f'that is fine. you can {action[0]} {address_user(obj)} first.'])


def clarification(rng):
    action, obj = pick_action(rng)
    when = rng.choice(['today', 'tomorrow', 'this morning', 'this afternoon', 'this evening'])
    difficulty = rng.choice(['i do not know when to start', 'i have very little time',
                             'i do not know what to do first', 'it feels hard to begin'])
    response = {'i do not know when to start': 'choose a time when you are free. start with one small step.',
                'i have very little time': 'choose one small part to do first. you can do the rest later.',
                'i do not know what to do first': 'start with the easiest part. you do not have to do everything at once.',
                'it feels hard to begin': 'try one small step first. you can take a short break after that.'}[difficulty]
    return Dialogue('clarification_and_help', '0', f'{action[0]}|{obj}|{when}|{difficulty}', [
        'can you help me with it?', 'what would you like help with?',
        f'i want to {action[0]} {obj} {when}, but {difficulty}.', response,
        'can you say that in a shorter way?', 'start small. take one step at a time.'])


def counting(rng):
    singular, plural = rng.choice(COUNT_OBJECTS)
    a, b = rng.randint(1, 5), rng.randint(1, 5)
    total = a + b
    give = rng.randint(1, total)
    left = total - give
    person = rng.choice(['a friend', 'my sister', 'my brother', 'a child', 'a student'])
    return Dialogue('small_counts', '0', f'{singular}|{a}|{b}|{give}|{person}', [
        f'i have {NUMBERS[a]} {singular if a == 1 else plural}. i get {NUMBERS[b]} more. how many do i have now?',
        f'you have {NUMBERS[total]} {plural}.',
        f'i give {NUMBERS[give]} to {person}. how many do i have left?',
        f'you have {NUMBERS[left]} {singular if left == 1 else plural} left.'])


GENERATORS = [spatial, possessions, plans, description, story, tense,
              correction, preferences, feelings, clarification, counting]
WEIGHTS = [25, 13, 17, 11, 12, 7, 3, 4, 3, 2, 3]


def content_tokens(text):
    if not text or text != text.lower():
        raise ValueError(f'Expected nonempty lowercase text: {text!r}')
    tokens = TOKEN_RE.findall(text)
    if ''.join(tokens) != re.sub(r'\s+', '', text):
        raise ValueError(f'Unrecognized character or punctuation: {text!r}')
    return tokens


def initialize_vocabulary(path):
    required = set()
    rng = random.Random(SEED)
    for generator in GENERATORS:
        for _ in range(3000):
            for turn in generator(rng).turns:
                required.update(WORD_RE.findall(turn))
    required.update('please say the word put before after yes no hello hi thanks thank you welcome goodbye okay sorry again'.split())
    if len(required) > 800:
        raise ValueError(f'Grammar inventory exceeds limit: {len(required)} words')
    words = set(required)
    for word in COMMON_WORDS:
        if len(words) == 800:
            break
        if not re.fullmatch('[a-z]+', word):
            raise ValueError(word)
        words.add(word)
    if len(words) != 800:
        raise ValueError(f'Need more curated candidates: {len(words)}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(sorted(words)) + '\n')
    print(f'Frozen vocabulary: {len(required)} grammar words + {800-len(required)} additional conversational words = 800', flush=True)
    return required


def encode_dialogue(dialogue, token_to_id):
    if len(dialogue.turns) < 2 or len(dialogue.turns) % 2:
        raise ValueError('Dialogue must contain complete user/assistant pairs')
    ids = [token_to_id['<bos>']]
    mask = [0]
    words = []
    for i, turn in enumerate(dialogue.turns):
        assistant = i % 2 == 1
        tokens = content_tokens(turn)
        unknown = [t for t in tokens if t not in token_to_id]
        if unknown:
            raise ValueError(f'Out-of-vocabulary words in {dialogue.family}: {unknown}: {turn}')
        words.extend(t for t in tokens if t.isalpha())
        ids.append(token_to_id['<assistant>' if assistant else '<user>'])
        mask.append(0)
        ids.extend(token_to_id[t] for t in tokens)
        mask.extend([int(assistant)] * len(tokens))
        ids.append(token_to_id['<turn>'])
        mask.append(int(assistant))
    ids.append(token_to_id['<eos>'])
    mask.append(1)
    return ids, mask, words


def split_for(dialogue):
    key = f'{dialogue.family}|{dialogue.group}'.encode()
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:4], 'big') % 10000
    return 'validation' if bucket < 100 else 'test' if bucket < 200 else 'train'


class SplitWriter:
    def __init__(self, root, name, shard_tokens, token_to_id):
        self.root, self.name, self.shard_tokens = root, name, shard_tokens
        self.tokens = self.words = self.dialogues = self.shard = self.shard_size = 0
        self.files = []
        self.family_tokens = Counter()
        self.family_examples = Counter()
        self.word_counts = Counter()
        self.natural_word_counts = Counter()
        self.lengths = Counter()
        self.token_to_id = token_to_id
        self.file = None
        self.raw = None
        self.binary = (root / f'{name}.bin').open('wb')
        self.loss = (root / f'{name}.assistant_mask.bin').open('wb')
        self.offsets = (root / f'{name}.offsets.jsonl').open('w')

    def write(self, dialogue, ids, mask, words, digest):
        if self.file is None or self.shard_size >= self.shard_tokens:
            if self.file:
                self.file.close()
                self.raw.close()
            name = f'{self.name}-{self.shard:03d}.jsonl.gz'
            self.files.append(name)
            # No wall-clock timestamp in gzip headers.
            self.raw = (self.root / name).open('wb')
            self.file = gzip.GzipFile(filename='', mode='wb', fileobj=self.raw, mtime=0)
            self.shard += 1
            self.shard_size = 0
        group_id = hashlib.sha256(f'{dialogue.family}|{dialogue.group}'.encode()).hexdigest()
        record = dict(id=digest, group_id=group_id, family=dialogue.family, pattern=dialogue.pattern,
                      messages=dialogue.messages, token_count=len(ids))
        self.file.write((json.dumps(record, separators=(',', ':')) + '\n').encode())
        self.offsets.write(json.dumps(dict(id=digest, offset=self.tokens, length=len(ids))) + '\n')
        packed = array('H', ids)
        if sys.byteorder != 'little':
            packed.byteswap()
        self.binary.write(packed.tobytes())
        self.loss.write(bytes(mask))
        self.tokens += len(ids)
        self.shard_size += len(ids)
        self.words += len(words)
        self.dialogues += 1
        self.family_tokens[dialogue.family] += len(ids)
        self.family_examples[dialogue.family] += 1
        self.word_counts.update(words)
        if dialogue.family != 'vocabulary_copy':
            self.natural_word_counts.update(words)
        self.lengths[len(ids)] += 1

    def close(self):
        if self.file:
            self.file.close()
            self.raw.close()
        self.binary.close()
        self.loss.close()
        self.offsets.close()

    def report(self):
        return dict(tokens=self.tokens, english_word_occurrences=self.words,
                    dialogues=self.dialogues, unique_english_words=len(self.word_counts),
                    min_dialogue_tokens=min(self.lengths) if self.lengths else 0,
                    max_dialogue_tokens=max(self.lengths) if self.lengths else 0,
                    family_tokens=dict(self.family_tokens), family_examples=dict(self.family_examples),
                    word_counts=dict(sorted(self.word_counts.items())),
                    words_only_in_copy_tasks=sorted(set(self.word_counts)-set(self.natural_word_counts)),
                    jsonl_shards=self.files)


def build(args):
    vocab_path = ROOT / 'vocabulary_800.txt'
    if args.init_vocab:
        if vocab_path.exists():
            raise ValueError('Vocabulary already exists. It will not be silently replaced.')
        initialize_vocabulary(vocab_path)
    if not vocab_path.exists():
        raise ValueError('Initialize the vocabulary first with --init-vocab')
    vocab = vocab_path.read_text().splitlines()
    if len(vocab) != 800 or len(set(vocab)) != 800 or any(not re.fullmatch('[a-z]+', w) for w in vocab):
        raise ValueError('Vocabulary must contain exactly 800 unique lowercase English word forms')
    tokens = SPECIAL + PUNCTUATION + vocab
    token_to_id = {token: i for i, token in enumerate(tokens)}
    targets = dict(train=args.train_tokens, validation=args.validation_tokens, test=args.test_tokens)
    if min(targets.values()) < 0 or sum(targets.values()) == 0:
        raise ValueError('At least one positive split target is required')
    root = Path(args.output_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError(f'Output directory is not empty: {root}')
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = dict(type='closed_word_level', lowercase=True, unknown_policy='reject',
                     pattern=TOKEN_RE.pattern, english_words=800, vocabulary_size=len(tokens),
                     special_tokens=SPECIAL, punctuation=PUNCTUATION,
                     tokens=tokens, token_to_id=token_to_id, binary_dtype='uint16_le',
                     english_vocabulary_sha256=hashlib.sha256(vocab_path.read_bytes()).hexdigest())
    (root / 'tokenizer.json').write_text(json.dumps(tokenizer, indent=2) + '\n')
    (root / 'vocabulary_800.txt').write_bytes(vocab_path.read_bytes())
    writers = {s: SplitWriter(root, s, args.shard_tokens, token_to_id) for s in targets}
    rng = random.Random(args.seed)
    seen = set()
    rejections = Counter()
    sample = []
    start = time.monotonic()
    next_report = 1000000

    def add(dialogue, split):
        ids, mask, words = encode_dialogue(dialogue, token_to_id)
        canonical = json.dumps(dialogue.messages, separators=(',', ':'))
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        if digest in seen:
            rejections['exact_duplicate'] += 1
            return False
        seen.add(digest)
        writers[split].write(dialogue, ids, mask, words, digest)
        if len(sample) < 200 and dialogue.family != 'vocabulary_copy':
            sample.append(dict(id=digest, split=split, family=dialogue.family, messages=dialogue.messages))
        return True

    try:
        # Explicit lexical copying is a small disclosed coverage task, not a definition.
        if targets['train'] > 20000:
            for word in vocab:
                add(Dialogue('vocabulary_copy', 'coverage', f'coverage|{word}',
                             [f'please say the word {word}.', f'{word}.']), 'train')
        attempts = 0
        while any(writers[s].tokens < targets[s] for s in targets):
            attempts += 1
            if attempts > max(100000, sum(targets.values()) * 10):
                raise RuntimeError('Unable to meet targets without excessive duplicate attempts')
            if rng.random() < .015:
                a, b = rng.sample(vocab, 2)
                dialogue = Dialogue('vocabulary_copy', 'order', f'{a}|{b}', [
                    f'please say {a} before {b}.', f'{a} {b}.',
                    f'now say {b} before {a}.', f'{b} {a}.'])
            else:
                dialogue = rng.choices(GENERATORS, weights=WEIGHTS, k=1)[0](rng)
            split = split_for(dialogue)
            if writers[split].tokens >= targets[split]:
                rejections['split_target_already_met'] += 1
                continue
            add(dialogue, split)
            total = sum(w.tokens for w in writers.values())
            if total >= next_report:
                print(f'{total:,} tokens; {sum(w.dialogues for w in writers.values()):,} unique dialogues', flush=True)
                next_report += 1000000
    finally:
        for writer in writers.values():
            writer.close()
    with (root / 'sample_200.jsonl').open('w') as out:
        for item in sample:
            out.write(json.dumps(item) + '\n')
    report = dict(dataset='picoboi-800-v1', source='authored_patterns_with_programmatic_variation',
                  independently_llm_generated=False, seed=args.seed, generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  english_vocabulary_sha256=tokenizer['english_vocabulary_sha256'],
                  token_definition='800 lowercase word forms + 7 structural tokens + 5 punctuation tokens',
                  model_vocabulary_size=len(tokens), requested_token_targets=targets,
                  total_tokens=sum(w.tokens for w in writers.values()),
                  total_dialogues=sum(w.dialogues for w in writers.values()),
                  split_method='semantic group SHA256; 98/1/1 buckets; complete dialogues; train-only lexical coverage tasks',
                  exact_duplicate_dialogues_written=0, out_of_vocabulary_words=0,
                  rejected_candidates=dict(rejections), elapsed_seconds=round(time.monotonic()-start, 2),
                  limitations=['Template families are shared between splits.',
                               'Novel semantic groups are held out, not all linguistic structures.',
                               'Words used only in copy tasks are not taught their meanings.',
                               'Vocabulary and grammar checks do not establish conversational quality.'],
                  splits={s: w.report() for s, w in writers.items()})
    (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    hashes = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name not in {'sha256.json', 'audit.json'}:
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / 'sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['total_tokens', 'total_dialogues', 'model_vocabulary_size', 'elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=str(ROOT / 'data' / 'picoboi-800-v1'))
    parser.add_argument('--init-vocab', action='store_true')
    parser.add_argument('--train-tokens', type=int, default=9800000)
    parser.add_argument('--validation-tokens', type=int, default=100000)
    parser.add_argument('--test-tokens', type=int, default=100000)
    parser.add_argument('--shard-tokens', type=int, default=1000000)
    parser.add_argument('--seed', type=int, default=SEED)
    build(parser.parse_args())
