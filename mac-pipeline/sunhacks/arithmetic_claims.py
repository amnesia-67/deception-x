"""Conservative, deterministic checks of explicit arithmetic assertions.

This module does not evaluate Python or verify a person's honesty. Unsupported,
reported, hypothetical and negated claims are deliberately left unchecked.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import re

MAX_TEXT = 8000
MAX_CLAUSE = 180
MAX_OPERATIONS = 3
MAX_OPERAND = Fraction(10**9)
MAX_RESULT = Fraction(10**12)
_UNITS = dict(zip(
    ('zero','one','two','three','four','five','six','seven','eight','nine','ten',
     'eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen'),
    range(20)))
_TENS = dict(zip(('twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety'), range(20,100,10)))
_NUMBER_WORDS = frozenset((*_UNITS, *_TENS))
_CONTEXT = re.compile(r"\b(?:if|suppose|supposing|assuming|imagine|pretend|hypothetical|could|would|might|not|never|no|said|says|asked|quote|quoted|claimed|claims|told|report|reports|reported|wrote|writes|whether|question|believe|think|false|incorrect|wrong|untrue)\b|n['’]t\b", re.I)
_EQUAL = re.compile(r"\s+(?:is\s+equal\s+to|equals?|equal\s+to|is)\s+|=")


class Unsupported(ValueError):
    pass


class Uncheckable(ValueError):
    pass


@dataclass
class Budget:
    operations: int = 0

    def use(self):
        self.operations += 1
        if self.operations > MAX_OPERATIONS:
            raise Uncheckable('This arithmetic expression exceeds the supported operation bound.')


def _normalise(value):
    value = value.lower().strip()
    value = value.replace('−','-').replace('×','*').replace('÷','/')
    value = re.sub(r'\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)-(one|two|three|four|five|six|seven|eight|nine)\b',r'\1 \2',value)
    for words, symbol in ((r'\bmultiplied\s+by\b','*'),(r'\bdivided\s+by\b','/'),
                          (r'\bplus\b','+'),(r'\bminus\b','-'),(r'\btimes\b','*'),
                          (r'\bnegative\s+','-'),(r'\bpositive\s+','+')):
        value = re.sub(words,symbol,value)
    return value.strip()


def _number(token):
    token = token.strip()
    if re.fullmatch(r'\d{1,10}(?:\.\d{1,6})?',token):
        value = Fraction(token)
    else:
        words = token.split()
        if len(words)==1 and words[0] in _UNITS:
            value = Fraction(_UNITS[words[0]])
        elif len(words)==1 and words[0] in _TENS:
            value = Fraction(_TENS[words[0]])
        elif len(words)==2 and words[0] in _TENS and words[1] in _UNITS and 1<=_UNITS[words[1]]<=9:
            value = Fraction(_TENS[words[0]]+_UNITS[words[1]])
        else:
            raise Unsupported('Unsupported operand')
    if abs(value)>MAX_OPERAND:
        raise Uncheckable('The claimed arithmetic uses numbers outside the supported bound.')
    return value


def _bounded(value):
    if abs(value)>MAX_RESULT or value.denominator>10**12:
        raise Uncheckable('The arithmetic result exceeds the supported numerical bound.')
    return value


class _Parser:
    def __init__(self, expression, budget):
        self.tokens = [x.strip() for x in re.findall(r'[+\-*/]|[^+\-*/]+',expression) if x.strip()]
        self.position = 0
        self.budget = budget

    def next(self):
        return self.tokens[self.position] if self.position<len(self.tokens) else None

    def factor(self):
        sign = 1
        if self.next() in ('+','-'):
            if self.next()=='-':sign=-1
            self.position+=1
        token = self.next()
        if token is None or token in ('+','-','*','/'):
            raise Unsupported('Missing operand')
        self.position+=1
        return sign*_number(token)

    def term(self):
        value = self.factor()
        while self.next() in ('*','/'):
            operator = self.next();self.position+=1;self.budget.use()
            right = self.factor()
            if operator=='/' and right==0:
                raise Uncheckable('Division by zero has no defined arithmetic result.')
            value = _bounded(value*right if operator=='*' else value/right)
        return value

    def expression(self):
        value = self.term()
        while self.next() in ('+','-'):
            operator = self.next();self.position+=1;self.budget.use()
            right = self.term()
            value = _bounded(value+right if operator=='+' else value-right)
        if self.position!=len(self.tokens):
            raise Unsupported('Unsupported arithmetic syntax')
        return value


def _display(value):
    return str(value.numerator) if value.denominator==1 else str(value)


def _candidate(clause):
    if not clause or len(clause)>MAX_CLAUSE:
        return None
    pieces = _EQUAL.split(clause)
    if len(pieces)!=2 or not all(part.strip() for part in pieces):
        return None
    left,right = (_normalise(part) for part in pieces)
    # No unknown names, executable syntax, brackets or free-form prose enters the parser.
    for expression in (left,right):
        if re.search(r'[^a-z0-9. +\-*/]',expression):
            return None
        for word in re.findall(r'[a-z]+',expression):
            if word not in _NUMBER_WORDS:
                return None
    budget = Budget()
    try:
        first = _Parser(left,budget).expression()
        second = _Parser(right,budget).expression()
    except Unsupported:
        return None
    except Uncheckable as exc:
        return {'status':'unverifiable','kind':'math','reason':str(exc),
                'evidence':[], 'expression':left,'claimed_expression':right}
    contradictory = first!=second
    return {'status':'contradiction' if contradictory else 'no_contradiction','kind':'math',
            'reason':('The transcribed arithmetic claim contradicts exact arithmetic.' if contradictory else
                      'No arithmetic contradiction was found in this parsed equation; this does not verify other claims.'),
            'evidence':[f'Parsed equation: {left} = {right}.',
                        f'Calculated values: {_display(first)} and {_display(second)}.'],
            'expression':left,'claimed_expression':right,
            'calculated_left':_display(first),'calculated_right':_display(second)}


def check_arithmetic(text):
    """Check at most three supported direct assertions; never return a truth verdict.

    ``None`` means no safe supported arithmetic assertion was found. The caller
    must not reinterpret that as correctness. Scope exclusions intentionally err
    toward leaving contextual/ambiguous language unchecked.
    """
    if not isinstance(text,str) or not text.strip() or len(text)>MAX_TEXT:
        return None
    if _CONTEXT.search(text) or any(mark in text for mark in ('"','“','”','`','?')) or re.search(r"'[^']+'",text):
        return None
    clauses = re.split(r'(?<!\d)\.|\.(?!\d)|[!;]',text.lower())
    results = []
    for clause in clauses[:6]:
        result = _candidate(clause.strip(' ,\n\t'))
        if result is not None:
            results.append(result)
            if result['status']=='contradiction':return result
            if len(results)>=3:break
    if not results:return None
    return next((value for value in results if value['status']=='unverifiable'),results[0])
