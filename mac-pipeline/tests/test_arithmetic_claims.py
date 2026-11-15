import pytest
from sunhacks.arithmetic_claims import check_arithmetic


@pytest.mark.parametrize('claim,left,right',[
    ('1+1 = 3','2','3'),('one plus one equals three.','2','3'),
    ('One plus one is three.','2','3'),('ten minus two equals nine','8','9'),
    ('two times three is seven','6','7'),('six divided by two equals four','3','4'),
    ('two multiplied by three is five','6','5'),('1 + 2 * 3 = 8','7','8'),
    ('negative one plus two is three','1','3'),('0.1 + 0.2 = 0.4','3/10','2/5'),
    ('twenty-one plus nine is thirty one','30','31'),('1=2','1','2'),
    ('1+1=2. 2+2=5.','4','5'),('one plus one is equal to three','2','3')])
def test_false_arithmetic(claim,left,right):
    value=check_arithmetic(claim)
    assert value['status']=='contradiction' and value['kind']=='math'
    assert value['calculated_left']==left and value['calculated_right']==right
    assert 'truth' not in value


@pytest.mark.parametrize('claim',[
    '1+1=2','one plus one equals two','0.1+0.2=.3','1/3=2/6',
    '-1+2=1','one minus negative two is three','two plus two times two is six',
    'twenty one plus nine is thirty','1+1=2+0'])
def test_correct_arithmetic_never_claims_truth(claim):
    value=check_arithmetic(claim)
    if claim=='0.1+0.2=.3':
        assert value is None
    else:
        assert value['status']=='no_contradiction'
        assert 'does not verify' in value['reason']
        assert 'verified' not in value and 'truth' not in value


@pytest.mark.parametrize('claim',[
    'If 1+1=3, then everything changes.','I do not think one plus one equals three.',
    "I don't think 1+1=3.",'He said one plus one equals three.',
    'She told me 1+1=3.','Imagine one plus one equals three.',
    '"1+1=3" is an example.','Does one plus one equal three?',
    'My shirt is purple.','I am black.','I think one plus one is three.',
    'one plus one is not three.','Suppose 1+1=3.',
    "__import__('os').system('touch /tmp/unsafe')=0",'1+1=__import__("os")',
    '(1+1)=3','one two plus three is four','1**2=3','2^4=16','1+1==3',
    'the answer is 3','1e100+1=3','three hundred plus one is two',
    'one plus a number is three','1+1=3 ' + 'x'*8000])
def test_unsupported_context_or_code_is_unchecked(claim):
    assert check_arithmetic(claim) is None


@pytest.mark.parametrize('claim',['1/0=3','1+2+3+4+5=15','9999999999+1=2','1000000000/0.000001=1'])
def test_undefined_or_bounded_out_is_unverifiable(claim):
    value=check_arithmetic(claim)
    assert value['status']=='unverifiable'
    assert value['evidence']==[]


def test_input_contract():
    for value in (None,1,{},'', ' '):assert check_arithmetic(value) is None


@pytest.mark.parametrize('claim',[
    'one and one is two',
    'two and two equals four',
    '1+1=3, but that is false',
    '1+1=3. That is incorrect.',
])
def test_conjunction_or_retraction_is_not_misread_as_false_assertion(claim):
    assert check_arithmetic(claim) is None
