"""OpenRouter contract tests; generated inputs, fake HTTP, no real API calls."""
import asyncio,base64,io,json,threading,time
import httpx,pytest
from PIL import Image
from sunhacks.claim_check import ClaimChecker,ENDPOINT,MODEL,RESOLVED_MODEL,MAX_RESPONSE_BYTES

def frames():
    out=io.BytesIO();Image.new('RGB',(64,64),(0,180,0)).save(out,format='JPEG')
    value=base64.b64encode(out.getvalue()).decode();return(value,value)

def envelope(choice='contradiction',**changes):
    p={x:(.98 if x==choice else .01) for x in ('contradiction','no_contradiction','insufficient_evidence')}
    return {'model':RESOLVED_MODEL,'provider':'DeepInfra','answers':{'contradiction':{'type':'choice','choice':choice,'probabilities':p}},**changes}

def checker(monkeypatch,data=None,handler=None,**cfg):
    monkeypatch.setenv('OPENROUTER_API_KEY','test-private-key');calls=[]
    async def wrapped(request):
        calls.append(request)
        return await handler(request) if handler else httpx.Response(200,json=data or envelope())
    return ClaimChecker(cfg,_transport=httpx.MockTransport(wrapped)),calls

@pytest.mark.parametrize('text',['1 + 1 = 3.','My shirt is purple.','There is an elephant beside me.','My name is Sam.'])
def test_one_request_gets_entire_recognized_transcript_and_two_photos(monkeypatch,text):
    c,calls=checker(monkeypatch);r=c.check(text,frames(),visual_authorized=True,source_window_id=7)
    assert r['status']=='contradiction' and r['kind']=='multimodal'
    assert r['source_window_id']==7 and r['sent_transcript'] is True and r['sent_keyframes']==2
    assert r['provider']=='openrouter' and r['upstream_provider']=='DeepInfra' and r['model']==RESOLVED_MODEL
    assert r['model_version_verified'] is False and r['never_confirms_truth'] is True
    assert len(calls)==1;request=calls[0];body=json.loads(request.content)
    assert str(request.url)==ENDPOINT and request.headers['authorization']=='Bearer test-private-key'
    assert body['model']==MODEL and body['provider']=={'only':['DeepInfra'],'allow_fallbacks':False}
    assert json.loads(body['state'][0])['recognized_transcript']==text
    assert len(body['state'])==3 and all(p['type']=='image_url' for p in body['state'][1:])
    assert set(body['questions'])=={'contradiction'} and body['questions']['contradiction']['type']=='choice'
    assert 'messages' not in body and 'thinking' not in body
    assert 'test-private-key' not in json.dumps(r) and text not in json.dumps(r)

@pytest.mark.parametrize('choice,status',[('contradiction','contradiction'),('no_contradiction','no_contradiction'),('insufficient_evidence','unverifiable')])
def test_typed_outcomes_do_not_verify_truth_or_lying_intent(monkeypatch,choice,status):
    c,_=checker(monkeypatch,envelope(choice));r=c.check('Generated statement.',frames(),visual_authorized=True)
    assert r['status']==status and r['never_confirms_truth'] and r['decision_choice']==choice
    assert 'confidence' not in r and 'probability_of_lying' not in r

@pytest.mark.parametrize('authorized',[False,None,'true',1])
def test_no_network_without_explicit_transcript_and_photo_authorization(monkeypatch,authorized):
    c,calls=checker(monkeypatch);r=c.check('1+1=3',frames(),visual_authorized=authorized)
    assert not calls and not r['call_made']

@pytest.mark.parametrize('images',[(),('bad','bad'),frames()[:1],('x'*3000000,'x')])
def test_missing_invalid_or_oversized_photos_never_fall_back_to_text(monkeypatch,images):
    c,calls=checker(monkeypatch);r=c.check('1+1=3',images,visual_authorized=True)
    assert r['status']=='unverifiable' and not calls

@pytest.mark.parametrize('text',[None,'',' '*5,'x'*4001])
def test_invalid_transcript_does_not_dispatch(monkeypatch,text):
    c,calls=checker(monkeypatch);assert c.check(text,frames(),visual_authorized=True)['status']=='unverifiable' and not calls

@pytest.mark.parametrize('source',[True,'3',1.1,-1,2**53])
def test_invalid_source_does_not_dispatch(monkeypatch,source):
    c,calls=checker(monkeypatch);assert c.check('1+1=3',frames(),visual_authorized=True,source_window_id=source)['status']=='unverifiable' and not calls

@pytest.mark.parametrize('changes',[{'model':'other/model'},{'provider':'OtherProvider'},{'answers':{}},{'answers':{'contradiction':{'type':'noul','noul':1}}}])
def test_wrong_model_provider_or_schema_is_rejected(monkeypatch,changes):
    c,_=checker(monkeypatch,envelope(**changes));r=c.check('Generated',frames(),visual_authorized=True)
    assert r['status']=='unverifiable'

@pytest.mark.parametrize('change',[
 {'choice':'unknown'}, {'choice':'no_contradiction'}, {'probabilities':{'contradiction':True,'no_contradiction':0,'insufficient_evidence':0}},
 {'probabilities':{'contradiction':.9,'no_contradiction':.9,'insufficient_evidence':.9}},
 {'confidence':float('nan')}, {'refusal':'refused'},
])
def test_malformed_choice_probabilities_and_refusal_do_not_override(monkeypatch,change):
    data=envelope();data['answers']['contradiction'].update(change);c,_=checker(monkeypatch,data)
    assert c.check('Generated',frames(),visual_authorized=True)['status']=='unverifiable'

def test_probability_tie_becomes_insufficient_evidence(monkeypatch):
    data=envelope();data['answers']['contradiction']['probabilities']={'contradiction':.5,'no_contradiction':.5,'insufficient_evidence':0}
    c,_=checker(monkeypatch,data);assert c.check('Generated',frames(),visual_authorized=True)['status']=='unverifiable'

@pytest.mark.parametrize('code',[401,402,403,429,500])
def test_provider_error_does_not_retry_or_expose_bodies(monkeypatch,code):
    async def handler(request):return httpx.Response(code,text='SECRET BODY test-private-key')
    c,calls=checker(monkeypatch,handler=handler);r=c.check('Generated',frames(),visual_authorized=True)
    assert r['status']=='unverifiable' and len(calls)==1 and 'SECRET' not in json.dumps(r)

def test_missing_key_fails_without_a_request(monkeypatch):
    c,calls=checker(monkeypatch);monkeypatch.delenv('OPENROUTER_API_KEY');r=c.check('Generated',frames(),visual_authorized=True)
    assert r['status']=='unverifiable' and not calls

def test_total_deadline_cancels_single_request(monkeypatch):
    async def handler(request):await asyncio.sleep(.4);return httpx.Response(200,json=envelope())
    c,calls=checker(monkeypatch,handler=handler,deadline_seconds=.1);started=time.monotonic()
    r=c.check('Generated',frames(),visual_authorized=True)
    assert r['status']=='unverifiable' and time.monotonic()-started<.35 and len(calls)==1

def test_cancellation_drops_decision_evidence(monkeypatch):
    stop=threading.Event()
    async def handler(request):stop.set();await asyncio.sleep(.1);return httpx.Response(200,json=envelope())
    c,_=checker(monkeypatch,handler=handler);r=c.check('Generated',frames(),visual_authorized=True,stop_event=stop)
    assert r['status']=='cancelled' and r['evidence']==[] and 'decision_probabilities' not in r

def test_response_byte_limit(monkeypatch):
    async def handler(request):return httpx.Response(200,content=b'x'*(MAX_RESPONSE_BYTES+1))
    c,_=checker(monkeypatch,handler=handler);assert c.check('Generated',frames(),visual_authorized=True)['status']=='unverifiable'

@pytest.mark.parametrize('cfg',[{'provider':'deepseek'},{'model':'nace-ai/drex-max-finance'},{'deadline_seconds':True},{'deadline_seconds':0}])
def test_unsupported_provider_model_or_deadline_rejected(cfg):
    with pytest.raises(ValueError):ClaimChecker(cfg)
