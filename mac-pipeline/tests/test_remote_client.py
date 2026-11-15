import json

import pytest

from sunhacks import remote_client as remote

LEGACY = {'status':'ok','message':'pong','models':['v2_tabular_xgboost', 'branch1_jepa_neural'], 'features_expected':250,'jepa_loaded':True}

class FakeSocket:
    def __init__(self, chunks=(), connect_error=None):
        self.chunks = iter(chunks)
        self.connect_error = connect_error
        self.sent = []
        self.timeouts = []
        self.closed = False
        self.destination = None
    def __enter__(self): return self
    def __exit__(self, *args): self.closed = True
    def settimeout(self, value): self.timeouts.append(value)
    def connect(self, address):
        self.destination = address
        if self.connect_error: raise self.connect_error
    def sendall(self, data): self.sent.append(data)
    def recv(self, maximum):
        chunk = next(self.chunks, b'')
        if isinstance(chunk, BaseException): raise chunk
        assert len(chunk) <= maximum
        return chunk


def run_probe(monkeypatch, chunks, *, max_response=65536, clock=None, connect_error=None):
    stream = FakeSocket(chunks, connect_error)
    monkeypatch.setattr(remote.socket, 'socket', lambda *args: stream)
    if clock is not None:
        times = iter(clock)
        monkeypatch.setattr(remote.time, 'monotonic', lambda: next(times))
    output = remote.probe_socket_health('192.0.2.1', timeout_seconds=3, max_response_bytes=max_response)
    return output, stream


def test_legacy_health_only_never_calls_random_jepa(monkeypatch):
    result, stream = run_probe(monkeypatch, [json.dumps(LEGACY).encode()+b'\n'])
    assert stream.sent == [b'{"action":"ping"}\n']
    assert stream.destination == ('192.0.2.1', 5002)
    assert stream.closed
    assert result['status'] == 'needs_encoder_protocol'
    assert result['service_kind'] == 'legacy_head_and_tabular'
    assert result['fusion_head_loaded'] is True
    assert result['encoder_verified'] is False
    assert result['participant_data_sent'] is False
    assert result['reachable'] is True


def test_chunked_line_and_trailing_data(monkeypatch):
    encoded = json.dumps(LEGACY).encode()
    result, stream = run_probe(monkeypatch, [encoded[:30], encoded[30:]+b'\n{}'])
    assert result['health'] == LEGACY
    assert result['status'] == 'needs_encoder_protocol'
    assert stream.closed


def test_encoder_claim_cannot_become_verified(monkeypatch):
    health = dict(LEGACY, encoder_loaded=True, encoder_model='facebook/vjepa2-vitl-fpc64-256')
    result, _ = run_probe(monkeypatch, [json.dumps(health).encode()+b'\n'])
    assert result['status'] == 'encoder_protocol_unverified'
    assert result['encoder_verified'] is False


def test_plain_pong_does_not_imply_known_service():
    result = remote.classify_health({'message':'pong'})
    assert result['service_kind'] == 'unknown_pong_service'
    assert result['status'] == 'needs_encoder_protocol'


def test_server_error():
    result = remote.classify_health({'status':'error','error':'not ready'})
    assert result['status'] == 'server_error'


def test_refused_connection(monkeypatch):
    result, stream = run_probe(monkeypatch, [], connect_error=ConnectionRefusedError(61, 'private detail'))
    assert result['status'] == 'network_error'
    assert result['reachable'] is False
    assert stream.sent == []
    assert 'private detail' not in result['reason']
    assert stream.closed


def test_total_deadline_is_not_reset_per_chunk(monkeypatch):
    result, stream = run_probe(monkeypatch, [b'{'], clock=[0, 0.1, 0.2, 1.5, 3.1, 3.2])
    assert result['status'] == 'timeout'
    assert result['reachable'] is True
    assert stream.timeouts == [2.9, 2.8, 1.5]
    assert result['elapsed_ms'] == 3200
    assert stream.closed


def test_response_limit(monkeypatch):
    result, stream = run_probe(monkeypatch, [b'x'*129], max_response=128)
    assert result['status'] == 'protocol_error'
    assert 'byte limit' in result['reason']
    assert stream.closed


@pytest.mark.parametrize('payload', [b'', b'not json\n', b'{"x":1,"x":2}\n', b'{"x":NaN}\n', b'{"x":1e999}\n', b'[]\n', b'\xff\n'])
def test_invalid_responses_are_bounded_errors(monkeypatch, payload):
    result, stream = run_probe(monkeypatch, [payload])
    assert result['status'] == 'protocol_error'
    assert result['encoder_verified'] is False
    assert stream.closed


def test_sensitive_health_metadata_redacted(monkeypatch):
    health = dict(LEGACY, api_key='secret-value', auth_required=True, nested={'access_token':'secret-value'})
    result, _ = run_probe(monkeypatch, [json.dumps(health).encode()+b'\n'])
    assert result['health']['api_key'] == '[redacted]'
    assert result['health']['nested']['access_token'] == '[redacted]'
    assert result['health']['auth_required'] is True
    assert 'secret-value' not in json.dumps(result)


@pytest.mark.parametrize('host', ['0.0.0.0','::','255.255.255.255','224.0.0.1','example.org'])
def test_unsafe_or_unresolved_destinations_rejected(host):
    with pytest.raises(ValueError): remote.SocketHealthClient(host)


@pytest.mark.parametrize('kwargs', [{'port':True},{'port':0},{'port':65536},{'timeout_seconds':0},{'timeout_seconds':31},{'timeout_seconds':float('nan')},{'timeout_seconds':True},{'max_response_bytes':127},{'max_response_bytes':True}])
def test_invalid_limits_rejected(kwargs):
    with pytest.raises(ValueError): remote.SocketHealthClient('192.0.2.1', **kwargs)
