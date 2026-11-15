"""Removal regression: old settings/requests cannot transmit explanation evidence."""
import threading
import httpx
from sunhacks.pipeline import Pipeline
from sunhacks.server import make_server
from test_claim_pipeline_integration import run_answer


def test_legacy_enabled_manager_config_has_no_runtime_or_public_surface():
    p=Pipeline({'manager':{'enabled':True,'provider':'deepseek','model':'deepseek-flash'}})
    try:
        assert 'manager' not in p.config
        assert not hasattr(p,'manager') and not hasattr(p,'request_manager')
        assert not hasattr(p,'images') and not hasattr(p,'_keyframes')
        state=p.state()
        assert 'manager' not in state and 'manager_busy' not in state
        assert not any(k.startswith('manager_') for k in state['deployment'])
    finally:p.close()


def test_finished_answer_cannot_trigger_removed_explainer_even_with_old_flags(monkeypatch):
    p,task,claims=run_answer(monkeypatch)
    p.config['manager']={'enabled':True};p.answer['manager_summary']=True
    monkeypatch.setattr(httpx,'AsyncClient',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Unexpected remote explanation')))
    p._infer_answer()
    assert p.answer['status']=='complete'
    assert 'manager' not in p.latest
    assert p.latest['claim_check']['status']=='contradiction'
    assert not hasattr(p,'manager_result')


def test_removed_http_endpoint_is_unavailable_and_does_not_publish_summary():
    p=Pipeline();server=make_server(p,0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{server.server_port}',timeout=2,trust_env=False) as c:
            response=c.post('/api/manager',json={})
            assert response.status_code==404
            state=c.get('/api/state').json()
            assert 'manager' not in state and state['latest'] is None
    finally:server.shutdown();server.server_close();thread.join(1);p.close()
