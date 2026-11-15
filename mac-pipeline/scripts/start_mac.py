"""Start this Mac checkout once and open its local app. No capture on launch."""
from pathlib import Path
import json,subprocess,sys,time,urllib.request,webbrowser
ROOT=Path(__file__).resolve().parents[1]
URL="http://127.0.0.1:8765/"
def running():
    try:
        with urllib.request.urlopen(URL+"api/state",timeout=1) as response:return json.load(response)
    except OSError:return None
state=running()
if state is not None:
    if state.get("deployment",{}).get("app_repository")!="amnesia-67/deception-x":
        raise SystemExit("Port 8765 is used by another app. Close it before starting DeceptionX.")
else:
    runtime=ROOT/".runtime";runtime.mkdir(exist_ok=True)
    with (runtime/"app.log").open("ab",buffering=0) as log:
        child=subprocess.Popen([sys.executable,"-m","sunhacks","serve","--config",str(ROOT/"config.split-answer.local.json")],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (runtime/"app.pid").write_text(str(child.pid))
    for _ in range(80):
        state=running()
        if state:break
        if child.poll() is not None:raise SystemExit("App exited. See "+str(runtime/"app.log"))
        time.sleep(.2)
    else:raise SystemExit("App is starting; see "+str(runtime/"app.log"))
if "--no-browser" not in sys.argv:webbrowser.open(URL)
print("DeceptionX is running at "+URL)
