"""Explicit local secret-file loading without shell evaluation or interpolation."""
from pathlib import Path
import os
import re
import shlex

def load_env_file(path, allowed_keys=("DEEPSEEK_API_KEY", "OPENROUTER_API_KEY")):
    file = Path(path).expanduser()
    if not file.is_file(): raise ValueError("Configured env file is missing")
    if file.stat().st_size>65536: raise ValueError("Env file exceeds size limit")
    loaded = []
    for raw in file.read_text().splitlines():
        line=raw.strip()
        if not line or line.startswith("#"): continue
        if line.startswith("export "): line=line[7:].strip()
        name,sep,value=line.partition("=");name=name.strip()
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*",name):
            raise ValueError("Invalid env-file assignment; secret values are not shown")
        if name not in allowed_keys: continue
        try: parts=shlex.split(value.strip(),comments=True,posix=True)
        except ValueError: raise ValueError("Invalid quoted env-file value; secret values are not shown") from None
        if len(parts)!=1 or not parts[0] or any(x in parts[0] for x in ("\r","\n")):
            raise ValueError("API key assignment must contain one nonempty value")
        os.environ[name]=parts[0]
        loaded.append(name)
    if not loaded: raise ValueError("No supported nonempty API key found in env file")
    return {"status":"loaded","variable_names":loaded,"values_exposed":False}
