import os
import pytest
from sunhacks.credentials import load_env_file

def test_explicit_file_load_does_not_interpolate_or_expose_values(tmp_path,monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY",raising=False)
    file=tmp_path/"env.env";file.write_text('DEEPSEEK_API_KEY="literal-${NOT_EXPANDED}-key"\nOTHER=ignore\n')
    result=load_env_file(file)
    assert os.getenv("DEEPSEEK_API_KEY")=="literal-${NOT_EXPANDED}-key"
    assert "literal" not in str(result)
    assert result["variable_names"]==["DEEPSEEK_API_KEY"]
    monkeypatch.delenv("DEEPSEEK_API_KEY")

def test_shell_command_not_executed(tmp_path):
    target=tmp_path/"must-not-exist";file=tmp_path/"env.env"
    file.write_text('DEEPSEEK_API_KEY=$(touch '+str(target)+')\n')
    with pytest.raises(ValueError):load_env_file(file)
    assert not target.exists()

def test_missing_key_or_multiline_assignment_rejected(tmp_path):
    file=tmp_path/"env.env";file.write_text('OTHER=ignore\n')
    with pytest.raises(ValueError):load_env_file(file)


def test_openrouter_key_is_loaded_as_literal_and_not_exposed(tmp_path,monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY",raising=False)
    file=tmp_path/"env.env";file.write_text('OPENROUTER_API_KEY="private-router-key"\n')
    result=load_env_file(file)
    assert os.getenv("OPENROUTER_API_KEY")=="private-router-key"
    assert result["variable_names"]==["OPENROUTER_API_KEY"]
    assert "private-router-key" not in str(result)
    monkeypatch.delenv("OPENROUTER_API_KEY")
