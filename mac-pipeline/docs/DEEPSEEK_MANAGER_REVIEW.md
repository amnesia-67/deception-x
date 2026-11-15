# DeepSeek Flash manager adapter

Research date:2026-11-11. Only official DeepSeek documentation was consulted. No real API key was read and no provider request was made in this subtask.

## Verified current contract

- Model ID:`deepseek-flash`; the currently served version is DeepSeek-V4.1-Flash-1017. Use the current ID rather than the retired `deepseek-v4-flash` or `deepseek-v4-flash-vision-exp` aliases. [Models](https://api-docs.deepseek.com/quick_start/pricing/)
- Endpoint:`POST https://api.deepseek.com/chat/completions`, OpenAI-compatible messages and `Authorization: Bearer` authentication. [First call](https://api-docs.deepseek.com/quick_start), [Chat API](https://api-docs.deepseek.com/api/create-chat-completion/)
- Flash supports images. The current Pro model does not. Images use `user` message content parts `{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,...","detail":"low"}}`. The `low` setting is documented for faster/cheaper image processing. This integration keeps the existing explicit image opt-in and at most four keyframes; it does not upload whole video or raw audio. [Vision](https://api-docs.deepseek.com/guides/vision/)
- Thinking is enabled by default. Send `thinking:{"type":"disabled"}` for the requested non-thinking path. This is a latency-oriented setting, not evidence of a measured speed advantage or unchanged quality. [Thinking mode](https://api-docs.deepseek.com/guides/thinking_mode/)
- `response_format:{"type":"json_object"}` requires a prompt explicitly requesting JSON and describing its shape. Blank/space-only or truncated responses remain possible and must be rejected. The adapter validates final content as summary/observations/limitations and never displays provider `reasoning_content`. [JSON output](https://api-docs.deepseek.com/guides/json_mode/)

## Staged files

- `/tmp/sunhacks-deepseek-stage/manager.py`: replacement provider adapter.
- `/tmp/sunhacks-deepseek-stage/test_deepseek_manager.py`:18 local-stub checks, no key or external traffic.

The provider defaults to `gemini` only to preserve existing configurations/tests. Set `provider:"deepseek"` explicitly to switch. DeepSeek requests always use the hard-coded DeepSeek endpoint; no Gemini fallback or arbitrary base URL is used. A configured fallback remains within that same provider. HTTP401/402/403 stop fallback attempts; only sanitized status codes reach the dashboard.

The helper accepts only the two currently documented DeepSeek IDs (`deepseek-flash`, `deepseek-v4-pro`), rejects unsupported providers/legacy IDs, refuses images on Pro, and uses4-second phase timeouts inside a configurable whole-response deadline. Existing opt-in Gemini behavior remains available but is not selected by a DeepSeek configuration. Application image limits are intentionally smaller than provider limits.

## Root integration

Copy staged manager.py to the repository's sunhacks/manager.py. Copy tests to tests/test_deepseek_manager.py, replacing their staged dynamic import block with `import sunhacks.manager as manager`; the test module currently imports its sibling manager.py to validate the staged implementation without modifying the live application.

Configure the runtime:

```json
{
  "manager": {
    "enabled": true,
    "provider": "deepseek",
    "model": "deepseek-flash",
    "fallback_model": "",
    "api_key_env": "DEEPSEEK_API_KEY",
    "total_timeout_seconds": 15,
    "send_keyframes": false
  }
}
```

The adapter reads only `os.environ[DEEPSEEK_API_KEY]`; it does not parse or source a .env file. Root should load the user-specified Desktop/env file with a non-executing dotenv parser, verify the exact key variable name without printing its value, and populate the server process environment. Do not shell-source an unknown .env and do not copy credentials into JSON, reports or the browser.

Start with a minimal synthetic text-only connectivity check if root is performing live authorization verification. User video/images require the selected `send_keyframes` mode and an explicitly requested manager summary. A successful connectivity response establishes only that provider access works. Actual live manager latency, image behavior and output usefulness remain to be measured. Model summaries do not establish reliable deception detection or verify a teammate checkpoint's training history.
