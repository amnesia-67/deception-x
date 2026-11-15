"""Deployment quarantine shared by every trained-head entry point."""


def artifact_unavailable_reason(config, branch):
    deployment = config.get("deployment", {})
    if deployment.get("artifacts_status") in ("awaiting_retrained_weights", "awaiting_deployment_export", "awaiting_remote_verification", "quarantined"):
        return deployment.get("artifact_reason") or "Previous teammate artifacts are quarantined; reviewed replacement weights and input contracts are required."
    return config.get("models", {}).get(branch, {}).get("disabled_reason")
