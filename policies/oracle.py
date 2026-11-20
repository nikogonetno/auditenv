"""Baseline «Oracle только для оценщика» (case §06): submits the truth as findings.

Grader-side by definition. It runs only in-process through auditenv.runner.run_inprocess_episode
and gets truth from the grader side of the env, never through a subprocess or an observation.
It checks the maximum of the reward, not the skill of an agent: on every episode where truth
holds a complete evidence set for each issue, it must get 1.0 at zero inspection steps.
"""


def findings_from_truth(truth, episode_id):
    """findings.json of case §05 with one finding per truth issue and its first evidence set."""
    findings = []
    for issue in truth.get("issues") or []:
        evidence_sets = issue.get("evidence_sets") or [[]]
        findings.append({
            "type": issue["type"],
            "affected_ids": list(issue["affected_ids"]),
            "claimed_amount_minor": issue["expected_amount_minor"],
            "currency": issue["currency"],
            "evidence_ids": list(evidence_sets[0]),
            "explanation": "oracle",
        })
    return {"episode_id": episode_id, "findings": findings}


def make_oracle(truth):
    """policy_fn(message) -> action that submits the truth at its first message.

    The episode_id comes from the manifest of the reset observation, as any policy would read it.
    """
    state = {"episode_id": None}

    def policy_fn(message):
        observation = message.get("observation") or {}
        manifest = observation.get("manifest")
        if isinstance(manifest, dict) and state["episode_id"] is None:
            state["episode_id"] = manifest.get("episode_id")
        return {"action": "submit", "findings": findings_from_truth(truth, state["episode_id"])}

    return policy_fn


def make_env_oracle(env):
    """An oracle that reads truth from the env's grader side after reset (env._truth).

    Grader-side only: the env object never crosses into a policy process.
    """
    inner = []

    def policy_fn(message):
        if not inner:
            inner.append(make_oracle(env._truth))
        return inner[0](message)

    return policy_fn
