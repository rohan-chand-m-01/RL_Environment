"""
inference.py — AI Workplace Simulator entry point.

Conforms exactly to the Meta OpenEnv Hackathon output format:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

import os
import sys
import json
import traceback
from typing import Any, Dict, List, Optional

from openai import OpenAI

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.base_env import AIWorkplaceEnv
from models.action import Action

# ---------------------------------------------------------------------------
# Config — API_BASE_URL and MODEL_NAME must have defaults per spec
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    print("[END] success=false steps=0 rewards=", flush=True)
    raise ValueError("HF_TOKEN environment variable is required")

ENV_NAME = "ai_workplace_simulator"

SYSTEM_PROMPT = """You are an AI workplace assistant agent operating inside a reinforcement learning environment.

You receive observations describing the current task. Respond ONLY with a valid JSON object:
  {"action_type": "<action>", "payload": {<key>: <value>}}

Valid actions per task:
- email_triage:   {"action_type": "classify", "payload": {"label": "urgent"|"normal"|"spam"}}
- code_review:    {"action_type": "detect_bug", "payload": {"description": "<bug description>"}}
                  {"action_type": "suggest_fix", "payload": {"fix": "<fix description>"}}
- data_cleaning:  {"action_type": "remove_null"}
                  {"action_type": "normalize"}
                  {"action_type": "fix_schema"}

Rules:
- Respond ONLY with raw JSON. No markdown fences, no explanation.
- For data_cleaning actions, payload can be omitted or set to {}.
- Look at completed_steps in the observation and pick the next required step that is not yet done."""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class WorkplaceAgent:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        self.history: List[Dict[str, Any]] = []

    def act(self, observation_dict: Dict[str, Any]) -> Optional[Action]:
        user_msg = (
            f"Current observation:\n{json.dumps(observation_dict, indent=2)}\n\n"
            "What action do you take? Respond with JSON only."
        )
        self.history.append({"role": "user", "content": user_msg})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self.history,
            temperature=0.0,
            max_tokens=256,
        )

        raw = response.choices[0].message.content.strip()
        self.history.append({"role": "assistant", "content": raw})

        # Strip markdown fences if the model wraps the JSON anyway
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else parts[0]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data = json.loads(raw)
        return Action(action_type=data["action_type"], payload=data.get("payload"))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run():
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    env = AIWorkplaceEnv()
    agent = WorkplaceAgent(client, MODEL_NAME)

    obs = env.reset()
    rewards: List[float] = []
    step = 0
    done = False

    # [START] — one line per episode
    print(f"[START] task={obs.task_type} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    while not done:
        step += 1
        error_str = "null"
        reward_val = 0.0
        action_str = "null"

        try:
            action = agent.act(obs.model_dump())
            action_str = action.action_type

            obs, reward, done, info = env.step(action)
            reward_val = reward.value
            rewards.append(reward_val)

            if obs.error_feedback:
                # Sanitise — no newlines inside a [STEP] line
                error_str = obs.error_feedback.replace("\n", " ").strip()

        except json.JSONDecodeError as exc:
            error_str = f"json_parse_error: {exc}".replace("\n", " ")
            rewards.append(0.0)
            # Don't terminate — let the agent retry next step

        except Exception as exc:
            error_str = str(exc).replace("\n", " ").strip()
            rewards.append(0.0)
            done = True

        # [STEP] — one per step, immediately after env.step()
        print(
            f"[STEP] step={step} action={action_str} "
            f"reward={reward_val:.2f} done={str(done).lower()} error={error_str}",
            flush=True,
        )

    # Determine success: env is done and cumulative reward is positive
    state = env.state()
    success = state["done"] and state["cumulative_reward"] > 0
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    # [END] — always emitted
    print(f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}", flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        traceback.print_exc()
        print("[END] success=false steps=0 rewards=", flush=True)
        sys.exit(1)
