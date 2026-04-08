import asyncio
import json
import os
from openai import AsyncOpenAI
import traceback

from env.base_env import AIWorkplaceEnv
from models.action import Action

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


API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN: str = os.getenv("HF_TOKEN")

class AutoAgentController:
    def __init__(self, env: AIWorkplaceEnv):
        self.env = env
        self.client = AsyncOpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
        self.model = MODEL_NAME
        self.history = []

    async def act(self, observation_dict: dict) -> Action:
        user_msg = (
            f"Current observation:\n{json.dumps(observation_dict, indent=2)}\n\n"
            "What action do you take? Respond with JSON only."
        )
        self.history.append({"role": "user", "content": user_msg})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self.history,
            temperature=0.0,
            max_tokens=256,
        )

        raw = response.choices[0].message.content.strip()
        self.history.append({"role": "assistant", "content": raw})

        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else parts[0]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data = json.loads(raw)
        return Action(action_type=data["action_type"], payload=data.get("payload"))

async def run_agent_loop(env: AIWorkplaceEnv):
    try:
        agent = AutoAgentController(env)
        obs = env.current_observation()
        
        while not obs.done:
            try:
                # Add a tiny sleep to avoid slamming the API too quickly if there are failures
                await asyncio.sleep(0.5)
                
                action = await agent.act(obs.model_dump())
                obs, reward, done, info = env.step(action)
                
            except json.JSONDecodeError as exc:
                print(f"[AutoAgent] JSON Decode Error: {exc}")
                # Provide an empty action to trigger an error state and let agent try again
                obs, reward, done, info = env.step(Action(action_type="invalid_json", payload={}))
            except Exception as exc:
                print(f"[AutoAgent] Loop Exception: {exc}")
                traceback.print_exc()
                break
                
    except Exception as e:
        print(f"[AutoAgent] Fatal Error: {e}")
        traceback.print_exc()

def start_auto_agent_background(env: AIWorkplaceEnv):
    asyncio.create_task(run_agent_loop(env))
