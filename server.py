from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional
import os

from env.base_env import AIWorkplaceEnv
from models.action import Action
from models.observation import Observation
from models.reward import Reward
from agent_controller import start_auto_agent_background

app = FastAPI(title="Heapify API")

# Singleton environment for simplicity in this simulator
env = AIWorkplaceEnv()

class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]

@app.get("/")
async def root():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Heapify is running. index.html not found.", "version": "1.0.0"}

@app.post("/api/reset", response_model=Observation)
async def reset():
    return env.reset()

@app.post("/api/step", response_model=StepResponse)
async def step(action: Action):
    obs, reward, done, info = env.step(action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)

@app.get("/api/state")
async def get_state():
    return env.state()

@app.get("/api/current_obs", response_model=Observation)
async def get_current_obs():
    return env.current_observation()

@app.post("/api/auto_agent/start")
async def start_auto_agent():
    start_auto_agent_background(env)
    return {"status": "started"}

if __name__ == "__main__":
    import uvicorn
    # Make sure we run on 7860 which HF Spaces expects by default
    uvicorn.run(app, host="0.0.0.0", port=7860)
