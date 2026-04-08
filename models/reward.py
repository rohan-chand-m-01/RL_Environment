from pydantic import BaseModel
from typing import Optional


class Reward(BaseModel):
    value: float
    reason: str
    cumulative: float = 0.0
    success: Optional[bool] = None
