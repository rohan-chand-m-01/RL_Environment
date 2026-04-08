from pydantic import BaseModel
from typing import Any, Dict, Optional


class Action(BaseModel):
    action_type: str  # e.g. "classify", "suggest_fix", "detect_bug", "remove_null", etc.
    payload: Optional[Dict[str, Any]] = None  # action-specific parameters
