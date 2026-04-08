from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class Observation(BaseModel):
    task_id: str
    task_type: str  # "email_triage" | "code_review" | "data_cleaning"
    step: int
    content: Dict[str, Any]  # task-specific payload
    action_history: List[str] = []
    error_feedback: Optional[str] = None
    done: bool = False
    metadata: Dict[str, Any] = {}
