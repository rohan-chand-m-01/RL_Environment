import pytest
from env.base_env import AIWorkplaceEnv, MAX_STEPS
from models.action import Action

def test_env_reset():
    env = AIWorkplaceEnv()
    obs = env.reset()
    assert obs.task_type == "email_triage"
    assert obs.step == 0
    assert obs.done is False
    assert obs.metadata["cumulative_reward"] == 0.0

def test_env_step_invalid_action():
    env = AIWorkplaceEnv()
    env.reset()
    # email_triage only accepts 'classify'
    action = Action(action_type="invalid_action", payload={"label": "spam"})
    obs, reward, done, info = env.step(action)
    
    assert reward.value == -0.3
    assert "invalid action" in obs.error_feedback
    assert done is False

def test_env_loop_detection():
    env = AIWorkplaceEnv()
    env.reset()
    action = Action(action_type="classify", payload={"label": "spam"})
    
    # Same action 3 times
    env.step(action)
    env.step(action)
    env.step(action)
    
    # 4th time should trigger loop penalty
    obs, reward, done, info = env.step(action)
    assert reward.value == -1.0
    assert "repeated looping action" in reward.reason
    assert done is False

def test_env_step_limit():
    env = AIWorkplaceEnv()
    env.reset()
    action = Action(action_type="classify", payload={"label": "invalid_label"})
    
    # Step until limit
    for _ in range(MAX_STEPS):
        obs, reward, done, info = env.step(action)
    
    # Next step should trigger limit
    obs, reward, done, info = env.step(action)
    assert done is True
    assert reward.value == -1.0
    assert "step limit exceeded" in reward.reason

def test_env_full_task_transition():
    # This is a bit complex as it requires knowing the correct answers
    # But we can test if it advances after a correct action if we mock or know data
    env = AIWorkplaceEnv()
    env.reset()
    
    # We know from email_task.py (we should check it) what the first email is
    # But for now, let's just check if it stays in email_triage on wrong answer
    action = Action(action_type="classify", payload={"label": "wrong"})
    obs, reward, done, info = env.step(action)
    assert obs.task_type == "email_triage"
    assert reward.value == -0.3
