import sys
from env.base_env import AIWorkplaceEnv
from models.action import Action

def run_smoke_test():
    print("[SMOKE TEST] Initializing environment...")
    env = AIWorkplaceEnv()
    obs = env.reset()
    
    total_reward = 0.0
    step_count = 0
    
    print(f"[START] task={obs.task_type}")
    
    while not obs.done:
        step_count += 1
        action = None
        
        # Rule-based agent logic
        if obs.task_type == "email_triage":
            # The emails are in a fixed order, but we can look at the subject to be sure
            subject = obs.content.get("subject", "").lower()
            if "urgent" in subject or "critical" in subject or "server down" in subject:
                label = "urgent"
            elif "won" in subject or "pills" in subject or "prize" in subject:
                label = "spam"
            else:
                label = "normal"
            action = Action(action_type="classify", payload={"label": label})
            
        elif obs.task_type == "code_review":
            # If bug not detected, detect it
            if not obs.content.get("bug_detected"):
                code = obs.content.get("code", "").lower()
                if "division by zero" in code or "calculate_average" in code:
                    desc = "potential zero division error"
                elif "find_duplicates" in code or "j in range" in code:
                    desc = "self comparison bug in range"
                elif "binary_search" in code:
                    desc = "off by one error in right boundary"
                else:
                    desc = "code bug"
                action = Action(action_type="detect_bug", payload={"description": desc})
            else:
                # Suggest fix
                code = obs.content.get("code", "").lower()
                if "calculate_average" in code:
                    fix = "check if list is empty"
                elif "find_duplicates" in code:
                    fix = "use range(i+1, len(lst))"
                elif "binary_search" in code:
                    fix = "use len(arr) - 1"
                else:
                    fix = "fix code"
                action = Action(action_type="suggest_fix", payload={"fix": fix})
                
        elif obs.task_type == "data_cleaning":
            required = obs.content.get("required_steps", [])
            completed = obs.content.get("completed_steps", [])
            
            # Find next required step
            next_step = None
            for s in ["remove_null", "fix_schema", "normalize"]: # Order matters for success
                if s in required and s not in completed:
                    next_step = s
                    break
            
            if next_step:
                action = Action(action_type=next_step)
            else:
                # Should not happen if logic is correct
                action = Action(action_type="remove_null")
        
        if not action:
            print("No action determined, stopping.")
            break
            
        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        
        print(f"[STEP] {step_count}: task={info['task']} action={action.action_type} reward={reward.value:.2f} total={total_reward:.2f}")
        
        if obs.error_feedback:
            print(f"       ERROR: {obs.error_feedback}")

    print(f"\n[END] Success: {obs.done} Steps: {step_count} Total Reward: {total_reward:.2f}")
    
    if total_reward > 5.0 and obs.done:
        print("Smoke test PASSED!")
        return True
    else:
        print("Smoke test FAILED or low score.")
        return False

if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
