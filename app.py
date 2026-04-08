"""
app.py — HuggingFace Spaces entry point.
Launches an interactive Gradio UI on 0.0.0.0:7860.
"""

import sys
import os
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.base_env import AIWorkplaceEnv
from models.action import Action

# ---------------------------------------------------------------------------
# Environment singleton
# ---------------------------------------------------------------------------
env = AIWorkplaceEnv()
current_obs = env.reset()
episode_log = []


def _fmt_obs(obs):
    content_lines = "\n".join(f"  {k}: {v}" for k, v in obs.content.items())
    history = obs.action_history[-5:] if obs.action_history else []
    return (
        f"### Task: **{obs.task_type.replace('_', ' ').title()}** — Step {obs.step}\n\n"
        f"**Content:**\n```\n{content_lines}\n```\n\n"
        f"**Cumulative Reward:** `{obs.metadata.get('cumulative_reward', 0.0):.2f}`  "
        f"| **Done:** `{obs.done}`  "
        f"| **Error:** `{obs.error_feedback or 'None'}`\n\n"
        f"**Recent Actions:** {history}"
    )


def do_step(action: Action):
    global current_obs
    obs, reward, done, info = env.step(action)
    current_obs = obs
    log_line = (
        f"[STEP] step={obs.step} action={action.action_type} "
        f"reward={reward.value:.2f} done={str(done).lower()} "
        f"error={obs.error_feedback or 'null'}"
    )
    episode_log.append(log_line)
    return _fmt_obs(obs), "\n".join(episode_log[-20:])


def handle_classify(label):
    return do_step(Action(action_type="classify", payload={"label": label}))


def handle_detect_bug(description):
    return do_step(Action(action_type="detect_bug", payload={"description": description}))


def handle_suggest_fix(fix):
    return do_step(Action(action_type="suggest_fix", payload={"fix": fix}))


def handle_data(action_type):
    return do_step(Action(action_type=action_type))


def reset_env():
    global current_obs, episode_log
    episode_log = []
    current_obs = env.reset()
    return _fmt_obs(current_obs), ""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="Heapify — OpenEnv",
    theme=gr.themes.Monochrome(),
    css="""
    .gr-button { font-family: monospace; }
    .observation-box { font-family: monospace; background: #f8f8f8; }
    """,
) as demo:

    gr.Markdown(
        """
# Heapify
**OpenEnv-compatible RL environment** · email triage · code review · data cleaning

> Interact manually with the environment. Each task has a programmatic grader that returns deterministic rewards.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            obs_view = gr.Markdown(value=_fmt_obs(current_obs), label="Observation")
        with gr.Column(scale=2):
            log_view = gr.Textbox(
                label="Episode Log (last 20 steps)",
                lines=12,
                max_lines=12,
                interactive=False,
            )

    gr.Markdown("---")
    gr.Markdown("### 📧 Task 1 — Email Triage")
    with gr.Row():
        label_radio = gr.Radio(
            choices=["urgent", "normal", "spam"],
            label="Classification Label",
            value="urgent",
        )
        classify_btn = gr.Button("▶ Classify Email", variant="primary")

    gr.Markdown("---")
    gr.Markdown("### 🐛 Task 2 — Code Review")
    with gr.Row():
        with gr.Column():
            bug_desc = gr.Textbox(label="Bug Description", placeholder="e.g. zero division error when list is empty")
            detect_btn = gr.Button("▶ Detect Bug")
        with gr.Column():
            fix_text = gr.Textbox(label="Fix Description", placeholder="e.g. check if list is empty before dividing")
            fix_btn = gr.Button("▶ Suggest Fix")

    gr.Markdown("---")
    gr.Markdown("### 🧹 Task 3 — Data Cleaning")
    with gr.Row():
        data_action = gr.Dropdown(
            choices=["remove_null", "normalize", "fix_schema"],
            label="Transformation",
            value="remove_null",
        )
        data_btn = gr.Button("▶ Apply Transformation")

    gr.Markdown("---")
    reset_btn = gr.Button("🔄 Reset Environment", variant="stop")

    # Wire up
    classify_btn.click(handle_classify, inputs=[label_radio], outputs=[obs_view, log_view])
    detect_btn.click(handle_detect_bug, inputs=[bug_desc], outputs=[obs_view, log_view])
    fix_btn.click(handle_suggest_fix, inputs=[fix_text], outputs=[obs_view, log_view])
    data_btn.click(handle_data, inputs=[data_action], outputs=[obs_view, log_view])
    reset_btn.click(reset_env, outputs=[obs_view, log_view])



if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
