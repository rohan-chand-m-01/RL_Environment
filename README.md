# Heapify AI Simulator / OpenEnv 1.1 

<div align="center">
  <br />
  <img src="https://img.shields.io/badge/Status-Hackathon_Ready-success?style=for-the-badge" alt="Status" />
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <br />
  <p><h3>A Complete RL Environment & Autonomous Agent Sandbox</h3></p>
</div>

---

## 🚀 Overview

**Heapify** (OpenEnv v1.1) is a lightweight, strictly deterministic Reinforcement Learning environment designed for evaluating and visualizing Large Language Model (LLM) agents. 

Built entirely with standard libraries and served via **FastAPI**, this system provides a premium, zero-dependency HTML/JS/CSS frontend—crafted in a brutalist, monochrome IDE aesthetic—to visualize the RL pipeline in real-time.

### Key Features
* 🧠 **Dense Reward Tracking:** Granular scoring systems (+0.3 for correct reasoning, -0.5 to -1.0 logic loop penalties).
* ⚡ **Auto-Agent Mode:** An asynchronous `agent_controller.py` that hooks seamlessly into Hugging Face/OpenAI standard inference endpoints. Let the LLM run on autopilot while you watch the logs.
* 🛡️ **Episode Consistency:** Strict RL lifecycle enforcement. Task switching is disabled during active episodes to ensure uncorrupted evaluation.
* 🖥️ **Brutalist IDE Interface:** High-contrast, hyper-responsive UI with auto-scrolling terminal logs, active step tracking, and dynamic action spaces.

---

## 🛠️ Installation & Setup

1. **Clone the Repository**
```bash
git clone https://github.com/rohan-chand-m-01/RL_Environment.git
cd RL_Environment
```

2. **Install Dependencies**
Ensure you have Python 3.9+ installed, then run:
```bash
pip install -r requirements.txt
```

3. **Set Environment Variables**
Configure your LLM provider credentials securely (do not commit these!):
```bash
export HF_TOKEN="your_hugging_face_token"
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
```
*(On Windows, use `set` or configure a `.env` file).*

4. **Launch the Server**
```bash
python server.py
```
> The application will be live at `http://localhost:7860`.

---

## 🎯 Task Environments

Current implemented environments inside the `env/graders/` directory:
1. **Email Triage:** Classify incoming emails (`urgent`, `normal`, `spam`).
2. **Code Review:** Detect syntax anomalies and deploy logical fixes.
3. **Data Cleaning:** Handle database integrity (remove nulls, normalize metrics, fix schema typos).

---

## 💻 Architecture

* `server.py`: FastAPI application serving REST endpoints and static assets.
* `agent_controller.py`: Asynchronous task loop that handles LLM inferences.
* `env/base_env.py`: The core state machine, observation builder, and dense reward governor.
* `index.html`: The zero-dependency, pure vanilla JS front-end terminal.

---

<div align="center">
  <i>Built for performance. Designed for humans. Executed by AI.</i>
</div>
