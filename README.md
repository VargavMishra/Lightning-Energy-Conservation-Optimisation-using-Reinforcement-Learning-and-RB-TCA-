🧠 Overview
The system mimics how an intelligent controller might decide between:

Storing energy for future use,

Converting it for immediate usage,

Dissipating it to avoid overloading or inefficiencies.

The approach combines:

A Deep Q-Network (DQN)-based RL agent.

A fallback rule-based controller that decides based on energy thresholds.

📁 Files
RLRB_TCAcode.py: Main simulation and training script.

No external config or data files are needed.

🚀 Features
Hybrid decision-making: 70% RL, 30% rule-based.

Dynamic environment with stochastic lightning energy.

Energy saturation and dissipation boundaries.

Training loop for policy improvement over episodes.

⚙️ Requirements
Ensure the following Python packages are installed:

bash
Copy
Edit
pip install torch numpy
🧪 How It Works
Environment (LightningEnv)
Initializes with random energy levels.

Accepts three actions: Store, Convert, and Dissipate.

Computes reward based on energy efficiency and safety.

RL Agent
Deep Q-Network (DQN) with 2 hidden layers.

Experience replay buffer.

ε-greedy exploration strategy.

Rule-Based Logic
Stores energy if below 3000 units.

Converts if energy is moderate (3000–8000).

Dissipates if energy is too high (>8000).

🧬 Training
Run the training loop:

bash
Copy
Edit
python RLRB_TCAcode.py
Sample output every 50 episodes:

yaml
Copy
Edit
Episode 250, Avg Reward: 0.93, Avg Energy: 7643.25, Epsilon: 0.223, Actions: Store=20, Convert=45, Dissipate=10
🎯 Goal
To demonstrate how hybrid intelligent systems can efficiently manage natural stochastic resources (like lightning energy) using both learned behavior and domain-based rules.
