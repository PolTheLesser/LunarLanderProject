import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

# 1) Environment
env = gym.make(
    "LunarLander-v3",           # use "LunarLander-v2" if your gymnasium version doesn’t have v3
    render_mode=None,           # "human" for on-screen rendering
    continuous=False,
    gravity=-10.0,
    enable_wind=False,
    wind_power=15.0,
    turbulence_power=1.5,
)

# 2) DQN agent
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=50_000,
    learning_starts=1_000,
    batch_size=64,
    gamma=0.99,
    tau=1.0,
    target_update_interval=1_000,
    train_freq=4,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    exploration_fraction=0.3,
    verbose=1,
)

# 3) Train
model.learn(total_timesteps=300_000)

# 4) Evaluate
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"Mean reward: {mean_reward:.1f} +/- {std_reward:.1f}")

# 5) Watch one episode
test_env = gym.make("LunarLander-v3", render_mode="human", continuous=False)
obs, info = test_env.reset(seed=42)
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    done = terminated or truncated

test_env.close()
env.close()
