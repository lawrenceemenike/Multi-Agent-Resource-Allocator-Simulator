# MARAS: Multi-Agent Resource Allocation Simulator

A high-performance deterministic double-auction and multi-agent resource allocation simulation platform with PettingZoo, Centralized Critic PPO, Gemma Cognitive Explainer, and interactive Streamlit/Plotly dashboard.

---

## 🚀 Features by Milestone

1. **Milestone 1: Deterministic Double Auction & Escrow Validation**
   - Discrete double auction clearing with uniform equilibrium price determination.
   - Pre-clearing escrow scaling: strictly guarantees zero negative cash balances and prevents naked shorting.
   - Pro-rata tie-breaking when bids or asks match at identical marginal price levels.
   - Core files: [`maras/schemas/contracts.py`](file:///c:/Users/clene/Downloads/MULTI-AGENT%20RESOURCE%20ALLOCATION%20SIMULATOR/maras/schemas/contracts.py), [`maras/environment/auction.py`](file:///c:/Users/clene/Downloads/MULTI-AGENT%20RESOURCE%20ALLOCATION%20SIMULATOR/maras/environment/auction.py).

2. **Milestone 2: PettingZoo Parallel Environment & Production Functions**
   - PettingZoo `ParallelEnv` compliant simulation (`EconomyEnv`).
   - Production functions: Cobb-Douglas ($Y = A \prod X_i^{\alpha_i}$) and CES ($Y = A (\sum \alpha_i X_i^\rho)^{\gamma/\rho}$).
   - Action projection: log-uniform price scaling and softmax budget allocation with strict resource conservation.
   - Core files: [`maras/environment/economy_env.py`](file:///c:/Users/clene/Downloads/MULTI-AGENT%20RESOURCE%20ALLOCATION%20SIMULATOR/maras/environment/economy_env.py).

3. **Milestone 3: MAPPO Policy with Centralized Critic**
   - Multi-Agent PPO featuring persona-conditioned decentralized actors and centralized state-value critics.
   - Reward schedules: Competitive, Cooperative (Social Welfare), and Mixed ($\lambda$-weighted).
   - Core files: [`maras/policies/ppo_wrapper.py`](file:///c:/Users/clene/Downloads/MULTI-AGENT%20RESOURCE%20ALLOCATION%20SIMULATOR/maras/policies/ppo_wrapper.py).

4. **Milestone 4: Gemma Cognitive Explainer (Isolated Replay Mode)**
   - Bounded async worker queue for non-blocking macroeconomic replay analysis.
   - Ollama integration (`gemma4:12b` / `gemma:latest`) with instant local fallback.
   - Core files: [`maras/cognition/gemma_explainer.py`](file:///c:/Users/clene/Downloads/MULTI-AGENT%20RESOURCE%20ALLOCATION%20SIMULATOR/maras/cognition/gemma_explainer.py).

5. **Milestone 5: Adversarial Stress Testing & Anomaly Metrics**
   - Adversary attacks: `ADV-01` (Spoofing) and `ADV-02` (Monopoly Cornering).
   - Econometric metrics: Wealth Gini coefficient and Resource Herfindahl-Hirschman Index (HHI).
   - Anomaly detection: Triggers alerts whenever energy market $HHI > 4,000$.
   - Core files: [`maras/agents/adversary.py`](file:///c:/Users/clene/Downloads/MULTI-AGENT%20RESOURCE%20ALLOCATION%20SIMULATOR/maras/agents/adversary.py), [`maras/metrics/econometrics.py`](file:///c:/Users/clene/Downloads/MULTI-AGENT%20RESOURCE%20ALLOCATION%20SIMULATOR/maras/metrics/econometrics.py).

6. **Milestone 6: Streamlit / Plotly Interactive Console**
   - Turn-by-turn episode scrubbing slider.
   - Interactive Marshallian cross supply & demand curves with equilibrium detection.
   - Sankey trade flow routing across agents and commodity markets.
   - Gemma cognitive narrative pane and cached simulation replay.
   - Core files: [`maras/visualization/app.py`](file:///c:/Users/clene/Downloads/MULTI-AGENT%20RESOURCE%20ALLOCATION%20SIMULATOR/maras/visualization/app.py).

---

## 🛠️ Quickstart

### 1. Run Unit & Integration Tests
```bash
.\.venv\Scripts\pytest.exe tests/ -v
```

### 2. Launch Interactive Dashboard
```bash
.\.venv\Scripts\streamlit.exe run maras/visualization/app.py
```
