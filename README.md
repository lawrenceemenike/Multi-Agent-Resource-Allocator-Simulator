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

## 🎮 Interactive Dashboard Controls Guide (Plain English)

The MARAS Streamlit dashboard lets you act as the **Chief Economic Architect**. Every slider, dropdown, and toggle in the sidebar directly alters the economic physics of the world and reshapes how the AI agents behave, survive, and trade.

```
                          ┌───────────────────────────────────┐
                          │   Sidebar Configuration Knobs     │
                          └─────────────────┬─────────────────┘
                                            │
           ┌────────────────────────┬───────┴────────┬────────────────────────┐
           ▼                        ▼                ▼                        ▼
 👥 Population & Time      🏛️ Economic Rules   🦹 Market Sabotage      ⏱️ Turn-by-Turn Replay
  - Number of Agents       - Economy System    - Inject Adversary       - Timeline Scrubber
  - Simulation Turns       - Factory Engine    - Attack Strategy        - Commodity Selectors
```

---

### 1. World Configuration & Population

| Control & Type | What It Does (Plain English) | How It Affects the Players & Outcome |
| :--- | :--- | :--- |
| **Number of AI Agents**<br>`Slider: 3 to 12` | Sets the total population of independent AI participants in the market. | • **Low (3–4 agents):** Creates thin market liquidity. A single player can monopolize a resource, leading to wide bid-ask spreads, price volatility, and frequent shortages.<br>• **High (8–12 agents):** Creates fierce competition. More sellers compete for buyers, narrowing bid-ask spreads, stabilizing clearing prices, and boosting total manufacturing output. |
| **Simulation Turns**<br>`Slider: 10 to 100` | Determines the length of the economic timeline (how many trading days or epochs elapse). | • **Short (10–20 turns):** Tests immediate, short-term resilience to supply shocks.<br>• **Long (50–100 turns):** Reveals long-term wealth divergence. You will see whether impoverished households can recover or get trapped in permanent debt, and whether factory profits compound over time. |
| **Random World Seed**<br>`Number Input: 1 to 9999` | Sets the mathematical seed controlling initial wallet balances and starting inventories. | • Allows **100% reproducible experiments**. If you discover an interesting market crash or cartel event at Seed `42`, setting the seed back to `42` will replay the exact same market conditions. |

---

### 2. Economic System & Production Physics

| Control & Type | What It Does (Plain English) | How It Affects the Players & Outcome |
| :--- | :--- | :--- |
| **Economy System (Reward)**<br>`Dropdown: competitive, cooperative, mixed` | Governs what the AI agents are rewarded for—defining the societal morality of the economy. | • **Competitive (Pure Capitalism):** Each agent’s reward is 100% tied to its own personal wealth and profit. Players aggressively undercut rivals and hoard cash. **Outcome:** High GDP output, but extreme wealth inequality (Gini > 0.6) and struggling consumer households.<br>• **Cooperative (Social Welfare):** Each agent is rewarded based on the *average happiness of all citizens*. Players trade at fairer prices to prevent anyone from starving. **Outcome:** Near-zero poverty and low Gini inequality, but slightly lower aggregate factory output.<br>• **Mixed (Social Market Economy):** Blends 50% individual profit with 50% societal health. Creates healthy commercial competition while maintaining a safety net for essential commodities. |
| **Factory Engine**<br>`Dropdown: cobb_douglas, ces` | Governs the engineering physics of how factories convert raw inputs (Energy, Labor, Materials) into Finished Goods. | • **Cobb-Douglas (Strict Dependency):** Multiplicative production ($Y = A \cdot E^\alpha \cdot L^\beta \cdot M^\gamma$). If **even one input drops to zero** (e.g., a power blackout), total production drops to **zero**. **Outcome:** Factories will panic-bid whatever price it takes to secure scarce energy.<br>• **CES (Flexible Substitution):** Constant Elasticity of Substitution. If energy becomes unaffordable, factories can substitute it by hiring more manual labor or using alternative materials. **Outcome:** Dampens inflationary price spikes and grants factories more bargaining power during energy crunches. |

---

### 3. Market Sabotage & Adversarial Stress-Testing

| Control & Type | What It Does (Plain English) | How It Affects the Players & Outcome |
| :--- | :--- | :--- |
| **Inject Adversary Player**<br>`Checkbox: ON / OFF` | Spawns a predatory villain (`agent_0`) endowed with a massive $25,000 cash war chest designed to exploit the market. | • **OFF:** The market runs peacefully under pure organic supply and demand. Prices track genuine resource abundance.<br>• **ON:** Injects systemic stress. Tests whether honest producers and consumers can adapt their bidding strategies or if the market collapses into runaway inflation. |
| **Attack Strategy**<br>`Dropdown: ADV-02 (Cornering), ADV-01 (Spoofing)` | Chooses the specific financial exploit executed by the hostile agent. | • **ADV-02 Cornering (The Hoarder / Cartel):** The adversary aggressively outbids all honest players to buy up all available Energy, locks it in storage, and demands extortionate sell prices. **Outcome:** Factories face power blackouts, manufacturing output craters, energy prices spike, and societal happiness plunges.<br>• **ADV-01 Spoofing (The Manipulator):** The adversary places massive phantom buy orders right below the market price to trick other players into believing demand is soaring, then cancels them before settlement. **Outcome:** Misleads honest traders into overbidding, creating artificial market volatility. |

---

### 4. Replay & Microstructure Inspection

| Control & Type | What It Does (Plain English) | How It Affects the Players & Outcome |
| :--- | :--- | :--- |
| **Turn Scrubbing Timeline**<br>`Main Slider: Step 1 to Step N` | Acts as a time-travel DVR scrubber. Lets you slide back and forth across every turn in the simulation history. | • Updates the entire dashboard in real time: reveals who owned what, who traded with whom, what bids were placed, and what the Gini inequality score was at that exact split second. |
| **Dashboard View**<br>`Radio: Story Mode vs. Pro Analyst Mode` | Switches between a high-level narrative summary and deep econometric analytics. | • **Story Mode:** Shows news headlines, character portfolio cards (cash, inventory), visual trade cards, and Sankey resource flows.<br>• **Pro Analyst Mode:** Unlocks interactive **Marshallian Cross** supply & demand curves with equilibrium detection ($P^*, Q^*$) and order book bid/ask depth. |
| **Select Commodity Market**<br>`Dropdown (Pro Mode): ENERGY, LABOR, RAW_MATERIALS, FINISHED_GOODS` | Isolates the supply and demand curves for a single specific good. | • Displays the exact clearing price where buyer bids crossed seller asks, revealing consumer surplus, producer profit margins, or unmet shortage gaps. |

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

