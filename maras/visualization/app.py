"""Streamlit & Plotly Interactive Console for MARAS.

Upgraded with Story Mode (Avatars, Live Headlines, Health Gauges, Visual Cards)
and Pro Analyst Mode (Marshallian Cross, Sankey Flow Routing, Econometrics).
"""
from __future__ import annotations

import copy
import importlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, List, Optional, Any

import maras.environment.economy_env
import maras.cognition.gemma_explainer
import maras.metrics.econometrics

# Force fresh module reload on every Streamlit script execution
importlib.reload(maras.environment.economy_env)
importlib.reload(maras.cognition.gemma_explainer)
importlib.reload(maras.metrics.econometrics)

from maras.schemas.contracts import ResourceType, PersonaType, EpisodeStepLog
from maras.environment.economy_env import EconomyEnv
from maras.metrics.econometrics import calculate_gini, calculate_hhi, detect_anomalies
from maras.agents.adversary import AdversaryAgent
from maras.cognition.gemma_explainer import GemmaExplainer


# Page configuration
st.set_page_config(
    page_title="MARAS | Living AI Economy Simulator",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for modern dark glassmorphism aesthetic
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b0e14 0%, #121824 100%);
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    .headline-card {
        background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
        border-left: 5px solid #38bdf8;
        border-radius: 8px;
        padding: 14px 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .headline-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .headline-sub {
        font-size: 0.9rem;
        color: #94a3b8;
    }
    .agent-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 14px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease;
    }
    .agent-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
    }
    .role-badge-producer { background-color: #0284c7; color: white; padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }
    .role-badge-consumer { background-color: #10b981; color: white; padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }
    .role-badge-speculator { background-color: #8b5cf6; color: white; padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }
    .role-badge-adversary { background-color: #ef4444; color: white; padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }
    .alert-banner {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 12px 16px;
        color: #fca5a5;
        font-weight: 600;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def run_simulation(
    num_agents: int = 6,
    num_steps: int = 35,
    inject_adversary: bool = True,
    adversary_mode: str = "ADV-02",
    reward_mode: str = "competitive",
    production_type: str = "cobb_douglas",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Runs a multi-step simulation episode and returns serialized step history logs."""
    env = EconomyEnv(
        num_agents=num_agents,
        max_steps=num_steps,
        reward_mode=reward_mode,
        production_type=production_type,
        seed=seed,
    )
    obs, _ = env.reset(seed=seed)

    adv_agent = None
    if inject_adversary and len(env.possible_agents) > 0:
        adv_id = env.possible_agents[0]
        env.personas[adv_id] = PersonaType.ADVERSARY
        env.accounts[adv_id].persona = PersonaType.ADVERSARY
        env.accounts[adv_id].cash = 25000.0  # War-chest for cornering
        adv_agent = AdversaryAgent(
            agent_id=adv_id,
            attack_mode=adversary_mode,
            target_resource=ResourceType.ENERGY.value,
        )

    for step in range(num_steps):
        actions = {}
        for agent_id in env.agents:
            if inject_adversary and agent_id == env.possible_agents[0] and adv_agent is not None:
                actions[agent_id] = adv_agent.get_continuous_action(env.resources, env.accounts[agent_id])
            else:
                actions[agent_id] = np.random.uniform(-1.5, 1.5, size=env.action_spaces[agent_id].shape).astype(np.float32)

        obs, rewards, terminations, truncations, infos = env.step(actions)
        if any(truncations.values()) or not env.agents:
            break

    return [log.model_dump() for log in env.step_history]


def generate_turn_headline(step_log: Dict[str, Any], step_num: int) -> Tuple[str, str, str]:
    """Generates an intuitive, story-driven news headline for the current turn."""
    alerts = step_log.get("anomaly_alerts", [])
    total_trades = sum(len(clr.get("trades", [])) for clr in step_log.get("clearing_results", {}).values())
    finished_goods_made = sum(step_log.get("production_output", {}).values())

    if alerts:
        # Check which resource was attacked
        for res in ["ENERGY", "LABOR", "RAW_MATERIALS", "FINISHED_GOODS"]:
            if any(res in a for a in alerts):
                return (
                    f"🚨 Turn {step_num} Market Panic: Monopolist Corners {res} Supply!",
                    f"A dominant buyer has stockpiled the majority of the world's {res}. Prices and scarcity have surged, threatening other players.",
                    "alert",
                )

    if finished_goods_made > 5.0:
        return (
            f"🏭 Turn {step_num} Industrial Boom: Factories Produce {finished_goods_made:.1f} Goods!",
            f"Producers successfully sourced power and materials to manufacture goods, keeping store shelves stocked.",
            "positive",
        )

    if total_trades > 0:
        return (
            f"🛒 Turn {step_num} Active Trading Day: {total_trades} Commercial Deals Settled",
            "Buyers and sellers smoothly exchanged labor, materials, and currency at competitive equilibrium prices.",
            "neutral",
        )

    return (
        f"⏳ Turn {step_num} Quiet Market: Bids & Asks Stalled",
        "Prices bid by buyers were below sellers' minimum asking prices. No trades crossed this turn.",
        "neutral",
    )


def render_health_gauge(step_log: Dict[str, Any]) -> go.Figure:
    """Renders an intuitive 0-100% Economy Health Gauge."""
    gini = step_log.get("gini_wealth", 0.5)
    max_hhi = max(step_log.get("hhi_by_resource", {}).values()) if step_log.get("hhi_by_resource") else 2500.0

    # Calculate overall health score (0-100)
    # Low Gini (good) + Low HHI (good) = High Score
    gini_penalty = gini * 40.0
    hhi_penalty = (max(0.0, max_hhi - 2500.0) / 7500.0) * 50.0
    health_score = max(5.0, min(100.0, 100.0 - gini_penalty - hhi_penalty))

    if health_score >= 70:
        color = "#10b981"  # Emerald Green
        status_text = "Thriving & Fair"
    elif health_score >= 45:
        color = "#f59e0b"  # Amber
        status_text = "Strained Supply"
    else:
        color = "#ef4444"  # Red
        status_text = "Monopoly Crisis"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=health_score,
            title={"text": f"<b>Economy Health</b><br><span style='font-size:0.8em;color:{color}'>{status_text}</span>", "font": {"size": 16}},
            number={"suffix": "%", "font": {"size": 28, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white"},
                "bar": {"color": color, "thickness": 0.3},
                "bgcolor": "rgba(255,255,255,0.05)",
                "borderwidth": 1,
                "bordercolor": "rgba(255,255,255,0.1)",
                "steps": [
                    {"range": [0, 45], "color": "rgba(239, 68, 68, 0.2)"},
                    {"range": [45, 70], "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [70, 100], "color": "rgba(16, 185, 129, 0.2)"},
                ],
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=200,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_marshallian_cross(clearing_result: Dict[str, Any], resource_name: str) -> go.Figure:
    """Renders Marshallian Supply and Demand curves with equilibrium crossing."""
    fig = go.Figure()

    demand_curve = clearing_result.get("demand_curve", [])
    supply_curve = clearing_result.get("supply_curve", [])
    clearing_p = clearing_result.get("clearing_price")
    clearing_q = clearing_result.get("clearing_volume", 0.0)

    if demand_curve:
        d_p = [p for p, q in demand_curve]
        d_q = [q for p, q in demand_curve]
        fig.add_trace(
            go.Scatter(
                x=d_q,
                y=d_p,
                mode="lines+markers",
                name="Demand Curve (Bids)",
                line=dict(color="#38bdf8", width=3, shape="hv"),
                marker=dict(size=6, color="#38bdf8"),
            )
        )

    if supply_curve:
        s_p = [p for p, q in supply_curve]
        s_q = [q for p, q in supply_curve]
        fig.add_trace(
            go.Scatter(
                x=s_q,
                y=s_p,
                mode="lines+markers",
                name="Supply Curve (Asks)",
                line=dict(color="#34d399", width=3, shape="hv"),
                marker=dict(size=6, color="#34d399"),
            )
        )

    if clearing_p is not None and clearing_q > 0:
        fig.add_trace(
            go.Scatter(
                x=[clearing_q],
                y=[clearing_p],
                mode="markers",
                name=f"Equilibrium (P*=${clearing_p:.2f}, Q*={clearing_q:.2f})",
                marker=dict(color="#f43f5e", size=14, symbol="star"),
            )
        )
        fig.add_hline(y=clearing_p, line_dash="dash", line_color="#f43f5e", opacity=0.6)
        fig.add_vline(x=clearing_q, line_dash="dash", line_color="#f43f5e", opacity=0.6)

    fig.update_layout(
        title=f"<b>Marshallian Cross: {resource_name} Market Equilibrium</b>",
        xaxis_title="Quantity",
        yaxis_title="Price ($)",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=380,
    )
    return fig


def plot_sankey_routing(trades_list: List[Dict[str, Any]], resource_filter: Optional[str] = None) -> go.Figure:
    """Constructs Plotly Sankey diagram representing multi-agent resource and currency routing."""
    filtered_trades = [t for t in trades_list if (resource_filter is None or t["resource_id"] == resource_filter)]

    if not filtered_trades:
        fig = go.Figure()
        fig.update_layout(
            title="<b>Trade Flow Routing</b> (No matched trades crossing in this turn)",
            template="plotly_dark",
            height=380,
        )
        return fig

    nodes = []
    node_indices = {}

    def get_node_idx(name: str) -> int:
        if name not in node_indices:
            node_indices[name] = len(nodes)
            nodes.append(name)
        return node_indices[name]

    sources = []
    targets = []
    values = []

    for t in filtered_trades:
        seller = f"Seller: {t['seller_id']}"
        buyer = f"Buyer: {t['buyer_id']}"
        market = f"Market: {t['resource_id']}"

        s_idx = get_node_idx(seller)
        m_idx = get_node_idx(market)
        b_idx = get_node_idx(buyer)

        sources.append(s_idx)
        targets.append(m_idx)
        values.append(t["quantity"])

        sources.append(m_idx)
        targets.append(b_idx)
        values.append(t["quantity"])

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=nodes,
                    color="#6366f1",
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color="rgba(56, 189, 248, 0.35)",
                ),
            )
        ]
    )
    fig.update_layout(
        title="<b>Multi-Agent Resource Settlement Flow</b>",
        template="plotly_dark",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def main():
    # Sidebar: Simulation Controls
    st.sidebar.title("🏙️ MARAS Simulator")
    view_mode = st.sidebar.radio("Dashboard View", ["🌟 Story Mode (Beginner)", "🔬 Pro Analyst Mode"], index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ World Configuration")
    num_agents = st.sidebar.slider("Number of AI Agents", min_value=3, max_value=12, value=6)
    num_steps = st.sidebar.slider("Simulation Turns", min_value=10, max_value=100, value=35)
    reward_mode = st.sidebar.selectbox("Economy System (Reward)", ["competitive", "cooperative", "mixed"], index=0,
                                       help="Competitive = Free market greed. Cooperative = Shared social welfare. Mixed = Balanced.")
    production_type = st.sidebar.selectbox("Factory Engine", ["cobb_douglas", "ces"], index=0)

    st.sidebar.markdown("### 🦹 Adversary Threat Injection")
    inject_adversary = st.sidebar.checkbox("Inject Adversary Player", value=True, help="Spawns an aggressive hostile agent trying to break the market.")
    adversary_mode = st.sidebar.selectbox("Attack Strategy", ["ADV-02 (Cornering)", "ADV-01 (Spoofing)"])
    adv_mode_code = "ADV-02" if "ADV-02" in adversary_mode else "ADV-01"

    sim_seed = st.sidebar.number_input("Random World Seed", min_value=1, max_value=9999, value=42)

    # Run Simulation
    history_raw = run_simulation(
        num_agents=num_agents,
        num_steps=num_steps,
        inject_adversary=inject_adversary,
        adversary_mode=adv_mode_code,
        reward_mode=reward_mode,
        production_type=production_type,
        seed=sim_seed,
    )

    if not history_raw:
        st.error("No simulation history available.")
        return

    # Main Header
    st.title("🏙️ Multi-Agent Resource Allocation Simulator")
    st.caption("A living AI mini-economy where Autonomous Agents trade energy, labor, and goods in real-time double auctions.")

    total_recorded_steps = len(history_raw)
    current_step_idx = st.slider(
        "⏱️ **Turn Scrubbing Timeline**",
        min_value=1,
        max_value=total_recorded_steps,
        value=total_recorded_steps,
        help="Drag slider back and forth to inspect turn-by-turn trades and player decisions.",
    )

    current_log = history_raw[current_step_idx - 1]

    # Live Headline News Card
    head_title, head_sub, head_type = generate_turn_headline(current_log, current_step_idx)
    border_color = "#ef4444" if head_type == "alert" else ("#10b981" if head_type == "positive" else "#38bdf8")
    st.markdown(
        f"""
        <div class="headline-card" style="border-left-color: {border_color};">
            <div class="headline-title">{head_title}</div>
            <div class="headline-sub">{head_sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top KPI Metrics & Health Gauge
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns([1.2, 1, 1, 1])

    with kpi_col1:
        st.plotly_chart(render_health_gauge(current_log), use_container_width=True)

    with kpi_col2:
        gini = current_log.get("gini_wealth", 0.0)
        st.metric("💰 Wealth Inequality (Gini)", f"{gini:.2f}",
                  help="0.0 = Everyone equal. 1.0 = 1 person has all the money.")
        st.caption("High inequality" if gini > 0.6 else "Moderate distribution")

    with kpi_col3:
        welfare = current_log.get("social_welfare", 0.0)
        st.metric("❤️ Society Happiness", f"{welfare:.1f}", help="Average satisfaction score across all players.")
        st.caption("Higher is healthier")

    with kpi_col4:
        prod_total = sum(current_log.get("production_output", {}).values())
        st.metric("📦 Goods Manufactured", f"{prod_total:.1f} units", help="Total finished products made by factories this turn.")
        st.caption(f"Step {current_step_idx} output")

    st.markdown("---")

    # TABS
    tab_agents, tab_market, tab_macro, tab_gemma = st.tabs([
        "👥 AI Character Portfolios",
        "🏪 Interactive Marketplace",
        "📈 Economy Trajectories",
        "🧠 Gemma Chief Economist Report",
    ])

    # TAB 1: Character Portfolios
    with tab_agents:
        st.markdown("### 👥 The Players of the Economy")
        st.caption("Inspect each AI player's bank balance, raw supplies, and current status in the economy.")

        cols = st.columns(3)
        agent_items = list(current_log["agent_accounts"].items())

        for idx, (aid, acc) in enumerate(agent_items):
            col_target = cols[idx % 3]
            persona = acc["persona"]

            if persona == "PRODUCER":
                icon = "🏭"
                badge = '<span class="role-badge-producer">FACTORY OWNER</span>'
                desc = "Buys Energy & Ore → Manufactures Finished Goods"
            elif persona == "CONSUMER":
                icon = "👨‍👩‍👧"
                badge = '<span class="role-badge-consumer">HOUSEHOLD WORKER</span>'
                desc = "Sells Labor → Buys Finished Goods & Energy"
            elif persona == "SPECULATOR":
                icon = "💼"
                badge = '<span class="role-badge-speculator">COMMODITY TRADER</span>'
                desc = "Buys low and sells high across turns"
            else:
                icon = "🦹"
                badge = '<span class="role-badge-adversary">MARKET VILLAIN</span>'
                desc = "Attempts hostile cornering & price manipulation attacks"

            inv = acc["inventory"]
            cash = acc["cash"]

            with col_target:
                st.markdown(
                    f"""
                    <div class="agent-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                            <span style="font-size:1.1rem; font-weight:700;">{icon} {aid}</span>
                            {badge}
                        </div>
                        <div style="font-size:0.8rem; color:#94a3b8; margin-bottom:12px;">{desc}</div>
                        <div style="font-size:1.1rem; font-weight:700; color:#38bdf8; margin-bottom:12px;">
                            💵 Wallet: ${cash:,.2f}
                        </div>
                        <div style="font-size:0.85rem;">
                            <div>⚡ <b>Energy:</b> {inv.get('ENERGY', 0.0):.1f} units</div>
                            <div>🪵 <b>Materials:</b> {inv.get('RAW_MATERIALS', 0.0):.1f} units</div>
                            <div>🔨 <b>Labor:</b> {inv.get('LABOR', 0.0):.1f} units</div>
                            <div>📦 <b>Finished Goods:</b> {inv.get('FINISHED_GOODS', 0.0):.1f} units</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # TAB 2: Marketplace
    with tab_market:
        if view_mode == "🌟 Story Mode (Beginner)":
            st.markdown("### 🏪 Turn Settlement Feed: Who Traded With Whom?")
            all_trades = []
            for res, clr in current_log["clearing_results"].items():
                all_trades.extend(clr.get("trades", []))

            if all_trades:
                trade_cols = st.columns(2)
                for t_idx, t in enumerate(all_trades):
                    c = trade_cols[t_idx % 2]
                    c.info(
                        f"🤝 **{t['resource_id']} Deal:** Buyer `{t['buyer_id']}` bought **{t['quantity']:.1f} units** from Seller `{t['seller_id']}` at **${t['price']:.2f} / unit** (Total: **${t['price']*t['quantity']:.2f}**)"
                    )
            else:
                st.warning("No matched trades crossing this turn. Buyers' bids were lower than sellers' asks.")

            st.markdown("---")
            st.markdown("### 📊 Market Flow Routing")
            fig_sankey = plot_sankey_routing(all_trades)
            st.plotly_chart(fig_sankey, use_container_width=True)

        else:  # Pro Analyst Mode
            st.markdown("### 🔬 Pro Market Microstructure & Marshallian Curves")
            col_curve, col_sankey = st.columns([1, 1])

            with col_curve:
                available_res = list(current_log["clearing_results"].keys())
                selected_res = st.selectbox("Select Commodity Market", available_res, index=0)
                res_clearing = current_log["clearing_results"].get(selected_res, {})
                fig_cross = plot_marshallian_cross(res_clearing, selected_res)
                st.plotly_chart(fig_cross, use_container_width=True)

            with col_sankey:
                all_trades = []
                for res, clr in current_log["clearing_results"].items():
                    all_trades.extend(clr.get("trades", []))
                fig_sankey = plot_sankey_routing(all_trades)
                st.plotly_chart(fig_sankey, use_container_width=True)

    # TAB 3: Macro Trajectories
    with tab_macro:
        st.markdown("### 📈 Economic Health Over Time")
        steps = [log["step"] for log in history_raw]
        ginis = [log.get("gini_wealth", 0.0) for log in history_raw]
        welfares = [log.get("social_welfare", 0.0) for log in history_raw]

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=steps, y=ginis, mode="lines+markers", name="Inequality (Gini)", line=dict(color="#f59e0b", width=2.5)))
        fig_ts.add_trace(go.Scatter(x=steps, y=welfares, mode="lines+markers", name="Society Happiness (Welfare)", line=dict(color="#38bdf8", width=2.5), yaxis="y2"))
        fig_ts.add_vline(x=current_step_idx, line_dash="dash", line_color="#f43f5e", opacity=0.8)

        fig_ts.update_layout(
            title="<b>Macroeconomic Trajectory: Inequality vs Happiness</b>",
            xaxis_title="Simulation Turn",
            yaxis=dict(title="Gini Coefficient (0=Equal, 1=Unequal)", range=[0, 1]),
            yaxis2=dict(title="Social Welfare Score", overlaying="y", side="right"),
            template="plotly_dark",
            height=400,
            hovermode="x unified",
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    # TAB 4: Gemma Report
    with tab_gemma:
        st.markdown("### 🧠 Gemma Chief Economist Report")
        st.caption("AI-powered strategic debrief analyzing inflation, winners/losers, supply bottlenecks, and market integrity.")

        col_btn, _ = st.columns([1, 3])
        with col_btn:
            refresh = st.button("🔄 Re-Analyze World Data", use_container_width=True)

        config_signature = f"{num_agents}_{num_steps}_{inject_adversary}_{adversary_mode}_{reward_mode}_{production_type}_{sim_seed}"
        if refresh or st.session_state.get("last_config") != config_signature or "gemma_analysis" not in st.session_state:
            with st.spinner("Analyzing macroeconomic dynamics..."):
                pydantic_logs = [EpisodeStepLog.model_validate(log) for log in history_raw]
                explainer = GemmaExplainer(ollama_url="http://127.0.0.1:11434")
                analysis_text = explainer.generate_episode_explanation(pydantic_logs)
                explainer.shutdown()
                st.session_state["gemma_analysis"] = analysis_text
                st.session_state["last_config"] = config_signature

        if "gemma_analysis" in st.session_state:
            st.markdown(st.session_state["gemma_analysis"])


if __name__ == "__main__":
    main()
