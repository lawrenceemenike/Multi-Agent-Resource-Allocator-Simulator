"""Unit tests for Milestone 6: Streamlit / Plotly Interactive Console."""
import pytest
from maras.visualization.app import run_simulation, plot_marshallian_cross, plot_sankey_routing


def test_simulation_data_caching_and_chart_generation():
    """Verify that simulation runs and produces valid Plotly figures for Marshallian curves and Sankey routing."""
    history = run_simulation(
        num_agents=4,
        num_steps=10,
        inject_adversary=True,
        adversary_mode="ADV-02",
        seed=123,
    )

    assert len(history) == 10
    first_step = history[0]
    assert "clearing_results" in first_step
    assert "agent_accounts" in first_step

    # Test Marshallian cross figure generation
    energy_clearing = first_step["clearing_results"].get("ENERGY", {})
    fig_cross = plot_marshallian_cross(energy_clearing, "ENERGY")
    assert fig_cross is not None
    assert len(fig_cross.data) >= 1

    # Test Sankey diagram figure generation
    trades = energy_clearing.get("trades", [])
    fig_sankey = plot_sankey_routing(trades, resource_filter="ENERGY")
    assert fig_sankey is not None
