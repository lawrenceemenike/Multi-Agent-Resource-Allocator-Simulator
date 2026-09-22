"""Unit tests for Milestone 4: Gemma Cognitive Explainer."""
import time
import pytest
from maras.schemas.contracts import (
    EpisodeStepLog,
    AgentAccount,
    PersonaType,
    ResourceType,
    AuctionClearingResult,
)
from maras.cognition.gemma_explainer import GemmaExplainer


def create_sample_episode_logs(num_steps: int = 5) -> list[EpisodeStepLog]:
    logs = []
    for s in range(1, num_steps + 1):
        accounts = {
            "agent_0": AgentAccount(
                agent_id="agent_0",
                cash=500.0 + s * 10,
                inventory={ResourceType.ENERGY.value: 10.0, ResourceType.FINISHED_GOODS.value: 5.0},
                persona=PersonaType.PRODUCER,
            ),
            "agent_1": AgentAccount(
                agent_id="agent_1",
                cash=700.0 - s * 5,
                inventory={ResourceType.LABOR.value: 8.0},
                persona=PersonaType.CONSUMER,
            ),
        }
        clearing = {
            ResourceType.ENERGY.value: AuctionClearingResult(
                resource_id=ResourceType.ENERGY.value,
                clearing_price=12.5,
                clearing_volume=20.0,
            )
        }
        logs.append(
            EpisodeStepLog(
                step=s,
                clearing_results=clearing,
                agent_accounts=accounts,
                gini_wealth=0.25,
                social_welfare=18.5,
            )
        )
    return logs


def test_explainer_prompt_construction():
    explainer = GemmaExplainer(ollama_url="http://127.0.0.1:9999", max_queue_size=10)
    logs = create_sample_episode_logs(5)
    prompt = explainer.build_prompt(logs)

    assert "5 steps" in prompt
    assert "ENERGY" in prompt
    assert "Gini" in prompt
    explainer.shutdown()


def test_explainer_bounded_queue_and_async_processing():
    explainer = GemmaExplainer(ollama_url="http://127.0.0.1:9999", max_queue_size=2)
    logs = create_sample_episode_logs(3)

    submitted = explainer.submit_async(task_id="task_1", episode_logs=logs)
    assert submitted is True

    processed = False
    for _ in range(20):
        time.sleep(0.05)
        if "task_1" in explainer.results_cache:
            processed = True
            break

    assert processed is True
    explanation = explainer.results_cache["task_1"]
    assert len(explanation) > 50
    assert "Chief Economist" in explanation or "Macroeconomic" in explanation

    explainer.shutdown()
