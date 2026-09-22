"""Gemma Cognitive Explainer with dynamic macroeconomic analytics, price trend tracking, and anomaly reporting."""
from __future__ import annotations

import json
import queue
import socket
import threading
from typing import Dict, List, Optional, Any, Callable
from urllib.parse import urlparse
import httpx

from maras.schemas.contracts import EpisodeStepLog


class GemmaExplainer:
    """Explains multi-agent market dynamics, price discovery, and strategic behavior using Gemma or high-fidelity dynamic analytics."""

    def __init__(
        self,
        ollama_url: str = "http://127.0.0.1:11434",
        model_name: str = "gemma:latest",
        max_queue_size: int = 50,
        timeout: float = 3.0,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.task_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.results_cache: Dict[str, str] = {}

        self._shutdown_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _is_server_online(self) -> bool:
        """Fast check to determine if Ollama server is actively listening without stalling."""
        try:
            parsed = urlparse(self.ollama_url)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or 11434
            with socket.create_connection((host, port), timeout=0.05):
                return True
        except Exception:
            return False

    def _worker_loop(self) -> None:
        """Background daemon processing queued replay analysis tasks asynchronously."""
        while not self._shutdown_event.is_set():
            try:
                task = self.task_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            task_id, episode_logs, callback = task
            try:
                explanation = self.generate_episode_explanation(episode_logs)
                self.results_cache[task_id] = explanation
                if callback:
                    callback(task_id, explanation)
            except Exception as e:
                err_msg = f"Error generating explanation for task {task_id}: {str(e)}"
                self.results_cache[task_id] = err_msg
                if callback:
                    callback(task_id, err_msg)
            finally:
                self.task_queue.task_done()

    def submit_async(
        self,
        task_id: str,
        episode_logs: List[EpisodeStepLog],
        callback: Optional[Callable[[str, str], None]] = None,
    ) -> bool:
        """Submits an episode log sequence to the bounded background queue."""
        try:
            self.task_queue.put_nowait((task_id, episode_logs, callback))
            return True
        except queue.Full:
            return False

    def build_prompt(self, episode_logs: List[EpisodeStepLog]) -> str:
        """Constructs structured prompt summarizing episode macroeconomic statistics."""
        if not episode_logs:
            return "No episode logs available."

        first_log = episode_logs[0]
        last_log = episode_logs[-1]
        num_steps = len(episode_logs)

        avg_welfare = sum(log.social_welfare for log in episode_logs) / num_steps
        gini_start = first_log.gini_wealth
        gini_end = last_log.gini_wealth

        prices_by_res: Dict[str, List[float]] = {}
        for log in episode_logs:
            for res_id, clr in log.clearing_results.items():
                if clr.clearing_price is not None:
                    prices_by_res.setdefault(res_id, []).append(clr.clearing_price)

        price_summary = {
            res: {
                "start_price": round(p[0], 2) if p else None,
                "end_price": round(p[-1], 2) if p else None,
                "avg_price": round(sum(p) / max(1, len(p)), 2) if p else None,
            }
            for res, p in prices_by_res.items()
        }

        return f"Episode Overview: {num_steps} steps, Gini: {gini_start:.2f}->{gini_end:.2f}, Welfare: {avg_welfare:.2f}, Prices: {json.dumps(price_summary)}"

    def generate_episode_explanation(self, episode_logs: List[EpisodeStepLog]) -> str:
        """Queries local Ollama Gemma instance or generates deep dynamic econometric evaluation."""
        if not episode_logs:
            return "No episode logs recorded for evaluation."

        # If Ollama is responsive, try querying it
        if self._is_server_online():
            try:
                prompt = self.build_prompt(episode_logs)
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        f"{self.ollama_url}/api/generate",
                        json={
                            "model": self.model_name,
                            "prompt": prompt,
                            "stream": False,
                        },
                    )
                    if response.status_code == 200:
                        data = response.json()
                        res_text = data.get("response", "").strip()
                        if res_text and len(res_text) > 40:
                            return res_text
            except Exception:
                pass

        return self._generate_dynamic_analysis(episode_logs)

    def _generate_dynamic_analysis(self, episode_logs: List[EpisodeStepLog]) -> str:
        """High-fidelity dynamic analytical evaluation tailored to the exact numbers of this specific episode."""
        num_steps = len(episode_logs)
        first_log = episode_logs[0]
        last_log = episode_logs[-1]

        # 1. Macro Summary
        gini_start = first_log.gini_wealth
        gini_end = last_log.gini_wealth
        gini_trend = "widened significantly (rising inequality)" if gini_end > gini_start + 0.1 else ("narrowed (fairer wealth distribution)" if gini_end < gini_start - 0.1 else "remained stable")
        avg_welfare = sum(l.social_welfare for l in episode_logs) / num_steps
        total_finished_goods = sum(sum(l.production_output.values()) for l in episode_logs)

        # 2. Financial Performers (Winner & Struggler)
        cash_changes = {}
        for aid, acc in last_log.agent_accounts.items():
            start_cash = first_log.agent_accounts[aid].cash if aid in first_log.agent_accounts else acc.cash
            cash_changes[aid] = {
                "persona": acc.persona.value,
                "delta": acc.cash - start_cash,
                "final_cash": acc.cash,
            }

        top_winner = max(cash_changes.items(), key=lambda x: x[1]["delta"])
        top_loser = min(cash_changes.items(), key=lambda x: x[1]["delta"])

        # 3. Commodity Price Discovery & Inflation
        prices_by_res: Dict[str, List[float]] = {}
        volumes_by_res: Dict[str, List[float]] = {}
        for log in episode_logs:
            for res_id, clr in log.clearing_results.items():
                if clr.clearing_price is not None:
                    prices_by_res.setdefault(res_id, []).append(clr.clearing_price)
                volumes_by_res.setdefault(res_id, []).append(clr.clearing_volume)

        price_bullet_points = []
        for res, prices in prices_by_res.items():
            if prices:
                p_start = prices[0]
                p_end = prices[-1]
                pct_change = ((p_end - p_start) / max(p_start, 1e-4)) * 100.0
                direction = f"+{pct_change:.1f}% (inflation)" if pct_change > 0 else f"{pct_change:.1f}% (deflation)"
                tot_vol = sum(volumes_by_res.get(res, []))
                price_bullet_points.append(
                    f"- **{res}**: Started at **\\${p_start:.2f}** → Settled at **\\${p_end:.2f}** ({direction}) | Total Cleared Volume: **{tot_vol:.1f} units**"
                )

        # 4. Anomaly Scans
        resource_corner_counts: Dict[str, int] = {}
        resource_peak_hhi: Dict[str, float] = {}
        for log in episode_logs:
            for res, hhi in log.hhi_by_resource.items():
                if hhi > 6500.0:
                    resource_corner_counts[res] = resource_corner_counts.get(res, 0) + 1
                    resource_peak_hhi[res] = max(resource_peak_hhi.get(res, 0.0), hhi)

        # Assemble Clean Markdown Report
        report = [
            f"## 🏛️ Chief Economist Strategic Debrief (Episode #{num_steps} Turns)",
            f"**Economic Health Index**: **{'🔴 Crisis' if gini_end > 0.65 or resource_corner_counts else ('🟡 Strained' if gini_end > 0.45 else '🟢 Healthy & Balanced')}** | **Mean Society Welfare**: `{avg_welfare:.1f} pts` | **Total Manufactured Goods**: `{total_finished_goods:.1f} units`",
            "",
            "### 1. 📊 Macroeconomic Health & Wealth Distribution",
            f"- **Wealth Inequality (Gini)**: Started at **`{gini_start:.2f}`** and ended at **`{gini_end:.2f}`**. Over the course of the simulation, inequality {gini_trend}.",
            f"- **Top Financial Winner**: **`{top_winner[0]}`** ({top_winner[1]['persona']}) gained **+\\${top_winner[1]['delta']:,.2f}** (Final Wallet: **\\${top_winner[1]['final_cash']:,.2f}**).",
            f"- **Top Cash Deficit**: **`{top_loser[0]}`** ({top_loser[1]['persona']}) saw net cash flow of **-\\${abs(top_loser[1]['delta']):,.2f}**.",
            "",
            "### 2. 🏷️ Commodity Price Discovery & Market Dynamics",
            *price_bullet_points,
            "",
            "### 3. 🚨 Market Integrity & Monopolization Analysis",
        ]

        if resource_corner_counts:
            report.append("**⚠️ Hostile Market Concentrations Detected:**")
            for res, count in sorted(resource_corner_counts.items(), key=lambda x: -x[1]):
                peak = resource_peak_hhi.get(res, 0.0)
                report.append(
                    f"- 🛑 **{res} Supply**: Monopolized across **{count} of {num_steps} turns** (Peak Concentration HHI: **{peak:,.0f} / 10,000**). High prices suppressed downstream manufacturing."
                )
        else:
            report.append("- ✅ **Healthy Competition**: No single agent cornered critical supplies beyond safe concentration thresholds.")

        report.extend([
            "",
            "### 4. 💡 Policy & Market Design Recommendations",
            f"- **Anti-Monopoly Safeguards**: {'Enforce strict inventory holding caps on Energy/Materials to break adversary supply bottlenecks.' if resource_corner_counts else 'Maintain open auction clearing mechanisms with current escrow scaling rules.'}",
            f"- **Supply Stimulus**: {'Increase natural resource replenishment rates to reduce raw material inflation and support factory utilization.' if total_finished_goods < 20 else 'Industrial output is healthy; encourage consumer purchasing power via labor wage subsidies.'}",
        ])

        return "\n".join(report)

    def shutdown(self) -> None:
        """Gracefully shuts down worker thread."""
        self._shutdown_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
