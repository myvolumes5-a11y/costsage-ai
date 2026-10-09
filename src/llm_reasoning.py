"""
CostSage AI - Conversational Advisory Engine
File: src/llm_reasoning.py

Architecture Overview:
1. System Prompt Construction:
   - Injects quantitative PyTorch telemetry (KLOC, Person-Months, Cost, AI Tier, Risk).
   - Instructs model to deliver plain-English executive summaries first, followed by technical audit steps.
2. Dual-Engine Dispatcher:
   - Primary: OpenAI gpt-4o-mini for dynamic, context-aware trade-off reasoning.
   - Secondary (Offline Fallback): Deterministic rule-based template logic ensuring zero crashes
     even when offline or running without API keys.
3. Conversational State Tracking:
   - Maintains chat message history to support continuous trade-off exploration.
"""

import os
from typing import Dict, Any, List

# Safely import openai without breaking the app if not installed
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ============================================================================
# SECTION 1: SYSTEM PROMPT DEFINITION
# ============================================================================
ADVISORY_SYSTEM_PROMPT = """You are CostSage AI, an expert software delivery director and technical financial auditor.
Your job is to advise founders, product managers, and engineering leads on project feasibility, cost reduction, and delivery risks.

--- CURRENT PROJECT TELEMETRY ---
- Equivalent Sizing: {kloc:.1f} KLOC
- Calibrated Effort: {effort_pm:.1f} Person-Months
- Estimated Budget: ${total_budget:,.0f}
- Projected Timeline: {duration_months:.1f} Months across {team_size} developers
- AI Tooling Strategy: {ai_strategy}
- Risk Level: {risk_label}

--- RESPONSE FORMAT & TONE ---
1. Plain-English Verdict: Start with a 1-2 sentence executive feasibility verdict that a non-technical founder can immediately understand.
2. Cost & Scope Levers: Provide 2-3 specific, high-impact levers (e.g., using Backend-as-a-Service, MVP feature deferrals, AI assistant adoption).
3. Quantify Trade-Offs: State estimated dollar savings and timeline adjustments for each recommendation.
4. Technical Balance: Keep the tone helpful, direct, and pragmatic without dense academic jargon.
"""


# ============================================================================
# SECTION 2: ADVISOR CLASS & DISPATCH LOGIC
# ============================================================================
class CostSageAdvisor:
    """
    Manages conversational audits and feasibility checks using gpt-4o-mini
    with an automatic offline fallback.
    """

    def __init__(self, telemetry: Dict[str, Any], api_key: str = None):
        """
        Args:
            telemetry: Dictionary containing live metrics from the PyTorch model & UI.
            api_key: Optional OpenAI API key passed via UI or environment variable.
        """
        self.telemetry = telemetry
        # Read API key from parameter or environment variable
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.history: List[Dict[str, str]] = []

        # Format system prompt with current project telemetry
        system_content = ADVISORY_SYSTEM_PROMPT.format(
            kloc=self.telemetry.get("kloc", 35.0),
            effort_pm=self.telemetry.get("effort_pm", 30.0),
            total_budget=self.telemetry.get("total_budget", 250000.0),
            duration_months=self.telemetry.get("duration_months", 6.0),
            team_size=self.telemetry.get("team_size", 4),
            ai_strategy=self.telemetry.get("ai_strategy", "Traditional"),
            risk_label=self.telemetry.get("risk_label", "Moderate Risk")
        )
        self.history.append({"role": "system", "content": system_content})

    def respond(self, user_query: str) -> str:
        """
        Processes a user question, queries OpenAI if available,
        or falls back to deterministic rule logic.
        """
        self.history.append({"role": "user", "content": user_query})

        # Branch A: Use OpenAI API if available and key is provided
        if OPENAI_AVAILABLE and self.api_key:
            try:
                client = openai.OpenAI(api_key=self.api_key)
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=self.history,
                    temperature=0.4,
                    max_tokens=450
                )
                answer = completion.choices[0].message.content
                self.history.append({"role": "assistant", "content": answer})
                return answer
            except Exception:
                # If network fails, quota expires, or key is invalid, fall through cleanly
                pass

        # Branch B: Deterministic Offline Fallback
        fallback_answer = self._rule_based_fallback(user_query)
        self.history.append({"role": "assistant", "content": fallback_answer})
        return fallback_answer

    # ========================================================================
    # SECTION 3: DETERMINISTIC RULE-BASED FALLBACK
    # ========================================================================
    def _rule_based_fallback(self, query: str) -> str:
        """
        Generates realistic advice when running offline or without an API key.
        Matches keywords related to budget cuts, feasibility, and team sizing.
        """
        q = query.lower()
        budget = self.telemetry.get("total_budget", 250000.0)
        months = self.telemetry.get("duration_months", 6.0)
        team = self.telemetry.get("team_size", 4)
        risk = self.telemetry.get("risk_label", "Moderate Risk")

        # Intent 1: Cost Reduction / Budget Cuts
        if any(w in q for w in ["cut", "save", "reduce", "budget", "cost", "cheaper"]):
            return (
                f"**Here are 3 ways to reduce your ${budget:,.0f} budget:**\n\n"
                f"1. **Adopt Managed Services / BaaS (Saves ~15%–20% / ~${budget * 0.18:,.0f}):**\n"
                f"   Avoid writing custom authentication, notification, and database plumbing. "
                f"   Using Supabase, Firebase, or Stripe Checkout cuts architectural complexity from 3.5 down to 2.5.\n\n"
                f"2. **Equip Developers with Premium AI Tools (Saves ~20%–25% net payroll):**\n"
                f"   Tools like Cursor Pro or GitHub Copilot ($30/dev/mo) accelerate boilerplate coding by ~45%, "
                f"   easily saving tens of thousands in payroll despite minor pull-request review overhead.\n\n"
                f"3. **Freeze MVP Scope:** Lock features for Version 1 to prevent mid-sprint rework, "
                f"   which eliminates roughly 20% of unplanned lines of code."
            )

        # Intent 2: Timeline & Feasibility Checks
        elif any(w in q for w in ["feasible", "time", "deadline", "schedule", "when", "months"]):
            return (
                f"**Feasibility Verdict:** Shipping in **{months} months** with **{team} engineers** is **achievable**, "
                f"provided you don't face unmocked third-party API delays. Your current risk status is **{risk}**.\n\n"
                f"*To shorten the timeline:* Do not simply add more developers (Brooks' Law creates communication drag). "
                f"Instead, defer secondary integrations to Phase 2."
            )

        # Intent 3: Team Adjustments / Headcount Changes
        elif any(w in q for w in ["team", "headcount", "dev", "fewer", "hire", "person", "engineers"]):
            alt_team = max(1, team - 1)
            alt_months = round(self.telemetry.get("effort_pm", 30.0) / alt_team, 1)
            return (
                f"**Team Sizing Analysis:**\n\n"
                f"* If you reduce your squad from **{team} to {alt_team} developers**, your monthly burn drops, "
                f"  but your delivery schedule extends from **{months} months to ~{alt_months} months**.\n"
                f"* Total development cost remains relatively constant, but market launch is delayed."
            )

        # Default Catch-all Response
        return (
            f"Your project is currently calibrated at **{months} months** and **${budget:,.0f}** ({risk}).\n\n"
            f"You can ask me questions like:\n"
            f"- *'Where can I cut $40,000 from this budget?'*\n"
            f"- *'Is this timeline feasible with only 2 developers?'*\n"
            f"- *'What happens to our schedule if we switch to Premium AI tools?'*"
        )


if __name__ == "__main__":
    # Smoke test for advisor fallback
    sample_telemetry = {
        "kloc": 35.0,
        "effort_pm": 28.5,
        "total_budget": 242250.0,
        "duration_months": 5.7,
        "team_size": 5,
        "ai_strategy": "Premium AI",
        "risk_label": "Low Risk"
    }
    advisor = CostSageAdvisor(sample_telemetry)
    print("Smoke Test Offline Response:")
    print(advisor.respond("Where can I cut budget?"))
