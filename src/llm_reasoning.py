"""
CostSage AI - Conversational Advisory Engine
File: src/llm_reasoning.py

Dynamically parses team sizing, AI tool differences, and cost-cutting trade-offs.
"""

import os
import re
from typing import Dict, Any, List

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


ADVISORY_SYSTEM_PROMPT = """You are CostSage AI, an expert software delivery director.
You give direct, mathematically accurate answers to founders and engineering leads.

Current Project Specs:
- Project: {project_title}
- Total Work Volume: {effort_pm:.1f} Person-Months
- Current Team: {team_size} developers
- Current Timeline: {duration_months:.1f} Months
- Projected Spend: ${total_budget:,.0f} (Cap: ${investment_budget:,.0f})
- AI Tool Tier: {ai_strategy}

Instructions:
Answer the user's specific question directly. If they ask about changing team size, calculate the new schedule (Effort / new_team) with communication overhead.
"""


class CostSageAdvisor:
    """Manages conversational audits and dynamic scenario modeling."""

    def __init__(self, telemetry: Dict[str, Any], api_key: str = None):
        self.telemetry = telemetry
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.history: List[Dict[str, str]] = []

        system_content = ADVISORY_SYSTEM_PROMPT.format(
            project_title=self.telemetry.get("project_title", "Custom Project"),
            effort_pm=self.telemetry.get("effort_pm", 7.0),
            team_size=self.telemetry.get("team_size", 2),
            duration_months=self.telemetry.get("duration_months", 3.5),
            total_budget=self.telemetry.get("total_budget", 20000.0),
            investment_budget=self.telemetry.get("investment_budget", 25000.0),
            ai_strategy=self.telemetry.get("ai_strategy", "Free AI")
        )
        self.history.append({"role": "system", "content": system_content})

    def respond(self, user_query: str) -> str:
        self.history.append({"role": "user", "content": user_query})

        # Use OpenAI if available
        if OPENAI_AVAILABLE and self.api_key:
            try:
                client = openai.OpenAI(api_key=self.api_key)
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=self.history,
                    temperature=0.3,
                    max_tokens=400
                )
                ans = res.choices[0].message.content
                self.history.append({"role": "assistant", "content": ans})
                return ans
            except Exception:
                pass

        # Robust, dynamic fallback
        reply = self._dynamic_fallback(user_query)
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def _dynamic_fallback(self, query: str) -> str:
        q = query.lower()
        title = self.telemetry.get("project_title", "this app")
        effort = self.telemetry.get("effort_pm", 7.0)
        curr_team = int(self.telemetry.get("team_size", 2))
        curr_months = float(self.telemetry.get("duration_months", 3.5))
        budget_cap = float(self.telemetry.get("investment_budget", 25000))
        cost = float(self.telemetry.get("total_budget", 20000))
        ai_strat = self.telemetry.get("ai_strategy", "Free AI")

        # ---------------------------------------------------------
        # INTENT 1: TEAM SIZING / HEADCOUNT CHANGES
        # ---------------------------------------------------------
        if any(w in q for w in ["person", "persons", "people", "team", "dev", "devs", "developer", "developers", "hire", "add"]):
            # Extract number from query if present (e.g., "3 persons", "team of 4", "add 1")
            nums = re.findall(r"\b\d+\b", q)
            target_team = None

            if "add" in q and nums:
                target_team = curr_team + int(nums[0])
            elif nums:
                target_team = int(nums[0])
            elif "more" in q or "add" in q or "increase" in q:
                target_team = curr_team + 1
            elif "less" in q or "fewer" in q or "reduce" in q:
                target_team = max(1, curr_team - 1)

            if target_team and target_team != curr_team:
                # Brooks' Law overhead factor for larger teams (coordination tax)
                comm_overhead = 1.0 + (0.05 * (target_team - 1))
                new_months = round((effort / target_team) * comm_overhead, 1)
                
                # Check direction
                if target_team > curr_team:
                    diff_months = round(curr_months - new_months, 1)
                    return (
                        f"**Scaling Team from {curr_team} to {target_team} developers for {title}:**\n\n"
                        f"- **New Timeline:** Delivery drops from **{curr_months} months** down to **~{new_months} months** "
                        f"(saving ~{diff_months} months).\n"
                        f"- **Trade-off:** Monthly burn rate increases, but you hit the market significantly faster.\n"
                        f"- **Recommendation:** For a modular scope (e.g., 1 frontend dev on mobile UI, 1 backend dev on APIs/DB, 1 on integrations), a 3-person team works with minimal communication drag."
                    )
                else:
                    return (
                        f"**Reducing Team from {curr_team} to {target_team} developer(s):**\n\n"
                        f"- **New Timeline:** Extends from **{curr_months} months** up to **~{new_months} months**.\n"
                        f"- **Trade-off:** Lower monthly payroll spend, but time-to-market is delayed by ~{round(new_months - curr_months, 1)} months."
                    )

        # ---------------------------------------------------------
        # INTENT 2: WHY DOES FREE AI TAKE THIS LONG?
        # ---------------------------------------------------------
        if any(w in q for w in ["why", "free", "slow", "take this", "long"]):
            return (
                f"**Why Free AI takes ~{curr_months} months for {title}:**\n\n"
                f"1. **Autocomplete vs Full Logic:** Free AI tools (like standard Copilot autocomplete or small local models) only predict the next line or short function. They cannot architect entire features or refactor multi-file codebases.\n"
                f"2. **The Verification Tax (+10%–12% Drag):** Free models frequently hallucinate deprecated syntax or subtle logic bugs. Developers spend ~12% extra time reviewing and testing generated code.\n"
                f"3. **How to cut it to under 2 months:** Upgrading to **Premium AI (Cursor Pro / Claude 3.5 Sonnet)** allows drafting whole screens and API endpoints in minutes, cutting net delivery down to **~2.2 months** with 2 devs."
            )

        # ---------------------------------------------------------
        # INTENT 3: COST CUTTING & BUDGET
        # ---------------------------------------------------------
        if any(w in q for w in ["cut", "save", "budget", "reduce", "cheaper", "deficit", "shortfall"]):
            return (
                f"**3 concrete ways to cut costs on {title} (Target: <${budget_cap:,.0f}):**\n\n"
                f"1. **Use Backend-as-a-Service (Saves ~$5,000–$8,000):** Use Supabase for user auth, Postgres DB, and storage instead of building custom backend plumbing.\n"
                f"2. **Stick to Single Codebase (React Native / Flutter):** Builds iOS and Android simultaneously, avoiding duplicate engineering.\n"
                f"3. **Drop In-Person Hardware for MVP:** Avoid custom POS card-reader hardware. Use web Stripe checkout or cash/counter payments for V1."
            )

        # ---------------------------------------------------------
        # INTENT 4: FEASIBILITY
        # ---------------------------------------------------------
        if any(w in q for w in ["feasible", "realistic", "possible"]):
            status = "fully within budget" if cost <= budget_cap else f"over budget by ${cost - budget_cap:,.0f}"
            return (
                f"**Feasibility Analysis for {title}:**\n\n"
                f"- **Schedule Feasibility:** **{curr_months} months** with {curr_team} developer(s) is realistic for an MVP.\n"
                f"- **Financial Feasibility:** The project is currently **{status}**.\n"
                f"- **Path to 2-Month Delivery:** Moving to a 3-person team or using Premium AI tools reduces this to under 2.5 months."
            )

        # Default catch-all
        return (
            f"For **{title}**, your baseline is **{curr_months} months** with **{curr_team} developers** (${cost:,.0f}).\n\n"
            f"You can ask me:\n"
            f"- *'What happens if we have 3 developers?'*\n"
            f"- *'Why is Free AI slower than Premium AI?'*\n"
            f"- *'How can I get this done in under 2 months?'*"
        )
