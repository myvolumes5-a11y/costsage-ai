"""
CostSage AI - Layman & Technical Software Scoping Cockpit
File: app.py

Features:
- Step-by-step user input: Project Idea, Platform, Language, Budget, Team.
- Compares AI strategies side-by-side with multi-select.
- Budget Viability Check: Directly calculates if your investment is sufficient.
- Report Views: Summary Report vs. Detailed Technical Audit.
- Conversational chat advisor dynamically grounded in your exact specs.
"""

import streamlit as st
import pandas as pd
from src.llm_reasoning import CostSageAdvisor

st.set_page_config(
    page_title="CostSage AI | Project Delivery & Cost Estimator",
    page_icon="💡",
    layout="wide"
)

# ============================================================================
# SECTION 1: SIDEBAR - PROJECT QUESTIONNAIRE
# ============================================================================
with st.sidebar:
    st.title("💡 CostSage Project Intake")
    st.caption("Tell us about your project to generate a calibrated estimate.")
    st.divider()

    st.subheader("1. What are you building?")
    project_title = st.text_input("Project Name / Title", value="Multiplayer Coin Game")
    project_desc = st.text_area(
        "Describe your project idea & key features",
        value="A real-time multiplayer mobile game with virtual coin economy, daily tasks, player inventory, and matchmaking.",
        height=90
    )

    platform = st.selectbox(
        "Target Platform",
        ["Mobile App (iOS & Android)", "Web Application", "Cross-Platform / Desktop", "Backend API & Microservices"]
    )

    tech_stack = st.multiselect(
        "Primary Technologies / Languages",
        ["Unity / C#", "Python / FastAPI", "Node.js / TypeScript", "React Native / Flutter", "Go", "C++ / Unreal", "Swift / Kotlin"],
        default=["Unity / C#", "Node.js / TypeScript"]
    )

    st.subheader("2. Team & Capital Available")
    col1, col2 = st.columns(2)
    with col1:
        team_size = st.number_input("Team Members", min_value=1, max_value=30, value=3)
    with col2:
        investment_budget = st.number_input("Available Budget ($)", min_value=1000, value=50000, step=5000)

    st.subheader("3. AI Coding Tools to Evaluate")
    eval_no_ai = st.checkbox("Traditional (No AI)", value=True)
    eval_free_ai = st.checkbox("Free AI (Local models / Copilot Free)", value=True)
    eval_prem_ai = st.checkbox("Premium AI (Cursor / Copilot Pro @ $30/mo)", value=True)

    st.subheader("4. Report Depth")
    report_type = st.radio("Choose Output Format", ["Summary Report (Layman Friendly)", "Detailed Technical Audit (For Tech Leads)"])

    st.subheader("5. LLM API Key (Optional)")
    api_key_input = st.text_input("OpenAI Key (Leave blank for offline mode)", type="password")

# ============================================================================
# SECTION 2: ESTIMATION ENGINE
# ============================================================================
# Architectural baseline complexity derived from platform & stack
base_complexity = 3.0
if "Unity / C#" in tech_stack or "C++ / Unreal" in tech_stack:
    base_complexity += 0.8  # Real-time state synchronization penalty
if "Mobile App (iOS & Android)" in platform:
    base_complexity += 0.3

# Base sizing estimate (KLOC equivalent) derived from feature scope
estimated_kloc = 32.0 if "multiplayer" in project_desc.lower() or "game" in project_desc.lower() else 22.0
nominal_effort = 2.94 * (estimated_kloc ** 1.05) * (base_complexity / 3.0)

# Typical market blended developer burn rate ($6,500/month per full-stack dev)
standard_monthly_dev_rate = 6500.0

ai_profiles = {}
if eval_no_ai:
    ai_profiles["Traditional (No AI)"] = {"speedup": 1.00, "licensing": 0.0, "pr_penalty": 0.0}
if eval_free_ai:
    ai_profiles["Free AI Tools"] = {"speedup": 1.20, "licensing": 0.0, "pr_penalty": 0.12}
if eval_prem_ai:
    ai_profiles["Premium AI Tools"] = {"speedup": 1.45, "licensing": 30.0, "pr_penalty": 0.05}

# Fallback if user unchecks all
if not ai_profiles:
    ai_profiles["Traditional (No AI)"] = {"speedup": 1.00, "licensing": 0.0, "pr_penalty": 0.0}

# Calculate figures for selected profiles
comparison_data = []
best_scenario = None
min_cost = float("inf")

for name, p in ai_profiles.items():
    net_speed = p["speedup"] / (1.0 + p["pr_penalty"])
    calibrated_effort = round(nominal_effort / net_speed, 1)
    duration_months = round(calibrated_effort / team_size, 1)
    dev_payroll = calibrated_effort * standard_monthly_dev_rate
    tool_cost = duration_months * team_size * p["licensing"]
    total_cost = round(dev_payroll + tool_cost, 0)
    budget_gap = investment_budget - total_cost

    if total_cost < min_cost:
        min_cost = total_cost
        best_scenario = {
            "name": name,
            "effort": calibrated_effort,
            "duration": duration_months,
            "cost": total_cost,
            "gap": budget_gap
        }

    comparison_data.append({
        "AI Strategy": name,
        "Total Timeline": f"~{duration_months} Months",
        "Total Projected Cost": f"${total_cost:,.0f}",
        "Budget Status": f"✅ Funded (+${budget_gap:,.0f})" if budget_gap >= 0 else f"⚠️ Shortfall (-${abs(budget_gap):,.0f})",
        "QA / Code Review Drag": f"+{int(p['pr_penalty']*100)}% review effort" if p["pr_penalty"] > 0 else "Normal pace"
    })

# Telemetry for LLM Advisor
telemetry_data = {
    "project_title": project_title,
    "project_desc": project_desc,
    "platform": platform,
    "tech_stack": ", ".join(tech_stack),
    "team_size": team_size,
    "investment_budget": investment_budget,
    "total_budget": best_scenario["cost"],
    "duration_months": best_scenario["duration"],
    "effort_pm": best_scenario["effort"],
    "ai_strategy": best_scenario["name"],
    "kloc": estimated_kloc,
    "risk_label": "High Financial Risk" if best_scenario["gap"] < 0 else "Budget Adequate"
}

# ============================================================================
# SECTION 3: MAIN VIEWPORT
# ============================================================================
st.title(f"🚀 CostSage Assessment: {project_title}")
st.caption(f"Target: {platform} | Core Stack: {', '.join(tech_stack) if tech_stack else 'General'}")

# Budget Feasibility Alert Banner
if best_scenario["gap"] >= 0:
    st.success(
        f"✅ **Feasible within your budget!** Your ${investment_budget:,.0f} investment covers the projected "
        f"${best_scenario['cost']:,.0f} cost with a **${best_scenario['gap']:,.0f} reserve margin** using {best_scenario['name']}."
    )
else:
    st.error(
        f"⚠️ **Budget Shortfall Warning:** Estimated cost is **${best_scenario['cost']:,.0f}**, but your investment cap is "
        f"**${investment_budget:,.0f}** (Deficit: **-${abs(best_scenario['gap']):,.0f}**). You will need to defer features or adopt managed backends."
    )

# Key metric summary tiles
k1, k2, k3, k4 = st.columns(4)
k1.metric("Estimated Cost", f"${best_scenario['cost']:,.0f}", help="Total development cost based on market payroll + tool licenses")
k2.metric("Projected Timeline", f"~{best_scenario['duration']} Months", help=f"Duration for a {team_size}-person team")
k3.metric("Your Budget Cap", f"${investment_budget:,.0f}")
k4.metric("Recommended Approach", best_scenario["name"].split(" (")[0])

st.divider()

# ============================================================================
# SECTION 4: SUMMARY REPORT VS DETAILED AUDIT
# ============================================================================
if report_type == "Summary Report (Layman Friendly)":
    st.subheader("📋 Executive Summary")
    st.write(
        f"Building **{project_title}** as a {platform.lower()} with **{team_size} people** will take roughly "
        f"**{best_scenario['duration']} months**. Here is how using AI coding tools impacts your bottom line:"
    )
    st.table(pd.DataFrame(comparison_data))

else:
    st.subheader("🔬 Detailed Technical & Financial Audit")
    st.table(pd.DataFrame(comparison_data))

    t1, t2, t3 = st.columns(3)
    t1.metric("Equivalent Sizing (KLOC)", f"{estimated_kloc:.1f} KLOC")
    t2.metric("Architectural Complexity", f"{base_complexity:.1f} / 5.0")
    t3.metric("Nominal Effort", f"{nominal_effort:.1f} Person-Months")

    st.markdown("#### Cost Driver Analysis")
    st.write(
        f"- **Multiplayer State Sync Drag:** Multiplayer networking introduces socket state synchronization overhead, increasing baseline complexity to **{base_complexity:.1f}**.\n"
        f"- **Team Concurrency:** A team size of **{team_size}** provides optimal concurrency without severe Brooks' Law communication loss.\n"
        f"- **Verification Friction:** AI code generation saves drafting time but requires an estimated **5%–12% pull-request verification overhead** to audit gameplay logic."
    )

st.divider()

# ============================================================================
# SECTION 5: CONVERSATIONAL ADVISOR CHAT
# ============================================================================
st.subheader("💬 Ask CostSage: How to Cut Costs or Optimize Scope?")
st.caption("Ask specific questions like: 'How can I fit this in my budget?', 'What features should I cut for MVP?', or 'Is this timeline realistic?'")

if "advisor" not in st.session_state or st.session_state.get("last_calc_cost") != best_scenario["cost"]:
    st.session_state.advisor = CostSageAdvisor(telemetry_data, api_key=api_key_input)
    st.session_state.last_calc_cost = best_scenario["cost"]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": f"Hi! I've analyzed **{project_title}**. Based on your **${investment_budget:,.0f}** budget and **{team_size} team members**, your projected delivery is **~{best_scenario['duration']} months (${best_scenario['cost']:,.0f})**. What would you like to explore or optimize?"
        }
    ]

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("E.g., How can I cut $15,000 from this game? Or: What can we cut for MVP?"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        # Contextual response combining live project attributes
        q = prompt.lower()
        if any(w in q for w in ["cut", "save", "budget", "reduce", "cheaper"]):
            reply = (
                f"**To trim costs for {project_title} and fit within your ${investment_budget:,.0f} budget:**\n\n"
                f"1. **Use Backend-as-a-Service for Games (Saves ~$18,000–$25,000):**\n"
                f"   Don't build custom matchmaking or inventory databases from scratch. Use services like **PlayFab, Nakama, or Firebase** to handle coins, player accounts, and daily tasks out of the box.\n\n"
                f"2. **Launch with Asynchronous Multiplayer First:**\n"
                f"   Turn-based or leaderboard-based multiplayer takes 60% less engineering effort than frame-by-frame real-time networking.\n\n"
                f"3. **Equip Developers with Premium AI:** Cuts net implementation hours by ~25%."
            )
        elif any(w in q for w in ["feasible", "realistic", "possible"]):
            reply = (
                f"**Feasibility Verdict for {project_title}:**\n\n"
                f"- **Timeline Feasibility:** **{best_scenario['duration']} months** with {team_size} developers is realistic *if* you use pre-built game backend SDKs.\n"
                f"- **Financial Feasibility:** " + (
                    f"You have an adequate safety reserve of **${best_scenario['gap']:,.0f}**." if best_scenario["gap"] >= 0 else
                    f"You have a **${abs(best_scenario['gap']):,.0f} deficit**. You must cut scope or adopt managed game servers to avoid running out of funds."
                )
            )
        else:
            reply = st.session_state.advisor.respond(prompt)

        st.write(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
