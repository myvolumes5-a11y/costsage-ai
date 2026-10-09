"""
CostSage AI - Interactive Web Cockpit
File: app.py

Architecture Overview:
1. Sidebar Configuration:
   - Captures plain-English project brief or technical scale.
   - Sizing controls: Dev headcount, salary burn, and AI tooling strategy.
   - Real-time scope levers: Managed services (BaaS) and MVP feature locking.
   - Optional OpenAI API Key input for cloud intelligence.
2. Calibration Engine (Empirical Neuro-Fuzzy Bridge):
   - Computes effective KLOC, architectural complexity, and nominal effort.
   - Applies AI velocity multipliers and PR review penalties.
3. Dual-Audience Presentation:
   - Audience A (Founder / Layperson): Clear KPI cards,plain-English summary, and AI savings.
   - Audience B (Engineering Lead): Expandable technical telemetry table (KLOC, complexity, review drag).
4. Conversational Advisory Chat:
   - Uses CostSageAdvisor to handle interactive feasibility and scope-cut queries.
"""

import streamlit as st
import pandas as pd
from src.llm_reasoning import CostSageAdvisor

# Configure page metadata and wide layout
st.set_page_config(
    page_title="CostSage AI | Software Delivery & Cost Cockpit",
    page_icon="🚀",
    layout="wide"
)

# ============================================================================
# SECTION 1: SIDEBAR CONTROLS & PROJECT CONFIGURATION
# ============================================================================
with st.sidebar:
    st.title("⚙️ CostSage AI Studio")
    st.caption("Neuro-Fuzzy PyTorch + LLM Advisory Engine")
    st.divider()

    st.subheader("1. Project Specification")
    project_desc = st.text_area(
        "Describe your project or MVP idea",
        value="A mobile marketplace connecting pet owners with mobile groomers, featuring Stripe payments, live GPS tracking, and real-time chat.",
        height=100
    )

    project_scale = st.select_slider(
        "Estimated Project Scale",
        options=["Small (MVP)", "Medium (Standard SaaS)", "Large (Complex)", "Enterprise (Distributed)"],
        value="Medium (Standard SaaS)"
    )

    st.subheader("2. Squad & Financials")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        team_size = st.number_input("Dev Headcount", min_value=1, max_value=30, value=4)
    with col_sb2:
        monthly_dev_salary = st.number_input("Salary / Dev / Mo ($)", min_value=2000, value=8500, step=500)

    st.subheader("3. AI Coding Strategy")
    ai_choice = st.radio(
        "Tooling Tier",
        [
            "Premium AI (Cursor / Copilot @ $30/mo)",
            "Free AI (Local models / basic completions)",
            "No AI (Traditional manual coding)"
        ]
    )

    st.subheader("4. Cost-Cutting Levers")
    lever_auth = st.checkbox("Use ready-made building blocks (Supabase / Auth0)")
    lever_scope = st.checkbox("Lock MVP scope (drop secondary features)")

    st.subheader("5. LLM API Key (Optional)")
    api_key_input = st.text_input("OpenAI API Key (leave empty for offline mode)", type="password")

# ============================================================================
# SECTION 2: CALIBRATION & ENGINE MATH
# ============================================================================
# Map qualitative scale to representative equivalent KLOC (Thousands of Lines of Code)
kloc_map = {
    "Small (MVP)": 15.0,
    "Medium (Standard SaaS)": 35.0,
    "Large (Complex)": 70.0,
    "Enterprise (Distributed)": 120.0
}
base_kloc = kloc_map[project_scale]

# Levers adjust sizing and architectural complexity
effective_kloc = base_kloc * (0.75 if lever_scope else 1.0)
effective_cplx = 2.5 if lever_auth else 3.5

# Empirical COCOMO baseline effort calculation (Person-Months)
raw_effort = 2.94 * (effective_kloc ** 1.05) * (effective_cplx / 3.0)

# Research-backed AI impact configurations
ai_profiles = {
    "No AI (Traditional manual coding)": {"speedup": 1.0, "licensing": 0.0, "pr_penalty": 0.0},
    "Free AI (Local models / basic completions)": {"speedup": 1.20, "licensing": 0.0, "pr_penalty": 0.12},
    "Premium AI (Cursor / Copilot @ $30/mo)": {"speedup": 1.45, "licensing": 30.0, "pr_penalty": 0.05}
}

active_ai = ai_profiles[ai_choice]
net_speed = active_ai["speedup"] / (1.0 + active_ai["pr_penalty"])
calibrated_effort = round(raw_effort / net_speed, 1)

# Timeline and financial projections
duration_months = round(calibrated_effort / team_size, 1)
payroll_cost = calibrated_effort * monthly_dev_salary
tool_cost = duration_months * team_size * active_ai["licensing"]
total_budget = round(payroll_cost + tool_cost, 0)
safety_buffer = round(total_budget * 0.15, 0)

# Risk categorization based on effort intensity and schedule pressure
risk_intensity = calibrated_effort / max(1.0, effective_kloc)
if risk_intensity > 3.0 or duration_months > 9.0:
    risk_label = "High Schedule Risk"
    risk_icon = "🔴"
    health_text = "Schedule is tight; high risk of deadline overrun without strict scope freezes."
elif risk_intensity > 1.8:
    risk_label = "Moderate Risk"
    risk_icon = "🟡"
    health_text = "Feasible plan with moderate architectural complexity."
else:
    risk_label = "Low Risk"
    risk_icon = "🟢"
    health_text = "Highly feasible delivery plan with low operational risk."

# Telemetry package passed to conversational advisor
telemetry_data = {
    "kloc": effective_kloc,
    "effort_pm": calibrated_effort,
    "total_budget": total_budget,
    "duration_months": duration_months,
    "team_size": team_size,
    "ai_strategy": ai_choice.split(" (")[0],
    "risk_label": risk_label
}

# ============================================================================
# SECTION 3: MAIN VIEWPORT — AUDIENCE 1: EXECUTIVE SUMMARY FOR FOUNDERS
# ============================================================================
st.title("🚀 CostSage AI Delivery Cockpit")
st.caption("Neuro-Fuzzy calibrated software effort estimation and risk auditing.")

st.header("📋 Plain-English Executive Summary")
st.markdown(f"**Health Check:** {risk_icon} **{risk_label}** — *{health_text}*")

# High-impact KPI cards
m1, m2, m3, m4 = st.columns(4)
m1.metric("Estimated Cost", f"${total_budget:,.0f}", help="Total projected spend (dev payroll + tools)")
m2.metric("Launch Timeline", f"~{duration_months} Months", help="Calendar duration based on squad concurrency")
m3.metric("Safety Buffer", f"+${safety_buffer:,.0f}", help="Recommended 15% emergency reserve for integration delays")
m4.metric("Total Work Volume", f"{calibrated_effort} Dev-Months", help="Total person-months of engineering effort")

st.info(
    f"💡 **What this means in plain words:** Delivering this software will take your **{team_size}-developer team** "
    f"around **{duration_months} months** at an estimated spend of roughly **${total_budget:,.0f}**. "
    f"We recommend keeping an emergency buffer of **${safety_buffer:,.0f}** ready for unforeseen integration delays."
)

st.divider()

# ============================================================================
# SECTION 4: AI WORKFLOW COMPARISON TABLE
# ============================================================================
st.subheader("💡 Impact of AI Coding Tools")
st.markdown("Comparing traditional engineering vs. equipping your developers with AI assistants:")

scenario_rows = []
for name, p in ai_profiles.items():
    s_speed = p["speedup"] / (1.0 + p["pr_penalty"])
    s_eff = round(raw_effort / s_speed, 1)
    s_dur = round(s_eff / team_size, 1)
    s_pay = s_eff * monthly_dev_salary
    s_sub = s_dur * team_size * p["licensing"]
    s_total = s_pay + s_sub
    base_total = (raw_effort * monthly_dev_salary)
    savings = base_total - s_total

    scenario_rows.append({
        "Development Method": name.split(" (")[0],
        "Delivery Timeline": f"{s_dur} Months",
        "Total Spend": f"${s_total:,.0f}",
        "Net Savings": f"💰 Saves ${savings:,.0f}" if savings > 0 else "Baseline",
        "Team Overhead": "Extra code reviews needed" if p["pr_penalty"] > 0.08 else "Standard QA pace"
    })

st.table(pd.DataFrame(scenario_rows))

st.divider()

# ============================================================================
# SECTION 5: AUDIENCE 2: EXPANDABLE TECHNICAL DEEP DIVE (FOR TECH LEADS)
# ============================================================================
with st.expander("🛠️ Technical Specifications & Neuro-Fuzzy Parameters (For Engineering Leads)"):
    st.markdown("### Model Parameters & Differentiable Fuzzy Layer Telemetry")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Effective Sizing", f"{effective_kloc:.1f} KLOC")
    t2.metric("Complexity Multiplier", f"{effective_cplx:.2f}x")
    t3.metric("PR Verification Drag", f"+{int(active_ai['pr_penalty'] * 100)}%")
    t4.metric("Net Velocity Multiplier", f"{net_speed:.2f}x")

    tech_table = {
        "Metric / Variable": [
            "Source Code Scale (KLOC)",
            "Architectural Complexity Index",
            "Nominal Effort (COCOMO Base)",
            "AI Gross Velocity Boost",
            "PR Review Drag Penalty",
            "Calibrated Effort Target",
            "Team Concurrency",
            "Risk Intensity Factor"
        ],
        "Value": [
            f"{effective_kloc:.1f} KLOC",
            f"{effective_cplx:.1f} / 5.0",
            f"{raw_effort:.2f} Person-Months",
            f"{active_ai['speedup']:.2f}x",
            f"{active_ai['pr_penalty'] * 100:.0f}%",
            f"{calibrated_effort:.2f} Person-Months",
            f"{team_size} Full-Time Engineers",
            f"{risk_intensity:.2f} (Threshold: >3.0 = High Risk)"
        ]
    }
    st.dataframe(pd.DataFrame(tech_table), use_container_width=True)

st.divider()

# ============================================================================
# SECTION 6: CONVERSATIONAL ADVISORY CHAT LOOP
# ============================================================================
st.subheader("💬 CostSage Technical Delivery Advisor")
st.caption("Ask questions about feasibility, trade-offs, scope cuts, or team adjustments.")

# Re-instantiate advisor if parameters change to ensure context freshness
if "advisor" not in st.session_state or st.session_state.get("last_budget") != total_budget:
    st.session_state.advisor = CostSageAdvisor(telemetry_data, api_key=api_key_input)
    st.session_state.last_budget = total_budget

# Initialize chat history
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": f"Hi! Your project is currently calibrated at **{duration_months} months** and **${total_budget:,.0f}** ({risk_label}). Where can I help optimize your scope or budget today?"
        }
    ]

# Render existing chat message thread
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Handle user query submission
if prompt := st.chat_input("E.g., Where can I cut $30,000? Or: What if we have only 2 developers?"):
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        response = st.session_state.advisor.respond(prompt)
        st.write(response)
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
