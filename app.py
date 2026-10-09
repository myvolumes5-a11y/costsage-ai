"""
CostSage AI - Modern Software Delivery & Cost Scoping Cockpit
File: app.py

Features:
- Initial state is blank: Prompts user to enter their project details first.
- Clear All Inputs button: Clears form contents and resets calculations.
- Realistic calibrated software effort engine for modern web/mobile stacks.
- AI strategy comparison and contextual chat advisor.
"""

import streamlit as st
import pandas as pd
from src.llm_reasoning import CostSageAdvisor

st.set_page_config(
    page_title="CostSage AI | Project Delivery & Cost Estimator",
    page_icon="💡",
    layout="wide"
)

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.has_submitted = False
    st.session_state.form_title = ""
    st.session_state.form_desc = ""
    st.session_state.form_platform = "Mobile App (iOS & Android)"
    st.session_state.form_stack = ["React Native / Flutter", "Node.js / TypeScript"]
    st.session_state.form_team = 2
    st.session_state.form_budget = 40000
    st.session_state.form_no_ai = True
    st.session_state.form_free_ai = True
    st.session_state.form_prem_ai = True
    st.session_state.form_managed = True
    st.session_state.form_mvp = True
    st.session_state.form_report = "Summary Report (Layman Friendly)"
    st.session_state.form_apikey = ""
    st.session_state.chat_history = []

def reset_all_fields():
    st.session_state.has_submitted = False
    st.session_state.form_title = ""
    st.session_state.form_desc = ""
    st.session_state.form_stack = []
    st.session_state.form_team = 2
    st.session_state.form_budget = 25000
    st.session_state.form_apikey = ""
    st.session_state.chat_history = []
    if "advisor" in st.session_state:
        del st.session_state["advisor"]

with st.sidebar:
    st.title("💡 CostSage Project Intake")
    st.caption("Calibrated for modern frameworks, cloud backends & AI tooling.")

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        submit_clicked = st.button("🚀 Estimate", type="primary", use_container_width=True)
    with btn_col2:
        st.button("🗑️ Clear All", on_click=reset_all_fields, use_container_width=True)

    st.divider()

    st.subheader("1. What are you building?")
    project_title = st.text_input("Project Name / Title", value=st.session_state.form_title, key="form_title", placeholder="E.g., Multiplayer Coin Game")
    project_desc = st.text_area("Describe your project idea & key features", value=st.session_state.form_desc, key="form_desc", placeholder="E.g., 2-player turn-based board game with coin betting and matchmaking.", height=90)

    platform_options = ["Mobile App (iOS & Android)", "Web Application", "Cross-Platform (Web + Mobile)", "Backend API & Microservices"]
    platform = st.selectbox("Target Platform", platform_options, index=platform_options.index(st.session_state.form_platform) if st.session_state.form_platform in platform_options else 0, key="form_platform")

    all_stacks = ["React Native / Flutter", "Node.js / TypeScript", "Python / FastAPI", "Swift / Kotlin", "Unity / C#", "Go"]
    tech_stack = st.multiselect("Primary Technologies / Languages", all_stacks, default=st.session_state.form_stack, key="form_stack")

    st.subheader("2. Team & Capital Available")
    col1, col2 = st.columns(2)
    with col1:
        team_size = st.number_input("Team Members", min_value=1, max_value=30, value=st.session_state.form_team, key="form_team")
    with col2:
        investment_budget = st.number_input("Available Budget ($)", min_value=1000, value=st.session_state.form_budget, step=2500, key="form_budget")

    st.subheader("3. AI Coding Tools to Evaluate")
    eval_no_ai = st.checkbox("Traditional (No AI)", value=st.session_state.form_no_ai, key="form_no_ai")
    eval_free_ai = st.checkbox("Free AI (Autocomplete / Basic)", value=st.session_state.form_free_ai, key="form_free_ai")
    eval_prem_ai = st.checkbox("Premium AI (Cursor / Copilot Pro @ $30/mo)", value=st.session_state.form_prem_ai, key="form_prem_ai")

    st.subheader("4. Architecture & Scope Levers")
    use_managed_backend = st.checkbox("Use Managed Game BaaS (Nakama / Photon / Supabase)", value=st.session_state.form_managed, key="form_managed")
    strict_mvp = st.checkbox("Strict MVP Scope (Core Mechanics Only)", value=st.session_state.form_mvp, key="form_mvp")

    st.subheader("5. Report Format")
    report_type = st.radio("Choose Depth", ["Summary Report (Layman Friendly)", "Detailed Technical Audit (For Tech Leads)"], key="form_report")

    st.subheader("6. Optional LLM Key")
    api_key_input = st.text_input("OpenAI Key (Leave blank for offline mode)", value=st.session_state.form_apikey, type="password", key="form_apikey")

if submit_clicked:
    if not project_title.strip() and not project_desc.strip():
        st.warning("⚠️ Please enter a Project Name or Brief Description before generating an estimate.")
    else:
        st.session_state.has_submitted = True

if not st.session_state.has_submitted:
    st.title("🚀 CostSage AI Software Cost & Delivery Cockpit")
    st.markdown("### Ready to scope your software build?")
    st.write("1. Describe your project idea in the sidebar.\n2. Set your available investment and squad size.\n3. Click **🚀 Estimate** to run the calibrated model.")
    st.info("👈 Enter your project specifications on the left to begin.")
    st.stop()

# ============================================================================
# MODERN CALIBRATION ENGINE
# ============================================================================
desc_lower = (project_title + " " + project_desc).lower()

# Baseline KLOC for modern high-level languages
if any(w in desc_lower for w in ["multiplayer", "game"]):
    base_kloc = 14.0 if use_managed_backend else 26.0
elif any(w in desc_lower for w in ["salon", "booking", "crm", "scheduling"]):
    base_kloc = 6.0
elif any(w in desc_lower for w in ["marketplace", "ecommerce"]):
    base_kloc = 12.0
else:
    base_kloc = 8.0

effective_kloc = base_kloc * (0.75 if strict_mvp else 1.0)

complexity = 2.4
if "Mobile App" in platform:
    complexity += 0.3
if any(w in desc_lower for w in ["multiplayer", "real-time"]):
    complexity += (0.4 if use_managed_backend else 1.0)
if use_managed_backend:
    complexity -= 0.5
complexity = max(1.5, round(complexity, 2))

# Modernized effort formula: High-level SDKs deliver MVPs in 3-8 person-months
nominal_effort = round(1.25 * (effective_kloc ** 0.82) * (complexity / 2.5), 1)
blended_dev_rate = 5500.0

ai_profiles = {}
if eval_no_ai:
    ai_profiles["Traditional (No AI)"] = {"speedup": 1.00, "licensing": 0.0, "pr_penalty": 0.0}
if eval_free_ai:
    ai_profiles["Free AI Tools"] = {"speedup": 1.25, "licensing": 0.0, "pr_penalty": 0.08}
if eval_prem_ai:
    ai_profiles["Premium AI Tools"] = {"speedup": 1.75, "licensing": 30.0, "pr_penalty": 0.04}

if not ai_profiles:
    ai_profiles["Traditional (No AI)"] = {"speedup": 1.00, "licensing": 0.0, "pr_penalty": 0.0}

comparison_data = []
best_scenario = None
min_cost = float("inf")

for name, p in ai_profiles.items():
    net_speed = p["speedup"] / (1.0 + p["pr_penalty"])
    calibrated_effort = round(nominal_effort / net_speed, 1)
    # Realistic minimum threshold: 1.0 month for an MVP
    duration_months = round(max(1.0, calibrated_effort / team_size), 1)
    dev_payroll = calibrated_effort * blended_dev_rate
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
        "QA / Review Overhead": f"+{int(p['pr_penalty']*100)}% audit drag" if p["pr_penalty"] > 0 else "Baseline"
    })

telemetry_data = {
    "project_title": project_title or "Custom Application",
    "project_desc": project_desc,
    "platform": platform,
    "tech_stack": ", ".join(tech_stack),
    "team_size": team_size,
    "investment_budget": investment_budget,
    "total_budget": best_scenario["cost"],
    "duration_months": best_scenario["duration"],
    "effort_pm": best_scenario["effort"],
    "ai_strategy": best_scenario["name"],
    "kloc": effective_kloc,
    "risk_label": "High Financial Risk" if best_scenario["gap"] < 0 else "Budget Adequate"
}

# ============================================================================
# RESULTS VIEWPORT
# ============================================================================
st.title(f"🚀 CostSage Assessment: {project_title or 'Custom Application'}")
st.caption(f"Target: {platform} | Core Stack: {', '.join(tech_stack) if tech_stack else 'Standard Stack'}")

# Clean formatting without broken asterisks
if best_scenario["gap"] >= 0:
    st.success(
        f"✅ **Feasible within your budget!** Your **${investment_budget:,.0f}** budget covers the projected "
        f"**${best_scenario['cost']:,.0f}** build cost with a **${best_scenario['gap']:,.0f} reserve margin** using {best_scenario['name']}."
    )
else:
    st.error(
        f"⚠️ **Budget Shortfall Warning:** Estimated build cost is **${best_scenario['cost']:,.0f}**, while your budget cap is "
        f"**${investment_budget:,.0f}** (Deficit: **-${abs(best_scenario['gap']):,.0f}**). Consider enabling managed game servers or reducing scope."
    )

k1, k2, k3, k4 = st.columns(4)
k1.metric("Estimated Cost", f"${best_scenario['cost']:,.0f}")
k2.metric("Projected Timeline", f"~{best_scenario['duration']} Months")
k3.metric("Your Budget Cap", f"${investment_budget:,.0f}")
k4.metric("Optimal Strategy", best_scenario["name"].split(" (")[0])

st.divider()

if report_type == "Summary Report (Layman Friendly)":
    st.subheader("📋 Executive Summary")
    st.write(f"Building **{project_title}** with **{team_size} developer(s)** is estimated at **{best_scenario['duration']} months** using modern tools:")
    st.table(pd.DataFrame(comparison_data))
else:
    st.subheader("🔬 Detailed Technical & Financial Audit")
    st.table(pd.DataFrame(comparison_data))
    t1, t2, t3 = st.columns(3)
    t1.metric("Effective Sizing", f"{effective_kloc:.1f} KLOC")
    t2.metric("Complexity Factor", f"{complexity:.1f} / 5.0")
    t3.metric("Nominal Effort", f"{nominal_effort:.1f} Dev-Months")

st.divider()

# ============================================================================
# CONVERSATIONAL ADVISOR CHAT
# ============================================================================
st.subheader("💬 Ask CostSage: Scenario Modeling & Trade-Offs")
st.caption("Ask questions like: 'If I increase developers to 5', 'What if I switch to Java?', or 'How can I fit this in a $30,000 budget?'")

session_signature = f"{project_title}_{investment_budget}_{team_size}_{best_scenario['cost']}"
if "last_signature" not in st.session_state or st.session_state.last_signature != session_signature:
    st.session_state.last_signature = session_signature
    st.session_state.advisor = CostSageAdvisor(telemetry_data, api_key=api_key_input)
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": f"Hi! I've evaluated **{project_title or 'your project'}**. With a **${investment_budget:,.0f}** budget and **{team_size} developers**, modern delivery is estimated at **~{best_scenario['duration']} months (${best_scenario['cost']:,.0f})**. What scenario would you like to explore?"
        }
    ]

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("E.g., If I increase developers to 5, how does the timeline change?"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        reply = st.session_state.advisor.respond(prompt)
        st.write(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
