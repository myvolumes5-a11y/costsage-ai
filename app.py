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

# ============================================================================
# SECTION 1: SESSION STATE INITIALIZATION & RESET HANDLERS
# ============================================================================
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.has_submitted = False
    st.session_state.form_title = ""
    st.session_state.form_desc = ""
    st.session_state.form_platform = "Mobile App (iOS & Android)"
    st.session_state.form_stack = ["React Native / Flutter"]
    st.session_state.form_team = 2
    st.session_state.form_budget = 20000
    st.session_state.form_no_ai = True
    st.session_state.form_free_ai = True
    st.session_state.form_prem_ai = True
    st.session_state.form_managed = True
    st.session_state.form_mvp = True
    st.session_state.form_report = "Summary Report (Layman Friendly)"
    st.session_state.form_apikey = ""
    st.session_state.chat_history = []

def reset_all_fields():
    """Wipes all user inputs, results, and conversation history."""
    st.session_state.has_submitted = False
    st.session_state.form_title = ""
    st.session_state.form_desc = ""
    st.session_state.form_platform = "Mobile App (iOS & Android)"
    st.session_state.form_stack = []
    st.session_state.form_team = 1
    st.session_state.form_budget = 10000
    st.session_state.form_no_ai = True
    st.session_state.form_free_ai = True
    st.session_state.form_prem_ai = True
    st.session_state.form_managed = False
    st.session_state.form_mvp = False
    st.session_state.form_apikey = ""
    st.session_state.chat_history = []
    if "advisor" in st.session_state:
        del st.session_state["advisor"]

# ============================================================================
# SECTION 2: SIDEBAR CONTROLS
# ============================================================================
with st.sidebar:
    st.title("💡 CostSage Project Intake")
    st.caption("Realistic, calibrated estimation for modern software development.")

    # Action Buttons: Generate and Clear
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        submit_clicked = st.button("🚀 Estimate", type="primary", use_container_width=True)
    with btn_col2:
        st.button("🗑️ Clear All", on_click=reset_all_fields, use_container_width=True)

    st.divider()

    st.subheader("1. What are you building?")
    project_title = st.text_input(
        "Project Name / Title",
        value=st.session_state.form_title,
        key="form_title",
        placeholder="E.g., Salon Management App"
    )

    project_desc = st.text_area(
        "Describe your project idea & key features",
        value=st.session_state.form_desc,
        key="form_desc",
        placeholder="E.g., Mobile app for booking appointments, staff schedules, and customer payments.",
        height=100
    )

    platform_options = [
        "Mobile App (iOS & Android)",
        "Web Application",
        "Cross-Platform (Web + Mobile)",
        "Backend API & Admin Portal"
    ]
    platform = st.selectbox(
        "Target Platform",
        platform_options,
        index=platform_options.index(st.session_state.form_platform) if st.session_state.form_platform in platform_options else 0,
        key="form_platform"
    )

    all_stacks = [
        "React Native / Flutter",
        "Node.js / TypeScript",
        "Python / FastAPI",
        "Swift / Kotlin",
        "Unity / C#",
        "Go",
        "PHP / Laravel"
    ]
    tech_stack = st.multiselect(
        "Primary Technologies / Languages",
        all_stacks,
        default=st.session_state.form_stack,
        key="form_stack"
    )

    st.subheader("2. Team & Capital Available")
    col1, col2 = st.columns(2)
    with col1:
        team_size = st.number_input("Team Members", min_value=1, max_value=30, value=st.session_state.form_team, key="form_team")
    with col2:
        investment_budget = st.number_input("Available Budget ($)", min_value=1000, value=st.session_state.form_budget, step=2500, key="form_budget")

    st.subheader("3. AI Coding Tools to Evaluate")
    eval_no_ai = st.checkbox("Traditional (No AI)", value=st.session_state.form_no_ai, key="form_no_ai")
    eval_free_ai = st.checkbox("Free AI (Basic Autocomplete / Local)", value=st.session_state.form_free_ai, key="form_free_ai")
    eval_prem_ai = st.checkbox("Premium AI (Cursor / Copilot Pro @ $30/mo)", value=st.session_state.form_prem_ai, key="form_prem_ai")

    st.subheader("4. Architecture & Scope Levers")
    use_managed_backend = st.checkbox("Use Managed Services (Supabase / Firebase / Stripe)", value=st.session_state.form_managed, key="form_managed")
    strict_mvp = st.checkbox("Strict MVP Scope (Core User Loop Only)", value=st.session_state.form_mvp, key="form_mvp")

    st.subheader("5. Report Format")
    report_type = st.radio("Choose Depth", ["Summary Report (Layman Friendly)", "Detailed Technical Audit (For Tech Leads)"], key="form_report")

    st.subheader("6. Optional LLM Key")
    api_key_input = st.text_input("OpenAI Key (Leave blank for offline mode)", value=st.session_state.form_apikey, type="password", key="form_apikey")

if submit_clicked:
    if not project_title.strip() and not project_desc.strip():
        st.warning("⚠️ Please enter a Project Name or Brief Description before generating an estimate.")
    else:
        st.session_state.has_submitted = True

# ============================================================================
# SECTION 3: EMPTY STATE (INITIAL VIEW BEFORE USER INPUT)
# ============================================================================
if not st.session_state.has_submitted:
    st.title("🚀 CostSage AI Software Cost & Delivery Cockpit")
    st.markdown("### Welcome! Ready to estimate your project cost and timeline?")
    st.write(
        "To get an accurate, calibrated breakdown comparing traditional delivery against AI-assisted developer workflows:\n\n"
        "1. **Describe your project idea** in the left sidebar.\n"
        "2. **Specify your team size and available investment capital**.\n"
        "3. Click **🚀 Estimate** at the top of the sidebar to calculate your budget viability."
    )
    st.info("👈 Enter your project specifications on the left to begin.")
    st.stop()

# ============================================================================
# SECTION 4: CALIBRATED ESTIMATION ENGINE
# ============================================================================
desc_lower = (project_title + " " + project_desc).lower()

# Sizing heuristics based on feature scope
if any(w in desc_lower for w in ["salon", "booking", "crm", "scheduling", "management", "restaurant", "store"]):
    base_kloc = 6.5
elif any(w in desc_lower for w in ["multiplayer", "game", "real-time engine", "unreal", "unity"]):
    base_kloc = 24.0
elif any(w in desc_lower for w in ["marketplace", "ecommerce", "fintech", "payment gateway"]):
    base_kloc = 14.0
else:
    base_kloc = 8.0

# Apply scope levers
effective_kloc = base_kloc * (0.75 if strict_mvp else 1.0)

# Architectural complexity index
complexity = 2.2
if "Mobile App" in platform or "Cross-Platform" in platform:
    complexity += 0.4
if any(w in desc_lower for w in ["multiplayer", "socket", "real-time"]):
    complexity += 0.8
if use_managed_backend:
    complexity -= 0.5

complexity = max(1.5, round(complexity, 2))

# Calibrated modern effort (Person-Months)
nominal_effort = round((1.8 * (effective_kloc ** 0.92) * (complexity / 2.5)), 1)
blended_dev_rate = 5500.0

ai_profiles = {}
if eval_no_ai:
    ai_profiles["Traditional (No AI)"] = {"speedup": 1.00, "licensing": 0.0, "pr_penalty": 0.0}
if eval_free_ai:
    ai_profiles["Free AI Tools"] = {"speedup": 1.20, "licensing": 0.0, "pr_penalty": 0.10}
if eval_prem_ai:
    ai_profiles["Premium AI Tools"] = {"speedup": 1.45, "licensing": 30.0, "pr_penalty": 0.05}

if not ai_profiles:
    ai_profiles["Traditional (No AI)"] = {"speedup": 1.00, "licensing": 0.0, "pr_penalty": 0.0}

comparison_data = []
best_scenario = None
min_cost = float("inf")

for name, p in ai_profiles.items():
    net_speed = p["speedup"] / (1.0 + p["pr_penalty"])
    calibrated_effort = round(nominal_effort / net_speed, 1)
    duration_months = round(max(0.8, calibrated_effort / team_size), 1)
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
        "QA / Code Review Drag": f"+{int(p['pr_penalty']*100)}% review effort" if p["pr_penalty"] > 0 else "Normal pace"
    })

telemetry_data = {
    "project_title": project_title or "Custom Software Project",
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
# SECTION 5: MAIN VIEWPORT (ACTIVE ESTIMATE)
# ============================================================================
st.title(f"🚀 CostSage Assessment: {project_title or 'Custom Software Project'}")
st.caption(f"Target: {platform} | Core Stack: {', '.join(tech_stack) if tech_stack else 'Standard Stack'}")

if best_scenario["gap"] >= 0:
    st.success(
        f"✅ **Feasible within your budget!** Your **${investment_budget:,.0f}** investment covers the projected "
        f"**${best_scenario['cost']:,.0f}** build cost with a **${best_scenario['gap']:,.0f} reserve margin** using {best_scenario['name']}."
    )
else:
    st.error(
        f"⚠️ **Budget Shortfall:** Estimated build cost is **${best_scenario['cost']:,.0f}**, while your budget cap is "
        f"**${investment_budget:,.0f}** (Deficit: **-${abs(best_scenario['gap']):,.0f}**). Consider enabling managed backends or trimming non-essential screens."
    )

k1, k2, k3, k4 = st.columns(4)
k1.metric("Estimated Cost", f"${best_scenario['cost']:,.0f}", help="Total development cost based on blended market rates + tools")
k2.metric("Projected Timeline", f"~{best_scenario['duration']} Months", help=f"Duration for a {team_size}-person team")
k3.metric("Your Budget Cap", f"${investment_budget:,.0f}")
k4.metric("Optimal Strategy", best_scenario["name"].split(" (")[0])

st.divider()

# ============================================================================
# SECTION 6: SUMMARY REPORT VS DETAILED AUDIT
# ============================================================================
if report_type == "Summary Report (Layman Friendly)":
    st.subheader("📋 Executive Summary")
    st.write(
        f"Building **{project_title or 'this project'}** as a {platform.lower()} with **{team_size} developer(s)** will take roughly "
        f"**{best_scenario['duration']} months**. Below is how AI-assisted workflows alter your timeline and budget:"
    )
    st.table(pd.DataFrame(comparison_data))
else:
    st.subheader("🔬 Detailed Technical & Financial Audit")
    st.table(pd.DataFrame(comparison_data))

    t1, t2, t3 = st.columns(3)
    t1.metric("Effective Sizing (KLOC)", f"{effective_kloc:.1f} KLOC")
    t2.metric("Architectural Complexity", f"{complexity:.1f} / 5.0")
    t3.metric("Nominal Effort", f"{nominal_effort:.1f} Person-Months")

    st.markdown("#### Domain Cost Drivers")
    if "salon" in desc_lower or "booking" in desc_lower:
        st.write(
            f"- **Appointment & Calendar State:** Multi-staff slot reservation and SMS/Push notifications drive the baseline complexity to **{complexity:.1f}**.\n"
            f"- **Managed Service Offset:** Using services like Supabase, Firebase Auth, or Stripe Billing reduces custom backend engineering hours by ~25%.\n"
            f"- **Developer Concurrency:** A team size of **{team_size}** is appropriate for this MVP scope without communication bottlenecks."
        )
    elif "game" in desc_lower or "multiplayer" in desc_lower:
        st.write(
            f"- **Network State Synchronization:** Real-time client-server synchronization drives baseline complexity to **{complexity:.1f}**.\n"
            f"- **Asset & Logic Verification:** AI tools accelerate boilerplate C# or shaders, but game-loop bugs require ~10% dedicated testing overhead."
        )
    else:
        st.write(
            f"- **Core Architecture:** Standard REST/GraphQL integration with relational database schemas.\n"
            f"- **AI Workflow Leverage:** Routine boilerplate, forms, validation, and unit tests show strong acceleration under modern coding assistants."
        )

st.divider()

# ============================================================================
# SECTION 7: CONVERSATIONAL ADVISOR CHAT
# ============================================================================
st.subheader("💬 Ask CostSage: How to Cut Costs or Optimize Scope?")
st.caption("Ask questions like: 'How can I fit this in my budget?', 'What features should I cut for MVP?', or 'Is this timeline realistic?'")

# Reset advisor and chat history if project parameters change
session_signature = f"{project_title}_{investment_budget}_{team_size}_{best_scenario['cost']}"
if "last_signature" not in st.session_state or st.session_state.last_signature != session_signature:
    st.session_state.last_signature = session_signature
    st.session_state.advisor = CostSageAdvisor(telemetry_data, api_key=api_key_input)
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": f"Hi! I've analyzed **{project_title or 'your project'}**. With a **${investment_budget:,.0f}** budget and **{team_size} team member(s)**, the projected delivery is **~{best_scenario['duration']} months (${best_scenario['cost']:,.0f})**. How can I help optimize your build?"
        }
    ]

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("E.g., How can I build this for under budget?"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        q = prompt.lower()
        if any(w in q for w in ["cut", "save", "budget", "reduce", "cheaper"]):
            if "salon" in desc_lower or "booking" in desc_lower:
                reply = (
                    f"**Ways to trim costs for {project_title or 'your app'} to match your ${investment_budget:,.0f} budget:**\n\n"
                    f"1. **Use Pre-Built Booking SDKs (Saves ~$6,000–$10,000):**\n"
                    f"   Do not write custom recurring appointment slot logic from scratch. Integrate solutions like **Cal.com embed/API or Supabase Calendar templates**.\n\n"
                    f"2. **Cut Custom Payment Hardware (Saves ~$4,000):**\n"
                    f"   For Version 1, avoid building in-person POS hardware integrations. Let users pay via Stripe links or mark as 'Pay at Counter'.\n\n"
                    f"3. **Standardize on React Native or Flutter:**\n"
                    f"   A single codebase for both iOS and Android cuts UI engineering effort in half compared to native Swift and Kotlin."
                )
            else:
                reply = (
                    f"**To reduce the budget for {project_title or 'your project'}:**\n\n"
                    f"1. **Leverage Backend-as-a-Service (BaaS):** Offload auth, file storage, and transactional emails to Firebase or Supabase.\n"
                    f"2. **Freeze Scope to Core User Loop:** Eliminate secondary features like social sharing, dark-mode toggles, or custom reporting for V1.\n"
                    f"3. **Equip Developers with AI Assistants:** Net delivery effort drops ~25% with tools like Cursor or GitHub Copilot."
                )
        elif any(w in q for w in ["feasible", "realistic", "possible"]):
            reply = (
                f"**Feasibility Verdict for {project_title or 'your project'}:**\n\n"
                f"- **Timeline:** Delivering in **~{best_scenario['duration']} months** with {team_size} developer(s) is realistic for an MVP.\n"
                f"- **Financials:** " + (
                    f"Fully funded with a safety buffer of **${best_scenario['gap']:,.0f}**." if best_scenario["gap"] >= 0 else
                    f"Currently over budget by **${abs(best_scenario['gap']):,.0f}**. Enabling 'Managed Services' and keeping scope tight will bridge this gap."
                )
            )
        else:
            reply = st.session_state.advisor.respond(prompt)

        st.write(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
