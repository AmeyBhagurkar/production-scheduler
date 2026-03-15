import streamlit as st
from huggingface_hub import InferenceClient
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import BytesIO
import time

# ─────────────────────────────────────────────
# 1. Load HF Token
# ─────────────────────────────────────────────
try:
    with open("hf_token.txt", "r") as f:
        HF_TOKEN = f.read().strip()
except FileNotFoundError:
    HF_TOKEN = ""

if not HF_TOKEN or HF_TOKEN == "paste_your_token_here":
    st.error("❌ Token not found. Please open `hf_token.txt` and replace the placeholder with your real HF token.")
    st.stop()

client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    token=HF_TOKEN
)

# ─────────────────────────────────────────────
# 2. Machine Model
# ─────────────────────────────────────────────
class Machine:
    def __init__(self, name, capacity_per_hour=300, status="running"):
        self.name = name
        self.capacity_per_hour = capacity_per_hour
        self.status = status
        self.total_assigned = 0
        self.gallons_produced = 0
        self.is_standby_activated = False  # tracks if this was originally standby

    def assign_production(self, gallons):
        self.total_assigned = gallons
        self.gallons_produced = 0

    def remaining(self):
        return max(0, self.total_assigned - self.gallons_produced)

    def produce_for(self, hours):
        if self.status != "running":
            return 0
        can_produce = hours * self.capacity_per_hour
        actual = min(can_produce, self.remaining())
        self.gallons_produced += actual
        return actual


def initialize_machines():
    machines = [
        Machine("Machine 1"),
        Machine("Machine 2"),
        Machine("Machine 3", status="standby"),
    ]
    machines[2].is_standby_activated = False
    return machines

# ─────────────────────────────────────────────
# 3. Scheduler & Breakdown Logic
# ─────────────────────────────────────────────
def redistribute_to_running(machines, extra_gallons):
    """Evenly distribute extra_gallons across all running machines."""
    running = [m for m in machines if m.status == "running"]
    if not running:
        return machines, "🔔 No running machines to redistribute to."
    per_machine = extra_gallons / len(running)
    for m in running:
        m.total_assigned += per_machine
    return machines, None


def handle_breakdown(machines, machine_name):
    """Mark machine as down, activate standby if available, redistribute remaining gallons."""
    found = False
    broken_remaining = 0

    for machine in machines:
        if machine.name == machine_name:
            found = True
            if machine.status == "down":
                return machines, f"⚠️ {machine_name} is already down."
            broken_remaining = machine.remaining()
            machine.total_assigned = machine.gallons_produced  # zero out remaining
            machine.status = "down"

    if not found:
        return machines, f"🔔 Machine '{machine_name}' not found."

    # Activate standby machine if available
    standby = next((m for m in machines if m.status == "standby"), None)
    if standby:
        standby.status = "running"
        standby.is_standby_activated = True

    # Redistribute broken machine's remaining gallons to all running machines
    machines, err = redistribute_to_running(machines, broken_remaining)
    alert = err if err else None
    return machines, alert


def handle_repair(machines, machine_name):
    """Repair a broken machine and rebalance: deactivate standby, redistribute evenly."""
    found = False
    for machine in machines:
        if machine.name == machine_name:
            found = True
            if machine.status != "down":
                return machines, f"⚠️ {machine_name} is not currently down."
            machine.status = "running"
            machine.is_standby_activated = False

    if not found:
        return machines, f"🔔 Machine '{machine_name}' not found."

    # Deactivate standby machine that was covering — put it back to standby
    standby_active = next((m for m in machines if m.is_standby_activated and m.status == "running"), None)
    if standby_active:
        standby_active.status = "standby"
        standby_active.is_standby_activated = False
        # Redistribute standby's remaining gallons back to running machines
        extra = standby_active.remaining()
        standby_active.total_assigned = standby_active.gallons_produced
        if extra > 0:
            machines, _ = redistribute_to_running(machines, extra)

    # Now rebalance all remaining gallons equally across all running machines
    total_remaining = sum(m.remaining() for m in machines if m.status == "running")
    running = [m for m in machines if m.status == "running"]
    if running and total_remaining > 0:
        per_machine = total_remaining / len(running)
        for m in running:
            m.total_assigned = m.gallons_produced + per_machine

    return machines, None

# ─────────────────────────────────────────────
# 4. Plant Controller
# ─────────────────────────────────────────────
class PlantController:
    def __init__(self):
        self.machines = []
        self.production_started = False
        self.production_ended = False
        self.total_demand = 0
        self.daily_limit = 3000
        self.start_time = None
        self.end_time = None
        self.last_synced_time = None
        self.last_snapshot_time = None  # FIX #4: for 5-min snapshots
        self.history = []
        self.alerts = []               # FIX #2: unified alert list
        self.snapshot_log = []

    def _log(self, event, details=""):
        self.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Event": event,
            "Details": details
        })

    def _alert(self, msg):
        """FIX #2: All alerts go to same list, visible in Alerts tab."""
        entry = {"Time": datetime.now().strftime("%H:%M:%S"), "Alert": msg}
        self.alerts.append(entry)
        return msg

    def elapsed_hours_since_start(self):
        if not self.start_time:
            return 0
        end = self.end_time if self.production_ended else datetime.now()
        return (end - self.start_time).total_seconds() / 3600

    def sync_production(self):
        """FIX #3: Sync anytime — called by auto-refresh too."""
        if not self.production_started or self.production_ended:
            return
        now = datetime.now()
        hours_since_last_sync = (now - self.last_synced_time).total_seconds() / 3600
        if hours_since_last_sync <= 0:
            return
        for machine in self.machines:
            machine.produce_for(hours_since_last_sync)
        self.last_synced_time = now

        total_remaining = sum(m.remaining() for m in self.machines)
        total_produced = self.total_demand - total_remaining

        # FIX #4: snapshot every 5 minutes
        mins_since_snap = (now - self.last_snapshot_time).total_seconds() / 60 if self.last_snapshot_time else 999
        if mins_since_snap >= 5:
            self.snapshot_log.append({
                "Timestamp": now.strftime("%H:%M:%S"),
                "Elapsed (hrs)": round(self.elapsed_hours_since_start(), 4),
                "Produced (gal)": round(total_produced, 1),
                "Remaining (gal)": round(total_remaining, 1),
                "Daily Limit (gal)": self.daily_limit
            })
            self.last_snapshot_time = now

        if total_remaining <= 0 and not self.production_ended:
            self.production_ended = True
            self.end_time = now
            self._alert("✅ All production complete!")

    def set_daily_limit(self, new_limit):
        """FIX #8: Only updates the limit, never restarts production."""
        if new_limit <= self.total_demand and self.production_started:
            return f"⚠️ New limit ({new_limit} gal) is less than current demand ({self.total_demand} gal). Please set a higher limit."
        old_limit = self.daily_limit
        self.daily_limit = new_limit
        msg = f"Daily limit updated {old_limit} → {new_limit} gallons."
        self._log("LIMIT_CHANGE", msg)
        return f"✅ {msg} Production continues unaffected."

    def start_production(self, demand):
        """FIX #7: Does NOT restart if already running — warns instead."""
        if self.production_started and not self.production_ended:
            return (
                f"⚠️ Production is already running ({self.total_demand} gal). "
                f"Use 'add X gallons' to increase demand, or reset the plant to start fresh."
            )
        if demand > self.daily_limit:
            msg = f"🔔 Demand {demand} gal exceeds daily limit {self.daily_limit} gal!"
            self._alert(msg)
            return msg

        self.machines = initialize_machines()
        running = [m for m in self.machines if m.status == "running"]
        per_machine = demand / len(running)
        for m in running:
            m.assign_production(per_machine)

        self.total_demand = demand
        self.production_started = True
        self.production_ended = False
        self.start_time = datetime.now()
        self.last_synced_time = datetime.now()
        self.last_snapshot_time = datetime.now()
        msg = f"Production started at {self.start_time.strftime('%H:%M:%S')} for {demand} gallons. (Limit: {self.daily_limit} gal)"
        self._log("START", msg)
        return f"✅ {msg}"

    def add_production(self, extra_gallons):
        if not self.production_started:
            return "⚠️ Production has not started yet."
        if self.production_ended:
            return "⚠️ Production has already ended. Reset the plant to start a new run."

        self.sync_production()
        new_total = self.total_demand + extra_gallons

        if new_total > self.daily_limit:
            over = new_total - self.daily_limit
            msg = (
                f"🔔 Adding {extra_gallons} gal would bring total to {new_total} gal, "
                f"exceeding daily limit by {over} gal.\n\n"
                f"💡 Increase the daily limit in the sidebar first, then try again."
            )
            self._alert(f"Blocked ADD_PRODUCTION: {extra_gallons} gal would exceed limit.")
            return msg

        self.machines, err = redistribute_to_running(self.machines, extra_gallons)
        if err:
            self._alert(err)
            return err

        self.total_demand = new_total
        msg = f"{extra_gallons} gal added at {datetime.now().strftime('%H:%M:%S')}. New total: {new_total} gal."
        self._log("ADD_PRODUCTION", msg)
        return f"✅ {msg}"

    def breakdown(self, machine_name):
        if not self.production_started:
            return "⚠️ Production has not started."
        if self.production_ended:
            return "⚠️ Production has already ended."
        self.sync_production()
        self.machines, alert = handle_breakdown(self.machines, machine_name)
        if alert:
            self._alert(alert)  # FIX #2: goes to alerts tab
        msg = f"{machine_name} broke down at {datetime.now().strftime('%H:%M:%S')}. Gallons redistributed."
        self._log("BREAKDOWN", msg)
        result = f"🔧 {msg}"
        if alert:
            result += f"\n\n{alert}"
        return result

    def repair(self, machine_name):
        """FIX #6: Repair machine, deactivate standby, rebalance."""
        if not self.production_started:
            return "⚠️ Production has not started."
        self.sync_production()
        self.machines, alert = handle_repair(self.machines, machine_name)
        if alert:
            self._alert(alert)
            return f"⚠️ {alert}"
        msg = f"{machine_name} repaired at {datetime.now().strftime('%H:%M:%S')}. Standby deactivated. Production rebalanced."
        self._log("REPAIR", msg)
        return f"🔧 ✅ {msg}"

    def end_production(self):
        """FIX #7: Explicit end production command."""
        if not self.production_started:
            return "⚠️ Production has not started."
        if self.production_ended:
            return "⚠️ Production has already ended."
        self.sync_production()
        self.production_ended = True
        self.end_time = datetime.now()
        total_produced = sum(m.gallons_produced for m in self.machines)
        msg = f"Production ended at {self.end_time.strftime('%H:%M:%S')}. Total produced: {total_produced:.1f} gal."
        self._log("END", msg)
        self._alert(f"🛑 Production manually ended. Total produced: {total_produced:.1f} gal.")
        return f"🛑 {msg}"

    def show_status(self):
        if not self.production_started:
            return "⚠️ Production has not started."
        self.sync_production()

        now = datetime.now()
        elapsed = (self.end_time if self.production_ended else now) - self.start_time
        elapsed_str = str(elapsed).split(".")[0]

        lines = []
        lines.append(
            f"🕐 **Started:** {self.start_time.strftime('%H:%M:%S')} | "
            f"**Now:** {now.strftime('%H:%M:%S')} | "
            f"**Elapsed:** {elapsed_str}"
        )
        lines.append(f"📋 **Daily Limit:** {self.daily_limit} gal | **Total Demand:** {self.total_demand} gal")
        lines.append("")

        total_remaining = 0
        for m in self.machines:
            rem = m.remaining()
            total_remaining += rem
            icon = {"running": "🟢", "standby": "🟡", "down": "🔴"}.get(m.status, "⚪")
            lines.append(f"{icon} **{m.name}** | {m.status} | Produced: {m.gallons_produced:.1f} gal | Remaining: {rem:.1f} gal")

        total_produced = self.total_demand - total_remaining
        lines.append(f"\n📦 **Total Produced:** {total_produced:.1f} gal | **Remaining:** {total_remaining:.1f} gal")

        if self.production_ended:
            lines.append("🛑 **Production has ended.**")
        else:
            running = [m for m in self.machines if m.status == "running"]
            if running and total_remaining > 0:
                total_capacity = sum(m.capacity_per_hour for m in running)
                hours_left = total_remaining / total_capacity
                eta = now + timedelta(hours=hours_left)
                lines.append(f"⏳ **ETA:** {eta.strftime('%H:%M:%S')} ({hours_left*60:.0f} mins from now)")
            elif total_remaining <= 0:
                lines.append("✅ **Production complete!**")

        return "\n\n".join(lines)

# ─────────────────────────────────────────────
# 5. LLM Interpreter
# ─────────────────────────────────────────────
def local_keyword_fallback(command):
    """
    Keyword-based fallback interpreter — runs without any API call.
    Catches short / ambiguous inputs the LLM might miss.
    Returns an action string or None if no match found.
    """
    cmd = command.lower().strip()
    import re

    # SHOW_STATUS keywords
    status_keywords = ["status", "update", "how's it going", "how is it going",
                       "progress", "report", "check", "current", "overview",
                       "how much", "how many", "whats going on", "what's going on",
                       "what is the status", "what's the status", "give me status",
                       "show status", "show me status", "give update", "any update",
                       "how is production", "how's production", "production status"]
    if any(k in cmd for k in status_keywords):
        return "SHOW_STATUS"

    # END_PRODUCTION keywords
    end_keywords = ["stop", "end", "finish", "halt", "done", "complete", "shut down", "shutdown"]
    if any(k in cmd for k in end_keywords):
        return "END_PRODUCTION"

    # ADD_PRODUCTION — catch "add X", "extra X", "more X", "increase by X"
    add_patterns = [
        r"add\s+(\d+)", r"(\d+)\s+more", r"extra\s+(\d+)", r"(\d+)\s+extra",
        r"increase\s+by\s+(\d+)", r"boost\s+by\s+(\d+)", r"(\d+)\s+additional"
    ]
    for pattern in add_patterns:
        match = re.search(pattern, cmd)
        if match:
            return f"ADD_PRODUCTION {match.group(1)}"

    # START_PRODUCTION — catch "start X", "produce X", "make X", "run X gallons"
    start_patterns = [
        r"start\s+(\d+)", r"produce\s+(\d+)", r"make\s+(\d+)",
        r"run\s+(\d+)", r"need\s+(\d+)", r"want\s+(\d+)\s+gallon"
    ]
    for pattern in start_patterns:
        match = re.search(pattern, cmd)
        if match:
            return f"START_PRODUCTION {match.group(1)}"

    # BREAKDOWN — catch "machine X broke / failed / down / crash"
    breakdown_patterns = [
        r"machine\s+(\d+)\s+(?:broke|failed|down|crash|stop|fault|error|not working)",
        r"(?:breakdown|broke|failed|fault)\s+(?:machine\s+)?(\d+)"
    ]
    for pattern in breakdown_patterns:
        match = re.search(pattern, cmd)
        if match:
            return f"BREAKDOWN {match.group(1)}"

    # REPAIR — catch "machine X fixed / repaired / back / working / is repaired"
    repair_patterns = [
        r"machine\s+(\d+)\s+(?:is\s+)?(?:fixed|repaired|back|working|restored|online|ok|okay|ready)",
        r"(?:fix|repair|restore|repaired|fixed)\s+(?:machine\s+)?(\d+)",
        r"(\d+)\s+(?:is\s+)?(?:fixed|repaired|restored|working|back online)",
        r"machine\s+(\d+).*(?:repair|fix|back|working|restored)"
    ]
    for pattern in repair_patterns:
        match = re.search(pattern, cmd)
        if match:
            return f"REPAIR {match.group(1)}"

    return None  # no match — let LLM handle it


def llm_interpret(command, chat_history):
    system_prompt = """You are an AI supervisor for a water bottling plant.

Machines: Machine 1, Machine 2, Machine 3

Allowed actions:
1. START_PRODUCTION <gallons>   — only when production has NOT started yet
2. ADD_PRODUCTION <gallons>     — only the EXTRA gallons to add to ongoing production
3. BREAKDOWN <machine number>
4. REPAIR <machine number>
5. END_PRODUCTION
6. SHOW_STATUS

CRITICAL RULES:
- If the user says "increase by X", "add X more", "need X extra", "increase production to X" while production is already running, use ADD_PRODUCTION with ONLY the extra amount (X), never the new total.
- ADD_PRODUCTION <gallons> means the NUMBER OF EXTRA GALLONS TO ADD, never the new total.
- START_PRODUCTION is ONLY for when production has not started at all.
- ONE command only. No explanation. No extra text.

Examples:
"start production for 2000 gallons" → START_PRODUCTION 2000
"we need 2000 gallons today" → START_PRODUCTION 2000
"run 1500 gallons" → START_PRODUCTION 1500
"add 500 more gallons" → ADD_PRODUCTION 500
"increase production by 500" → ADD_PRODUCTION 500
"we need 500 extra gallons" → ADD_PRODUCTION 500
"increase to 2500 gallons" (already running 2000) → ADD_PRODUCTION 500
"boost production by 300" → ADD_PRODUCTION 300
"I need 300 more gallons" → ADD_PRODUCTION 300
"machine 2 failed" → BREAKDOWN 2
"machine 2 is not working" → BREAKDOWN 2
"machine 2 is fixed" → REPAIR 2
"fix machine 2" → REPAIR 2
"end production" → END_PRODUCTION
"stop production" → END_PRODUCTION
"stop" → END_PRODUCTION
"show status" → SHOW_STATUS
"status" → SHOW_STATUS
"update" → SHOW_STATUS
"how much is done" → SHOW_STATUS
"progress" → SHOW_STATUS
"check" → SHOW_STATUS
"""
    messages = [{"role": "system", "content": system_prompt}]
    for entry in chat_history[-6:]:
        messages.append({"role": entry["role"], "content": entry["content"]})
    messages.append({"role": "user", "content": command})

    response = client.chat_completion(messages=messages, max_tokens=20)
    return response.choices[0].message["content"].strip()


def llm_chat(controller, message, chat_history):
    # Try local keyword fallback first — fast, no API call needed
    action = local_keyword_fallback(message)
    if action:
        debug_info = f"🤖 Local Match: `{action}`"
    else:
        # Fall back to LLM if local keywords didn't catch it
        action = llm_interpret(message, chat_history)
        debug_info = f"🤖 LLM Decision: `{action}`"

    if action.startswith("START_PRODUCTION"):
        try:
            gallons = int(action.split()[1])
            # Safety net: if production already running, treat as ADD_PRODUCTION for the difference
            if controller.production_started and not controller.production_ended:
                if gallons > controller.total_demand:
                    extra = gallons - controller.total_demand
                    debug_info += f" ⚠️ *(auto-corrected to ADD_PRODUCTION {extra})*"
                    return debug_info, controller.add_production(extra)
                else:
                    return debug_info, controller.start_production(gallons)
            return debug_info, controller.start_production(gallons)
        except (IndexError, ValueError):
            return debug_info, "❌ Could not parse gallons."

    if action.startswith("ADD_PRODUCTION"):
        try:
            gallons = int(action.split()[1])
            return debug_info, controller.add_production(gallons)
        except (IndexError, ValueError):
            return debug_info, "❌ Could not parse gallons to add."

    if action.startswith("BREAKDOWN"):
        try:
            machine_num = action.split()[1]
            return debug_info, controller.breakdown("Machine " + machine_num)
        except IndexError:
            return debug_info, "❌ Could not parse machine number."

    if action.startswith("REPAIR"):
        try:
            machine_num = action.split()[1]
            return debug_info, controller.repair("Machine " + machine_num)
        except IndexError:
            return debug_info, "❌ Could not parse machine number."

    if action in ("END_PRODUCTION", "STOP_PRODUCTION", "FINISH_PRODUCTION"):
        return debug_info, controller.end_production()

    if action == "SHOW_STATUS":
        return debug_info, controller.show_status()

    return debug_info, "❓ Not recognized. Try: 'start production for 2000 gallons', 'add 500 gallons', 'show status', 'machine 1 broke down', 'machine 1 repaired', 'end production'."

# ─────────────────────────────────────────────
# 6. Export
# ─────────────────────────────────────────────
def export_to_excel(controller):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_hist = pd.DataFrame(controller.history) if controller.history else pd.DataFrame({"Info": ["No history"]})
        df_hist.to_excel(writer, sheet_name="Production History", index=False)

        df_snaps = pd.DataFrame(controller.snapshot_log) if controller.snapshot_log else pd.DataFrame({"Info": ["No snapshots"]})
        df_snaps.to_excel(writer, sheet_name="Timeline Snapshots", index=False)

        df_alerts = pd.DataFrame(controller.alerts) if controller.alerts else pd.DataFrame({"Info": ["No alerts"]})
        df_alerts.to_excel(writer, sheet_name="Alerts", index=False)

        machine_data = [{
            "Machine": m.name, "Status": m.status,
            "Total Assigned (gal)": round(m.total_assigned, 1),
            "Produced (gal)": round(m.gallons_produced, 1),
            "Remaining (gal)": round(m.remaining(), 1),
            "Capacity/Hour": m.capacity_per_hour
        } for m in controller.machines] if controller.machines else [{"Info": "No machines"}]
        pd.DataFrame(machine_data).to_excel(writer, sheet_name="Machine Status", index=False)

    output.seek(0)
    return output

# ─────────────────────────────────────────────
# 7. Charts
# ─────────────────────────────────────────────
def render_machine_cards(controller):
    controller.sync_production()
    machines = controller.machines
    cols = st.columns(len(machines))
    icons = {"running": "🟢", "standby": "🟡", "down": "🔴"}
    for col, m in zip(cols, machines):
        with col:
            st.metric(
                label=f"{icons.get(m.status,'⚪')} {m.name}",
                value=f"{m.remaining():.0f} gal left",
                delta=f"{m.gallons_produced:.0f} gal produced"
            )

def render_timeline_chart(controller):
    if not controller.snapshot_log:
        st.info("Snapshots recorded every 5 minutes will appear here.")
        return
    df = pd.DataFrame(controller.snapshot_log)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Elapsed (hrs)"], y=df["Produced (gal)"],
        mode="lines+markers", name="Produced", line=dict(color="#4F8BF9", width=3)))
    fig.add_trace(go.Scatter(x=df["Elapsed (hrs)"], y=df["Remaining (gal)"],
        mode="lines+markers", name="Remaining", line=dict(color="#F94F4F", width=3, dash="dash")))
    fig.add_hline(y=controller.daily_limit, line_dash="dot", line_color="#F9C74F",
        annotation_text=f"Daily Limit ({controller.daily_limit} gal)", annotation_position="top left")
    fig.update_layout(title="Production Timeline (5-min snapshots)",
        xaxis_title="Hours Elapsed", yaxis_title="Gallons",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"), legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)

def render_machine_pie(machines):
    if not machines:
        return
    labels = [m.name for m in machines]
    values = [max(m.remaining(), 0.01) for m in machines]
    colors = ["#4F8BF9" if m.status == "running" else "#F9C74F" if m.status == "standby" else "#F94F4F" for m in machines]
    fig = go.Figure(go.Pie(labels=labels, values=values, marker=dict(colors=colors), hole=0.4))
    fig.update_layout(title="Remaining Gallons by Machine",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# 8. Streamlit UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="🏭 Production Scheduler", layout="wide")
st.title("🏭 Water Bottling Plant — AI Supervisor")
st.caption("Powered by Hugging Face · Meta-Llama-3-8B-Instruct | Real-Time Production Tracking")

if "controller" not in st.session_state:
    st.session_state.controller = PlantController()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm_history" not in st.session_state:
    st.session_state.llm_history = []

# Always read controller fresh from session_state on every rerun
controller = st.session_state.controller

# Sync production on every page load so all tabs see latest state
if controller.production_started and not controller.production_ended:
    controller.sync_production()

# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ Controls")

    # FIX #1: Live clock — always visible, updates on each rerun
    now_str = datetime.now().strftime("%H:%M:%S")
    st.markdown(f"🕐 **Now:** `{now_str}`")

    if controller.production_started:
        elapsed = (controller.end_time if controller.production_ended else datetime.now()) - controller.start_time
        elapsed_str = str(elapsed).split(".")[0]
        st.markdown(f"▶️ **Started:** `{controller.start_time.strftime('%H:%M:%S')}`")
        st.markdown(f"⏱️ **Elapsed:** `{elapsed_str}`")
        st.divider()
        running = [m.name for m in controller.machines if m.status == "running"]
        standby = [m.name for m in controller.machines if m.status == "standby"]
        down = [m.name for m in controller.machines if m.status == "down"]
        st.markdown(f"🟢 **Running:** {', '.join(running) if running else 'None'}")
        if standby:
            st.markdown(f"🟡 **Standby:** {', '.join(standby)}")
        if down:
            st.markdown(f"🔴 **Down:** {', '.join(down)}")
        st.markdown(f"🎯 **Total demand:** {controller.total_demand} gal")
        if controller.production_ended:
            st.error("🛑 Production Ended")

    st.divider()

    # FIX #8: Daily limit adjuster — only updates limit, never restarts
    st.markdown("### 📏 Daily Limit")
    new_limit = st.number_input(
        "Set daily production limit (gal)",
        min_value=100, max_value=100000,
        value=controller.daily_limit, step=500,
        help="Adjust anytime — only changes the limit, never restarts production."
    )
    if new_limit != controller.daily_limit:
        if st.button("✅ Apply New Limit"):
            msg = controller.set_daily_limit(new_limit)
            st.success(msg)
            st.rerun()

    st.divider()
    if st.button("🔄 Reset Plant"):
        st.session_state.controller = PlantController()
        st.session_state.messages = []
        st.session_state.llm_history = []
        st.rerun()

    st.divider()
    st.markdown("**💬 Sample commands:**")
    st.markdown("- `start production for 2000 gallons`")
    st.markdown("- `add 500 more gallons`")
    st.markdown("- `machine 1 broke down`")
    st.markdown("- `machine 1 is repaired`")
    st.markdown("- `end production`")
    st.markdown("- `what is the status?`")

# FIX #2: Alert badge count on tab label
alert_count = len(controller.alerts)
alert_label = f"🔔 Alerts ({alert_count})" if alert_count > 0 else "🔔 Alerts"

tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Dashboard", alert_label, "📝 History & Export"])

# ── Tab 1: Chat ──
with tab1:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Type a command...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.llm_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                debug_info, response = llm_chat(controller, user_input, st.session_state.llm_history)
            st.markdown(debug_info)
            st.markdown(response)
            full_reply = f"{debug_info}\n\n{response}"

        st.session_state.messages.append({"role": "assistant", "content": full_reply})
        st.session_state.llm_history.append({"role": "assistant", "content": response})
        st.rerun()

# ── Tab 2: Dashboard ──
with tab2:
    st.subheader("📊 Live Machine Status")

    if controller.machines:
        render_machine_cards(controller)
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            render_timeline_chart(controller)
        with col2:
            render_machine_pie(controller.machines)

        if controller.production_started:
            controller.sync_production()
            total_remaining = sum(m.remaining() for m in controller.machines)
            total_produced = controller.total_demand - total_remaining
            st.divider()
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("📏 Daily Limit", f"{controller.daily_limit} gal")
            c2.metric("🎯 Total Demand", f"{controller.total_demand:.0f} gal")
            c3.metric("✅ Produced", f"{total_produced:.0f} gal")
            c4.metric("⏳ Remaining", f"{total_remaining:.0f} gal")
            elapsed_hrs = controller.elapsed_hours_since_start()
            c5.metric("⏱️ Elapsed", f"{elapsed_hrs*60:.1f} mins")
    else:
        st.info("No machines active yet. Start production from the Chat tab.")

    # FIX #3: Auto-refresh dashboard every 10 seconds
    if controller.production_started and not controller.production_ended:
        st.caption("🔄 Dashboard auto-refreshes every 10 seconds")
        time.sleep(10)
        st.rerun()

# ── Tab 3: Alerts ──
with tab3:
    st.subheader("🔔 Alerts & Warnings")
    # Always read fresh from session_state
    alerts = list(st.session_state.controller.alerts)
    if alerts:
        st.info(f"Total alerts received: **{len(alerts)}**")
        for alert in reversed(alerts):
            st.warning(f"**{alert['Time']}** — {alert['Alert']}")
    else:
        st.success("✅ No alerts. Everything is running smoothly.")

# ── Tab 4: History & Export ──
with tab4:
    st.subheader("📝 Production History")

    # Always read fresh from session_state
    _c = st.session_state.controller
    history = list(_c.history)
    snapshots = list(_c.snapshot_log)

    st.caption(f"Total events logged: **{len(history)}** · Auto-updates every 10 seconds")

    if history:
        st.dataframe(pd.DataFrame(history), use_container_width=True)
    else:
        st.info("No history yet. Start issuing commands from the Chat tab.")
    
    # Auto-refresh every 10 seconds just like the dashboard
    if _c.production_started and not _c.production_ended:
        time.sleep(10)
        st.rerun()

    if snapshots:
        st.subheader("📈 Timeline Snapshots (every 5 mins)")
        st.dataframe(pd.DataFrame(snapshots), use_container_width=True)

    if history or snapshots:
        excel_data = export_to_excel(controller)
        st.download_button(
            label="⬇️ Export Full Report to Excel",
            data=excel_data,
            file_name=f"production_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
