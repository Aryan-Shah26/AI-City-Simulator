"""Renders WorldState.history into a single self-contained HTML file (Chart.js via CDN).
No server needed - open the file directly in a browser. Call after a run:

    from dashboard import render_dashboard
    render_dashboard(world.history, "dashboard.html")

or via CLI: `python main.py --ticks 20 --dashboard dashboard.html`
"""
import json

_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>City Sim Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  body {{ font-family: system-ui, sans-serif; background: #0f1115; color: #e6e6e6; margin: 0; padding: 24px; }}
  h1 {{ font-weight: 600; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 24px; }}
  .card {{ background: #1a1d24; border-radius: 12px; padding: 16px; }}
  canvas {{ max-height: 320px; }}
</style></head>
<body>
  <h1>City Sim Dashboard</h1>
  <p>{tick_count} ticks recorded</p>
  <div class="grid">
    <div class="card"><canvas id="cashChart"></canvas></div>
    <div class="card"><canvas id="repChart"></canvas></div>
    <div class="card"><canvas id="marketChart"></canvas></div>
    <div class="card"><canvas id="debtChart"></canvas></div>
  </div>
<script>
const history = {history_json};
const labels = history.map(h => h.tick);
const agentIds = history.length ? Object.keys(history[0].agents) : [];
const colors = ["#5b8def","#e0575b","#4caf7d","#e0a355","#a76ee0","#3ec1c9"];
const seriesFor = (field) => agentIds.map((id, i) => ({{
  label: history[0].agents[id].name,
  data: history.map(h => h.agents[id][field]),
  borderColor: colors[i % colors.length],
  fill: false, tension: 0.2
}}));

new Chart(document.getElementById('cashChart'), {{
  type: 'line', data: {{ labels, datasets: seriesFor('cash') }},
  options: {{ plugins: {{ title: {{ display: true, text: 'Cash over time', color: '#e6e6e6' }} }} }}
}});
new Chart(document.getElementById('repChart'), {{
  type: 'line', data: {{ labels, datasets: seriesFor('reputation') }},
  options: {{ plugins: {{ title: {{ display: true, text: 'Reputation over time', color: '#e6e6e6' }} }} }}
}});
new Chart(document.getElementById('debtChart'), {{
  type: 'line', data: {{ labels, datasets: seriesFor('debt') }},
  options: {{ plugins: {{ title: {{ display: true, text: 'Debt over time', color: '#e6e6e6' }} }} }}
}});
new Chart(document.getElementById('marketChart'), {{
  type: 'line',
  data: {{ labels, datasets: [
    {{ label: 'Wheat price', data: history.map(h => h.market.wheat.price), borderColor: '#e0a355', fill: false }},
    {{ label: 'Bread price', data: history.map(h => h.market.bread.price), borderColor: '#5b8def', fill: false }}
  ] }},
  options: {{ plugins: {{ title: {{ display: true, text: 'Market prices over time', color: '#e6e6e6' }} }} }}
}});
</script>
</body></html>"""


def render_dashboard(history: list, out_path: str):
    """history: WorldState.history (list of per-tick snapshot dicts). Writes a single HTML file."""
    html = _TEMPLATE.format(tick_count=len(history), history_json=json.dumps(history))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)