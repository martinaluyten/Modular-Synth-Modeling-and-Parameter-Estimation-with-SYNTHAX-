import pandas as pd
import os
import webbrowser
import json

def generate_interactive_report(csv_path, output_folder):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    raw_cols = sorted([c for c in df.columns if 'raw_' in c])
    target_cols = sorted([c for c in df.columns if 'target_' in c])
    
    filters_config = {}
    for col in (raw_cols + target_cols):
        filters_config[col] = {
            "min": round(float(df[col].min()), 2),
            "max": round(float(df[col].max()), 2),
            "label": col.replace('raw_', 'RAW: ').replace('target_', 'NORM: ').replace('_', ' ')
        }

    html_header = f"""
    <html>
    <head>
        <title>SynthAX Advanced Explorer</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; margin: 0; background: #f0f2f5; display: flex; }}
            #sidebar {{ width: 320px; background: #2c3e50; color: white; padding: 20px; overflow-y: auto; height: 100vh; position: sticky; top: 0; font-size: 0.8em; }}
            #main-content {{ flex: 1; padding: 20px; background: #f0f2f5; }}
            
            .tab-bar {{ display: flex; border-bottom: 2px solid #ddd; margin-bottom: 20px; gap: 10px; }}
            .tab {{ padding: 10px 20px; cursor: pointer; background: #e0e0e0; border-radius: 8px 8px 0 0; font-weight: bold; }}
            .tab.active {{ background: #2980b9; color: white; }}
            .tab-content {{ display: none; }}
            .tab-content.active {{ display: block; }}

            .filter-group {{ margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #3e4f5f; }}
            .filter-label {{ display: block; color: #bdc3c7; font-weight: bold; margin-bottom: 4px; }}
            input[type="number"] {{ width: 45%; background: #34495e; border: 1px solid #555; color: white; padding: 3px; border-radius: 3px; }}

            .data-card {{ background: white; border-radius: 10px; margin-bottom: 15px; display: flex; border: 1px solid #ddd; overflow: hidden; }}
            .spec-container {{ width: 400px; padding: 10px; background: #fafafa; border-right: 1px solid #eee; }}
            .spec-img {{ width: 100%; border-radius: 4px; }}
            .params-container {{ flex: 1; padding: 15px; display: flex; gap: 15px; }}
            
            .param-column {{ flex: 1; }}
            .column-header {{ font-size: 0.7em; font-weight: bold; text-transform: uppercase; margin-bottom: 10px; padding: 4px; border-radius: 4px; text-align: center; }}
            .header-raw {{ background: #e3f2fd; color: #1565c0; }}
            .header-norm {{ background: #e8f5e9; color: #2e7d32; }}

            .param-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(85px, 1fr)); gap: 4px; }}
            .param-box {{ padding: 4px; border-radius: 3px; border: 1px solid #eee; }}
            .raw-box {{ background: #f1f8ff; border-left: 2px solid #2196f3; }}
            .norm-box {{ background: #f6fff6; border-left: 2px solid #4caf50; }}
            .p-lab {{ font-size: 0.55em; color: #7f8c8d; text-transform: uppercase; display: block; }}
            .p-val {{ font-size: 0.75em; font-family: monospace; font-weight: bold; }}

            .hist-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            .chart-card {{ background: white; padding: 15px; border-radius: 8px; border: 1px solid #ddd; }}
            tr.hidden {{ display: none; }}

            .time-ruler {{
                position: relative;
                width: 100%;
                height: 18px;
                margin-top: 2px;
                border-top: 1px solid #cbd5e0;
            }}
            .tick {{
                position: absolute;
                border-left: 1px solid #94a3b8;
                height: 5px;
                font-size: 0.65em;
                color: #64748b;
                padding-left: 3px;
                top: 0;
                white-space: nowrap;
            }}
        </style>
    </head>
    <body>
        <div id="sidebar">
            <h2>Filters</h2>
            <div id="filter-controls"></div>
        </div>
        <div id="main-content">
            <div class="tab-bar">
                <div class="tab active" onclick="showTab('gallery')">Gallery View</div>
                <div class="tab" onclick="showTab('analytics')">Distributions</div>
            </div>

            <div id="gallery" class="tab-content active">
                <div style="margin-bottom: 10px; color: #7f8c8d;">Matching: <span id="visible-count">{len(df)}</span> / {len(df)}</div>
                <div id="data-list">
    """

    rows = ""
    # Locate this loop in your script and update the 'rows' construction
    for idx, row in df.round(4).iterrows():
        data_attrs = " ".join([f'data-{c.replace("_","-")}="{row[c]}"' for c in (raw_cols + target_cols)])
        
        # Generate the 0s to 6s ticks
        duration = 6 
        ticks_html = "".join([f'<div class="tick" style="left:{(i/duration)*100}%">{i}s</div>' for i in range(duration + 1)])
        time_grid = f'<div class="time-ruler">{ticks_html}</div>'
        
        def build_grid(cols, box_class):
            html = "<div class='param-grid'>"
            for c in cols:
                label = c.split('_', 1)[1].replace('_', ' ')
                html += f"<div class='param-box {box_class}'><span class='p-lab'>{label}</span><span class='p-val'>{row[c]}</span></div>"
            return html + "</div>"

        rows += f"""
        <div class="data-card data-row" {data_attrs}>
            <div class="spec-container">
                <div style="font-size: 0.8em; font-weight: bold; margin-bottom: 5px;">{row['filename']}</div>
                <img src="{row['spec_path']}" class="spec-img">
                {time_grid}  <!-- Added the ruler here -->
            </div>
            <div class="params-container">
                <div class="param-column">
                    <div class="column-header header-raw">Raw (Original)</div>
                    {build_grid(raw_cols, 'raw-box')}
                </div>
                <div class="param-column">
                    <div class="column-header header-norm">Target (Normalized)</div>
                    {build_grid(target_cols, 'norm-box')}
                </div>
            </div>
        </div>
        """

    analytics_html = """
            <div id="analytics" class="tab-content">
                <h2>Parameter Histograms</h2>
                <div class="hist-grid" id="hist-container"></div>
            </div>
    """

    script = f"""
    <script>
        const filtersConfig = {json.dumps(filters_config)};
        const fullData = {df.to_json(orient='records')};
        
        function showTab(id) {{
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(id).classList.add('active');
            event.currentTarget.classList.add('active');
            if(id === 'analytics') renderCharts();
        }}

        // Filter Logic
        const filterContainer = document.getElementById('filter-controls');
        Object.keys(filtersConfig).forEach(id => {{
            const conf = filtersConfig[id];
            const div = document.createElement('div');
            div.className = 'filter-group';
            div.innerHTML = `
                <span class="filter-label">${{conf.label}}</span>
                <div style="display:flex; justify-content:space-between">
                    <input type="number" step="0.0001" id="min-${{id}}" value="${{conf.min}}" oninput="applyFilters()">
                    <input type="number" step="0.0001" id="max-${{id}}" value="${{conf.max}}" oninput="applyFilters()">
                </div>
            `;
            filterContainer.appendChild(div);
        }});

        function applyFilters() {{
            const rows = document.querySelectorAll('.data-row');
            let count = 0;
            rows.forEach(row => {{
                let visible = true;
                for (let id in filtersConfig) {{
                    const val = parseFloat(row.getAttribute('data-' + id.replace(/_/g, '-')));
                    const min = parseFloat(document.getElementById('min-' + id).value);
                    const max = parseFloat(document.getElementById('max-' + id).value);
                    if (val < min || val > max) {{ visible = false; break; }}
                }}
                row.style.display = visible ? 'flex' : 'none';
                if (visible) count++;
            }});
            document.getElementById('visible-count').innerText = count;
        }}

        function renderCharts() {{
            const container = document.getElementById('hist-container');
            if(container.children.length > 0) return;
            
            Object.keys(filtersConfig).forEach(col => {{
                const vals = fullData.map(r => r[col]);
                const card = document.createElement('div');
                card.className = 'chart-card';
                card.innerHTML = `<h4>${{filtersConfig[col].label}}</h4><canvas id="chart-${{col}}"></canvas>`;
                container.appendChild(card);
                
                const min = Math.min(...vals);
                const max = Math.max(...vals);
                const bins = 15;
                const step = (max - min) / bins;
                const freq = new Array(bins).fill(0);
                
                vals.forEach(v => {{
                    let b = Math.floor((v - min) / step);
                    if(b === bins) b--;
                    freq[b]++;
                }});

                new Chart(document.getElementById('chart-' + col), {{
                    type: 'bar',
                    data: {{
                        labels: freq.map((_, i) => (min + (i * step)).toFixed(2)),
                        datasets: [{{ label: 'Frequency', data: freq, backgroundColor: col.startsWith('raw') ? '#2196f3' : '#4caf50' }}]
                    }},
                    options: {{ plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ beginAtZero: true }} }} }}
                }});
            }});
        }}
    </script>
    """

    full_html = html_header + rows + "</div></div>" + analytics_html + script + "</body></html>"
    
    with open(os.path.join(output_folder, "advanced_inspector.html"), "w") as f:
        f.write(full_html)
    # webbrowser.open('file://' + os.path.realpath(os.path.join(output_folder, "advanced_inspector.html")))

if __name__ == "__main__":
    generate_interactive_report("SynthDataset/metadata.csv", "SynthDataset")