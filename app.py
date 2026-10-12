"""
Streamlit Web Application: Neuro-Symbolic LLM Interface
A logic-constrained reasoning dashboard fusing Neural Extraction with Deterministic Verification.
"""

import streamlit as st
import json
import networkx as nx
import streamlit.components.v1 as components
import core_engine as ce

# Page configuration
st.set_page_config(
    page_title="Neuro-Symbolic LLM Reasoning Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Sleek Dark Modern Aesthetic)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }

    /* Header styling */
    .header-badge {
        display: inline-flex;
        align-items: center;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(168, 85, 247, 0.15));
        border: 1px solid rgba(139, 92, 246, 0.35);
        color: #a78bfa;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
    }

    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        line-height: 1.5;
        margin-bottom: 1.5rem;
    }

    /* Metric Card */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(51, 65, 85, 0.8);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.6);
        transform: translateY(-2px);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
    }
    .metric-lbl {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        font-weight: 600;
    }

    /* Status Banners */
    .banner-pass {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.2) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 14px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px -5px rgba(16, 185, 129, 0.2);
    }
    .banner-fail {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(185, 28, 28, 0.2) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 14px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px -5px rgba(239, 68, 68, 0.2);
    }

    /* Status badge pill */
    .badge-pass {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-fail {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-missing {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    /* Explanation Box */
    .explanation-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 1.2rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        line-height: 1.6;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# Lazy-loaded LLM model caching
@st.cache_resource(show_spinner="Loading Qwen 2.5-1.5B Instruct model weights (this may take a minute)...")
def get_neural_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    if device == "cuda":
        model = model.to(device)
    model.eval()
    return model, tokenizer, device


# Helper to generate Interactive HTML Vis.js graph
def generate_vis_network_html(graph: nx.DiGraph) -> str:
    nodes = []
    edges = []

    color_map = {
        "entity": {"bg": "#3b82f6", "border": "#60a5fa", "shape": "box"},
        "fact": {"bg": "#06b6d4", "border": "#22d3ee", "shape": "ellipse"},
        "value": {"bg": "#f59e0b", "border": "#fbbf24", "shape": "box"},
        "rule": {"bg": "#8b5cf6", "border": "#a78bfa", "shape": "diamond"},
        "condition": {"bg": "#f97316", "border": "#fb923c", "shape": "ellipse"},
        "conclusion": {"bg": "#ec4899", "border": "#f472b6", "shape": "box"}
    }

    for node_id, attrs in graph.nodes(data=True):
        ntype = attrs.get("node_type", "fact")
        cfg = color_map.get(ntype, {"bg": "#94a3b8", "border": "#cbd5e1", "shape": "ellipse"})
        label = attrs.get("label", str(node_id))

        nodes.append({
            "id": node_id,
            "label": label,
            "title": f"Type: {ntype.upper()}<br>ID: {node_id}",
            "color": {
                "background": cfg["bg"],
                "border": cfg["border"],
                "highlight": {"background": "#ffffff", "border": cfg["border"]}
            },
            "font": {"color": "#ffffff", "size": 13, "face": "Inter"},
            "shape": cfg["shape"],
            "margin": 10,
            "borderWidth": 2
        })

    for source, target, attrs in graph.edges(data=True):
        rel = attrs.get("relation", "")
        edges.append({
            "from": source,
            "to": target,
            "label": rel,
            "arrows": "to",
            "color": {"color": "#475569", "highlight": "#818cf8"},
            "font": {"color": "#94a3b8", "size": 10, "align": "middle", "strokeWidth": 0},
            "smooth": {"type": "cubicBezier", "roundness": 0.3}
        })

    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
                background: #090d16;
                overflow: hidden;
            }}
            #mynetwork {{
                width: 100%;
                height: 520px;
                border: 1px solid #1e293b;
                border-radius: 12px;
            }}
        </style>
    </head>
    <body>
        <div id="mynetwork"></div>
        <script type="text/javascript">
            var nodes = new vis.DataSet({nodes_json});
            var edges = new vis.DataSet({edges_json});

            var container = document.getElementById('mynetwork');
            var data = {{ nodes: nodes, edges: edges }};
            var options = {{
                physics: {{
                    enabled: true,
                    barnesHut: {{
                        gravitationalConstant: -3500,
                        centralGravity: 0.3,
                        springLength: 95,
                        springConstant: 0.04
                    }},
                    stabilization: {{ iterations: 150 }}
                }},
                interaction: {{
                    hover: true,
                    zoomView: true,
                    dragView: true,
                    navigationButtons: false
                }}
            }};
            var network = new vis.Network(container, data, options);
        </script>
    </body>
    </html>
    """


# Helper to generate Graphviz DOT representation
def generate_dot_chart(graph: nx.DiGraph) -> str:
    lines = ["digraph KnowledgeGraph {", "  graph [rankdir=LR, bgcolor=transparent];", "  node [fontname=\"Inter\", fontsize=10, style=filled, fontcolor=white];", "  edge [fontname=\"Inter\", fontsize=8, fontcolor=\"#94a3b8\", color=\"#475569\"];"]
    
    color_map = {
        "entity": "#2563eb",
        "fact": "#0891b2",
        "value": "#d97706",
        "rule": "#7c3aed",
        "condition": "#ea580c",
        "conclusion": "#db2777"
    }

    for node_id, attrs in graph.nodes(data=True):
        ntype = attrs.get("node_type", "fact")
        color = color_map.get(ntype, "#64748b")
        label = attrs.get("label", str(node_id)).replace('"', '\\"')
        lines.append(f'  "{node_id}" [label="{label}", fillcolor="{color}"];')

    for s, t, attrs in graph.edges(data=True):
        rel = attrs.get("relation", "")
        lines.append(f'  "{s}" -> "{t}" [label="{rel}"];')

    lines.append("}")
    return "\n".join(lines)


# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:
    st.markdown('<div class="header-badge">Control Center</div>', unsafe_allow_html=True)
    st.subheader("⚙️ System Configuration")

    engine_choice = st.radio(
        "Reasoning Engine:",
        [
            "⚡ Fast Mode (Benchmark / Heuristic)",
            "🧠 Live Neural Model (Qwen 2.5 1.5B)"
        ],
        index=0,
        help="Fast Mode runs instant deterministic logic on benchmarks. Live Neural Model executes the local Qwen LLM for knowledge extraction."
    )

    st.markdown("---")
    st.subheader("📚 Benchmark Scenarios")

    preset_names = list(ce.BENCHMARK_SCENARIOS.keys()) + ["Custom Scenario"]
    selected_preset = st.selectbox(
        "Choose a test scenario:",
        preset_names,
        index=0
    )

    if selected_preset in ce.BENCHMARK_SCENARIOS:
        st.info(ce.BENCHMARK_SCENARIOS[selected_preset]["description"])

    st.markdown("---")
    st.markdown("### 🔍 Pipeline Architecture")
    with st.expander("How Neuro-Symbolic Logic Works"):
        st.markdown("""
        1. **Neural Phase**: Qwen 2.5 extracts unstructured facts and rules into structured schema without guessing decisions.
        2. **Knowledge Graph**: Ingests facts and rules into an Entity-Attribute-Value network.
        3. **Deterministic Reasoning**: Relational engine evaluates each condition (`>=`, `<=`, `==`).
        4. **Audit Trail**: Generates a 100% verified, hallucination-free decision breakdown.
        """)

    st.caption("Neuro-Symbolic AI Reasoning Engine • v1.0")


# ==========================================================
# MAIN INTERFACE
# ==========================================================
st.markdown('<div class="header-badge">Deterministic AI Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Neuro-Symbolic Reasoning Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Fusing deep language comprehension with 100% auditable, '
    'deterministic mathematical logic. Eliminates reasoning hallucinations on boundary constraints.</div>',
    unsafe_allow_html=True
)

# Initialize Session State
if "scenario_text" not in st.session_state:
    st.session_state["scenario_text"] = ce.BENCHMARK_SCENARIOS["Instant Loan Approval (Pass)"]["text"]

if selected_preset in ce.BENCHMARK_SCENARIOS:
    if st.session_state.get("last_preset") != selected_preset:
        st.session_state["scenario_text"] = ce.BENCHMARK_SCENARIOS[selected_preset]["text"]
        st.session_state["last_preset"] = selected_preset
elif selected_preset == "Custom Scenario" and st.session_state.get("last_preset") != selected_preset:
    st.session_state["last_preset"] = selected_preset

col_input, col_meta = st.columns([3, 1])

with col_input:
    input_text = st.text_area(
        "Scenario & Rules Text (Input)",
        value=st.session_state["scenario_text"],
        height=210,
        placeholder="Enter unstructured text describing the rules and facts...",
        help="Describe domain rules, conditions, and customer facts in natural language."
    )
    st.session_state["scenario_text"] = input_text

with col_meta:
    st.markdown("#### ⚡ Quick Actions")
    run_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)
    reset_btn = st.button("🔄 Reload Current Preset", use_container_width=True)
    if reset_btn and selected_preset in ce.BENCHMARK_SCENARIOS:
        st.session_state["scenario_text"] = ce.BENCHMARK_SCENARIOS[selected_preset]["text"]
        st.rerun()

    st.markdown("#### 🛡️ Guarantees")
    st.markdown("""
    - **Zero Hallucination** in reasoning
    - **Deterministic** operator evaluation
    - **Granular** root cause audit
    """)

# Pipeline Execution
if run_btn or "last_result" in st.session_state:
    if run_btn:
        model = None
        tokenizer = None

        if "Live Neural Model" in engine_choice:
            try:
                model, tokenizer, device = get_neural_model()
                st.toast(f"Running Qwen 2.5 on {device.upper()}...", icon="🧠")
            except Exception as e:
                st.error(f"Error loading neural model: {e}. Falling back to deterministic mode.")

        with st.spinner("Executing Neuro-Symbolic reasoning pipeline..."):
            try:
                result = ce.run_pipeline(
                    user_input=input_text,
                    model=model,
                    tokenizer=tokenizer
                )
                st.session_state["last_result"] = result
            except Exception as e:
                st.error(f"Pipeline error: {str(e)}")
                st.stop()

    result = st.session_state.get("last_result")
    if not result:
        st.stop()

    st.markdown("---")

    # ==========================================================
    # RESULTS DASHBOARD
    # ==========================================================
    status = result["overall_status"]
    is_pass = (status == "SATISFIED")

    # Top Status Banner
    if is_pass:
        st.markdown(f"""
        <div class="banner-pass">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span class="badge-pass">STATUS: {status}</span>
                    <h2 style="margin: 8px 0 4px 0; color: #10b981; font-weight: 800;">Decision: {result['decision']}</h2>
                    <p style="margin: 0; color: #a7f3d0; font-size: 0.95rem;">{result['explanation']}</p>
                </div>
                <div style="font-size: 3.5rem;">✅</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="banner-fail">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span class="badge-fail">STATUS: {status}</span>
                    <h2 style="margin: 8px 0 4px 0; color: #ef4444; font-weight: 800;">Decision: {result['decision']}</h2>
                    <p style="margin: 0; color: #fecaca; font-size: 0.95rem;">{result['explanation'].splitlines()[0]}</p>
                </div>
                <div style="font-size: 3.5rem;">❌</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Metric Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    total_facts = len(result["knowledge"].get("facts", []))
    total_rules = len(result["knowledge"].get("rules", []))
    total_conditions = sum(len(r.get("conditions", [])) for r in result["knowledge"].get("rules", []))
    total_passed = sum(1 for r in result["reasoning"] for c in r["conditions"] if c["status"] == "PASS")

    with m_col1:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Extracted Facts</div><div class="metric-val">{total_facts}</div></div>', unsafe_allow_html=True)
    with m_col2:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Rules Defined</div><div class="metric-val">{total_rules}</div></div>', unsafe_allow_html=True)
    with m_col3:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Total Conditions</div><div class="metric-val">{total_conditions}</div></div>', unsafe_allow_html=True)
    with m_col4:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Conditions Passed</div><div class="metric-val" style="color: #34d399;">{total_passed} / {total_conditions}</div></div>', unsafe_allow_html=True)

    st.write("")

    # Interactive Tabs
    tab_graph, tab_reasoning, tab_knowledge, tab_audit = st.tabs([
        "🕸️ Interactive Knowledge Graph",
        "⚖️ Symbolic Verification Matrix",
        "🧠 Extracted Schema (JSON)",
        "📋 Full Audit Trail"
    ])

    # TAB 1: KNOWLEDGE GRAPH
    with tab_graph:
        st.subheader("Entity-Attribute-Value (EAV) Knowledge Graph")
        st.caption("Drag nodes to interact. Zoom in/out to inspect relationships between entities, facts, rules, and conditions.")

        # Legend
        st.markdown("""
        <div style="display: flex; gap: 12px; margin-bottom: 12px; flex-wrap: wrap;">
            <span style="background: #3b82f6; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;">Entity</span>
            <span style="background: #06b6d4; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;">Fact Attribute</span>
            <span style="background: #f59e0b; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;">Fact Value</span>
            <span style="background: #8b5cf6; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;">Rule</span>
            <span style="background: #f97316; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;">Condition</span>
            <span style="background: #ec4899; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;">Conclusion</span>
        </div>
        """, unsafe_allow_html=True)

        kg = result["knowledge_graph"]
        vis_html = generate_vis_network_html(kg)
        components.html(vis_html, height=540)

        with st.expander("View Graphviz DOT Format"):
            st.graphviz_chart(generate_dot_chart(kg))

    # TAB 2: SYMBOLIC VERIFICATION MATRIX
    with tab_reasoning:
        st.subheader("Condition-Level Deterministic Verification")
        st.markdown("Each logical constraint is strictly verified against factual nodes in the Knowledge Graph.")

        for rule_res in result["reasoning"]:
            rule_id = rule_res["rule_id"]
            rule_stat = rule_res["status"]
            stat_color = "#34d399" if rule_stat == "SATISFIED" else "#f87171"

            st.markdown(f"#### Rule {rule_id}: Target Conclusion `{rule_res['conclusion'].get('attribute')} = {rule_res['conclusion'].get('value')}` — <span style='color:{stat_color}; font-weight:700;'>{rule_stat}</span>", unsafe_allow_html=True)

            table_rows = []
            for cond in rule_res["conditions"]:
                c_stat = cond["status"]
                badge = f'<span class="badge-pass">PASS</span>' if c_stat == "PASS" else (f'<span class="badge-fail">FAIL</span>' if c_stat == "FAIL" else '<span class="badge-missing">MISSING</span>')
                
                table_rows.append({
                    "Attribute": cond["attribute"],
                    "Operator": f"`{cond['operator']}`",
                    "Expected Value": str(cond["expected"]),
                    "Actual Ingested Fact": str(cond["actual"]) if cond["actual"] is not None else "(None)",
                    "Evaluated Entity": cond.get("entity") or "N/A",
                    "Condition Status": badge
                })

            # Render HTML Table for styling
            html_table = """
            <table style="width:100%; border-collapse: collapse; margin-bottom: 1.5rem;">
                <thead>
                    <tr style="border-bottom: 1px solid #334155; text-align: left; color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">
                        <th style="padding: 10px;">Attribute</th>
                        <th style="padding: 10px;">Operator</th>
                        <th style="padding: 10px;">Expected Threshold</th>
                        <th style="padding: 10px;">Actual Fact in KG</th>
                        <th style="padding: 10px;">Entity</th>
                        <th style="padding: 10px;">Status</th>
                    </tr>
                </thead>
                <tbody>
            """
            for row in table_rows:
                html_table += f"""
                    <tr style="border-bottom: 1px solid #1e293b; font-size: 0.9rem;">
                        <td style="padding: 10px; font-weight:600; color:#f8fafc;">{row['Attribute']}</td>
                        <td style="padding: 10px; color:#cbd5e1;"><code>{row['Operator']}</code></td>
                        <td style="padding: 10px; color:#cbd5e1;">{row['Expected Value']}</td>
                        <td style="padding: 10px; color:#cbd5e1;">{row['Actual Ingested Fact']}</td>
                        <td style="padding: 10px; color:#94a3b8;">{row['Evaluated Entity']}</td>
                        <td style="padding: 10px;">{row['Condition Status']}</td>
                    </tr>
                """
            html_table += "</tbody></table>"
            st.markdown(html_table, unsafe_allow_html=True)

    # TAB 3: EXTRACTED SCHEMA (JSON)
    with tab_knowledge:
        st.subheader("Structured Knowledge Representation")
        st.caption("Normalized JSON output extracted by the neural phase and validated by the schema validator.")

        col_j1, col_j2 = st.columns(2)
        with col_j1:
            st.markdown("##### Extracted Facts")
            st.json(result["knowledge"].get("facts", []))
        with col_j2:
            st.markdown("##### Extracted Rules & Conditions")
            st.json(result["knowledge"].get("rules", []))

        st.download_button(
            label="💾 Download Knowledge JSON",
            data=json.dumps(result["knowledge"], indent=2),
            file_name="neuro_symbolic_knowledge.json",
            mime="application/json"
        )

    # TAB 4: FULL AUDIT TRAIL
    with tab_audit:
        st.subheader("Explainable Reasoning & Audit Trail")
        st.markdown("""
        Every inference step is deterministic and audit-compliant. The root cause analysis details all passing and failing constraints:
        """)

        st.markdown(f'<div class="explanation-box">{result["explanation"]}</div>', unsafe_allow_html=True)

        if result.get("failed_conditions"):
            st.markdown("##### ⚠️ Root Cause Breakdown (Violations):")
            for c in result["failed_conditions"]:
                st.error(f"Constraint Failed: **{c['attribute']} {c['operator']} {c['expected']}** but actual fact was **{c['actual']}**")

        st.markdown("---")
        st.caption("Neuro-Symbolic Reasoning Engine • 100% Deterministic Guarantee")
