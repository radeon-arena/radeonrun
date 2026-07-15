#!/usr/bin/env python3
"""Build a static recipe explorer for radeonrun.

The output is a single HTML file that can be opened directly or published via any
static file host. It intentionally has no frontend build step.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from radeonrun_config import image_tag, list_run_configs, load_run_config, resolve_image  # noqa: E402


def _engine_of(recipe: dict[str, Any]) -> str:
    runtime = str(recipe.get("runtime") or "").strip()
    if runtime:
        return runtime
    container = str(recipe.get("container") or "").lower()
    command = str(recipe.get("command") or "").lower()
    if "llamacpp" in container or "llama-server" in command:
        return "llamacpp"
    if "vllm-main" in container:
        return "vllm-main"
    return "vllm"


def _family(name: str, model: str, source: str) -> str:
    text = " ".join([name, model, source]).lower()
    if "qwen" in text:
        return "Qwen"
    if "gemma" in text:
        return "Gemma"
    if "llama" in text:
        return "Llama"
    if "mimo" in text:
        return "MiMo"
    if "step" in text:
        return "Step"
    if "diffusion" in text:
        return "Diffusion"
    return "Other"


def _serve_name(command: str) -> str | None:
    match = re.search(r"--served-model-name(?:=|\s+)(\S+)", command) or re.search(r"--alias(?:=|\s+)(\S+)", command)
    return match.group(1).rstrip("\\") if match else None


def _best_measurement(measurements: list[dict[str, Any]]) -> dict[str, Any] | None:
    best = None
    for item in measurements:
        value = item.get("decode_toks_per_s")
        if isinstance(value, (int, float)) and (best is None or value > best.get("decode_toks_per_s", 0)):
            best = item
    return best


def _load_results(root: Path) -> dict[str, dict[str, Any]]:
    bundle_path = root / "results" / "bundle.json"
    if not bundle_path.exists():
        return {}
    bundle = json.loads(bundle_path.read_text())
    by_recipe: dict[str, dict[str, Any]] = {}
    for device, records in (bundle.get("records") or {}).items():
        for record in records:
            recipe = record.get("recipe") or {}
            data = record.get("data") or {}
            name = recipe.get("name") or Path(record.get("recipe_file", "")).stem
            measurements = data.get("measurements") or []
            best = _best_measurement(measurements)
            if not best:
                continue
            current = by_recipe.get(name)
            if current is None or best.get("decode_toks_per_s", 0) > current.get("best", {}).get("decode_toks_per_s", 0):
                by_recipe[name] = {
                    "device": device,
                    "profile": data.get("profile"),
                    "points": len(measurements),
                    "best": best,
                    "generated_at": data.get("generated_at"),
                }
    return by_recipe


def _recipe_to_card(recipe_id: str, result: dict[str, Any] | None) -> dict[str, Any]:
    recipe = load_run_config(recipe_id)
    metadata = recipe.get("metadata") or {}
    measured = metadata.get("measured") or {}
    command = str(recipe.get("command") or "").strip()
    name = str(recipe.get("name") or recipe_id)
    model = str(recipe.get("model") or "")
    source = str(recipe.get("source") or "")
    defaults = recipe.get("defaults") or {}
    env = recipe.get("env") or {}
    image = resolve_image(recipe, str((recipe.get("_device") or {}).get("id") or "halo"))
    return {
        "file": (recipe.get("_spec_files") or {}).get("matrix") or (recipe.get("_spec_files") or {}).get("legacy_recipe"),
        "spec_files": recipe.get("_spec_files") or {},
        "config_source": recipe.get("_config_source") or "legacy",
        "name": name,
        "title": str(recipe.get("description") or metadata.get("description") or name),
        "family": _family(name, model, source),
        "runtime": _engine_of(recipe),
        "container": str(recipe.get("container") or ""),
        "image": image,
        "image_tag": image_tag(image) or "",
        "quantization": str(metadata.get("quantization") or ""),
        "model": model,
        "source": source,
        "served_model_name": _serve_name(command),
        "defaults": defaults,
        "env": env,
        "command": command,
        "measured": measured,
        "result": result,
        "patches": recipe.get("model_patches") or [],
    }


def _build_data(root: Path) -> dict[str, Any]:
    results = _load_results(root)
    recipes = []
    for recipe_id in list_run_configs():
        config = load_run_config(recipe_id)
        recipes.append(_recipe_to_card(recipe_id, results.get(config.get("name") or recipe_id)))
    facets = {
        "families": sorted({r["family"] for r in recipes}),
        "runtimes": sorted({r["runtime"] for r in recipes}),
        "quantizations": sorted({r["quantization"] for r in recipes if r["quantization"]}),
    }
    return {"recipes": recipes, "facets": facets}


def _html(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Radeon Arena Recipes</title>
  <style>
    :root {{
      color-scheme: light;
      --ink:#14211f; --muted:#61716e; --line:#d8e1de; --paper:#f7faf8; --panel:#ffffff;
      --accent:#b8322b; --accent-2:#0f766e; --chip:#edf4f1; --code:#10201e;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background:linear-gradient(180deg,#f6faf7 0,#edf4f1 48%,#f8faf9 100%); color:var(--ink); }}
    header {{ padding:32px clamp(20px,4vw,48px) 22px; border-bottom:1px solid var(--line); background:rgba(255,255,255,.72); backdrop-filter: blur(10px); position:sticky; top:0; z-index:5; }}
    .brand {{ display:flex; align-items:center; gap:14px; margin-bottom:18px; }}
    .mark {{ width:38px; height:38px; border-radius:8px; background:linear-gradient(135deg,#e43d30,#0f766e); box-shadow:0 8px 22px rgba(184,50,43,.25); }}
    h1 {{ font-size:clamp(28px,4vw,46px); line-height:1; margin:0; letter-spacing:0; }}
    .subtitle {{ color:var(--muted); margin:8px 0 0; max-width:820px; }}
    .toolbar {{ display:grid; grid-template-columns:minmax(240px,1fr) repeat(3,minmax(150px,190px)); gap:10px; margin-top:18px; }}
    input, select {{ width:100%; border:1px solid var(--line); border-radius:8px; background:#fff; padding:11px 12px; color:var(--ink); font:inherit; }}
    main {{ padding:24px clamp(20px,4vw,48px) 48px; }}
    .stats {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:18px; }}
    .stat {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px 12px; min-width:120px; }}
    .stat strong {{ display:block; font-size:22px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); gap:14px; }}
    article {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; box-shadow:0 10px 24px rgba(20,33,31,.05); }}
    .card-head {{ padding:16px; display:grid; gap:10px; }}
    .name {{ font-size:17px; font-weight:750; line-height:1.22; overflow-wrap:anywhere; }}
    .desc {{ color:var(--muted); font-size:13px; line-height:1.45; min-height:36px; }}
    .chips {{ display:flex; gap:6px; flex-wrap:wrap; }}
    .chip {{ border:1px solid var(--line); background:var(--chip); border-radius:999px; padding:4px 8px; font-size:12px; color:#27413d; }}
    .perf {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }}
    .perf div {{ border:1px solid var(--line); border-radius:8px; padding:8px; background:#fbfdfc; }}
    .perf b {{ display:block; font-size:16px; }}
    .perf span {{ color:var(--muted); font-size:11px; }}
    details {{ border-top:1px solid var(--line); }}
    summary {{ cursor:pointer; padding:12px 16px; color:var(--accent-2); font-weight:700; }}
    .detail {{ padding:0 16px 16px; display:grid; gap:12px; }}
    dl {{ display:grid; grid-template-columns:110px 1fr; gap:6px 10px; margin:0; font-size:13px; }}
    dt {{ color:var(--muted); }} dd {{ margin:0; overflow-wrap:anywhere; }}
    pre {{ margin:0; background:var(--code); color:#d7fff3; border-radius:8px; padding:12px; overflow:auto; font-size:12px; line-height:1.45; max-height:280px; }}
    button {{ border:1px solid var(--line); border-radius:8px; background:#fff; color:var(--ink); padding:8px 10px; font-weight:700; cursor:pointer; width:max-content; }}
    .empty {{ padding:42px; text-align:center; color:var(--muted); border:1px dashed var(--line); border-radius:8px; background:rgba(255,255,255,.7); }}
    @media (max-width:860px) {{ .toolbar {{ grid-template-columns:1fr; }} header {{ position:static; }} }}
  </style>
</head>
<body>
  <header>
    <div class=\"brand\"><div class=\"mark\"></div><div><h1>Radeon Arena Recipes</h1><p class=\"subtitle\">Serve commands, model sources, image pins, environment flags, and measured results for every reproducible Radeon recipe.</p></div></div>
    <div class=\"toolbar\">
      <input id=\"q\" placeholder=\"Search model, source, command...\" />
      <select id=\"family\"><option value=\"\">All families</option></select>
      <select id=\"runtime\"><option value=\"\">All runtimes</option></select>
      <select id=\"quant\"><option value=\"\">All quantization</option></select>
    </div>
  </header>
  <main>
    <section class=\"stats\" id=\"stats\"></section>
    <section class=\"grid\" id=\"grid\"></section>
  </main>
  <script id=\"recipe-data\" type=\"application/json\">{html.escape(payload)}</script>
  <script>
    const data = JSON.parse(document.getElementById('recipe-data').textContent);
    const state = {{ q:'', family:'', runtime:'', quant:'' }};
    const el = id => document.getElementById(id);
    const text = value => value == null || value === '' ? '—' : String(value);
    const fmt = value => typeof value === 'number' ? value.toFixed(value >= 100 ? 0 : 2) : '—';
    function optionize(id, values) {{
      const select = el(id);
      values.forEach(v => {{ const o=document.createElement('option'); o.value=v; o.textContent=v; select.appendChild(o); }});
      select.addEventListener('change', e => {{ state[id]=e.target.value; render(); }});
    }}
    optionize('family', data.facets.families); optionize('runtime', data.facets.runtimes); optionize('quant', data.facets.quantizations);
    el('q').addEventListener('input', e => {{ state.q=e.target.value.toLowerCase(); render(); }});
    function matches(r) {{
      const hay = [r.name,r.title,r.model,r.source,r.command,r.quantization,r.runtime,r.container,r.image,r.image_tag].join(' ').toLowerCase();
      return (!state.q || hay.includes(state.q)) && (!state.family || r.family===state.family) && (!state.runtime || r.runtime===state.runtime) && (!state.quant || r.quantization===state.quant);
    }}
    function perf(r) {{
      const best = r.result?.best;
      const measured = r.measured || {{}};
      const value = best?.decode_toks_per_s ?? measured.decode_toks_per_s;
      const profile = r.result?.profile ?? measured.profile;
      const conc = best?.concurrency;
      return `<div><b>${{fmt(value)}}</b><span>best decode tok/s</span></div><div><b>${{text(profile)}}</b><span>profile</span></div><div><b>${{conc ? 'c'+conc : '—'}}</b><span>best concurrency</span></div>`;
    }}
    function card(r) {{
      const env = Object.entries(r.env || {{}}).map(([k,v]) => `${{k}}=${{v}}`).join('\n');
      const defaults = Object.entries(r.defaults || {{}}).map(([k,v]) => `${{k}}: ${{v}}`).join('\n');
      return `<article>
        <div class=\"card-head\">
          <div class=\"name\">${{r.name}}</div>
          <div class=\"desc\">${{r.title}}</div>
          <div class=\"chips\"><span class=\"chip\">${{r.family}}</span><span class=\"chip\">${{r.runtime}}</span><span class=\"chip\">${{r.quantization || 'unquantized'}}</span><span class=\"chip\">${{r.image_tag || 'latest'}}</span></div>
          <div class=\"perf\">${{perf(r)}}</div>
        </div>
        <details><summary>Recipe details</summary><div class=\"detail\">
          <dl><dt>Model</dt><dd>${{text(r.model)}}</dd><dt>Source</dt><dd>${{text(r.source)}}</dd><dt>Served name</dt><dd>${{text(r.served_model_name)}}</dd><dt>Image</dt><dd>${{text(r.image)}}</dd><dt>Config</dt><dd>${{r.file}}</dd></dl>
          ${{defaults ? `<pre>${{defaults}}</pre>` : ''}}
          ${{env ? `<pre>${{env}}</pre>` : ''}}
          <button data-copy=\"${{r.name}}\">Copy command</button><pre id=\"cmd-${{r.name}}\">${{r.command.replace(/[&<>]/g, s => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[s]))}}</pre>
        </div></details>
      </article>`;
    }}
    function render() {{
      const rows = data.recipes.filter(matches);
      const measured = rows.filter(r => r.result || r.measured?.decode_toks_per_s).length;
      el('stats').innerHTML = `<div class=\"stat\"><strong>${{rows.length}}</strong><span>visible recipes</span></div><div class=\"stat\"><strong>${{data.recipes.length}}</strong><span>total recipes</span></div><div class=\"stat\"><strong>${{measured}}</strong><span>with measurements</span></div>`;
      el('grid').innerHTML = rows.length ? rows.map(card).join('') : '<div class=\"empty\">No recipes match the current filters.</div>';
      document.querySelectorAll('button[data-copy]').forEach(btn => btn.onclick = async () => {{
        const name = btn.getAttribute('data-copy'); const text = document.getElementById('cmd-'+name).textContent;
        await navigator.clipboard.writeText(text); btn.textContent='Copied'; setTimeout(()=>btn.textContent='Copy command',900);
      }});
    }}
    render();
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build static Radeon Arena recipe explorer")
    parser.add_argument("--out", default="docs/recipes.html", help="Output HTML path")
    args = parser.parse_args()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_html(_build_data(ROOT)))
    try:
      shown = out.relative_to(ROOT)
    except ValueError:
      shown = out
    print(f"wrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
