import html
import json
import platform
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .config import settings


def export_solution(run_id, package, result, name):
    directory = settings.data_dir / "models" / run_id
    directory.mkdir(exist_ok=True)
    portable_app = directory / "app"
    portable_app.mkdir(exist_ok=True)
    (portable_app / "__init__.py").write_text("")
    (portable_app / "features.py").write_text(Path(__file__).with_name("features.py").read_text(), encoding="utf-8")
    joblib.dump(package, directory / "pipeline.joblib", compress=3)
    for filename, data in (("metrics.json", result), ("input_schema.json", result["input_schema"]), ("pipeline.json", result["pipeline"])):
        (directory / filename).write_text(json.dumps(data, indent=2), encoding="utf-8")
    (directory / "requirements.txt").write_text(f"scikit-learn=={sklearn.__version__}\npandas=={pd.__version__}\nnumpy=={np.__version__}\njoblib=={joblib.__version__}\n", encoding="utf-8")
    (directory / "predict.py").write_text(STANDALONE_PREDICT, encoding="utf-8")
    (directory / "README.md").write_text(f"# {name}: {package['task']} solution\n\nUse Python {platform.python_version()}.\n\n```bash\npip install -r requirements.txt\npython predict.py input.json\n```\n\nInput: JSON array matching input_schema.json. All feature keys are required; values may be null. Keep the included app/features.py module with the pipeline.\n\n{result['report']}\n", encoding="utf-8")
    (directory / "Dockerfile").write_text('FROM python:3.12-slim\nWORKDIR /solution\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nENTRYPOINT ["python", "predict.py"]\n', encoding="utf-8")
    (directory / "report.html").write_text(build_report_html(name, result), encoding="utf-8")
    with zipfile.ZipFile(directory / "solution.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in directory.rglob("*"):
            if file.is_file() and file.name != "solution.zip" and "__pycache__" not in file.parts:
                archive.write(file, file.relative_to(directory))
    return directory


def build_report_html(name, result):
    score_cards = "".join(f"<div class='metric'><strong>{html.escape(key.upper())}</strong><span>{value:.4f}</span></div>" for key, value in result["metrics"].items())
    pipeline = "".join(f"<li><strong>{html.escape(stage['name'])}</strong><p>{html.escape(stage['detail'])}</p></li>" for stage in result["pipeline"])
    comparisons = "".join(f"<tr><td>{html.escape(experiment['name'])}</td><td>{html.escape(experiment['family'])}</td><td>{html.escape(experiment['status'])}</td><td>{experiment.get('selection_score', '—')}</td></tr>" for experiment in result["experiments"])
    importance = "".join(f"<tr><td>{html.escape(item['feature'])}</td><td>{item['importance']:.5f}</td></tr>" for item in result["feature_importance"])
    shap = result["shap"]
    shap_rows = "".join(f"<tr><td>{html.escape(item['feature'])}</td><td>{item['importance']:.5f}</td></tr>" for item in shap.get("feature_importance", []))
    evidence = html.escape(json.dumps(result["error_analysis"], indent=2))
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(name)} — Analytiq report</title>
    <style>body{{font:15px system-ui;background:#f7f8fc;color:#292c43;max-width:1080px;margin:40px auto;padding:24px}}header{{border-bottom:2px solid #9d84cf;padding-bottom:20px}}h1,h2{{letter-spacing:-.6px}}section{{background:white;border:1px solid #e9ebf3;border-radius:14px;padding:24px;margin:24px 0}}.metrics{{display:flex;flex-wrap:wrap;gap:14px;margin-top:25px}}.metric{{background:white;border:1px solid #e9ebf3;border-radius:10px;padding:18px;min-width:110px}}.metric span{{display:block;font-size:27px;margin-top:8px;color:#a88acb}}table{{width:100%;border-collapse:collapse}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #e9ebf3}}pre{{font:inherit;line-height:1.9;white-space:pre-wrap;overflow-wrap:anywhere}}p,small{{color:#7c8199;line-height:1.8}}li{{margin:16px 0}}code{{overflow-wrap:anywhere}}</style></head><body>
    <header><small>ANALYTIQ / FINAL ANALYSIS REPORT</small><h1>{html.escape(name)}</h1><p>{html.escape(result['model_name'])} · {html.escape(result['task'])} · {html.escape(result.get('target') or 'Unsupervised grouping')}</p></header>
    <div class="metrics">{score_cards}</div><section><h2>Analysis summary</h2><pre>{html.escape(result['report'])}</pre></section>
    <section><h2>Executed pipeline</h2><ol>{pipeline}</ol></section>
    <section><h2>Model comparison</h2><table><tr><th>Model</th><th>Family</th><th>Status</th><th>Selection score</th></tr>{comparisons}</table><p>{html.escape(result['selection_note'])}</p></section>
    <section><h2>Feature importance</h2><table>{importance}</table></section><section><h2>SHAP explanation</h2><p>{html.escape(shap.get('method') or shap.get('reason') or '')} · {html.escape(shap.get('output') or '')}</p><table>{shap_rows}</table></section>
    <section><h2>Error / assignment analysis</h2><pre>{evidence}</pre></section>
    <small>Dataset SHA-256: <code>{result['dataset_fingerprint']}</code> · Seed {result['seed']} · scikit-learn {result['sklearn_version']}</small></body></html>"""


STANDALONE_PREDICT = '''import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

package = joblib.load(Path(__file__).with_name("pipeline.joblib"))
records = json.loads(Path(sys.argv[1]).read_text())
for index, record in enumerate(records):
    missing = set(package["features"]) - set(record)
    if missing:
        raise ValueError(f"Record {index} is missing: {sorted(missing)}")
frame = pd.DataFrame(records)[package["features"]]
for col in package["numeric"]:
    frame[col] = pd.to_numeric(frame[col], errors="raise")
for col in package["categorical"]:
    frame[col] = frame[col].map(lambda value: str(value) if pd.notna(value) else np.nan)
prediction = package["pipeline"].predict(frame)
if package["encoder"] is not None:
    prediction = package["encoder"].inverse_transform(prediction.astype(int))
print(json.dumps({"task": package["task"], "predictions": prediction.tolist()}, indent=2))
'''
