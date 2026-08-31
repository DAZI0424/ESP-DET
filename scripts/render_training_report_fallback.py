from __future__ import annotations

import base64
import html
import io
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "runs/detect/welland_v04_yolo_espdet_pico_416_500e_p50/training_report"


def inline(text: str) -> str:
    value = html.escape(text)
    value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)
    return value


def markdown_to_html(text: str) -> str:
    output: list[str] = []
    paragraph: list[str] = []
    list_kind: str | None = None

    def flush_paragraph() -> None:
        if paragraph:
            output.append(f"<p>{inline(' '.join(paragraph))}</p>")
            paragraph.clear()

    def close_list() -> None:
        nonlocal list_kind
        if list_kind:
            output.append(f"</{list_kind}>")
            list_kind = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush_paragraph()
            close_list()
            continue
        heading = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            close_list()
            level = len(heading.group(1))
            output.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
            continue
        item = re.match(r"^(?:[-*]|\d+\.)\s+(.+)$", line)
        if item:
            flush_paragraph()
            wanted = "ol" if re.match(r"^\d+\.", line) else "ul"
            if list_kind != wanted:
                close_list()
                output.append(f"<{wanted}>")
                list_kind = wanted
            output.append(f"<li>{inline(item.group(1))}</li>")
            continue
        close_list()
        paragraph.append(line)
    flush_paragraph()
    close_list()
    return "\n".join(output)


def svg_data_uri(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    return "data:image/svg+xml;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def line_chart(rows: list[dict], group_field: str, percent: bool, reference: float | None) -> str:
    fig, ax = plt.subplots(figsize=(11.5, 4.4), dpi=130)
    groups: list[str] = []
    for row in rows:
        key = str(row[group_field])
        if key not in groups:
            groups.append(key)
    colors = ["#2563eb", "#f97316", "#16a34a", "#7c3aed"]
    for idx, key in enumerate(groups):
        data = [row for row in rows if str(row[group_field]) == key]
        label = key
        if group_field == "split":
            label = "Train" if idx == 0 else "Validation"
        ax.plot([row["epoch"] for row in data], [row["value"] for row in data],
                label=label, color=colors[idx % len(colors)], linewidth=1.8)
    if reference is not None:
        ax.axvline(reference, color="#64748b", linewidth=1.1, linestyle="--", label=f"Best epoch {int(reference)}")
    ax.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score" if percent else "Loss")
    if percent:
        ax.set_ylim(0, 1.02)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=max(2, len(groups)))
    fig.subplots_adjust(bottom=0.24)
    return svg_data_uri(fig)


def bar_chart(rows: list[dict]) -> str:
    ordered = sorted(rows, key=lambda row: row["map5095"])
    fig, ax = plt.subplots(figsize=(11.5, 4.6), dpi=130)
    bars = ax.barh([row["class"] for row in ordered], [row["map5095"] for row in ordered], color="#2563eb")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("mAP50-95")
    ax.grid(True, axis="x", color="#e2e8f0", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.bar_label(bars, labels=[f"{row['map5095']:.3f}" for row in ordered], padding=4, fontsize=9)
    return svg_data_uri(fig)


def format_value(value, fmt: str | None) -> str:
    if value is None:
        return "—"
    if fmt == "percent":
        return f"{float(value):.2%}"
    if fmt == "number" and isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return html.escape(str(value))


def main() -> None:
    artifact = json.loads((REPORT_DIR / "artifact.json").read_text(encoding="utf-8"))
    manifest = artifact["manifest"]
    datasets = artifact["snapshot"]["datasets"]
    cards = {card["id"]: card for card in manifest["cards"]}
    charts = {chart["id"]: chart for chart in manifest["charts"]}
    tables = {table["id"]: table for table in manifest["tables"]}
    chart_images: dict[str, str] = {}
    for chart_id, chart in charts.items():
        rows = datasets[chart["dataset"]]
        if chart["type"] == "line":
            group = "metric" if "metric" in rows[0] else "split"
            reference = chart.get("referenceLines", [{}])[0].get("value") if chart.get("referenceLines") else None
            chart_images[chart_id] = line_chart(rows, group, chart_id == "validation_trend", reference)
        else:
            chart_images[chart_id] = bar_chart(rows)

    sections: list[str] = []
    summary_row = datasets["summary"][0]
    for block in manifest["blocks"]:
        if block["type"] == "markdown":
            sections.append(f"<section class='text-block'>{markdown_to_html(block['body'])}</section>")
        elif block["type"] == "metric-strip":
            items = []
            for card_id in block["cardIds"]:
                card = cards[card_id]
                metric = card["metrics"][0]
                items.append(
                    "<article class='metric'>"
                    f"<div class='metric-label'>{html.escape(metric['label'])}</div>"
                    f"<div class='metric-value'>{format_value(summary_row.get(metric['field']), metric.get('format'))}</div>"
                    f"<div class='metric-note'>{html.escape(card.get('description', ''))}</div>"
                    "</article>"
                )
            sections.append("<section class='metrics'>" + "".join(items) + "</section>")
        elif block["type"] == "chart":
            chart = charts[block["chartId"]]
            sections.append(
                "<section class='panel'>"
                f"<h3>{html.escape(chart['title'])}</h3><p class='subtitle'>{html.escape(chart.get('subtitle', ''))}</p>"
                f"<img class='chart' src='{chart_images[chart['id']]}' alt='{html.escape(chart['title'])}'>"
                "</section>"
            )
        elif block["type"] == "table":
            table = tables[block["tableId"]]
            header = "".join(f"<th>{html.escape(col['label'])}</th>" for col in table["columns"])
            body = []
            for row in datasets[table["dataset"]]:
                cells = "".join(
                    f"<td>{format_value(row.get(col['field']), col.get('format'))}</td>" for col in table["columns"]
                )
                body.append(f"<tr>{cells}</tr>")
            sections.append(
                "<section class='panel table-panel'>"
                f"<h3>{html.escape(table['title'])}</h3><p class='subtitle'>{html.escape(table.get('subtitle', ''))}</p>"
                f"<div class='table-wrap'><table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
                "</section>"
            )

    source_items = "".join(
        f"<li><strong>{html.escape(source['label'])}</strong><br><code>{html.escape(source['path'])}</code></li>"
        for source in artifact["sources"]
    )
    generated_at = html.escape(manifest.get("generatedAt", ""))
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(manifest['title'])}</title>
<style>
:root{{--ink:#172033;--muted:#64748b;--line:#dbe4f0;--blue:#2563eb;--paper:#fff;--bg:#f4f7fb}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:"Segoe UI","Microsoft YaHei",sans-serif;line-height:1.7}}
main{{max-width:1180px;margin:0 auto;padding:42px 24px 72px}} section{{margin:0 0 24px}} h1{{font-size:34px;line-height:1.2;margin:0 0 8px}} h2{{font-size:24px;margin:34px 0 10px;border-left:4px solid var(--blue);padding-left:12px}} h3{{font-size:19px;margin:0 0 3px}} p{{margin:7px 0}} code{{font-family:Consolas,monospace;background:#eef2f7;padding:2px 5px;border-radius:4px}}
.text-block,.panel{{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:22px 26px;box-shadow:0 5px 18px rgba(30,55,90,.05)}}
.text-block:first-child{{background:linear-gradient(135deg,#173a70,#2563eb);color:white;border:0;padding:30px}} .text-block:first-child h1{{margin:0}} .subtitle,.metric-note{{color:var(--muted)}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}} .metric{{background:white;border:1px solid var(--line);border-radius:14px;padding:18px}} .metric-label{{font-size:13px;color:var(--muted)}} .metric-value{{font-size:28px;font-weight:750;color:#173a70;margin:4px 0}} .metric-note{{font-size:12px;line-height:1.45}}
.chart{{width:100%;height:auto;display:block;margin-top:12px}} .table-wrap{{overflow:auto;margin-top:12px}} table{{border-collapse:collapse;width:100%;font-size:14px}} th{{background:#edf3fb;color:#30405d;text-align:left}} th,td{{border-bottom:1px solid var(--line);padding:10px 12px;white-space:nowrap}} tbody tr:hover{{background:#f8fafc}}
.sources{{background:#eaf1fb;border:1px solid #cad9ef;border-radius:14px;padding:22px 26px}} .sources li{{margin:9px 0}} footer{{color:var(--muted);font-size:12px;text-align:center;margin-top:30px}}
@media(max-width:800px){{.metrics{{grid-template-columns:repeat(2,1fr)}}main{{padding:20px 12px 48px}}.text-block,.panel{{padding:17px}}}}
@media print{{body{{background:white}}main{{max-width:none;padding:0}}.text-block,.panel,.metric{{box-shadow:none;break-inside:avoid}}}}
</style></head><body><main>{''.join(sections)}
<section class="sources"><h2>来源与复核材料</h2><ol>{source_items}</ol></section>
<footer>生成时间：{generated_at} · 单文件离线 HTML · 图表与数据均已内嵌</footer>
</main></body></html>"""
    output = REPORT_DIR / "report.html"
    output.write_text(document, encoding="utf-8")
    print(f"REPORT_HTML={output}")


if __name__ == "__main__":
    main()
