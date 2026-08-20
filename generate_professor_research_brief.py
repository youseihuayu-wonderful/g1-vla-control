#!/usr/bin/env python3
"""Generate a public, print-ready professor brief from the evidence-bound Markdown."""

from __future__ import annotations

import html
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "PROFESSOR_RESEARCH_BRIEF_CN.md"
OUTPUT = ROOT / "professor_research_brief.html"
PUBLIC_OUTPUT = ROOT / "vercel_public" / "research-brief" / "index.html"


def inline(text: str) -> str:
    rendered = html.escape(text.strip())
    rendered = re.sub(r"`([^`]+)`", r"<code>\1</code>", rendered)
    rendered = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", rendered)
    return rendered


def cells(line: str) -> list[str]:
    return [part.strip() for part in line.strip().strip("|").split("|")]


def markdown_to_html(markdown: str) -> tuple[str, list[tuple[str, str]]]:
    lines = markdown.replace("\ufeff", "").splitlines()
    output: list[str] = []
    toc: list[tuple[str, str]] = []
    paragraph: list[str] = []
    list_type: str | None = None
    index = 0
    section_id = 0

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            output.append(f"<p>{inline(' '.join(paragraph))}</p>")
            paragraph = []

    def close_list() -> None:
        nonlocal list_type
        if list_type:
            output.append(f"</{list_type}>")
            list_type = None

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_paragraph()
            close_list()
            language = stripped[3:].strip()
            code: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(lines[index])
                index += 1
            output.append(
                f'<pre data-language="{html.escape(language)}"><code>'
                f"{html.escape(chr(10).join(code))}</code></pre>"
            )
        elif re.match(r"^#{1,4}\s+", stripped):
            flush_paragraph()
            close_list()
            hashes, title = stripped.split(" ", 1)
            level = len(hashes)
            section_id += 1
            anchor = f"section-{section_id}"
            output.append(f'<h{level} id="{anchor}">{inline(title)}</h{level}>')
            if level == 2:
                toc.append((anchor, title))
        elif stripped == "---":
            flush_paragraph()
            close_list()
            output.append("<hr>")
        elif (
            stripped.startswith("|")
            and index + 1 < len(lines)
            and re.match(r"^\s*\|?\s*:?-{3,}", lines[index + 1])
        ):
            flush_paragraph()
            close_list()
            headers = cells(line)
            index += 2
            body: list[list[str]] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                body.append(cells(lines[index]))
                index += 1
            index -= 1
            output.append('<div class="table-frame"><table><thead><tr>')
            output.extend(f"<th>{inline(value)}</th>" for value in headers)
            output.append("</tr></thead><tbody>")
            for row in body:
                output.append("<tr>")
                output.extend(f"<td>{inline(value)}</td>" for value in row)
                output.append("</tr>")
            output.append("</tbody></table></div>")
        elif re.match(r"^[-*]\s+", stripped):
            flush_paragraph()
            if list_type != "ul":
                close_list()
                output.append("<ul>")
                list_type = "ul"
            item = re.sub(r"^[-*]\\s+", "", stripped)
            output.append(f"<li>{inline(item)}</li>")
        elif re.match(r"^\d+\.\s+", stripped):
            flush_paragraph()
            if list_type != "ol":
                close_list()
                output.append("<ol>")
                list_type = "ol"
            item = re.sub(r"^\\d+\\.\\s+", "", stripped)
            output.append(f"<li>{inline(item)}</li>")
        elif not stripped:
            flush_paragraph()
            close_list()
        else:
            close_list()
            paragraph.append(stripped)
        index += 1
    flush_paragraph()
    close_list()
    return "".join(output), toc


def build() -> str:
    body, toc = markdown_to_html(SOURCE.read_text())
    navigation = "".join(
        f'<a href="#{html.escape(anchor)}"><span>{number:02d}</span>{inline(title)}</a>'
        for number, (anchor, title) in enumerate(toc, start=1)
    )
    css = """
    :root{--ink:#172033;--muted:#647087;--line:#dfe5ee;--blue:#2855d9;--cyan:#0f91a8;--green:#148260;--amber:#a8610a;--red:#b4324b;--paper:#fff;--wash:#f4f7fb}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--wash);color:var(--ink);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif;line-height:1.7}.top{position:sticky;top:0;z-index:20;display:flex;align-items:center;justify-content:space-between;padding:12px max(22px,calc((100vw - 1420px)/2));border-bottom:1px solid #dce3edcc;background:#f8fafddd;backdrop-filter:blur(18px)}.brand{display:flex;align-items:center;gap:10px;font-weight:850}.mark{display:grid;place-items:center;width:34px;height:34px;border-radius:10px;background:linear-gradient(145deg,#2855d9,#1395ad);color:#fff;font-size:11px}.top-meta{display:flex;align-items:center;gap:8px}.chip{padding:6px 10px;border:1px solid var(--line);border-radius:999px;background:#fff;color:var(--muted);font-size:10px;font-weight:750}.print{border:0;border-radius:10px;padding:8px 12px;background:var(--ink);color:#fff;font-weight:750;cursor:pointer}.layout{display:grid;grid-template-columns:260px minmax(0,880px);justify-content:center;gap:34px;width:min(1420px,calc(100% - 36px));margin:34px auto 70px}.toc{position:sticky;top:82px;align-self:start;max-height:calc(100vh - 110px);overflow:auto;padding:14px;border:1px solid var(--line);border-radius:18px;background:#ffffffd9;box-shadow:0 16px 50px #18305a0d}.toc-label{padding:8px 10px 12px;color:#8995a8;font-size:9px;font-weight:900;letter-spacing:.16em}.toc a{display:grid;grid-template-columns:24px 1fr;gap:7px;padding:8px 9px;border-radius:10px;color:#657086;text-decoration:none;font-size:10px;line-height:1.35}.toc a:hover{background:#edf3ff;color:var(--blue)}.toc a span{color:#9aa6b8;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.paper{min-width:0;padding:64px 70px;border:1px solid var(--line);border-radius:24px;background:var(--paper);box-shadow:0 28px 80px #1a315715}.paper h1{margin:0 0 22px;font-size:40px;line-height:1.15;letter-spacing:-.04em}.paper h1+p,.paper h1+p+p,.paper h1+p+p+p{margin:2px 0;color:var(--muted);font-size:12px}.paper h2{margin:58px 0 18px;padding-top:8px;border-top:2px solid var(--ink);font-size:26px;line-height:1.25;letter-spacing:-.025em}.paper h3{margin:30px 0 10px;color:#263752;font-size:17px}.paper h4{margin:22px 0 7px;font-size:14px}.paper p{margin:12px 0;color:#344158}.paper ul,.paper ol{padding-left:23px;color:#344158}.paper li+li{margin-top:6px}.paper strong{color:#101827}.paper hr{margin:30px 0;border:0;border-top:1px solid var(--line)}.paper code{padding:2px 5px;border-radius:5px;background:#eef2f7;color:#243b6b;font:600 .88em ui-monospace,SFMono-Regular,Menlo,monospace}.paper pre{overflow:visible;margin:18px 0;padding:18px 20px;border:1px solid #d8e1ee;border-radius:14px;background:#f6f8fb;color:#263752;white-space:pre-wrap;overflow-wrap:anywhere;font:600 11px/1.7 ui-monospace,SFMono-Regular,Menlo,monospace}.table-frame{margin:18px 0;border:1px solid var(--line);border-radius:14px;overflow:hidden}table{width:100%;table-layout:fixed;border-collapse:collapse;font-size:10.5px;line-height:1.5}th{padding:11px 10px;background:#edf2f8;color:#4d5c72;text-align:left;font-size:9px;letter-spacing:.05em;text-transform:uppercase}td{padding:10px;vertical-align:top;border-top:1px solid #e6ebf2;color:#344158;overflow-wrap:anywhere}tr:nth-child(even) td{background:#fafbfd}.evidence-banner{display:grid;grid-template-columns:repeat(3,1fr);gap:9px;margin:0 0 28px}.evidence-banner div{padding:13px;border:1px solid var(--line);border-radius:12px;background:#f8fafc}.evidence-banner small{display:block;color:#8a97aa;font-size:8px;font-weight:900;letter-spacing:.1em}.evidence-banner b{display:block;margin-top:3px;font-size:12px}.footer-note{margin-top:44px;padding-top:18px;border-top:1px solid var(--line);color:#8793a5;font-size:9px}.footer-note a{color:var(--blue)}
    @media(max-width:960px){.layout{display:block;width:min(900px,calc(100% - 22px));margin-top:18px}.toc{position:static;max-height:none;margin-bottom:14px}.toc nav{display:grid;grid-template-columns:repeat(2,1fr)}.paper{padding:42px 35px}.top-meta .chip{display:none}}
    @media(max-width:620px){.top{padding:10px 12px}.paper{padding:30px 20px;border-radius:17px}.paper h1{font-size:31px}.paper h2{font-size:22px}.toc nav{grid-template-columns:1fr}.evidence-banner{grid-template-columns:1fr}table{font-size:9px}th,td{padding:8px 6px}}
    @media print{body{background:#fff}.top,.toc{display:none}.layout{display:block;width:auto;margin:0}.paper{width:auto;padding:0;border:0;border-radius:0;box-shadow:none}.paper h2{break-before:page}.paper h2:first-of-type{break-before:auto}.table-frame,pre,li{break-inside:avoid}.footer-note{display:none}@page{size:A4;margin:16mm}}
    """
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="author" content="Shihua Yu"><meta name="description" content="LGG100 × Unitree G1 adaptive-speed manipulation research overview"><title>LGG100 × G1 · Professor Research Brief</title><style>{css}</style></head><body>
    <header class="top"><div class="brand"><span class="mark">G1</span><span>Research Brief</span></div><div class="top-meta"><span class="chip">Evidence-bound</span><span class="chip">Shihua Yu</span><button class="print" type="button" onclick="window.print()">Print / PDF</button></div></header>
    <main class="layout"><aside class="toc"><div class="toc-label">DOCUMENT OUTLINE</div><nav>{navigation}</nav></aside><article class="paper"><div class="evidence-banner"><div><small>PROJECT OWNER</small><b>Shihua Yu</b></div><div><small>CURRENT STAGE</small><b>Simulation Validation</b></div><div><small>HARDWARE AUTHORITY</small><b style="color:var(--red)">OFF</b></div></div>{body}<div class="footer-note">Evidence dashboard: <a href="https://g1-vla-simulation-readiness.vercel.app">g1-vla-simulation-readiness.vercel.app</a> · No hardware action was performed.</div></article></main></body></html>"""


def main() -> None:
    rendered = build()
    OUTPUT.write_text(rendered)
    PUBLIC_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_OUTPUT.write_text(rendered)
    print(OUTPUT)
    print(PUBLIC_OUTPUT)


if __name__ == "__main__":
    main()
