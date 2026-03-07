from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def parse_markdown_to_flowables(md_text: str):
    styles = getSampleStyleSheet()
    style_title = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=19,
        spaceAfter=6,
    )
    style_h1 = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=14,
        spaceBefore=6,
        spaceAfter=3,
    )
    style_h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=12,
        spaceBefore=4,
        spaceAfter=2,
    )
    style_body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.6,
        leading=10.8,
        spaceAfter=1.3,
    )
    style_bullet = ParagraphStyle(
        "Bullet",
        parent=style_body,
        leftIndent=12,
        bulletIndent=6,
        spaceAfter=1,
    )
    style_code = ParagraphStyle(
        "Code",
        parent=style_body,
        fontName="Courier",
        fontSize=8.0,
        leading=10,
        backColor=colors.whitesmoke,
        leftIndent=6,
        rightIndent=6,
        spaceBefore=1,
        spaceAfter=2,
    )

    def esc(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    lines = md_text.splitlines()
    flowables = []
    i = 0
    in_code = False
    code_buf = []

    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip("\n")

        if line.strip().startswith("```"):
            if not in_code:
                in_code = True
                code_buf = []
            else:
                in_code = False
                if code_buf:
                    flowables.append(Paragraph(esc("<br/>".join(code_buf)), style_code))
                    flowables.append(Spacer(1, 0.03 * inch))
            i += 1
            continue

        if in_code:
            code_buf.append(line)
            i += 1
            continue

        # Markdown table block
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1
            rows = []
            for tline in table_lines:
                if re.match(r"^\|\s*-", tline):
                    continue
                cells = [c.strip() for c in tline.strip().strip("|").split("|")]
                rows.append(cells)
            if rows:
                t = Table(rows, repeatRows=1)
                t.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, -1), 7.2),
                            ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 2),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                            ("TOPPADDING", (0, 0), (-1, -1), 1),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                        ]
                    )
                )
                flowables.append(t)
                flowables.append(Spacer(1, 0.03 * inch))
            continue

        if not line.strip():
            flowables.append(Spacer(1, 0.01 * inch))
            i += 1
            continue

        if line.startswith("# "):
            flowables.append(Paragraph(esc(line[2:].strip()), style_title))
            i += 1
            continue
        if line.startswith("## "):
            flowables.append(Paragraph(esc(line[3:].strip()), style_h1))
            i += 1
            continue
        if line.startswith("### "):
            flowables.append(Paragraph(esc(line[4:].strip()), style_h2))
            i += 1
            continue

        if line.startswith("- "):
            flowables.append(Paragraph(esc(line[2:].strip()), style_bullet, bulletText="•"))
            i += 1
            continue

        # Horizontal rule
        if line.strip() == "---":
            flowables.append(Spacer(1, 0.03 * inch))
            i += 1
            continue

        # Inline code markdown
        para = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", esc(line))
        # Bold markdown
        para = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", para)
        flowables.append(Paragraph(para, style_body))
        i += 1

    return flowables


def build_pdf(src_md: Path, out_pdf: Path) -> None:
    text = src_md.read_text(encoding="utf-8")
    flowables = parse_markdown_to_flowables(text)
    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=LETTER,
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
        title="Economics-First Robotics Stack White Paper",
        author="robotics-vp-core",
    )
    doc.build(flowables)


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    src = repo / "docs" / "whitepaper_objective_tensor_stack.md"
    out = repo / "docs" / "whitepaper_objective_tensor_stack.pdf"
    build_pdf(src, out)
    print(str(out))
