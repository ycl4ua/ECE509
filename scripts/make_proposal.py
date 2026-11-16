"""Render proposal/proposal.md to a one-page PDF using ReportLab."""

import os
import re

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_PATH = os.path.join(ROOT, "proposal", "proposal.md")
PDF_PATH = os.path.join(ROOT, "proposal", "proposal.pdf")

FONT_FILES = {
    "TimesNewRoman": r"C:\Windows\Fonts\times.ttf",
    "TimesNewRoman-Bold": r"C:\Windows\Fonts\timesbd.ttf",
    "TimesNewRoman-Italic": r"C:\Windows\Fonts\timesi.ttf",
}
if all(os.path.exists(p) for p in FONT_FILES.values()):
    for name, path in FONT_FILES.items():
        pdfmetrics.registerFont(TTFont(name, path))
    BODY, BOLD, ITALIC = "TimesNewRoman", "TimesNewRoman-Bold", "TimesNewRoman-Italic"
else:
    BODY, BOLD, ITALIC = "Times-Roman", "Times-Bold", "Times-Italic"

# ReportLab's standard fonts do not cover every Unicode glyph used in the source.
REPLACEMENTS = [
    ("ρa", "rho_a"),
    ("ρn", "rho_n"),
    ("ρ", "rho"),
    ("ŷ", "y_hat"),
    ("×", "x"),
    ("≤", "<="),
    ("≥", ">="),
    ("→", "->"),
    ("∈", "in"),
    ("–", "-"),
    ("—", "-"),
    ("·", "|"),
    ("’", "'"),
    ("‘", "'"),
    ("“", '"'),
    ("”", '"'),
    ("…", "..."),
    ("±", "+/-"),
    ("≈", "~"),
]


def normalize(text: str) -> str:
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    # Escape XML special characters before inserting our own <b> tags.
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return text


def parse_markdown(text: str):
    blocks = []
    paragraph = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if paragraph:
                blocks.append(("p", " ".join(paragraph)))
                paragraph = []
            continue
        if line.startswith("# "):
            if paragraph:
                blocks.append(("p", " ".join(paragraph)))
                paragraph = []
            blocks.append(("title", line[2:].strip()))
        elif line.startswith("## "):
            if paragraph:
                blocks.append(("p", " ".join(paragraph)))
                paragraph = []
            blocks.append(("h2", line[3:].strip()))
        elif line.startswith("- "):
            if paragraph:
                blocks.append(("p", " ".join(paragraph)))
                paragraph = []
            blocks.append(("bullet", line[2:].strip()))
        else:
            paragraph.append(line)
    if paragraph:
        blocks.append(("p", " ".join(paragraph)))
    return blocks


def main():
    with open(MD_PATH, "r", encoding="utf-8") as handle:
        blocks = parse_markdown(handle.read())

    title_style = ParagraphStyle(
        "title", fontName=BOLD, fontSize=14.5, leading=17, alignment=TA_CENTER, spaceAfter=3
    )
    meta_style = ParagraphStyle(
        "meta", fontName=BODY, fontSize=8.6, leading=10.5, alignment=TA_CENTER, spaceAfter=5
    )
    heading_style = ParagraphStyle(
        "heading", fontName=BOLD, fontSize=10.1, leading=12.2, spaceBefore=4.5, spaceAfter=1.5
    )
    body_style = ParagraphStyle(
        "body", fontName=BODY, fontSize=9.15, leading=10.95, alignment=TA_JUSTIFY, spaceAfter=2.4
    )
    bullet_style = ParagraphStyle(
        "bullet", parent=body_style, leftIndent=10, bulletIndent=1.5, spaceAfter=1.2
    )
    ref_style = ParagraphStyle(
        "ref", parent=body_style, fontSize=8.0, leading=9.5, spaceAfter=1.0
    )

    story = []
    seen_title = False
    seen_meta = False
    in_references = False
    for kind, text in blocks:
        if kind == "title":
            story.append(Paragraph(normalize(text), title_style))
            seen_title = True
            continue
        if kind == "h2":
            in_references = text.strip().lower() == "references"
            story.append(Paragraph(normalize(text), heading_style))
            continue
        if kind == "bullet":
            story.append(Paragraph(normalize(text), bullet_style, bulletText="•"))
            continue
        if seen_title and not seen_meta:
            story.append(Paragraph(normalize(text), meta_style))
            seen_meta = True
            continue
        story.append(Paragraph(normalize(text), ref_style if in_references else body_style))

    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=A4,
        leftMargin=0.70 * inch,
        rightMargin=0.70 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="ECE 509 Proposal: Reproducing WaNet",
        author="ECE 509 Team",
    )
    doc.build(story)

    from pypdf import PdfReader

    reader = PdfReader(PDF_PATH)
    print(f"Wrote {PDF_PATH} ({len(reader.pages)} page(s))")
    print(reader.pages[0].extract_text()[:600])


if __name__ == "__main__":
    main()
