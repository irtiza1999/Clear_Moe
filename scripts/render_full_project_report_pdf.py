from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, Flowable
from svglib.svglib import svg2rlg

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "FULL_PROJECT_REPORT.md"
OUTPUT = ROOT / "FULL_PROJECT_REPORT.pdf"


class SVGFlowable(Flowable):
    def __init__(self, drawing: Drawing, max_width: float):
        super().__init__()
        self.drawing = drawing
        self.max_width = max_width
        self._scale = 1.0
        self._draw_width = drawing.width
        self._draw_height = drawing.height

    def wrap(self, availWidth, availHeight):
        width = min(self.max_width, availWidth)
        if self.drawing.width:
            self._scale = width / self.drawing.width
            self._draw_width = self.drawing.width * self._scale
            self._draw_height = self.drawing.height * self._scale
        return self._draw_width, self._draw_height

    def draw(self):
        self.canv.saveState()
        self.canv.scale(self._scale, self._scale)
        renderPDF.draw(self.drawing, self.canv, 0, 0)
        self.canv.restoreState()


def inline_to_rl(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda match: f"{match.group(1)} ({match.group(2)})", text)
    text = escape(text)
    text = re.sub(r"`([^`]+)`", lambda match: f"<font face='Courier'>{escape(match.group(1))}</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", lambda match: f"<b>{match.group(1)}</b>", text)
    return text


def is_table_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)


def split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def make_image_flowable(image_path: Path, max_width: float, max_height: float = 5.9 * inch):
    suffix = image_path.suffix.lower()
    if suffix == ".svg":
        drawing = svg2rlg(str(image_path))
        if drawing is None:
            return Paragraph(f"[Could not render SVG: {image_path.name}]", None)
        return SVGFlowable(drawing, max_width)

    with PILImage.open(image_path) as img:
        width_px, height_px = img.size
    scale = min(max_width / width_px, max_height / height_px, 1.0)
    return Image(str(image_path), width=width_px * scale, height=height_px * scale)


def add_paragraph(story, text: str, style: ParagraphStyle):
    clean = text.strip()
    if clean:
        story.append(Paragraph(inline_to_rl(clean), style))
        story.append(Spacer(1, 0.08 * inch))


def add_image_block(story, image_path: Path, alt: str, max_width: float, caption_style: ParagraphStyle):
    flowable = make_image_flowable(image_path, max_width=max_width)
    story.append(flowable)
    story.append(Spacer(1, 0.06 * inch))
    if alt:
        story.append(Paragraph(escape(alt), caption_style))
        story.append(Spacer(1, 0.12 * inch))


def add_table_block(story, rows: list[list[str]], body_style: ParagraphStyle, max_width: float):
    table_data = [[Paragraph(inline_to_rl(cell), body_style) for cell in row] for row in rows]
    num_cols = max(len(row) for row in rows)
    col_width = max_width / num_cols if num_cols else max_width
    table = Table(table_data, colWidths=[col_width] * num_cols, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17324D")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#6B7A90")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.14 * inch))


def build_pdf():
    text = SOURCE.read_text(encoding="utf-8")
    lines = text.splitlines()

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#102A43"),
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportHeading1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=21,
            textColor=colors.HexColor("#16324F"),
            spaceBefore=10,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportHeading2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13.5,
            leading=17,
            textColor=colors.HexColor("#243B53"),
            spaceBefore=8,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportHeading3",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            textColor=colors.HexColor("#334E68"),
            spaceBefore=7,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=12.3,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#102A43"),
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportCaption",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=10.5,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#627D98"),
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportBullet",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=11.8,
            leftIndent=14,
            firstLineIndent=-7,
            spaceAfter=2,
            alignment=TA_LEFT,
        )
    )

    story = []
    story.append(Paragraph(escape("CLEAR-MoE Full Project Report"), styles["ReportTitle"]))
    story.append(Paragraph(escape("Rendered from FULL_PROJECT_REPORT.md with embedded figures and tables."), styles["ReportCaption"]))
    story.append(Spacer(1, 0.18 * inch))

    image_re = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")
    heading_re = re.compile(r"^(#{1,6})\s+(.*)$")
    bullet_re = re.compile(r"^([*-])\s+(.*)$")
    ordered_re = re.compile(r"^(\d+)\.\s+(.*)$")

    buffer: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            if buffer:
                add_paragraph(story, " ".join(buffer), styles["ReportBody"])
                buffer = []
            i += 1
            continue

        heading_match = heading_re.match(stripped)
        if heading_match:
            if buffer:
                add_paragraph(story, " ".join(buffer), styles["ReportBody"])
                buffer = []
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2)
            if level == 1:
                story.append(PageBreak())
                story.append(Paragraph(inline_to_rl(heading_text), styles["ReportHeading1"]))
            elif level == 2:
                story.append(Paragraph(inline_to_rl(heading_text), styles["ReportHeading2"]))
            else:
                story.append(Paragraph(inline_to_rl(heading_text), styles["ReportHeading3"]))
            i += 1
            continue

        image_match = image_re.match(stripped)
        if image_match:
            if buffer:
                add_paragraph(story, " ".join(buffer), styles["ReportBody"])
                buffer = []
            alt_text, rel_path = image_match.groups()
            image_path = (ROOT / rel_path).resolve()
            if image_path.exists():
                add_image_block(story, image_path, alt_text, max_width=7.0 * inch, caption_style=styles["ReportCaption"])
            else:
                add_paragraph(story, f"Missing image: {rel_path}", styles["ReportBody"])
            i += 1
            continue

        if i + 1 < len(lines) and "|" in stripped and is_table_separator(lines[i + 1]):
            if buffer:
                add_paragraph(story, " ".join(buffer), styles["ReportBody"])
                buffer = []
            table_rows = [split_table_row(stripped), split_table_row(lines[i + 1].strip())]
            i += 2
            while i < len(lines):
                candidate = lines[i].rstrip()
                if not candidate.strip() or "|" not in candidate:
                    break
                table_rows.append(split_table_row(candidate))
                i += 1
            add_table_block(story, table_rows, styles["ReportBody"], max_width=7.0 * inch)
            continue

        if bullet_re.match(stripped) or ordered_re.match(stripped):
            if buffer:
                add_paragraph(story, " ".join(buffer), styles["ReportBody"])
                buffer = []
            story.append(Paragraph(inline_to_rl(stripped), styles["ReportBullet"]))
            i += 1
            continue

        buffer.append(stripped)
        i += 1

    if buffer:
        add_paragraph(story, " ".join(buffer), styles["ReportBody"])

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#627D98"))
        canvas.drawRightString(doc.pagesize[0] - 0.55 * inch, 0.45 * inch, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=LETTER,
        leftMargin=0.62 * inch,
        rightMargin=0.62 * inch,
        topMargin=0.72 * inch,
        bottomMargin=0.68 * inch,
        title="CLEAR-MoE Full Project Report",
        author="GitHub Copilot",
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


if __name__ == "__main__":
    build_pdf()
    print(OUTPUT)
