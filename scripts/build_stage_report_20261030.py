#!/usr/bin/env python3
"""Regenerate the bilingual 2026-10-30 supplement from Markdown.

Requires ReportLab and a full local Chinese TrueType/TTC font. This entry never
targets the historical technical-report files. It copies no font into the repo.
"""
from __future__ import annotations
import argparse
import hashlib
from html import escape
import json
import os
from pathlib import Path
import re
import reportlab
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import HRFlowable, LongTable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
STEM = "memory-bounded-stage-20261030"
PW, PH = A4
MARGIN = 23 * mm
WIDTH = PW - 2 * MARGIN
INK, MUTED = colors.HexColor("#17212B"), colors.HexColor("#53616E")
RULE, PALE = colors.HexColor("#B9C2CA"), colors.HexColor("#F0F3F5")
TOKEN = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))")

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def register_latin():
    windows = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    families = [
        (windows, ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf")),
        (Path("/mnt/c/Windows/Fonts"), ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf")),
        (Path("/usr/share/fonts/truetype/liberation2"), ("LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf", "LiberationSerif-Italic.ttf", "LiberationSerif-BoldItalic.ttf")),
        (Path(reportlab.__file__).resolve().parent / "fonts", ("Vera.ttf", "VeraBd.ttf", "VeraIt.ttf", "VeraBI.ttf")),
    ]
    font_dir, files = next((directory, names) for directory, names in families
                           if all((directory / name).is_file() for name in names))
    for name, file in zip(("StageLatin", "StageLatinBold", "StageLatinItalic", "StageLatinBoldItalic"), files):
        pdfmetrics.registerFont(TTFont(name, str(font_dir / file)))
    pdfmetrics.registerFontFamily("StageLatin", normal="StageLatin", bold="StageLatinBold",
                                 italic="StageLatinItalic", boldItalic="StageLatinBoldItalic")

def register_cjk(explicit):
    windir = Path(os.environ.get("WINDIR", "C:/Windows"))
    paths = [Path(explicit)] if explicit else [
        windir / "Fonts/simsun.ttc",
        Path("/mnt/c/Windows/Fonts/simsun.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
    ]
    font = next((p for p in paths if p.is_file()), None)
    if font is None:
        raise RuntimeError("Pass --cjk-font with a full local Chinese TrueType/TTC font")
    pdfmetrics.registerFont(TTFont("StageCJK", str(font), subfontIndex=0))
    heading = font.parent / "simhei.ttf"
    pdfmetrics.registerFont(TTFont("StageCJKHeading", str(heading if heading.is_file() else font), subfontIndex=0))
    pdfmetrics.registerFontFamily("StageCJK", normal="StageCJK", bold="StageCJKHeading",
                                 italic="StageCJK", boldItalic="StageCJKHeading")

def inline(text, lang):
    parts, cursor = [], 0
    size = 8.0 if lang == "zh-CN" else 8.4
    for match in TOKEN.finditer(text):
        parts.append(escape(text[cursor:match.start()]))
        token = match.group(0)
        if token.startswith("**"):
            parts.append("<b>" + escape(token[2:-2]) + "</b>")
        elif token.startswith(chr(96)):
            parts.append(f'<font name="StageLatin" size="{size}">' + escape(token[1:-1]) + "</font>")
        else:
            link = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", token)
            parts.append('<link href="' + escape(link.group(2), quote=True) + '" color="#243E54">'
                         + escape(link.group(1)) + "</link>")
        cursor = match.end()
    parts.append(escape(text[cursor:]))
    return "".join(parts)

def read_source(path):
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError("Missing Markdown metadata block")
    front, body = text[4:].split("\n---\n", 1)
    metadata = {}
    for line in front.splitlines():
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    required = ("title", "subtitle", "author", "edition", "scope", "footer")
    if any(not metadata.get(key) for key in required):
        raise ValueError("Incomplete report metadata")
    if "drive.google.com" in text or re.search(r"[A-Z]:[/\\\\]Users[/\\\\]|/home/[^/]+/", text):
        raise ValueError("Report contains an internal link or local user path")
    return metadata, body

def make_styles(lang):
    cjk = lang == "zh-CN"
    normal, bold = ("StageCJK", "StageCJKHeading") if cjk else ("StageLatin", "StageLatinBold")
    body = ParagraphStyle("Body", fontName=normal, fontSize=10.2 if cjk else 10.5,
                          leading=15.0 if cjk else 14.6, textColor=INK, spaceAfter=7,
                          wordWrap="CJK" if cjk else "LTR", splitLongWords=True)
    return {
        "body": body,
        "h2": ParagraphStyle("H2", parent=body, fontName=bold, fontSize=15, leading=20,
                             spaceBefore=2, spaceAfter=10, keepWithNext=True),
        "h3": ParagraphStyle("H3", parent=body, fontName=bold, fontSize=11.6, leading=16,
                             spaceBefore=9, spaceAfter=6, keepWithNext=True),
        "bullet": ParagraphStyle("Bullet", parent=body, leftIndent=10, bulletIndent=0, bulletFontName="StageLatin", spaceAfter=5),
        "caption": ParagraphStyle("Caption", parent=body, fontName=bold, fontSize=9.2,
                                  leading=12.3, spaceBefore=5, spaceAfter=5, keepWithNext=True),
        "cell": ParagraphStyle("Cell", parent=body, fontSize=8.6 if cjk else 8.9,
                               leading=12.2, spaceAfter=0, spaceBefore=0),
        "headcell": ParagraphStyle("HeadCell", parent=body, fontName=bold,
                                   fontSize=8.5 if cjk else 8.7, leading=11.8, spaceAfter=0),
        "code": ParagraphStyle("Code", parent=body, fontName="StageLatin", fontSize=8.1,
                               leading=11.2, leftIndent=8, rightIndent=8, backColor=PALE,
                               borderPadding=7, spaceAfter=9),
        "title": ParagraphStyle("Title", parent=body, fontName=bold,
                                fontSize=30 if cjk else 31, leading=39, spaceAfter=12),
        "subtitle": ParagraphStyle("Subtitle", parent=body, fontSize=13, leading=19, spaceAfter=16),
        "covercell": ParagraphStyle("CoverCell", parent=body, fontSize=10.5, leading=15, spaceAfter=0),
        "eyebrow": ParagraphStyle("Eyebrow", parent=body, fontName="StageLatinBold",
                                  fontSize=9.8, leading=13, spaceAfter=18),
        "small": ParagraphStyle("Small", parent=body, fontSize=9.2, leading=13.4, textColor=MUTED),
    }

def column_widths(n):
    ratios = {2:[.27,.73], 3:[.32,.34,.34], 4:[.27,.29,.22,.22],
              5:[.13,.16,.23,.23,.25], 6:[.13,.15,.18,.18,.18,.18]}.get(n, [1/n]*n)
    return [WIDTH*x for x in ratios]

def make_table(lines, lang, styles):
    rows = []
    for line in lines:
        cells = [x.strip() for x in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", x) for x in cells):
            continue
        rows.append(cells)
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Inconsistent Markdown table")
    rendered = [[Paragraph(inline(cell, lang), styles["headcell"] if i == 0 else styles["cell"])
                 for cell in row] for i,row in enumerate(rows)]
    table = LongTable(rendered, colWidths=column_widths(len(rows[0])), repeatRows=1, hAlign="LEFT")
    commands = [
        ("VALIGN",(0,0),(-1,-1),"TOP"), ("BACKGROUND",(0,0),(-1,0),PALE),
        ("LINEABOVE",(0,0),(-1,0),.8,INK), ("LINEBELOW",(0,0),(-1,0),.4,RULE),
        ("LINEBELOW",(0,-1),(-1,-1),.7,INK), ("LEFTPADDING",(0,0),(-1,-1),6),
        ("RIGHTPADDING",(0,0),(-1,-1),6), ("TOPPADDING",(0,0),(-1,-1),5),
        ("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]
    for i in range(1,len(rows)-1):
        commands.append(("LINEBELOW",(0,i),(-1,i),.2,colors.HexColor("#DCE1E5")))
    table.setStyle(TableStyle(commands))
    return [table, Spacer(1,8)]

def body_story(body, lang, styles):
    lines, story, index = body.splitlines(), [], 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
        elif line == "<!-- pagebreak -->":
            story.append(PageBreak()); index += 1
        elif line.startswith("### "):
            story.append(Paragraph(inline(line[4:],lang),styles["h3"])); index += 1
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:],lang),styles["h2"])); index += 1
        elif line.startswith("|"):
            block = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                block.append(lines[index]); index += 1
            story.extend(make_table(block,lang,styles))
        elif line.startswith(chr(96)*3):
            block = []; index += 1
            while index < len(lines) and not lines[index].strip().startswith(chr(96)*3):
                block.append(lines[index]); index += 1
            story.append(Paragraph("<br/>".join(escape(x).replace(" ","&#160;") for x in block),styles["code"]))
            index += 1
        elif line.startswith("- "):
            story.append(Paragraph(inline(line[2:],lang),styles["bullet"],bulletText="-")); index += 1
        else:
            block = [line]; index += 1
            while index < len(lines):
                nxt = lines[index].strip()
                if not nxt or nxt.startswith(("## ","### ","|","- ","<!--",chr(96)*3)):
                    break
                block.append(nxt); index += 1
            text = " ".join(block)
            key = "caption" if re.match(r"^(?:Table|表)\s*\d",text) else "body"
            story.append(Paragraph(inline(text,lang),styles[key]))
    return story

def cover(meta, lang, styles):
    cjk = lang == "zh-CN"
    labels = ["作者","版本","范围"] if cjk else ["Author","Edition","Scope"]
    rows = [[Paragraph(a,styles["covercell"]),Paragraph(inline(meta[b],lang),styles["covercell"])]
            for a,b in zip(labels,("author","edition","scope"))]
    table = Table(rows,colWidths=[86,WIDTH-86],hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),0),
        ("RIGHTPADDING",(0,0),(-1,-1),8),("TOPPADDING",(0,0),(-1,-1),3),
        ("BOTTOMPADDING",(0,0),(-1,-1),4),
    ]))
    return [
        Spacer(1,33*mm), Paragraph("AI INFRASTRUCTURE / ENGINEERING SUPPLEMENT",styles["eyebrow"]),
        Paragraph(inline(meta["title"],lang),styles["title"]),
        Paragraph(inline(meta["subtitle"],lang),styles["subtitle"]),
        HRFlowable(width="100%",thickness=.7,color=INK,spaceAfter=13), table, Spacer(1,16),
        PageBreak(),
    ]

def decorate(meta, lang):
    def draw(canvas, doc):
        canvas.saveState()
        cjk = lang == "zh-CN"
        if doc.page > 1:
            canvas.setFont("StageCJKHeading" if cjk else "StageLatin",8.8)
            canvas.setFillColor(MUTED)
            canvas.drawString(MARGIN,PH-12*mm,meta["title"])
            canvas.setStrokeColor(RULE); canvas.setLineWidth(.3)
            canvas.line(MARGIN,PH-14.5*mm,PW-MARGIN,PH-14.5*mm)
        canvas.setFont("StageCJK" if cjk else "StageLatin",8.3); canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN,11.5*mm,meta["footer"])
        canvas.drawRightString(PW-MARGIN,11.5*mm,str(doc.page))
        canvas.restoreState()
    return draw

def build(source, target, lang):
    meta, body = read_source(source)
    styles = make_styles(lang)
    target.parent.mkdir(parents=True,exist_ok=True)
    doc = SimpleDocTemplate(str(target),pagesize=A4,leftMargin=MARGIN,rightMargin=MARGIN,
                            topMargin=21*mm,bottomMargin=21*mm,title=meta["title"],author=meta["author"],
                            subject=meta["scope"],creator="ReportLab Markdown report generator",
                            pageCompression=1,invariant=1)
    callback = decorate(meta,lang)
    doc.build(cover(meta,lang,styles)+body_story(body,lang,styles),onFirstPage=callback,onLaterPages=callback)
    return {"source":source.name,"source_sha256":sha256(source),"pdf":target.name,
            "pdf_sha256":sha256(target),"bytes":target.stat().st_size}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir",type=Path,default=ROOT/"docs/reports")
    p.add_argument("--output-dir",type=Path,default=ROOT/"docs/reports")
    p.add_argument("--languages",nargs="+",choices=("en","zh-CN"),default=["en","zh-CN"])
    p.add_argument("--cjk-font")
    args = p.parse_args()
    register_latin()
    if "zh-CN" in args.languages:
        register_cjk(args.cjk_font)
    results = [build(args.source_dir/f"{STEM}.{lang}.md",args.output_dir/f"{STEM}.{lang}.pdf",lang)
               for lang in args.languages]
    print(json.dumps({"reportlab":reportlab.Version,"outputs":results},ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()