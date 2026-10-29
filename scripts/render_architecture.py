#!/usr/bin/env python3
"""Render both localizations from the same editable topology."""
import argparse
import json
import os
from pathlib import Path
from xml.sax.saxutils import escape
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs/architecture/topology.json'
OUT = ROOT / 'docs/architecture'
COLORS = {'existing': ('#eaf0f6', '#34516a'), 'proposed': ('#fff9ee', '#94611e'), 'prior': ('#f2eff7', '#685780')}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--font', default=os.environ.get('CJK_FONT', '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'))
    args = parser.parse_args()
    font = Path(args.font)
    if not font.is_file():
        raise SystemExit('Set CJK_FONT to an installed Noto Serif CJK font file: ' + str(font))
    data = json.loads(DATA.read_text(encoding='utf8'))
    for lang in ('en', 'zh-CN'):
        W, H = data['width'], data['height']
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="title desc">',
               f'<title id="title">{escape(data["title"][lang])}</title>',
               f'<desc id="desc">{escape(data["subtitle"][lang])}</desc>',
               '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#3d5263"/></marker></defs>',
               f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
        im = Image.new('RGB', (W, H), 'white')
        draw = ImageDraw.Draw(im)
        header = ImageFont.truetype(str(font), 46)
        title = ImageFont.truetype(str(font), 30)
        body = ImageFont.truetype(str(font), 25)
        small = ImageFont.truetype(str(font), 23)
        def text(x, y, value, size, color, face, weight='normal'):
            svg.append(f'<text x="{x}" y="{y}" font-family="Noto Serif SC, Noto Serif CJK SC, serif" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(value)}</text>')
            draw.text((x, y-size), value, font=face, fill=color)
        text(70, 68, data['title'][lang], 46, '#172d40', header, 'bold')
        text(70, 117, data['subtitle'][lang], 25, '#576a79', body)
        for edge in data['edges']:
            pts=edge['points']; points=' '.join(f'{x},{y}' for x,y in pts)
            dash=' stroke-dasharray="12 8"' if edge['kind']=='proposed' else ''
            svg.append(f'<polyline points="{points}" fill="none" stroke="#3d5263" stroke-width="4"{dash} marker-end="url(#arrow)"/>')
            # Raster arrows use the exact same routing; dash style signals proposed paths.
            for a,b in zip(pts,pts[1:]):
                if edge['kind']=='proposed':
                    import math
                    length=math.hypot(b[0]-a[0],b[1]-a[1]); steps=max(1,int(length//20))
                    for i in range(0,steps,2):
                        s=i/steps; t=min((i+1)/steps,1)
                        draw.line((a[0]+(b[0]-a[0])*s,a[1]+(b[1]-a[1])*s,a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t),fill='#3d5263',width=4)
                else: draw.line((a,b),fill='#3d5263',width=4)
            x,y=pts[-1]; px,py=pts[-2]; dx=x-px;dy=y-py
            if abs(dx)>abs(dy): tri=[(x,y),(x-14*(1 if dx>0 else -1),y-8),(x-14*(1 if dx>0 else -1),y+8)]
            else: tri=[(x,y),(x-8,y-14*(1 if dy>0 else -1)),(x+8,y-14*(1 if dy>0 else -1))]
            draw.polygon(tri,fill='#3d5263')
        for n in data['nodes']:
            x,y,w,h=n['x'],n['y'],n['w'],n['h']; fill,stroke=COLORS[n['kind']]
            dash=' stroke-dasharray="12 8"' if n['kind']=='proposed' else ''
            svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="3"{dash}/>')
            draw.rounded_rectangle((x,y,x+w,y+h),radius=14,fill=fill,outline=stroke,width=3)
            for i,line in enumerate(n[lang]):
                size = 30 if i == 0 else 25
                face = ImageFont.truetype(str(font), size)
                while draw.textlength(line, font=face) > w - 40 and size > 20:
                    size -= 1
                    face = ImageFont.truetype(str(font), size)
                if draw.textlength(line, font=face) > w - 40:
                    raise ValueError(f"Label exceeds box: {lang} {n['id']}: {line}")
                text(x+20,y+45+i*38,line,size,stroke if i==0 else '#233847',face,'bold' if i==0 else 'normal')
        for i,(kind,label) in enumerate(zip(('existing','proposed','prior'), data['legend'][lang])):
            x=70+i*520; y=1455; fill,stroke=COLORS[kind]
            svg.append(f'<rect x="{x}" y="{y-24}" width="35" height="25" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
            draw.rectangle((x,y-24,x+35,y+1),fill=fill,outline=stroke,width=2)
            text(x+48,y,label,23,'#233847',small)
        svg.append('</svg>')
        (OUT/f'architecture.{lang}.svg').write_text('\n'.join(svg)+'\n',encoding='utf8')
        im.save(OUT/f'architecture.{lang}.png', optimize=True)
        print('rendered',lang)

if __name__ == '__main__':
    main()
