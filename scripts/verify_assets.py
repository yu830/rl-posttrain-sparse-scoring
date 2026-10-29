#!/usr/bin/env python3
"""Check document links, paired assets, topology, bibliography and PDF metadata."""
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree as ET
import re
import hashlib
import json
import platform
import shutil
import subprocess
import PIL
import sys
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
errors = []
checks = []

def check(condition, message):
    (checks if condition else errors).append(message)

def run(*args):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        errors.append(f"Command failed: {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout

def read(path):
    return (ROOT / path).read_text(encoding='utf8')

required = ['README.md', 'README.zh-CN.md', 'docs/architecture/topology.json',
            'scripts/render_architecture.py', 'scripts/build_assets.sh', 'assets/fonts/project-cjk-serif.ttf', 'assets/fonts/OFL.txt', 'docs/build-environment.md', 'docs/build-environment.lock.json', 'docs/requirements-docs.txt',
            'docs/reports/common-equations.tex', 'docs/reports/references.bib']
for lang in ('en', 'zh-CN'):
    required += [f'docs/architecture/architecture.{lang}.{ext}' for ext in ('svg','png')]
    required += [f'docs/reports/technical-report.{lang}.{ext}' for ext in ('tex','pdf')]
    required += [f'assets/report-cover.{lang}.png', f'docs/benchmark-methodology.{lang}.md']
for p in required:
    check((ROOT / p).is_file(), f'required asset: {p}')
if errors:
    print('\n'.join('FAIL '+e for e in errors)); sys.exit(1)

check(hashlib.sha256((ROOT/'assets/fonts/project-cjk-serif.ttf').read_bytes()).hexdigest() == '56bd5d7eb730ed49aa33ff19e2bbbb8dbb78ddd656797013ad88f63814f6016d', 'bundled font hash')

if '--locked-environment' in sys.argv:
    lock = json.loads(read('docs/build-environment.lock.json'))
    check(platform.python_version() == lock['python']['version'], 'locked Python version')
    check(PIL.__version__ == lock['python']['pillow'], 'locked Pillow version')
    for name, info in lock['executables'].items():
        path = shutil.which(name)
        check(bool(path) and hashlib.sha256(Path(path).read_bytes()).hexdigest() == info['sha256'], f'locked executable: {name}')
    for name, digest in lock['tex_packages_sha256'].items():
        path = run('kpsewhich', name).strip()
        check(bool(path) and hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, f'locked TeX package: {name}')

for md in ROOT.rglob('*.md'):
    if 'build' in md.parts:
        continue
    text = md.read_text(encoding='utf8')
    for target in re.findall(r'\]\(([^)]+)\)', text):
        url = target.split()[0]
        parsed = urlsplit(url)
        if parsed.scheme or url.startswith('#'):
            continue
        resolved = (md.parent / unquote(parsed.path)).resolve()
        check(resolved.is_file() and (resolved == ROOT or ROOT in resolved.parents), f'link in {md.relative_to(ROOT)}: {url}')

switch = '[English](./README.md) | [简体中文](./README.zh-CN.md)'
for name, lang in [('README.md','en'), ('README.zh-CN.md','zh-CN')]:
    s = read(name)
    check(s.startswith(switch), f'{name}: language switch')
    for p in [f'./docs/architecture/architecture.{lang}.png',f'./docs/reports/technical-report.{lang}.pdf',f'./assets/report-cover.{lang}.png']:
        check(p in s, f'{name}: matching {p}')
    other = 'zh-CN' if lang == 'en' else 'en'
    check(f'./docs/reports/technical-report.{other}.pdf' not in s, f'{name}: no wrong-language report link')

tex = {lang:read(f'docs/reports/technical-report.{lang}.tex') for lang in ('en','zh-CN')}
def cite_keys(s):
    return {k.strip() for group in re.findall(r'\\cite\{([^}]+)\}',s) for k in group.split(',')}
refs = set(re.findall(r'@\w+\s*\{\s*([^,]+),',read('docs/reports/references.bib')))
check(cite_keys(tex['en']) == cite_keys(tex['zh-CN']), 'reports cite the same keys')
check(cite_keys(tex['en']) <= refs, 'all cited keys exist in bibliography')
for macro in ('ScoreEquations','KLEquations','Decomposition','CounterExample','CostEquations'):
    check(all(s.count('\\'+macro)==1 for s in tex.values()), f'shared equation block: {macro}')

geometries=[]
for lang in ('en','zh-CN'):
    path=ROOT/f'docs/architecture/architecture.{lang}.svg'
    root=ET.parse(path).getroot()
    geom=[(el.tag.rsplit('}',1)[-1],tuple(sorted((k,v) for k,v in el.attrib.items() if k not in ('aria-labelledby',)))) for el in root.iter() if el.tag.rsplit('}',1)[-1] in ('rect','polyline')]
    geometries.append(geom)
    with Image.open(ROOT/f'docs/architecture/architecture.{lang}.png') as im:
        check(im.size==(1800,1500), f'{lang}: architecture PNG resolution')
    with Image.open(ROOT/f'assets/report-cover.{lang}.png') as im:
        check(im.width>=900 and im.height>=1300, f'{lang}: cover resolution')
check(geometries[0]==geometries[1], 'SVG geometry and edges match across languages')

for lang in ('en','zh-CN'):
    pdf=f'docs/reports/technical-report.{lang}.pdf'
    meta=run('pdfinfo',pdf)
    check(bool(re.search(r'^Author:\s+Jiacheng Yu\s*$',meta,re.M)), f'{lang}: PDF author metadata')
    m=re.search(r'^Pages:\s+(\d+)',meta,re.M)
    check(bool(m) and int(m.group(1))>=2, f'{lang}: readable multipage PDF')
    text=run('pdftotext',pdf,'-')
    review='Author reviewed the c1ab1f4 report; AI assisted' if lang=='en' else '作者已审核 c1ab1f4 版报告；AI 辅助'
    check(review in text, f'{lang}: version-scoped author review and AI assistance')
    for pattern in (r'drive\.google\.com',r'AIINFRA-POSTTRAIN-',r'\bhandoff/',r'\bevidence/',r'AI-assisted engineering writer for'):
        check(not re.search(pattern,text,re.I), f'{lang}: no internal marker {pattern}')
    log=ROOT/f'build/{lang}/technical-report.{lang}.log'
    if log.is_file():
        contents=log.read_text(encoding='utf8',errors='replace')
        bad=r'Overfull \\hbox|Missing character:|LaTeX Font Warning|There were undefined (?:references|citations)|Citation .* undefined'
        check(not re.search(bad,contents), f'{lang}: no missing glyphs, overfull boxes or unresolved references')

for p in ROOT.rglob('*'):
    if not p.is_file() or 'build' in p.parts or p.suffix not in ('.md','.tex','.bib','.json','.svg'):
        continue
    s=p.read_text(encoding='utf8')
    for pattern in (r'drive\.google\.com',r'AIINFRA-POSTTRAIN-',r'\bhandoff/',r'\bevidence/',r'AI-assisted engineering writer for'):
        check(not re.search(pattern,s,re.I), f'{p.relative_to(ROOT)}: no internal marker {pattern}')
print(f'{len(checks)} checks passed; {len(errors)} failed')
for e in errors: print('FAIL',e)
sys.exit(bool(errors))
