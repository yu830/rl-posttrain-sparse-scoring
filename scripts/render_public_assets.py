"""Generate the editable system diagrams and results chart from project data."""
from pathlib import Path
from html import escape
import json

ROOT = Path(__file__).resolve().parents[1]
BLUE = '#163b63'
INK = '#182330'
MUTED = '#596675'
LINE = '#cbd3dc'
PALE = '#f4f7fa'

labels = {
 'en': {
  'eyebrow':'SYSTEM OVERVIEW', 'title':'Teacher scoring, from input to validation',
  'input':'01  INPUT', 'in1':'Student candidate token IDs', 'in2':'Padded input IDs  /  attention mask',
  'worker':'02  TEACHER WORKER', 'w1':'TeacherFSDPWorker', 'w2':'FSDP2-wrapped Qwen2  /  normal forward lifecycle',
  'backbone':'03  BACKBONE', 'b1':'Transformer backbone', 'b2':'Hidden states',
  'branch':'04  READOUT', 'a0':'A0  Dense baseline', 'a01':'Full LM head readout', 'a02':'Materialize full N × V logits',
  'a1':'A1  Memory-bounded readout', 'a11':'Row-chunked LM head', 'a12':'Per-chunk logsumexp  /  candidate gather', 'a13':'Teacher Top-K  /  no full N × V logits retained',
  'out':'05  OUTPUT', 'o1':'Compact teacher outputs', 'o2':'Candidate logprobs  /  teacher Top-K IDs', 'o3':'Nested offsets  /  mask-aligned rows',
  'val':'06  VALIDATION', 'v1':'Worker-to-student validation path', 'v2':'Reverse-KL consumer  /  student gradients', 'v3':'Restricted final RMSNorm update',
 },
 'zh-CN': {
  'eyebrow':'系统概览', 'title':'从输入到验证的教师评分路径',
  'input':'01  输入', 'in1':'学生候选 token ID', 'in2':'填充后的输入 ID  /  attention mask',
  'worker':'02  教师工作进程', 'w1':'TeacherFSDPWorker', 'w2':'FSDP2 封装的 Qwen2  /  保留正常 forward 生命周期',
  'backbone':'03  主干网络', 'b1':'Transformer 主干', 'b2':'隐藏状态',
  'branch':'04  输出头', 'a0':'A0  稠密基线', 'a01':'完整 LM Head 读出', 'a02':'生成完整 N × V logits',
  'a1':'A1  显存受控读出', 'a11':'按行分块执行 LM Head', 'a12':'逐块 logsumexp  /  收集候选分数', 'a13':'教师 Top-K  /  不保留完整 N × V logits',
  'out':'05  输出', 'o1':'紧凑教师输出', 'o2':'候选 logprob  /  教师 Top-K ID', 'o3':'嵌套张量 offset  /  与 mask 对齐的行',
  'val':'06  验证', 'v1':'Worker 到 Student 的验证路径', 'v2':'反向 KL 消费端  /  学生梯度', 'v3':'受限的最终 RMSNorm 更新',
 }
}

def t(x, y, s, size=29, color=INK, weight=400, anchor='start', spacing=0):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}" letter-spacing="{spacing}">{escape(s)}</text>'

def line(x1,y1,x2,y2,color=LINE,width=2):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>'

def diagram(lang):
    d=labels[lang]
    p=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1440" viewBox="0 0 1600 1440" role="img" aria-label="{escape(d["title"])}">',
       '<rect width="1600" height="1440" fill="#ffffff"/>',
       '<g font-family="Arial, Microsoft YaHei, Noto Sans CJK SC, sans-serif">',
       t(90,70,d['eyebrow'],19,BLUE,700,spacing=2),t(90,120,d['title'],38,INK,700),
       line(90,148,1510,148,INK,2)]
    # Rows share a single central reading axis, with short labels in the left margin.
    p += [t(90,211,d['input'],19,BLUE,700),t(800,209,d['in1'],31,INK,700,'middle'),t(800,250,d['in2'],26,MUTED,400,'middle'),
          line(360,281,1240,281),line(800,282,800,316,BLUE,2),'<path d="M793 309 L800 318 L807 309" fill="none" stroke="#163b63" stroke-width="2"/>',
          t(90,360,d['worker'],19,BLUE,700),t(800,359,d['w1'],31,INK,700,'middle'),t(800,400,d['w2'],26,MUTED,400,'middle'),
          line(360,430,1240,430),line(800,431,800,465,BLUE,2),'<path d="M793 458 L800 467 L807 458" fill="none" stroke="#163b63" stroke-width="2"/>',
          t(90,509,d['backbone'],19,BLUE,700),t(800,507,d['b1'],31,INK,700,'middle'),t(800,547,d['b2'],26,MUTED,400,'middle'),
          '<path d="M800 570 V600 M430 600 H1170 M430 600 V635 M1170 600 V635" fill="none" stroke="#163b63" stroke-width="2"/>',
          '<path d="M423 628 L430 637 L437 628 M1163 628 L1170 637 L1177 628" fill="none" stroke="#163b63" stroke-width="2"/>',
          t(90,624,d['branch'],19,BLUE,700),
          '<rect x="832" y="657" width="600" height="226" fill="#f4f7fa"/>',
          line(168,657,768,657,'#9aa7b5',3),line(832,657,1432,657,BLUE,3),
          t(194,711,d['a0'],30,INK,700),t(194,765,d['a01'],25,MUTED),t(194,808,d['a02'],25,MUTED),
          t(858,711,d['a1'],30,BLUE,700),t(858,765,d['a11'],25,INK),t(858,808,d['a12'],25,INK),t(858,851,d['a13'],25,INK),
          '<path d="M430 886 V915 H800 M1170 886 V915 H800 M800 915 V947" fill="none" stroke="#163b63" stroke-width="2"/>',
          '<path d="M793 940 L800 949 L807 940" fill="none" stroke="#163b63" stroke-width="2"/>',
          t(90,997,d['out'],19,BLUE,700),t(800,995,d['o1'],31,INK,700,'middle'),t(800,1036,d['o2'],26,MUTED,400,'middle'),t(800,1076,d['o3'],26,MUTED,400,'middle'),
          line(360,1108,1240,1108),line(800,1109,800,1145,BLUE,2),'<path d="M793 1138 L800 1147 L807 1138" fill="none" stroke="#163b63" stroke-width="2"/>',
          t(90,1197,d['val'],19,BLUE,700),t(800,1195,d['v1'],31,INK,700,'middle'),t(800,1237,d['v2'],26,MUTED,400,'middle'),t(800,1277,d['v3'],26,MUTED,400,'middle'),
          line(90,1340,1510,1340,INK,2),t(90,1381,'MEMORY-BOUNDED TEACHER SCORING',17,MUTED,700,spacing=1),
          '</g></svg>']
    return '\n'.join(p)

def chart():
    data=json.loads((ROOT/'results/worker-benchmark-summary.json').read_text(encoding='utf-8'))
    groups={(g['length'],g['mode']):g for g in data['groups'] if g['mode']=='dense' or g['chunk_rows']==256}
    p=['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="650" viewBox="0 0 1600 650" role="img" aria-label="Worker allocation and timing results">',
       '<rect width="1600" height="650" fill="#ffffff"/>','<g font-family="Arial, sans-serif">',
       t(55,51,'QWEN2.5-0.5B  /  BF16  /  K=32  /  FSDP2 OFFLOAD  /  FRACTION 0.50',18,BLUE,700,spacing=0.4),
       t(55,104,'Peak allocated',31,INK,700), t(55,138,'MiB per Worker call',22,MUTED),
       t(855,104,'Warm Worker span',31,INK,700), t(855,138,'p50 to p95, milliseconds',22,MUTED),
       '<rect x="1135" y="46" width="22" height="22" fill="#aebac6"/>',t(1168,64,'Dense',21,MUTED),
       f'<rect x="1280" y="46" width="22" height="22" fill="{BLUE}"/>',t(1313,64,'Chunked',21,MUTED)]
    for v in (0,1000,2000,3000,4000):
        y=520-v*0.082
        p += [line(115,y,730,y,'#e3e8ed',1),t(99,y+7,f'{v:,}',19,MUTED,anchor='end')]
    for v in (0,250,500,750,1000):
        y=520-v*.328
        p += [line(915,y,1530,y,'#e3e8ed',1),t(899,y+7,f'{v:,}',19,MUTED,anchor='end')]
    for idx,n in enumerate((1024,4096,8192)):
        x=220+idx*195
        for offset,mode,color in ((-32,'dense','#aebac6'),(32,'chunked',BLUE)):
            g=groups[n,mode]; h=g['max_allocated_mib']*.082
            p += [f'<rect x="{x+offset-25}" y="{520-h:.1f}" width="50" height="{h:.1f}" fill="{color}"/>']
        p += [t(x,562,f'{n:,}',22,MUTED,anchor='middle')]
        x2=1020+idx*195
        for off,mode,color in ((-17,'dense','#8593a2'),(17,'chunked',BLUE)):
            g=groups[n,mode]; y50=520-g['p50_perf_ms']*.328; y95=520-g['p95_perf_ms']*.328
            p += [line(x2+off,y50,x2+off,y95,color,4),f'<circle cx="{x2+off}" cy="{y50:.1f}" r="7" fill="{color}"/>',line(x2+off-7,y95,x2+off+7,y95,color,3)]
        p += [t(x2,562,f'{n:,}',22,MUTED,anchor='middle')]
    p += [line(55,598,1530,598,LINE,1),t(55,630,'N = projected rows  ·  dots: p50  ·  caps: p95  ·  2 processes / 10 calls per mode  ·  chunk rows 256',19,MUTED),'</g></svg>']
    return '\n'.join(p)

out=ROOT/'docs/architecture'
for lang in labels:
    (out/f'system-overview.{lang}.svg').write_text(diagram(lang),encoding='utf-8')
(ROOT/'docs/figures/worker-summary.svg').write_text(chart(),encoding='utf-8')
print('Wrote editable SVG source assets.')
