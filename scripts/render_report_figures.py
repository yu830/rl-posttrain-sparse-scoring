"""Render the monochrome figures used in the bilingual technical report.

Requires Matplotlib 3.11.1. Pass --cjk-font if SimHei is not available at the
Windows font path. The benchmark values come from the checked-in summary.
"""
from pathlib import Path
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'figures'

LABELS = {
    'en': {
        'input': 'Student candidate IDs  ·  padded inputs and attention mask',
        'worker': 'TeacherFSDPWorker',
        'worker_note': 'Qwen2 with FSDP2  ·  normal forward and module hooks',
        'backbone': 'Transformer backbone',
        'hidden': 'hidden states',
        'dense': 'A0   Dense baseline',
        'dense_a': 'Full LM head readout',
        'dense_b': 'Materialize full N × V logits',
        'chunk': 'A1   Memory-bounded readout',
        'chunk_a': 'Row-chunked LM head',
        'chunk_b': 'Log-sum-exp, candidate gather, teacher Top-K',
        'chunk_c': 'No full N × V logits retained',
        'output': 'Compact teacher outputs',
        'output_note': 'candidate logprobs  ·  teacher Top-K IDs  ·  nested offsets',
        'validate': 'Worker-to-student validation',
        'validate_note': 'reverse-KL consumer  ·  gradients  ·  restricted RMSNorm update',
        'alloc': 'Peak allocated (MiB)', 'rows': 'Projected rows, N',
        'dense_legend': 'Dense', 'chunk_legend': 'Chunked',
    },
    'zh-CN': {
        'input': '学生候选 ID  ·  填充输入与 attention mask',
        'worker': 'TeacherFSDPWorker',
        'worker_note': 'FSDP2 封装的 Qwen2  ·  正常 forward 与模块 hooks',
        'backbone': 'Transformer 主干网络',
        'hidden': '隐藏状态',
        'dense': 'A0   稠密基线',
        'dense_a': '完整 LM Head 读出',
        'dense_b': '生成完整 N × V logits',
        'chunk': 'A1   显存受控读出',
        'chunk_a': '按行分块执行 LM Head',
        'chunk_b': '逐块 log-sum-exp、候选 gather 与教师 Top-K',
        'chunk_c': '不保留完整 N × V logits',
        'output': '紧凑教师输出',
        'output_note': '候选 logprob  ·  教师 Top-K ID  ·  嵌套张量 offset',
        'validate': 'Worker 到 Student 的验证',
        'validate_note': '反向 KL 消费端  ·  学生梯度  ·  受限的 RMSNorm 更新',
        'alloc': 'Peak allocated（MiB）', 'rows': '投影行数 N',
        'dense_legend': 'Dense', 'chunk_legend': 'Chunked',
    },
}

def text(ax, x, y, value, font, *, size=10, weight='normal', align='center', color='#202020'):
    ax.text(x, y, value, transform=ax.transAxes, ha=align, va='center',
            fontproperties=font, fontsize=size, fontweight=weight, color=color)

def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes,
                 arrowstyle='-|>', mutation_scale=8, linewidth=.8, color='#404040'))

def segment(ax, x1, y1, x2, y2, *, color='#777777', width=.7):
    ax.add_line(Line2D([x1, x2], [y1, y2], transform=ax.transAxes,
                       linewidth=width, color=color))

def save_figure(fig, stem):
    for ext in ('pdf', 'svg'):
        path=OUT/f'{stem}.{ext}'
        fig.savefig(path,facecolor='white')
        if ext=='svg':
            content=path.read_text(encoding='utf-8')
            path.write_bytes(('\n'.join(line.rstrip() for line in content.splitlines())+'\n').encode('utf-8'))

def draw_system(lang, font):
    d=LABELS[lang]
    fig, ax=plt.subplots(figsize=(6.5, 5.65))
    fig.subplots_adjust(left=.02, right=.98, top=.99, bottom=.01)
    ax.set_axis_off()
    text(ax,.5,.970,d['input'],font,size=10.0,weight='bold')
    arrow(ax,(.5,.941),(.5,.905))
    text(ax,.5,.867,d['worker'],font,size=11.1,weight='bold')
    text(ax,.5,.824,d['worker_note'],font,size=9.1,color='#505050')
    arrow(ax,(.5,.798),(.5,.763))
    text(ax,.5,.728,d['backbone'],font,size=10.7,weight='bold')
    text(ax,.5,.689,d['hidden'],font,size=9.2,color='#505050')
    segment(ax,.5,.668,.5,.643)
    segment(ax,.255,.643,.745,.643)
    arrow(ax,(.255,.643),(.255,.606))
    arrow(ax,(.745,.643),(.745,.606))
    ax.add_patch(Rectangle((.512,.377),.465,.208,transform=ax.transAxes,
                           facecolor='#f3f3f3',edgecolor='none',zorder=0))
    segment(ax,.025,.585,.488,.585,color='#555555',width=1)
    segment(ax,.512,.585,.977,.585,color='#202020',width=1)
    text(ax,.043,.550,d['dense'],font,size=10.1,weight='bold',align='left')
    text(ax,.043,.498,d['dense_a'],font,size=9.2,align='left')
    text(ax,.043,.450,d['dense_b'],font,size=9.2,align='left')
    text(ax,.530,.550,d['chunk'],font,size=10.1,weight='bold',align='left')
    text(ax,.530,.498,d['chunk_a'],font,size=9.2,align='left')
    text(ax,.530,.450,d['chunk_b'],font,size=8.7,align='left')
    text(ax,.530,.402,d['chunk_c'],font,size=9.1,align='left')
    segment(ax,.255,.355,.255,.332)
    segment(ax,.745,.355,.745,.332)
    segment(ax,.255,.332,.745,.332)
    arrow(ax,(.5,.332),(.5,.295))
    text(ax,.5,.258,d['output'],font,size=10.5,weight='bold')
    text(ax,.5,.216,d['output_note'],font,size=9.0,color='#505050')
    arrow(ax,(.5,.189),(.5,.157))
    text(ax,.5,.119,d['validate'],font,size=10.3,weight='bold')
    text(ax,.5,.077,d['validate_note'],font,size=8.8,color='#505050')
    save_figure(fig, f'report-system.{lang}')
    plt.close(fig)

def draw_memory(lang, font, groups):
    d=LABELS[lang]
    x=[0,1,2]
    lengths=[1024,4096,8192]
    dense=[groups[(n,'dense')]['max_allocated_mib'] for n in lengths]
    chunk=[groups[(n,'chunked')]['max_allocated_mib'] for n in lengths]
    fig, ax=plt.subplots(figsize=(6.5,3.10))
    fig.subplots_adjust(left=.13,right=.98,bottom=.19,top=.88)
    ax.set_axisbelow(True)
    ax.grid(axis='y',color='#dedede',linewidth=.6)
    ax.bar([a-.18 for a in x],dense,.34,color='#b6b6b6',edgecolor='#696969',linewidth=.4,label=d['dense_legend'])
    ax.bar([a+.18 for a in x],chunk,.34,color='#333333',label=d['chunk_legend'])
    for a,b in zip(x,dense): ax.text(a-.18,b+85,f'{b:,.0f}',ha='center',va='bottom',fontsize=8.8,color='#333333')
    for a,b in zip(x,chunk): ax.text(a+.18,b+85,f'{b:,.0f}',ha='center',va='bottom',fontsize=8.8,color='#333333')
    ax.set_xticks(x, [f'{n:,}' for n in lengths])
    ax.set_ylim(0, 4250)
    ax.set_yticks([0,1000,2000,3000,4000],['0','1,000','2,000','3,000','4,000'])
    ax.set_ylabel(d['alloc'],fontproperties=font,fontsize=9.5,labelpad=7)
    ax.set_xlabel(d['rows'],fontproperties=font,fontsize=9.5,labelpad=5)
    ax.tick_params(axis='both',labelsize=9,length=0,pad=5,colors='#333333')
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color('#666666')
    ax.legend(loc='upper left',bbox_to_anchor=(0,1.15),ncol=2,frameon=False,
              prop=font_manager.FontProperties(fname=font.get_file(),size=9) if font.get_file() else {'size':9},
              handlelength=1.25,columnspacing=1.7)
    save_figure(fig, f'report-memory.{lang}')
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cjk-font',type=Path)
    args=parser.parse_args()
    default_font=(Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts/simhei.ttf'
                  if os.name=='nt' else Path('/mnt/c/Windows/Fonts/simhei.ttf'))
    cjk_path=args.cjk_font or default_font
    font_manager.fontManager.addfont(str(cjk_path))
    cjk_name=font_manager.FontProperties(fname=str(cjk_path)).get_name()
    cjk_font=font_manager.FontProperties(family=['DejaVu Sans',cjk_name])
    en_font=font_manager.FontProperties(family='DejaVu Sans')
    plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False})
    data=json.loads((ROOT/'results/worker-benchmark-summary.json').read_text(encoding='utf-8'))
    groups={(g['length'],g['mode']):g for g in data['groups']
            if g['mode']=='dense' or g['chunk_rows']==256}
    for lang,font in [('en',en_font),('zh-CN',cjk_font)]:
        draw_system(lang,font)
        draw_memory(lang,font,groups)
    print('Rendered monochrome report figures from the checked-in benchmark summary.')

if __name__=='__main__':
    main()
