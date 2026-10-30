"""Generate figures from the measured, scope-limited Worker summary."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
groups = json.loads((ROOT/'results/worker-benchmark-summary.json').read_text())['groups']
lookup = {(x['length'], x['mode'], x['chunk_rows']): x for x in groups}
out = ROOT/'docs/figures'
out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':150,'savefig.dpi':200,'axes.titleweight':'bold'})
blue, amber = '#25657d', '#bd7935'
ns = [1024,4096,8192]
x = np.arange(len(ns)); width=.30
fig, (a,b) = plt.subplots(1,2,figsize=(7.5,3.5),layout='constrained')
for shift, mode, color, chunk in [(-width/2,'dense',blue,0),(width/2,'chunked',amber,256)]:
    rows=[lookup[(n,mode,chunk)] for n in ns]
    allocated=[r['max_allocated_mib'] for r in rows]
    reserved=[r['max_reserved_mib'] for r in rows]
    a.bar(x+shift,allocated,width,color=color,label=mode.title(),alpha=.88)
    a.scatter(x+shift,reserved,marker='_',s=190,color='#262626',linewidth=1.6,zorder=4)
    y=[r['p50_perf_ms'] for r in rows]; upper=[r['p95_perf_ms']-r['p50_perf_ms'] for r in rows]
    b.errorbar(x+shift,y,yerr=[np.zeros(3),upper],fmt='o-',capsize=4,color=color,label=mode.title())
a.set_title('Peak memory during Worker calls')
a.set_ylabel('PyTorch allocator peak (MiB)')
a.set_xticks(x, ['1,024','4,096','8,192']); a.set_xlabel('Sequence length N')
a.grid(axis='y',alpha=.18);a.set_axisbelow(True)
a.plot([],[],marker='_',linestyle='None',markersize=12,markeredgewidth=1.6,color='#262626',label='Reserved peak')
a.legend(frameon=False,loc='upper left',fontsize=8)
b.set_title('Warm Worker timing: p50 to p95')
b.set_ylabel('Synchronized perf_counter span (ms)')
b.set_xticks(x,['1,024','4,096','8,192']);b.set_xlabel('Sequence length N')
b.grid(axis='y',alpha=.18);b.set_axisbelow(True);b.legend(frameon=False,loc='upper left')
fig.suptitle('Real Qwen2.5 Teacher Worker · BF16 · K=32 · allocator fraction 0.50',fontsize=12,fontweight='bold')
fig.text(.5,-.02,'Two fresh processes per mode and N; 1 warmup + 5 measured calls each. Timing is descriptive: A/A and clocks vary.',ha='center',fontsize=8)
fig.savefig(out/'worker-memory-timing.png',bbox_inches='tight',facecolor='white')
fig.savefig(out/'worker-memory-timing.pdf',bbox_inches='tight',facecolor='white')
plt.close(fig)

fig,(a,b)=plt.subplots(1,2,figsize=(7.5,3.5),layout='constrained')
for n,color in [(4096,blue),(8192,amber)]:
    rows=[lookup[(n,'chunked',c)] for c in (128,256,512)]
    a.plot([128,256,512],[r['max_allocated_mib'] for r in rows],'-o',color=color,label=f'N={n:,}')
    y=[r['p50_perf_ms'] for r in rows]; upper=[r['p95_perf_ms']-r['p50_perf_ms'] for r in rows]
    b.errorbar([128,256,512],y,yerr=[np.zeros(3),upper],fmt='o-',capsize=4,color=color,label=f'N={n:,}')
for ax in (a,b):
    ax.set_xticks([128,256,512]); ax.set_xlabel('LM-head chunk rows'); ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
a.set_title('Allocated peak');a.set_ylabel('PyTorch peak allocated (MiB)');a.legend(frameon=False)
b.set_title('Warm p50 to p95');b.set_ylabel('Synchronized perf_counter span (ms)');b.legend(frameon=False)
fig.suptitle('Chunk size trade-off in the actual Worker',fontsize=12,fontweight='bold')
fig.text(.5,-.02,'256 rows: two processes / 10 samples; 128 and 512: one process / 5 samples. Timing is descriptive.',ha='center',fontsize=8)
fig.savefig(out/'worker-chunk-tradeoff.png',bbox_inches='tight',facecolor='white')
fig.savefig(out/'worker-chunk-tradeoff.pdf',bbox_inches='tight',facecolor='white')
plt.close(fig)
