"""Report figures from the frozen CPU replay; no model calls or policy changes."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_accuracy_cost import RELEASE,read_archive,jsonl

COLORS={'adaptive':'#0072B2','vote':'#D55E00','selector':'#009E73','chosen':'#CC79A7','base':'#222222'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--skill-dir',type=Path)
    a=p.parse_args()
    if a.skill_dir:
        sys.path.insert(0,str(a.skill_dir/'scripts'))
        from setup_style import setup_style
        setup_style(journal='general',lang='en',use_sciplots=False)
    plt.rcParams.update({'figure.constrained_layout.use':False,'figure.autolayout':False,'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':9,'axes.titlesize':10,
                         'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
                         'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
                         'svg.fonttype':'none','savefig.dpi':300})
    s=json.loads((RELEASE/'summary.json').read_text());qa={}
    def export(fig,name):
        if a.skill_dir:
            from visual_qa import audit_layout
            qa[name]=audit_layout(fig)
        for ext in ('png','pdf','svg'):
            kw={'metadata':{'Date':None}} if ext=='svg' else {'metadata':{'CreationDate':None,'ModDate':None}} if ext=='pdf' else {}
            fig.savefig(RELEASE/f'{name}.{ext}',dpi=300,**kw)
            if ext == 'svg':
                path = RELEASE/f'{name}.{ext}'
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        from PIL import Image
        with Image.open(RELEASE/f'{name}.png') as im:im.convert('L').save(RELEASE/f'{name}_grayscale.png')
        plt.close(fig)
    for role in ('test','confirm'):
        table=s['roles'][role]['strategies']
        fig,ax=plt.subplots(figsize=(7.2,4.5))
        fig.subplots_adjust(left=.105,right=.97,top=.86,bottom=.22)
        ax.axhline(0,color='#AAAAAA',linewidth=.7)
        seen=set()
        # Chosen and theta0,tau0.8 are the same policy: plot only once.
        for name,r in table.items():
            if name=='theta0.0_tau0.8_selector':continue
            if name=='chosen':family,marker,label='chosen','*','Frozen product'
            elif name.startswith('theta'):family,marker,label='adaptive','o','Gated / early-stop selection'
            elif name.startswith('fixed'):
                family='vote' if name.endswith('vote') else 'selector'
                marker='v' if family=='vote' else 's';label='Fixed-context vote' if family=='vote' else 'Fixed-context selector'
            else:family,marker,label='base','D','Base / retrieval-only'
            d=r['gain_vs_b_bm25'];x=r['calls_mean'];y=d['delta_pp']
            ax.scatter(x,y,s=95 if name=='chosen' else 24 if name.startswith('theta') else 32,
                       marker=marker,c=COLORS[family],label=label if label not in seen else None,
                       linewidths=.4,edgecolors='white',zorder=4)
            seen.add(label)
        # Main comparisons have paired 95% CIs. Others remain visible without implying certainty.
        annotate={'c1':('C1',(9,0)),'b_bm25':('B+BM25',(6,-16)),
                  'chosen':('Product',(12,-16 if role=='test' else 18)),
                  'fixed3_selector':('F3',(-16,15)), 'fixed5_selector':('F5',(20,10 if role=='test' else -16)),
                  'fixed8_selector':('F8',(7,6)), 'fixed8_vote':('V8',(7,-7)),
                  'theta0.39_tau0.6_selector':('L',(0,-21)), 'theta0.3_tau0.6_selector':('M',(0,17))}
        if role=='test':annotate.update({'b':('B',(-27,4)),'p3_all':('P3 all',(9,-3))})
        for name,(label,offset) in annotate.items():
            r=table[name];d=r['gain_vs_b_bm25'];x=r['calls_mean'];y=d['delta_pp'];lo,hi=d['ci95_pp']
            ax.errorbar(x,y,yerr=[[y-lo],[hi-y]],fmt='none',ecolor='#888888',linewidth=.6,capsize=2,zorder=2)
            ax.annotate(label,(x,y),xytext=offset,textcoords='offset points',fontsize=8,
                        arrowprops={'arrowstyle':'-','lw':.4,'color':'#777777'},zorder=5)
        ax.set(xlim=(.75,6.65),ylim=(-4.0,3.7),xlabel='Total Qwen generations per question (C1 included)',
               ylabel='Judge accuracy gain over B+BM25 (percentage points)')
        n=s['roles'][role]['n'];ax.set_title(f'NQ {role.capitalize()} (n = {n:,}): frozen quality–cost trade-offs',loc='left')
        ax.grid(axis='y',color='#DDDDDD',linewidth=.4)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles,labels,loc='center',bbox_to_anchor=(.54,.10),ncol=3,frameon=False,columnspacing=1.3,handletextpad=.4)
        fig.text(.105,.945,'L: θ=0.39, τ=0.6    M: θ=0.3, τ=0.6    F/V: fixed selector/vote; number = contexts',fontsize=8)
        fig.text(.105,.027,'Whiskers: paired item bootstrap 95% CIs (10,000 resamples). All points are existing policies; no new selection.',fontsize=7)
        export(fig,f'pareto_{role}')
    # Online measurement is a separate confirm subset, never mixed with Test quality.
    receipt=read_archive(RELEASE/'inputs.tar.gz');timed=jsonl(receipt['timing/latency.jsonl'])
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.2,3.8),gridspec_kw={'width_ratios':[1,1.1]})
    fig.subplots_adjust(left=.12,right=.97,bottom=.28,top=.85,wspace=.36)
    comp=s['timing_components'];names=['C1','B+BM25','Product']
    components=[('C1',comp['c1_s'],'#999999'),('Diagnosis',comp['diagnosis_features_s']+comp['diagnosis_classifier_s'],'#E69F00'),
                ('B / BM25',comp['controller_s'],'#0072B2'),('Answer selection',comp['answer_selection_s'],'#CC79A7')]
    bottoms=np.zeros(3)
    for j,(name,val,color) in enumerate(components):
        vals=np.array([val if j==0 or (i>=1 and j<3) or i==2 else 0 for i in range(3)])
        ax.barh(names,vals,left=bottoms,color=color,label=name,height=.55,hatch=['','//','','..'][j]);bottoms+=vals
    ax.set(xlabel='Mean measured component time (s)',xlim=(0,3.05));ax.invert_yaxis()
    ax.set_title('(a) Component means',loc='left')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='center',bbox_to_anchor=(.28,.10),ncol=2,frameon=False,fontsize=7)
    for i,v in enumerate(bottoms):ax.text(v+.06,i,f'{v:.2f}',va='center',fontsize=8)
    arrays={'C1':[t['c1_s'] for t in timed],
            'B+BM25':[t['c1_s']+t['diagnosis_features_s']+t['diagnosis_classifier_s']+t['controller_s'] for t in timed],
            'Product':[t['total_s'] for t in timed]}
    for (name,v),color,style in zip(arrays.items(),['#777777','#0072B2','#CC79A7'],[':', '--','-']):
        xx=np.sort(v);bx.step(xx,np.arange(1,len(xx)+1)/len(xx),where='post',label=name,color=color,linestyle=style,lw=1.3)
    bx.set(xlabel='Measured time per question (s)',ylabel='Cumulative fraction of questions',ylim=(0,1.02),xlim=(0,max(max(v) for v in arrays.values())*1.02))
    bx.set_title('(b) Per-question distribution',loc='left');bx.legend(frameon=False,loc='lower right');bx.grid(axis='y',lw=.4,color='#DDDDDD')
    fig.suptitle('Online timing: 300 confirm questions, RTX 5090 D, after 10 warm-up questions',fontsize=10,x=.53,y=.97)
    fig.text(.12,.025,'One query at a time. C1 / B+BM25 are measured prefixes of the product; loading (82.19 s) excluded.',fontsize=7)
    export(fig,'latency_measured')
    (RELEASE/'figure_qa.json').write_text(json.dumps(qa,indent=2,default=str)+'\n')


if __name__=='__main__':main()
