# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo", "numpy", "matplotlib", "scikit-learn", "scipy"]
# ///
# Pattern Recognition — Clustering. Marimo slides, one concept per cell.
# Open with: marimo edit clustering_lezione_2.py; switch to presentation/slides mode.

import marimo

__generated_with = "0.23.13"
app = marimo.App(
    width="medium",
    layout_file="layouts/clustering_lezione_2.slides.json",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons, make_blobs
    from scipy.cluster.hierarchy import linkage, dendrogram

    return dendrogram, linkage, make_moons, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Clustering II

    - Sequential clustering: BSAS and incremental mean.
    - Hierarchical clustering: linkage and dendrogram.
    - Ward: the connection to k-means.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 24</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From batch to sequential

    - K-means repeatedly reconsiders all data.
    - Sequential algorithms read one point at a time.
    - Each new point is assigned immediately or starts a new group.
    - Earlier decisions can influence later ones.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 24</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    control_3_s=mo.ui.slider(start=0,stop=4,step=1,value=0,label="Advance both algorithms",show_value=True)
    return (control_3_s,)


@app.cell(hide_code=True)
def _(control_3_s, mo, np, plt):
    _controls=control_3_s
    _seq=[[[0]],[[0,1]],[[0,1],[2]],[[0,1],[2,3]],[[0,1,4],[2,3]]]
    _hier=[[[0],[1],[2],[3],[4]],[[0,1],[2],[3],[4]],[[0,1],[2,3],[4]],[[0,1],[2,3,4]],[[0,1,2,3,4]]]
    _x=np.array([0.,1.,6.,7.,2.]);_fig,_axs=plt.subplots(1,2,figsize=(10,3.2),sharex=True,sharey=True)
    for _ax,_groups,_title in zip(_axs,[_seq[control_3_s.value],_hier[control_3_s.value]],['Sequential · points arrive','Hierarchical · clusters merge']):
        for _r,_group in enumerate(_groups):
            for _idx in _group:_ax.scatter(_x[_idx],0,c=f'C{_r%10}',s=140,zorder=4);_ax.annotate('ABCDE'[_idx],(_x[_idx],0),textcoords='offset points',xytext=(0,12),ha='center')
            if len(_group)>1:_ax.plot(_x[_group],[0]*len(_group),lw=5,alpha=.2,color=f'C{_r%10}')
        _ax.set_title(_title);_ax.set_xlim(-1,9);_ax.set_ylim(-1,1);_ax.set_yticks([]);_ax.grid(axis='x',alpha=.2)
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ### Animated comparison: sequential vs agglomerative

    - Left: a new point arrives and joins a group or creates one.
    - Right: all points exist from the start; groups are merged.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 24</div>
    """), _controls, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## BSAS: parameters and state

    - Distance threshold $\theta>0$.
    - Maximum number of clusters $q_{\max}$.
    - Current clusters: $q\leq q_{\max}$ and centroids $\boldsymbol\mu_1,\ldots,\boldsymbol\mu_q$.

    $$\theta>0,\qquad q\leq q_{\max}$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 24</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## BSAS: assignment rule

    - First point forms the first cluster.
    - Find nearest existing center and its distance $\delta_i$.
    - If too far and room remains, create a new group.
    - Otherwise, assign to nearest group and update its mean.

    $$r^*=\arg\min_{1\leq r\leq q}\|\mathbf x_i-\boldsymbol\mu_r\|,\quad\delta_i=\|\mathbf x_i-\boldsymbol\mu_{r^*}\|$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 24</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    control_6_s=mo.ui.slider(start=1,stop=4,step=1,value=1,label="Points received",show_value=True)
    control_6_theta=mo.ui.slider(start=.5,stop=4.,step=.5,value=2.5,label="Threshold θ",show_value=True)
    return control_6_s, control_6_theta


@app.cell(hide_code=True)
def _(control_6_s, control_6_theta, mo, np, plt):
    _controls=mo.hstack([control_6_s,control_6_theta])
    _points=[0.,2.,3.,4.];_clusters=[]
    for _v in _points[:control_6_s.value]:
        if not _clusters:_clusters=[[_v]];continue
        _centers=np.array([np.mean(_c) for _c in _clusters]);_j=int(np.argmin(abs(_centers-_v)))
        if abs(_centers[_j]-_v)>control_6_theta.value and len(_clusters)<4:_clusters.append([_v])
        else:_clusters[_j].append(_v)
    _fig,_ax=plt.subplots(figsize=(8,3.1))
    for _r,_group in enumerate(_clusters):_ax.scatter(_group,[0]*len(_group),s=140,color=f'C{_r}',label=f'Cluster {_r+1}, μ={np.mean(_group):.2f}');_ax.scatter(np.mean(_group),.27,marker='X',s=120,color=f'C{_r}')
    _ax.set_xlim(-1,5);_ax.set_ylim(-.4,.75);_ax.set_yticks([]);_ax.set_xticks(range(5));_ax.set_title(f'{len(_clusters)} cluster(s) after {control_6_s.value} observations');_ax.grid(axis='x',alpha=.2);_ax.legend(loc='upper left')
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ### BSAS: one point at a time

    - Data arrive in the order $0,2,3,4$.
    - Change the threshold and advance one observation at a time.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 24</div>
    """), _controls, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Incremental centroid: no need to reread data

    - Suppose a group contains $n_r$ points and has mean $\boldsymbol\mu_r$.
    - Add one observation $\mathbf x_i$.
    - The new mean uses only old mean, count and new point.

    $$\boxed{\boldsymbol\mu_r^{\mathrm{new}}=\frac{n_r\boldsymbol\mu_r+\mathbf x_i}{n_r+1}=\boldsymbol\mu_r+\frac{\mathbf x_i-\boldsymbol\mu_r}{n_r+1}}$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 24</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    control_8_s=mo.ui.slider(start=1,stop=6,step=1,value=2,label="Observations received",show_value=True)
    return (control_8_s,)


@app.cell(hide_code=True)
def _(control_8_s, mo, np, plt):
    _controls=control_8_s
    _data=np.array([0.,2.,3.,4.,3.,4.]);_mean=np.mean(_data[:control_8_s.value]);_fig,_ax=plt.subplots(figsize=(8,3))
    _ax.scatter(_data[:control_8_s.value],np.zeros(control_8_s.value),s=110,zorder=3);_ax.scatter([_mean],[.24],marker='X',c='black',s=160,label=f'Mean = {_mean:.2f}');_ax.set_xlim(-1,5);_ax.set_ylim(-.45,.7);_ax.set_yticks([]);_ax.set_xticks(range(5));_ax.grid(axis='x',alpha=.2);_ax.legend()
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ### Incremental mean: the centroid moves

    - Add points to the same group.
    - The influence of each additional point becomes smaller as $n$ grows.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 24</div>
    """), _controls, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    control_9_s=mo.ui.slider(start=1,stop=4,step=1,value=1,label="Observations received",show_value=True)
    return (control_9_s,)


@app.cell(hide_code=True)
def _(control_9_s, mo, np, plt):
    _controls=control_9_s
    _orders=[[0.,2.,3.,4.],[0.,4.,2.,3.]];_fig,_axs=plt.subplots(1,2,figsize=(10,3),sharex=True,sharey=True)
    for _ax,_points,_name in zip(_axs,_orders,['Order: 0, 2, 3, 4','Order: 0, 4, 2, 3']):
        _clusters=[]
        for _v in _points[:control_9_s.value]:
            if not _clusters:_clusters=[[_v]];continue
            _centers=np.array([np.mean(_c) for _c in _clusters]);_r=int(np.argmin(abs(_centers-_v)))
            if abs(_centers[_r]-_v)>2.5 and len(_clusters)<3:_clusters.append([_v])
            else:_clusters[_r].append(_v)
        for _r,_group in enumerate(_clusters):_ax.scatter(_group,[0]*len(_group),color=f'C{_r}',s=120);_ax.scatter([np.mean(_group)],[.3],marker='X',color=f'C{_r}',s=120)
        _ax.set_title(f'{_name} · {len(_clusters)} cluster(s)');_ax.set_xlim(-1,5);_ax.set_ylim(-.4,.8);_ax.set_yticks([]);_ax.grid(axis='x',alpha=.2)
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ### BSAS: same observations, different order

    - Compare $0,2,3,4$ against $0,4,2,3$.
    - With $\theta=2.5$, one order produces one cluster, the other two.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 24</div>
    """), _controls, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sequential variants: the idea

    - MBSAS: separate seed-selection from final assignment.
    - TTSAS: two thresholds; defer ambiguous decisions.
    - Order dependence is reduced in some variants, not universally eliminated.

    $$0<\theta_1<\theta_2$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 24</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hierarchical clustering: why a hierarchy?

    - Rather than commit immediately to a single partition, build nested partitions.
    - Agglomerative: begin with $N$ singleton clusters.
    - At each step, merge the two closest clusters.
    - Merges are irreversible.

    $$N\longrightarrow N-1\longrightarrow\cdots\longrightarrow 2\longrightarrow1$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 24</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    control_12_s=mo.ui.slider(start=0,stop=3,step=1,value=0,label="Number of merges",show_value=True)
    return (control_12_s,)


@app.cell(hide_code=True)
def _(control_12_s, mo, plt):
    _controls=control_12_s
    _groups=[[[0],[2],[3],[10]],[[0],[2,3],[10]],[[0,2,3],[10]],[[0,2,3,10]]][control_12_s.value]
    _fig,_ax=plt.subplots(figsize=(8,3))
    for _i,_group in enumerate(_groups):_ax.scatter(_group,[0]*len(_group),s=135,color=f'C{_i}');_ax.plot([min(_group),max(_group)],[.1,.1],lw=5,color=f'C{_i}',alpha=.6)
    _ax.set_xlim(-1,11);_ax.set_ylim(-.5,.8);_ax.set_xticks([0,2,3,10]);_ax.set_yticks([]);_ax.grid(axis='x',alpha=.2);_ax.set_title(f'{len(_groups)} cluster(s) · {control_12_s.value} merge(s)')
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ### Hierarchical merging: watch every fusion

    - All four observations exist from the beginning.
    - Each step fuses two clusters; none is created by arriving data.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 24</div>
    """), _controls, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What is linkage?

    - Distance between points is not distance between sets.
    - Single: closest cross-cluster pair.
    - Complete: farthest cross-cluster pair.
    - Average: mean over all cross-cluster pairs.

    $$\begin{aligned}D_{\mathrm{single}}(A,B)&=\min_{i\in A,j\in B}d(\mathbf x_i,\mathbf x_j)\\ D_{\mathrm{complete}}(A,B)&=\max_{i\in A,j\in B}d(\mathbf x_i,\mathbf x_j)\end{aligned}$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">13 / 24</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    control_14_mode=mo.ui.slider(start=0,stop=2,step=1,value=0,label="0: single · 1: complete · 2: average",show_value=True)
    return (control_14_mode,)


@app.cell(hide_code=True)
def _(control_14_mode, mo, np, plt):
    _controls=control_14_mode
    _A=np.array([0.,2.]);_B=np.array([3.,10.]);_pairs=[(float(abs(_a-_b)),_a,_b) for _a in _A for _b in _B];_sel=sorted(_pairs,key=lambda z:z[0])
    _fig,_ax=plt.subplots(figsize=(8,3))
    for _v in _A:_ax.scatter(_v,0,s=110,color='tab:blue')
    for _v in _B:_ax.scatter(_v,.9,s=110,color='tab:orange')
    for _d,_a,_b in _pairs:
        if control_14_mode.value==2:_ax.plot([_a,_b],[0,.9],color='gray',alpha=.5)
    if control_14_mode.value<2:
        _d,_a,_b=_sel[0] if control_14_mode.value==0 else _sel[-1]
        _ax.plot([_a,_b],[0,.9],lw=3,color='black')
    else:_d=np.mean([_p[0] for _p in _pairs])
    _ax.set_xlim(-1,11);_ax.set_ylim(-.4,1.5);_ax.set_yticks([0,.9],['A','B']);_ax.set_xticks([0,2,3,10]);_ax.grid(axis='x',alpha=.2);_ax.set_title(f'{["Single", "Complete", "Average"][control_14_mode.value]} linkage distance = {_d:.2f}')
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ### Linkage animation: nearest / farthest / average

    - Two existing groups, four cross-group distances.
    - The definition of inter-cluster proximity changes with linkage.

    $$D_{\mathrm{average}}(A,B)=\frac1{|A||B|}\sum_{i\in A}\sum_{j\in B}d(\mathbf x_i,\mathbf x_j)$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">14 / 24</div>
    """), _controls, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Agglomerative algorithm

    - Initialize $\mathcal P^{(0)}$ with $N$ singleton sets.
    - Compute inter-cluster linkage distances.
    - Merge the pair with smallest linkage distance.
    - Repeat until the desired number of clusters (or one).

    $$(A^*,B^*)=\operatorname*{arg\,min}_{A\neq B}D_{\mathrm{linkage}}(A,B)$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">15 / 24</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    control_16_mode=mo.ui.slider(start=0,stop=2,step=1,value=0,label="0: single · 1: complete · 2: average",show_value=True)
    control_16_cut=mo.ui.slider(start=1,stop=10,step=1,value=5,label="Cut height",show_value=True)
    return control_16_cut, control_16_mode


@app.cell(hide_code=True)
def _(control_16_cut, control_16_mode, dendrogram, linkage, mo, np, plt):
    _controls=mo.hstack([control_16_mode,control_16_cut])
    _X=np.array([[0.],[2.],[3.],[10.]])
    _Z=linkage(_X,method=['single','complete','average'][control_16_mode.value],metric='euclidean')
    _fig,_ax=plt.subplots(figsize=(8,3.8));dendrogram(_Z,labels=['0','2','3','10'],ax=_ax,color_threshold=0,above_threshold_color='tab:blue');_ax.axhline(control_16_cut.value,ls='--',color='tab:red',label='Cut');_ax.set_title(['Single','Complete','Average'][control_16_mode.value]+' linkage dendrogram');_ax.set_ylabel('Linkage distance');_ax.legend(loc='upper left')
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ### Dendrogram: four points, three linkage choices

    - The same merges can occur at different heights.
    - Cut the tree to obtain a chosen partition.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">16 / 24</div>
    """), _controls, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Worked example: dendrogram heights

    - Observations on a line: $0,2,3,10$.
    - First merge: $2$ and $3$ at distance $1$.
    - Second merge: $0$ with $\{2,3\}$.
    - Final merge with $10$: heights depend on linkage.

    $$\begin{array}{c|ccc}&\text{single}&\text{complete}&\text{average}\\ \text{second merge}&2&3&2.5\\ \text{last merge}&7&10&25/3\end{array}$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">17 / 24</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    control_18_mode=mo.ui.slider(start=0,stop=1,step=1,value=0,label="0: single · 1: complete",show_value=True)
    return (control_18_mode,)


@app.cell(hide_code=True)
def _(control_18_mode, linkage, make_moons, mo, plt):
    _controls=control_18_mode
    _X,_y=make_moons(n_samples=120,noise=.045,random_state=7)
    from scipy.cluster.hierarchy import fcluster
    _Z=linkage(_X,method=['single','complete'][control_18_mode.value]);_lab=fcluster(_Z,2,criterion='maxclust')
    _fig,_ax=plt.subplots(figsize=(8,3.5));_ax.scatter(*_X.T,c=_lab,cmap='coolwarm',s=28);_ax.set_aspect('equal');_ax.set_title(['Single linkage','Complete linkage'][control_18_mode.value]+' · two clusters')
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ### Two moons: single vs complete linkage

    - Single linkage can connect a curved chain.
    - Complete linkage tends to favor tighter, smaller-diameter groups.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">18 / 24</div>
    """), _controls, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ward: connect hierarchy to k-means

    - Instead of closest / farthest pair, measure the increase in squared within-cluster error.
    - Merge the pair producing the smallest increase.
    - Same squared-error idea, but merges are irreversible.

    $$S(A)=\sum_{i\in A}\|\mathbf x_i-\boldsymbol\mu_A\|^2,\qquad \Delta(A,B)=S(A\cup B)-S(A)-S(B)$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">19 / 24</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ward: final formula (derive on the board)

    - Write the mean of the union.
    - Expand squared distances around the two old means.
    - Cross-terms vanish because deviations sum to zero.

    $$\boxed{\Delta(A,B)=\frac{n_A n_B}{n_A+n_B}\|\boldsymbol\mu_A-\boldsymbol\mu_B\|^2}$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">20 / 24</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    control_21_n=mo.ui.slider(start=1,stop=15,step=1,value=2,label="Number of points in A",show_value=True)
    control_21_dist=mo.ui.slider(start=1,stop=5,step=1,value=2,label="Centroid separation",show_value=True)
    return control_21_dist, control_21_n


@app.cell(hide_code=True)
def _(control_21_dist, control_21_n, mo, np, plt):
    _controls=mo.hstack([control_21_n,control_21_dist])
    _nb=3;_delta=(control_21_n.value*_nb/(control_21_n.value+_nb))*control_21_dist.value**2
    _t=np.arange(1,16);_y=_t*_nb/(_t+_nb)*control_21_dist.value**2
    _fig,_ax=plt.subplots(figsize=(8,3.4));_ax.plot(_t,_y);_ax.scatter([control_21_n.value],[_delta],s=120,zorder=3);_ax.set_xlabel('Size of group A');_ax.set_ylabel('Ward increment Δ');_ax.set_title(f'nB = 3 · separation = {control_21_dist.value} · Δ = {_delta:.2f}');_ax.grid(alpha=.2)
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ### Ward: distance and cluster sizes both matter

    - Fix centroid separation and change the size of cluster A.
    - Ward weighs squared separation by group sizes.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">21 / 24</div>
    """), _controls, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ward vs k-means

    - K-means: reassign observations and update centroids repeatedly.
    - Ward: merge entire groups; cannot undo past merges.
    - Both are built around within-cluster squared errors.

    $$J_{\mathrm{kmeans}}=\sum_r S(C_r),\qquad\text{Ward: choose }\arg\min_{A,B}\Delta(A,B)$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">22 / 24</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary: no universal clustering algorithm

    - K-means: compactness around centers.
    - Sequential BSAS: decisions upon arrival; order matters.
    - Hierarchical: nested structure via irreversible merges.
    - Single / complete / average / Ward express different notions of group.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">23 / 24</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Final takeaway

    - Start with the meaning of proximity.
    - Then choose what a cluster should represent.
    - Only then choose the algorithm.

    $$\text{proximity}\;\longrightarrow\;\text{cluster notion}\;\longrightarrow\;\text{algorithm}$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">24 / 24</div>
    """)
    return


if __name__ == "__main__":
    app.run()
