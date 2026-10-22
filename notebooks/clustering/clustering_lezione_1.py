# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo", "numpy", "matplotlib", "scikit-learn", "scipy"]
# ///
# Pattern Recognition — Clustering. Marimo slides, one concept per cell.
# Open with: marimo edit clustering_lezione_1.py; switch to presentation/slides mode.

import marimo

__generated_with = "0.23.13"
app = marimo.App(
    width="medium",
    layout_file="layouts/clustering_lezione_1.slides.json",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons, make_blobs
    from scipy.cluster.hierarchy import linkage, dendrogram

    return make_blobs, make_moons, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Clustering I

    - Unsupervised learning: basic concepts
    - Proximity measures
    - K-means: objective, updates, convergence

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 25</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From dimensionality reduction to clustering

    - PCA / KPCA: coordinates preserving variance.
    - Laplacian Eigenmaps: preserve local neighborhoods.
    - Clustering: discover groups, without known labels.

    $$\mathbf{x}_1,\ldots,\mathbf{x}_N\in\mathbb{R}^d\quad\longrightarrow\quad c_i\in\{1,\ldots,k\}$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 25</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What makes a cluster?

    - Compact cloud around a centroid.
    - Chain of nearby points: another idea of a group.
    - No universal notion of a “correct” partition.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 25</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    control_4_mode=mo.ui.slider(start=0,stop=2,step=1,value=0,label="Geometry (0: clouds · 1: arcs · 2: elongated)",show_value=True)
    return (control_4_mode,)


@app.cell(hide_code=True)
def _(control_4_mode, mo, np, plt):
    _controls=control_4_mode
    _fig,_ax=plt.subplots(figsize=(8,3.6))
    _rng=np.random.default_rng(12)
    if control_4_mode.value==0:
        _A=_rng.normal([-2,0],[.35,.35],(45,2)); _B=_rng.normal([2,0],[.35,.35],(45,2))
    elif control_4_mode.value==1:
        _t=np.linspace(.05,np.pi-.05,45)
        _A=np.c_[2*np.cos(_t),2*np.sin(_t)]; _B=np.c_[1.15*np.cos(_t),1.15*np.sin(_t)]
    else:
        _A=_rng.normal([-2,0],[.32,.32],(45,2)); _t=np.linspace(-1.8,1.8,45);_B=np.c_[_t+1.7, _t*.45+.08*_rng.normal(size=45)]
    _ax.scatter(*_A.T,s=27,label="Group A");_ax.scatter(*_B.T,s=27,label="Group B");_ax.set_aspect('equal');_ax.legend(loc='upper right');_ax.set_xlim(-3.3,4);_ax.set_ylim(-2.5,3);_ax.grid(alpha=.15)
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ### Three geometries, three intuitions

    - Compare compact clouds, nested arcs and elongated groups.
    - Which shape is easiest to describe with a center?

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 25</div>
    """), _controls, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Partition, labels, centroids

    - Each observation belongs to exactly one cluster.
    - The centroid is the mean: not necessarily an observed point.

    $$C_r=\{i:c_i=r\},\qquad n_r=|C_r|,\qquad \boldsymbol\mu_r=\frac1{n_r}\sum_{i\in C_r}\mathbf x_i$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 25</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Three algorithmic families

    - K-means: iteratively reassign points to k centers.
    - Sequential: decide what to do with the next arriving point.
    - Hierarchical: merge currently existing groups, one pair at a time.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 25</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Proximity: what does “near” mean?

    - Distances: smaller means more similar.
    - Similarities: larger means more similar.
    - The notion of proximity affects the resulting clusters.

    $$d_2(\mathbf x_i,\mathbf x_j)=\|\mathbf x_i-\mathbf x_j\|_2$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 25</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    control_8_s=mo.ui.slider(start=1,stop=6,step=1,value=4,label="Vertical separation",show_value=True)
    return (control_8_s,)


@app.cell(hide_code=True)
def _(control_8_s, mo, np, plt):
    _controls=control_8_s
    _x,_y=3,control_8_s.value
    _fig,_ax=plt.subplots(figsize=(7.5,3.6));_ax.plot([0,_x],[0,_y],lw=2.5,label=f'Euclidean = {np.hypot(_x,_y):.2f}');_ax.plot([0,_x,_x],[0,0,_y],lw=2,ls='--',label=f'Manhattan = {_x+_y}');_ax.scatter([0,_x],[0,_y],s=70,zorder=4);_ax.set_xlim(-.7,6);_ax.set_ylim(-.7,7);_ax.set_aspect('equal');_ax.grid(alpha=.2);_ax.legend(loc='upper left')
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ### Euclidean vs Manhattan distance

    - Same endpoints; different routes through the plane.
    - Move the point to compare $d_2$, $d_1$ and $d_\infty$.

    $$d_1=\sum_{\ell=1}^d|x_{i\ell}-x_{j\ell}|,\qquad d_\infty=\max_\ell |x_{i\ell}-x_{j\ell}|$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 25</div>
    """), _controls, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Distances: properties and squared distance

    - A metric is nonnegative, symmetric and satisfies the triangle inequality.
    - Squared Euclidean distance is useful for k-means but is not generally a metric.

    $$\operatorname*{arg\,min}_r\|\mathbf x_i-\boldsymbol\mu_r\|=\operatorname*{arg\,min}_r\|\mathbf x_i-\boldsymbol\mu_r\|^2$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 25</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cosine similarity

    - Focuses on direction rather than vector length.
    - Common for document and high-dimensional vector comparisons.
    - Cosine dissimilarity is not automatically a metric.

    $$s_{\mathrm{cos}}(\mathbf x,\mathbf y)=\frac{\mathbf x^T\mathbf y}{\|\mathbf x\|\,\|\mathbf y\|},\qquad\delta_{\mathrm{cos}}=1-s_{\mathrm{cos}}$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 25</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    control_11_ang=mo.ui.slider(start=0,stop=180,step=10,value=40,label="Angle (degrees)",show_value=True)
    control_11_len=mo.ui.slider(start=1,stop=4,step=1,value=2,label="Length",show_value=True)
    return control_11_ang, control_11_len


@app.cell(hide_code=True)
def _(control_11_ang, control_11_len, mo, np, plt):
    _controls=mo.hstack([control_11_ang,control_11_len])
    _t=np.deg2rad(control_11_ang.value)
    _fig,_ax=plt.subplots(figsize=(7.5,3.4));_ax.arrow(0,0,2.3,0,head_width=.13,length_includes_head=True,color='tab:blue');_ax.arrow(0,0,control_11_len.value*np.cos(_t),control_11_len.value*np.sin(_t),head_width=.13,length_includes_head=True,color='tab:orange');_ax.set_xlim(-4.6,4.6);_ax.set_ylim(-.6,4.6);_ax.set_aspect('equal');_ax.grid(alpha=.2);_ax.set_title(f'Cosine similarity = {np.cos(_t):.3f}')
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ### Cosine similarity: rotate a vector

    - Changing the angle changes the similarity.
    - Changing only the length does not.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 25</div>
    """), _controls, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature scales matter

    - Large-scale features can dominate Euclidean distances.
    - Standardization changes the geometry before clustering.
    - Handle constant features separately.

    $$\widetilde x_{i\ell}=\frac{x_{i\ell}-\bar x_\ell}{s_\ell}\qquad(s_\ell>0)$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 25</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    control_13_scale=mo.ui.switch(value=False,label="Standardize features")
    return (control_13_scale,)


@app.cell(hide_code=True)
def _(control_13_scale, mo, np, plt):
    _controls=control_13_scale
    _P=np.array([[0.,0.],[2.,45.],[4.,4.]])
    _Q=np.array([2.,3.]);_all=np.vstack([_P,_Q]);_sd=_all.std(axis=0);_avg=_all.mean(axis=0)
    _Z=(_all-_avg)/_sd if control_13_scale.value else _all
    _fig,_ax=plt.subplots(figsize=(7.5,3.5));_ax.scatter(_Z[:3,0],_Z[:3,1],s=90,label="Candidates");_ax.scatter(_Z[3,0],_Z[3,1],marker='*',s=230,label="Query");_j=np.argmin(np.linalg.norm(_Z[:3]-_Z[3],axis=1));_ax.plot([_Z[_j,0],_Z[3,0]],[_Z[_j,1],_Z[3,1]],ls='--');_ax.set_title(f'Nearest candidate: {"ABC"[_j]}');_ax.legend();_ax.grid(alpha=.2)
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ### Scaling can change the nearest point

    - A large feature range dominates before standardization.
    - Toggle rescaling to see how the nearest neighbor changes.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">13 / 25</div>
    """), _controls, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Proximity: quick recap

    - Physical separation → Euclidean.
    - Sum of coordinate deviations → Manhattan.
    - Vector orientation → cosine similarity.
    - Next: fix squared Euclidean distance and optimize compactness.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">14 / 25</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## K-means: a precise optimization objective

    - Input: $k$ clusters and $N$ observations.
    - Penalize each squared distance to its assigned centroid.
    - Optimize labels and centroids alternately.

    $$\boxed{J(\mathbf c,\boldsymbol\mu)=\sum_{i=1}^N\|\mathbf x_i-\boldsymbol\mu_{c_i}\|^2}$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">15 / 25</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    control_16_seed=mo.ui.slider(start=0,stop=2,step=1,value=0,label="Initialization",show_value=True)
    return (control_16_seed,)


@app.cell(hide_code=True)
def _(control_16_seed, make_blobs, mo, np, plt):
    _controls=control_16_seed
    _X,_=make_blobs(n_samples=110,centers=[[-2,0],[2,0]],cluster_std=.65,random_state=11)
    _C=np.array([[ -3.,0.],[3.,0.]]) if control_16_seed.value==0 else (np.array([[-2.,2.],[0.,2.]]) if control_16_seed.value==1 else np.array([[0.,-2.],[1.,-2.]]))
    _labels=np.argmin((( _X[:,None,:]-_C[None,:,:])**2).sum(2),axis=1)
    _fig,_ax=plt.subplots(figsize=(7.7,3.8));_ax.scatter(*_X.T,c=_labels,cmap='coolwarm',s=21);_ax.scatter(*_C.T,marker='X',s=160,c='black');_ax.set_xlim(-5,5);_ax.set_ylim(-3.2,3.2);_ax.set_aspect('equal');_ax.set_title('One assignment step; initial centroids marked X')
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ### K-means: initialization changes the story

    - Move the chosen seed configuration.
    - Assignments depend on which centroids are initially selected.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">16 / 25</div>
    """), _controls, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why is the optimal centroid the mean?

    - Hold assignments fixed.
    - Minimize within-cluster squared error with respect to the center.
    - Derivation on the board → stationary point is the mean.

    $$J_r(\boldsymbol\mu_r)=\sum_{i\in C_r}\|\mathbf x_i-\boldsymbol\mu_r\|^2\quad\Longrightarrow\quad \boxed{\boldsymbol\mu_r=\frac1{n_r}\sum_{i\in C_r}\mathbf x_i}$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">17 / 25</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    control_18_center=mo.ui.slider(start=-1,stop=5,step=.25,value=1,label="Candidate centroid",show_value=True)
    return (control_18_center,)


@app.cell(hide_code=True)
def _(control_18_center, mo, np, plt):
    _controls=control_18_center
    _xx=np.array([0.,2.,4.]);_t=np.linspace(-1,5,140);_loss=((_xx[:,None]-_t[None,:])**2).sum(0)
    _fig,_ax=plt.subplots(figsize=(7.7,3.5));_ax.plot(_t,_loss);_ax.axvline(_xx.mean(),ls='--',label='Mean = 2');_ax.scatter([control_18_center.value],[((_xx-control_18_center.value)**2).sum()],s=90,zorder=3);_ax.set_title(f'J({ control_18_center.value:g}) = {((_xx-control_18_center.value)**2).sum():.2f}');_ax.set_xlabel('Candidate centroid');_ax.set_ylabel('Squared error');_ax.legend();_ax.grid(alpha=.2)
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ### The mean minimizes squared error

    - Slide the candidate center.
    - The error curve is minimized at the sample mean.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">18 / 25</div>
    """), _controls, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why assign to the nearest centroid?

    - Now hold centroid positions fixed.
    - Each observation affects only its own distance term.
    - Minimize independently, one point at a time.

    $$\boxed{c_i\leftarrow\operatorname*{arg\,min}_{1\leq r\leq k}\|\mathbf x_i-\boldsymbol\mu_r\|^2}$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">19 / 25</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Lloyd algorithm: alternating updates

    - 1. Initialize $k$ centroids.
    - 2. Assignment: nearest centroid.
    - 3. Update: arithmetic mean of each nonempty group.
    - 4. Repeat until assignments stabilize.

    $$c_i\leftarrow\arg\min_r\|\mathbf x_i-\boldsymbol\mu_r\|^2,\qquad\boldsymbol\mu_r\leftarrow\frac1{n_r}\sum_{i\in C_r}\mathbf x_i$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">20 / 25</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    control_21_step=mo.ui.slider(start=0,stop=8,step=1,value=0,label="Algorithm step",show_value=True)
    return (control_21_step,)


@app.cell(hide_code=True)
def _(control_21_step, make_blobs, mo, np, plt):
    _controls=control_21_step
    _X,_=make_blobs(n_samples=105,centers=[[-2.5,-.4],[2,1]],cluster_std=[.8,.85],random_state=6)
    _C=np.array([[-3.,2.5],[.5,-1.8]]);_lab=np.argmin(((_X[:,None,:]-_C[None,:,:])**2).sum(2),axis=1)
    _for=range(control_21_step.value)
    for _st in _for:
        if _st%2==0:_lab=np.argmin(((_X[:,None,:]-_C[None,:,:])**2).sum(2),axis=1)
        else:
            for _r in range(2):
                if np.any(_lab==_r):_C[_r]=_X[_lab==_r].mean(axis=0)
    _J=np.sum((_X-_C[_lab])**2)
    _fig,_ax=plt.subplots(figsize=(8,3.7));_ax.scatter(*_X.T,c=_lab,cmap='coolwarm',s=25,alpha=.85);_ax.scatter(*_C.T,marker='X',s=200,c='black',label='Centroids');_ax.legend(loc='upper left');_ax.set_xlim(-5,5);_ax.set_ylim(-4,4);_ax.set_aspect('equal');_ax.set_title(f'Step {control_21_step.value} · J = {_J:.1f}')
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ### K-means animation: assignments and moving centroids

    - Advance through initialization, assignment and centroid updates.
    - Watch the objective decrease as centers move.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">21 / 25</div>
    """), _controls, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why does the objective never increase?

    - Assignment step minimizes $J$ with centers fixed.
    - Update step minimizes $J$ with labels fixed.
    - A decreasing cost does not guarantee a global optimum.

    $$J(\mathbf c^{(t)},\boldsymbol\mu^{(t)})\geq J(\mathbf c^{(t+1)},\boldsymbol\mu^{(t)})\geq J(\mathbf c^{(t+1)},\boldsymbol\mu^{(t+1)})$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">22 / 25</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    control_23_stage=mo.ui.slider(start=0,stop=2,step=1,value=0,label="0: initialize · 1: assign · 2: update",show_value=True)
    return (control_23_stage,)


@app.cell(hide_code=True)
def _(control_23_stage, mo, np, plt):
    _controls=control_23_stage
    _points=np.array([0,2,7,9]);_means=np.array([0,7]) if control_23_stage.value<2 else np.array([1,8]);_lab=np.array([0,0,1,1]);_J=sum((_points-_means[_lab])**2)
    _fig,_ax=plt.subplots(figsize=(8,3));_ax.scatter(_points,np.zeros(4),c=_lab,cmap='coolwarm',s=120,zorder=3);_ax.scatter(_means,[.2,.2],s=170,marker='X',c='black',label='Centroids');_ax.set_yticks([]);_ax.set_ylim(-.35,.6);_ax.set_xlim(-1,10);_ax.set_xticks(range(10));_ax.grid(axis='x',alpha=.2);_ax.set_title(f'Step {control_23_stage.value} · cost J = {_J:g}');_ax.legend()
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ### Worked example: four observations

    - Data: $0,2,7,9$; initial centers $0,7$.
    - Assignment → groups $(0,2)$ and $(7,9)$; update → means $1,8$.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">23 / 25</div>
    """), _controls, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    control_24_k=mo.ui.slider(start=1,stop=4,step=1,value=2,label="Number of clusters k",show_value=True)
    return (control_24_k,)


@app.cell(hide_code=True)
def _(control_24_k, make_moons, mo, plt):
    _controls=control_24_k
    _X,_y=make_moons(n_samples=220,noise=.065,random_state=42)
    from sklearn.cluster import KMeans
    _lab=KMeans(n_clusters=control_24_k.value,n_init=10,random_state=2).fit_predict(_X)
    _fig,_ax=plt.subplots(figsize=(8,3.6));_ax.scatter(*_X.T,c=_lab,cmap='tab10',s=24);_ax.set_aspect('equal');_ax.set_title('K-means partitions of curved data')
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ### Geometric limitation: two moons

    - K-means creates centroid-based regions, not curved chains.
    - The two arcs are not necessarily recovered as separate groups.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">24 / 25</div>
    """), _controls, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## K-means: takeaways

    - Fast, intuitive; compact center-based groups.
    - Sensitive to scale, outliers and initialization.
    - Each iteration reduces (or preserves) squared error.
    - No guarantee of global optimum; elongated / curved groups may be difficult.

    $$\boxed{J=\sum_i\|\mathbf x_i-\boldsymbol\mu_{c_i}\|^2\qquad J^{(t+1)}\leq J^{(t)}}$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">25 / 25</div>
    """)
    return


if __name__ == "__main__":
    app.run()
