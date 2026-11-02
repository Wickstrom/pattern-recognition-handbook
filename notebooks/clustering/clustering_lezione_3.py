# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo>=0.23", "numpy", "matplotlib", "scikit-learn", "scipy"]
# ///
# Pattern Recognition — Clustering III. Marimo slides, one concept per visible cell.
# Run from this directory: marimo run clustering_lezione_3.py

import marimo

__generated_with = "0.23.13"
app = marimo.App(
    width="medium",
    layout_file="layouts/clustering_lezione_3.slides.json",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons, make_blobs
    from sklearn.cluster import KMeans, SpectralClustering
    from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
    from scipy.spatial.distance import cdist
    from scipy.linalg import eigh
    from scipy.sparse.csgraph import connected_components

    X_moons, y_moons = make_moons(n_samples=160, noise=0.075, random_state=8)
    X_blobs, _ = make_blobs(n_samples=180, centers=[[-3, 0], [0, 2], [3, 0]], cluster_std=0.65, random_state=8)
    km_moons = KMeans(n_clusters=2, n_init=20, random_state=8).fit_predict(X_moons)
    sp_moons = SpectralClustering(n_clusters=2, affinity="nearest_neighbors", n_neighbors=10, assign_labels="kmeans", random_state=8).fit_predict(X_moons)
    inertias = [KMeans(n_clusters=k, n_init=10, random_state=8).fit(X_blobs).inertia_ for k in range(1, 9)]
    blob_labels = KMeans(n_clusters=3, n_init=20, random_state=8).fit_predict(X_blobs)
    silhouette_values = silhouette_samples(X_blobs, blob_labels)
    graph_xy = np.array([[-2.2, 0.3], [-1.45, 1.0], [-1.35, -0.7], [1.2, 0.6], [2.0, 0.9], [1.65, -0.6], [-3.1, -0.55]])
    return (
        KMeans,
        X_blobs,
        X_moons,
        adjusted_rand_score,
        blob_labels,
        cdist,
        eigh,
        graph_xy,
        inertias,
        km_moons,
        mo,
        np,
        plt,
        silhouette_score,
        silhouette_values,
        sp_moons,
        y_moons,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Clustering III: Spectral Clustering and Validation


    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 30</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From LE to Spectral Clustering

    You already know **Laplacian Eigenmaps**: build a neighborhood graph and use its low-frequency eigenvectors as coordinates.

    Spectral clustering takes one more step: **cluster those coordinates**.

    $$\text{points} \to \text{similarity graph} \to \text{eigenvectors} \to \text{k-means labels}$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 30</div>
    """)
    return


@app.cell(hide_code=True)
def _(X_moons, km_moons, mo, plt, y_moons):
    _fig,_axs=plt.subplots(1,2,figsize=(9,3.1),sharex=True,sharey=True)
    for _ax,_lab,_title in zip(_axs,[y_moons,km_moons],["Connected shapes","K-means partition"]):
        _ax.scatter(X_moons[:,0],X_moons[:,1],c=_lab,cmap="coolwarm",s=20,edgecolors="none")
        _ax.set_title(_title); _ax.set_aspect("equal"); _ax.grid(alpha=.15)
    _fig.tight_layout()
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ## The problem with centers on curved groups

    K-means prefers groups that are compact around a mean. Two interlocking moons are connected locally but have overlapping centers.

    **Question:** should a cluster be defined by its centroid or by paths of strong neighbor links?

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 30</div>
    """), mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Replace distances to centers with a graph

    A point is a **vertex**. A strong pairwise similarity is an **edge**. Dense connections within a group and weak connections between groups suggest a partition.

    $$W_{ij}\geq0,\quad W_{ij}=W_{ji},\quad W_{ii}=0$$

    The graph records **local relationships**, so the cluster boundary can curve.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 30</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    edge_limit = mo.ui.slider(start=0.7,stop=4.5,step=0.2,value=1.5,label="Distance threshold",show_value=True)
    return (edge_limit,)


@app.cell(hide_code=True)
def _(cdist, edge_limit, graph_xy, mo, plt):
    _controls=edge_limit
    _d=cdist(graph_xy,graph_xy)
    _fig,_ax=plt.subplots(figsize=(7.6,3.0))
    for _i in range(len(graph_xy)):
        for _j in range(_i+1,len(graph_xy)):
            if _d[_i,_j] <= edge_limit.value:
                _ax.plot(graph_xy[[_i,_j],0],graph_xy[[_i,_j],1],color="#7a8aa0",lw=1.5,zorder=1)
    _ax.scatter(graph_xy[:,0],graph_xy[:,1],c=[0,0,0,1,1,1,0],cmap="coolwarm",vmin=0,vmax=1,s=120,zorder=2)
    for _i,(_x,_y) in enumerate(graph_xy): _ax.text(_x,_y+.16,str(_i+1),ha="center",fontsize=9)
    _ax.set(xlim=(-3.8,2.7),ylim=(-1.2,1.55),title=f"Edges with distance ≤ {edge_limit.value:.1f}")
    _ax.set_aspect("equal"); _ax.axis("off")
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ##   the edge threshold changes the graph

    Raise the distance threshold to add edges. Too small: isolated fragments. Too large: the two groups merge.

    **Look for:** a range that preserves within-group links without many cross-group links.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 30</div>
    """), _controls, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to assign edge weights

    A Gaussian affinity gives nearby points stronger links:

    $$W_{ij}=\exp\!\left(-\frac{\|x_i-x_j\|^2}{2\sigma^2}\right)$$

    Usually we retain only **k nearest neighbors** (or points within a radius). $\sigma$ and graph sparsity change the result; feature scaling still matters.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 30</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Degree and volume: how much connection?

    The **degree** of vertex $i$ is the sum of its edge weights. The volume of a set adds its degrees.

    $$d_i=\sum_jW_{ij},\qquad \operatorname{vol}(A)=\sum_{i\in A}d_i$$

    Volume measures how strongly a group participates in the graph. It will stop us rewarding tiny isolated clusters.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 30</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The graph Laplacian

    Define $D=\operatorname{diag}(d_1,\ldots,d_n)$ and $L=D-W$.

    $$f^TLf=\frac12\sum_{i,j}W_{ij}(f_i-f_j)^2$$

    A low-energy vector changes slowly across **strong** edges. Constant values on a connected component have zero energy.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 30</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    cut_choice=mo.ui.slider(start=0,stop=1,step=1,value=0,label="0: isolate leaf · 1: split dense groups",show_value=True)
    return (cut_choice,)


@app.cell(hide_code=True)
def _(cut_choice, graph_xy, mo, np, plt):
    _controls=cut_choice
    _W=np.zeros((7,7))
    for _group in ([0,1,2],[3,4,5]):
        for _i in _group:
            for _j in _group:
                if _i!=_j:_W[_i,_j]=1.0
    _W[2,3]=_W[3,2]=0.7; _W[0,6]=_W[6,0]=0.2
    _A={6} if cut_choice.value==0 else {0,1,2,6}
    _c=float(_W[list(_A)][:,list(set(range(7))-_A)].sum())
    _fig,_ax=plt.subplots(figsize=(7.6,3.0))
    for _i in range(7):
        for _j in range(_i+1,7):
            if _W[_i,_j]>0:
                _ax.plot(graph_xy[[_i,_j],0],graph_xy[[_i,_j],1],color=("#d62728" if ((_i in _A)!=(_j in _A)) else "#aab4c1"),lw=2+2*_W[_i,_j],zorder=1)
    _ax.scatter(graph_xy[:,0],graph_xy[:,1],c=[0 if i in _A else 1 for i in range(7)],cmap="coolwarm",vmin=0,vmax=1,s=135,zorder=2)
    _ax.set(title=f"Cut weight = {_c:.1f}",xlim=(-3.8,2.7),ylim=(-1.2,1.55));_ax.set_aspect("equal");_ax.axis("off")
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ##   choose a cut

    A **cut** is the total weight of edges crossing a proposed partition. Move between a leaf split and a split of the two dense groups.

    $$\operatorname{cut}(A,\bar A)=\sum_{i\in A,j\in\bar A} W_{ij}$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 30</div>
    """), _controls, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    partition_pick=mo.ui.slider(start=0,stop=1,step=1,value=0,label="Selected partition",show_value=True)
    return (partition_pick,)


@app.cell(hide_code=True)
def _(mo, np, partition_pick, plt):
    _controls=partition_pick
    _W=np.zeros((7,7))
    for _g in ([0,1,2],[3,4,5]):
        for _i in _g:
            for _j in _g:
                if _i!=_j:_W[_i,_j]=1
    _W[2,3]=_W[3,2]=.7;_W[0,6]=_W[6,0]=.2
    _A={6} if partition_pick.value==0 else {0,1,2,6}
    _B=set(range(7))-_A;_d=_W.sum(axis=1);_cut=_W[np.ix_(list(_A),list(_B))].sum();_ncut=_cut/_d[list(_A)].sum()+_cut/_d[list(_B)].sum()
    _fig,_axs=plt.subplots(1,2,figsize=(8.4,2.8))
    for _ax,_vals,_title in zip(_axs,[[.2,.7],[1.015,.203]],["Raw cut (smaller is better)","Normalized cut (smaller is better)"]):
        _bars=_ax.bar(["Leaf","Dense groups"],_vals,color=["#d62728" if partition_pick.value==0 else "#aab4c1","#d62728" if partition_pick.value==1 else "#aab4c1"])
        _ax.set_ylim(0,1.2);_ax.set_title(_title,fontsize=10)
    _fig.suptitle(f"Selected: cut={_cut:.2f}, Ncut={_ncut:.2f}")
    _fig.tight_layout()
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ##   Cut can prefer a tiny group

    The single leaf has the **smaller cut**, but its normalized cut is much worse. The denominator penalizes separating a low-volume fragment.

    $$\operatorname{Ncut}(A,\bar A)=\frac{\operatorname{cut}(A,\bar A)}{\operatorname{vol}(A)}+\frac{\operatorname{cut}(A,\bar A)}{\operatorname{vol}(\bar A)}$$

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 30</div>
    """), _controls, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From a hard partition to a continuous vector

    The discrete Ncut problem is combinatorial. Its relaxation replaces a binary group indicator with a real-valued vector.

    For the normalized Laplacian $L_{\mathrm{sym}}=I-D^{-1/2}WD^{-1/2}$, low eigenvectors minimize a smoothness objective under normalization.

    **Then:** discretize the embedding into labels with k-means.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 30</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bridge_weight=mo.ui.slider(start=0.1,stop=2.0,step=0.1,value=0.3,label="Bridge weight",show_value=True)
    return (bridge_weight,)


@app.cell(hide_code=True)
def _(bridge_weight, eigh, mo, np, plt):
    _controls=bridge_weight
    _W=np.zeros((6,6))
    for _g in ([0,1,2],[3,4,5]):
        for _i in _g:
            for _j in _g:
                if _i!=_j:_W[_i,_j]=1
    _W[2,3]=_W[3,2]=bridge_weight.value
    _L=np.diag(_W.sum(axis=1))-_W
    _lam,_vec=eigh(_L);_v=_vec[:,1]
    if _v[:3].mean()<0:_v=-_v
    _fig,_ax=plt.subplots(figsize=(7.3,3.0))
    _ax.bar(range(1,7),_v,color=["#377eb8"]*3+["#e46c55"]*3)
    _ax.axhline(0,color="#333",lw=.7);_ax.set(xlabel="Vertex",ylabel="Second eigenvector",ylim=(-.5,.5),title=f"λ₂ = {_lam[1]:.3f}")
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ##   the Fiedler vector follows the bridge

    For this connected graph, the second eigenvector of $L$ separates its two dense halves. Increase bridge weight and the split becomes less pronounced.

    **Sign is arbitrary:** swapping positive and negative leaves the partition unchanged.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 30</div>
    """), _controls, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The link to Laplacian Eigenmaps

    Both methods encode **strong neighbors as nearby coordinates**.

    - Laplacian Eigenmaps uses the low eigenvectors as a representation.
    - Spectral clustering applies a final **partition** in that representation.

    The exact eigenproblem depends on normalization: $L$, $(L,D)$ or $L_{\mathrm{sym}}$. Do not mix coordinates from one formulation with another.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">13 / 30</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    embedding_stage=mo.ui.slider(start=0,stop=2,step=1,value=0,label="Stage",show_value=True)
    return (embedding_stage,)


@app.cell(hide_code=True)
def _(X_moons, cdist, eigh, embedding_stage, mo, np, plt, y_moons):
    _controls=embedding_stage
    _D=cdist(X_moons,X_moons);_neighbors=np.argsort(_D,axis=1)[:,1:11]
    _W=np.zeros_like(_D)
    for _i in range(len(X_moons)):_W[_i,_neighbors[_i]]=1
    _W=np.maximum(_W,_W.T);_deg=_W.sum(axis=1)
    _L=np.eye(len(_W))-_W/np.sqrt(_deg[:,None]*_deg[None,:])
    _lam,_U=eigh(_L,subset_by_index=[0,2]);_Y=_U[:,:2];_Y=_Y/np.maximum(np.linalg.norm(_Y,axis=1,keepdims=True),1e-12)
    _fig,_ax=plt.subplots(figsize=(7.4,3.1))
    if embedding_stage.value==0:
        _ax.scatter(X_moons[:,0],X_moons[:,1],c=y_moons,cmap="coolwarm",s=20)
        _ax.set_title("1 · Original coordinates")
    elif embedding_stage.value==1:
        for _i in range(0,len(X_moons),2):
            for _j in _neighbors[_i,:4]: _ax.plot(X_moons[[_i,_j],0],X_moons[[_i,_j],1],color="#9aa9b9",lw=.4,alpha=.5)
        _ax.scatter(X_moons[:,0],X_moons[:,1],c=y_moons,cmap="coolwarm",s=16,zorder=2)
        _ax.set_title("2 · Neighbor graph")
    else:
        _ax.scatter(_Y[:,0],_Y[:,1],c=y_moons,cmap="coolwarm",s=20)
        _ax.set_title("3 · Spectral coordinates")
    _ax.grid(alpha=.15)
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ##   a spectral embedding unfolds the moons

    Step through the raw data, the graph, and its two-dimensional low-frequency embedding. The embedding makes the two groups easier for k-means to separate.

    For the symmetric normalized Laplacian, row-normalize the selected eigenvectors before k-means.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">14 / 30</div>
    """), _controls, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spectral clustering: the practical recipe

    1. Scale features and choose a similarity graph $W$.
    2. Form a graph Laplacian; select $k$ low eigenvectors.
    3. Form one $k$-dimensional row per point; row-normalize for the $L_{\mathrm{sym}}$ recipe.
    4. Run k-means on these rows; return labels for the **original** points.

    **Key choice:** the graph can matter more than the final k-means run.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">15 / 30</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    moons_view=mo.ui.slider(start=0,stop=2,step=1,value=0,label="Partition",show_value=True)
    return (moons_view,)


@app.cell(hide_code=True)
def _(X_moons, km_moons, mo, moons_view, plt, sp_moons, y_moons):
    _controls=moons_view
    _labels=[y_moons,km_moons,sp_moons][moons_view.value]
    _titles=["Reference shapes","K-means in original space","Spectral clustering"][moons_view.value]
    _fig,_ax=plt.subplots(figsize=(7.4,3.0))
    _ax.scatter(X_moons[:,0],X_moons[:,1],c=_labels,cmap="coolwarm",s=22)
    _ax.set_title(_titles);_ax.set_aspect("equal");_ax.grid(alpha=.15)
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ##   k-means versus spectral on two moons

    Move from the true shapes to k-means labels, then spectral labels. Both methods use $k=2$; they differ in the space where k-means operates.

    **Watch:** spectral clustering follows local connectivity.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">16 / 30</div>
    """), _controls, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Graph design can change the answer

    - **Few neighbors:** fragile or disconnected graph.
    - **Too many neighbors:** shortcuts across the gap between groups.
    - **Gaussian width $\sigma$:** very small weights can effectively delete edges; very large widths flatten similarities.

    Inspect graph connectivity and repeat the analysis over plausible settings.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">17 / 30</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Limits and computational cost

    A dense affinity matrix needs $O(n^2)$ storage, and a full dense eigendecomposition can cost $O(n^3)$ time. Sparse neighbor graphs and partial eigensolvers help at scale.

    Disconnected graphs have multiple zero eigenvalues. Eigenvectors can rotate within that zero eigenspace, so interpret the **subspace/partition**, not one vector entry by entry.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">18 / 30</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spectral clustering: three takeaways

    **Idea:** a cluster can be a well-connected region of a graph.

    **Math:** low Laplacian eigenvectors encode weakly connected groups.

    **Practice:** graph construction, normalization, $k$, and final k-means all require choices. Now we ask how to assess them.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">19 / 30</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Validation: what are we checking?

    **Internal:** compactness and separation from the data alone (inertia, silhouette).

    **External:** agreement with trusted reference labels (ARI), if labels exist.

    **Stability:** whether small changes to data, seed, or graph choices change the partition.

    No single score proves that clusters are meaningful for a task.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">20 / 30</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    elbow_k=mo.ui.slider(start=1,stop=8,step=1,value=3,label="Number of clusters k",show_value=True)
    return (elbow_k,)


@app.cell(hide_code=True)
def _(elbow_k, inertias, mo, np, plt):
    _controls=elbow_k
    _ks=np.arange(1,9)
    _fig,_ax=plt.subplots(figsize=(7.3,3.0))
    _ax.plot(_ks,inertias,"o-",color="#527ca3")
    _ax.scatter([elbow_k.value],[inertias[elbow_k.value-1]],s=180,color="#e46c55",zorder=3)
    _ax.axvline(elbow_k.value,color="#e46c55",alpha=.4,ls="--")
    _ax.set(xticks=_ks,xlabel="Number of clusters k",ylabel="Within-cluster sum of squares",title=f"k={elbow_k.value} · inertia={inertias[elbow_k.value-1]:.0f}")
    _ax.grid(alpha=.2)
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ##   the elbow is a model-selection clue

    Inertia always decreases as $k$ increases. Select $k$ and look for a **bend** where extra groups buy diminishing improvement.

    $$J_k=\sum_{r=1}^{k}\sum_{i\in C_r}\|x_i-\mu_r\|^2$$

    The elbow is a heuristic; some datasets have no clear bend.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">21 / 30</div>
    """), _controls, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Silhouette: a point-level question

    For point $i$, let $a_i$ be mean distance to its **own** cluster and $b_i$ the smallest mean distance to **another** cluster.

    $$s_i=\frac{b_i-a_i}{\max(a_i,b_i)}\in[-1,1]$$

    Near $1$: well placed. Near $0$: boundary. Below $0$: another cluster is closer on average. Singleton clusters use $s_i=0$ by convention.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">22 / 30</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    point_pick=mo.ui.slider(start=0,stop=5,step=1,value=0,label="Example point",show_value=True)
    return (point_pick,)


@app.cell(hide_code=True)
def _(X_blobs, blob_labels, cdist, mo, plt, point_pick):
    _controls=point_pick
    _choices=[0,12,30,60,95,140];_i=_choices[point_pick.value]
    _d=cdist(X_blobs[[_i]],X_blobs)[0];_own=blob_labels==blob_labels[_i];_a=_d[_own].sum()/(_own.sum()-1)
    _b=min(_d[blob_labels==_r].mean() for _r in set(blob_labels) if _r!=blob_labels[_i]);_s=(_b-_a)/max(_a,_b)
    _fig,_axs=plt.subplots(1,2,figsize=(8.6,3.0),gridspec_kw={"width_ratios":[1.4,1]})
    _ax=_axs[0];_ax.scatter(X_blobs[:,0],X_blobs[:,1],c=blob_labels,cmap="tab10",s=13,alpha=.75)
    _ax.scatter(*X_blobs[_i],facecolors="none",edgecolors="black",s=260,lw=2.2);_ax.set_title(f"Point {_i}");_ax.grid(alpha=.15)
    _ax=_axs[1];_ax.bar(["aᵢ: own","bᵢ: nearest other"],[_a,_b],color=["#527ca3","#e46c55"])
    _ax.set_title(f"silhouette = {_s:.2f}");_ax.set_ylim(0,max(_a,_b)*1.35);_ax.tick_params(axis="x",labelsize=8)
    _fig.tight_layout()
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ##   inspect one silhouette point

    Select a point. The circle marks its own cluster; $a_i$ and $b_i$ compare mean distances, not just the nearest neighbor.

    A negative silhouette suggests the assignment deserves inspection.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">23 / 30</div>
    """), _controls, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(blob_labels, mo, np, plt, silhouette_values):
    _fig,_ax=plt.subplots(figsize=(7.4,3.0))
    _y=0
    for _r in sorted(set(blob_labels)):
        _vals=np.sort(silhouette_values[blob_labels==_r]);_end=_y+len(_vals)
        _ax.fill_betweenx(np.arange(_y,_end),0,_vals,alpha=.7,label=f"Cluster {_r+1}")
        _y=_end+5
    _ax.axvline(float(np.mean(silhouette_values)),color="black",ls="--",label=f"Mean {np.mean(silhouette_values):.2f}")
    _ax.axvline(0,color="#777",lw=.7);_ax.set(xlabel="Silhouette value",ylabel="Points grouped by cluster",title="Silhouette profile")
    _ax.legend(fontsize=8,loc="lower right")
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ## From one silhouette to a full partition

    Plot point silhouettes within each group. Look for a broadly positive distribution and any cluster with many negative members.

    The mean silhouette is convenient, but inspect **cluster-level shapes and sizes** too.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">24 / 30</div>
    """), mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## When reference labels exist: ARI

    The **Adjusted Rand Index** compares pairwise agreements between two partitions and corrects for chance agreement.

    - $1$: identical partitions, even if label names are swapped.
    - About $0$: chance-level agreement under the adjustment model.
    - It can be **negative** for worse-than-chance agreement.

    Reference classes are only a useful target if they match the clustering goal.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">25 / 30</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    flipped_count=mo.ui.slider(start=0,stop=60,step=5,value=0,label="Assignments flipped",show_value=True)
    return (flipped_count,)


@app.cell(hide_code=True)
def _(X_moons, adjusted_rand_score, flipped_count, mo, np, plt, y_moons):
    _controls=flipped_count
    _pred=y_moons.copy();_rng=np.random.default_rng(12);_order=_rng.permutation(len(_pred))
    _pred[_order[:flipped_count.value]]=1-_pred[_order[:flipped_count.value]]
    _score=adjusted_rand_score(y_moons,_pred)
    _fig,_axs=plt.subplots(1,2,figsize=(8.6,3.0),sharex=True,sharey=True)
    for _ax,_lab,_title in zip(_axs,[y_moons,_pred],["Reference","Changed partition"]):
        _ax.scatter(X_moons[:,0],X_moons[:,1],c=_lab,cmap="coolwarm",s=17)
        _ax.set_title(_title);_ax.set_aspect("equal");_ax.grid(alpha=.15)
    _fig.suptitle(f"{flipped_count.value} assignments changed · ARI = {_score:.2f}")
    _fig.tight_layout()
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ##   ARI falls as assignments change

    Start from the two true moon groups, then flip an increasing number of assignments. ARI responds to the **partition**, not the numeric names of labels.

    The same score can hide different error patterns; always inspect the points.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">26 / 30</div>
    """), _controls, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    noise_scale=mo.ui.slider(start=0.0,stop=1.0,step=0.2,value=0.0,label="Added noise",show_value=True)
    return (noise_scale,)


@app.cell(hide_code=True)
def _(
    KMeans,
    X_blobs,
    adjusted_rand_score,
    blob_labels,
    mo,
    noise_scale,
    np,
    plt,
):
    _controls=noise_scale
    _rng=np.random.default_rng(42);_scores=[]
    for _repeat in range(8):
        _X=X_blobs+_rng.normal(scale=noise_scale.value,size=X_blobs.shape)
        _lab=KMeans(n_clusters=3,n_init=5,random_state=_repeat).fit_predict(_X)
        _scores.append(adjusted_rand_score(blob_labels,_lab))
    _fig,_ax=plt.subplots(figsize=(7.4,3.0))
    _ax.scatter(range(1,9),_scores,s=80,color="#527ca3")
    _ax.axhline(np.mean(_scores),color="#e46c55",ls="--",label=f"Mean ARI = {np.mean(_scores):.2f}")
    _ax.set(xlim=(.5,8.5),ylim=(-.05,1.05),xticks=range(1,9),xlabel="Perturbation repeat",ylabel="ARI vs original partition",title=f"Noise standard deviation = {noise_scale.value:.1f}")
    _ax.grid(alpha=.2);_ax.legend()
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ##   test partition stability

    Add measurement noise, refit k-means eight times, and compare each partition with the original using ARI.

    **Stable** means the result survives this perturbation. It does not by itself mean the groups answer the scientific question.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">27 / 30</div>
    """), _controls, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(X_moons, km_moons, mo, plt, silhouette_score, sp_moons):
    _scores=[silhouette_score(X_moons,km_moons),silhouette_score(X_moons,sp_moons)]
    _fig,_ax=plt.subplots(figsize=(7.3,3.0))
    _ax.bar(["K-means","Spectral"],_scores,color=["#527ca3","#e46c55"])
    for _i,_s in enumerate(_scores):_ax.text(_i,_s+.01,f"{_s:.2f}",ha="center")
    _ax.set(ylabel="Euclidean silhouette",ylim=(0,max(_scores)*1.3),title="Same moons, different partitions")
    plt.close(_fig)
    mo.vstack([mo.md(r"""
    ## A good score depends on the geometry

    Euclidean silhouette rewards compact, separated clouds. On curved moons it may prefer the **wrong shape** because its distance notion differs from the graph notion.

    Treat internal metrics as evidence about their own assumptions, not as a universal verdict.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">28 / 30</div>
    """), mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Common validation traps

    - A high score can reflect the wrong **distance or cluster notion**.
    - Choosing parameters on a reference label set and then reporting ARI on those same labels overstates external evidence.
    - High stability can preserve a consistently wrong answer.
    - Inspect outliers, cluster sizes, and sensitivity to feature scaling and graph settings.

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">29 / 30</div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A practical decision sequence

    1. State what a useful group means for this dataset.
    2. Inspect scaling, geometry, graph connectivity, and cluster sizes.
    3. Compare plausible $k$ and graph settings with internal scores **and plots**.
    4. Use ARI only when trusted labels match the goal.
    5. Perturb data and parameters; report instability.

    **Final question:** would these groups support a decision or explanation?

    <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">30 / 30</div>
    """)
    return


if __name__ == "__main__":
    app.run()
