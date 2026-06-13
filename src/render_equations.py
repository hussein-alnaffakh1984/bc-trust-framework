import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, json

os.makedirs("eq", exist_ok=True)
plt.rcParams["mathtext.fontset"] = "cm"

EQS = {
 "eq1": r"$\varphi_{j}^{(r)} \;=\; \dfrac{1}{N}\sum_{i}\,\left|\,s_{ij}^{(r)}\,\right|$",
 "eq2": r"$CV_{j} \;=\; \dfrac{\sigma_{j}}{\bar{\varphi}_{j}} \qquad m_{j} \;=\; \dfrac{1}{1+CV_{j}}$",
 "eq3": r"$S_{mag} \;=\; \dfrac{\sum_{j}\,\bar{\varphi}_{j}\,m_{j}}{\sum_{j}\,\bar{\varphi}_{j}}$",
 "eq4": r"$\bar{\tau} \;=\; \dfrac{2}{R(R-1)}\sum_{r<r'}\tau\!\left(\pi^{(r)},\pi^{(r')}\right) \qquad S_{rank} \;=\; \dfrac{1+\bar{\tau}}{2}$",
 "eq5": r"$KI \;=\; \dfrac{\left|\,T^{(r)}\cap T^{(r')}\,\right| \;-\; k^{2}/p}{k \;-\; k^{2}/p}$",
 "eq6": r"$\hat{\Phi} \;=\; 1 \;-\; \dfrac{\dfrac{1}{p}\sum_{j} s_{j}^{2}}{\dfrac{\bar{k}}{p}\left(1-\dfrac{\bar{k}}{p}\right)}$",
 "eq7": r"$ESI \;=\; S_{mag}^{\,w_{1}}\cdot S_{rank}^{\,w_{2}}\cdot S_{sel}^{\,w_{3}}\,, \qquad w_{1}+w_{2}+w_{3}=1$",
 "eq8": r"$ESI_{rel} \;=\; \dfrac{ESI \;-\; \mathbb{E}[ESI_{0}]}{1 \;-\; \mathbb{E}[ESI_{0}]}$",
 "eq9": r"$f_{j} \;=\; \dfrac{1}{R}\sum_{r}\mathbf{1}\!\left[\,j\in T^{(r)}\,\right] \qquad \psi_{j} \;=\; f_{j}\,m_{j}$",
 "eq10": r"$score_{j} \;=\; \alpha\,\tilde{\varphi}_{j} \;+\; (1-\alpha)\,\psi_{j}$",
}

DPI = 300
COLOR = "#10243a"
info = {}
for name, tex in EQS.items():
    fig = plt.figure(figsize=(0.1, 0.1))
    t = fig.text(0, 0, tex, fontsize=15, color=COLOR)
    fig.canvas.draw()
    bbox = t.get_window_extent()
    w_in = bbox.width / DPI
    h_in = bbox.height / DPI
    plt.close(fig)
    fig = plt.figure(figsize=(w_in + 0.08, h_in + 0.06), dpi=DPI)
    fig.text(0.5, 0.5, tex, fontsize=15, color=COLOR, ha="center", va="center")
    path = f"eq/{name}.png"
    fig.savefig(path, dpi=DPI, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    # record pixel size
    from PIL import Image
    im = Image.open(path)
    info[name] = {"w": im.width, "h": im.height}

json.dump(info, open("eq/info.json", "w"))
print(json.dumps(info))
