import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sccm = [1, 2, 4, 8, 23, 58, 127]
M = [1.078, 1.040, 1.031, 1.045, 1.079, 1.053, 1.019]
R = [0.639, 1.251, 2.603, 5.569, 18.224, 46.784, 104.538]

fig, ax1 = plt.subplots(figsize=(7, 5))
ax2 = ax1.twinx()

ax1.plot(sccm, M, "o", color="tab:blue", markersize=9, label="M (Mach number)")
ax2.plot(sccm, R, "s", color="tab:red", markersize=9, label="R (Reynolds number)")

ax1.set_xlabel("He flow rate (SCCM)")
ax1.set_ylabel("Mach number, M", color="tab:blue")
ax2.set_ylabel("Reynolds number, R", color="tab:red")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:red")

ax1.set_ylim(0, 1.5)

ax1.grid(True, which="both", alpha=0.3)
ax1.set_title("Takahashi et al. Fig. 2 reproduction:\nMach and Reynolds number at the aperture vs. He flow rate")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left")

fig.tight_layout()
fig.savefig("results/inspection/takahashi-fig2.png", dpi=150)
print("wrote results/inspection/takahashi-fig2.png")
