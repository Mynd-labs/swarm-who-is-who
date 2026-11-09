import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R=json.load(open("results.json")); S=["Name","Operator","Key","Fingerprint","Key+Fingerprint"]
def ms(v): v=np.array(v,float); return v.mean(), 1.96*v.std(ddof=1)/np.sqrt(len(v))
col={"Name":"#888","Operator":"#c9a227","Key":"#1f77b4","Fingerprint":"#d62728","Key+Fingerprint":"#2ca02c"}
fig,ax=plt.subplots(2,2,figsize=(7.2,5.6))
for a,(k,xl) in zip(ax.flat,[("rename","rename prob. per round"),("clones","clone fraction"),("noise","fingerprint noise $\\sigma$"),("rotation","key rotation prob. per round")]):
    xs=list(R[k].keys())
    for s in S:
        m=[ms(R[k][x][s]["acc"]) for x in xs]
        a.errorbar([float(x) for x in xs],[i[0] for i in m],yerr=[i[1] for i in m],label=s,color=col[s],marker="o",ms=3,lw=1.2,capsize=2)
    a.set_xlabel(xl,fontsize=8); a.set_ylabel("attribution accuracy",fontsize=8); a.set_ylim(0,1.02); a.tick_params(labelsize=7); a.grid(alpha=.3)
ax[0,0].legend(fontsize=6.5,loc="center right")
plt.tight_layout(); plt.savefig("fig1.pdf"); plt.savefig("fig1.png",dpi=140)
fig,ax=plt.subplots(1,2,figsize=(7.2,2.6))
xs=list(R["impersonation"].keys())
for s in ["Name","Key","Fingerprint","Key+Fingerprint"]:
    m=[ms(R["impersonation"][x][s]["framed"] if x!="0.0" else [0]*30) for x in xs]
    ax[0].errorbar([float(x) for x in xs],[i[0] for i in m],yerr=[i[1] for i in m],label=s,color=col[s],marker="o",ms=3,capsize=2)
ax[0].set_xlabel("impersonator fraction",fontsize=8); ax[0].set_ylabel("forged events blamed on victim",fontsize=8); ax[0].legend(fontsize=6.5); ax[0].tick_params(labelsize=7); ax[0].grid(alpha=.3)
xs=list(R["rename"].keys())
for s in ["Name","Key","Fingerprint"]:
    m=[ms(R["rename"][x][s]["frag"]) for x in xs]
    ax[1].errorbar([float(x) for x in xs],[i[0] for i in m],yerr=[i[1] for i in m],label=s,color=col[s],marker="o",ms=3,capsize=2)
ax[1].set_xlabel("rename prob. per round",fontsize=8); ax[1].set_ylabel("identities per true agent",fontsize=8); ax[1].legend(fontsize=6.5); ax[1].tick_params(labelsize=7); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("fig2.pdf"); plt.savefig("fig2.png",dpi=140)
d=R["default_run"]["0.1"]
print("TABLE default")
for s in S:
    print(s," & ".join("%.2f $\\pm$ %.2f"%ms(d[s][m]) for m in ["acc","cont","frag","framed"]))
for k in ["rename","clones","noise","rotation"]:
    print(k)
    for x in R[k]: print(x,{s:"%.2f±%.2f"%ms(R[k][x][s]["acc"]) for s in S})
print("framed imp")
for x in R["impersonation"]: print(x,{s:"%.3f"%ms(R["impersonation"][x][s]["framed"])[0] for s in S})
