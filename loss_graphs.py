import csv
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import settings

latex=True
weight_loss=True
compute_total=False
experiments_to_plot = [
                        # "spatial_ae_conv",
                        "regular_conv_ae_big_bottleneck"
                      ]
losses_to_plot =      [
                        "Error loss",
                        # "Error loss (test)",
                        "Smooth loss",
                        # "Smooth loss (test)",
                        "Presence loss",
                        # "Presence loss (test)",
                        "Total loss",
                        # "Total loss (test)",
                      ]

experiments = {}

for e in experiments_to_plot:
    s = settings.parse_conf( settings.get_conf(e) )
    experiments[s["name"]] = s
def load_losslog(x):
    try:
        with open(x,'rb') as f:
            return pickle.load(f)
    except:
        return {}
datasets = {}
for e in experiments:
    datasets[e] = load_losslog("projects/"+experiments[e]["project_folder"]+"/losslog.pkl")
def smoothen(x, alpha=0.0):
    out = np.array(x)
    z = out[0]
    for i,_ in enumerate(out):
        a = (1-np.e**(-0.3*i))*alpha
        z = a * z + (1-a) * out[i]
        out[i] = z
    return out

max_t = 0
max_val = 0
data = {}
for d in datasets:
    data[d] = {}
    for loss in datasets[d]:
        if "weight" in datasets[d][loss] and weight_loss:
            w = datasets[d][loss]['weight']
        else:
            w = 1
        if datasets[d][loss]['weight'] == 0.0:
            continue
        data[d][loss] = {"t" : [], "val" : [], "weight" : w}
        t = 0
        while t in datasets[d][loss]:
            val = datasets[d][loss][t]
            data[d][loss]["t"].append(t)
            data[d][loss]["val"].append(val)
            if loss in losses_to_plot:
                max_t = max(max_t,t)
                max_val = max(max_val, val)
            t += 1

if compute_total:
    for d in data:
        n = data[d][list(data[d].keys())[0]]["t"][-1]+1
        test_total = [0]*n
        total = [0]*n
        for l in data[d]:
            if "(test)" in l:
                tot = test_total
            else:
                tot = total
            w = data[d][l]["weight"]
            for i,val in enumerate(data[d][l]["val"]):
                tot[i] += w*val
        data[d]["Total loss"] = {"val" : total, "t" : [t for t in range(n)], "weight" : 1.0}
        data[d]["Total loss (test)"] = {"val" : test_total, "t" : [t for t in range(n)], "weight" : 1.0}

# Plot stuff!
fontsize_ytick = 14
fontsize_xtick = 12
font_size_axis = 18
lines = []
names = []
plt.style.use(plt.style.available[0])
fig, ax = plt.subplots()
ax.set_xlabel('Time steps', fontsize=font_size_axis)
ax.set_ylabel('Loss', fontsize=font_size_axis)
palette = ["red", "green", "blue"]
colors = []

for d in data:
    print(d+"\n-------------------")
    z = None
    for loss in data[d]:
        if loss not in losses_to_plot:
            continue
        print("\t"+loss+ " w:"+str(data[d][loss]["weight"]) )
        style_str = ""
        name = "{}: {}".format(d,loss)
        if "error" in loss:
            style_str += "r"
        if "smooth" in loss:
            style_str += "g"
        if "presence" in loss:
            style_str += "b"
        if "test" in loss:
            style_str += "--"
        else:
            style_str += "-"
        y = [x*(data[d][loss]['weight'] if weight_loss else 1) for x in data[d][loss]['val']]
        if z is None:
            z = [data[d][loss]["t"]]
        z.append(y)
        colors.append(palette[len(names)])
        names.append(name)
        # p = ax.plot(data[d][loss]['t'], y, style_str, label=name, linewidth=1.75)
        # lines.append(p[0])
        # ax2.plot([data[d][loss]['t'][0], data[d][loss]['t'][-1]], [data[d]['max'], data[d]['max']], color=p[0].get_c(), linestyle='--', linewidth=1 )
    ax.stackplot(*z, labels=names, colors=colors)

plt.legend(handles=[mpatches.Patch(color=c, label=l) for c,l in zip(colors,names)])
plt.show()
# plt.yticks([data[x]['max'] for x in data], calculate_relative(data), fontsize=fontsize_ytick )
ax.minorticks_on()
plt.sca(ax)
# plt.xticks(np.linspace(0,max_t,num=6), ["{:}".format(x) for x in np.linspace(0,max_t,num=6)], rotation=15, fontsize=fontsize_xtick)
plt.xticks([x for x in range(max_t) if x%50==0], [x for x in range(max_t) if x%50==0], rotation=0, fontsize=fontsize_xtick)
plt.setp(ax.get_yticklabels(), fontsize=fontsize_ytick)
ax.grid(axis='both')
fig.set_tight_layout(True)
ax.legend(handles=lines, loc=4)
xlim,_ = plt.xlim()
ax.set_xlim([0, max_t])
ax.set_ylim([0, max_val])
# performance_calc(data, low='Uniformly random agent', high='Vector + LSTM (Baseline)', latex=latex)
plt.show()
