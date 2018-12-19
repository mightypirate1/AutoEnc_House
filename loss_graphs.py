import csv
import pickle
import matplotlib.pyplot as plt
import numpy as np
import settings

latex=True
weight_loss=True
experiments = ["ConvAE", "SAEV"]

def load_losslog(x):
    try:
        with open(x,'rb') as f:
            return pickle.load(f)
    except:
        return [{}]
datasets = {}
for x in [x for x in settings.quick_list if x["name"] in experiments]:
    datasets[x["name"]] = load_losslog("projects/"+x["project_folder"]+"/losslog.pkl")

def smoothen(x, alpha=0.0):
    out = np.array(x)
    z = out[0]
    for i,_ in enumerate(out):
        a = (1-np.e**(-0.3*i))*alpha
        z = a * z + (1-a) * out[i]
        out[i] = z
    return out

max_t = 0
data = {}
for d in datasets:
    data[d] = {}
    for loss in datasets[d][0]:
        if weight_loss:
            if "weight" in loss:
                w = loss['weight']
            else:
                w = 1
        else:
            w = 1
        data[d][loss] = {"t" : [], "val" : [], "weight" : w}
        t = 0
        while t in datasets[d][0][loss]:
            val = datasets[d][0][loss][t] * data[d][loss]["weight"]
            data[d][loss]["t"].append(t)
            data[d][loss]["val"].append(val)
            max_t = max(max_t,t)
            t += 1

for d in data:
    n = len(data[d][list(data[d].keys())[0]]["t"])
    total       =  {"weight" : 0, "val" : [0 for x in range(n)], "t" : [x for x in range(n)]}
    test_total  =  {"weight" : 0, "val" : [0 for x in range(n)], "t" : [x for x in range(n)]}
    for l in data[d]:
        for i,val in enumerate(data[d][l]["val"]):
            if "test" not in l:
                total["weight"] += data[d][l]["weight"]
                total["val"][i] += val*data[d][l]["weight"]
            else:
                test_total["weight"] += data[d][l]["weight"]
                test_total["val"][i] += val*data[d][l]["weight"]
    data[d]["Total loss"] = total
    data[d]["Total loss (test)"] = test_total

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
ax2 = ax.twinx()
ax2.set_ylabel('No need?', fontsize=font_size_axis)

for d in data:
    print(d+"\n-------------------")
    for loss in data[d]:
        print("\t"+loss)
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
        p = ax.plot(data[d][loss]['t'], data[d][loss]['val'], style_str, label=name, linewidth=0.75)
        lines.append(p[0])
        names.append(name)
        # ax2.plot([data[d][loss]['t'][0], data[d][loss]['t'][-1]], [data[d]['max'], data[d]['max']], color=p[0].get_c(), linestyle='--', linewidth=1 )

# plt.yticks([data[x]['max'] for x in data], calculate_relative(data), fontsize=fontsize_ytick )
ax.minorticks_on()
plt.sca(ax)
# plt.xticks(np.linspace(0,max_t,num=6), ["{:}".format(x) for x in np.linspace(0,max_t,num=6)], rotation=15, fontsize=fontsize_xtick)
plt.xticks([x for x in range(max_t) if x%50==0], [x for x in range(max_t) if x%500==0], rotation=0, fontsize=fontsize_xtick)
plt.setp(ax.get_yticklabels(), fontsize=fontsize_ytick)
ax.grid(axis='both')
fig.set_tight_layout(True)
ax.legend(handles=lines, loc=4)
xlim,_ = plt.xlim()
plt.xlim(0,max_t)
ax2.set_ylim(ax.get_ylim())
# performance_calc(data, low='Uniformly random agent', high='Vector + LSTM (Baseline)', latex=latex)
plt.show()
