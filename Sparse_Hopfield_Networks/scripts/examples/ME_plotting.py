import argparse
from Memory_evaluation import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle

def get_args():
    parser = argparse.ArgumentParser(description='MNIST arguments:')
    parser.add_argument('--dataset', default='ImageNet', type=str, choices=["MNIST", "CIFAR10", "ImageNet"])
    parser.add_argument('--yomega', default='entmax', type=str, choices=["identity", "entmax", "normmax", "10_poly", "5_poly", "exp"])
    parser.add_argument('--alpha', help='alpha', default=2, type=float)
    parser.add_argument('--ypsi', default='normalization', type=str, choices=["none", "normalization", "layer_normalization", "tanh", "identity"])
    parser.add_argument('--beta', help='inverse of temperature', default=100, type=float)
    parser.add_argument('--inner_beta', help='inverse of temperature for the exp DAM inner operation', default=0.05, type=float)
    parser.add_argument('--perc', help="mas percentage", default=0.5, type=float)
    parser.add_argument('--std', help='std for noise in queries', default=0.5, type=float)
    parser.add_argument('--memory_size', help='number of memories', default=100, type=int)
    parser.add_argument('--device', help='gpu', default=2, type=int)
    args = parser.parse_args()
    return args

data = {"tanh": ["identity", "10_poly", "exp"],
        #"identity": ["entmax", "normmax"],
        #"normalization": ["entmax", "normmax"],
        #"layer_normalization": ["entmax", "normmax"], 
        }

def memory_capacity(args):

    results = {}
    args.std = 0

  
    beta = args.beta
    #---------------------------MNIST--------------------------------            
    datasets = ["ImageNet", "MNIST", "CIFAR10"]
    for dataset in datasets:
        if dataset == "MNIST":
            memory_sizes = [2, 4, 8,16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
            args.inner_beta = 0.05
        else:
            memory_sizes = [2, 4, 8,16, 32, 64, 128, 256, 512, 1024]
            args.inner_beta = 0.01

        args.dataset = dataset
        results[dataset] = {}
        for key, value in data.items():
            results[args.dataset][key] = {}
            if key == "normalization":
                args.beta = 20
            else:
                args.beta = beta
            for yomega in value:
                args.yomega = yomega
                args.ypsi =  key
                if yomega ==  "entmax":
                    alphas = [1, 2]
                    for alpha in alphas:
                        args.alpha = alpha
                        results[args.dataset][key][str(alpha) + "-" + yomega] = {"success_rate": [],
                                                                    "CI": []}
                        for memory_size in memory_sizes:
                            args.memory_size = memory_size
                            acc = Hopfield(args)
                            results[args.dataset][key][str(alpha) + "-" + yomega]["success_rate"].append(np.median(acc))
                            results[args.dataset][key][str(alpha) + "-" + yomega]["CI"].append([np.percentile(acc, 25), np.percentile(acc, 75)])
                elif yomega ==  "normmax":
                    alphas = [2,5]
                    for alpha in alphas:
                        args.alpha = alpha
                        results[args.dataset][key][str(alpha) + "-" + yomega] = {"success_rate": [],
                                                                    "CI": []}
                        for memory_size in memory_sizes:
                            args.memory_size = memory_size
                            acc = Hopfield(args)
                            results[args.dataset][key][str(alpha) + "-" + yomega]["success_rate"].append(np.median(acc))
                            results[args.dataset][key][str(alpha) + "-" + yomega]["CI"].append([np.percentile(acc, 25), np.percentile(acc, 75)])
                else:
                    results[args.dataset][key][yomega] = {"success_rate": [],
                                                        "CI": []}
                    for memory_size in memory_sizes:
                        args.memory_size = memory_size
                        acc = Hopfield(args)
                        results[args.dataset][key][yomega]["success_rate"].append(np.median(acc))
                        results[args.dataset][key][yomega]["CI"].append([np.percentile(acc, 25), np.percentile(acc, 75)])

    return results

def memory_robustness(args):

    results = {}
    beta = args.beta
    stds = [0, 2.5 ,5,  7.5, 10, 12.5, 15]
    args.memory_size = 100
    #---------------------------MNIST--------------------------------            
    datasets = ["ImageNet", "MNIST", "CIFAR10"]
    for dataset in datasets:
        if dataset == "MNIST":
            args.inner_beta = 0.05
        else:
            args.inner_beta = 0.01

        args.dataset = dataset
        results[dataset] = {}
        for key, value in data.items():
            results[args.dataset][key] = {}
            if key == "normalization":
                args.beta = 20
            else:
                args.beta = beta
            for yomega in value:
                args.yomega = yomega
                args.ypsi =  key
                if yomega ==  "entmax":
                    alphas = [1, 2]
                    for alpha in alphas:
                        args.alpha = alpha
                        results[args.dataset][key][str(alpha) + "-" + yomega] = {"success_rate": [],
                                                                    "CI": []}
                        for std in stds:
                            args.std = std
                            acc = Hopfield(args)
                            results[args.dataset][key][str(alpha) + "-" + yomega]["success_rate"].append(np.median(acc))
                            results[args.dataset][key][str(alpha) + "-" + yomega]["CI"].append([np.percentile(acc, 25), np.percentile(acc, 75)])
                elif yomega ==  "normmax":
                    alphas = [2,5]
                    for alpha in alphas:
                        args.alpha = alpha
                        results[args.dataset][key][str(alpha) + "-" + yomega] = {"success_rate": [],
                                                                    "CI": []}
                        for std in stds:
                            args.std = std
                            acc = Hopfield(args)
                            results[args.dataset][key][str(alpha) + "-" + yomega]["success_rate"].append(np.median(acc))
                            results[args.dataset][key][str(alpha) + "-" + yomega]["CI"].append([np.percentile(acc, 25), np.percentile(acc, 75)])
                else:
                    results[args.dataset][key][yomega] = {"success_rate": [],
                                                        "CI": []}
                    for std in stds:
                        args.std = std
                        acc = Hopfield(args)
                        results[args.dataset][key][yomega]["success_rate"].append(np.median(acc))
                        results[args.dataset][key][yomega]["CI"].append([np.percentile(acc, 25), np.percentile(acc, 75)])

    return results
                
def plotting(args, data):
    datasets = ["MNIST", "CIFAR10", "ImageNet"]

    # Create a larger figure
    fig, axs = plt.subplots(2, 3, figsize=(3*12, 2*10), sharex=False, sharey=True)
    sns.set_theme(style='whitegrid', context='talk', font='sans-serif', font_scale=1.0, palette="pastel")
    sns.despine(left=False, top=True, right=True, bottom=False)
    # Iterate through datasets, methods, and yomegas to plot data
    labels_all = ["Classic HNs", "10-Poly DAM", "Exp DAM", "1-entmax", "2-entmax", "2-normmax", "5-normmax", "norm 2-entmax", "lnorm 2-entmax"]
    methods = ["tanh", "identity", "normalization", "layer_normalization"]
    colors = [
    '#1f77b4', '#ff7f0e', '#e377c2', '#d62728', '#9467bd', '#8c564b', '#2ca02c', '#7f7f7f', '#bcbd22']
    markers = ['o', '^', 'v', 's', '*', 'x', 'D', 'P', 'h']
    for p, beta in enumerate([1, 0.1]):
        results = data[str(beta)]["memory_capacity"]
        for i, dataset in enumerate(datasets):
            if dataset == "MNIST":
                x_axis = [2, 4, 8,16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
            else:
                x_axis = [2, 4, 8,16, 32, 64, 128, 256, 512, 1024]
            l = 0
            for j, method in enumerate(methods):
                if method == "normalization" or method == "layer_normalization":
                    yomegas = ["2-entmax"]
                else:
                    yomegas = list(results[datasets[0]][method].keys())
                for k, yomega in enumerate(yomegas):
                    retrieval_mean = results[dataset][method][yomega]['success_rate']
                    retrieval_std = results[dataset][method][yomega]['CI']
                    first_elements = [sublist[0] for sublist in retrieval_std]
                    second_elements = [sublist[1] for sublist in retrieval_std]
                    marker_indices = (np.arange(0, len(x_axis), 3) + l) % len(x_axis)
                    # Create subplot
                    axs[p,i].plot(x_axis, retrieval_mean, label= labels_all[l], color= colors[l], marker=markers[l], markersize=15, markevery=marker_indices, alpha=0.7, linewidth=3)   
                    axs[p,i].fill_between(x_axis, first_elements,
                                        second_elements, alpha=0.1, linewidth=0, color= colors[l], label='_nolegend_')
                    l += 1
                    if p==1:
                        axs[p,i].set_xlabel("Number of Memories", fontsize=40)
                        axs[p,i].set_xticks(x_axis)
                    else:
                        axs[p, i].tick_params(labelbottom=False)
                        axs[p,i].set_title(r'{}'.format(dataset), fontsize= 45)

                    axs[p,i].set_ylim(bottom=0, top=1.1)
                    axs[p,i].grid(True, color='white')  # Display the white grid
                    axs[p,i].set_facecolor('#f0f0f0')  # Set the background color
                    axs[p,i].set_xscale('log', base=2)
                    axs[p, i].set_xticks(x_axis, minor=False)
                    axs[p,i].tick_params(axis='y', labelsize=35, which='major')
                    axs[p,i].tick_params(axis='x', labelsize=35, which='major', pad=10)

    handles, labels = axs[p, i].get_legend_handles_labels()
    axs[0, 0].set_ylabel(r"$\beta=1$", fontsize=45, fontweight = "extra bold")
    axs[1, 0].set_ylabel(r"$\beta=0.1$", fontsize=45, fontweight = "extra bold")
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.52, 0.0175), ncol=5, fontsize=40)
    fig.text(0.015, 0.56, "Success Retrieval Rate", va='center', rotation='vertical', fontsize=45)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, left=0.09)  # Adjust to create space for legend
    plt.savefig('figure1.pdf')

    labels = ["norm", "lnorm", "identity"]
    results = data["1"]["memory_capacity"]
     # Create a larger figure
    fig, axs = plt.subplots(4, 3, figsize=(3*12, 4*8), sharex=False, sharey=True)
    sns.set_theme(style='whitegrid', context='talk', font='sans-serif', font_scale=1.0, palette="pastel")
    sns.despine(left=False, top=True, right=True, bottom=False)
    methods = ["normalization", "layer_normalization", "identity"] 
    colors_detail = ['red', 'black', 'blue']      
    for i, dataset in enumerate(datasets):
        if dataset == "MNIST":
            x_axis = [2, 4, 8,16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        else:
            x_axis = [2, 4, 8,16, 32, 64, 128, 256, 512, 1024]
        l = 0
        results = data[str(1)]["memory_capacity"]
        yomegas = ["1-entmax", "2-entmax", "2-normmax", "5-normmax"]
        for j, yomega in enumerate(yomegas):
            for k, method in enumerate(methods):
                retrieval_mean = results[dataset][method][yomega]['success_rate']
                retrieval_std = results[dataset][method][yomega]['CI']
                first_elements = [sublist[0] for sublist in retrieval_std]
                second_elements = [sublist[1] for sublist in retrieval_std]

                axs[j,i].plot(x_axis, retrieval_mean, label= labels[k], color=colors_detail[k], marker=markers[k], markersize=15, alpha=0.7, linewidth=3)   
                l += 1
                axs[j,i].fill_between(x_axis, first_elements,
                                    second_elements, alpha=0.25, color=colors_detail[k], linewidth=0, label='_nolegend_')

                axs[j, i].set_title(f'{yomega} - {dataset}', fontsize= 40)
                if i==0 and j==0:
                    axs[j, i].legend(fontsize=32, loc='lower left')
                if j==3:
                    axs[j, i].set_xlabel("Number of Memories", fontsize=40)
                    axs[j,i].set_xticks(x_axis)
                else:
                    axs[j, i].tick_params(labelbottom=False)  
                axs[j,i].set_ylim(bottom=0, top=1.1)
                axs[j,i].grid(True, color='white')  # Display the white grid
                axs[j,i].set_facecolor('#f0f0f0')  # Set the background color
                axs[j,i].set_xscale('log', base=2)
                axs[j, i].set_xticks(x_axis, minor=False)
                axs[j,i].tick_params(axis='y', labelsize=35, which='major')
                axs[j,i].tick_params(axis='x', labelsize=35, which='major', pad=10)

    plt.tight_layout()
    fig.text(0.01, 0.5, "Success Retrieval Rate", va='center', rotation='vertical', fontsize=45)
    plt.subplots_adjust(left=0.07)  # Adjust to create space for legend
    plt.savefig('figure2.pdf')
    
    # Create a larger figure
    fig, axs = plt.subplots(2, 3, figsize=(3*12, 2*10), sharex=True, sharey=True)
    sns.set_theme(style='whitegrid', context='talk', font='sans-serif', font_scale=1.0, palette="pastel")
    sns.despine(left=False, top=True, right=True, bottom=False)
    for p, beta in enumerate([1, 0.1]):
        results = data[str(beta)]["memory_robustness"]
        for i, dataset in enumerate(datasets):
            x_axis = [0, 2.5 ,5,  7.5, 10, 12.5, 15]
            methods = ["tanh", "identity", "normalization", "layer_normalization"]
            l = 0
            for j, method in enumerate(methods):
                if method == "normalization" or method == "layer_normalization":
                    yomegas = ["2-entmax"]
                else:
                    yomegas = list(results[datasets[0]][method].keys())
                for k, yomega in enumerate(yomegas):
                    retrieval_mean = results[dataset][method][yomega]['success_rate']
                    retrieval_std = results[dataset][method][yomega]['CI']
                    first_elements = [sublist[0] for sublist in retrieval_std]
                    second_elements = [sublist[1] for sublist in retrieval_std]
                    marker_indices = (np.arange(0, len(x_axis), 3) + l) % len(x_axis)
                    # Create subplot
                    axs[p,i].plot(x_axis, retrieval_mean, label= labels_all[l], color= colors[l], marker=markers[l], markersize=15, markevery=marker_indices, alpha=0.7, linewidth=3)   
                    axs[p, i].fill_between(x_axis, first_elements,
                                        second_elements, alpha=0.1, linewidth=0, color= colors[l], label='_nolegend_')
                    l += 1
                    if p ==1:
                        axs[p,i].set_xlabel("Noise std ($\sigma$)", fontsize=40)
                        axs[p,i].set_xticks(x_axis)
                    else:
                        axs[p, i].tick_params(labelbottom=False)
                        axs[p,i].set_title(r'{}'.format(dataset), fontsize= 45)
                    axs[p,i].set_ylim(bottom=0, top=1.1)
                    axs[p,i].grid(True, color='white')  # Display the white grid
                    axs[p,i].set_facecolor('#f0f0f0')  # Set the background color
                    axs[p, i].set_xticks(x_axis, minor=False)
                    axs[p,i].tick_params(axis='y', labelsize=35, which='major')
                    axs[p,i].tick_params(axis='x', labelsize=35, which='major', pad=10)

    handles, labels = axs[p, i].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.52, 0.0175), ncol=5, fontsize=40)
    fig.text(0.015, 0.56, "Success Retrieval Rate", va='center', rotation='vertical', fontsize=45)
    axs[0, 0].set_ylabel(r"$\beta=1$", fontsize=45, fontweight = "extra bold")
    axs[1, 0].set_ylabel(r"$\beta=0.1$", fontsize=45, fontweight = "extra bold")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, left=0.09)  # Adjust to create space for legend
    plt.savefig('figure3.pdf')
    
    labels = ["norm", "lnorm", "identity"]
    results = data[str(1)]["memory_robustness"]
    fig, axs = plt.subplots(4, 3, figsize=(3*12, 4*8), sharex=False, sharey=True)
    sns.set_theme(style='whitegrid', context='talk', font='sans-serif', font_scale=1.0, palette="pastel")
    sns.despine(left=False, top=True, right=True, bottom=False)
    methods = ["normalization", "layer_normalization", "identity"] 
    colors_detail = ['red', 'black', 'blue']      
    for i, dataset in enumerate(datasets):
        x_axis=[0, 2.5 ,5,  7.5, 10, 12.5, 15]
        l = 0
        yomegas = ["1-entmax", "2-entmax", "2-normmax", "5-normmax"]
        for j, yomega in enumerate(yomegas):
            for k, method in enumerate(methods):
                retrieval_mean = results[dataset][method][yomega]['success_rate']
                retrieval_std = results[dataset][method][yomega]['CI']
                first_elements = [sublist[0] for sublist in retrieval_std]
                second_elements = [sublist[1] for sublist in retrieval_std]

                axs[j,i].plot(x_axis, retrieval_mean, label= labels[k], color=colors_detail[k], marker=markers[k], markersize=15, alpha=0.7, linewidth=3)   
                l += 1
                axs[j,i].fill_between(x_axis, first_elements,
                                    second_elements, alpha=0.25, color=colors_detail[k], linewidth=0, label='_nolegend_')

                axs[j, i].set_title(f'{yomega} - {dataset}', fontsize= 40)
                if i==0 and j==0:
                    axs[j, i].legend(fontsize=32, loc='upper right')
                if j==3:
                    axs[j, i].set_xlabel("Noise std ($\sigma$)", fontsize=40)
                    axs[j,i].set_xticks(x_axis)
                else:
                    axs[j, i].tick_params(labelbottom=False)  
                axs[j,i].set_ylim(bottom=0, top=1.1)
                axs[j,i].grid(True, color='white')  # Display the white grid
                axs[j,i].set_facecolor('#f0f0f0')  # Set the background color
                axs[j, i].set_xticks(x_axis, minor=False)
                axs[j,i].tick_params(axis='y', labelsize=35, which='major')
                axs[j,i].tick_params(axis='x', labelsize=35, which='major', pad=10)

    plt.tight_layout()
    fig.text(0.01, 0.5, "Success Retrieval Rate", va='center', rotation='vertical', fontsize=45)
    plt.subplots_adjust(left=0.07)  # Adjust to create space for legend
    plt.savefig('figure4.pdf')
if __name__ == '__main__':
    args = get_args()
    args.device =  "cuda:" + str(args.device)
    betas = [0.1, 1]
    data_final = {}
    for beta in betas:
        args.beta = beta
        results = memory_capacity(args)
        data_results = {}
        data_results["memory_capacity"]=results
        results = memory_robustness(args)
        data_results["memory_robustness"] = results
        data_final[str(beta)] = data_results
    
    plotting(args, data_final)