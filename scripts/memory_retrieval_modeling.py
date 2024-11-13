import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import argparse
import torchvision
from utils import *
import pickle
import seaborn as sns
import textdistance
import statistics

def get_args():
    parser = argparse.ArgumentParser(description='MNIST arguments:')
    parser.add_argument('--paradigm', default="free_recall", type=str, choices=["free_recall", "seq_recall"])
    parser.add_argument('--dataset', default="CIFAR10", type=str, choices=["MNIST", "CIFAR10", "ImageNet"])
    parser.add_argument('--method', default='entmax', type=str, choices=["entmax", "csparsemax"])
    parser.add_argument('--alpha', help='alpha', default=2, type=float)
    parser.add_argument('--beta', help='inverse of temperature', default=0.1, type=float)
    parser.add_argument('--memory_size', help='number of memories', default=128, type=int)
    parser.add_argument('--max_iter', help='maximum number of inner iterations', default=20, type=int)
    parser.add_argument('--gamma', help='step size', default=1e9, type=float)
    parser.add_argument('--decay_rate', help='decay_rate', default=0.001, type=int)
    parser.add_argument('--omega', help='bonus parameter for seq recall', default=1.1, type=float)
    parser.add_argument('--device', help='gpu', default=4, type=int)
    args = parser.parse_args()
    return args


def load_MNIST():
    # Define the transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5,0.5),  # Normalize to [-1, 1]
    ])

    # Load the MNIST dataset
    mnist_dataset = datasets.MNIST(root='../datasets', train=True, download=False, transform=transform)
    mnist_dataset_test = datasets.MNIST(root='../datasets', train=False, download=False, transform=transform)

    # Create a DataLoader to iterate over the dataset
    data_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=len(mnist_dataset), shuffle=True)
    # Create a DataLoader to iterate over the dataset
    data_loader_test = torch.utils.data.DataLoader(mnist_dataset_test, batch_size=len(mnist_dataset_test), shuffle=True)

    for data in data_loader:
        X_train, labels_train = data

    for data in data_loader_test:
        X_test, labels_test = data
    
    return X_train, labels_train

def load_tiny_ImageNet():
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.ImageFolder(root='../datasets/tiny-imagenet-200/train',
                                                transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset),
                                              shuffle=True, num_workers=4)

    train_data = next(iter(trainloader))

    testset = torchvision.datasets.ImageFolder(root='../datasets/tiny-imagenet-200/test',
                                               transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                             shuffle=True, num_workers=4)

    test_data = next(iter(testloader))

    return train_data[0], test_data[0]

def load_cifar10():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    trainset = torchvision.datasets.CIFAR10(root='../datasets', train=True,
                                            download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset),
                                      shuffle=True)
    train_data = next(iter(trainloader))
    testset = torchvision.datasets.CIFAR10(root='../datasets', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                      shuffle=True)
    test_data = next(iter(testloader))
    
    return train_data[0], test_data[0]

def seq_recall(args, X_train):
    max_iter = args.max_iter
    gamma = args.gamma
    decay_rate= args.decay_rate
    num_patterns = args.memory_size
    
    X_train = X_train.view(num_patterns, -1)
    Q = X_train[0].flatten().view(X_train.shape[1], 1)
    Q_values_all = []
    memories_trigered = []
    Q_hopfield = []
    avg = torch.zeros_like(X_train[:, 0]).unsqueeze(1).to(args.device)
    for _ in range(num_patterns):
        Q_values = []
        pair = SparseMAP_sequence_exactly_k(args.beta*(X_train.mm(Q) - gamma*avg), 1e8, 2).detach()
        Q = X_train.T @ pair - Q
        for _ in range(max_iter):
            p = entmax(X_train.mm(Q), dim=0, alpha = args.alpha)
            Q = X_train.T @ p
            Q_values.append(Q)
        avg =  decay_rate * (pair-args.omega*p)  +(1 - decay_rate) * avg
        Q_values_all.append(Q_values)
        memories_trigered.append(torch.argmax(p).item())
    return memories_trigered, Q_hopfield, Q_values_all


def free_recall(args, X_train):
    max_iter = args.max_iter
    gamma = args.gamma
    decay_rate= args.decay_rate
    num_patterns = args.memory_size
    
    X_train = X_train.view(num_patterns, -1)
    Q = X_train[0].flatten().view(X_train.shape[1], 1)
    Q_values_all = []
    memories_trigered = []
    Q_hopfield = []
    if args.method == "entmax":
        avg = torch.zeros_like(X_train[:, 0]).unsqueeze(1).to(args.device)
    else:
        upper = torch.ones(1, num_patterns, dtype=float).to(args.device)

    for _ in range(num_patterns):
        Q_values = []
        if args.method == "entmax":
            p = entmax(args.beta*(X_train.mm(Q) - gamma * avg), alpha=args.alpha, dim=0)
            avg =  decay_rate * p  +(1 - decay_rate) * avg
            Q = X_train.T @ p 
        for _ in range(max_iter):
            if args.method == "entmax":
                p = entmax(args.beta*X_train.mm(Q), alpha=args.alpha, dim=0)
            else:
                p = constrained_sparsemax_function(args.beta*X_train.mm(Q).T, upper).T
            Q = X_train.T @ p 
            Q_values.append(Q)
        if args.method != "entmax":
            upper -= p.T
            upper = upper.clamp(min=0.)
        Q_hopfield.append(Q)
        Q_values_all.append(Q_values)
        memories_trigered.append(torch.argmax(p).item())
    return memories_trigered, Q_hopfield, Q_values_all

def entmax_mapping(args):
    if args.method == "softmax":
        args.alpha = 1
        args.method = "entmax"
    elif args.method == "entmax":
        args.alpha = 1.5
        args.method = "entmax"
    elif args.method == "sparsemax":
        args.alpha = 2
        args.method = "entmax"
    return args

def Hopfield(args):
    torch.random.manual_seed(42)
    if args.dataset == "MNIST":        
        X, _ = load_MNIST()
    
    elif args.dataset == "CIFAR10":
        X,_ = load_cifar10()
    
    elif args.dataset == "ImageNet":
        X,_ = load_tiny_ImageNet()
    
    acc = []
    taus = []
    for i in range(5):
        torch.manual_seed(i)

        # Shuffle the dataset
        all_indices = torch.randperm(X.shape[0])

        # Select the first 10 indices
        indices = all_indices[:args.memory_size]
        X_train = X[indices].to(args.device)
        if args.paradigm == "free_recall":
            x, _, _ = free_recall(args, X_train)
        else:
            x, _, _ = seq_recall(args, X_train)
            gt = list(range(1, len(x) + 1))
            distance = textdistance.levenshtein.distance(x, gt)
            taus.append(1 - distance / len(x))
        acc.append(len(set(x))/args.memory_size)
    
    print(f"Unique Memory Ratio - Mean: {np.mean(acc).item()}, Standard Deviation: {np.std(acc).item()}")
    if args.paradigm == "seq_recall":
        print(f"Levensthein Coefficient - Mean: {np.mean(taus).item()}, Standard Deviation: {np.std(taus).item()}")
        return acc, taus
    return acc
def memory_size_sentivity(args):
    memory_sizes = [2,4,8,16,32,64,128,256,512,1024]
    methods = ["csparsemax", "entmax", "softmax", "sparsemax"]
    datasets = ["MNIST", "ImageNet",  "CIFAR10"]
    results = {}

    for dataset in datasets:
        args.dataset = dataset
        results[dataset] = {}
        for method in methods:
            results[dataset][method]= {"acc": [], "CI":[]}
            args.method = method
            args = entmax_mapping(args)
            for memory_size in memory_sizes:
                args.memory_size=memory_size
                acc = Hopfield(args)
                results[dataset][method]["acc"].append(statistics.median(acc))
                results[dataset][method]["CI"].append([np.percentile(acc, 25), np.percentile(acc, 75)])
    return results

def beta_sensitivity(args):
    args.memory_size = 128
    betas = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
    methods = ["csparsemax", "entmax", "softmax", "sparsemax"]
    datasets = ["ImageNet", "MNIST", "CIFAR10"]
    results = {}

    for dataset in datasets:
        args.dataset = dataset
        results[dataset] = {}
        for method in methods:
            results[dataset][method]= {"acc": [], "CI":[]}
            args.method = method
            args = entmax_mapping(args)
            for beta in betas:
                args.beta = beta
                acc = Hopfield(args)
                results[dataset][method]["acc"].append(statistics.median(acc))
                results[dataset][method]["CI"].append([np.percentile(acc, 25), np.percentile(acc, 75)])
    return results

def MS_plotting(memory_capacity, beta_sensitivity):
    # Iterate through datasets, methods, and yomegas to plot data
    # Create a larger figure
    fig, axs = plt.subplots(2, 3, figsize=(3*12, 2*8), sharex=False, sharey=True)
    sns.set_theme(style='whitegrid', context='talk', font='sans-serif', font_scale=1.0, palette="pastel")
    sns.despine(left=False, top=True, right=True, bottom=False)
    labels_all = ["softmax", "1.5-entmax", "sparsemax", "constrained sparsemax"]
    methods = ["softmax", "entmax", "sparsemax","csparsemax"]
    colors = ['red', 'black', 'blue', 'brown']
    datasets = ["MNIST", "CIFAR10", "ImageNet"]
    x_axis=[[2,4,8,16,32,64,128,256,512,1024], [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]]
    for i, dataset in enumerate(datasets):
        for k, results_dict in enumerate([memory_capacity, beta_sensitivity]):
            results = results_dict[dataset]
            for j, method in enumerate(methods):
                retrieval_mean = results[method]['acc']
                retrieval_std = results[method]['CI']
                axs[k, i].plot(x_axis[k], retrieval_mean, label=labels_all[j], color=colors[j], markersize=15, alpha=0.7, linewidth=3)
                first_elements = [sublist[0] for sublist in retrieval_std]
                second_elements = [sublist[1] for sublist in retrieval_std]
                axs[k, i].fill_between(x_axis[k], first_elements,
                                    second_elements, color=colors[j], alpha=0.1)
                axs[k, i].set_ylim(bottom=0, top=1.1)
                axs[k, i].grid(True, color='white')  # Display the white grid
                axs[k, i].set_facecolor('#f0f0f0')  # Set the background color
                axs[k, i].tick_params(axis='y', labelsize=35, which='major')
                axs[k, i].tick_params(axis='x', labelsize=35, which='major', pad =7.5)
                axs[k, i].set_xscale('log', base=2)
                axs[k, i].set_xticks(x_axis[k])
                if k ==0:
                    axs[k, i].set_title(f'{dataset}', fontsize=40)

        axs[0, i].set_xlabel("Number of Memories", fontsize=40)
        axs[1, i].set_xlabel("$\\beta$", fontsize=40)
        
    fig.text(0.01, 0.5, "Unique Memory Ratio", va='center', rotation='vertical', fontsize=40)
    # Add legend to the first subplot
    axs[0, 0].legend(fontsize=32, loc='lower left')
    plt.tight_layout()
    plt.subplots_adjust(left=0.06)  # Adjust to create space for legend
    plt.savefig('free_recall.pdf')


def MS_plotting2(memory_capacity, beta_sensitivity):

    fig, axs = plt.subplots(2, 3, figsize=(3*12, 2*8), sharex=False, sharey=True)
    sns.set_theme(style='whitegrid', context='talk', font='sans-serif', font_scale=1.0, palette="pastel")
    sns.despine(left=False, top=True, right=True, bottom=False)

    labels_all = ["seq. $2$-subsets + softmax", "seq. $2$-subsets + 1.5-entmax", "seq. $2$-subsets + sparsemax"]
    methods = ["softmax", "entmax", "sparsemax"]
    colors = ['red', 'black', 'blue', 'brown']
    datasets = ["MNIST", "CIFAR10", "ImageNet"]
    marker = [None, "o", None]
    x_axis = [[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], [8, 16, 32, 64, 128, 256, 512, 1024]]

    # Loop through datasets and methods to create plots
    for i, dataset in enumerate(datasets):
        for k, results_dict in enumerate([memory_capacity, beta_sensitivity]):
            results = results_dict[dataset]
            for j, method in enumerate(methods):
                retrieval_mean = results[method]['acc']
                retrieval_std = results[method]['CI']
                axs[k, i].plot(x_axis[k], retrieval_mean, marker=marker[j], label=labels_all[j], color=colors[j], markersize=15, alpha=0.7, linewidth=3)
                first_elements = [sublist[0] for sublist in retrieval_std]
                second_elements = [sublist[1] for sublist in retrieval_std]
                axs[k, i].fill_between(x_axis[k], first_elements,
                                    second_elements, color=colors[j], alpha=0.1)

                axs[k, i].set_ylim(bottom=0, top=1.1)
                axs[k, i].grid(True, color='white')  # Display the white grid
                axs[k, i].set_facecolor('#f0f0f0')  # Set the background color
                axs[k, i].tick_params(axis='y', labelsize=35, which='major')
                axs[k, i].tick_params(axis='x', labelsize=35, which='major')
                axs[k, i].set_xscale('log', base=2)
                if k == 0:
                    axs[k, i].set_title(f'{dataset}', fontsize=40)
                axs[k, i].set_xticks(x_axis[k])
        axs[1, i].set_xlabel("Number of Memories", fontsize=40)
        axs[0, 0].set_ylabel("Unique Memory Ratio", fontsize=40)
        axs[1, 0].set_ylabel("Levenshtein Coefficient", fontsize=40)

    # Add legend to the first subplot
    axs[0, 0].legend(fontsize=32, loc='lower left')

    plt.tight_layout()
    plt.savefig('seq_recall.pdf')
    plt.show()

if __name__ == '__main__':
    args = get_args()
    args.device =  "cuda:0" 
    args.paradigm = "seq_recall"
    results = {}
    results1={}
    #Hopfield(args)
    args.beta = 0.1
    datasetss = ["MNIST", "ImageNet", "CIFAR10"]
    for dataset in datasetss:
        args.dataset = dataset
        results1[dataset] = {}
        results[dataset] = {}
        for method in ["softmax", "entmax", "sparsemax"]:
            args.method = method
            args = entmax_mapping(args)
            results1[dataset][method] = {"acc":[], "CI":[]}
            results[dataset][method] = {"acc":[], "CI":[]}
            for memory_size in [2,4,8,16,32,64,128,256, 512, 1024]:
                args.memory_size = memory_size
                acc, taus = Hopfield(args)
                if memory_size>4:
                    results[dataset][method]["acc"].append(statistics.median(taus))
                    results[dataset][method]["CI"].append([np.percentile(taus, 25), np.percentile(taus, 75)])
        
                results1[dataset][method]["acc"].append(statistics.median(acc))
                results1[dataset][method]["CI"].append([np.percentile(acc, 25), np.percentile(acc, 75)])

    args.beta = 0.1
    args.paradigm = "free_recall"
    results1 = memory_size_sentivity(args)
    results2 = beta_sensitivity(args)
    with open("dics/free_recall", 'wb') as file:
        pickle.dump(results1, file)
    with open("dics/free_recall_beta", 'wb') as file:
        pickle.dump(results2, file)
    MS_plotting(results1, results2)
