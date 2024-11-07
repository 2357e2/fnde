import torch
import numpy as np
import matplotlib.pyplot as plt

def print_from_numpy(file_location, file_name, index):
    file = np.load(f"{file_location}/{file_name}.npy")
    output = file
    if type(index) is int:
        output = output[index]
    else:
        for i in index:
            try:
                output = output[i]
            except IndexError:
                print(f'Error: np array dim {file.shape} does not match index {index}')
    print(output)

def plot(file_path, model, file_name_x, file_name_y):
    x = np.load(f"{file_path}/{model}/{file_name_x}.npy")
    y = np.load(f"{file_path}/{model}/{file_name_y}.npy")

    fig, ax0 = plt.subplots()
    
    ax0.scatter(x, y, label='FNDE', c='black', s=2)
    ax0.set_xlabel('epoch')
    ax0.set_ylabel('mean fractional loss')
    ax0.legend()
    plt.show()

def plot_compare(file_paths, model_names, plot_title, file_name_x, file_name_y, log_loss=True):
    #x = np.load(f"{file_paths[0]}/{file_name_x}.npy")
    x = np.linspace(1, 400, 400)
    y_ls = []
    for file in file_paths:
        y_ls.append(np.load(f"{file}/{file_name_y}.npy"))
    
    y_fnde = np.empty(len(y_ls[0]))
    y_fnde_mod = np.empty(len(y_ls[0]))
    y_fno = np.empty(len(y_ls[0]))
    y_node = np.empty(len(y_ls[0]))
    for k in range(len(y_ls)):
        if k % 4 == 0:
            for j in range(len(y_ls[0])):
                y_fnde[j] += 1/5 * y_ls[k][j]
        if k % 4 == 1:
            for j in range(len(y_ls[0])):
                y_fnde_mod[j] += 1/5 * y_ls[k][j]
        if k % 4 == 2:
            for j in range(len(y_ls[0])):
                y_fno[j] += 1/5 * y_ls[k][j]
        if k % 4 == 3:
            for j in range(len(y_ls[0])):
                y_node[j] += 1/5 * y_ls[k][j]



    if log_loss:
        for j in range(len(y_fnde)):
            y_fnde[j] = np.log10(y_fnde[j])

        for j in range(len(y_fnde_mod)):
            y_fnde_mod[j] = np.log10(y_fnde_mod[j])

        for j in range(len(y_fno)):
            y_fno[j] = np.log10(y_fno[j])
        
        for j in range(len(y_node)):
            y_node[j] = np.log10(y_node[j])
    
    fig, ax = plt.subplots(figsize=(5,4))

    
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 13
    plt.rcParams['figure.dpi'] = 500
    plt.rcParams['savefig.dpi'] = 500

    ax.plot(x, y_fnde, label='FNDE', c='black')
    ax.plot(x, y_fnde_mod, label='FNDE mod', c='blue')
    ax.plot(x, y_fno, label='FNO', c='red')
    ax.plot(x, y_node, label='NODE', c='green')
    ax.set_xlabel('epoch', fontsize=14)
    ax.set_ylabel('log 10 (mean fractional loss)', fontsize=14)
    
    plt.tight_layout()
    plt.title(plot_title, pad=0.2, loc='center')
    plt.legend(loc='upper right')
    plt.savefig(f'C:/Users/fnde/fnde/figures/{plot_title}', bbox_inches='tight')
    plt.show()




if __name__ == '__main__':
    theories=['Phi 4 (1 loop)','Scalar QED (tree level)','Scalar Yukawa (tree level)']
    theory=theories[0]
    if theory=='Scalar QED (tree level)':
        suffix = '___Scalar_QED_tree'
        prefixes = ['0020', '0021', '0022', '0023', '0024']
    elif theory=='Scalar Yukawa (tree level)':
        suffix = '___Scalar_Yukawa_tree'
        prefixes = ['0030', '0031', '0032', '0033', '0034']
    else:
        suffix = '___Phi4_1_loop'
        prefixes = ['0010', '0011', '0012', '0013', '0014']
    suffix = '___Phi4_1_loop'
    #suffix = '___Scalar_QED_Tree'
    #suffix = '___Scalar_Yukawa_Tree'
    prefixes = ['0040', '0041', '0042', '0043', '0044']
    path='C:/Users/fnde/fnde/data/'
    models = ['FNDE_2/', 'FNDE_2_mod/', 'FNO/', 'NODE/']
    plot_compare([f'{path}{models[i]}__{prefixes[j]}{suffix}' for j in range(len(prefixes)) for i in range(3)], ['FNDE', 'FNDE_mod','FNO', 'NODE'], f'{theory} training loss', 'epoch_arr', 'loss_arr')