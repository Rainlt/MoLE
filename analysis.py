import json
import os
import pdb
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def save_attn_hallu_fig(attn_data):

    # 优化配色方案
    background_color = "#ffffff"  # 背景色
    curve_color = "#e85939"  # 曲线颜色（深红色）
    bar_color = "#4e658f"  # 条形图颜色（深蓝色）
    edge_color = "#2f4f4f"  # 条形图边缘颜色（深石板灰色）
    fill_color = "#ff953d"  # 填充颜色（浅红色）
    hatch_color = "#ffa634"  # 虚线颜色（暗红色）
    
    # 假设你已经有96个数据值
    data = attn_data 
    x_values = np.arange(0, 96, 1)

    # 创建条形图的数据
    hallu_values = [14, 34, 42, 59, 88, 124]

    # 创建图表
    fig, ax1 = plt.subplots(figsize=(10, 8))

    # 设置背景颜色
    fig.patch.set_facecolor(background_color)
    ax1.set_facecolor(background_color)

    # 设置x轴范围
    ax1.set_xlim(0, 96)
    ax1.set_xticks(np.arange(0, 96, 16)) 
    
    # 设置左侧y轴的标签和颜色
    ax1.set_ylabel('Attention for Prompt', color=edge_color, fontsize=14)
    ax1.set_xlabel('Position Index', fontsize=14)
    ax1.set_ylim(0.68, 1.0)

    # 绘制曲线
    line1, = ax1.plot(x_values, data, label='Attention Curve', color=curve_color, linewidth=1.5, zorder=10)
    
    # 添加填充颜色
    ax1.fill_between(x_values, data, color=fill_color, alpha=0.3, zorder=8)

    # 添加虚线填充
    ax1.fill_between(x_values, data, color='none', hatch='//', edgecolor=hatch_color, alpha=0.5, zorder=9)

    # 创建第二个y轴用于条形图
    ax1.tick_params(axis='x', labelsize=12)  # 设置x轴刻度标签的字体大小
    ax1.tick_params(axis='y', labelsize=12)  # 设置左侧y轴刻度标签的字体大小
    ax2 = ax1.twinx()
    ax2.tick_params(axis='y', labelsize=12)  # 设置右侧y轴刻度标签的字体大小
    ax2.set_ylabel('Number of Hallucinations', color=edge_color, fontsize=14)
    ax2.set_ylim(0, 150)

    # 计算条形图的中心位置
    x_centers = [8, 24, 40, 56, 72, 88]

    # 绘制条形图
    bar1 = ax2.bar(x_centers, hallu_values, width=16, color=bar_color, edgecolor=edge_color, linewidth=2, alpha=0.8, zorder=5, label="Hallu Number")

    # 合并图例
    lines = [line1, bar1]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=10)

    # 保存图表
    output_path = "./combined_density_bar_plot.png"
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.show()
    
def read_dir_results(before_dir, file_list,metric):
    results_dict = {}
    for file in file_list:
        data = json.load(open(os.path.join(before_dir, file)))
        file_index = file.split('_')[-2]
        state = data[metric]
        results_dict[file_index] = state
    return results_dict
def plot_reslut_line():
    base_dir = './generated_captions/pope/'
    results_dir_ln = base_dir +'llava-1.5_LN'
    results_dir_woln = base_dir + 'all_layer_woLN'

    #获取results_dir_ln下以results.json结尾的文件，并结合路径
    #导入plt包
    
    results_files_ln = [f for f in os.listdir(results_dir_ln) if f.endswith('results.json')]
    results_files_woln = [f for f in os.listdir(results_dir_woln) if f.endswith('results.json')]

    results_ln = read_dir_results(results_dir_ln, results_files_ln, 'Accuracy')
    results_woln = read_dir_results(results_dir_woln, results_files_woln, 'Accuracy')

    # Convert keys to float and sort
    sorted_keys_ln = sorted([int(key) for key in results_ln.keys()])
    sorted_keys_woln = sorted([int(key) for key in results_woln.keys()])

    # Get corresponding values
    sorted_values_ln = [results_ln[str(key)] for key in sorted_keys_ln]
    sorted_values_woln = [results_woln[str(key)] for key in sorted_keys_woln]

    plt.figure()
    plt.plot(sorted_keys_ln, sorted_values_ln, label='LN')
    plt.plot(sorted_keys_woln, sorted_values_woln, label='woLN')
    plt.legend()
    plt.savefig('/home/liangtian/project/HALC/line_plot.png')




def compare_logits(tokenizer, all_layer_logits,all_layer_tokens, attn_matrix,premature_layer_list,index,hallu):
    first_token_logits = all_layer_logits[0]#[33, 20] 33层，top20
    #将first_token_logits中的所有值保留四位小数
    first_token_logits = [[round(i, 4) for i in j] for j in first_token_logits]
    first_token = all_layer_tokens[0]#[33, 20]

    first_token_for_final = first_token[-1]#最后一层的预测，所以是-1
    first_token_for_final = [tokenizer.convert_ids_to_tokens(token) for token in first_token_for_final]
    first_token_logits_for_final = first_token_logits[-1]
    first_token_logits_for_premature = first_token_logits[premature_layer_list[0]]

    max_attn_index = max(range(len(attn_matrix[:,0])), key=attn_matrix[:,0].__getitem__)#取第0个token的最大attn的index
    #min_attn_index = min(range(len(attn_matrix[:,0])), key=attn_matrix[:,0].__getitem__)#取第0个token的最小attn的index
    #这里改成第31层
    first_token_logits_for_max_attn = first_token_logits[max_attn_index]#[20]
    first_token_logits_for_min_attn = first_token_logits[31]#[20]

    #制表并输出
    table = f"{'index':<6}{'tokens':<20}{'final_logits':<20}{'premature_logits':<20}{'dola_logits':<20}{'max_attn':<20}{'31_layer':<20}\n"
    for i in range(len(first_token_for_final)):
        dola_logits = round(first_token_logits_for_final[i] - first_token_logits_for_premature[i],4)
        table += f"{i:<6}{first_token_for_final[i]:<20}{first_token_logits_for_final[i]:<20}{first_token_logits_for_premature[i]:<20}{dola_logits:<20}{first_token_logits_for_max_attn[i]:<20}{first_token_logits_for_min_attn[i]:<20}\n"
    
    with open(f'/home/liangtian/project/HALC/visualize/{hallu}_logits_table_{index}.txt', 'w') as f:
        f.write(table)
'''
TODO: 分析当final为hallu，但31层为True时的Pattern
1. 找出final为hallu的sample
2. 可视化final为hallu时的jsd，attn_matrix,logits,log_logits
3. 可视化final为True时的jsd，attn_matrix,logits,log_logits
4. 比较两者的差异
'''
def draw_jsd_matrix(jsd_matrix, output_tokens, img_id, hallu, mode):
    plt.figure(figsize=(48, 8))  # Increase the figsize to accommodate longer x-axis
    sns.heatmap(jsd_matrix, cmap='coolwarm', xticklabels=output_tokens, yticklabels=range(0, 32), annot=True, fmt=".2f", annot_kws={"fontsize": 6})
    plt.xlabel('Image ID: ' + str(img_id))
    plt.title('Attn Matrix Heatmap')
    plt.xticks(rotation=90)  # Rotate x-axis labels
    plt.yticks(rotation=0)  # Rotate y-axis labels
    plt.gca().invert_yaxis()  # Invert y-axis
    plt.show()
    #将热力图保存为图片
    plt.savefig(f'/home/liangtian/project/HALC/visualize/{mode}/{img_id}.png')
    plt.close()

def analysis_logits():
    #data_list = json.load(open('token_list_caption_128_addID.json'))
    data_list = json.load(open('token_list_caption_128.json'))

    #merged_ckpt = '/sda/liangtian/model/llava-v1.5-7b/'
    #tokenizer = AutoTokenizer.from_pretrained(merged_ckpt, use_fast=False)
    premature_layer_dist = {l:0 for l in range(33)}
    sum_attn_matrix = np.zeros((32, 96))
    None_96_count = 0
    token_dict = {}
    for index, data in tqdm(enumerate(data_list)):
        # analysis data sample
        #img_id = data["image_id"]
        output_tokens = data["output_tokens"]  # [n]
        jsd_matrix = data["jsd_matrix"]  # [32, seq_len]
        
        jsd_matrix_topN = data["jsd_matrix_topN"]  # [32, seq_len]
        jsd_matrix_remain = data["jsd_matrix_remain"]  # [32, seq_len]
        attn_matrix = data["attn_matrix"]  # [32, seq_len]
        # 将attn_matrix中的所有值保留四位小数
        attn_matrix = np.array([[round(i, 4) for i in j] for j in attn_matrix])#np.array(attn_matrix).shape=(32,64)
        np_attn_matrix = np.array(attn_matrix)
        #如果np_attn_matrix的shape的第二个维度<96，就跳过，大于96的就取前96个
        if np_attn_matrix.shape[1] < 96:
            None_96_count += 1
            continue
        np_attn_matrix = np_attn_matrix[:, :96]
        sum_attn_matrix += np_attn_matrix
        
        
        all_layer_logits = data["all_layer_logits"]  # [seq_len], [33, 20] 20是因为取的top20的logits
        all_layer_tokens = data["all_layer_tokens"]  # [seq_len], [33, 20] 20是因为取的top20的logits
        #all_layer_real_tokens = data["all_layer_real_logits"]
        #all_layer_real_logits = data["all_layer_real_logits"]
        #values = {"tokens": all_layer_real_tokens, "logits": all_layer_real_logits}
        #token_dict[img_id] = values
        
        premature_layer_list = data["premature_layer_list"]  # [n]
        for layer in premature_layer_list:
            premature_layer_dist[layer] += 1
        #hallu = data["hallu"]  # [1]
        hallu = 'oo'
        #print("output tokens: ", output_tokens)
        #print("hallu: ", hallu)

        #compare_logits(tokenizer, all_layer_log_logits, all_layer_tokens, attn_matrix, premature_layer_list, index, hallu)
        #draw_jsd_matrix(jsd_matrix, output_tokens, img_id, hallu, 'all')

        #pdb.set_trace()
        #draw_jsd_matrix(jsd_matrix_topN, output_tokens, index, hallu, 'top')
        #draw_jsd_matrix(jsd_matrix_remain, output_tokens, index, hallu, 'remain')
    #将token_dict保存为json文件 
    #with open('/home/liangtian/project/HALC/visualize/token_dict.json', 'w') as f:
        #json.dump(token_dict, f)
    #plot attn-hallu fig
    num_96 = len(data_list) - None_96_count
    print(sum_attn_matrix.shape)#[32,128]
    #time-wise sum
    #sum_attn_matrix = np.sum(sum_attn_matrix, axis=0)
    sum_attn_matrix = sum_attn_matrix[0] / num_96

    
    save_attn_hallu_fig(sum_attn_matrix)
    
    '''
        first_token_for_final = first_token[-1]
        first_token_for_final = [tokenizer.convert_ids_to_tokens(token) for token in first_token_for_final]
        first_token_logits_for_final = first_token_logits[-1]
        token_logits_dict_final = dict(zip(first_token_for_final, first_token_logits_for_final))
        print("token_logits_dict_final: ", token_logits_dict_final)
        print("\n")
        first_token_for_premature = first_token[premature_layer_list[0]]#[20]
        first_token_for_premature = [tokenizer.convert_ids_to_tokens(token) for token in first_token_for_premature]
        first_token_logits_for_premature = first_token_logits[premature_layer_list[0]]#[20]
        token_logits_dict_premature = dict(zip(first_token_for_premature, first_token_logits_for_premature))
        print("token_logits_dict_premature: ", token_logits_dict_premature)
        print("\n")

        max_attn_index = max(range(len(attn_matrix[:,0])), key=attn_matrix[:,0].__getitem__)#取第0个token的最大attn的index
        min_attn_index = min(range(len(attn_matrix[:,0])), key=attn_matrix[:,0].__getitem__)#取第0个token的最小attn的index
        print("max_attn_index: ", max_attn_index)
        print("min_attn_index: ", min_attn_index)
        print("attn_matrix: ", attn_matrix[:,0])
        print("\n")

        first_token_for_max_attn = first_token[max_attn_index]#[20]  
        first_token_for_max_attn = [tokenizer.convert_ids_to_tokens(token) for token in first_token_for_max_attn]  
        first_token_logits_for_max_attn = first_token_logits[max_attn_index]#[20]
        token_logits_dict_max_attn = dict(zip(first_token_for_max_attn, first_token_logits_for_max_attn))
        print("token_logits_dict_max_attn: ", token_logits_dict_max_attn)
        print("\n")

        first_token_for_min_attn = first_token[min_attn_index]#[20]
        first_token_for_min_attn = [tokenizer.convert_ids_to_tokens(token) for token in first_token[min_attn_index]]#[20]
        first_token_logits_for_min_attn = first_token_logits[min_attn_index]#[20]
        token_logits_dict_min_attn = dict(zip(first_token_for_min_attn, first_token_logits_for_min_attn))
        print("token_logits_dict_min_attn: ", token_logits_dict_min_attn)
        '''
    '''
    num_layer = sum(premature_layer_dist.values())
    premature_layer_dist = {k:v/num_layer for k,v in premature_layer_dist.items()}
    print(premature_layer_dist)
    '''
    #TODO: 以final layer的top 20 tokens为基准，找到premature layer对应的20个logits，attn_layer对应的20个logits
analysis_logits()