from src.Stacking_model.Stacking_model_archi_para import EnsembleModelLoader
from src.data_processing.minmax_scaler import MinMax
from src.material_descriptor.calculate_descriptors import DescriptorsCalculator
import numpy as np
import pandas as pd
import torch
import json
import ternary
import re
import matplotlib.pyplot as plt

# 加载配置文件
with open(r'config.json', 'r') as f:
    config = json.load(f)

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载归一化参数
scaler = MinMax()
feature_scaler = scaler.transform_X
target_scaler = scaler.transform_y

# 加载训练好的模型
loader = EnsembleModelLoader()
loader.load_model(config["Stacking.pth"])
Stacking_model = loader.get_model()
columns = ['Fe', 'Co', 'Ni', 'Cr', 'Mn', 'Al', 'Cu', 'Ti', 'Zr',
           'Nb', 'V', 'Mo', 'Hf', 'Ta', 'Si', 'W', 'YS', 'Elo', 'True probability']

# 加载描述符计算
desc_cal = DescriptorsCalculator()

# 定义计算描述符和拼接的函数
def compute_and_concatenate(individual, desc_calculator):
    # 使用描述符计算类计算描述符
    descriptors_dict = desc_calculator.compute_descriptors(columns[:16], individual)

    # 从字典中提取描述符值
    descriptors_values = list(descriptors_dict.values())

    # 拼接 individual 和描述符
    concatenated = individual + descriptors_values

    return concatenated, descriptors_values

# 定义计算目标值
def cal_target(individual):

    # 拼接特征和描述符
    feature_with_descri, descriptors_values = compute_and_concatenate(individual, desc_cal)
    # 归一化特征
    feature_with_descri_reshaped = np.array(feature_with_descri).reshape(1, -1)
    normalized_features = feature_scaler(feature_with_descri_reshaped)
    # 将数据转换为PyTorch张量
    normalized_features_tensor = torch.tensor(normalized_features, dtype=torch.float32)
    # 使用模型进行预测
    normalized_predictions = Stacking_model(normalized_features_tensor)

    # 将预测结果转换为numpy数组并进行反归一化
    normalized_predictions_np = normalized_predictions.detach().numpy()
    normalized_predictions_reverse = scaler.inverse_transform_nor_y(normalized_predictions_np)
    prediction1 = normalized_predictions_reverse[0, 0]
    prediction2 = normalized_predictions_reverse[0, 1]

    return prediction1, prediction2, descriptors_values

# 定义识别用户输入
def parse_composition(composition_str):
    elements_order = ["Fe", "Co", "Ni", "Cr", "Mn", "Al", "Cu", "Ti", "Zr", "Nb", "V", "Mo", "Hf", "Ta", "Si", "W"]
    composition_elements = re.findall(r'[A-Za-z]+', composition_str)
    composition_values = [float(x) for x in re.findall(r'\d+\.\d+|\d+', composition_str)]

    # 将元素和它们的百分比组合成字典
    composition_dict = dict(zip(composition_elements, composition_values))

    # 计算总和
    total = sum(composition_dict.values())

    # 创建一个16维向量，所有元素的初始值为0
    composition_vector = [0] * 16

    # 填充并规范化向量
    for element, value in composition_dict.items():
        if element in elements_order:
            index = elements_order.index(element)
            normalized_value = (value / total) * 100  # 规范化值
            composition_vector[index] = normalized_value
    return composition_vector



element_index = {"Fe": 0, "Co": 1, "Ni": 2, "Cr": 3, "Mn": 4, "Al": 5, "Cu": 6, "Ti": 7, "Zr": 8, "Nb": 9, "V": 10, "Mo": 11, "Hf": 12, "Ta": 13, "Si": 14, "W": 15}

# 单元素
def study_single_element(base_composition, element, min_val, max_val, step):
    base_vector = parse_composition(base_composition)
    base_sum = sum(base_vector)
    element_values = np.arange(min_val, max_val + step, step)
    results1 = []
    results2 = []

    for val in element_values:
        adjusted_vector = [(x / base_sum) * (100 - val) for x in base_vector]
        adjusted_vector[element_index[element]] = val  # 添加变化的元素
        prediction1, prediction2, _ = cal_target(adjusted_vector)  # 使用 cal_target 函数
        results1.append(prediction1)
        results2.append(prediction2)

    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 绘制第一个目标
    ax1.plot(element_values, results1, color='blue')
    ax1.set_title('Output 1')
    ax1.set_xlabel(f"{element} Content (%)")
    ax1.set_ylabel("Output 1 Value")

    # 绘制第二个目标
    ax2.plot(element_values, results2, color='red')
    ax2.set_title('Output 2')
    ax2.set_xlabel(f"{element} Content (%)")
    ax2.set_ylabel("Output 2 Value")

    plt.tight_layout()
    plt.show()

# 双元素
def study_two_elements_variation(base_composition, element1, min_val1, max_val1, step1, element2, min_val2, max_val2, step2):
    base_vector = parse_composition(base_composition)
    base_sum = sum(base_vector)
    results1 = []
    results2 = []
    x_values, y_values = [], []

    for val1 in np.arange(min_val1, max_val1 + step1, step1):
        for val2 in np.arange(min_val2, max_val2 + step2, step2):
            adjusted_vector = [(x / base_sum) * (100 - val1 - val2) for x in base_vector]
            adjusted_vector[element_index[element1]] = val1
            adjusted_vector[element_index[element2]] = val2
            prediction1, prediction2, _ = cal_target(adjusted_vector)
            results1.append(prediction1)
            results2.append(prediction2)
            x_values.append(val1)
            y_values.append(val2)

    # 转换为numpy数组
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    results1 = np.array(results1)
    results2 = np.array(results2)

    # 绘制热力图
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.tricontourf(x_values, y_values, results1, levels=14, cmap='coolwarm')
    plt.colorbar(label='YS')
    plt.xlabel(f'{element1} Content (%)')
    plt.ylabel(f'{element2} Content (%)')
    plt.title(f'Effect of {element1} and {element2} on YS')

    plt.subplot(1, 2, 2)
    plt.tricontourf(x_values, y_values, results2, levels=14, cmap='coolwarm')
    plt.colorbar(label='Elogation')
    plt.xlabel(f'{element1} Content (%)')
    plt.ylabel(f'{element2} Content (%)')
    plt.title(f'Effect of {element1} and {element2} on ELO')

    plt.tight_layout()
    plt.show()

# 三元素
def study_three_elements_variation(base_composition, elements, total_sum, step):
    base_vector = parse_composition(base_composition)
    base_sum = sum(base_vector)
    results1 = []
    results2 = []
    ternary_data = []

    for i in np.arange(0, total_sum + step, step):
        for j in np.arange(0, total_sum - i + step, step):
            k = total_sum - i - j
            if k < 0:
                continue
            adjusted_vector = [(x / base_sum) * (100 - total_sum) for x in base_vector]
            adjusted_vector[element_index[elements[0]]] = i
            adjusted_vector[element_index[elements[1]]] = j
            adjusted_vector[element_index[elements[2]]] = k
            prediction1, prediction2, _ = cal_target(adjusted_vector)
            results1.append(prediction1)
            results2.append(prediction2)
            ternary_data.append((i, j, k))

    # 绘制三元图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 绘制第一个输出
    tax1 = ternary.TernaryAxesSubplot(ax=ax1, scale=total_sum)
    tax1.heatmap({point: result for point, result in zip(ternary_data, results1)}, style="triangular", cmap="coolwarm")
    tax1.boundary(linewidth=2.0)
    tax1.set_title("Output 1 (YS)")
    tax1.left_corner_label(elements[0], offset=0.14)
    tax1.right_corner_label(elements[1], offset=0.14)
    tax1.top_corner_label(elements[2], offset=0.16)
    tax1.ticks(axis='lbr', linewidth=1, multiple=5)
    tax1.clear_matplotlib_ticks()

    # 绘制第二个输出
    tax2 = ternary.TernaryAxesSubplot(ax=ax2, scale=total_sum)
    tax2.heatmap({point: result for point, result in zip(ternary_data, results2)}, style="triangular", cmap="coolwarm")
    tax2.boundary(linewidth=2.0)
    tax2.set_title("Output 2 (ELO)")
    tax2.left_corner_label(elements[0], offset=0.14)
    tax2.right_corner_label(elements[1], offset=0.14)
    tax2.top_corner_label(elements[2], offset=0.16)
    tax2.ticks(axis='lbr', linewidth=1, multiple=5)
    tax2.clear_matplotlib_ticks()

    plt.tight_layout()
    plt.show()

# 描述符名称列表
descriptors_names = [
    "delta_chi_allen", "delta_chi_pauling", "delta_S_mix", "delta_H_mix",
    "delta_radii", "delta_a", "delta_Tm", "Tm_mean", "am_mean", "VEC_mean",
    "delta_G", "Gm_mean", "lambda", "Omega"
]

# 定义输出材料描述符的函数
def output_descriptors(composition_str):
    input_vector = parse_composition(composition_str)

    # 获取描述符
    _, _, descriptors_values = cal_target(input_vector)

    # 输出描述符结果
    for name, value in zip(descriptors_names, descriptors_values):
        print(f"{name}: {value}")

# 定义输出预测性能的函数
def output_predictions(composition_str):
    input_vector = parse_composition(composition_str)

    # 预测
    prediction1, prediction2, _ = cal_target(input_vector)

    # 输出预测结果
    print("Prediction YS (屈服强度):", prediction1)
    print("Prediction ELO (延伸率):", prediction2)

# 功能 1
composition_input = input('请输入合金成分（输出描述符）：')
output_descriptors(composition_input)

# 功能 2
composition_input = input('请输入合金成分（输出预测性能）：')
output_predictions(composition_input)

# 功能 3

# 功能 3.1 单元素变化
base_composition_input = input("请输入基础合金成分及比例（例如 Ni1Cr1Fe1）：")
element_input = input("请输入要研究的元素（例如 Al）：")
min_val = float(input("请输入元素变化的最小值："))
max_val = float(input("请输入元素变化的最大值："))
step = float(input("请输入步长："))
study_single_element(base_composition_input, element_input, min_val, max_val, step)

# 功能 3.2 双元素变化
base_composition_input = input("请输入基础合金成分及比例（例如 Ni1Cr1Fe1）：")
element1_input = input("请输入第一个元素（例如 Al）：")
min_val1 = float(input("请输入第一个元素的最小值："))
max_val1 = float(input("请输入第一个元素的最大值："))
step1 = float(input("请输入第一个元素的步长："))
step2 = step1
element2_input = input("请输入第二个元素（例如 Ti）：")
min_val2 = float(input("请输入第二个元素的最小值："))
max_val2 = float(input("请输入第二个元素的最大值："))

study_two_elements_variation(base_composition_input, element1_input, min_val1, max_val1, step1, element2_input, min_val2, max_val2, step2)

# 功能 3.3 三元素变化
base_composition_input = input("请输入基础合金成分及比例（例如 Ni1Cr1Fe1）：")
elements_input = input("请输入三个元素，用英文逗号分隔（例如 Al,Ti,Mo）：").split(',')
total_sum = int(input("请输入三个元素的总和："))
step = float(input("请输入步长："))
study_three_elements_variation(base_composition_input, elements_input, total_sum, step)
