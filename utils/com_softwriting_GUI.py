from src.Stacking_model.Stacking_model_archi_para import EnsembleModelLoader
from src.data_processing.minmax_scaler import MinMax
from src.material_descriptor.calculate_descriptors import DescriptorsCalculator
import numpy as np
import torch
import json
import ternary
import re
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog, messagebox, Toplevel, \
    Label, Entry, Button, font, StringVar, Frame

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

# 格式化数字，保留四位小数
def format_numbers(numbers):
    return [f"{num:.4f}" for num in numbers]

# 定义输出材料描述符的函数
def output_descriptors(composition_str):
    input_vector = parse_composition(composition_str)
    _, _, descriptors_values = cal_target(input_vector)
    formatted_descriptors = format_numbers(descriptors_values)
    descriptors_str = '\n'.join(f"{name}: {value}" for name, value in zip(descriptors_names, formatted_descriptors))
    return descriptors_str

# 定义输出预测性能的函数
def output_predictions(composition_str):
    input_vector = parse_composition(composition_str)
    prediction1, prediction2, _ = cal_target(input_vector)
    formatted_predictions = format_numbers([prediction1, prediction2])
    predictions_str = f"Prediction YS (屈服强度): {formatted_predictions[0]}\nPrediction ELO (延伸率): {formatted_predictions[1]}"
    return predictions_str


# GUI

def create_custom_dialog(title, prompt):
    dialog = Toplevel(root)
    dialog.title(title)

    Label(dialog, text=prompt).pack()

    entry_var = StringVar()
    entry = Entry(dialog, textvariable=entry_var)
    entry.pack()

    elements = ["Fe", "Co", "Ni", "Cr", "Mn", "Al", "Cu", "Ti", "Zr", "Nb", "V", "Mo", "Hf", "Ta", "Si", "W"]
    for i, elem in enumerate(elements):
        Button(dialog, text=elem, command=lambda e=elem: (entry_var.set(entry_var.get() + e), entry.focus_set(), entry.icursor(tk.END))).pack(side='left', padx=1)

    def on_submit():
        user_input = entry_var.get()
        dialog.destroy()
        return user_input

    Button(dialog, text="提交", command=on_submit).pack()

    dialog.wait_window()
    return entry_var.get()

# 修改功能1和功能2以使用自定义对话框
def output_descriptors_gui():
    composition_input = create_custom_dialog("输出描述符", "请输入合金成分：")
    if composition_input:
        descriptors = output_descriptors(composition_input)
        messagebox.showinfo("描述符结果", descriptors)

def output_predictions_gui():
    composition_input = create_custom_dialog("输出预测性能", "请输入合金成分：")
    if composition_input:
        predictions = output_predictions(composition_input)
        messagebox.showinfo("预测结果", predictions)



# 功能3.1：研究单元素变化
def study_single_element_gui():
    window = Toplevel(root)
    window.title("研究单元素变化")

    # 当前活动的输入框
    active_entry_var = StringVar(window, "base_comp")

    # 更新活动输入框的函数
    def set_active_entry(entry_name):
        active_entry_var.set(entry_name)

    # 向输入框添加元素的函数
    def add_element_to_entry(element):
        active_entry = base_comp_entry if active_entry_var.get() == "base_comp" else element_entry
        active_entry.insert(tk.END, element)
        active_entry.focus()

    # 输入字段 - 使用grid布局
    Label(window, text="基础合金成分及比例").grid(row=0, column=0, columnspan=4)
    base_comp_entry = Entry(window)
    base_comp_entry.grid(row=1, column=0, columnspan=4)
    base_comp_entry.bind("<FocusIn>", lambda e: set_active_entry("base_comp"))

    Label(window, text="要研究的元素").grid(row=2, column=0, columnspan=4)
    element_entry = Entry(window)
    element_entry.grid(row=3, column=0, columnspan=4)
    element_entry.bind("<FocusIn>", lambda e: set_active_entry("element"))

    # 添加合金元素按钮
    elements = ["Fe", "Co", "Ni", "Cr", "Mn", "Al", "Cu", "Ti", "Zr", "Nb", "V", "Mo", "Hf", "Ta", "Si", "W"]
    for i, elem in enumerate(elements):
        Button(window, text=elem, command=lambda e=elem: add_element_to_entry(e)).grid(row=4 + i // 4, column=i % 4)
        # 使用grid布局管理器进行布局
        Label(window, text="元素变化的最小值").grid(row=8, column=0, columnspan=4)
        min_val_entry = Entry(window)
        min_val_entry.grid(row=9, column=0, columnspan=4)

        Label(window, text="元素变化的最大值").grid(row=10, column=0, columnspan=4)
        max_val_entry = Entry(window)
        max_val_entry.grid(row=11, column=0, columnspan=4)

        Label(window, text="步长").grid(row=12, column=0, columnspan=4)
        step_entry = Entry(window)
        step_entry.grid(row=13, column=0, columnspan=4)

        Button(window, text="分析", command=lambda: study_single_element(
            base_comp_entry.get(), element_entry.get(),
            float(min_val_entry.get()), float(max_val_entry.get()), float(step_entry.get())
        )).grid(row=14, column=0, columnspan=4)

        # 调整布局
        for i in range(4):
            window.grid_columnconfigure(i, weight=1)

def study_two_elements_variation_gui():
    window = Toplevel(root)
    window.title("研究双元素变化")

    # 当前活动的输入框
    active_entry_var = StringVar(window, "base_comp")

    # 更新活动输入框的函数
    def set_active_entry(entry_name):
        active_entry_var.set(entry_name)

    # 向输入框添加元素的函数
    def add_element_to_entry(element):
        active_entry = base_comp_entry if active_entry_var.get() == "base_comp" else \
                       element1_entry if active_entry_var.get() == "element1" else \
                       element2_entry
        active_entry.insert(tk.END, element)
        active_entry.focus()

    # 输入字段 - 使用pack布局
    Label(window, text="基础合金成分及比例").pack()
    base_comp_entry = Entry(window)
    base_comp_entry.pack()
    base_comp_entry.bind("<FocusIn>", lambda e: set_active_entry("base_comp"))

    Label(window, text="第一个元素").pack()
    element1_entry = Entry(window)
    element1_entry.pack()
    element1_entry.bind("<FocusIn>", lambda e: set_active_entry("element1"))

    Label(window, text="第一个元素的最小值").pack()
    min_val1_entry = Entry(window)
    min_val1_entry.pack()

    Label(window, text="第一个元素的最大值").pack()
    max_val1_entry = Entry(window)
    max_val1_entry.pack()

    Label(window, text="第二个元素").pack()
    element2_entry = Entry(window)
    element2_entry.pack()
    element2_entry.bind("<FocusIn>", lambda e: set_active_entry("element2"))

    Label(window, text="第二个元素的最小值").pack()
    min_val2_entry = Entry(window)
    min_val2_entry.pack()

    Label(window, text="第二个元素的最大值").pack()
    max_val2_entry = Entry(window)
    max_val2_entry.pack()

    Label(window, text="元素的步长").pack()
    step1_entry = Entry(window)
    step1_entry.pack()

    # 添加合金元素按钮
    elements = ["Fe", "Co", "Ni", "Cr", "Mn", "Al", "Cu", "Ti", "Zr", "Nb", "V", "Mo", "Hf", "Ta", "Si", "W"]
    for i, elem in enumerate(elements):
        Button(window, text=elem, command=lambda e=elem: add_element_to_entry(e)).pack(side='left', padx=1)

    # 分析按钮
    Button(window, text="分析", command=lambda: study_two_elements_variation(
        base_comp_entry.get(),
        element1_entry.get(), float(min_val1_entry.get()), float(max_val1_entry.get()), float(step1_entry.get()),
        element2_entry.get(), float(min_val2_entry.get()), float(max_val2_entry.get()), float(step1_entry.get())
    )).pack()

    # 调整布局
    for i in range(4):
        window.pack_columnconfigure(i, weight=1)

def study_three_elements_variation_gui():
    window = Toplevel(root)
    window.title("研究三元素变化")

    # 当前活动的输入框
    active_entry_var = StringVar(window, "base_comp")

    # 更新活动输入框的函数
    def set_active_entry(entry_name):
        active_entry_var.set(entry_name)

    # 向输入框添加元素的函数
    def add_element_to_entry(element):
        active_entry = base_comp_entry if active_entry_var.get() == "base_comp" else elements_entry
        current_text = active_entry.get()
        # 仅在elements_entry中添加逗号
        if active_entry_var.get() == "elements" and current_text and not current_text.endswith(','):
            element = ',' + element
        active_entry.insert(tk.END, element)
        active_entry.focus()

    # 输入字段 - 使用pack布局
    Label(window, text="基础合金成分及比例").pack()
    base_comp_entry = Entry(window)
    base_comp_entry.pack()
    base_comp_entry.bind("<FocusIn>", lambda e: set_active_entry("base_comp"))

    Label(window, text="三个元素（用逗号分隔）").pack()
    elements_entry = Entry(window)
    elements_entry.pack()
    elements_entry.bind("<FocusIn>", lambda e: set_active_entry("elements"))

    Label(window, text="三个元素的总和").pack()
    total_sum_entry = Entry(window)
    total_sum_entry.pack()

    Label(window, text="步长").pack()
    step_entry = Entry(window)
    step_entry.pack()

    # 添加合金元素按钮
    elements = ["Fe", "Co", "Ni", "Cr", "Mn", "Al", "Cu", "Ti", "Zr", "Nb", "V", "Mo", "Hf", "Ta", "Si", "W"]
    for elem in elements:
        Button(window, text=elem, command=lambda e=elem: add_element_to_entry(e)).pack(side='left', padx=1)

    # 分析按钮
    Button(window, text="分析", command=lambda: study_three_elements_variation(
        base_comp_entry.get(),
        elements_entry.get().split(','),
        int(total_sum_entry.get()),
        float(step_entry.get())
    )).pack()

# 创建主窗口
root = tk.Tk()
root.title("HEAs压缩性能分析工具")
root.geometry("800x600")  # 设置窗口初始大小
root.configure(bg='light grey')  # 设置窗口背景颜色

# 设置字体
button_font = font.Font(family="Times New Roman", size=18, weight="bold")

# 创建按钮并应用样式
Button(root, text="HEAs描述符计算", command=output_descriptors_gui, font=button_font, fg='white', bg='#3498db', padx=10, pady=5).pack(expand=True, fill='both', pady=10)
Button(root, text="预测HEAs压缩力学性能", command=output_predictions_gui, font=button_font, fg='white', bg='#2ecc71', padx=10, pady=5).pack(expand=True, fill='both', pady=10)
Button(root, text="单元素变化对HEAs性能影响", command=study_single_element_gui, font=button_font, fg='white', bg='#9b59b6', padx=10, pady=5).pack(expand=True, fill='both', pady=10)
Button(root, text="双元素变化对HEAs性能影响", command=study_two_elements_variation_gui, font=button_font, fg='white', bg='#e67e22', padx=10, pady=5).pack(expand=True, fill='both', pady=10)
Button(root, text="三元素变化对HEAs性能影响", command=study_three_elements_variation_gui, font=button_font, fg='white', bg='#34495e', padx=10, pady=5).pack(expand=True, fill='both', pady=10)

# 运行主循环
root.mainloop()
