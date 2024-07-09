import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import json
import math

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载配置文件
with open(r'config.json', 'r') as f:
    config = json.load(f)

# 定义注意力层
class FeatureEmbedding(nn.Module):
    def __init__(self, input_dim, embed_size, dropout_rate=0.2):
        super(FeatureEmbedding, self).__init__()
        self.fc1 = nn.Linear(input_dim, embed_size)
        self.bn1 = nn.BatchNorm1d(embed_size)  # 批归一化层
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)  # Dropout层
        self.fc2 = nn.Linear(embed_size * 2, embed_size)
        self.bn2 = nn.BatchNorm1d(embed_size)  # 批归一化层

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  # 应用批归一化
        x = self.relu(x)
        # x = self.dropout(x)  # 应用dropout
        # x = self.fc2(x)
        # x = self.bn2(x)  # 应用批归一化
        return x

class SelfAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(SelfAttention, self).__init__()
        self.attention_dim = attention_dim
        self.input_dim = input_dim

        # # 特征嵌入层，为每个特征创建一个嵌入层
        # self.feature_embeddings = nn.ModuleList([nn.Linear(1, attention_dim) for _ in range(input_dim)])
        # 特征嵌入层，使用MLP替代单一的线性层
        self.feature_embeddings = nn.ModuleList([FeatureEmbedding(1, attention_dim) for _ in range(input_dim)])
        # 自注意力层的查询（Q）、键（K）和值（V）层
        self.query = nn.Linear(attention_dim, attention_dim)
        self.key = nn.Linear(attention_dim, attention_dim)
        self.value = nn.Linear(attention_dim, attention_dim)

        # 添加批归一化层
        self.bn_query = nn.BatchNorm1d(attention_dim)
        self.bn_key = nn.BatchNorm1d(attention_dim)
        self.bn_value = nn.BatchNorm1d(attention_dim)

        # 缩放因子，用于调整点积的大小
        self.scale = torch.sqrt(torch.tensor([attention_dim], dtype=torch.float32)).to(device)
    def forward(self, x):
        # 嵌入每个特征并将结果存储到列表中
        embedded_features_list = [self.feature_embeddings[i](x[:, i].view(-1, 1)) for i in range(self.input_dim)]
        # 沿特征维度拼接嵌入特征，以形成一个新的张量
        embedded_features = torch.cat(embedded_features_list, dim=1).view(-1, self.input_dim, self.attention_dim)

        # 生成 Q、K 和 V
        Q = self.query(embedded_features)
        K = self.key(embedded_features)
        V = self.value(embedded_features)

        # 批归一化
        Q = self.bn_query(Q.transpose(1, 2)).transpose(1, 2)
        K = self.bn_key(K.transpose(1, 2)).transpose(1, 2)
        V = self.bn_value(V.transpose(1, 2)).transpose(1, 2)

        # 计算注意力分数
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale

        # 使用 softmax 函数获取注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 通过注意力权重加权 V 以得到最终的输出
        out = torch.bmm(attention_weights, V)

        # 展平输出的后两维以便输入到回归模型中
        out_flat = out.flatten(start_dim=1)

        return out_flat, attention_weights

# 自适应权重参数
class SelfAttention(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.2):
        super(SelfAttention, self).__init__()
        # 第一层隐藏层
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.BatchNorm1d(input_dim * 2),  # 批归一化
            nn.ReLU(),
            nn.Dropout(dropout_rate)  # Dropout
        )
        # # 第二层隐藏层
        # self.layer2 = nn.Sequential(
        #     nn.Linear(input_dim * 2, input_dim * 2),
        #     nn.BatchNorm1d(input_dim * 2),  # 批归一化
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate)  # Dropout
        # )
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.layer1(x)
        # x1 = self.layer2(x1)
        weights = self.output_layer(x1)
        weighted_features = x * weights
        return weighted_features, weights


# 定义初级学习器的神经网络模型
class BaseModel(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, hidden_units, dropout_rate, output_features, attention_model=None):
        super(BaseModel, self).__init__()
        # 自注意力层
        self.self_attention = attention_model
        # 线性层和批归一化层
        self.extra_fc = nn.Linear(input_dim, hidden_units)
        self.extra_bn = nn.BatchNorm1d(hidden_units)
        # 隐藏层
        layers = []
        for _ in range(num_hidden_layers):
            layers.extend([
                nn.Linear(hidden_units if len(layers) == 0 else hidden_units, hidden_units),
                nn.BatchNorm1d(hidden_units),  # 批归一化
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
        self.hidden_layers = nn.Sequential(*layers)
        # 输出层
        self.output_layer = nn.Linear(hidden_units, output_features)

    def forward(self, x):
        # 自注意力层
        if self.self_attention:
            x, _ = self.self_attention(x)
        x = self.extra_fc(x)
        x = self.extra_bn(x)
        x = nn.ReLU()(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

# 定义元学习器的神经网络模型
class MetaModel(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, hidden_units, dropout_rate):
        super(MetaModel, self).__init__()

        # Extra layers before hidden layers
        self.extra_fc = nn.Linear(input_dim, hidden_units)
        self.extra_bn = nn.BatchNorm1d(hidden_units)

        # Hidden layers
        layers = []
        for _ in range(num_hidden_layers):
            layers.extend([
                nn.Linear(hidden_units if len(layers) == 0 else hidden_units, hidden_units),
                nn.BatchNorm1d(hidden_units),  # Batch normalization
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(hidden_units, 2)  # Two outputs

    def forward(self, x):
        # Extra layers before hidden layers
        x = self.extra_fc(x)
        x = self.extra_bn(x)
        x = nn.ReLU()(x)

        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

# 定义Stacking集成学习模型
class StackingModel(nn.Module):
    def __init__(self, base_models, meta_model, num_base_models, attention_model=None):
        super(StackingModel, self).__init__()
        self.base_models = nn.ModuleList(base_models)
        self.meta_model = meta_model
        self.num_base_models = num_base_models

        # 自注意力层
        self.self_attention = attention_model

    def forward(self, x):

        # 检查是否传入注意力，应用自注意力机制
        if self.self_attention:
            x, _ = self.self_attention(x)
        base_outputs = [model(x) for model in self.base_models]
        meta_input = torch.cat(base_outputs, dim=1)
        return self.meta_model(meta_input)

# 定义超参数模型初始化
class StackingModelInitializer:
    def __init__(self, num_base_models, device, config, use_attention_in_base=False, use_attention_in_stacking=False):
        self.num_base_models = num_base_models
        self.device = device
        self.config = config
        self.use_attention_in_base = use_attention_in_base
        self.use_attention_in_stacking = use_attention_in_stacking
        self.input_features_meta = num_base_models * config["output_features_base"]
        # self.input_base = config["input_size"] * config["attention_dim"]
        self.input_base = config["input_size"]

    def create_stacking_model(self):

        # 注意力模型参数
        attention_model_params = {
            "input_dim": self.config["input_size"],
            # # 使用自注意力时使用
            # "attention_dim": self.config["attention_dim"]
        }
        attention_model = SelfAttention(**attention_model_params)

        # 基学习器参数
        base_model_params = {
            "input_dim": self.input_base,
            "num_hidden_layers": self.config["num_hidden_layers_base"],
            "hidden_units": self.config["hidden_units_base"],
            "dropout_rate": self.config["dropout_rate_base"],
            "output_features": self.config["output_features_base"],
        }

        # 检查自注意力模型用在哪
        if self.use_attention_in_base:
            base_model_params["attention_model"] = attention_model

        base_models = [BaseModel(**base_model_params) for _ in range(self.num_base_models)]

        # 元学习器参数
        meta_model_params = {
            "input_dim": self.input_features_meta,
            "num_hidden_layers": self.config["num_hidden_layers_meta"],
            "hidden_units": self.config["hidden_units_meta"],
            "dropout_rate": self.config["dropout_rate_meta"]
        }
        meta_model = MetaModel(**meta_model_params)

        stacking_model = StackingModel(base_models, meta_model, self.num_base_models, attention_model if self.use_attention_in_stacking else None)
        return stacking_model.to(self.device)

# 定义调用训练好的模型
class EnsembleModelLoader:
    def __init__(self):
        self.ensemble_model = None

    def load_model(self, model_path):
        # 加载模型
        self.ensemble_model = torch.load(model_path, map_location=torch.device('cpu'))

        # 确保模型处于评估模式
        self.ensemble_model.eval()

        # 如果StackingModel有自注意力模块，确保它处于评估模式
        if hasattr(self.ensemble_model, "attention"):
            self.ensemble_model.attention.eval()

        # 确保元学习器模型处于评估模式
        self.ensemble_model.meta_model.eval()

        # 确保所有基学习器模型处于评估模式
        for base_model in self.ensemble_model.base_models:
            base_model.eval()

            # 如果基模型有自注意力模块，确保它处于评估模式
            if hasattr(base_model, "attention"):
                base_model.attention.eval()

    def get_model(self):
        return self.ensemble_model

    # 获取注意力权重
    def get_attention_weights(self, x):
        # 确保调用的模型有自注意力模块
        if hasattr(self.ensemble_model, "self_attention"):
            # 将输入传递到自注意力模块并返回注意力权重
            _, attention_weights = self.ensemble_model.self_attention(x)
            return attention_weights
        else:
            return None