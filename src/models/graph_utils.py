"""
图结构工具模块

该模块提供了用于创建和操作图结构的工具函数，
这些函数用于构建立体匹配中的马尔可夫随机场。
"""

import torch
import torch.nn.functional as F
import numpy as np
import dgl


def build_grid_graph(height, width, connectivity=4):
    """
    构建网格图
    
    参数:
        height (int): 图像高度
        width (int): 图像宽度
        connectivity (int): 连接度，可选4（上下左右）或8（上下左右及对角线）
        
    返回:
        dgl.DGLGraph: 构建的网格图
    """
    # 创建节点
    num_nodes = height * width
    src_nodes = []
    dst_nodes = []
    
    # 定义邻居的相对位置
    if connectivity == 4:
        # 上下左右
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        # 上下左右及对角线
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        raise ValueError(f"不支持的连接度: {connectivity}，只支持4或8")
    
    # 构建边
    for i in range(height):
        for j in range(width):
            node_id = i * width + j
            
            for di, dj in neighbors:
                ni, nj = i + di, j + dj
                
                # 检查邻居是否在网格内
                if 0 <= ni < height and 0 <= nj < width:
                    neighbor_id = ni * width + nj
                    src_nodes.append(node_id)
                    dst_nodes.append(neighbor_id)
    
    # 使用DGL创建图
    graph = dgl.graph((torch.tensor(src_nodes), torch.tensor(dst_nodes)))
    
    # 添加节点坐标作为特征
    node_coordinates = torch.zeros((num_nodes, 2))
    for i in range(height):
        for j in range(width):
            node_id = i * width + j
            node_coordinates[node_id, 0] = i  # y坐标
            node_coordinates[node_id, 1] = j  # x坐标
    
    graph.ndata['coordinates'] = node_coordinates
    
    return graph


def build_hierarchical_graph(height, width, num_levels=3, scale_factor=0.5):
    """
    构建层次图
    
    参数:
        height (int): 原始图像高度
        width (int): 原始图像宽度
        num_levels (int): 层级数
        scale_factor (float): 相邻层级之间的缩放因子
        
    返回:
        list: 包含不同层级网格图的列表
    """
    graphs = []
    
    for level in range(num_levels):
        # 计算当前层级的大小
        current_height = int(height * (scale_factor ** level))
        current_width = int(width * (scale_factor ** level))
        
        # 构建当前层级的网格图
        graph = build_grid_graph(current_height, current_width)
        graphs.append(graph)
    
    return graphs


def add_image_features_to_graph(graph, features):
    """
    将图像特征添加到图节点
    
    参数:
        graph (dgl.DGLGraph): 输入图
        features (torch.Tensor): 图像特征，形状为 [C, H, W]
        
    返回:
        dgl.DGLGraph: 添加了特征的图
    """
    # 重新排列特征形状为 [H*W, C]
    c, h, w = features.shape
    features_flat = features.permute(1, 2, 0).reshape(-1, c)
    
    # 添加到图的节点
    graph.ndata['features'] = features_flat
    
    return graph


def compute_edge_weights(graph, feature_weight=1.0, position_weight=0.1):
    """
    计算边权重
    
    参数:
        graph (dgl.DGLGraph): 输入图
        feature_weight (float): 特征相似性的权重
        position_weight (float): 位置相似性的权重
        
    返回:
        dgl.DGLGraph: 添加了边权重的图
    """
    # 获取源节点和目标节点的索引
    src, dst = graph.edges()
    
    # 计算特征相似性
    if 'features' in graph.ndata:
        src_feat = graph.ndata['features'][src]
        dst_feat = graph.ndata['features'][dst]
        # 使用余弦相似度
        feature_sim = F.cosine_similarity(src_feat, dst_feat, dim=1)
    else:
        feature_sim = torch.ones(src.shape[0])
    
    # 计算位置相似性
    src_pos = graph.ndata['coordinates'][src]
    dst_pos = graph.ndata['coordinates'][dst]
    # 使用欧几里得距离的倒数
    pos_dist = torch.sqrt(torch.sum((src_pos - dst_pos) ** 2, dim=1) + 1e-6)
    position_sim = 1.0 / pos_dist
    
    # 组合权重
    edge_weights = feature_weight * feature_sim + position_weight * position_sim
    
    # 添加到图的边
    graph.edata['weight'] = edge_weights
    
    return graph


def message_func(edges):
    """
    边的消息函数
    
    参数:
        edges: 边批次
        
    返回:
        dict: 包含消息的字典
    """
    # 组合边的权重和源节点的特征
    return {'msg': edges.src['h'] * edges.data['weight'].unsqueeze(1)}


def reduce_func(nodes):
    """
    节点的消息聚合函数
    
    参数:
        nodes: 节点批次
        
    返回:
        dict: 包含聚合后特征的字典
    """
    # 求和聚合所有传入消息
    return {'h_new': torch.sum(nodes.mailbox['msg'], dim=1)}


def apply_func(nodes):
    """
    节点的特征更新函数
    
    参数:
        nodes: 节点批次
        
    返回:
        dict: 包含更新后特征的字典
    """
    # 使用阻尼系数更新节点特征
    alpha = 0.5
    return {'h': (1 - alpha) * nodes.data['h'] + alpha * nodes.data['h_new']}


class GraphMessagePasser:
    """
    图消息传递器
    
    该类实现了基于DGL的消息传递算法，用于信念传播。
    """
    
    def __init__(self, iterations=5):
        """
        初始化图消息传递器
        
        参数:
            iterations (int): 消息传递迭代次数
        """
        self.iterations = iterations
    
    def run_message_passing(self, graph, initial_node_features):
        """
        运行消息传递
        
        参数:
            graph (dgl.DGLGraph): 输入图
            initial_node_features (torch.Tensor): 初始节点特征
            
        返回:
            torch.Tensor: 更新后的节点特征
        """
        # 初始化节点特征
        graph.ndata['h'] = initial_node_features
        
        # 迭代消息传递
        for _ in range(self.iterations):
            graph.update_all(message_func, reduce_func, apply_func)
        
        # 返回更新后的节点特征
        return graph.ndata['h']


def stereo_matching_with_graph(left_features, right_features, max_disp, iterations=5):
    """
    使用图结构进行立体匹配
    
    参数:
        left_features (torch.Tensor): 左图像特征，形状为 [C, H, W]
        right_features (torch.Tensor): 右图像特征，形状为 [C, H, W]
        max_disp (int): 最大视差值
        iterations (int): 消息传递迭代次数
        
    返回:
        torch.Tensor: 视差图，形状为 [H, W]
    """
    c, h, w = left_features.shape
    
    # 构建网格图
    graph = build_grid_graph(h, w)
    
    # 添加图像特征
    graph = add_image_features_to_graph(graph, left_features)
    
    # 计算初始匹配代价
    matching_cost = torch.zeros((h * w, max_disp), device=left_features.device)
    
    for d in range(max_disp):
        # 移位右图像特征
        shifted_right = torch.zeros_like(right_features)
        if d == 0:
            shifted_right = right_features.clone()
        else:
            shifted_right[:, :, d:] = right_features[:, :, :-d]
        
        # 计算特征相似度
        similarity = F.cosine_similarity(
            left_features.unsqueeze(3),
            shifted_right.unsqueeze(3),
            dim=0
        )
        
        # 填充匹配代价
        matching_cost[:, d] = similarity.view(-1)
    
    # 初始化图消息传递器
    message_passer = GraphMessagePasser(iterations=iterations)
    
    # 运行消息传递
    refined_cost = message_passer.run_message_passing(graph, matching_cost)
    
    # 获取最佳视差
    disparity = torch.argmin(refined_cost, dim=1).float()
    
    # 重塑为图像形状
    disparity = disparity.reshape(h, w)
    
    return disparity


def create_factor_graph(height, width, max_disp):
    """
    创建用于BP的因子图
    
    参数:
        height (int): 图像高度
        width (int): 图像宽度
        max_disp (int): 最大视差值
        
    返回:
        dgl.DGLGraph: 因子图
    """
    # 变量节点（像素）
    num_variables = height * width
    
    # 因子节点（相邻像素对）
    # 水平相邻的像素对
    horizontal_factors = [(i * width + j, i * width + j + 1) 
                         for i in range(height) 
                         for j in range(width - 1)]
    
    # 垂直相邻的像素对
    vertical_factors = [(i * width + j, (i + 1) * width + j) 
                       for i in range(height - 1) 
                       for j in range(width)]
    
    # 合并所有因子
    all_factors = horizontal_factors + vertical_factors
    num_factors = len(all_factors)
    
    # 构建边: 变量到因子
    src_var_to_factor = []
    dst_var_to_factor = []
    
    for factor_id, (var1, var2) in enumerate(all_factors):
        factor_node = num_variables + factor_id
        
        # 变量1连接到因子
        src_var_to_factor.append(var1)
        dst_var_to_factor.append(factor_node)
        
        # 变量2连接到因子
        src_var_to_factor.append(var2)
        dst_var_to_factor.append(factor_node)
    
    # 构建边: 因子到变量
    src_factor_to_var = dst_var_to_factor
    dst_factor_to_var = src_var_to_factor
    
    # 合并所有边
    src_nodes = src_var_to_factor + src_factor_to_var
    dst_nodes = dst_var_to_factor + dst_factor_to_var
    
    # 创建DGL图
    g = dgl.graph((torch.tensor(src_nodes), torch.tensor(dst_nodes)))
    
    # 添加节点类型
    node_types = ['variable'] * num_variables + ['factor'] * num_factors
    g.ndata['type'] = torch.tensor([0 if t == 'variable' else 1 for t in node_types])
    
    # 添加为每个变量节点可能的标签数量（视差值）
    g.ndata['num_labels'] = torch.zeros(g.number_of_nodes(), dtype=torch.int64)
    g.ndata['num_labels'][:num_variables] = max_disp
    
    return g


def initialize_beliefs(graph, data_cost):
    """
    初始化节点信念
    
    参数:
        graph (dgl.DGLGraph): 因子图
        data_cost (torch.Tensor): 数据项代价，形状为 [H*W, max_disp]
        
    返回:
        dgl.DGLGraph: 初始化了信念的图
    """
    num_variables = torch.sum(graph.ndata['type'] == 0).item()
    max_disp = data_cost.shape[1]
    
    # 初始化变量节点的信念为数据项代价
    beliefs = torch.zeros((graph.number_of_nodes(), max_disp), device=data_cost.device)
    beliefs[:num_variables] = -data_cost  # 负数据代价作为初始信念
    
    graph.ndata['belief'] = beliefs
    
    # 初始化消息
    graph.edata['message'] = torch.zeros((graph.number_of_edges(), max_disp), 
                                       device=data_cost.device)
    
    return graph


def run_belief_propagation_on_graph(graph, smoothness_cost, iterations=5):
    """
    在因子图上运行信念传播
    
    参数:
        graph (dgl.DGLGraph): 因子图
        smoothness_cost (torch.Tensor): 平滑项代价，形状为 [max_disp, max_disp]
        iterations (int): 迭代次数
        
    返回:
        dgl.DGLGraph: 更新了信念的图
    """
    for _ in range(iterations):
        # 变量到因子的消息
        graph.update_all(
            lambda edges: {'msg': edges.src['belief'] - edges.data['message']},
            lambda nodes: {'tmp': torch.stack([m for m in nodes.mailbox['msg']], dim=1)},
            lambda nodes: {'belief_update': nodes.data['tmp'].sum(dim=1)},
            etype=('variable', 'to', 'factor')
        )
        
        # 因子到变量的消息
        # 这里需要应用平滑项代价
        # 简化的实现，实际情况可能需要更复杂的处理
        graph.update_all(
            lambda edges: {'msg': F.softmax(edges.src['belief_update'], dim=1)},
            lambda nodes: {'belief': nodes.mailbox['msg'].sum(dim=1)},
            etype=('factor', 'to', 'variable')
        )
    
    return graph


def disparity_from_graph_beliefs(graph, height, width):
    """
    从图信念中获取视差图
    
    参数:
        graph (dgl.DGLGraph): 因子图
        height (int): 图像高度
        width (int): 图像宽度
        
    返回:
        torch.Tensor: 视差图，形状为 [H, W]
    """
    # 获取变量节点的信念
    variable_beliefs = graph.ndata['belief'][graph.ndata['type'] == 0]
    
    # 获取最可能的视差
    disparity = torch.argmax(variable_beliefs, dim=1).float()
    
    # 重塑为图像形状
    disparity = disparity.reshape(height, width)
    
    return disparity