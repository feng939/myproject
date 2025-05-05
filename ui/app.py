"""
Web应用主程序

该模块提供了一个用于立体视差估计的交互式Web应用。
"""

import os
import sys
import time
import numpy as np
import cv2
from PIL import Image
import io
import base64
import torch
from flask import Flask, render_template, request, jsonify

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bpnn import BPNN
from src.inference.predict import DisparityPredictor
from src.utils.visualization import colorize_disparity, create_anaglyph


app = Flask(__name__, static_folder='static', template_folder='templates')

# 全局变量
model = None
predictor = None
config = {
    'max_disp': 32,
    'feature_channels': 16,
    'iterations': 3,
    'use_attention': True,
    'use_refinement': True,
    'use_half_precision': True,
    'model_path': None,
    'target_size': (200, 200)  # 目标大小
}


def load_model():
    """加载模型"""
    global model, predictor
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = BPNN(
        max_disp=config['max_disp'],
        feature_channels=config['feature_channels'],
        iterations=config['iterations'],
        use_attention=config['use_attention'],
        use_refinement=config['use_refinement'],
        use_half_precision=config['use_half_precision']
    )
    
    # 加载模型权重（如果有）
    if config['model_path'] and os.path.exists(config['model_path']):
        checkpoint = torch.load(config['model_path'], map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    # 先将模型移动到设备
    model = model.to(device)
    
    # 如果启用半精度，立即转换模型
    if config['use_half_precision'] and device.type == 'cuda':
        model = model.half()

    # 设置为评估模式
    model.eval()
    
    # 创建预测器
    predictor = DisparityPredictor(
        model=model,
        device=device,
        target_size=config['target_size']
    )
    
    return model, predictor


def preprocess_image(image):
    """
    预处理上传的图像
    
    参数:
        image: 上传的图像文件
        
    返回:
        numpy.ndarray: 预处理后的图像
    """
    # 读取上传的图像
    img = Image.open(image)
    img = np.array(img)
    
    # 如果是RGBA格式，转换为RGB
    if len(img.shape) > 2 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    return img


@app.route('/')
def index():
    """首页路由"""
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    """处理上传的图像对"""
    # 检查模型是否已加载
    global model, predictor
    if model is None or predictor is None:
        model, predictor = load_model()
    
    # 获取上传的图像
    if 'left_image' not in request.files or 'right_image' not in request.files:
        return jsonify({'error': '请上传左右图像'})
    
    left_file = request.files['left_image']
    right_file = request.files['right_image']
    
    # 检查文件类型
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if (not left_file.filename.lower().endswith(tuple('.' + ext for ext in allowed_extensions)) or
        not right_file.filename.lower().endswith(tuple('.' + ext for ext in allowed_extensions))):
        return jsonify({'error': '仅支持PNG、JPG、JPEG格式的图像'})
    
    try:
        # 预处理图像
        left_img = preprocess_image(left_file)
        right_img = preprocess_image(right_file)

        # 确保左右图像尺寸一致
        if left_img.shape[:2] != right_img.shape[:2]:
            # 将右图像调整为左图像大小
            right_img = cv2.resize(right_img, (left_img.shape[1], left_img.shape[0]))
        
        # 预测视差图
        start_time = time.time()
        result = predictor.predict(left_img, right_img)
        disparity = result['disparity']
        processing_time = result['processing_time']
        
        # 生成彩色视差图
        color_disparity = colorize_disparity(disparity)
        
        # 生成红青立体图
        anaglyph = create_anaglyph(left_img, right_img)
        
        # 将图像转换为base64格式
        color_disparity_base64 = image_to_base64(color_disparity)
        anaglyph_base64 = image_to_base64(anaglyph)
        
        # 计算统计信息
        valid_mask = disparity > 0
        min_disp = disparity[valid_mask].min() if np.any(valid_mask) else 0
        max_disp = disparity[valid_mask].max() if np.any(valid_mask) else 0
        mean_disp = disparity[valid_mask].mean() if np.any(valid_mask) else 0
        
        # 返回结果
        return jsonify({
            'success': True,
            'disparity': color_disparity_base64,
            'anaglyph': anaglyph_base64,
            'processing_time': f"{processing_time:.3f}秒",
            'stats': {
                'min_disparity': f"{min_disp:.2f}",
                'max_disparity': f"{max_disp:.2f}",
                'mean_disparity': f"{mean_disp:.2f}"
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'处理图像时出错: {str(e)}'})


@app.route('/models', methods=['GET'])
def list_models():
    """列出可用模型"""
    # 查找checkpoints目录中的模型文件
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
    
    if not os.path.exists(model_dir):
        return jsonify({'models': []})
    
    # 查找所有.pth文件
    models = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    
    return jsonify({'models': models})


@app.route('/load_model', methods=['POST'])
def load_model_route():
    """加载选择的模型"""
    global config
    
    # 获取模型信息
    data = request.json
    model_path = data.get('model_path')
    use_attention = data.get('use_attention', True)
    max_disp = int(data.get('max_disp', 32))
    iterations = int(data.get('iterations', 3))
    use_refinement = data.get('use_refinement', True)
    
    # 构造完整的模型路径
    if model_path:
        full_model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            'checkpoints', 
            model_path
        )
    else:
        full_model_path = None
    
    # 更新配置
    config['model_path'] = full_model_path
    config['use_attention'] = use_attention
    config['max_disp'] = max_disp
    config['iterations'] = iterations
    config['use_refinement'] = use_refinement
    
    # 重新加载模型
    load_model()
    
    return jsonify({'success': True, 'message': f'已加载模型: {model_path}'})


@app.route('/benchmark', methods=['POST'])
def benchmark():
    """执行基准测试"""
    # 检查模型是否已加载
    global model, predictor
    if model is None or predictor is None:
        model, predictor = load_model()
    
    # 获取上传的图像
    if 'left_image' not in request.files or 'right_image' not in request.files:
        return jsonify({'error': '请上传左右图像'})
    
    left_file = request.files['left_image']
    right_file = request.files['right_image']
    
    try:
        # 预处理图像
        left_img = preprocess_image(left_file)
        right_img = preprocess_image(right_file)
        
        # 确保左右图像尺寸一致
        if left_img.shape[:2] != right_img.shape[:2]:
            # 将右图像调整为左图像大小
            right_img = cv2.resize(right_img, (left_img.shape[1], left_img.shape[0]))
        
        # 执行多次预测以测试性能
        num_tests = 5
        times = []
        
        for _ in range(num_tests):
            start_time = time.time()
            predictor.predict(left_img, right_img)
            end_time = time.time()
            times.append(end_time - start_time)
        
        # 计算统计信息
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # 计算每秒帧数（FPS）
        fps = 1 / avg_time
        
        # 返回结果
        return jsonify({
            'success': True,
            'benchmark': {
                'avg_time': f"{avg_time:.3f}秒",
                'min_time': f"{min_time:.3f}秒",
                'max_time': f"{max_time:.3f}秒",
                'fps': f"{fps:.2f}帧/秒"
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'基准测试出错: {str(e)}'})


def image_to_base64(image):
    """
    将图像转换为base64格式
    
    参数:
        image (numpy.ndarray): 输入图像，RGB格式
        
    返回:
        str: base64编码的图像字符串
    """
    # 确保图像是RGB格式
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # 将图像转换为PIL图像
    pil_img = Image.fromarray(image.astype(np.uint8))
    
    # 将PIL图像转换为base64
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return f"data:image/jpeg;base64,{img_str}"


if __name__ == '__main__':
    # 加载模型
    load_model()
    
    # 启动应用
    app.run(host='0.0.0.0', port=5000, debug=True)