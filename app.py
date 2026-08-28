from flask import Flask, request, jsonify
import os
import json
import copy
from main import run_test
import config

app = Flask(__name__)

# 确保results目录存在
os.makedirs('results', exist_ok=True)

@app.route('/')
def home():
    return "Welcome to Bias Mitigation and Accuracy Enhancement API"

@app.route('/run', methods=['POST'])
def run_algorithm():
    """Run the bias mitigation and accuracy enhancement algorithm"""
    try:
        # 获取请求参数
        data = request.json or {}
        
        # 更新配置参数
        if 'config' in data:
            for key, value in data['config'].items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # 运行算法
        final_metrics, changed_dict, results_history = run_test()
        
        # 读取保存的结果
        results_file = os.path.join('results', 'all_results.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = {
                'config_parameters': {},
                'initial_metrics': {},
                'initial_epsilon': {},
                'iterations': [],
                'final_results': {
                    'metrics': final_metrics,
                    'changed_dict': changed_dict
                }
            }
        
        return jsonify({
            'status': 'success',
            'final_metrics': final_metrics,
            'changed_dict': changed_dict,
            'results_history': results_history,
            'all_results': all_results
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    config_params = {}
    for param_name in dir(config):
        if not param_name.startswith('__') and param_name.isupper():
            config_params[param_name] = getattr(config, param_name)
    return jsonify(config_params)

@app.route('/config', methods=['POST'])
def update_config():
    """Update configuration parameters"""
    try:
        data = request.json
        updated_params = {}
        
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
                updated_params[key] = value
        
        return jsonify({
            'status': 'success',
            'updated_params': updated_params
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/results', methods=['GET'])
def get_results():
    """Get latest results"""
    results_file = os.path.join('results', 'all_results.json')
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
        return jsonify(all_results)
    else:
        return jsonify({
            'status': 'error',
            'message': 'No results found. Run the algorithm first.'
        }), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)