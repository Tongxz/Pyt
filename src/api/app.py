# Flask application
# Flask应用主文件

from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
from typing import Dict, Any

# 创建Flask应用
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 支持中文JSON

# 启用CORS
CORS(app)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.errorhandler(404)
def not_found(error) -> tuple:
    """
    404错误处理
    """
    return jsonify({
        'error': 'Not Found',
        'message': '请求的资源不存在',
        'status_code': 404
    }), 404


@app.errorhandler(500)
def internal_error(error) -> tuple:
    """
    500错误处理
    """
    return jsonify({
        'error': 'Internal Server Error',
        'message': '服务器内部错误',
        'status_code': 500
    }), 500


@app.errorhandler(400)
def bad_request(error) -> tuple:
    """
    400错误处理
    """
    return jsonify({
        'error': 'Bad Request',
        'message': '请求参数错误',
        'status_code': 400
    }), 400


@app.before_request
def before_request():
    """
    请求前处理
    """
    # 记录请求信息
    logger.info(f"{request.method} {request.url} - {request.remote_addr}")


@app.after_request
def after_request(response):
    """
    请求后处理
    """
    # 添加响应头
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


@app.route('/health', methods=['GET'])
def health_check() -> Dict[str, Any]:
    """
    健康检查接口
    """
    return jsonify({
        'status': 'healthy',
        'message': '服务运行正常',
        'timestamp': int(time.time())
    })


@app.route('/api/v1/info', methods=['GET'])
def get_api_info() -> Dict[str, Any]:
    """
    获取API信息
    """
    return jsonify({
        'name': 'Human Behavior Detection API',
        'version': '1.0.0',
        'description': '人体行为检测系统API',
        'endpoints': {
            'health': '/health',
            'detection': '/api/v1/detection',
            'tracking': '/api/v1/tracking',
            'behavior': '/api/v1/behavior',
            'regions': '/api/v1/regions',
            'config': '/api/v1/config'
        }
    })


def create_app(config_name: str = 'default') -> Flask:
    """
    应用工厂函数
    
    Args:
        config_name: 配置名称
    
    Returns:
        Flask应用实例
    """
    # 注册蓝图
    from .routes import detection, tracking, behavior, regions, config
    
    app.register_blueprint(detection.bp)
    app.register_blueprint(tracking.bp)
    app.register_blueprint(behavior.bp)
    app.register_blueprint(regions.bp)
    app.register_blueprint(config.bp)
    
    return app


if __name__ == '__main__':
    import time
    
    # 创建应用
    application = create_app()
    
    # 运行应用
    application.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )