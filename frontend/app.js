// 人体行为检测系统前端应用
class HumanDetectionApp {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.websocket = null;
        this.videoStream = null;
        this.isDetecting = false;
        this.detectionInterval = null;
        
        this.initializeElements();
        this.bindEvents();
        this.checkApiHealth();
    }

    initializeElements() {
        // 图像上传相关元素
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.uploadBtn = document.getElementById('uploadBtn');
        this.detectBtn = document.getElementById('detectBtn');
        this.uploadStatus = document.getElementById('uploadStatus');
        this.uploadResult = document.getElementById('uploadResult');
        
        // 实时检测相关元素
        this.videoElement = document.getElementById('videoElement');
        this.canvasOverlay = document.getElementById('canvasOverlay');
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.realtimeStatus = document.getElementById('realtimeStatus');
        this.realtimeResult = document.getElementById('realtimeResult');
    }

    bindEvents() {
        // 图像上传事件
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        this.uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        this.uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        
        this.fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        this.uploadBtn.addEventListener('click', () => this.fileInput.click());
        this.detectBtn.addEventListener('click', this.detectImage.bind(this));
        
        // 实时检测事件
        this.startBtn.addEventListener('click', this.startRealTimeDetection.bind(this));
        this.stopBtn.addEventListener('click', this.stopRealTimeDetection.bind(this));
    }

    // 检查API健康状态
    async checkApiHealth() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            const data = await response.json();
            
            if (data.status === 'healthy') {
                this.showStatus('uploadStatus', '✅ API服务连接正常', 'success');
                this.showStatus('realtimeStatus', '✅ 实时检测服务就绪', 'success');
            } else {
                throw new Error('API服务异常');
            }
        } catch (error) {
            this.showStatus('uploadStatus', '❌ API服务连接失败', 'error');
            this.showStatus('realtimeStatus', '❌ 无法连接到检测服务', 'error');
            console.error('API健康检查失败:', error);
        }
    }

    // 拖拽事件处理
    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleFile(files[0]);
        }
    }

    // 文件选择处理
    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.handleFile(file);
        }
    }

    // 处理选中的文件
    handleFile(file) {
        if (!file.type.startsWith('image/')) {
            this.showStatus('uploadStatus', '❌ 请选择有效的图片文件', 'error');
            return;
        }

        if (file.size > 10 * 1024 * 1024) { // 10MB限制
            this.showStatus('uploadStatus', '❌ 文件大小不能超过10MB', 'error');
            return;
        }

        this.selectedFile = file;
        this.detectBtn.disabled = false;
        this.showStatus('uploadStatus', `✅ 已选择文件: ${file.name}`, 'success');
        
        // 预览图片
        const reader = new FileReader();
        reader.onload = (e) => {
            this.uploadResult.innerHTML = `
                <div class="result-container">
                    <img src="${e.target.result}" class="result-image" alt="预览图片">
                    <p style="text-align: center; color: #666;">点击"开始检测"进行人体检测</p>
                </div>
            `;
        };
        reader.readAsDataURL(file);
    }

    // 图像检测
    async detectImage() {
        if (!this.selectedFile) {
            this.showStatus('uploadStatus', '❌ 请先选择图片', 'error');
            return;
        }

        this.detectBtn.disabled = true;
        this.showStatus('uploadStatus', '🔄 正在检测中...', 'info');

        try {
            const formData = new FormData();
            formData.append('file', this.selectedFile);

            const response = await fetch(`${this.apiBaseUrl}/api/v1/detect/image`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`检测失败: ${response.status}`);
            }

            const result = await response.json();
            this.displayDetectionResult(result);
            this.showStatus('uploadStatus', '✅ 检测完成', 'success');

        } catch (error) {
            this.showStatus('uploadStatus', `❌ 检测失败: ${error.message}`, 'error');
            console.error('图像检测失败:', error);
        } finally {
            this.detectBtn.disabled = false;
        }
    }

    // 显示检测结果
    displayDetectionResult(result) {
        const { detections, detection_count, annotated_image } = result;
        
        let confidenceSum = 0;
        let maxConfidence = 0;
        
        detections.forEach(detection => {
            confidenceSum += detection.confidence;
            maxConfidence = Math.max(maxConfidence, detection.confidence);
        });
        
        const avgConfidence = detection_count > 0 ? confidenceSum / detection_count : 0;
        
        this.uploadResult.innerHTML = `
            <div class="result-container">
                <img src="${annotated_image}" class="result-image" alt="检测结果">
                <div class="detection-info">
                    <div class="info-card">
                        <div class="info-value">${detection_count}</div>
                        <div class="info-label">检测到的人数</div>
                    </div>
                    <div class="info-card">
                        <div class="info-value">${(avgConfidence * 100).toFixed(1)}%</div>
                        <div class="info-label">平均置信度</div>
                    </div>
                    <div class="info-card">
                        <div class="info-value">${(maxConfidence * 100).toFixed(1)}%</div>
                        <div class="info-label">最高置信度</div>
                    </div>
                </div>
            </div>
        `;
    }

    // 开始实时检测
    async startRealTimeDetection() {
        try {
            // 获取摄像头权限
            this.videoStream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 }
            });
            
            this.videoElement.srcObject = this.videoStream;
            this.setupCanvas();
            
            // 连接WebSocket
            this.connectWebSocket();
            
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.isDetecting = true;
            
            this.showStatus('realtimeStatus', '✅ 实时检测已启动', 'success');
            
            // 开始发送视频帧进行检测
            this.startFrameCapture();
            
        } catch (error) {
            this.showStatus('realtimeStatus', `❌ 启动失败: ${error.message}`, 'error');
            console.error('启动实时检测失败:', error);
        }
    }

    // 停止实时检测
    stopRealTimeDetection() {
        this.isDetecting = false;
        
        if (this.detectionInterval) {
            clearInterval(this.detectionInterval);
            this.detectionInterval = null;
        }
        
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        
        if (this.videoStream) {
            this.videoStream.getTracks().forEach(track => track.stop());
            this.videoStream = null;
        }
        
        this.videoElement.srcObject = null;
        this.clearCanvas();
        
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        
        this.showStatus('realtimeStatus', '⏹️ 实时检测已停止', 'info');
        this.realtimeResult.innerHTML = '';
    }

    // 设置画布
    setupCanvas() {
        const video = this.videoElement;
        const canvas = this.canvasOverlay;
        
        video.addEventListener('loadedmetadata', () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.style.width = video.offsetWidth + 'px';
            canvas.style.height = video.offsetHeight + 'px';
        });
    }

    // 清空画布
    clearCanvas() {
        const canvas = this.canvasOverlay;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    // 连接WebSocket
    connectWebSocket() {
        const wsUrl = this.apiBaseUrl.replace('http', 'ws') + '/ws';
        
        // 如果已有连接，先关闭
        if (this.websocket) {
            this.websocket.close();
        }
        
        this.websocket = new WebSocket(wsUrl);
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        
        this.websocket.onopen = () => {
            console.log('WebSocket连接已建立');
            this.reconnectAttempts = 0;
            this.showStatus('realtimeStatus', '✅ WebSocket连接成功', 'success');
        };
        
        this.websocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                
                if (data.type === 'detection_result') {
                    this.handleRealtimeResult(data);
                } else if (data.type === 'error') {
                    console.error('WebSocket错误:', data.message);
                    this.showStatus('realtimeStatus', `❌ ${data.message}`, 'error');
                } else if (data.type === 'ping') {
                    // 响应心跳
                    this.websocket.send(JSON.stringify({type: 'pong'}));
                }
            } catch (error) {
                console.error('WebSocket消息解析错误:', error);
            }
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket错误:', error);
            this.showStatus('realtimeStatus', '❌ WebSocket连接错误', 'error');
        };
        
        this.websocket.onclose = (event) => {
            console.log('WebSocket连接已关闭', event.code, event.reason);
            
            // 如果是正在检测且不是正常关闭，尝试重连
            if (this.isDetecting && event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;
                const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1); // 指数退避
                
                this.showStatus('realtimeStatus', `🔄 重连中... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`, 'warning');
                
                setTimeout(() => {
                    if (this.isDetecting) {
                        this.connectWebSocket();
                    }
                }, delay);
            } else if (this.reconnectAttempts >= this.maxReconnectAttempts) {
                this.showStatus('realtimeStatus', '❌ 重连失败，请手动重启', 'error');
                this.stopRealTimeDetection();
            }
        };
    }

    // 开始帧捕获
    startFrameCapture() {
        this.detectionInterval = setInterval(() => {
            if (this.isDetecting && this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.captureAndSendFrame();
            }
        }, 500); // 每500ms发送一帧
    }

    // 捕获并发送视频帧
    captureAndSendFrame() {
        const video = this.videoElement;
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        ctx.drawImage(video, 0, 0);
        
        canvas.toBlob((blob) => {
            const reader = new FileReader();
            reader.onload = () => {
                const base64 = reader.result;
                this.websocket.send(JSON.stringify({
                    type: 'image',
                    data: base64
                }));
            };
            reader.readAsDataURL(blob);
        }, 'image/jpeg', 0.8);
    }

    // 处理实时检测结果
    handleRealtimeResult(data) {
        const { detections, detection_count } = data;
        
        // 在画布上绘制检测框
        this.drawDetections(detections);
        
        // 更新检测信息
        let confidenceSum = 0;
        let maxConfidence = 0;
        
        detections.forEach(detection => {
            confidenceSum += detection.confidence;
            maxConfidence = Math.max(maxConfidence, detection.confidence);
        });
        
        const avgConfidence = detection_count > 0 ? confidenceSum / detection_count : 0;
        
        this.realtimeResult.innerHTML = `
            <div class="detection-info">
                <div class="info-card">
                    <div class="info-value">${detection_count}</div>
                    <div class="info-label">当前人数</div>
                </div>
                <div class="info-card">
                    <div class="info-value">${(avgConfidence * 100).toFixed(1)}%</div>
                    <div class="info-label">平均置信度</div>
                </div>
                <div class="info-card">
                    <div class="info-value">${(maxConfidence * 100).toFixed(1)}%</div>
                    <div class="info-label">最高置信度</div>
                </div>
            </div>
        `;
    }

    // 在画布上绘制检测框
    drawDetections(detections) {
        const canvas = this.canvasOverlay;
        const ctx = canvas.getContext('2d');
        const video = this.videoElement;
        
        // 清空画布
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // 计算缩放比例
        const scaleX = canvas.width / video.videoWidth;
        const scaleY = canvas.height / video.videoHeight;
        
        detections.forEach((detection, index) => {
            const [x1, y1, x2, y2] = detection.bbox;
            const confidence = detection.confidence;
            
            // 缩放坐标
            const scaledX1 = x1 * scaleX;
            const scaledY1 = y1 * scaleY;
            const scaledX2 = x2 * scaleX;
            const scaledY2 = y2 * scaleY;
            
            // 绘制检测框
            ctx.strokeStyle = '#00ff00';
            ctx.lineWidth = 3;
            ctx.strokeRect(scaledX1, scaledY1, scaledX2 - scaledX1, scaledY2 - scaledY1);
            
            // 绘制标签背景
            const label = `Person ${(confidence * 100).toFixed(1)}%`;
            ctx.font = '16px Arial';
            const textWidth = ctx.measureText(label).width;
            
            ctx.fillStyle = '#00ff00';
            ctx.fillRect(scaledX1, scaledY1 - 25, textWidth + 10, 25);
            
            // 绘制标签文字
            ctx.fillStyle = '#000';
            ctx.fillText(label, scaledX1 + 5, scaledY1 - 5);
        });
    }

    // 显示状态信息
    showStatus(elementId, message, type) {
        const element = document.getElementById(elementId);
        const loadingIcon = type === 'info' && message.includes('检测中') ? '<span class="loading"></span>' : '';
        element.innerHTML = `<div class="status ${type}">${loadingIcon}${message}</div>`;
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new HumanDetectionApp();
});