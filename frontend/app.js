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
        this.updateUIForFileType(); // 初始化UI状态
        this.checkApiHealth();
    }

    initializeElements() {
        // 图像/视频上传相关元素
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.uploadBtn = document.getElementById('uploadBtn');
        this.detectBtn = document.getElementById('detectBtn');
        this.uploadStatus = document.getElementById('uploadStatus');
        this.uploadResult = document.getElementById('uploadResult');
        this.uploadText = document.getElementById('uploadText');
        this.uploadFormats = document.getElementById('uploadFormats');
        this.fileTypeRadios = document.querySelectorAll('input[name="fileType"]');

        // 实时检测相关元素
        this.videoElement = document.getElementById('videoElement');
        this.canvasOverlay = document.getElementById('canvasOverlay');
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.realtimeStatus = document.getElementById('realtimeStatus');
        this.realtimeResult = document.getElementById('realtimeResult');

        // 当前选择的文件类型
        this.currentFileType = 'image';
    }

    bindEvents() {
        // 文件类型选择事件
        this.fileTypeRadios.forEach(radio => {
            radio.addEventListener('change', this.handleFileTypeChange.bind(this));
        });

        // 图像/视频上传事件
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        this.uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        this.uploadArea.addEventListener('drop', this.handleDrop.bind(this));

        this.fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        this.uploadBtn.addEventListener('click', () => this.fileInput.click());
        this.detectBtn.addEventListener('click', this.detectFile.bind(this));

        // 实时检测事件
        this.startBtn.addEventListener('click', this.startRealTimeDetection.bind(this));
        this.stopBtn.addEventListener('click', this.stopRealTimeDetection.bind(this));
    }

    // 检查API健康状态
    async checkApiHealth() {
        try {
            console.log('正在检查API健康状态:', `${this.apiBaseUrl}/health`);
            const response = await fetch(`${this.apiBaseUrl}/health`, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                },
                mode: 'cors'
            });
            
            console.log('API响应状态:', response.status);
            console.log('API响应头:', response.headers);
            
            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            
            const contentType = response.headers.get('content-type');
            console.log('响应内容类型:', contentType);
            
            if (!contentType || !contentType.includes('application/json')) {
                const text = await response.text();
                console.error('非JSON响应:', text);
                throw new Error('服务器返回非JSON格式数据');
            }
            
            const data = await response.json();
            console.log('API响应数据:', data);

            if (data.status === 'healthy') {
                this.showStatus('uploadStatus', '✅ API服务连接正常', 'success');
                this.showStatus('realtimeStatus', '✅ 实时检测服务就绪', 'success');
            } else {
                throw new Error('API服务异常');
            }
        } catch (error) {
            console.error('API健康检查详细错误:', error);
            this.showStatus('uploadStatus', `❌ API服务连接失败: ${error.message}`, 'error');
            this.showStatus('realtimeStatus', '❌ 无法连接到检测服务', 'error');
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

    // 处理文件类型切换
    handleFileTypeChange(e) {
        this.currentFileType = e.target.value;
        this.updateUIForFileType();

        // 清除已选择的文件
        this.selectedFile = null;
        this.fileInput.value = '';
        this.detectBtn.disabled = true;
        this.uploadResult.innerHTML = '';
        this.showStatus('uploadStatus', '', '');
    }

    // 根据文件类型更新UI
    updateUIForFileType() {
        if (this.currentFileType === 'image') {
            this.uploadText.textContent = '点击或拖拽图片到此处';
            this.uploadFormats.textContent = '支持 JPG、PNG、GIF 格式';
            this.fileInput.accept = 'image/*';
        } else if (this.currentFileType === 'video') {
            this.uploadText.textContent = '点击或拖拽视频到此处';
            this.uploadFormats.textContent = '支持 MP4、AVI、MOV 格式';
            this.fileInput.accept = 'video/*';
        }
    }

    // 处理选中的文件
    handleFile(file) {
        // 验证文件类型
        if (this.currentFileType === 'image' && !file.type.startsWith('image/')) {
            this.showStatus('uploadStatus', '❌ 请选择有效的图片文件', 'error');
            return;
        }

        if (this.currentFileType === 'video' && !file.type.startsWith('video/')) {
            this.showStatus('uploadStatus', '❌ 请选择有效的视频文件', 'error');
            return;
        }

        if (file.size > 10 * 1024 * 1024) { // 10MB限制
            this.showStatus('uploadStatus', '❌ 文件大小不能超过10MB', 'error');
            return;
        }

        this.selectedFile = file;
        this.detectBtn.disabled = false;

        // 显示文件信息
        const fileSize = (file.size / 1024 / 1024).toFixed(2);
        const fileType = this.currentFileType === 'image' ? '图片' : '视频';
        this.showStatus('uploadStatus', `✅ 已选择${fileType}: ${file.name} (${fileSize}MB)`, 'success');

        // 预览文件
        if (this.currentFileType === 'image') {
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
        } else {
            // 视频预览
            const reader = new FileReader();
            reader.onload = (e) => {
                this.uploadResult.innerHTML = `
                    <div class="result-container">
                        <video src="${e.target.result}" class="result-image" controls alt="预览视频"></video>
                        <p style="text-align: center; margin-top: 1rem; color: #666;">视频预览 - 点击播放按钮查看</p>
                    </div>
                `;
            };
            reader.readAsDataURL(file);
        }
    }

    // 检测文件（图片或视频）
    async detectFile() {
        if (!this.selectedFile) {
            const fileType = this.currentFileType === 'image' ? '图片' : '视频';
            this.showStatus('uploadStatus', `❌ 请先选择${fileType}`, 'error');
            return;
        }

        this.detectBtn.disabled = true;
        const fileType = this.currentFileType === 'image' ? '图片' : '视频';
        this.showStatus('uploadStatus', `🔄 正在检测${fileType}中...`, 'info');

        try {
            const formData = new FormData();
            formData.append('file', this.selectedFile);

            // 根据文件类型选择不同的API端点
            const endpoint = this.currentFileType === 'image'
                ? '/api/v1/detect/hairnet'
                : '/api/v1/detect/hairnet/video';

            const response = await fetch(`${this.apiBaseUrl}${endpoint}`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`检测失败: ${response.status}`);
            }

            const result = await response.json();
            this.displayDetectionResult(result);
            this.showStatus('uploadStatus', `✅ ${fileType}检测完成`, 'success');

        } catch (error) {
            this.showStatus('uploadStatus', `❌ ${fileType}检测失败: ${error.message}`, 'error');
            console.error(`${fileType}检测失败:`, error);
        } finally {
            this.detectBtn.disabled = false;
        }
    }

    // 显示检测结果
    displayDetectionResult(result) {
        const fileType = this.currentFileType === 'image' ? '图片' : '视频';

        let detectionData;

        // 根据文件类型处理不同的数据结构
        if (this.currentFileType === 'video') {
            // 视频检测返回的数据结构
            detectionData = result.overall_statistics || {};
        } else {
            // 图片检测返回的数据结构
            detectionData = result.detections || {};
        }

        // 发网检测结果数据
        const totalPersons = detectionData.total_persons || 0;
        const personsWithHairnet = detectionData.persons_with_hairnet || 0;
        const personsWithoutHairnet = detectionData.persons_without_hairnet || 0;
        const complianceRate = detectionData.compliance_rate || 0;
        const avgConfidence = detectionData.average_confidence || 0;

        // 视频特有信息
        let additionalInfo = '';
        if (this.currentFileType === 'video' && result.video_info) {
            additionalInfo = `
                <div class="info-card">
                    <div class="info-value">${result.video_info.processed_frames || 0}</div>
                    <div class="info-label">处理帧数</div>
                </div>
                <div class="info-card">
                    <div class="info-value">${(result.video_info.duration || 0).toFixed(1)}s</div>
                    <div class="info-label">视频时长</div>
                </div>
            `;
        }

        // 显示带标注的结果图片
        let annotatedImageHtml = '';
        if (result.annotated_image) {
            annotatedImageHtml = `
                <div class="annotated-image-container">
                    <div class="image-header">
                        <h4>🎯 智能标注结果</h4>
                        <div class="image-controls">
                            <button class="btn-secondary" onclick="this.parentElement.parentElement.parentElement.querySelector('.result-image').style.transform = this.parentElement.parentElement.parentElement.querySelector('.result-image').style.transform === 'scale(1.5)' ? 'scale(1)' : 'scale(1.5)'">🔍 放大/缩小</button>
                            <button class="btn-secondary" onclick="window.app.downloadAnnotatedImage('${result.annotated_image}')">💾 下载图片</button>
                        </div>
                    </div>
                    <div class="image-wrapper">
                        <img src="data:image/jpeg;base64,${result.annotated_image}" class="result-image enhanced-annotation" alt="智能标注结果图片" onclick="this.classList.toggle('fullscreen')">
                        <div class="image-overlay">
                            <div class="annotation-legend">
                                <div class="legend-item">
                                    <div class="legend-color compliant"></div>
                                    <span>✓ 合规 (佩戴发网)</span>
                                </div>
                                <div class="legend-item">
                                    <div class="legend-color non-compliant"></div>
                                    <span>✗ 违规 (未佩戴发网)</span>
                                </div>
                                <div class="legend-note">
                                    💡 点击图片可全屏查看详细标注
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        this.uploadResult.innerHTML = `
            <div class="result-container">
                <h3>🔍 ${fileType}发网检测结果</h3>
                ${annotatedImageHtml}
                <div class="detection-info">
                    <div class="info-card">
                        <div class="info-value">${totalPersons}</div>
                        <div class="info-label">检测到的人数</div>
                    </div>
                    <div class="info-card">
                        <div class="info-value">${personsWithHairnet}</div>
                        <div class="info-label">佩戴发网人数</div>
                    </div>
                    <div class="info-card">
                        <div class="info-value">${personsWithoutHairnet}</div>
                        <div class="info-label">未佩戴发网人数</div>
                    </div>
                    <div class="info-card">
                        <div class="info-value compliance-${complianceRate >= 0.8 ? 'good' : complianceRate >= 0.5 ? 'medium' : 'poor'}">${(complianceRate * 100).toFixed(1)}%</div>
                        <div class="info-label">合规率</div>
                    </div>
                    <div class="info-card">
                        <div class="info-value">${(avgConfidence * 100).toFixed(1)}%</div>
                        <div class="info-label">平均置信度</div>
                    </div>
                    ${additionalInfo}
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

                if (data.type === 'hairnet_detection_result') {
                    this.handleHairnetRealtimeResult(data);
                } else if (data.type === 'detection_result') {
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
            if (!blob) {
                console.error('Canvas toBlob failed - blob is null');
                return;
            }

            const reader = new FileReader();
            reader.onload = () => {
                const base64 = reader.result;
                if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                    this.websocket.send(JSON.stringify({
                        type: 'image',
                        data: base64
                    }));
                }
            };
            reader.onerror = (error) => {
                console.error('FileReader error:', error);
            };
            reader.readAsDataURL(blob);
        }, 'image/jpeg', 0.8);
    }

    // 下载标注图片
    downloadAnnotatedImage(base64Data) {
        try {
            // 创建下载链接
            const link = document.createElement('a');
            link.href = `data:image/jpeg;base64,${base64Data}`;
            link.download = `hairnet_detection_${new Date().getTime()}.jpg`;

            // 触发下载
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            // 显示成功提示
            this.showNotification('📥 标注图片已下载', 'success');
        } catch (error) {
            console.error('下载图片失败:', error);
            this.showNotification('❌ 下载失败，请重试', 'error');
        }
    }

    // 显示通知消息
    showNotification(message, type = 'info') {
        // 创建通知元素
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;

        // 添加样式
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 20px;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            z-index: 10000;
            animation: slideIn 0.3s ease-out;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        `;

        // 根据类型设置背景色
        switch(type) {
            case 'success':
                notification.style.backgroundColor = '#10b981';
                break;
            case 'error':
                notification.style.backgroundColor = '#ef4444';
                break;
            default:
                notification.style.backgroundColor = '#3b82f6';
        }

        // 添加到页面
        document.body.appendChild(notification);

        // 3秒后自动移除
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-in';
            setTimeout(() => {
                if (notification.parentNode) {
                    document.body.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }

    // 处理发网检测实时结果
    handleHairnetRealtimeResult(data) {
        const { detections, detection_count } = data;

        // 确保detections是数组
        const detectionsArray = Array.isArray(detections) ? detections : [];

        // 在画布上绘制检测框
        this.drawHairnetDetections(detectionsArray);

        // 统计发网佩戴情况
        const withHairnet = detectionsArray.filter(d => d.has_hairnet).length;
        const withoutHairnet = detection_count - withHairnet;
        const complianceRate = detection_count > 0 ? (withHairnet / detection_count * 100).toFixed(1) : '0.0';

        // 更新检测信息
        let confidenceSum = 0;
        let maxConfidence = 0;

        detectionsArray.forEach(detection => {
            confidenceSum += (detection.hairnet_confidence || 0);
            maxConfidence = Math.max(maxConfidence, (detection.hairnet_confidence || 0));
        });

        const avgConfidence = detectionsArray.length > 0 ? confidenceSum / detectionsArray.length : 0;

        this.realtimeResult.innerHTML = `
            <div class="detection-info">
                <div class="info-card">
                    <div class="info-value">${detection_count}</div>
                    <div class="info-label">当前人数</div>
                </div>
                <div class="info-card">
                    <div class="info-value">${withHairnet}</div>
                    <div class="info-label">佩戴发网</div>
                </div>
                <div class="info-card">
                    <div class="info-value">${withoutHairnet}</div>
                    <div class="info-label">未佩戴</div>
                </div>
                <div class="info-card">
                    <div class="info-value">${complianceRate}%</div>
                    <div class="info-label">合规率</div>
                </div>
                <div class="info-card">
                    <div class="info-value">${(avgConfidence * 100).toFixed(1)}%</div>
                    <div class="info-label">平均置信度</div>
                </div>
            </div>
        `;
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

    // 在画布上绘制发网检测框
    drawHairnetDetections(detections) {
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
            const confidence = detection.hairnet_confidence;
            const hasHairnet = detection.has_hairnet;

            // 缩放坐标
            const scaledX1 = x1 * scaleX;
            const scaledY1 = y1 * scaleY;
            const scaledX2 = x2 * scaleX;
            const scaledY2 = y2 * scaleY;

            // 根据发网佩戴情况设置颜色
            const color = hasHairnet ? '#00ff00' : '#ff0000';

            // 绘制检测框
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(scaledX1, scaledY1, scaledX2 - scaledX1, scaledY2 - scaledY1);

            // 绘制标签背景
            const status = hasHairnet ? '✅' : '❌';
            const label = `${status} ${(confidence * 100).toFixed(1)}%`;
            ctx.font = '16px Arial';
            const textWidth = ctx.measureText(label).width;

            ctx.fillStyle = color;
            ctx.fillRect(scaledX1, scaledY1 - 25, textWidth + 10, 25);

            // 绘制标签文字
            ctx.fillStyle = '#fff';
            ctx.fillText(label, scaledX1 + 5, scaledY1 - 5);
        });
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

    displayAnnotatedImage(imageData) {
        // 创建或更新标注图片显示区域
        let annotatedContainer = document.querySelector('.annotated-image-container');
        if (!annotatedContainer) {
            annotatedContainer = document.createElement('div');
            annotatedContainer.className = 'annotated-image-container';

            // 插入到结果容器的开头
            const resultContainer = this.uploadResult.querySelector('.result-container');
            if (resultContainer) {
                resultContainer.insertBefore(annotatedContainer, resultContainer.firstChild.nextSibling);
            }
        }

        annotatedContainer.innerHTML = `
            <h4>🎯 标注结果图片</h4>
            <img src="${imageData}" class="result-image" alt="标注结果图片" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
        `;
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    window.app = new HumanDetectionApp();
});
