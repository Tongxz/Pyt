// 区域配置管理系统
class RegionConfigManager {
    constructor() {
        this.canvas = document.getElementById('regionCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.regions = [];
        this.currentRegion = null;
        this.isDrawing = false;
        this.backgroundImage = null;
        this.selectedRegionId = null;
        
        // 绘制状态
        this.drawingPoints = [];
        this.tempPoint = null;
        
        // 编辑模式状态
        this.editingRegionId = null;
        this.editingRegionData = null;
        
        // 颜色配置
        this.colors = {
            entrance: '#28a745',
            work_area: '#007bff',
            restricted: '#dc3545',
            monitoring: '#ffc107',
            custom: '#6f42c1'
        };
        
        this.initEventListeners();
        this.loadExistingRegions();
        this.updateRegionList();
    }
    
    initEventListeners() {
        // 画布事件
        this.canvas.addEventListener('click', this.handleCanvasClick.bind(this));
        this.canvas.addEventListener('mousemove', this.handleCanvasMouseMove.bind(this));
        this.canvas.addEventListener('contextmenu', this.handleCanvasRightClick.bind(this));
        this.canvas.addEventListener('dblclick', this.finishDrawing.bind(this));
        
        // 背景图像上传
        document.getElementById('backgroundImage').addEventListener('change', this.handleImageUpload.bind(this));
        
        // 人数限制复选框
        document.getElementById('limitOccupancy').addEventListener('change', this.toggleOccupancyConfig.bind(this));
        
        // 键盘事件
        document.addEventListener('keydown', this.handleKeyDown.bind(this));
    }
    
    handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                this.backgroundImage = img;
                // 调整画布大小以适应图像
                const maxWidth = 800;
                const maxHeight = 600;
                const ratio = Math.min(maxWidth / img.width, maxHeight / img.height);
                
                this.canvas.width = img.width * ratio;
                this.canvas.height = img.height * ratio;
                
                this.redrawCanvas();
                this.showNotification('背景图像已加载', 'success');
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
    
    handleCanvasClick(event) {
        if (!this.isDrawing) return;
        
        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        this.drawingPoints.push({ x, y });
        this.redrawCanvas();
    }
    
    handleCanvasMouseMove(event) {
        if (!this.isDrawing) return;
        
        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        this.tempPoint = { x, y };
        this.redrawCanvas();
    }
    
    handleCanvasRightClick(event) {
        event.preventDefault();
        if (this.isDrawing && this.drawingPoints.length >= 3) {
            this.finishDrawing();
        }
    }
    
    handleKeyDown(event) {
        if (event.key === 'Escape' && this.isDrawing) {
            this.cancelDrawing();
        }
    }
    
    startDrawing() {
        const regionName = document.getElementById('regionName').value.trim();
        if (!regionName) {
            this.showNotification('请输入区域名称', 'error');
            return;
        }
        
        this.isDrawing = true;
        this.drawingPoints = [];
        this.tempPoint = null;
        this.canvas.style.cursor = 'crosshair';
        this.showNotification('开始绘制区域，双击或右键完成', 'success');
    }
    
    finishDrawing() {
        if (!this.isDrawing || this.drawingPoints.length < 3) {
            this.showNotification('至少需要3个点才能形成区域', 'error');
            return;
        }
        
        const regionData = this.getRegionFormData();
        
        if (this.editingRegionId) {
            // 编辑模式：更新现有区域
            const regionIndex = this.regions.findIndex(r => r.id === this.editingRegionId);
            if (regionIndex !== -1) {
                const updatedRegion = {
                    ...this.regions[regionIndex],
                    name: regionData.name,
                    type: regionData.type,
                    description: regionData.description,
                    points: [...this.drawingPoints],
                    rules: regionData.rules,
                    color: this.colors[regionData.type] || '#6f42c1',
                    updatedAt: new Date().toISOString()
                };
                
                this.regions[regionIndex] = updatedRegion;
                this.showNotification(`区域 "${updatedRegion.name}" 更新成功`, 'success');
            }
            
            // 清除编辑状态
            this.editingRegionId = null;
            this.editingRegionData = null;
        } else {
            // 创建模式：添加新区域
            const region = {
                id: 'region_' + Date.now(),
                name: regionData.name,
                type: regionData.type,
                description: regionData.description,
                points: [...this.drawingPoints],
                rules: regionData.rules,
                isActive: true,
                createdAt: new Date().toISOString(),
                color: this.colors[regionData.type] || '#6f42c1'
            };
            
            this.regions.push(region);
            this.showNotification(`区域 "${region.name}" 创建成功`, 'success');
        }
        
        this.isDrawing = false;
        this.drawingPoints = [];
        this.tempPoint = null;
        this.canvas.style.cursor = 'default';
        
        this.updateRegionList();
        this.redrawCanvas();
        this.clearForm();
    }
    
    cancelDrawing() {
        this.isDrawing = false;
        this.drawingPoints = [];
        this.tempPoint = null;
        this.canvas.style.cursor = 'default';
        
        // 清除编辑状态
        this.editingRegionId = null;
        this.editingRegionData = null;
        
        this.redrawCanvas();
        this.showNotification('绘制已取消', 'info');
    }
    
    getRegionFormData() {
        const rules = {};
        
        if (document.getElementById('requireHairnet').checked) {
            rules.require_hairnet = true;
        }
        
        if (document.getElementById('limitOccupancy').checked) {
            rules.max_occupancy = parseInt(document.getElementById('maxOccupancy').value) || 5;
        }
        
        if (document.getElementById('timeRestriction').checked) {
            rules.time_restriction = true;
        }
        
        return {
            name: document.getElementById('regionName').value.trim(),
            type: document.getElementById('regionType').value,
            description: document.getElementById('regionDescription').value.trim(),
            rules: rules
        };
    }
    
    clearForm() {
        document.getElementById('regionName').value = '';
        document.getElementById('regionDescription').value = '';
        document.getElementById('requireHairnet').checked = true;
        document.getElementById('limitOccupancy').checked = false;
        document.getElementById('timeRestriction').checked = false;
        this.toggleOccupancyConfig();
    }
    
    toggleOccupancyConfig() {
        const checkbox = document.getElementById('limitOccupancy');
        const config = document.getElementById('occupancyConfig');
        config.style.display = checkbox.checked ? 'block' : 'none';
    }
    
    redrawCanvas() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // 绘制背景图像
        if (this.backgroundImage) {
            this.ctx.drawImage(this.backgroundImage, 0, 0, this.canvas.width, this.canvas.height);
        }
        
        // 绘制已保存的区域
        this.regions.forEach(region => {
            this.drawRegion(region, region.id === this.selectedRegionId);
        });
        
        // 绘制正在绘制的区域
        if (this.isDrawing && this.drawingPoints.length > 0) {
            this.drawCurrentDrawing();
        }
    }
    
    drawRegion(region, isSelected = false) {
        if (region.points.length < 3) return;
        
        this.ctx.save();
        
        // 绘制区域填充
        this.ctx.beginPath();
        this.ctx.moveTo(region.points[0].x, region.points[0].y);
        for (let i = 1; i < region.points.length; i++) {
            this.ctx.lineTo(region.points[i].x, region.points[i].y);
        }
        this.ctx.closePath();
        
        // 设置填充样式
        this.ctx.fillStyle = region.color + (region.isActive ? '40' : '20');
        this.ctx.fill();
        
        // 绘制边框
        this.ctx.strokeStyle = isSelected ? '#ff0000' : region.color;
        this.ctx.lineWidth = isSelected ? 3 : 2;
        this.ctx.stroke();
        
        // 绘制顶点
        region.points.forEach((point, index) => {
            this.ctx.beginPath();
            this.ctx.arc(point.x, point.y, 4, 0, 2 * Math.PI);
            this.ctx.fillStyle = isSelected ? '#ff0000' : region.color;
            this.ctx.fill();
            
            // 绘制顶点编号
            this.ctx.fillStyle = '#fff';
            this.ctx.font = '12px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(index + 1, point.x, point.y + 4);
        });
        
        // 绘制区域标签
        const centerX = region.points.reduce((sum, p) => sum + p.x, 0) / region.points.length;
        const centerY = region.points.reduce((sum, p) => sum + p.y, 0) / region.points.length;
        
        this.ctx.fillStyle = '#000';
        this.ctx.font = 'bold 14px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(region.name, centerX, centerY);
        
        // 绘制状态指示器
        this.ctx.beginPath();
        this.ctx.arc(centerX + 50, centerY - 20, 6, 0, 2 * Math.PI);
        this.ctx.fillStyle = region.isActive ? '#28a745' : '#dc3545';
        this.ctx.fill();
        
        this.ctx.restore();
    }
    
    drawCurrentDrawing() {
        if (this.drawingPoints.length === 0) return;
        
        this.ctx.save();
        
        // 绘制已确定的线段
        this.ctx.strokeStyle = '#007bff';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);
        
        this.ctx.beginPath();
        this.ctx.moveTo(this.drawingPoints[0].x, this.drawingPoints[0].y);
        for (let i = 1; i < this.drawingPoints.length; i++) {
            this.ctx.lineTo(this.drawingPoints[i].x, this.drawingPoints[i].y);
        }
        
        // 绘制到鼠标位置的临时线段
        if (this.tempPoint) {
            this.ctx.lineTo(this.tempPoint.x, this.tempPoint.y);
            // 如果有足够的点，绘制闭合线
            if (this.drawingPoints.length >= 3) {
                this.ctx.lineTo(this.drawingPoints[0].x, this.drawingPoints[0].y);
            }
        }
        
        this.ctx.stroke();
        
        // 绘制顶点
        this.drawingPoints.forEach((point, index) => {
            this.ctx.beginPath();
            this.ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
            this.ctx.fillStyle = '#007bff';
            this.ctx.fill();
            
            this.ctx.fillStyle = '#fff';
            this.ctx.font = '12px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(index + 1, point.x, point.y + 4);
        });
        
        this.ctx.restore();
    }
    
    updateRegionList() {
        const listContainer = document.getElementById('regionList');
        listContainer.innerHTML = '';
        
        if (this.regions.length === 0) {
            listContainer.innerHTML = '<p style="text-align: center; color: #666; padding: 20px;">暂无区域配置</p>';
            return;
        }
        
        this.regions.forEach(region => {
            const regionItem = document.createElement('div');
            regionItem.className = 'region-item';
            if (region.id === this.selectedRegionId) {
                regionItem.classList.add('active');
            }
            
            const rulesText = Object.keys(region.rules).map(key => {
                switch (key) {
                    case 'require_hairnet': return '发网检测';
                    case 'max_occupancy': return `人数限制(${region.rules[key]})`;
                    case 'time_restriction': return '时间限制';
                    default: return key;
                }
            }).join(', ') || '无规则';
            
            regionItem.innerHTML = `
                <h4>
                    <span class="status-indicator ${region.isActive ? 'status-active' : 'status-inactive'}"></span>
                    ${region.name}
                </h4>
                <p><strong>类型:</strong> ${this.getTypeDisplayName(region.type)}</p>
                <p><strong>规则:</strong> ${rulesText}</p>
                <p><strong>描述:</strong> ${region.description || '无描述'}</p>
                <div class="region-actions">
                    <button class="btn btn-primary" onclick="regionManager.selectRegion('${region.id}')">选择</button>
                    <button class="btn btn-secondary" onclick="regionManager.editRegion('${region.id}')">编辑</button>
                    <button class="btn ${region.isActive ? 'btn-secondary' : 'btn-success'}" 
                            onclick="regionManager.toggleRegion('${region.id}')">
                        ${region.isActive ? '禁用' : '启用'}
                    </button>
                    <button class="btn btn-danger" onclick="regionManager.deleteRegion('${region.id}')">删除</button>
                </div>
            `;
            
            listContainer.appendChild(regionItem);
        });
    }
    
    getTypeDisplayName(type) {
        const typeNames = {
            entrance: '入口区域',
            work_area: '工作区域',
            restricted: '限制区域',
            monitoring: '监控区域',
            custom: '自定义'
        };
        return typeNames[type] || type;
    }
    
    selectRegion(regionId) {
        this.selectedRegionId = this.selectedRegionId === regionId ? null : regionId;
        this.updateRegionList();
        this.redrawCanvas();
    }
    
    editRegion(regionId) {
        const region = this.regions.find(r => r.id === regionId);
        if (!region) return;
        
        // 填充表单
        document.getElementById('regionName').value = region.name;
        document.getElementById('regionType').value = region.type;
        document.getElementById('regionDescription').value = region.description;
        
        // 设置规则
        document.getElementById('requireHairnet').checked = region.rules.require_hairnet || false;
        document.getElementById('limitOccupancy').checked = !!region.rules.max_occupancy;
        document.getElementById('timeRestriction').checked = region.rules.time_restriction || false;
        
        if (region.rules.max_occupancy) {
            document.getElementById('maxOccupancy').value = region.rules.max_occupancy;
        }
        
        this.toggleOccupancyConfig();
        
        // 设置编辑模式，保存原区域数据
        this.editingRegionId = regionId;
        this.editingRegionData = JSON.parse(JSON.stringify(region)); // 深拷贝
        
        // 将区域的点加载到绘制状态
        this.drawingPoints = [...region.points];
        this.isDrawing = true;
        
        // 重绘画布以显示编辑状态
        this.redrawCanvas();
        
        this.showNotification('区域已加载到编辑器，可以修改点位或直接保存', 'success');
    }
    
    toggleRegion(regionId) {
        const region = this.regions.find(r => r.id === regionId);
        if (!region) return;
        
        region.isActive = !region.isActive;
        this.updateRegionList();
        this.redrawCanvas();
        this.showNotification(`区域 "${region.name}" 已${region.isActive ? '启用' : '禁用'}`, 'success');
    }
    
    deleteRegion(regionId, showNotification = true) {
        const regionIndex = this.regions.findIndex(r => r.id === regionId);
        if (regionIndex === -1) return;
        
        const regionName = this.regions[regionIndex].name;
        this.regions.splice(regionIndex, 1);
        
        if (this.selectedRegionId === regionId) {
            this.selectedRegionId = null;
        }
        
        this.updateRegionList();
        this.redrawCanvas();
        
        if (showNotification) {
            this.showNotification(`区域 "${regionName}" 已删除`, 'success');
        }
    }
    
    async clearCanvas() {
        console.log('clearCanvas function called');
        if (confirm('确定要清空所有区域吗？此操作不可撤销。')) {
            console.log('User confirmed clear canvas');
            this.regions = [];
            this.selectedRegionId = null;
            this.updateRegionList();
            this.redrawCanvas();
            
            // 同步到服务器
            try {
                console.log('Sending clear request to server');
                const response = await fetch('/api/regions', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        regions: [],
                        canvas_size: {
                            width: this.canvas.width,
                            height: this.canvas.height
                        }
                    })
                });
                
                console.log('Server response:', response.status);
                if (response.ok) {
                    this.showNotification('画布已清空并同步到服务器', 'success');
                } else {
                    this.showNotification('画布已清空，但同步到服务器失败', 'warning');
                }
            } catch (error) {
                console.error('Clear canvas sync error:', error);
                this.showNotification('画布已清空，但同步到服务器失败', 'warning');
            }
        } else {
            console.log('User cancelled clear canvas');
        }
    }
    
    clearBackground() {
        this.backgroundImage = null;
        this.canvas.width = 800;
        this.canvas.height = 600;
        this.redrawCanvas();
        document.getElementById('backgroundImage').value = '';
        this.showNotification('背景图像已清除', 'success');
    }
    
    async saveRegions() {
        if (this.regions.length === 0) {
            this.showNotification('没有区域需要保存', 'warning');
            return;
        }
        
        try {
            const response = await fetch('/api/regions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    regions: this.regions,
                    canvas_size: {
                        width: this.canvas.width,
                        height: this.canvas.height
                    }
                })
            });
            
            if (response.ok) {
                this.showNotification('区域配置已保存到服务器', 'success');
            } else {
                throw new Error('保存失败');
            }
        } catch (error) {
            console.error('Save error:', error);
            this.showNotification('保存失败，请检查网络连接', 'error');
        }
    }
    
    async loadRegions() {
        try {
            const response = await fetch('/api/regions');
            if (response.ok) {
                const data = await response.json();
                this.regions = data.regions || [];
                
                if (data.canvas_size) {
                    this.canvas.width = data.canvas_size.width;
                    this.canvas.height = data.canvas_size.height;
                }
                
                this.updateRegionList();
                this.redrawCanvas();
                this.showNotification('区域配置已从服务器加载', 'success');
            } else {
                throw new Error('加载失败');
            }
        } catch (error) {
            console.error('Load error:', error);
            this.showNotification('加载失败，请检查网络连接', 'error');
        }
    }
    
    async loadExistingRegions() {
        // 尝试从服务器加载现有配置
        try {
            const response = await fetch('/api/regions');
            if (response.ok) {
                const data = await response.json();
                if (data.regions && data.regions.length > 0) {
                    this.regions = data.regions;
                    if (data.canvas_size) {
                        this.canvas.width = data.canvas_size.width;
                        this.canvas.height = data.canvas_size.height;
                    }
                    this.updateRegionList();
                    this.redrawCanvas();
                }
            }
        } catch (error) {
            console.log('No existing regions found');
        }
    }
    
    exportConfig() {
        if (this.regions.length === 0) {
            this.showNotification('没有区域可以导出', 'warning');
            return;
        }
        
        const config = {
            regions: this.regions,
            canvas_size: {
                width: this.canvas.width,
                height: this.canvas.height
            },
            exported_at: new Date().toISOString(),
            version: '1.0'
        };
        
        const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `region_config_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        this.showNotification('配置文件已导出', 'success');
    }
    
    showNotification(message, type = 'success') {
        const notification = document.getElementById('notification');
        notification.textContent = message;
        notification.className = `notification ${type} show`;
        
        setTimeout(() => {
            notification.classList.remove('show');
        }, 3000);
    }
}

// 全局函数
function startDrawing() {
    regionManager.startDrawing();
}

function clearCanvas() {
    regionManager.clearCanvas();
}

function clearBackground() {
    regionManager.clearBackground();
}

function saveRegions() {
    regionManager.saveRegions();
}

function loadRegions() {
    regionManager.loadRegions();
}

function exportConfig() {
    regionManager.exportConfig();
}

// 初始化
let regionManager;
document.addEventListener('DOMContentLoaded', () => {
    regionManager = new RegionConfigManager();
});