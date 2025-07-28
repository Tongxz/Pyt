-- 人体行为检测系统数据库初始化脚本
-- Human Behavior Detection System Database Initialization Script

-- 创建扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- 创建用户表
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建摄像头配置表
CREATE TABLE IF NOT EXISTS cameras (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    location VARCHAR(200),
    stream_url VARCHAR(500) NOT NULL,
    camera_type VARCHAR(50) DEFAULT 'ip_camera',
    resolution VARCHAR(20) DEFAULT '1920x1080',
    fps INTEGER DEFAULT 30,
    is_active BOOLEAN DEFAULT true,
    config JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建检测区域表
CREATE TABLE IF NOT EXISTS detection_zones (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    camera_id UUID REFERENCES cameras(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    polygon JSONB NOT NULL, -- 存储多边形坐标点
    zone_type VARCHAR(50) DEFAULT 'general',
    rules JSONB, -- 存储区域规则配置
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建检测记录表
CREATE TABLE IF NOT EXISTS detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    camera_id UUID REFERENCES cameras(id) ON DELETE CASCADE,
    zone_id UUID REFERENCES detection_zones(id) ON DELETE SET NULL,
    detection_type VARCHAR(50) NOT NULL, -- person, hairnet, etc.
    confidence FLOAT NOT NULL,
    bbox JSONB NOT NULL, -- 边界框坐标
    attributes JSONB, -- 检测属性（如是否佩戴发网）
    image_path VARCHAR(500),
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建行为记录表
CREATE TABLE IF NOT EXISTS behaviors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    detection_id UUID REFERENCES detections(id) ON DELETE CASCADE,
    behavior_type VARCHAR(50) NOT NULL, -- washing_hands, wearing_hairnet, etc.
    status VARCHAR(20) NOT NULL, -- detected, violated, compliant
    confidence FLOAT,
    duration INTEGER, -- 行为持续时间（秒）
    metadata JSONB,
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建告警记录表
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    camera_id UUID REFERENCES cameras(id) ON DELETE CASCADE,
    zone_id UUID REFERENCES detection_zones(id) ON DELETE SET NULL,
    detection_id UUID REFERENCES detections(id) ON DELETE SET NULL,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) DEFAULT 'medium', -- low, medium, high, critical
    title VARCHAR(200) NOT NULL,
    description TEXT,
    status VARCHAR(20) DEFAULT 'active', -- active, acknowledged, resolved
    acknowledged_by UUID REFERENCES users(id) ON DELETE SET NULL,
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建系统配置表
CREATE TABLE IF NOT EXISTS system_configs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_key VARCHAR(100) UNIQUE NOT NULL,
    config_value JSONB NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建统计数据表
CREATE TABLE IF NOT EXISTS statistics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    camera_id UUID REFERENCES cameras(id) ON DELETE CASCADE,
    zone_id UUID REFERENCES detection_zones(id) ON DELETE SET NULL,
    stat_type VARCHAR(50) NOT NULL, -- daily_count, hourly_avg, etc.
    stat_date DATE NOT NULL,
    stat_hour INTEGER, -- 0-23, NULL for daily stats
    metrics JSONB NOT NULL, -- 统计指标数据
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_detections_camera_id ON detections(camera_id);
CREATE INDEX IF NOT EXISTS idx_detections_detected_at ON detections(detected_at);
CREATE INDEX IF NOT EXISTS idx_detections_type ON detections(detection_type);
CREATE INDEX IF NOT EXISTS idx_behaviors_detection_id ON behaviors(detection_id);
CREATE INDEX IF NOT EXISTS idx_behaviors_type ON behaviors(behavior_type);
CREATE INDEX IF NOT EXISTS idx_alerts_camera_id ON alerts(camera_id);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at);
CREATE INDEX IF NOT EXISTS idx_statistics_camera_date ON statistics(camera_id, stat_date);
CREATE INDEX IF NOT EXISTS idx_zones_camera_id ON detection_zones(camera_id);

-- 创建更新时间触发器函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 为需要的表创建更新时间触发器
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_cameras_updated_at BEFORE UPDATE ON cameras
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_zones_updated_at BEFORE UPDATE ON detection_zones
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_configs_updated_at BEFORE UPDATE ON system_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 插入默认管理员用户
INSERT INTO users (username, email, password_hash, role) VALUES 
('admin', 'admin@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6hsxq5S/kS', 'admin')
ON CONFLICT (username) DO NOTHING;

-- 插入默认系统配置
INSERT INTO system_configs (config_key, config_value, description) VALUES 
('detection_confidence_threshold', '0.5', '检测置信度阈值'),
('max_detection_history_days', '30', '检测记录保留天数'),
('alert_notification_enabled', 'true', '是否启用告警通知'),
('system_timezone', '"Asia/Shanghai"', '系统时区设置')
ON CONFLICT (config_key) DO NOTHING;

-- 创建视图：最近24小时检测统计
CREATE OR REPLACE VIEW recent_detection_stats AS
SELECT 
    c.id as camera_id,
    c.name as camera_name,
    COUNT(d.id) as total_detections,
    COUNT(CASE WHEN d.detection_type = 'person' THEN 1 END) as person_detections,
    COUNT(CASE WHEN d.attributes->>'has_hairnet' = 'true' THEN 1 END) as hairnet_compliant,
    COUNT(CASE WHEN d.attributes->>'has_hairnet' = 'false' THEN 1 END) as hairnet_violations,
    AVG(d.confidence) as avg_confidence
FROM cameras c
LEFT JOIN detections d ON c.id = d.camera_id 
    AND d.detected_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
WHERE c.is_active = true
GROUP BY c.id, c.name;

-- 创建视图：活跃告警
CREATE OR REPLACE VIEW active_alerts AS
SELECT 
    a.*,
    c.name as camera_name,
    z.name as zone_name
FROM alerts a
LEFT JOIN cameras c ON a.camera_id = c.id
LEFT JOIN detection_zones z ON a.zone_id = z.id
WHERE a.status = 'active'
ORDER BY a.created_at DESC;

-- 创建清理历史数据的函数
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS void AS $$
DECLARE
    retention_days INTEGER;
BEGIN
    -- 获取数据保留天数配置
    SELECT (config_value::text)::integer INTO retention_days
    FROM system_configs 
    WHERE config_key = 'max_detection_history_days';
    
    IF retention_days IS NULL THEN
        retention_days := 30; -- 默认保留30天
    END IF;
    
    -- 删除过期的检测记录
    DELETE FROM detections 
    WHERE detected_at < CURRENT_TIMESTAMP - (retention_days || ' days')::interval;
    
    -- 删除过期的统计数据
    DELETE FROM statistics 
    WHERE stat_date < CURRENT_DATE - (retention_days || ' days')::interval;
    
    -- 删除已解决的告警（保留7天）
    DELETE FROM alerts 
    WHERE status = 'resolved' 
    AND resolved_at < CURRENT_TIMESTAMP - INTERVAL '7 days';
    
END;
$$ LANGUAGE plpgsql;

-- 提交事务
COMMIT;