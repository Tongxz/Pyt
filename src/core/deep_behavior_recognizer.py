import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class HandBehaviorDataset(Dataset):
    """手部行为数据集
    
    用于训练和推理的时序手部行为数据
    """

    def __init__(self, sequences: List[np.ndarray], labels: List[int], sequence_length: int = 30):
        """
        初始化数据集
        
        Args:
            sequences: 特征序列列表，每个序列形状为 (time_steps, feature_dim)
            labels: 标签列表 (0: 无行为, 1: 洗手, 2: 消毒)
            sequence_length: 序列长度
        """
        self.sequences = sequences
        self.labels = labels
        self.sequence_length = sequence_length
        
        # 数据预处理
        self.scaler = StandardScaler()
        self._preprocess_data()
    
    def _preprocess_data(self):
        """预处理数据"""
        # 收集所有特征用于标准化
        all_features = []
        for seq in self.sequences:
            if len(seq) > 0:
                all_features.extend(seq)
        
        if all_features:
            all_features = np.array(all_features)
            self.scaler.fit(all_features)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # 标准化特征
        if len(sequence) > 0:
            sequence = self.scaler.transform(sequence)
        
        # 填充或截断到固定长度
        if len(sequence) < self.sequence_length:
            # 填充零
            padding = np.zeros((self.sequence_length - len(sequence), sequence.shape[1]))
            sequence = np.vstack([sequence, padding])
        elif len(sequence) > self.sequence_length:
            # 截断
            sequence = sequence[-self.sequence_length:]
        
        return torch.FloatTensor(sequence), torch.LongTensor([label])


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerBehaviorClassifier(nn.Module):
    """基于Transformer的行为分类器
    
    使用Transformer架构进行时序手部行为识别
    """
    
    def __init__(self, 
                 input_dim: int = 50,  # 输入特征维度
                 d_model: int = 128,   # Transformer模型维度
                 nhead: int = 8,       # 注意力头数
                 num_layers: int = 4,  # Transformer层数
                 num_classes: int = 3, # 分类数量
                 dropout: float = 0.1,
                 max_seq_len: int = 60):
        """
        初始化Transformer分类器
        
        Args:
            input_dim: 输入特征维度
            d_model: Transformer模型维度
            nhead: 多头注意力的头数
            num_layers: Transformer编码器层数
            num_classes: 分类类别数
            dropout: Dropout比率
            max_seq_len: 最大序列长度
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # 注意力权重可视化
        self.attention_weights = None
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x: 输入序列 (batch_size, seq_len, input_dim)
            mask: 注意力掩码 (batch_size, seq_len)
            
        Returns:
            分类logits (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # 位置编码
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # 创建padding mask
        if mask is None:
            # 简单的全零检测作为padding mask
            mask = (x.sum(dim=-1) == 0)  # (batch_size, seq_len)
        
        # Transformer编码
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # 全局平均池化（忽略padding部分）
        if mask is not None:
            # 将padding位置设为0
            encoded = encoded.masked_fill(mask.unsqueeze(-1), 0)
            # 计算有效长度
            valid_lengths = (~mask).sum(dim=1, keepdim=True).float()
            # 平均池化
            pooled = encoded.sum(dim=1) / (valid_lengths + 1e-8)
        else:
            pooled = encoded.mean(dim=1)
        
        # 分类
        logits = self.classifier(pooled)
        
        return logits
    
    def get_attention_weights(self):
        """获取注意力权重（用于可视化）"""
        return self.attention_weights


class DeepBehaviorRecognizer:
    """深度学习行为识别器
    
    集成Transformer模型进行高精度行为识别
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = 'auto',
                 sequence_length: int = 30,
                 feature_dim: int = 50):
        """
        初始化深度行为识别器
        
        Args:
            model_path: 预训练模型路径
            device: 计算设备 ('cpu', 'cuda', 'auto')
            sequence_length: 输入序列长度
            feature_dim: 特征维度
        """
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        
        # 设备选择
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 初始化模型
        self.model = TransformerBehaviorClassifier(
            input_dim=feature_dim,
            d_model=128,
            nhead=8,
            num_layers=4,
            num_classes=3,
            dropout=0.1,
            max_seq_len=sequence_length
        ).to(self.device)
        
        # 加载预训练模型
        if model_path and self._load_model(model_path):
            logger.info(f"Loaded pre-trained model from {model_path}")
        else:
            logger.info("Using randomly initialized model")
        
        # 特征缓存
        self.feature_buffer = deque(maxlen=sequence_length)
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        
        # 行为标签映射
        self.label_map = {0: 'none', 1: 'handwash', 2: 'sanitize'}
        self.confidence_threshold = 0.7
        
        logger.info(f"DeepBehaviorRecognizer initialized on {self.device}")
    
    def _load_model(self, model_path: str) -> bool:
        """加载预训练模型"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 加载标准化器
            if 'scaler_mean' in checkpoint and 'scaler_scale' in checkpoint:
                self.scaler.mean_ = checkpoint['scaler_mean']
                self.scaler.scale_ = checkpoint['scaler_scale']
                self.scaler_fitted = True
            
            self.model.eval()
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def save_model(self, model_path: str):
        """保存模型"""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'sequence_length': self.sequence_length,
                'feature_dim': self.feature_dim,
            }
            
            # 保存标准化器参数
            if self.scaler_fitted:
                checkpoint['scaler_mean'] = self.scaler.mean_
                checkpoint['scaler_scale'] = self.scaler.scale_
            
            torch.save(checkpoint, model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def extract_features_from_motion_data(self, motion_data: Dict[str, Any]) -> np.ndarray:
        """从运动数据中提取特征向量
        
        Args:
            motion_data: 运动分析数据
            
        Returns:
            特征向量
        """
        features = []
        
        # 基础统计特征 (6维)
        features.extend([
            motion_data.get('avg_speed', 0.0),
            motion_data.get('max_speed', 0.0),
            motion_data.get('speed_variance', 0.0),
            motion_data.get('avg_acceleration', 0.0),
            motion_data.get('position_variance_x', 0.0),
            motion_data.get('position_variance_y', 0.0),
        ])
        
        # 轨迹形状特征 (5维)
        features.extend([
            motion_data.get('trajectory_length', 0.0),
            motion_data.get('displacement', 0.0),
            motion_data.get('tortuosity', 0.0),
            motion_data.get('convex_hull_area', 0.0),
            motion_data.get('direction_changes', 0),
        ])
        
        # 频域特征 (6维)
        features.extend([
            motion_data.get('dominant_frequency_x', 0.0),
            motion_data.get('dominant_frequency_y', 0.0),
            motion_data.get('spectral_centroid_x', 0.0),
            motion_data.get('spectral_centroid_y', 0.0),
            motion_data.get('spectral_rolloff_x', 0.0),
            motion_data.get('spectral_rolloff_y', 0.0),
        ])
        
        # 运动模式特征 (4维)
        features.extend([
            motion_data.get('periodicity_score', 0.0),
            motion_data.get('smoothness_index', 0.0),
            motion_data.get('directional_consistency', 0.0),
            motion_data.get('pause_ratio', 0.0),
        ])
        
        # 异常检测特征 (2维)
        features.extend([
            motion_data.get('anomaly_score', 0.0),
            motion_data.get('outlier_ratio', 0.0),
        ])
        
        # 手部交互特征 (27维) - 如果有双手数据
        if 'hand_interaction' in motion_data:
            interaction = motion_data['hand_interaction']
            features.extend([
                interaction.get('distance', 0.0),
                interaction.get('relative_speed', 0.0),
                interaction.get('synchronization', 0.0),
                # 可以添加更多交互特征...
            ])
        else:
            # 填充零值
            features.extend([0.0] * 27)
        
        # 确保特征维度正确
        features = features[:self.feature_dim]
        if len(features) < self.feature_dim:
            features.extend([0.0] * (self.feature_dim - len(features)))
        
        return np.array(features, dtype=np.float32)
    
    def update_features(self, motion_data: Dict[str, Any]):
        """更新特征缓存
        
        Args:
            motion_data: 运动分析数据
        """
        features = self.extract_features_from_motion_data(motion_data)
        self.feature_buffer.append(features)
        
        # 如果是第一次，初始化标准化器
        if not self.scaler_fitted and len(self.feature_buffer) >= 10:
            feature_array = np.array(list(self.feature_buffer))
            self.scaler.fit(feature_array)
            self.scaler_fitted = True
            logger.info("Feature scaler fitted")
    
    def predict_behavior(self) -> Dict[str, float]:
        """预测当前行为
        
        Returns:
            行为预测结果 {'handwash': confidence, 'sanitize': confidence, 'none': confidence}
        """
        if len(self.feature_buffer) < self.sequence_length // 2:
            return {'handwash': 0.0, 'sanitize': 0.0, 'none': 1.0}
        
        try:
            # 准备输入数据
            sequence = np.array(list(self.feature_buffer))
            
            # 标准化
            if self.scaler_fitted:
                sequence = self.scaler.transform(sequence)
            
            # 填充到固定长度
            if len(sequence) < self.sequence_length:
                padding = np.zeros((self.sequence_length - len(sequence), self.feature_dim))
                sequence = np.vstack([padding, sequence])
            elif len(sequence) > self.sequence_length:
                sequence = sequence[-self.sequence_length:]
            
            # 转换为tensor
            input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            # 预测
            self.model.eval()
            with torch.no_grad():
                logits = self.model(input_tensor)
                probabilities = F.softmax(logits, dim=1)
                probs = probabilities.cpu().numpy()[0]
            
            # 构建结果
            result = {
                'none': float(probs[0]),
                'handwash': float(probs[1]),
                'sanitize': float(probs[2])
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'handwash': 0.0, 'sanitize': 0.0, 'none': 1.0}
    
    def get_behavior_confidence(self, behavior_type: str) -> float:
        """获取特定行为的置信度
        
        Args:
            behavior_type: 行为类型 ('handwash' 或 'sanitize')
            
        Returns:
            置信度 (0.0-1.0)
        """
        predictions = self.predict_behavior()
        return predictions.get(behavior_type, 0.0)
    
    def is_behavior_detected(self, behavior_type: str, threshold: Optional[float] = None) -> bool:
        """检测是否存在特定行为
        
        Args:
            behavior_type: 行为类型
            threshold: 置信度阈值
            
        Returns:
            是否检测到行为
        """
        if threshold is None:
            threshold = self.confidence_threshold
        
        confidence = self.get_behavior_confidence(behavior_type)
        return confidence >= threshold
    
    def reset_buffer(self):
        """重置特征缓存"""
        self.feature_buffer.clear()
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'device': str(self.device),
            'sequence_length': self.sequence_length,
            'feature_dim': self.feature_dim,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'scaler_fitted': self.scaler_fitted,
            'buffer_size': len(self.feature_buffer),
        }


class BehaviorModelTrainer:
    """行为模型训练器
    
    用于训练Transformer行为识别模型
    """
    
    def __init__(self, model: TransformerBehaviorClassifier, device: str = 'auto'):
        """
        初始化训练器
        
        Args:
            model: Transformer模型
            device: 计算设备
        """
        self.model = model
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # 训练参数
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"BehaviorModelTrainer initialized on {self.device}")
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (sequences, labels) in enumerate(dataloader):
            sequences = sequences.to(self.device)
            labels = labels.squeeze().to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            logits = self.model(sequences)
            loss = self.criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, labels in dataloader:
                sequences = sequences.to(self.device)
                labels = labels.squeeze().to(self.device)
                
                logits = self.model(sequences)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                # 计算准确率
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def train(self, 
              train_dataloader: DataLoader, 
              val_dataloader: DataLoader, 
              num_epochs: int = 50,
              save_path: Optional[str] = None) -> Dict[str, List[float]]:
        """训练模型"""
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # 训练
            train_loss = self.train_epoch(train_dataloader)
            
            # 验证
            val_loss, val_accuracy = self.validate(val_dataloader)
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            # 保存最佳模型
            if val_loss < best_val_loss and save_path:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                }, save_path)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        return history