import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from pykalman import KalmanFilter
import warnings
import os
import json
import pickle

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOOKBACK = 20
TARGET_HORIZON = 1

# -------------------- 1. 从本地XLSX加载数据 --------------------
def load_local_data(filepath='指数行情_000300.xlsx'):
    df = pd.read_excel(filepath)
    # 映射中文列名
    col_map = {
        '交易日期': 'trade_date',
        '开盘指数': 'open',
        '最高指数': 'high',
        '最低指数': 'low',
        '收盘指数': 'close',
        '成交数量(股)': 'vol',
        '成交金额(元)': 'amount'
    }
    df.rename(columns=col_map, inplace=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values('trade_date').reset_index(drop=True)
    # 只保留需要的列
    df = df[['trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']]
    return df

# -------------------- 2. 特征工程与收益率计算 --------------------
def build_features(df):
    # 卡尔曼滤波预处理
    kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1],
                      initial_state_mean=df['close'].iloc[0],
                      observation_covariance=1, transition_covariance=0.01)
    state_means, _ = kf.filter(df['close'].values)
    df['smoothed_close'] = state_means.flatten()

    # 计算对数收益率
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

    # 构造技术指标特征
    df['ret_ma5'] = df['log_ret'].rolling(5).mean()
    df['ret_ma10'] = df['log_ret'].rolling(10).mean()
    df['ret_ma20'] = df['log_ret'].rolling(20).mean()
    df['vol_ratio'] = df['vol'] / df['vol'].rolling(5).mean()
    df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
    df['close_open_ratio'] = (df['close'] - df['open']) / df['open']
    df['volatility'] = df['log_ret'].rolling(10).std()
    # 新增：成交金额相关特征
    df['amount_ratio'] = df['amount'] / df['amount'].rolling(5).mean()
    df['avg_price'] = df['amount'] / df['vol']  # 均价
    df['avg_price_ratio'] = df['avg_price'].pct_change()

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# -------------------- 3. 构造监督学习样本（滑动窗口） --------------------
def create_sequences(data, targets, lookback, horizon=1):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i + lookback])
        y.append(targets[i + lookback + horizon - 1])
    return np.array(X), np.array(y)

# -------------------- 4. 数据准备（完整流程） --------------------
def prepare_data(filepath='指数行情_000300.xlsx'):
    df = load_local_data(filepath)
    df = build_features(df)

    feature_cols = ['log_ret', 'ret_ma5', 'ret_ma10', 'ret_ma20', 'vol_ratio',
                    'high_low_ratio', 'close_open_ratio', 'volatility', 'amount_ratio', 'avg_price_ratio']

    feature_data = df[feature_cols].values
    target_data = np.log(df['smoothed_close'] / df['smoothed_close'].shift(1)).fillna(0).values

    X, y = create_sequences(feature_data, target_data, LOOKBACK, TARGET_HORIZON)

    # 按时间顺序划分数据集
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))

    X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

    # 保存日期索引用于可视化
    date_index = df['trade_date'].values
    test_dates = date_index[train_size + val_size + LOOKBACK: train_size + val_size + LOOKBACK + len(y_test)]

    # 数据标准化
    n_features = X_train.shape[2]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))

    X_train = scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(DEVICE)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(DEVICE)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(DEVICE)

    batch_size = 64
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return {
        'df': df, 'feature_cols': feature_cols, 'n_features': n_features,
        'scaler': scaler, 'feature_data': feature_data, 'target_data': target_data,
        'train_loader': train_loader, 'val_loader': val_loader, 'test_loader': test_loader,
        'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'test_dates': test_dates,
        'train_size': train_size, 'val_size': val_size,
    }

# -------------------- 5. 定义模型 --------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class LSTMTransformerModel(nn.Module):
    def __init__(self, input_dim, lstm_hidden=64, d_model=32, nhead=2, num_layers=2, dim_feedforward=64, dropout=0.2):
        super(LSTMTransformerModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden, num_layers, batch_first=True, dropout=dropout)
        self.input_proj = nn.Linear(lstm_hidden, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.input_proj(lstm_out)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# -------------------- 6. 模型训练与评估 --------------------
def train_model(model, train_loader, val_loader, epochs=100, learning_rate=0.001):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses, val_losses = [], []
    best_val_loss, best_model = float('inf'), None
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward(); optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                val_loss += criterion(model(X_batch), y_batch).item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    model.load_state_dict(best_model)
    return model, train_losses, val_losses

def evaluate_model(model, test_loader):
    """评估模型，返回多维度指标：MSE/RMSE/MAE/MAPE/R2/DA"""
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            predictions.extend(model(X_batch).cpu().numpy().flatten())
            actuals.extend(y_batch.cpu().numpy().flatten())
    preds, acts = np.array(predictions), np.array(actuals)

    # 误差维度指标
    mse = mean_squared_error(acts, preds)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(acts - preds))
    # MAPE: 避免除以0，过滤掉真实值接近0的样本
    mask = np.abs(acts) > 1e-8
    mape = np.mean(np.abs((acts[mask] - preds[mask]) / acts[mask])) * 100 if mask.sum() > 0 else float('nan')
    r2 = r2_score(acts, preds)

    # 方向维度指标：DA (Directional Accuracy)
    # 判断预测涨跌方向与实际涨跌方向是否一致
    actual_dir = np.sign(acts)
    pred_dir = np.sign(preds)
    da = np.mean(actual_dir == pred_dir)

    print(f'MSE:  {mse:.6f}')
    print(f'RMSE: {rmse:.6f}')
    print(f'MAE:  {mae:.6f}')
    print(f'MAPE: {mape:.4f}%')
    print(f'R2:   {r2:.6f}')
    print(f'DA:   {da:.4f} ({da*100:.2f}%)')
    return preds, acts, mse, rmse, mae, mape, r2, da

# -------------------- 7. 滚动时间窗口验证（稳健性分析） --------------------
def rolling_window_evaluate(model_class, model_kwargs, data_dict, n_windows=3, train_ratio=0.7, val_ratio=0.15):
    """滚动时间窗口验证：在不同时间段上评估模型稳定性"""
    feature_data = data_dict['feature_data']
    target_data = data_dict['target_data']
    n_total = len(feature_data)
    
    # 计算窗口大小
    test_size = int(n_total * (1 - train_ratio - val_ratio))
    window_shift = test_size // n_windows
    
    rolling_results = []
    
    for w in range(n_windows):
        # 滚动测试集的起始位置
        test_start = n_total - test_size + w * window_shift
        test_end = min(test_start + window_shift, n_total)
        
        if test_end - test_start < 50:
            continue
            
        # 用前 test_start 的数据作为训练+验证
        train_end = int(test_start * train_ratio / (train_ratio + val_ratio))
        val_end = test_start
        
        X_all = feature_data
        y_all = target_data
        
        # 构造序列
        X, y = create_sequences(X_all, y_all, LOOKBACK, TARGET_HORIZON)
        
        # 划分
        X_train_w = X[:train_end]
        y_train_w = y[:train_end]
        X_val_w = X[train_end:val_end]
        y_val_w = y[train_end:val_end]
        X_test_w = X[test_start:test_end]
        y_test_w = y[test_start:test_end]
        
        if len(X_test_w) < 30:
            continue
        
        # 标准化
        n_feat = X_train_w.shape[2]
        scaler_w = StandardScaler()
        scaler_w.fit(X_train_w.reshape(-1, n_feat))
        
        X_train_w = scaler_w.transform(X_train_w.reshape(-1, n_feat)).reshape(X_train_w.shape)
        X_val_w = scaler_w.transform(X_val_w.reshape(-1, n_feat)).reshape(X_val_w.shape)
        X_test_w = scaler_w.transform(X_test_w.reshape(-1, n_feat)).reshape(X_test_w.shape)
        
        # 转为tensor
        X_train_t = torch.tensor(X_train_w, dtype=torch.float32).to(DEVICE)
        y_train_t = torch.tensor(y_train_w, dtype=torch.float32).view(-1, 1).to(DEVICE)
        X_val_t = torch.tensor(X_val_w, dtype=torch.float32).to(DEVICE)
        y_val_t = torch.tensor(y_val_w, dtype=torch.float32).view(-1, 1).to(DEVICE)
        X_test_t = torch.tensor(X_test_w, dtype=torch.float32).to(DEVICE)
        y_test_t = torch.tensor(y_test_w, dtype=torch.float32).view(-1, 1).to(DEVICE)
        
        train_loader_w = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=False)
        val_loader_w = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=64, shuffle=False)
        test_loader_w = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64, shuffle=False)
        
        # 训练
        model = model_class(**model_kwargs).to(DEVICE)
        model, _, _ = train_model(model, train_loader_w, val_loader_w, epochs=50)
        
        # 评估
        preds, acts, mse, rmse, mae, mape, r2, da = evaluate_model(model, test_loader_w)
        
        rolling_results.append({
            'window': w + 1,
            'test_samples': len(y_test_w),
            'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2, 'da': da
        })
        print(f"  Window {w+1}: R2={r2:.4f}, DA={da:.4f}, RMSE={rmse:.6f}")
    
    return rolling_results

# -------------------- 8. 未来预测 --------------------
def predict_future(model, last_sequence, scaler, n_features, days=10):
    """用训练好的模型预测未来N天的收益率"""
    model.eval()
    predictions = []
    current_seq = last_sequence.copy()  # shape: (lookback, n_features)

    for _ in range(days):
        # 标准化
        seq_scaled = scaler.transform(current_seq.reshape(-1, n_features)).reshape(1, LOOKBACK, n_features)
        seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            pred = model(seq_tensor).cpu().numpy().flatten()[0]
        predictions.append(pred)

        # 滑动窗口：移除最早一步，添加新预测
        new_row = current_seq[-1].copy()
        new_row[0] = pred  # 更新log_ret
        current_seq = np.vstack([current_seq[1:], new_row])

    return np.array(predictions)

# -------------------- 8. 保存结果 --------------------
def save_results(data_dict, models_results, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)

    # 保存模型
    for name, result in models_results.items():
        model = result['model']
        torch.save(model.state_dict(), os.path.join(output_dir, f'{name}.pt'))

    # 保存scaler
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(data_dict['scaler'], f)

    # 保存预测结果和指标
    save_data = {
        'feature_cols': data_dict['feature_cols'],
        'n_features': data_dict['n_features'],
        'test_dates': [str(d) for d in data_dict['test_dates']],
    }
    for name, result in models_results.items():
        save_data[f'{name}_preds'] = result['preds'].tolist()
        save_data[f'{name}_actuals'] = result['actuals'].tolist()
        save_data[f'{name}_train_losses'] = result['train_losses']
        save_data[f'{name}_val_losses'] = result['val_losses']
        save_data[f'{name}_metrics'] = {
            'mse': float(result['mse']),
            'rmse': float(result['rmse']),
            'mae': float(result['mae']),
            'mape': float(result['mape']),
            'r2': float(result['r2']),
            'da': float(result['da'])
        }
        save_data[f'{name}_future_preds'] = result['future_preds'].tolist()

    # 保存最近一段序列用于未来预测
    feature_data = data_dict['feature_data']
    last_seq = feature_data[-LOOKBACK:]
    save_data['last_sequence'] = last_seq.tolist()

    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    # 保存原始数据摘要
    df = data_dict['df']
    df[['trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount', 'log_ret']].to_csv(
        os.path.join(output_dir, 'data_summary.csv'), index=False)

    print(f"\n所有结果已保存到 {output_dir}/ 目录")

# -------------------- 9. 主流程 --------------------
def main():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '指数行情_000300.xlsx')

    print("=" * 60)
    print("沪深300指数收益率预测 - LSTM / Transformer / LSTM-Transformer")
    print("=" * 60)

    # 数据准备
    print("\n[1/4] 加载并处理数据...")
    data = prepare_data(data_path)
    df = data['df']
    print(f"数据时间范围：{df['trade_date'].min()} 至 {df['trade_date'].max()}")
    print(f"总样本数：{len(df)}")
    print(f"特征列：{data['feature_cols']}")
    print(f"训练集：{data['train_size']}, 验证集：{data['val_size']}, 测试集：{len(data['y_test'])}")

    n_features = data['n_features']
    model_classes = {
        'lstm': LSTMModel(input_dim=n_features),
        'transformer': TransformerModel(input_dim=n_features),
        'lstm_transformer': LSTMTransformerModel(input_dim=n_features),
    }

    # 训练与评估
    models_results = {}
    for name, model in model_classes.items():
        display_name = {'lstm': 'LSTM', 'transformer': 'Transformer', 'lstm_transformer': 'LSTM-Transformer'}[name]
        print(f"\n[2/4] 训练{display_name}模型...")
        model, train_losses, val_losses = train_model(model, data['train_loader'], data['val_loader'])

        print(f"\n[3/4] 评估{display_name}模型...")
        preds, actuals, mse, rmse, mae, mape, r2, da = evaluate_model(model, data['test_loader'])

        # 未来预测
        last_seq = data['feature_data'][-LOOKBACK:]
        future_preds = predict_future(model, last_seq, data['scaler'], n_features, days=10)

        models_results[name] = {
            'model': model, 'preds': preds, 'actuals': actuals,
            'train_losses': train_losses, 'val_losses': val_losses,
            'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2, 'da': da,
            'future_preds': future_preds,
        }

    # 保存结果
    print(f"\n[4/5] 保存结果...")
    save_results(data, models_results)

    # 滚动时间窗口验证（稳健性分析）
    print(f"\n[5/5] 滚动时间窗口验证（稳健性分析）...")
    rolling_all = {}
    for name, model_cls in [('lstm', LSTMModel), ('transformer', TransformerModel), ('lstm_transformer', LSTMTransformerModel)]:
        display_name = {'lstm': 'LSTM', 'transformer': 'Transformer', 'lstm_transformer': 'LSTM-Transformer'}[name]
        print(f"\n--- {display_name} 滚动验证 ---")
        kwargs = {'input_dim': n_features}
        rolling_all[name] = rolling_window_evaluate(model_cls, kwargs, data, n_windows=3)

    # 保存滚动验证结果
    rolling_save = {}
    for name, results_list in rolling_all.items():
        rolling_save[name] = [
            {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in r.items()}
            for r in results_list
        ]
    with open(os.path.join('results', 'rolling_results.json'), 'w', encoding='utf-8') as f:
        json.dump(rolling_save, f, ensure_ascii=False, indent=2)

    # 打印滚动验证汇总
    print("\n" + "=" * 60)
    print("滚动时间窗口验证汇总")
    print("=" * 60)
    for name, results_list in rolling_all.items():
        display_name = {'lstm': 'LSTM', 'transformer': 'Transformer', 'lstm_transformer': 'LSTM-Transformer'}[name]
        r2_vals = [r['r2'] for r in results_list]
        da_vals = [r['da'] for r in results_list]
        print(f"\n{display_name}: R2均值={np.mean(r2_vals):.4f}(std={np.std(r2_vals):.4f}), DA均值={np.mean(da_vals):.4f}(std={np.std(da_vals):.4f})")

    # 打印指标汇总
    print("\n" + "=" * 60)
    print("模型评估指标汇总")
    print("=" * 60)
    for name, result in models_results.items():
        display_name = {'lstm': 'LSTM', 'transformer': 'Transformer', 'lstm_transformer': 'LSTM-Transformer'}[name]
        print(f"\n{display_name}:")
        print(f"  MSE:  {result['mse']:.6f}")
        print(f"  RMSE: {result['rmse']:.6f}")
        print(f"  MAE:  {result['mae']:.6f}")
        print(f"  MAPE: {result['mape']:.4f}%")
        print(f"  R2:   {result['r2']:.6f}")
        print(f"  DA:   {result['da']:.4f} ({result['da']*100:.2f}%)")
        print(f"  未来10天预测收益率: {[f'{p:.6f}' for p in result['future_preds']]}")

    print("\n训练完成！请运行 streamlit run app.py 查看可视化界面")

if __name__ == '__main__':
    main()
