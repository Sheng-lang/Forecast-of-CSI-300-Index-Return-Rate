import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="沪深300预测系统", page_icon="📈", layout="wide")

MODEL_NAMES = {'lstm': 'LSTM', 'transformer': 'Transformer', 'lstm_transformer': 'LSTM-Transformer'}
M_COLORS = {'lstm': '#2563eb', 'transformer': '#dc2626', 'lstm_transformer': '#16a34a'}

# ── Plotly 通用模板 ──
PX_LAYOUT = dict(
    margin=dict(l=50, r=20, t=40, b=40),
    paper_bgcolor='white', plot_bgcolor='white',
    font=dict(family='system-ui', size=13, color='#334155'),
    xaxis=dict(gridcolor='#f1f5f9', zerolinecolor='#e2e8f0'),
    yaxis=dict(gridcolor='#f1f5f9', zerolinecolor='#e2e8f0'),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    hovermode='x unified',
)

def _fig(fig):
    """统一应用模板"""
    fig.update_layout(**PX_LAYOUT)
    fig.update_traces(hovertemplate='%{y:.6f}<extra></extra>')
    return fig


# ── 兼容旧版 pandas 的安全表格显示（避免 PyArrow 依赖） ──
def safe_dataframe(df, style_kwargs=None):
    """将 DataFrame 转为 HTML 表格显示，兼容 pandas 0.25+"""
    if style_kwargs:
        # 手动格式化各列，不使用 .style（Styler 在 0.25.1 没有 to_html）
        df = df.copy()
        for col, fmt in style_kwargs.items():
            if col in df.columns:
                df[col] = df[col].apply(lambda x, f=fmt: f.format(x) if pd.notnull(x) else '')
    html = df.to_html(index=False, escape=False, border=0,
                      classes='dataframe', justify='center')
    # 美化表格样式
    html = html.replace('<table', '<table style="border-collapse:collapse;width:100%;"')
    html = html.replace('<th', '<th style="padding:6px 10px;text-align:center;background:#f1f5f9;'
                        'font-weight:600;font-size:0.85rem;color:#334155;border-bottom:2px solid #e2e8f0;"')
    html = html.replace('<td', '<td style="padding:5px 10px;text-align:right;font-size:0.83rem;'
                        'border-bottom:1px solid #f1f5f9;"')
    st.markdown(html, unsafe_allow_html=True)


def safe_table(df):
    """安全显示简单表格"""
    safe_dataframe(df)


# ── 数据加载（全部缓存） ──
@st.cache_data
def load_all():
    rd = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    with open(os.path.join(rd, 'results.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
    ds = pd.read_csv(os.path.join(rd, 'data_summary.csv'), parse_dates=['trade_date'])

    # 下采样收盘价
    step = max(1, len(ds) // 500)
    chart_close = ds.iloc[::step][['trade_date', 'close']].copy()

    # 预计算各模型预测图数据（下采样到300点）
    pred_charts = {}
    for n in MODEL_NAMES:
        act = np.array(data[f'{n}_actuals'])
        prd = np.array(data[f'{n}_preds'])
        s = max(1, len(act) // 300)
        pred_charts[n] = {'actual': act[::s], 'predicted': prd[::s],
                          'residual': (act - prd)[::s],
                          'actual_full': act, 'predicted_full': prd}

    # 预计算未来预测
    future_charts = {}
    for n in MODEL_NAMES:
        fp = np.array(data[f'{n}_future_preds'])
        cum = np.exp(np.cumsum(fp)) - 1
        future_charts[n] = {'daily': fp, 'cumulative': cum}

    return data, ds, chart_close, pred_charts, future_charts

try:
    results, data_summary, chart_close, pred_charts, future_charts = load_all()
except Exception as e:
    st.error(f"数据加载失败: {e}")
    st.stop()

N = len(data_summary)
D_MIN = data_summary['trade_date'].min().strftime('%Y-%m-%d')
D_MAX = data_summary['trade_date'].max().strftime('%Y-%m-%d')

# ── 极简CSS ──
st.markdown("""<style>
[data-testid="stSidebar"]{display:none;}
header,#MainMenu,footer{visibility:hidden;height:0;margin:0;padding:0;overflow:hidden;}
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:10px 0;}
.kpi-card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:14px 16px;box-shadow:0 1px 2px rgba(0,0,0,0.05);}
.kpi-label{font-size:0.82rem;color:#64748b;font-weight:600;margin-bottom:6px;}
.kpi-val{font-size:1.15rem;font-weight:700;color:#0f172a;white-space:nowrap;overflow:visible;text-overflow:clip;}
</style>""", unsafe_allow_html=True)

# ── 顶部导航 ──
if 'page' not in st.session_state:
    st.session_state.page = 0

PAGES = ["数据概览","训练过程","预测对比","指标对比","未来预测","滚动验证"]
P_ICONS = ["📊","📉","🔮","📏","🚀","🔄"]

nav_cols = st.columns(len(PAGES))
for i, (col, name) in enumerate(zip(nav_cols, PAGES)):
    active = (st.session_state.page == i)
    if col.button(f"{P_ICONS[i]} {name}", key=f"nav_{i}",
                  type="primary" if active else "secondary"):
        st.session_state.page = i
        st.experimental_rerun()

P = st.session_state.page

st.markdown(f"""
<div style="padding:12px 0 4px 0;">
    <h1 style="margin:0;font-size:1.5rem;font-weight:700;color:#0f172a;">沪深300指数收益率预测系统</h1>
    <p style="margin:2px 0 0 0;font-size:0.85rem;color:#94a3b8;">
        LSTM / Transformer / LSTM-Transformer &nbsp;|&nbsp; {D_MIN} ~ {D_MAX} &nbsp;|&nbsp; {N:,} 条样本
    </p>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  P0 - 数据概览
# ═══════════════════════════════════════════════
if P == 0:
    avg_r = data_summary['log_ret'].mean() * 252
    st.markdown(f"""<div class="kpi-grid">
        <div class="kpi-card"><div class="kpi-label">起始日期</div><div class="kpi-val">{D_MIN}</div></div>
        <div class="kpi-card"><div class="kpi-label">截止日期</div><div class="kpi-val">{D_MAX}</div></div>
        <div class="kpi-card"><div class="kpi-label">样本数量</div><div class="kpi-val">{N:,}</div></div>
        <div class="kpi-card"><div class="kpi-label">最新收盘</div><div class="kpi-val">{data_summary['close'].iloc[-1]:.2f}</div></div>
        <div class="kpi-card"><div class="kpi-label">年化收益率</div><div class="kpi-val">{avg_r*100:.2f}%</div></div>
    </div>""", unsafe_allow_html=True)

    # 收盘价走势（Plotly WebGL）
    st.subheader("收盘价走势")
    fig_close = px.line(chart_close, x='trade_date', y='close')
    fig_close.update_traces(line=dict(color='#2563eb', width=1.5), hovertemplate='日期=%{x|%Y-%m-%d}<br>收盘=%{y:.2f}<extra></extra>')
    fig_close.update_layout(yaxis_title='收盘价', xaxis_title='')
    st.plotly_chart(_fig(fig_close), height=380)

    # 双列：收益率分布 + 月度收益热力图
    ca, cb = st.columns(2)
    with ca:
        st.subheader("对数收益率分布")
        ret = data_summary['log_ret'].dropna()
        fig_hist = px.histogram(ret, nbins=80, labels={'value': '对数收益率', 'count': '频次'})
        fig_hist.update_traces(marker_color='#2563eb', marker_line=dict(color='white', width=0.5))
        fig_hist.add_vline(x=0, line_dash='dash', line_color='#dc2626', annotation_text='零线')
        fig_hist.update_layout(yaxis_title='频次')
        st.plotly_chart(_fig(fig_hist), height=300)

    with cb:
        st.subheader("月度平均收益率热力图")
        ds_copy = data_summary.copy()
        ds_copy['year'] = ds_copy['trade_date'].dt.year
        ds_copy['month'] = ds_copy['trade_date'].dt.month
        monthly = ds_copy.groupby(['year', 'month'])['log_ret'].mean().reset_index()
        monthly_pivot = monthly.pivot(index='year', columns='month', values='log_ret')
        fig_heat = px.imshow(monthly_pivot, text_auto='.4f', aspect='auto',
                             color_continuous_scale='RdBu_r',
                             labels=dict(x='月份', y='年份', color='平均收益率'))
        fig_heat.update_xaxes(tickvals=list(range(1, 13)), ticktext=[f'{m}月' for m in range(1, 13)])
        st.plotly_chart(_fig(fig_heat), height=300)

    # 统计信息
    s = data_summary['close']
    r = data_summary['log_ret']
    pos_pct = (r > 0).sum() / len(r) * 100

    sc_a, sc_b = st.columns(2)
    with sc_a:
        st.subheader("收盘价统计")
        st.markdown(f"""<div class="kpi-grid" style="grid-template-columns:repeat(3,1fr);">
            <div class="kpi-card"><div class="kpi-label">均值</div><div class="kpi-val">{s.mean():.2f}</div></div>
            <div class="kpi-card"><div class="kpi-label">标准差</div><div class="kpi-val">{s.std():.2f}</div></div>
            <div class="kpi-card"><div class="kpi-label">中位数</div><div class="kpi-val">{s.median():.2f}</div></div>
        </div><div class="kpi-grid" style="grid-template-columns:repeat(2,1fr);">
            <div class="kpi-card"><div class="kpi-label">最小值</div><div class="kpi-val">{s.min():.2f}</div></div>
            <div class="kpi-card"><div class="kpi-label">最大值</div><div class="kpi-val">{s.max():.2f}</div></div>
        </div>""", unsafe_allow_html=True)
    with sc_b:
        st.subheader("对数收益率统计")
        st.markdown(f"""<div class="kpi-grid" style="grid-template-columns:repeat(3,1fr);">
            <div class="kpi-card"><div class="kpi-label">均值</div><div class="kpi-val">{r.mean():.6f}</div></div>
            <div class="kpi-card"><div class="kpi-label">标准差</div><div class="kpi-val">{r.std():.6f}</div></div>
            <div class="kpi-card"><div class="kpi-label">上涨占比</div><div class="kpi-val">{pos_pct:.1f}%</div></div>
        </div><div class="kpi-grid" style="grid-template-columns:repeat(2,1fr);">
            <div class="kpi-card"><div class="kpi-label">最小值</div><div class="kpi-val">{r.min():.6f}</div></div>
            <div class="kpi-card"><div class="kpi-label">最大值</div><div class="kpi-val">{r.max():.6f}</div></div>
        </div>""", unsafe_allow_html=True)

    st.subheader("最近20条数据")
    cols_show = ['trade_date', 'open', 'high', 'low', 'close', 'vol', 'log_ret']
    safe_dataframe(data_summary[cols_show].tail(20), {
        'open': '{:.2f}', 'high': '{:.2f}', 'low': '{:.2f}', 'close': '{:.2f}',
        'vol': '{:,.0f}', 'log_ret': '{:.6f}'
    })


# ═══════════════════════════════════════════════
#  P1 - 训练过程
# ═══════════════════════════════════════════════
elif P == 1:
    st.subheader("训练过程分析")

    sel_mods = st.multiselect("选择模型", list(MODEL_NAMES.keys()),
        format_func=lambda x: MODEL_NAMES[x], default=list(MODEL_NAMES.keys()))

    if sel_mods:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Training Loss")
            fig_train = go.Figure()
            for n in sel_mods:
                losses = results[f'{n}_train_losses']
                epochs = list(range(1, len(losses) + 1))
                # 下采样到500点
                s = max(1, len(epochs) // 500)
                fig_train.add_trace(go.Scattergl(x=epochs[::s], y=losses[::s],
                    name=MODEL_NAMES[n], line=dict(color=M_COLORS[n], width=1.5)))
            fig_train.update_layout(xaxis_title='Epoch', yaxis_title='Loss')
            st.plotly_chart(_fig(fig_train), height=320)

        with c2:
            st.subheader("Validation Loss")
            fig_val = go.Figure()
            for n in sel_mods:
                losses = results[f'{n}_val_losses']
                epochs = list(range(1, len(losses) + 1))
                s = max(1, len(epochs) // 500)
                fig_val.add_trace(go.Scattergl(x=epochs[::s], y=losses[::s],
                    name=MODEL_NAMES[n], line=dict(color=M_COLORS[n], width=1.5)))
            fig_val.update_layout(xaxis_title='Epoch', yaxis_title='Loss')
            st.plotly_chart(_fig(fig_val), height=320)

        # 训练/验证对比（单模型）
        st.subheader("训练 vs 验证 Loss 对比")
        for n in sel_mods:
            tl = results[f'{n}_train_losses']
            vl = results[f'{n}_val_losses']
            epochs = list(range(1, len(tl) + 1))
            s = max(1, len(epochs) // 500)
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scattergl(x=epochs[::s], y=tl[::s], name='Train', line=dict(color='#2563eb', width=1.5)))
            fig_cmp.add_trace(go.Scattergl(x=epochs[::s], y=vl[::s], name='Val', line=dict(color='#dc2626', width=1.5, dash='dash')))
            fig_cmp.update_layout(xaxis_title='Epoch', yaxis_title='Loss')
            fig_cmp.update_layout(title=MODEL_NAMES[n])
            st.plotly_chart(_fig(fig_cmp), height=280)

        st.subheader("模型训练指标")
        mcs = st.columns(len(sel_mods))
        for idx, name in enumerate(sel_mods):
            with mcs[idx]:
                bv = min(results[f'{name}_val_losses'])
                ftv = results[f'{name}_train_losses'][-1]
                best_ep = np.argmin(results[f'{name}_val_losses']) + 1
                c_a, c_b, c_c = st.columns(3)
                with c_a: st.markdown(f'<div class="kpi-card"><div class="kpi-label">最终Train</div><div class="kpi-val">{ftv:.7f}</div></div>', unsafe_allow_html=True)
                with c_b: st.markdown(f'<div class="kpi-card"><div class="kpi-label">最佳Val</div><div class="kpi-val">{bv:.7f}</div></div>', unsafe_allow_html=True)
                with c_c: st.markdown(f'<div class="kpi-card"><div class="kpi-label">最佳Epoch</div><div class="kpi-val">{best_ep}</div></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  P2 - 预测对比
# ═══════════════════════════════════════════════
elif P == 2:
    mc = st.selectbox("选择模型", list(MODEL_NAMES.keys()), format_func=lambda x: MODEL_NAMES[x])
    pc = pred_charts[mc]
    m = results[f'{mc}_metrics']

    st.subheader(f"预测效果分析 — {MODEL_NAMES[mc]}")

    st.markdown(f"""<div class="kpi-grid">
        <div class="kpi-card"><div class="kpi-label">MSE</div><div class="kpi-val">{m['mse']:.7f}</div></div>
        <div class="kpi-card"><div class="kpi-label">RMSE</div><div class="kpi-val">{m['rmse']:.7f}</div></div>
        <div class="kpi-card"><div class="kpi-label">MAE</div><div class="kpi-val">{m.get('mae', 0):.7f}</div></div>
        <div class="kpi-card"><div class="kpi-label">R²</div><div class="kpi-val">{m['r2']:.6f}</div></div>
        <div class="kpi-card"><div class="kpi-label">方向准确率DA</div><div class="kpi-val">{m.get('da', 0)*100:.2f}%</div></div>
    </div>""", unsafe_allow_html=True)

    # 实际 vs 预测（Plotly WebGL）
    st.subheader("实际值 vs 预测值")
    idx = list(range(len(pc['actual'])))
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scattergl(x=idx, y=pc['actual'], name='实际值',
        line=dict(color='#2563eb', width=1.2)))
    fig_pred.add_trace(go.Scattergl(x=idx, y=pc['predicted'], name='预测值',
        line=dict(color='#dc2626', width=1.2)))
    fig_pred.update_layout(xaxis_title='时间步', yaxis_title='收益率')
    st.plotly_chart(_fig(fig_pred), height=360)

    # 双列：误差分布直方图 + 散点图
    ca, cb = st.columns(2)
    with ca:
        st.subheader("误差分布直方图")
        residual = pc['residual']
        fig_res = px.histogram(residual, nbins=60, labels={'value': '误差'})
        fig_res.update_traces(marker_color='#f59e0b', marker_line=dict(color='white', width=0.5))
        fig_res.add_vline(x=0, line_dash='dash', line_color='#dc2626')
        fig_res.update_layout(yaxis_title='频次')
        st.plotly_chart(_fig(fig_res), height=300)

    with cb:
        st.subheader("实际值 vs 预测值 散点图")
        fig_scatter = px.scatter(x=pc['actual'], y=pc['predicted'],
                                 labels={'x': '实际值', 'y': '预测值'},
                                 opacity=0.4)
        fig_scatter.update_traces(marker=dict(color=M_COLORS[mc], size=3))
        # 45度参考线
        vmin = min(pc['actual'].min(), pc['predicted'].min())
        vmax = max(pc['actual'].max(), pc['predicted'].max())
        fig_scatter.add_trace(go.Scattergl(x=[vmin, vmax], y=[vmin, vmax],
            mode='lines', line=dict(dash='dash', color='#94a3b8', width=1),
            name='理想拟合', hoverinfo='skip'))
        st.plotly_chart(_fig(fig_scatter), height=300)

    # 误差统计
    res_full = pc['actual_full'] - pc['predicted_full']
    st.subheader("误差统计")
    st.markdown(f"""<div class="kpi-grid">
        <div class="kpi-card"><div class="kpi-label">最大高估</div><div class="kpi-val">{res_full.max():.6f}</div></div>
        <div class="kpi-card"><div class="kpi-label">最大低估</div><div class="kpi-val">{res_full.min():.6f}</div></div>
        <div class="kpi-card"><div class="kpi-label">平均偏差</div><div class="kpi-val">{res_full.mean():.6f}</div></div>
        <div class="kpi-card"><div class="kpi-label">标准差</div><div class="kpi-val">{res_full.std():.6f}</div></div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  P3 - 指标对比
# ═══════════════════════════════════════════════
elif P == 3:
    st.subheader("三模型综合指标对比")

    rows = []
    for n in MODEL_NAMES:
        mt = results[f'{n}_metrics']
        rows.append({'模型': MODEL_NAMES[n], 'MSE': mt['mse'], 'RMSE': mt['rmse'],
                     'MAE': mt.get('mae', 0), 'R²': mt['r2'], 'DA': mt.get('da', 0)})
    mf = pd.DataFrame(rows).sort_values('R²', ascending=False).reset_index(drop=True)

    # 排名卡片
    medals = ["🥇", "🥈", "🥉"]
    for i in range(min(3, len(mf))):
        row = mf.iloc[i]
        st.info(f"""**{medals[i]} {row['模型']}** &nbsp;&nbsp;
            R²={row['R²']:.6f} | DA={row['DA']*100:.2f}% | RMSE={row['RMSE']:.6f} | MSE={row['MSE']:.6f} | MAE={row['MAE']:.6f}""")

    # 双列：柱状图 + 雷达图
    ca, cb = st.columns(2)
    with ca:
        st.subheader("指标柱状对比")
        compare_df = pd.DataFrame({
            'Model': [MODEL_NAMES[n] for n in MODEL_NAMES],
            'R²': [results[f'{n}_metrics']['r2'] for n in MODEL_NAMES],
            'DA%': [results[f'{n}_metrics'].get('da', 0) * 100 for n in MODEL_NAMES],
            'RMSE': [results[f'{n}_metrics']['rmse'] for n in MODEL_NAMES],
        })
        fig_bar = go.Figure()
        for col_name in ['R²', 'DA%', 'RMSE']:
            fig_bar.add_trace(go.Bar(
                x=compare_df['Model'], y=compare_df[col_name],
                name=col_name, text=compare_df[col_name].round(4),
                textposition='auto',
            ))
        fig_bar.update_layout(yaxis_title='值')
        st.plotly_chart(_fig(fig_bar), height=340)

    with cb:
        st.subheader("雷达图")
        # 归一化到 0~1 用于雷达图
        radar_metrics = ['R²', 'DA', 'RMSE', 'MAE', 'MSE']
        fig_radar = go.Figure()
        for _, row in mf.iterrows():
            vals = []
            for met in radar_metrics:
                v = row[met]
                if met in ('RMSE', 'MAE', 'MSE'):
                    # 越小越好 → 反转归一化
                    col_max = mf[met].max()
                    col_min = mf[met].min()
                    vals.append(1 - (v - col_min) / (col_max - col_min) if col_max != col_min else 1)
                else:
                    col_max = mf[met].max()
                    col_min = mf[met].min()
                    vals.append((v - col_min) / (col_max - col_min) if col_max != col_min else 1)
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],  # 闭合
                theta=radar_metrics + [radar_metrics[0]],
                fill='toself', opacity=0.15,
                name=row['模型'],
                line=dict(color=M_COLORS.get(row['模型'].lower().replace('-', '_'), '#666'), width=2),
            ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                showlegend=True)
        st.plotly_chart(_fig(fig_radar), height=340)

    st.subheader("指标明细表")
    safe_dataframe(mf, {
        'MSE': '{:.6f}', 'RMSE': '{:.6f}', 'MAE': '{:.6f}', 'R²': '{:.6f}', 'DA': '{:.4f}'
    })


# ═══════════════════════════════════════════════
#  P4 - 未来预测
# ═══════════════════════════════════════════════
elif P == 4:
    fm = st.selectbox("选择预测模型", list(MODEL_NAMES.keys()), format_func=lambda x: MODEL_NAMES[x])
    fc = future_charts[fm]

    st.subheader(f"未来10日预测 — {MODEL_NAMES[fm]}")

    st.markdown(f"""<div class="kpi-grid">
        <div class="kpi-card"><div class="kpi-label">平均日收益</div><div class="kpi-val">{fc['daily'].mean():.6f}</div></div>
        <div class="kpi-card"><div class="kpi-label">累积收益</div><div class="kpi-val">{fc['cumulative'][-1]*100:.4f}%</div></div>
        <div class="kpi-card"><div class="kpi-label">最大单日</div><div class="kpi-val">{fc['daily'].max():.6f}</div></div>
        <div class="kpi-card"><div class="kpi-label">最小单日</div><div class="kpi-val">{fc['daily'].min():.6f}</div></div>
    </div>""", unsafe_allow_html=True)

    # 预测表格
    pred_data = []
    for i in range(10):
        up = fc['daily'][i] > 0
        pred_data.append([f'Day{i+1}', f"{fc['daily'][i]:+.6f}", f"{fc['cumulative'][i]*100:+.4f}%", "↑涨" if up else "↓跌"])
    df_pred = pd.DataFrame(pred_data, columns=["日期", "日收益率", "累积收益", "方向"])
    safe_table(df_pred)


    # 双列：累积走势 + 日收益
    days = [f'D{i+1}' for i in range(10)]
    ca, cb = st.columns(2)
    with ca:
        st.subheader("累积收益走势")
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=days, y=fc['cumulative']*100,
            mode='lines+markers', name='累积收益',
            line=dict(color=M_COLORS[fm], width=2),
            marker=dict(size=6)))
        fig_cum.add_hline(y=0, line_dash='dash', line_color='#94a3b8')
        fig_cum.update_layout(yaxis_title='累积收益 (%)')
        st.plotly_chart(_fig(fig_cum), height=280)

    with cb:
        st.subheader("每日收益分布")
        colors_bar = ['#16a34a' if v >= 0 else '#dc2626' for v in fc['daily']]
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Bar(x=days, y=fc['daily'],
            marker_color=colors_bar, name='日收益率'))
        fig_daily.add_hline(y=0, line_color='#94a3b8', line_width=1)
        fig_daily.update_layout(yaxis_title='收益率')
        st.plotly_chart(_fig(fig_daily), height=280)

    # 三模型预测对比折线图
    st.subheader("三模型未来预测对比")
    fig_all = go.Figure()
    for n in MODEL_NAMES:
        fc_n = future_charts[n]
        fig_all.add_trace(go.Scatter(x=days, y=fc_n['cumulative']*100,
            mode='lines+markers', name=MODEL_NAMES[n],
            line=dict(color=M_COLORS[n], width=2),
            marker=dict(size=5)))
    fig_all.add_hline(y=0, line_dash='dash', line_color='#94a3b8')
    fig_all.update_layout(yaxis_title='累积收益 (%)')
    st.plotly_chart(_fig(fig_all), height=320)

    # 三模型对比表
    st.subheader("三模型对比")
    cmp = {'Day': [f'D{i+1}' for i in range(10)]}
    for n in MODEL_NAMES:
        cmp[MODEL_NAMES[n]] = [f"{results[f'{n}_future_preds'][i]:+.6f}" for i in range(10)]
    safe_dataframe(pd.DataFrame(cmp))



# ═══════════════════════════════════════════════
#  P5 - 滚动验证
# ═══════════════════════════════════════════════
elif P == 5:
    st.subheader("滚动窗口稳健性验证")

    rp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'rolling_results.json')
    if not os.path.exists(rp):
        st.warning("未找到滚动验证结果，请运行完整训练脚本。")
    else:
        with open(rp, 'r', encoding='utf-8') as f:
            rdata = json.load(f)

        rs = []
        all_r2, all_da = {}, {}
        for n in MODEL_NAMES:
            if n in rdata and rdata[n]:
                r2v = [r['r2'] for r in rdata[n]]
                dav = [r['da'] for r in rdata[n]]
                rmsev = [r['rmse'] for r in rdata[n]]
                rs.append({'模型': MODEL_NAMES[n], '窗口数': len(rdata[n]),
                           'R²均值': np.mean(r2v), 'R²σ': np.std(r2v),
                           'DA均值': np.mean(dav), 'DAσ': np.std(dav), 'RMSE均值': np.mean(rmsev)})
                all_r2[n] = r2v
                all_da[n] = dav

        if rs:
            sdf = pd.DataFrame(rs)
            hc = st.columns(3)
            for i, row in sdf.iterrows():
                with hc[i % 3]:
                    st.info(f"""**{row['模型']}** ({int(row['窗口数'])}个窗口)\n\n
                        R²={row['R²均值']:.4f}(σ:{row['R²σ']:.4f})\n
                        DA={row['DA均值']*100:.1f}%(σ:{row['DAσ']:.4f})\n
                        RMSE={row['RMSE均值']:.6f}""")

            # 双列：R²箱线图 + DA箱线图
            ba, bb = st.columns(2)
            with ba:
                st.subheader("R² 窗口分布")
                fig_box_r2 = go.Figure()
                for n in all_r2:
                    fig_box_r2.add_trace(go.Box(y=all_r2[n], name=MODEL_NAMES[n],
                        marker_color=M_COLORS[n], boxmean='sd'))
                fig_box_r2.update_layout(yaxis_title='R²')
                st.plotly_chart(_fig(fig_box_r2), height=300)

            with bb:
                st.subheader("DA 窗口分布")
                fig_box_da = go.Figure()
                for n in all_da:
                    fig_box_da.add_trace(go.Box(y=all_da[n], name=MODEL_NAMES[n],
                        marker_color=M_COLORS[n], boxmean='sd'))
                fig_box_da.update_layout(yaxis_title='DA')
                st.plotly_chart(_fig(fig_box_da), height=300)

            # 窗口趋势折线图
            st.subheader("各窗口 R² 变化趋势")
            fig_trend = go.Figure()
            for n in all_r2:
                windows = list(range(1, len(all_r2[n]) + 1))
                fig_trend.add_trace(go.Scattergl(x=windows, y=all_r2[n],
                    mode='lines+markers', name=MODEL_NAMES[n],
                    line=dict(color=M_COLORS[n], width=2), marker=dict(size=5)))
            fig_trend.update_layout(xaxis_title='窗口编号', yaxis_title='R²')
            st.plotly_chart(_fig(fig_trend), height=300)

            st.subheader("汇总表")
            safe_dataframe(sdf, {
                'R²均值': '{:.4f}', 'R²σ': '{:.4f}', 'DA均值': '{:.4f}', 'DAσ': '{:.4f}', 'RMSE均值': '{:.6f}'
            })

            for n in MODEL_NAMES:
                if n in rdata and rdata[n]:
                    st.subheader(f"{MODEL_NAMES[n]} 详细结果")
                    rdf = pd.DataFrame(rdata[n])
                    dc = [c for c in ['window', 'test_samples', 'mse', 'rmse', 'mae', 'r2', 'da'] if c in rdf.columns]
                    rdd = rdf[dc].copy()
                    rdd.columns = ['窗口', '样本', 'MSE', 'RMSE', 'MAE', 'R²', 'DA']
                    safe_dataframe(rdd, {
                        'MSE': '{:.6f}', 'RMSE': '{:.6f}', 'MAE': '{:.6f}', 'R²': '{:.4f}', 'DA': '{:.4f}'
                    })
