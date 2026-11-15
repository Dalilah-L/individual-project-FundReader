# -*- coding: utf-8 -*-
"""
fund_plot.py
============
绘图模块：把历史净值和基金经理任职表现画成图，保存到本地 output/ 目录。

两张图：
1. plot_nav_history()      : 基金历史净值走势 + 回撤曲线
2. plot_manager_tenure()   : 基金经理上任前后净值对比 + 同类样本平均

所有图都只展示已经发生的历史数据，并在图内印上风险提示。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import matplotlib

# 必须在 pyplot 之前指定无界面后端，命令行程序才能稳定保存图片
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

from fund_fetch import OUTPUT_DIR, FundReaderError
from fund_interpret import fmt_date, fmt_pct


class PlotError(FundReaderError):
    """绘图失败（数据太少、文件写入失败等）。"""


# ---------------------------------------------------------------------------
# 中文字体与通用样式
# ---------------------------------------------------------------------------

def setup_chinese_font() -> str:
    """
    matplotlib 默认字体不支持中文，这里自动挑选系统里可用的中文字体。
    返回选中的字体名（找不到时返回空字符串，图里中文可能显示为方框）。
    """
    preferred = [
        "Microsoft YaHei",
        "SimHei",
        "DengXian",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "PingFang SC",
        "Arial Unicode MS",
    ]
    try:
        available = {font.name for font in font_manager.fontManager.ttflist}
    except Exception:
        available = set()

    chosen = ""
    for name in preferred:
        if name in available:
            chosen = name
            break

    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen] + list(plt.rcParams.get("font.sans-serif", []))
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#e6e6e6"
    plt.rcParams["grid.linewidth"] = 0.6
    return chosen


def _safe_filename(text: str) -> str:
    """去掉 Windows 文件名里不允许的字符。"""
    text = re.sub(r'[\\/:*?"<>|]+', "_", str(text)).strip()
    return text[:40] or "基金"


def _save_fig(fig, path: Path) -> Path:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    except Exception as exc:
        raise PlotError(f"图片保存失败：{exc}") from exc
    finally:
        plt.close(fig)
    return path


def _format_date_axis(ax) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.tick_params(axis="x", labelsize=9, colors="#555555")
    ax.tick_params(axis="y", labelsize=9, colors="#555555")


def _add_disclaimer(fig, extra: str = "") -> None:
    """在图底部加红色风险提示。"""
    text = "数据来源：tushare 公开基金数据 ｜ 历史数据不代表未来表现，不构成任何投资建议"
    if extra:
        text = f"{text}\n{extra}"
    fig.text(
        0.5,
        0.012,
        text,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#c00000",
        fontweight="bold",
        linespacing=1.5,
    )


# ---------------------------------------------------------------------------
# 图 1：基金历史净值走势
# ---------------------------------------------------------------------------

def plot_nav_history(result: dict, outdir: Optional[Path] = None) -> Path:
    """
    绘制基金历史净值走势图（上方净值，下方回撤），保存为 PNG。

    参数 result 来自 fund_interpret.analyze_fund()。
    返回图片路径。
    """
    setup_chinese_font()

    basic = result["basic"]
    nav = result.get("nav")
    stats = result.get("stats") or {}

    if nav is None or nav.empty or len(nav) < 2:
        raise PlotError("净值数据太少，画不出走势图。")

    outdir = Path(outdir or OUTPUT_DIR)
    dates = pd.to_datetime(nav["date"])
    values = pd.to_numeric(nav["value"], errors="coerce")
    valid = values.notna()
    dates = dates[valid]
    values = values[valid]
    if len(values) < 2:
        raise PlotError("净值数据里没有足够的有效数值，画不出走势图。")

    value_label = "净值"
    if "value_label" in nav.columns:
        mode_values = nav["value_label"].mode()
        if not mode_values.empty:
            value_label = str(mode_values.iloc[0])

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(11.5, 7.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1]},
    )
    fig.subplots_adjust(hspace=0.12, bottom=0.16, top=0.88)

    # 上方：净值曲线
    ax1.plot(dates, values, color="#1f6feb", linewidth=1.6, label=value_label)
    baseline = float(values.min()) * 0.98
    ax1.fill_between(dates, values, baseline, color="#1f6feb", alpha=0.08)
    ax1.set_ylabel(f"{value_label}（元）", fontsize=11)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # 最新点标注
    ax1.scatter([dates.iloc[-1]], [values.iloc[-1]], color="#1f6feb", s=28, zorder=5)
    ax1.annotate(
        f"{values.iloc[-1]:.4f}",
        xy=(dates.iloc[-1], values.iloc[-1]),
        xytext=(-8, 10),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color="#1f6feb",
        fontweight="bold",
    )

    # 统计信息小卡片
    cumulative = stats.get("cumulative")
    annualized = stats.get("annualized")
    mdd = (stats.get("max_drawdown") or {}).get("value")
    info = (
        f"区间涨跌：{fmt_pct(cumulative, 2, signed=True)}\n"
        f"年化折算：{fmt_pct(annualized, 2, signed=True)}\n"
        f"最大回撤：{fmt_pct(mdd, 2)}"
    )
    ax1.text(
        0.015,
        0.97,
        info,
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#dddddd", alpha=0.92),
    )

    ax1.set_title(
        f"{basic['name']}（{result['ts_code']}）历史净值走势\n"
        f"{fmt_date(dates.iloc[0])} 至 {fmt_date(dates.iloc[-1])} ｜ 只回顾历史，不预测未来",
        fontsize=15,
        fontweight="bold",
        color="#222222",
        pad=14,
    )

    # 下方：回撤曲线
    values_array = values.to_numpy(dtype=float)
    running_max = np.maximum.accumulate(values_array)
    drawdown = values_array / running_max - 1.0

    ax2.fill_between(dates, drawdown * 100, 0, color="#d9534f", alpha=0.28)
    ax2.plot(dates, drawdown * 100, color="#c0392b", linewidth=0.9)
    ax2.axhline(0, color="#999999", linewidth=0.8)
    ax2.set_ylabel("回撤（%）", fontsize=10)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    _format_date_axis(ax2)

    # 标出历史最大回撤的位置
    if mdd is not None and mdd < 0:
        trough_idx = int(np.argmin(drawdown))
        ax2.scatter(
            [dates.iloc[trough_idx]],
            [drawdown[trough_idx] * 100],
            color="#c0392b",
            s=30,
            zorder=5,
        )
        ax2.annotate(
            f"最大回撤 {fmt_pct(mdd, 1)}",
            xy=(dates.iloc[trough_idx], drawdown[trough_idx] * 100),
            xytext=(10, -18),
            textcoords="offset points",
            fontsize=9,
            color="#c0392b",
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="#c0392b", lw=0.8),
        )

    _add_disclaimer(
        fig,
        "最大回撤 = 从历史高点跌到最低点的幅度；年化 = 把整段历史平均摊到每年，二者都不是未来收益承诺。",
    )

    filename = f"{result['ts_code'].replace('.', '_')}_{_safe_filename(basic['name'])}_净值走势.png"
    return _save_fig(fig, outdir / filename)


# ---------------------------------------------------------------------------
# 图 2：基金经理上任前后净值对比
# ---------------------------------------------------------------------------

def plot_manager_tenure(report, outdir: Optional[Path] = None) -> Path:
    """
    绘制基金经理上任前后的归一化净值走势：
    - 蓝线：本基金（以上任日净值 = 1.00 归一化）
    - 灰虚线：同类样本平均（若有）
    - 橙色竖线：上任日期；橙色浅色区域：任职区间

    参数 report 来自 fund_manager_analysis.analyze_manager()。
    """
    setup_chinese_font()

    window = getattr(report, "window", None)
    if window is None or window.empty or "norm" not in window.columns:
        raise PlotError("没有足够的净值数据，画不出基金经理任职对比图。")

    window = window.dropna(subset=["norm"]).copy()
    if len(window) < 2:
        raise PlotError("上任前后的净值点太少，画不出对比图。")

    outdir = Path(outdir or OUTPUT_DIR)
    basic = report.basic
    begin = pd.Timestamp(report.primary_begin)
    end = pd.Timestamp(window["date"].max())
    dates = pd.to_datetime(window["date"])
    norm = pd.to_numeric(window["norm"], errors="coerce")

    fig, ax = plt.subplots(figsize=(11.8, 7.0))
    fig.subplots_adjust(bottom=0.19, top=0.86, left=0.075, right=0.97)

    # 上任前的背景区域
    if dates.min() < begin:
        ax.axvspan(dates.min(), begin, color="#999999", alpha=0.06, zorder=0)

    # 同类样本平均
    peer_avg = getattr(report, "peer_avg", None)
    if peer_avg is not None and len(peer_avg) > 1:
        ax.plot(
            pd.to_datetime(peer_avg.index),
            peer_avg.values,
            color="#7f8c8d",
            linewidth=1.4,
            linestyle="--",
            label=f"同类样本平均（{report.peer_count} 只，上任日=1）",
            zorder=2,
        )

    # 本基金净值
    ax.plot(
        dates,
        norm,
        color="#1f6feb",
        linewidth=2.0,
        label="本基金（上任日=1）",
        zorder=3,
    )

    # 任职区间
    ax.axvspan(begin, end, color="#f59e0b", alpha=0.07, zorder=0)
    ax.axvline(begin, color="#d97706", linewidth=1.8, linestyle="--", zorder=4)
    ax.axhline(1.0, color="#999999", linewidth=0.9, linestyle=":", zorder=1)

    ax.set_ylabel("净值（以上任日 = 1.00 归一化）", fontsize=11)
    ax.grid(True, axis="y", linestyle="--", alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _format_date_axis(ax)

    # 上方文字标注“上任前 / 上任后”
    y_min, y_max = ax.get_ylim()
    span = y_max - y_min if y_max > y_min else 1.0
    label_y = y_min + span * 0.06
    if dates.min() < begin:
        ax.text(
            dates.min() + (begin - dates.min()) / 2,
            label_y,
            "上任前（仅作背景）",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#777777",
        )
    ax.text(
        begin + (end - begin) * 0.015,
        label_y,
        "上任后（任职区间）",
        ha="left",
        va="bottom",
        fontsize=9,
        color="#b45309",
        fontweight="bold",
    )
    ax.text(
        begin,
        y_max - span * 0.005,
        " 上任日 ",
        ha="left",
        va="top",
        fontsize=9,
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#d97706", edgecolor="none"),
    )

    # 末端收益标注
    last_norm = float(norm.iloc[-1])
    ax.scatter([dates.iloc[-1]], [last_norm], color="#1f6feb", s=32, zorder=5)
    ax.annotate(
        f"任期累计 {last_norm - 1:+.1%}",
        xy=(dates.iloc[-1], last_norm),
        xytext=(-10, 16),
        textcoords="offset points",
        ha="right",
        fontsize=10,
        color="#1f6feb",
        fontweight="bold",
    )

    # 标题与说明
    manager_name = report.primary_name
    ax.set_title(
        f"{manager_name} 管理 {basic['name']}（{report.ts_code}）上任前后净值对比\n"
        f"上任日：{fmt_date(begin)} ｜ 仅基于历史净值做归一化，不预测未来",
        fontsize=15,
        fontweight="bold",
        color="#222222",
        pad=14,
    )

    ax.legend(loc="upper left", frameon=True, fontsize=9.5, framealpha=0.9)
    ax.margins(x=0.01)

    _add_disclaimer(
        fig,
        "纵轴为归一化净值（上任日=1.00），上任前曲线只用于观察起点位置；"
        "基金经理过往历史业绩不代表未来表现，不构成投资建议。",
    )

    filename = (
        f"{report.ts_code.replace('.', '_')}_{_safe_filename(basic['name'])}"
        f"_{_safe_filename(manager_name)}任期净值对比.png"
    )
    return _save_fig(fig, outdir / filename)
