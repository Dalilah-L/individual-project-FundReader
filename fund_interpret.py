# -*- coding: utf-8 -*-
"""
fund_interpret.py
=================
本项目的核心模块：把基金里的专业字段翻译成大学生听得懂的大白话。

包括：
- 管理费 / 托管费 -> “每年要从你本金里扣多少钱”
- 最大回撤 -> “历史上最多亏过多少”
- 波动率 -> “平时涨跌大概有多猛”
- 持仓 -> “它的钱主要押在哪几家公司上”
- 风险提示 -> 基金不保本，历史业绩不代表未来

重要原则：
只做客观解读和风险告知，不写“建议买/值得买”，不做任何收益预测。
"""

from __future__ import annotations

import os
import re
import sys
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from fund_fetch import (
    FundFetcher,
    FundReaderError,
    NoNavDataError,
)


# ---------------------------------------------------------------------------
# 终端输出小工具：中文对齐、颜色、标题
# ---------------------------------------------------------------------------

def enable_ansi() -> None:
    """在 Windows 控制台里开启 ANSI 颜色支持（失败也不影响程序）。"""
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            # 0x0004 = ENABLE_VIRTUAL_TERMINAL_PROCESSING
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def _color_enabled() -> bool:
    """只有在真正的终端里才输出颜色，重定向到文件时保持纯文本。"""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FUNDREADER_COLOR") == "1":
        return True
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def red_text(text: str, bold: bool = True) -> str:
    """返回红色文字（终端不支持颜色时原样返回）。"""
    if not _color_enabled():
        return text
    prefix = "\033[1;91m" if bold else "\033[91m"
    return f"{prefix}{text}\033[0m"


def print_title(text: str, width: int = 70) -> None:
    print()
    print("=" * width)
    print(text)
    print("=" * width)


def section(title: str) -> None:
    print()
    print(f"【{title}】")


def display_width(text: str) -> int:
    """计算字符串在终端里的显示宽度（中文算 2 格）。"""
    width = 0
    for char in str(text):
        width += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    return width


def pad_display(text: str, width: int, align: str = "left") -> str:
    text = str(text)
    gap = max(0, width - display_width(text))
    if align == "right":
        return " " * gap + text
    if align == "center":
        left = gap // 2
        return " " * left + text + " " * (gap - left)
    return text + " " * gap


def truncate_display(text: str, max_width: int) -> str:
    """按显示宽度截断长文本，末尾加省略号。"""
    text = str(text)
    if display_width(text) <= max_width:
        return text
    result = ""
    width = 0
    for char in text:
        char_width = 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
        if width + char_width > max_width - 1:
            break
        result += char
        width += char_width
    return result + "…"


def print_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    aligns: Optional[Sequence[str]] = None,
) -> None:
    """打印一张简单的文本表格，自动按中文宽度对齐。"""
    rows = [[str(cell) for cell in row] for row in rows]
    widths = [display_width(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], display_width(cell))

    aligns = list(aligns or ["left"] * len(headers))

    def render(cells: Sequence[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            align = aligns[i] if i < len(aligns) else "left"
            parts.append(pad_display(cell, widths[i], align))
        return "  ".join(parts).rstrip()

    print(render(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(render(row))


# ---------------------------------------------------------------------------
# 数字格式化
# ---------------------------------------------------------------------------

def fmt_pct(value: Optional[float], digits: int = 2, signed: bool = False, na: str = "数据不足") -> str:
    """把 0.1234 这样的小数格式化成 12.34% 或 +12.34%。"""
    if value is None or pd.isna(value):
        return na
    try:
        number = float(value) * 100
    except (TypeError, ValueError):
        return na
    if signed:
        return f"{number:+.{digits}f}%"
    return f"{number:.{digits}f}%"


def fmt_money(value: Optional[float], digits: int = 0) -> str:
    if value is None or pd.isna(value):
        return "数据不足"
    return f"{value:,.{digits}f} 元"


def fmt_date(value) -> str:
    if value is None or pd.isna(value):
        return "未知"
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value)


def fmt_years(days: Optional[int]) -> str:
    if not days or days <= 0:
        return "不到 1 年"
    years = days / 365.25
    return f"{years:.1f} 年"


# ---------------------------------------------------------------------------
# 统计计算
# ---------------------------------------------------------------------------

def max_drawdown(values: Sequence[float], dates: Sequence) -> Optional[dict]:
    """
    计算最大回撤（历史从最高点跌到最低点的最大幅度）。

    返回：
      value        : 回撤幅度，负数，例如 -0.623 表示 -62.3%
      peak_date    : 下跌前的高点日期
      trough_date  : 跌到的最低点日期
      recover_date : 重新涨回高点日期；还没涨回则为 None
      fall_days    : 从高点到低点用了多少天
      recover_days : 从低点涨回高点用了多少天
    """
    values = np.asarray(values, dtype=float)
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    if len(values) < 2:
        return None

    running_max = np.maximum.accumulate(values)
    drawdown = values / running_max - 1.0
    trough_idx = int(np.argmin(drawdown))
    mdd = float(drawdown[trough_idx])

    if mdd >= 0:
        return {
            "value": 0.0,
            "peak_date": dates.iloc[0],
            "trough_date": dates.iloc[0],
            "recover_date": dates.iloc[0],
            "fall_days": 0,
            "recover_days": 0,
            "peak_value": float(values[0]),
            "trough_value": float(values[0]),
        }

    peak_idx = int(np.where(values[: trough_idx + 1] == running_max[trough_idx])[0][-1])
    peak_value = float(values[peak_idx])
    trough_value = float(values[trough_idx])

    recover_idx = None
    if trough_idx < len(values) - 1:
        candidates = np.where(values[trough_idx + 1:] >= peak_value)[0]
        if len(candidates) > 0:
            recover_idx = trough_idx + 1 + int(candidates[0])

    peak_date = dates.iloc[peak_idx]
    trough_date = dates.iloc[trough_idx]
    recover_date = dates.iloc[recover_idx] if recover_idx is not None else None

    return {
        "value": mdd,
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recover_date": recover_date,
        "fall_days": int((trough_date - peak_date).days),
        "recover_days": int((recover_date - trough_date).days) if recover_date is not None else None,
        "peak_value": peak_value,
        "trough_value": trough_value,
    }


def annualized_volatility(values: Sequence[float]) -> Optional[float]:
    """用每日涨跌估算年化波动率（衡量平时波动大小，不是风险等级）。"""
    series = pd.Series(values, dtype="float64").dropna()
    if len(series) < 20:
        return None
    returns = series.pct_change().dropna()
    if len(returns) < 10:
        return None
    std = returns.std()
    if pd.isna(std) or std <= 0:
        return None
    return float(std * np.sqrt(252))


def volatility_label(vol: Optional[float]) -> str:
    """把波动率翻译成中文档位（只是历史波动粗略分档，不是官方风险评级）。"""
    if vol is None:
        return "数据不足"
    if vol < 0.05:
        return "很低"
    if vol < 0.10:
        return "中低"
    if vol < 0.18:
        return "中等"
    if vol < 0.25:
        return "中高"
    return "高"


def slice_nav(
    nav: pd.DataFrame,
    start=None,
    end=None,
) -> pd.DataFrame:
    """按日期截取净值区间，start/end 可以是字符串或 Timestamp。"""
    df = nav.copy()
    if start is not None and not pd.isna(start):
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None and not pd.isna(end):
        df = df[df["date"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


def period_stats(
    nav_slice: pd.DataFrame,
    min_days_for_annualized: int = 30,
    volatility_tail_days: Optional[int] = None,
) -> Optional[dict]:
    """
    计算一段区间的表现：累计涨跌、年化、最大回撤、波动率。

    参数 volatility_tail_days:
        只统计最近 N 天的波动率；不传则用整段区间。
        基金全历史报告里传 3 年，避免 20 年前的剧烈波动主导“当前风险感受”。

    说明：年化只是把这段历史的总涨跌平均摊到每年，不代表未来每年都能这样。
    """
    if nav_slice is None or len(nav_slice) < 2:
        return None

    df = nav_slice.sort_values("date").reset_index(drop=True)
    first_value = float(df["value"].iloc[0])
    last_value = float(df["value"].iloc[-1])
    if first_value <= 0:
        return None

    start_date = df["date"].iloc[0]
    end_date = df["date"].iloc[-1]
    days = int((end_date - start_date).days)
    cumulative = last_value / first_value - 1.0

    annualized = None
    if days >= min_days_for_annualized and (1 + cumulative) > 0:
        annualized = (1 + cumulative) ** (365.25 / days) - 1.0

    vol_df = df
    if volatility_tail_days and volatility_tail_days > 0:
        cutoff = end_date - pd.Timedelta(days=int(volatility_tail_days))
        candidate = df[df["date"] >= cutoff]
        if len(candidate) >= 20:
            vol_df = candidate

    vol = annualized_volatility(vol_df["value"].tolist())

    # 按月统计上涨月份占比：让小白直观感受“是不是经常涨”
    monthly_win = None
    month_count = 0
    try:
        series = df.set_index("date")["value"]
        try:
            monthly = series.resample("ME").last()
        except ValueError:
            monthly = series.resample("M").last()
        monthly_returns = monthly.pct_change().dropna()
        if len(monthly_returns) > 0:
            monthly_win = float((monthly_returns > 0).mean())
            month_count = int(len(monthly_returns))
    except Exception:
        pass

    return {
        "start_date": start_date,
        "end_date": end_date,
        "days": days,
        "years": days / 365.25,
        "first_value": first_value,
        "last_value": last_value,
        "cumulative": cumulative,
        "annualized": annualized,
        "max_drawdown": max_drawdown(df["value"].tolist(), df["date"].tolist()),
        "volatility": vol,
        "volatility_label": volatility_label(vol),
        "volatility_start_date": vol_df["date"].iloc[0] if vol is not None else None,
        "monthly_win": monthly_win,
        "month_count": month_count,
        "value_label": str(df["value_label"].mode().iloc[0]) if "value_label" in df.columns and not df["value_label"].mode().empty else "净值",
    }


# ---------------------------------------------------------------------------
# 基金类型 / 份额类别的大白话
# ---------------------------------------------------------------------------

FUND_TYPE_PLAIN = {
    "股票型": "主要把钱投进股票市场，长期看涨跌幅度通常比较大，遇到行情不好时跌得也会比较明显。",
    "混合型": "一般同时买股票和债券，股票多的时候波动大一些，债券多的时候相对温和一些，但依然可能亏钱。",
    "债券型": "主要买债券，平时波动通常比股票基金小，但并不等于保本，债券价格也会跌，特殊情况下也会亏。",
    "货币型": "主要投短期货币工具，波动通常很小，但收益也会随市场利率变化，同样不承诺保本保收益。",
    "指数型": "跟着某个指数买一篮子股票或债券，指数跌它就跌，属于被动跟踪，不是稳赚。",
    "QDII": "会投资境外市场，除了涨跌风险，还要承担汇率变化和境外市场休市等额外风险。",
    "FOF": "它不直接买股票，而是买其他基金，相当于“基金组合”，仍然会跟着市场涨跌。",
    "REITs": "主要投资基础设施等不动产项目，收益来源和股票债券不完全一样，但也有价格波动风险。",
}


def fund_type_plain(fund_type: str) -> str:
    text = (fund_type or "").strip()
    for key, value in FUND_TYPE_PLAIN.items():
        if key in text:
            return value
    return "这只基金的公开分类不太明确，具体投资范围要看基金合同，但只要是基金就不保本，仍可能亏损。"


def share_class(name: str) -> str:
    """从基金简称里粗略判断 A/C 类份额（只用于讲费用差异）。"""
    name = (name or "").strip().upper()
    if re.search(r"(?:[-（(]|\s)C(?:类)?[)）]?$", name) or name.endswith("C"):
        return "C"
    if re.search(r"(?:[-（(]|\s)A(?:类)?[)）]?$", name) or name.endswith("A"):
        return "A"
    return "未知"


# ---------------------------------------------------------------------------
# 主分析：获取并计算一只基金的全部解读素材
# ---------------------------------------------------------------------------

def analyze_fund(fetcher: FundFetcher, raw_code: str) -> dict:
    """
    获取一只基金的基础信息、净值、统计指标和持仓。

    返回字典，供 render_fund_report() 输出文字、供 fund_plot.py 画图。
    """
    ts_code, basic = fetcher.resolve_fund(raw_code)
    nav = fetcher.get_nav(ts_code)
    stats = period_stats(nav, volatility_tail_days=int(3 * 365.25))
    if stats is None:
        raise NoNavDataError(f"{ts_code} 的净值数据太少，无法做统计分析。")

    holdings = fetcher.get_portfolio(ts_code, top=10)
    stock_names: Dict[str, str] = {}
    if holdings is not None and not holdings.empty and "symbol" in holdings.columns:
        stock_names = fetcher.get_stock_names(holdings["symbol"].astype(str).tolist())

    return {
        "ts_code": ts_code,
        "basic": basic,
        "nav": nav,
        "stats": stats,
        "holdings": holdings,
        "stock_names": stock_names,
    }


# ---------------------------------------------------------------------------
# 文字报告
# ---------------------------------------------------------------------------

def _render_basic(result: dict) -> None:
    basic = result["basic"]
    stats = result["stats"]

    section("这只基金是干嘛的")
    found = fmt_date(pd.to_datetime(basic["found_date"], format="%Y%m%d", errors="coerce"))
    company = basic.get("management") or "基金公司（数据未提供）"
    custodian = basic.get("custodian") or "托管银行（数据未提供）"
    fund_type = basic.get("fund_type") or "未分类"

    print(f"全称：{basic['name']}（{result['ts_code']}）")
    print(f"类型：{fund_type}    成立日期：{found}")
    print(f"基金公司：{company}")
    print(f"托管银行：{custodian}")
    print()
    print("大白话：" + fund_type_plain(fund_type))
    print("它的净值每天会涨涨跌跌，买之前先想清楚：这笔钱如果短期跌了 10%~20%，自己扛不扛得住。")


def _render_fees(basic: dict) -> None:
    section("每年要扣你多少钱")
    m_fee = basic.get("m_fee")
    c_fee = basic.get("c_fee")
    principal = 10000.0

    print(f"先假设你投入 {fmt_money(principal)}、持有一整年，只做费用换算，不代表收益：")

    known_total = 0.0
    known_count = 0

    if m_fee is not None:
        amount = principal * m_fee / 100.0
        known_total += amount
        known_count += 1
        print(f"  - 管理费 {m_fee:.2f}%/年  ≈  每年 {fmt_money(amount)}")
        print("    这是基金公司帮你管钱收的“辛苦费”，每天从基金资产里扣，不用你另外掏钱。")
    else:
        print("  - 管理费：tushare 没有返回这只基金的费率，暂时算不出来。")

    if c_fee is not None:
        amount = principal * c_fee / 100.0
        known_total += amount
        known_count += 1
        print(f"  - 托管费 {c_fee:.2f}%/年  ≈  每年 {fmt_money(amount)}")
        print("    这是托管银行帮你看着这笔钱收的“保管费”，同样每天从净值里扣。")
    else:
        print("  - 托管费：tushare 没有返回这只基金的费率，暂时算不出来。")

    if known_count > 0:
        print(f"  - 已知两项合计  ≈  每年 {fmt_money(known_total)}，"
              f"相当于每 1 万元每年先被扣掉约 {known_total:,.0f} 元。")
    print("重要：这些费用不是另外找你收，而是每天从基金净值里扣掉了，所以你看到的净值已经是扣完费的。")

    # 申购赎回费：tushare 免费接口没有这项数据，只能讲行业常见情况
    print()
    print("买卖时还有两笔费用要注意：")
    print("（tushare 免费接口没有提供这只基金当前的申购/赎回费率，下面是行业常见情况，"
          "不是这只基金的实际费率，最终以购买页面和基金合同为准）")

    cls = share_class(basic.get("name", ""))
    if cls == "C":
        print("  - 申购费：C 类通常不收申购费，但会每天计提“销售服务费”（常见约 0.20%~0.80%/年），"
              "持有一年也会从净值里扣掉。")
    elif cls == "A":
        print("  - 申购费：A 类通常在买入时收，常见第三方平台打 1 折后约 0.10%~0.15%；"
              "如果在银行柜台买，可能明显更贵。")
    else:
        print("  - 申购费：这只基金名称里没有明显的 A/C 标记，常见场外基金约 0.10%~1.50%，"
              "很多平台会打折，具体看购买页面。")

    print("  - 赎回费：多数非货币基金持有不满 7 天会收 1.5% 的“惩罚性赎回费”；"
          "持有越久费率通常越低，部分基金持有 1~2 年以上可降到 0。")
    print("    注意：频繁买卖最容易在这两笔费用上吃亏：还没赚钱，先被手续费啃掉一块。")


def _render_risk(stats: dict) -> None:
    section("历史上最多亏过多少")

    print(f"统计区间：{fmt_date(stats['start_date'])} 至 {fmt_date(stats['end_date'])}，"
          f"共 {fmt_years(stats['days'])}。")
    if "复权" in stats["value_label"]:
        value_note = "已经把分红再投资算回去，更适合看长期收益"
    elif "累计" in stats["value_label"]:
        value_note = "已经把历史分红加回去，但没有考虑分红再投资的时间差"
    else:
        value_note = "用单位净值计算，分红带来的变化没有完全还原"
    print(f"净值口径：{stats['value_label']}（{value_note}）")
    print()
    print(f"- 累计涨跌：{fmt_pct(stats['cumulative'], 2, signed=True)}")
    hypothetical = 10000.0 * (1 + stats["cumulative"])
    print(f"  大白话：如果当时投 1 万元并一直持有、期间不取钱，按历史净值算，"
          f"账面会变成约 {hypothetical:,.0f} 元。")
    print("  注意：这是已经发生的历史数学换算，不是对未来收益的承诺，也不代表你能买在那个起点。")

    if stats.get("annualized") is not None:
        print(f"- 平均到每年：约 {fmt_pct(stats['annualized'], 2, signed=True)}/年")
        print("  这只是把整段历史的总涨跌平均摊到每年，实际过程是大起大落的，不是每年稳稳拿这个数。")
    else:
        print("- 平均到每年：成立时间太短，折算年化容易失真，这里不展示。")

    mdd = stats.get("max_drawdown")
    if mdd and mdd["value"] < 0:
        print(f"- 历史最大回撤：{fmt_pct(mdd['value'], 2)}")
        left = 10000.0 * (1 + mdd["value"])
        print(f"  大白话：从之前的高点跌到最低点，最多曾亏掉 {abs(mdd['value']) * 100:.1f}%。"
              f"如果当时账户里有 1 万元，最惨的时候大约只剩 {left:,.0f} 元（还没算手续费）。")
        print(f"  这次下跌：{fmt_date(mdd['peak_date'])} 的高点 → {fmt_date(mdd['trough_date'])} 的低点，"
              f"跌了约 {mdd['fall_days']} 天。")
        if mdd.get("recover_date") is not None:
            print(f"  之后一直到 {fmt_date(mdd['recover_date'])} 才重新涨回那个高点，"
                  f"从低点算起用了约 {mdd['recover_days'] / 365.25:.1f} 年。")
            print("  也就是说：亏钱可能很快，回本可能要很久，这段“熬”的时间也是真实成本。")
        else:
            print("  到最新数据为止，它还没有涨回那个高点。")
    else:
        print("- 历史最大回撤：统计区间内没有出现明显回撤（不代表以后不会跌）。")

    if stats.get("volatility") is not None:
        print(f"- 波动情况：年化约 {fmt_pct(stats['volatility'], 1)}，"
              f"按历史波动粗略分档属于【{stats['volatility_label']}】。")
        print(f"  统计的是 {fmt_date(stats.get('volatility_start_date'))} 以来的每日涨跌"
              "（成立不足 3 年时用全部历史）。")
        print("  大白话：这个数字越大，说明净值平时上蹿下跳越厉害；它只是描述过去的波动，"
              "不是官方风险评级，也不能用来预测下一步涨跌。")

    if stats.get("monthly_win") is not None:
        print(f"- 按月看：过去 {stats['month_count']} 个完整月份里，约 "
              f"{stats['monthly_win'] * 100:.0f}% 的月份是上涨的。")
        print("  也就是说，就算长期是涨的，也经常会有连续几个月不赚钱甚至亏钱的时候。")


def _render_holdings(result: dict) -> None:
    holdings = result.get("holdings")
    stock_names = result.get("stock_names") or {}

    section("它主要买了什么")
    if holdings is None or holdings.empty:
        print("这次没有取到持仓数据（可能接口权限不足、该基金不披露股票持仓，或数据尚未更新）。")
        print("不影响上面的净值解读，但你就少了一个判断它“押注了什么”的角度。")
        return

    period = str(holdings["end_date"].iloc[0]) if "end_date" in holdings.columns else "未知"
    print(f"最新持仓报告期：{fmt_date(pd.to_datetime(period, format='%Y%m%d', errors='coerce'))}，前十大股票如下：")

    rows = []
    total_ratio = 0.0
    for index, (_, row) in enumerate(holdings.iterrows(), start=1):
        symbol = str(row.get("symbol", ""))
        name = stock_names.get(symbol) or symbol
        ratio = row.get("stk_mkv_ratio")
        try:
            ratio_value = float(ratio)
        except (TypeError, ValueError):
            ratio_value = None
        if ratio_value is not None and not pd.isna(ratio_value):
            total_ratio += ratio_value
            ratio_text = f"{ratio_value:.2f}%"
        else:
            ratio_text = "未披露"
        rows.append([str(index), truncate_display(name, 14), symbol, ratio_text])

    print_table(["序号", "股票简称", "代码", "占基金资产"], rows, ["right", "left", "left", "right"])

    if total_ratio >= 60:
        level = "非常集中"
    elif total_ratio >= 40:
        level = "比较集中"
    elif total_ratio >= 20:
        level = "中等集中"
    else:
        level = "比较分散"

    print()
    print(f"前十大股票合计约占基金资产的 {total_ratio:.1f}%，集中度属于【{level}】。")
    print("大白话：这些就是基金钱袋子里的“重仓”。它们涨，基金跟着涨；它们一起大跌，基金也很难独善其身。")
    print("如果前十大占比很高，说明基金经理比较敢押注，涨跌都会更刺激；占比低则相对分散，但也不代表安全。")


def _render_glossary() -> None:
    section("术语小抄（下次看到这些词不用慌）")
    glossary = [
        ("净值", "基金每一份值多少钱。净值涨=赚钱，净值跌=亏钱。"),
        ("最大回撤", "历史上从最高点跌到最低点，最多亏了多少。数字越负，曾经跌得越狠。"),
        ("年化收益", "把一段时间的总涨跌平均摊到每年。只描述过去，不代表以后每年都能拿到。"),
        ("波动率", "净值平时上蹿下跳的幅度。越大越刺激，也越容易让人拿不住。"),
        ("管理费/托管费", "基金公司和托管银行每天从基金资产里扣的钱，已经体现在净值里。"),
        ("申购费/赎回费", "买入和卖出时可能收的手续费，短期频繁买卖时特别伤收益。"),
    ]
    for term, explanation in glossary:
        print(f"- {pad_display(term, 12)}：{explanation}")


def _render_disclaimer(stats: dict) -> None:
    section("给大学生的风险提醒")
    mdd = stats.get("max_drawdown") or {}
    mdd_text = fmt_pct(mdd.get("value"), 1) if mdd.get("value") is not None else "未知"
    print("1. 基金不保本。不管名字里有没有“稳健”“安心”，都可能亏钱，历史上就真的亏过。")
    print(f"2. 这只基金历史最大回撤是 {mdd_text}。你要先问自己：如果买入后账户出现这种幅度的浮亏，"
          "会不会影响吃饭、交学费、还花呗？")
    print("3. 上面所有数字都是已经发生的历史统计，不是对未来的承诺，也不能用来推算以后能赚多少。")
    print("4. 只用闲钱。生活费、学费、下个月的房租，不适合放进随时可能亏损的基金里。")
    print("5. 本程序只做公开数据的翻译和统计，不构成任何投资建议，也不推荐买入或卖出。")


def render_fund_report(result: dict) -> None:
    """把 analyze_fund() 的结果打印成一份大白话报告。"""
    basic = result["basic"]
    stats = result["stats"]

    print_title(
        f"基金大白话解读：{basic['name']}（{result['ts_code']}）\n"
        f"数据截至 {fmt_date(stats['end_date'])} ｜ 只解读历史，不预测未来"
    )

    _render_basic(result)
    _render_fees(basic)
    _render_risk(stats)
    _render_holdings(result)
    _render_glossary()
    _render_disclaimer(stats)
