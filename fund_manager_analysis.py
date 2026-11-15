# -*- coding: utf-8 -*-
"""
fund_manager_analysis.py
========================
基金经理历史业绩分析模块（本项目的扩展难点部分）。

本模块只做三件事：
1. 统计基金经理在“本基金”任职期间的 historical 表现（年化、最大回撤、任职时长）；
2. 统计该经理管理过的其他基金的历史片段，并做小样本同类对比；
3. 把所有指标翻译成大白话。

严禁：预测未来收益、推荐买卖、输出投资建议。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from fund_fetch import FundFetcher, FundReaderError
from fund_interpret import (
    fmt_date,
    fmt_pct,
    fmt_years,
    period_stats,
    print_table,
    print_title,
    red_text,
    section,
    slice_nav,
    truncate_display,
)


class ManagerDataMissingError(FundReaderError):
    """基金经理信息缺失，无法完成分析。"""


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class ManagerSegment:
    """一段任职记录 + 该区间的历史表现。"""
    ts_code: str
    fund_name: str
    begin: pd.Timestamp
    end: Optional[pd.Timestamp]
    days: int
    cumulative: Optional[float] = None
    annualized: Optional[float] = None
    max_drawdown: Optional[float] = None
    is_current_fund: bool = False
    note: str = ""


@dataclass
class ManagerReport:
    """基金经理分析的完整结果，既用于文字报告，也用于画图。"""
    ts_code: str
    basic: dict
    nav: pd.DataFrame
    primary_name: str
    primary_begin: pd.Timestamp
    primary_end: pd.Timestamp
    primary_edu: str
    primary_resume: str
    is_fallback: bool
    current_infos: List[dict] = field(default_factory=list)
    tenure: dict = field(default_factory=dict)
    history: List[ManagerSegment] = field(default_factory=list)
    history_total: int = 0
    hist_valid: int = 0
    hist_positive: int = 0
    hist_mean_ann: Optional[float] = None
    hist_min_ann: Optional[float] = None
    hist_max_ann: Optional[float] = None
    hist_median_mdd: Optional[float] = None
    peer_count: int = 0
    peer_ann_median: Optional[float] = None
    peer_mdd_median: Optional[float] = None
    peer_rank: Optional[int] = None
    peer_percentile: Optional[float] = None
    peer_deeper_count: int = 0
    peer_reason: str = ""
    window: Optional[pd.DataFrame] = None
    peer_avg: Optional[pd.Series] = None


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _annualized_mean(values: Sequence[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not nums:
        return None
    return float(np.mean(nums))


def _median(values: Sequence[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not nums:
        return None
    return float(np.median(nums))


def _safe_text(value) -> str:
    """把可能为 None / NaN / pd.NA 的字段安全转成字符串。"""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _prepare_chart_window(
    nav: pd.DataFrame,
    begin: pd.Timestamp,
) -> pd.DataFrame:
    """截取“上任前一年 + 上任后至今”的净值，并以任职起始日净值为 1 做归一化。"""
    window_start = max(begin - pd.Timedelta(days=365), nav["date"].min())
    window = nav[nav["date"] >= window_start].copy()
    if window.empty:
        return window

    at_begin = window[window["date"] >= begin]
    if at_begin.empty:
        base_value = float(window["value"].iloc[0])
    else:
        base_value = float(at_begin["value"].iloc[0])

    if base_value <= 0:
        window["norm"] = np.nan
    else:
        window["norm"] = window["value"] / base_value
    return window.reset_index(drop=True)


def _align_peer_average(
    peer_navs: Sequence[pd.DataFrame],
    target_dates: pd.Series,
    begin: pd.Timestamp,
    end: pd.Timestamp,
) -> Optional[pd.Series]:
    """
    把多只同类基金在“上任日”归一化成 1，再按日期对齐求平均。
    至少要有 2 只样本才返回，否则返回 None。
    """
    if not peer_navs or len(target_dates) == 0:
        return None

    frames = []
    for peer in peer_navs:
        try:
            series = peer.set_index("date")["value"].sort_index()
            series = series[~series.index.duplicated(keep="last")]
            series = series[(series.index >= begin) & (series.index <= end)]
            if series.empty:
                continue
            first_date = series.index[0]
            if (first_date - begin).days > 45:
                continue
            base = float(series.iloc[0])
            if base <= 0:
                continue
            frames.append(series / base)
        except Exception:
            continue

    if len(frames) < 2:
        return None

    combined = pd.concat(frames, axis=1).sort_index().ffill()
    average = combined.mean(axis=1, skipna=True).dropna()
    if average.empty:
        return None

    # 对齐到本基金的净值日期，方便画在同一张图里
    target_index = pd.DatetimeIndex(pd.to_datetime(target_dates)).drop_duplicates().sort_values()
    try:
        average = average.reindex(target_index, method="ffill")
    except Exception:
        pass
    return average


# ---------------------------------------------------------------------------
# 主分析
# ---------------------------------------------------------------------------

def analyze_manager(
    fetcher: FundFetcher,
    raw_code: str,
    peer_limit: int = 8,
    history_limit: int = 5,
) -> ManagerReport:
    """
    分析一只基金的现任基金经理（若已无在任经理，则分析最后一位）。

    参数：
      peer_limit   : 同类基金对比的样本数上限（小样本，避免接口调用过多）
      history_limit: 展示该经理最近几段任职记录
    """
    ts_code, basic = fetcher.resolve_fund(raw_code)
    nav = fetcher.get_nav(ts_code)
    managers = fetcher.get_managers(ts_code)

    if managers is None or managers.empty:
        raise ManagerDataMissingError(
            f"没有查到 {ts_code} 的基金经理任职数据。\n"
            "可能原因：该基金刚成立、已清盘，或 tushare 的基金经理数据尚未更新。\n"
            "你可以改用【基金大白话解读】查看它的净值和风险。"
        )

    current = managers[managers["is_current"] == True].copy()  # noqa: E712
    is_fallback = False

    if current.empty:
        # 基金可能已终止：退而求其次，分析最后一位有记录的经理
        historical = managers.dropna(subset=["begin_dt"]).sort_values("begin_dt", ascending=False)
        if historical.empty:
            raise ManagerDataMissingError(f"{ts_code} 的基金经理记录里没有有效的任职起始日期。")
        primary_row = historical.iloc[0]
        is_fallback = True
    else:
        primary_row = current.sort_values("begin_dt", ascending=True).iloc[0]

    primary_name = str(primary_row["name"]).strip()
    primary_begin = pd.Timestamp(primary_row["begin_dt"])
    if pd.isna(primary_begin):
        raise ManagerDataMissingError(f"{primary_name} 的任职起始日期缺失，无法统计任职区间。")

    primary_end_raw = primary_row.get("end_dt")
    if primary_end_raw is None or pd.isna(primary_end_raw):
        primary_end = pd.Timestamp(nav["date"].max())
    else:
        primary_end = min(pd.Timestamp(primary_end_raw), pd.Timestamp(nav["date"].max()))

    if primary_end <= primary_begin:
        raise ManagerDataMissingError(
            f"{primary_name} 的任职区间异常（起始 {fmt_date(primary_begin)}，"
            f"结束 {fmt_date(primary_end)}），无法统计。"
        )

    tenure_nav = slice_nav(nav, primary_begin, primary_end)
    tenure = period_stats(tenure_nav)
    if tenure is None:
        raise ManagerDataMissingError(
            f"{primary_name} 任职期间（{fmt_date(primary_begin)} 至 {fmt_date(primary_end)}）"
            "的净值数据太少，无法计算年化和回撤。"
        )

    # ------------------------------------------------------------------
    # 现任经理（可能不止一位）各自的任职成绩
    # ------------------------------------------------------------------
    current_infos: List[dict] = []
    for _, row in current.iterrows():
        name = str(row["name"]).strip()
        begin = pd.Timestamp(row["begin_dt"])
        if pd.isna(begin):
            continue
        end_raw = row.get("end_dt")
        end = pd.Timestamp(nav["date"].max()) if (end_raw is None or pd.isna(end_raw)) else pd.Timestamp(end_raw)
        sub = slice_nav(nav, begin, end)
        st = period_stats(sub)
        current_infos.append(
            {
                "name": name,
                "begin": begin,
                "end": end,
                "days": st["days"] if st else int((end - begin).days),
                "cumulative": st["cumulative"] if st else None,
                "annualized": st["annualized"] if st else None,
                "max_drawdown": st["max_drawdown"]["value"] if st and st.get("max_drawdown") else None,
            }
        )

    # 如果主经理不在 current_infos（历史兜底情况），补一条
    if is_fallback and not any(info["name"] == primary_name for info in current_infos):
        current_infos.append(
            {
                "name": primary_name,
                "begin": primary_begin,
                "end": primary_end,
                "days": tenure["days"],
                "cumulative": tenure["cumulative"],
                "annualized": tenure["annualized"],
                "max_drawdown": tenure["max_drawdown"]["value"] if tenure.get("max_drawdown") else None,
            }
        )

    # ------------------------------------------------------------------
    # 这位经理管过的其他基金
    # ------------------------------------------------------------------
    try:
        history_df = fetcher.get_manager_history(primary_name)
    except Exception:
        history_df = pd.DataFrame()

    history_total = 0
    if history_df is not None and not history_df.empty:
        history_full = history_df.dropna(subset=["begin_dt"]).sort_values("begin_dt", ascending=False)
        history_total = int(history_full["ts_code"].nunique())
        selected = history_full.head(history_limit).copy()
        # 本基金可能不是最近 5 段之一（老经理管过很多新基金），
        # 为保证列表里能看到本基金，用本基金替换掉最旧的一条。
        current_rows = history_full[history_full["ts_code"].astype(str) == ts_code]
        if not current_rows.empty and ts_code not in set(selected["ts_code"].astype(str)):
            selected = pd.concat(
                [current_rows.head(1), selected.head(max(history_limit - 1, 1))],
                ignore_index=True,
            )
            selected = selected.drop_duplicates(subset=["ts_code", "begin_date"], keep="first")
        history_df = selected.sort_values("begin_dt", ascending=False)
        codes = history_df["ts_code"].astype(str).tolist()
        name_map = fetcher.get_fund_names(codes)
    else:
        history_df = pd.DataFrame()
        name_map = {}

    history: List[ManagerSegment] = []
    for _, row in history_df.iterrows():
        code = str(row["ts_code"])
        fund_name = name_map.get(code, code)
        begin = pd.Timestamp(row["begin_dt"])
        end_raw = row.get("end_dt")
        end = None if (end_raw is None or pd.isna(end_raw)) else pd.Timestamp(end_raw)

        try:
            fund_nav = fetcher.get_nav(code)
        except Exception as exc:
            history.append(
                ManagerSegment(
                    ts_code=code,
                    fund_name=fund_name,
                    begin=begin,
                    end=end,
                    days=max(0, int(((end or pd.Timestamp.today()) - begin).days)),
                    is_current_fund=(code == ts_code),
                    note=f"没取到净值：{str(exc)[:40]}",
                )
            )
            continue

        effective_end = end if end is not None else pd.Timestamp(fund_nav["date"].max())
        sub = slice_nav(fund_nav, begin, effective_end)
        st = period_stats(sub)

        history.append(
            ManagerSegment(
                ts_code=code,
                fund_name=fund_name,
                begin=begin,
                end=end,
                days=st["days"] if st else max(0, int((effective_end - begin).days)),
                cumulative=st["cumulative"] if st else None,
                annualized=st["annualized"] if st else None,
                max_drawdown=st["max_drawdown"]["value"] if st and st.get("max_drawdown") else None,
                is_current_fund=(code == ts_code),
                note="" if st else "净值数据不足",
            )
        )

    history.sort(key=lambda item: item.begin, reverse=True)

    # 只把“至少 90 天”的片段纳入平均，避免十几天的小片段把平均值带偏
    valid_history = [seg for seg in history if seg.annualized is not None and seg.days >= 90]
    hist_mean_ann = _annualized_mean([seg.annualized for seg in valid_history])
    hist_min_ann = min((seg.annualized for seg in valid_history), default=None)
    hist_max_ann = max((seg.annualized for seg in valid_history), default=None)
    hist_positive = sum(1 for seg in valid_history if (seg.annualized or 0) > 0)
    hist_median_mdd = _median([seg.max_drawdown for seg in valid_history])

    # ------------------------------------------------------------------
    # 同类基金小样本对比
    # ------------------------------------------------------------------
    peer_reason = ""
    peer_count = 0
    peer_ann_median = None
    peer_mdd_median = None
    peer_rank = None
    peer_percentile = None
    peer_deeper_count = 0
    peer_navs: List[pd.DataFrame] = []

    fund_type = basic.get("fund_type", "")
    if tenure["days"] < 60:
        peer_reason = "任职时间不足 60 天，年化数字会被短期行情严重放大，本次不做同类对比。"
    elif not fund_type:
        peer_reason = "数据源没有返回这只基金的分类，无法找到同类基金。"
    else:
        before_date = primary_begin.strftime("%Y%m%d")
        candidates = fetcher.get_peer_candidates(
            fund_type=fund_type,
            exclude_code=ts_code,
            before_date=before_date,
            pool_size=peer_limit * 3,
        )

        if not candidates:
            peer_reason = "没有找到足够的同类基金样本（可能是权限或数据问题），已跳过同类对比。"
        else:
            print(f"  正在采集同类基金历史数据（最多 {peer_limit} 只，小样本对比）...", end="", flush=True)
            peer_anns: List[float] = []
            peer_mdds: List[float] = []
            attempts = 0
            for code in candidates:
                if len(peer_anns) >= peer_limit or attempts >= peer_limit * 2:
                    break
                attempts += 1
                try:
                    peer_nav = fetcher.get_nav(code)
                except Exception:
                    print(".", end="", flush=True)
                    continue

                sub = slice_nav(peer_nav, primary_begin, tenure["end_date"])
                if len(sub) < 2:
                    print(".", end="", flush=True)
                    continue

                # 要求样本在任职起始日附近就有净值，否则区间不可比
                lag_days = (sub["date"].iloc[0] - primary_begin).days
                if lag_days > 45:
                    print(".", end="", flush=True)
                    continue

                st = period_stats(sub)
                if st is None or st["annualized"] is None or st["days"] < 60:
                    print(".", end="", flush=True)
                    continue

                peer_anns.append(float(st["annualized"]))
                if st.get("max_drawdown"):
                    peer_mdds.append(float(st["max_drawdown"]["value"]))
                peer_navs.append(peer_nav)
                print(".", end="", flush=True)

            print(" 完成")

            peer_count = len(peer_anns)
            if peer_count >= 3:
                peer_ann_median = _median(peer_anns)
                peer_mdd_median = _median(peer_mdds)
                fund_ann = float(tenure["annualized"]) if tenure.get("annualized") is not None else None
                if fund_ann is not None:
                    peer_rank = 1 + sum(1 for value in peer_anns if value > fund_ann)
                    peer_percentile = sum(1 for value in peer_anns if value < fund_ann) / peer_count
                if tenure.get("max_drawdown"):
                    fund_mdd = float(tenure["max_drawdown"]["value"])
                    peer_deeper_count = sum(1 for value in peer_mdds if value < fund_mdd)
            else:
                peer_reason = f"只找到 {peer_count} 只有效同类样本，数量太少，本次不做排名。"

    # ------------------------------------------------------------------
    # 画图数据：上任前一年 + 上任后至今，归一化到上任日 = 1
    # ------------------------------------------------------------------
    window = _prepare_chart_window(nav, primary_begin)
    peer_avg = None
    if peer_navs and window is not None and not window.empty:
        peer_avg = _align_peer_average(
            peer_navs=peer_navs,
            target_dates=window["date"],
            begin=primary_begin,
            end=pd.Timestamp(nav["date"].max()),
        )

    return ManagerReport(
        ts_code=ts_code,
        basic=basic,
        nav=nav,
        primary_name=primary_name,
        primary_begin=primary_begin,
        primary_end=primary_end,
        primary_edu=_safe_text(primary_row.get("edu")),
        primary_resume=_safe_text(primary_row.get("resume")),
        is_fallback=is_fallback,
        current_infos=current_infos,
        tenure=tenure,
        history=history,
        history_total=history_total,
        hist_valid=len(valid_history),
        hist_positive=hist_positive,
        hist_mean_ann=hist_mean_ann,
        hist_min_ann=hist_min_ann,
        hist_max_ann=hist_max_ann,
        hist_median_mdd=hist_median_mdd,
        peer_count=peer_count,
        peer_ann_median=peer_ann_median,
        peer_mdd_median=peer_mdd_median,
        peer_rank=peer_rank,
        peer_percentile=peer_percentile,
        peer_deeper_count=peer_deeper_count,
        peer_reason=peer_reason,
        window=window,
        peer_avg=peer_avg,
    )


# ---------------------------------------------------------------------------
# 文字报告
# ---------------------------------------------------------------------------

def _render_manager_header(report: ManagerReport) -> None:
    basic = report.basic
    print_title(
        f"基金经理历史业绩分析：{basic['name']}（{report.ts_code}）\n"
        f"数据截至 {fmt_date(report.tenure['end_date'])} ｜ 只做历史统计，不预测未来"
    )
    print(red_text("提醒：本页全部是已经发生的历史数据，不代表未来表现，不构成投资建议。"))

    if report.is_fallback:
        print("\n注意：这只基金目前没有查到在任基金经理（可能已终止），下面分析的是最后一位有记录的经理。")


def _render_current_managers(report: ManagerReport) -> None:
    section("现任基金经理是谁")
    if report.is_fallback:
        print(f"最后一位有记录的经理：{report.primary_name}")
    else:
        names = "、".join(info["name"] for info in report.current_infos)
        if len(report.current_infos) > 1:
            print(f"这只基金现在由 {len(report.current_infos)} 位经理共同管理：{names}")
        else:
            print(f"现任基金经理：{names}")

    if report.primary_edu:
        print(f"公开资料里的学历：{report.primary_edu}")
    if report.primary_resume:
        resume = report.primary_resume.replace("\n", " ").strip()
        print(f"公开简历节选：{truncate_display(resume, 110)}")

    if len(report.current_infos) > 1:
        rows = []
        for info in report.current_infos:
            rows.append(
                [
                    info["name"],
                    fmt_date(info["begin"]),
                    f"{info['days']} 天",
                    fmt_pct(info["annualized"], 2, signed=True),
                    fmt_pct(info["max_drawdown"], 2),
                ]
            )
        print()
        print("各位现任经理在本基金上的任职数据（同一天上任则数字相同）：")
        print_table(
            ["姓名", "任职起始", "任职时长", "任期年化", "任期最大回撤"],
            rows,
            ["left", "left", "right", "right", "right"],
        )


def _render_tenure(report: ManagerReport) -> None:
    tenure = report.tenure
    section(f"{report.primary_name} 管这只基金的成绩单")

    print(f"统计区间：{fmt_date(tenure['start_date'])} 至 {fmt_date(tenure['end_date'])}")
    print(f"任职时长：约 {tenure['days']} 天（{fmt_years(tenure['days'])}）")
    print(f"任期累计涨跌：{fmt_pct(tenure['cumulative'], 2, signed=True)}")
    print(f"任期年化：{fmt_pct(tenure['annualized'], 2, signed=True)}")

    if tenure.get("annualized") is None:
        print("  任职时间太短，年化数字参考意义有限，这里主要看累计涨跌。")

    mdd = tenure.get("max_drawdown")
    if mdd and mdd["value"] < 0:
        print(f"任期最大回撤：{fmt_pct(mdd['value'], 2)}")
        left = 10000 * (1 + mdd["value"])
        print(f"  大白话：假设这位经理上任时你投入 1 万元，按历史净值算，"
              f"最难受的时候账户大约只剩 {left:,.0f} 元。")
        print("  这是账面最低点，不是最终结果；但如果正好在低点卖出，就会把浮亏变成真亏。")
        print(f"  这次下跌从 {fmt_date(mdd['peak_date'])} 到 {fmt_date(mdd['trough_date'])}，"
              f"用了约 {mdd['fall_days']} 天。")
        if mdd.get("recover_date") is not None:
            print(f"  之后到 {fmt_date(mdd['recover_date'])} 才重新涨回高点，"
                  f"从低点算用了约 {mdd['recover_days'] / 365.25:.1f} 年。")
        else:
            print("  到最新数据为止，还没有涨回那个高点。")
    else:
        print("任期最大回撤：这段区间没有出现明显回撤（不代表以后不会跌）。")

    if tenure.get("volatility") is not None:
        print(f"任期波动：年化约 {fmt_pct(tenure['volatility'], 1)}，"
              f"按历史波动粗略分档属于【{tenure['volatility_label']}】。")

    if tenure["days"] < 180:
        print()
        print("注意：这位经理在这只基金上的任职时间还不到半年。")
        print("  短期业绩很容易被一波行情放大或拖累，年化数字看看就好，别当成长期能力证明。")

    print()
    mdd_value = (mdd or {}).get("value")
    if mdd_value is None:
        print("大白话总结：这段任职里，基金累计 " +
              f"{fmt_pct(tenure['cumulative'], 1, signed=True)}；中途回撤数据不足，无法判断最难受的幅度。")
    else:
        print("大白话总结：这段任职里，基金整体是" +
              ("上涨的" if tenure["cumulative"] > 0 else "下跌的") +
              f"，累计 {fmt_pct(tenure['cumulative'], 1, signed=True)}；"
              f"但中途最多曾回撤 {fmt_pct(mdd_value, 1)}。")
    print("  年化只是把这段历史平均摊到每年，不代表他以后每年都能做出同样成绩。")


def _render_history(report: ManagerReport) -> None:
    section(f"{report.primary_name} 以前管过哪些基金")
    if not report.history:
        print("公开数据里没有查到这位经理的其他基金任职记录。")
        return

    rows = []
    for seg in report.history:
        end_text = fmt_date(seg.end) if seg.end is not None else "至今"
        period = f"{fmt_date(seg.begin)}~{end_text}"
        if seg.is_current_fund:
            period += "（本基金）"
        if seg.note and seg.annualized is None:
            ann_text = seg.note
            mdd_text = "-"
        else:
            ann_text = fmt_pct(seg.annualized, 2, signed=True)
            mdd_text = fmt_pct(seg.max_drawdown, 2)
        rows.append(
            [
                truncate_display(seg.fund_name, 16),
                seg.ts_code,
                period,
                f"{seg.days} 天",
                ann_text,
                mdd_text,
            ]
        )

    print(f"下表展示最近若干段任职记录（含本基金，按上任时间倒序）：")
    print_table(
        ["基金名称", "代码", "任职区间", "时长", "任期年化", "任期最大回撤"],
        rows,
        ["left", "left", "left", "right", "right", "right"],
    )
    print()
    print("说明：同一只基金的 A 类、C 类份额在 tushare 里是两条记录，")
    print("      所以表格里可能看到名字很像的两行，它们的净值走势基本一样。")
    if report.history_total > len(report.history):
        print(f"      公开记录里一共查到 {report.history_total} 只基金，"
              f"这里只展示最近 {len(report.history)} 段，避免一次抓取太多接口。")

    if report.hist_valid == 0:
        print()
        print("这些任职片段的净值数据不足 90 天，暂时算不出可靠的历史年化。")
        return

    print()
    print(f"在能算出结果的 {report.hist_valid} 段任职里：")
    print(f"  - 有 {report.hist_positive} 段任期为上涨，{report.hist_valid - report.hist_positive} 段为下跌；")
    print(f"  - 任期年化最高 {fmt_pct(report.hist_max_ann, 2, signed=True)}，"
          f"最低 {fmt_pct(report.hist_min_ann, 2, signed=True)}；")
    print(f"  - 简单平均约 {fmt_pct(report.hist_mean_ann, 2, signed=True)}/年。")
    if report.hist_median_mdd is not None:
        print(f"  - 这些片段任期最大回撤的中位数约 {fmt_pct(report.hist_median_mdd, 2)}。")

    print()
    print("大白话：" + (
        "这几段任职恰好都是上涨的，但样本很少、时间段也各不相同，不能说明下一次还会赚。"
        if report.hist_positive == report.hist_valid
        else (
            "这几段任职恰好都是下跌的，说明经理也会遇到很不顺的阶段。"
            if report.hist_positive == 0
            else "历史上有赚有亏才是常态。"
        )
    ))
    print("不同基金的类型、规模、上任时市场位置都不一样，")
    print("这些数字不能简单相加，更不能拿某一段的高收益去推断下一只基金。")
    print("另外，短期任职的年化数字会被放大，看的时候要结合任职时长一起看。")


def _render_peer_comparison(report: ManagerReport) -> None:
    section("和同类基金比一比（小样本参考）")

    if not report.peer_reason and report.peer_count >= 3:
        fund_ann = report.tenure.get("annualized")
        fund_mdd = (report.tenure.get("max_drawdown") or {}).get("value")
        fund_type = report.basic.get("fund_type", "同类")

        print(f"我们随机抽取了 {report.peer_count} 只同类型（{fund_type}）基金，"
              f"在相同时间段内做同样的历史计算。")
        print(f"对比区间：{fmt_date(report.tenure['start_date'])} 至 {fmt_date(report.tenure['end_date'])}")
        print()
        print(f"- 本基金任期年化：{fmt_pct(fund_ann, 2, signed=True)}")
        print(f"- 同类样本任期年化中位数：{fmt_pct(report.peer_ann_median, 2, signed=True)}")
        if report.peer_rank is not None:
            print(f"- 在 {report.peer_count} 只样本中约排第 {report.peer_rank} 位（从高到低）")
        print()
        print(f"- 本基金任期最大回撤：{fmt_pct(fund_mdd, 2)}")
        print(f"- 同类样本任期最大回撤中位数：{fmt_pct(report.peer_mdd_median, 2)}")
        print(f"- 样本里有 {report.peer_deeper_count} 只基金跌得比它更狠")

        print()
        if fund_ann is not None and report.peer_ann_median is not None:
            if fund_ann > report.peer_ann_median:
                print("大白话：这段历史里，它的收益比样本中位数高一些，")
                print("  但这只是小样本里的一次比较，不能说明经理以后也能保持。")
            elif fund_ann < report.peer_ann_median:
                print("大白话：这段历史里，它的收益比样本中位数低一些，")
                print("  短期跑输不代表以后一定差，短期跑赢也不代表以后一定好。")
            else:
                print("大白话：这段历史里，它的收益和样本中位数差不多。")

        if fund_mdd is not None and report.peer_mdd_median is not None:
            if fund_mdd < report.peer_mdd_median:
                print("  但跌起来也比一半以上的样本更狠：想拿住它，需要更能承受账户大幅波动。")
            elif fund_mdd > report.peer_mdd_median:
                print("  跌的时候相对样本中位数温和一些，但这不代表下次也会这样。")
            else:
                print("  回撤幅度和样本中位数接近。")

        print()
        print("注意：这只是一个很小的抽样，不是全市场排名，也不能说明这位经理比谁更厉害。")
        print("样本换一批、时间段换一段，结果都可能不一样。")

        invest_type = str(report.basic.get("invest_type", ""))
        fund_name = str(report.basic.get("name", ""))
        if ("指数" in invest_type) or ("ETF" in fund_name.upper()) or ("指数" in fund_name):
            print()
            print("补充：这是一只指数/ETF 类产品，而 tushare 的基金分类比较粗，")
            print("      同类样本里可能混有主动管理基金，投资策略并不完全相同，对比只能粗略看。")
    else:
        print(report.peer_reason or "本次没有足够的同类样本，已跳过对比。")
        print("这不影响前面基于本基金自身历史的统计。")


def _render_explainer() -> None:
    section("这几个数字到底怎么理解")
    print("- 任期年化：把他上任以来的总涨跌，平均摊到每一年。比如两年涨 21%，大约折合 10%/年。")
    print("  但它不是“每年稳稳 10%”，中间可能一年涨 40%、一年跌 20%。")
    print("- 任期最大回撤：从任期内最高点跌到最低点的幅度。它告诉你“最难受的时候有多难受”。")
    print("  回撤不等于最终亏损：如果之后涨回来，只是浮亏；如果低点卖出，才会变成真亏。")
    print("- 任职时长：时间越短，年化越容易被一两波行情放大，参考价值越低。")
    print("- 同类对比：只是拿一小批同类型基金做参照，不是评奖，也不是排名榜。")


def _render_mandatory_warning() -> None:
    """强制醒目提示：红字、加粗、带边框。"""
    print()
    print(red_text("!" * 70))
    print(red_text("  基金经理过往历史业绩不代表未来表现，"))
    print(red_text("  不能用来预判以后收益，不构成任何投资建议。"))
    print(red_text("!" * 70))
    print(red_text("  同一位经理换了基金、换了市场环境，结果可能完全不同；"))
    print(red_text("  历史高收益也可能伴随更大的回撤。请结合自己的风险承受能力，独立做决定。"))
    print(red_text("!" * 70))
    print()


def render_manager_report(report: ManagerReport) -> None:
    """输出基金经理历史业绩分析的大白话报告。"""
    _render_manager_header(report)
    _render_current_managers(report)
    _render_tenure(report)
    _render_history(report)
    _render_peer_comparison(report)
    _render_explainer()
    _render_mandatory_warning()
