# -*- coding: utf-8 -*-
"""
web_app.py
==========
FundReader 网页版（Streamlit）。

启动方式：
    streamlit run web_app.py

与命令行版的关系：
    - 数据获取、指标计算、画图全部复用 fund_fetch / fund_interpret /
      fund_manager_analysis / fund_plot，本文件只负责“把结果摆到网页上”；
    - 本文件不实现任何新的投资指标，也不做任何收益率预测。

合规底线（与命令行版一致）：
    - 只展示已经发生的历史统计；
    - 不出现“建议买/值得买/推荐”等表述；
    - 经理分析页底部固定展示强制风险提示。
"""

from __future__ import annotations

import contextlib
import io
from typing import Optional

import pandas as pd
import streamlit as st

from fund_fetch import (
    DataFetchError,
    FundFetcher,
    FundNotFoundError,
    FundReaderError,
    NoNavDataError,
    TokenError,
    split_fund_code,
)
from fund_interpret import (
    analyze_fund,
    fmt_date,
    fmt_money,
    fmt_pct,
    fmt_years,
    fund_type_plain,
    share_class,
)
from fund_manager_analysis import ManagerDataMissingError, analyze_manager
from fund_plot import plot_manager_tenure, plot_nav_history

# ---------------------------------------------------------------------------
# 页面基础设置（必须是第一个 Streamlit 调用）
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FundReader · 基金大白话解读",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2.2rem; max-width: 1180px;}
      .fr-hero {
          padding: 18px 22px; border-radius: 14px; margin-bottom: 14px;
          background: linear-gradient(135deg, #1f3b73 0%, #2f6fb3 100%); color: #fff;
      }
      .fr-hero h1 {font-size: 1.5rem; margin: 0 0 6px 0; color: #fff;}
      .fr-hero p {margin: 0; font-size: .88rem; opacity: .92; line-height: 1.7;}
      .fr-tip {
          padding: 12px 14px; border-radius: 10px; font-size: .9rem; line-height: 1.75;
          background: #f4f7fb; border: 1px solid #dbe4f0;
      }
      .fr-warn {
          padding: 14px 16px; border-radius: 10px; line-height: 1.8;
          background: #fff5f5; border: 1px solid #f0b7b7; border-left: 6px solid #d7263d;
      }
      .fr-muted {color: #6b7280; font-size: .85rem; line-height: 1.7;}
      div[data-testid="stMetricValue"] {font-size: 1.3rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 缓存：一次会话内复用 tushare 连接和查询结果，避免刷新页面就重复抓数据
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_fetcher() -> FundFetcher:
    """整个会话共用一个 FundFetcher（它内部还有一层接口级内存缓存）。"""
    return FundFetcher()


@st.cache_data(ttl=1800, show_spinner=False)
def load_fund(code: str) -> dict:
    """抓取并计算一只基金的解读数据，同时生成净值图。"""
    # 分析模块里有给命令行用的 print 进度提示，网页版不需要，这里静音掉
    with contextlib.redirect_stdout(io.StringIO()):
        result = analyze_fund(get_fetcher(), code)
        chart = ""
        try:
            chart = str(plot_nav_history(result))
        except Exception:
            chart = ""  # 画图失败不影响文字解读
    return {"result": result, "chart": chart}


@st.cache_data(ttl=1800, show_spinner=False)
def load_manager(code: str, peers: int) -> dict:
    """抓取并计算基金经理历史业绩，同时生成任期对比图。"""
    with contextlib.redirect_stdout(io.StringIO()):
        report = analyze_manager(get_fetcher(), code, peer_limit=int(peers))
        chart = ""
        try:
            chart = str(plot_manager_tenure(report))
        except Exception:
            chart = ""
    return {"report": report, "chart": chart}


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------

def show_error(exc: Exception) -> None:
    """把各类异常翻译成网页上的友好提示，不让页面直接崩掉。"""
    if isinstance(exc, TokenError):
        st.error(
            "**token 有问题**\n\n"
            f"{exc}\n\n"
            "请确认环境变量 `TUSHARE_TOKEN`，或项目目录下的 `.env` 文件里填的是真实 token。"
        )
    elif isinstance(exc, FundNotFoundError):
        st.error(f"**没找到这只基金**\n\n{exc}")
    elif isinstance(exc, ManagerDataMissingError):
        st.warning(f"**基金经理数据不完整**\n\n{exc}")
    elif isinstance(exc, NoNavDataError):
        st.warning(f"**没有足够的净值数据**\n\n{exc}")
    elif isinstance(exc, DataFetchError):
        st.error(f"**数据接口出错**\n\n{exc}")
    elif isinstance(exc, ValueError):
        st.warning(f"**输入有误**\n\n{exc}")
    elif isinstance(exc, FundReaderError):
        st.error(f"**这次没有完成**\n\n{exc}")
    else:
        st.error(f"**程序遇到意外问题**\n\n{exc}")
        st.caption("如需排查细节，可在启动前设置环境变量 FUNDREADER_DEBUG=1。")


def check_code(raw: str) -> Optional[str]:
    """校验基金代码格式，返回清理后的代码；不合法时在页面上提示并返回 None。"""
    code = (raw or "").strip()
    if not code:
        st.warning("请先在左侧输入基金代码，例如 000001。")
        return None
    try:
        split_fund_code(code)
    except ValueError as exc:
        st.warning(str(exc))
        return None
    return code


def metric_row(items: list) -> None:
    """一行等宽指标卡；items 是 (标题, 数值, 说明) 的列表。"""
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        if help_text:
            col.metric(label, value, help=help_text)
        else:
            col.metric(label, value)


# ---------------------------------------------------------------------------
# 页面：基金大白话解读
# ---------------------------------------------------------------------------

def render_basic(result: dict) -> None:
    basic = result["basic"]
    stats = result["stats"]

    st.subheader("这只基金是干嘛的")
    found = fmt_date(pd.to_datetime(basic.get("found_date"), format="%Y%m%d", errors="coerce"))
    cols = st.columns(4)
    cols[0].metric("基金类型", basic.get("fund_type") or "未分类")
    cols[1].metric("成立日期", found)
    cols[2].metric("基金公司", basic.get("management") or "未提供")
    cols[3].metric("托管银行", basic.get("custodian") or "未提供")

    st.markdown(
        f'<div class="fr-tip"><b>大白话：</b>{fund_type_plain(basic.get("fund_type") or "")}<br>'
        "它的净值每天会涨涨跌跌，买之前先想清楚：这笔钱如果短期跌了 10%~20%，自己扛不扛得住。"
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"统计区间：{fmt_date(stats['start_date'])} 至 {fmt_date(stats['end_date'])}，"
        f"共 {fmt_years(stats['days'])}"
    )


def render_chart(chart: str, caption: str) -> None:
    st.subheader("净值走势")
    if chart:
        try:
            st.image(chart, caption=caption)
        except Exception as exc:
            st.info(f"图表读取失败（{exc}），但下面的文字结论不受影响。")
    else:
        st.info("这次没有生成图表（可能数据点太少），但下面的文字结论仍然有效。")


def render_fees(basic: dict) -> None:
    st.subheader("每年要扣你多少钱")
    principal = 10000.0
    m_fee = basic.get("m_fee")
    c_fee = basic.get("c_fee")

    rows = []
    if m_fee is not None:
        rows.append(
            {
                "费用项": "管理费",
                "费率": f"{m_fee:.2f}% / 年",
                "1 万元每年约": f"{principal * m_fee / 100.0:,.0f} 元",
                "这是给谁的": "基金公司帮你管钱收的“辛苦费”，每天从基金资产里扣",
            }
        )
    if c_fee is not None:
        rows.append(
            {
                "费用项": "托管费",
                "费率": f"{c_fee:.2f}% / 年",
                "1 万元每年约": f"{principal * c_fee / 100.0:,.0f} 元",
                "这是给谁的": "托管银行帮你看着这笔钱收的“保管费”，同样每天从净值里扣",
            }
        )

    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        total = principal * ((m_fee or 0) + (c_fee or 0)) / 100.0
        st.caption(
            f"已知两项合计约 {total:,.0f} 元 / 年（按 1 万元测算）。"
            "这些费用不是另外找你收，而是每天从基金净值里扣掉了，你看到的净值已经是扣完费的。"
        )
    else:
        st.info("tushare 没有返回这只基金的费率，管理费和托管费暂时算不出来。")

    with st.expander("买卖时还有两笔费用（点开看）"):
        cls = share_class(basic.get("name", ""))
        if cls == "C":
            purchase = (
                "C 类通常不收申购费，但会每天计提“销售服务费”（常见约 0.20%~0.80%/年），"
                "持有一年也会从净值里扣掉。"
            )
        elif cls == "A":
            purchase = (
                "A 类通常在买入时收，常见第三方平台打 1 折后约 0.10%~0.15%；"
                "如果在银行柜台买，可能明显更贵。"
            )
        else:
            purchase = (
                "这只基金名称里没有明显的 A/C 标记，常见场外基金约 0.10%~1.50%，"
                "很多平台会打折，具体看购买页面。"
            )
        st.markdown(
            f"- **申购费**：{purchase}\n"
            "- **赎回费**：多数非货币基金持有不满 7 天会收 1.5% 的“惩罚性赎回费”；"
            "持有越久费率通常越低，部分基金持有 1~2 年以上可降到 0。\n"
        )
        st.caption(
            "tushare 免费接口没有提供这只基金当前的申购/赎回费率，以上是行业常见情况，"
            "不是这只基金的实际费率，最终以购买页面和基金合同为准。"
        )


def render_risk(stats: dict) -> None:
    st.subheader("历史上最多亏过多少")

    mdd = stats.get("max_drawdown") or {}
    metric_row(
        [
            ("累计涨跌", fmt_pct(stats["cumulative"], 2, signed=True), "整段统计区间的总涨跌"),
            ("平均到每年", fmt_pct(stats.get("annualized"), 2, signed=True), "把总涨跌平均摊到每年，不是每年都这样"),
            ("历史最大回撤", fmt_pct(mdd.get("value"), 2), "从最高点跌到最低点的幅度"),
            ("年化波动率", fmt_pct(stats.get("volatility"), 1), f"粗略分档：{stats.get('volatility_label')}"),
        ]
    )

    if "复权" in stats["value_label"]:
        value_note = "已经把分红再投资算回去，更适合看长期收益"
    elif "累计" in stats["value_label"]:
        value_note = "已经把历史分红加回去，但没有考虑分红再投资的时间差"
    else:
        value_note = "用单位净值计算，分红带来的变化没有完全还原"
    st.caption(f"净值口径：{stats['value_label']}（{value_note}）")

    hypothetical = 10000.0 * (1 + stats["cumulative"])
    lines = [
        f"**累计涨跌 {fmt_pct(stats['cumulative'], 2, signed=True)}**：如果当时投 1 万元并一直持有，"
        f"按历史净值算，账面会变成约 **{hypothetical:,.0f} 元**。",
    ]
    if stats.get("annualized") is not None:
        lines.append(
            f"**平均到每年约 {fmt_pct(stats['annualized'], 2, signed=True)}**：这只是把整段历史的总涨跌"
            "平均摊到每年，实际过程是大起大落的，不是每年稳稳拿这个数。"
        )
    else:
        lines.append("**平均到每年**：成立时间太短，折算年化容易失真，这里不展示。")

    if mdd.get("value") is not None and mdd["value"] < 0:
        left = 10000.0 * (1 + mdd["value"])
        lines.append(
            f"**最大回撤 {fmt_pct(mdd['value'], 2)}**：从 {fmt_date(mdd['peak_date'])} 的高点跌到 "
            f"{fmt_date(mdd['trough_date'])} 的低点，用了约 {mdd['fall_days']} 天。"
            f"如果当时账户里有 1 万元，最惨的时候大约只剩 **{left:,.0f} 元**（还没算手续费）。"
        )
        if mdd.get("recover_date") is not None:
            lines.append(
                f"之后一直到 {fmt_date(mdd['recover_date'])} 才重新涨回那个高点，"
                f"从低点算起用了约 {mdd['recover_days'] / 365.25:.1f} 年。"
                "亏钱可能很快，回本可能要很久，这段“熬”的时间也是真实成本。"
            )
        else:
            lines.append("到最新数据为止，它还没有涨回那个高点。")
    else:
        lines.append("**最大回撤**：统计区间内没有出现明显回撤（不代表以后不会跌）。")

    if stats.get("monthly_win") is not None:
        lines.append(
            f"**按月看**：过去 {stats['month_count']} 个完整月份里，约 {stats['monthly_win'] * 100:.0f}% "
            "的月份是上涨的。就算长期是涨的，也经常会有连续几个月不赚钱甚至亏钱的时候。"
        )

    st.markdown(
        '<div class="fr-tip">' + "<br><br>".join(lines) + "</div>",
        unsafe_allow_html=True,
    )


def render_holdings(result: dict) -> None:
    st.subheader("它主要买了什么")
    holdings = result.get("holdings")
    stock_names = result.get("stock_names") or {}

    if holdings is None or holdings.empty:
        st.info(
            "这次没有取到持仓数据（可能接口权限不足、该基金不披露股票持仓，或数据尚未更新）。"
            "不影响上面的净值解读，但你就少了一个判断它“押注了什么”的角度。"
        )
        return

    period = str(holdings["end_date"].iloc[0]) if "end_date" in holdings.columns else "未知"
    st.caption(
        f"最新持仓报告期：{fmt_date(pd.to_datetime(period, format='%Y%m%d', errors='coerce'))}，前十大股票如下"
    )

    rows = []
    total_ratio = 0.0
    for index, (_, row) in enumerate(holdings.iterrows(), start=1):
        symbol = str(row.get("symbol", ""))
        name = stock_names.get(symbol) or symbol
        try:
            ratio_value = float(row.get("stk_mkv_ratio"))
        except (TypeError, ValueError):
            ratio_value = None
        if ratio_value is not None and not pd.isna(ratio_value):
            total_ratio += ratio_value
            ratio_text = f"{ratio_value:.2f}%"
        else:
            ratio_text = "未披露"
        rows.append({"序号": index, "股票简称": name, "代码": symbol, "占基金资产": ratio_text})

    st.dataframe(pd.DataFrame(rows), hide_index=True)

    if total_ratio >= 60:
        level = "非常集中"
    elif total_ratio >= 40:
        level = "比较集中"
    elif total_ratio >= 20:
        level = "中等集中"
    else:
        level = "比较分散"

    st.markdown(
        f'<div class="fr-tip">前十大股票合计约占基金资产的 <b>{total_ratio:.1f}%</b>，'
        f"集中度属于<b>【{level}】</b>。<br>"
        "这些就是基金钱袋子里的“重仓”：它们涨，基金跟着涨；它们一起大跌，基金也很难独善其身。<br>"
        "前十大占比高，说明基金经理比较敢押注，涨跌都会更刺激；占比低则相对分散，但也不代表安全。"
        "</div>",
        unsafe_allow_html=True,
    )


def render_glossary() -> None:
    with st.expander("术语小抄：下次看到这些词不用慌"):
        glossary = [
            ("净值", "基金每一份值多少钱。净值涨=赚钱，净值跌=亏钱。"),
            ("最大回撤", "历史上从最高点跌到最低点，最多亏了多少。数字越负，曾经跌得越狠。"),
            ("年化收益", "把一段时间的总涨跌平均摊到每年。只描述过去，不代表以后每年都能拿到。"),
            ("波动率", "净值平时上蹿下跳的幅度。越大越刺激，也越容易让人拿不住。"),
            ("管理费/托管费", "基金公司和托管银行每天从基金资产里扣的钱，已经体现在净值里。"),
            ("申购费/赎回费", "买入和卖出时可能收的手续费，短期频繁买卖时特别伤收益。"),
        ]
        st.table(pd.DataFrame(glossary, columns=["名词", "大白话解释"]))


def render_fund_disclaimer(stats: dict) -> None:
    mdd = stats.get("max_drawdown") or {}
    mdd_text = fmt_pct(mdd.get("value"), 1) if mdd.get("value") is not None else "未知"
    st.markdown(
        '<div class="fr-warn">'
        "<b>给大学生的风险提醒</b><br>"
        "1. 基金不保本。不管名字里有没有“稳健”“安心”，都可能亏钱，历史上就真的亏过。<br>"
        f"2. 这只基金历史最大回撤是 <b>{mdd_text}</b>。你要先问自己：如果买入后账户出现这种幅度的浮亏，"
        "会不会影响吃饭、交学费、还花呗？<br>"
        "3. 上面所有数字都是已经发生的历史统计，不是对未来的承诺，也不能用来推算以后能赚多少。<br>"
        "4. 只用闲钱。生活费、学费、下个月的房租，不适合放进随时可能亏损的基金里。<br>"
        "5. 本程序只做公开数据的翻译和统计，不构成任何投资建议，也不推荐买入或卖出。"
        "</div>",
        unsafe_allow_html=True,
    )


def render_fund_tab() -> None:
    st.text_input(
        "基金代码",
        key="code_input",
        max_chars=12,
        placeholder="例如 000001 / 510300 / 002599",
        help="场外基金直接填 6 位数字；场内 ETF 也填 6 位，程序会自动补 .OF / .SH / .SZ 后缀",
    )
    if st.button("开始解读这只基金", type="primary", key="btn_fund"):
        code = check_code(st.session_state.get("code_input", ""))
        if code:
            st.session_state["fund_query"] = code

    code = st.session_state.get("fund_query", "")
    if not code:
        st.info("在上面输入 6 位基金代码，然后点「开始解读这只基金」。")
        return

    with st.spinner(f"正在获取 {code} 的数据并计算…（首次查询需要几秒到几十秒）"):
        try:
            data = load_fund(code)
        except Exception as exc:
            show_error(exc)
            return

    result = data["result"]
    chart = data["chart"]
    basic = result["basic"]
    stats = result["stats"]

    st.markdown(
        f"### {basic['name']} <span class='fr-muted'>（{result['ts_code']}）</span>",
        unsafe_allow_html=True,
    )
    st.caption(f"数据截至 {fmt_date(stats['end_date'])} ｜ 只解读历史，不预测未来")

    st.divider()
    render_basic(result)
    st.divider()
    render_chart(chart, "历史净值走势与回撤（图中数据全部为已经发生的历史）")
    st.divider()
    render_fees(basic)
    st.divider()
    render_risk(stats)
    st.divider()
    render_holdings(result)
    st.divider()
    render_glossary()
    st.divider()
    render_fund_disclaimer(stats)
    st.caption(f"图表已同时保存到本地：{data['chart']}" if data["chart"] else "")


# ---------------------------------------------------------------------------
# 页面：基金经理业绩分析
# ---------------------------------------------------------------------------

def render_manager_header(report) -> None:
    st.markdown(
        f"### {report.primary_name} "
        f"<span class='fr-muted'>（{report.basic.get('name', '')} · {report.ts_code}）</span>",
        unsafe_allow_html=True,
    )
    tenure = report.tenure
    st.caption(
        f"任职区间：{fmt_date(tenure['start_date'])} 至 {fmt_date(tenure['end_date'])}"
        f" ｜ 约 {tenure['days']} 天（{fmt_years(tenure['days'])}）"
    )
    if report.is_fallback:
        st.warning(
            "这只基金目前没有查询到在任基金经理，下面的统计对象是公开记录里最后一位经理。"
        )
    if report.primary_edu or report.primary_resume:
        with st.expander("经理简介（来自公开披露）"):
            if report.primary_edu:
                st.markdown(f"**学历/背景**：{report.primary_edu}")
            if report.primary_resume:
                st.markdown("**简历摘要**：")
                st.write(report.primary_resume[:1200])


def render_manager_tenure(report) -> None:
    tenure = report.tenure
    mdd = tenure.get("max_drawdown") or {}

    st.subheader(f"{report.primary_name} 管这只基金的成绩单")
    metric_row(
        [
            ("任职时长", f"{tenure['days']} 天", fmt_years(tenure["days"])),
            ("任期累计涨跌", fmt_pct(tenure["cumulative"], 2, signed=True), "这位经理上任以来的总涨跌"),
            ("任期年化", fmt_pct(tenure.get("annualized"), 2, signed=True), "把任期总涨跌平均摊到每年"),
            ("任期最大回撤", fmt_pct(mdd.get("value"), 2), "任期内最难受的时候"),
        ]
    )

    if tenure.get("annualized") is None:
        st.caption("任职时间太短，年化数字参考意义有限，主要看累计涨跌。")

    lines = []
    if mdd.get("value") is not None and mdd["value"] < 0:
        left = 10000 * (1 + mdd["value"])
        lines.append(
            f"假设这位经理上任时你投入 1 万元，按历史净值算，最难受的时候账户大约只剩 "
            f"**{left:,.0f} 元**。这是账面最低点，不是最终结果；但如果正好在低点卖出，就会把浮亏变成真亏。"
        )
        lines.append(
            f"这次下跌从 {fmt_date(mdd['peak_date'])} 到 {fmt_date(mdd['trough_date'])}，"
            f"用了约 {mdd['fall_days']} 天。"
        )
        if mdd.get("recover_date") is not None:
            lines.append(
                f"之后到 {fmt_date(mdd['recover_date'])} 才重新涨回高点，"
                f"从低点算用了约 {mdd['recover_days'] / 365.25:.1f} 年。"
            )
        else:
            lines.append("到最新数据为止，还没有涨回那个高点。")
    else:
        lines.append("这段区间没有出现明显回撤（不代表以后不会跌）。")

    if tenure.get("volatility") is not None:
        lines.append(
            f"任期波动：年化约 {fmt_pct(tenure['volatility'], 1)}，"
            f"按历史波动粗略分档属于【{tenure['volatility_label']}】。"
        )

    if tenure["days"] < 180:
        lines.append(
            "注意：这位经理在这只基金上的任职时间还不到半年。短期业绩很容易被一波行情放大或拖累，"
            "年化数字看看就好，别当成长期能力证明。"
        )

    lines.append(
        "**大白话总结**：这段任职里，基金整体是"
        + ("上涨的" if tenure["cumulative"] > 0 else "下跌的")
        + f"，累计 {fmt_pct(tenure['cumulative'], 1, signed=True)}；"
        + (f"但中途最多曾回撤 {fmt_pct(mdd.get('value'), 1)}。" if mdd.get("value") is not None
           else "中途回撤数据不足，无法判断最难受的幅度。")
        + " 年化只是把这段历史平均摊到每年，不代表他以后每年都能做出同样成绩。"
    )

    st.markdown('<div class="fr-tip">' + "<br><br>".join(lines) + "</div>", unsafe_allow_html=True)


def render_current_managers(report) -> None:
    if not report.current_infos:
        return
    if len(report.current_infos) == 1 and report.current_infos[0]["name"] == report.primary_name:
        return  # 只有一位经理且就是他本人，不必重复列表

    st.subheader("现在是谁在管")
    rows = []
    for info in report.current_infos:
        rows.append(
            {
                "经理": info["name"],
                "上任日期": fmt_date(info["begin"]),
                "任职天数": info["days"],
                "任期累计": fmt_pct(info["cumulative"], 2, signed=True),
                "任期年化": fmt_pct(info["annualized"], 2, signed=True),
                "任期最大回撤": fmt_pct(info["max_drawdown"], 2),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("同一只基金可能由多位经理共同管理，各自的任职起点不同，成绩不能简单相加。")


def render_manager_history(report) -> None:
    st.subheader(f"{report.primary_name} 以前管过哪些基金")
    if not report.history:
        st.info("公开数据里没有查到这位经理的其他基金任职记录。")
        return

    rows = []
    for seg in report.history:
        end_text = fmt_date(seg.end) if seg.end is not None else "至今"
        period = f"{fmt_date(seg.begin)} ~ {end_text}" + ("（本基金）" if seg.is_current_fund else "")
        if seg.note and seg.annualized is None:
            ann_text, mdd_text = seg.note, "—"
        else:
            ann_text = fmt_pct(seg.annualized, 2, signed=True)
            mdd_text = fmt_pct(seg.max_drawdown, 2)
        rows.append(
            {
                "基金名称": seg.fund_name,
                "代码": seg.ts_code,
                "任职区间": period,
                "时长": f"{seg.days} 天",
                "任期年化": ann_text,
                "任期最大回撤": mdd_text,
            }
        )

    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(
        "同一只基金的 A 类、C 类份额在 tushare 里是两条记录，所以可能看到名字很像的两行，"
        "它们的净值走势基本一样。"
    )
    if report.history_total > len(report.history):
        st.caption(
            f"公开记录里一共查到 {report.history_total} 只基金，这里只展示最近 {len(report.history)} 段，"
            "避免一次抓取太多接口。"
        )

    if report.hist_valid == 0:
        st.info("这些任职片段的净值数据不足 90 天，暂时算不出可靠的历史年化。")
        return

    st.markdown(
        f'<div class="fr-tip">在能算出结果的 <b>{report.hist_valid}</b> 段任职里：<br>'
        f"· 有 <b>{report.hist_positive}</b> 段任期为上涨，"
        f"<b>{report.hist_valid - report.hist_positive}</b> 段为下跌；<br>"
        f"· 任期年化最高 {fmt_pct(report.hist_max_ann, 2, signed=True)}，"
        f"最低 {fmt_pct(report.hist_min_ann, 2, signed=True)}，"
        f"简单平均约 {fmt_pct(report.hist_mean_ann, 2, signed=True)}/年；<br>"
        f"· 这些片段任期最大回撤的中位数约 {fmt_pct(report.hist_median_mdd, 2)}。<br><br>"
        "不同基金的类型、规模、上任时市场位置都不一样，这些数字不能简单相加，"
        "更不能拿某一段的高收益去推断下一只基金。短期任职的年化会被放大，要结合任职时长一起看。"
        "</div>",
        unsafe_allow_html=True,
    )


def render_peer(report) -> None:
    st.subheader("和同类基金比一比（小样本参考）")

    if report.peer_reason or report.peer_count < 3:
        st.info(report.peer_reason or "本次没有足够的同类样本，已跳过对比。这不影响前面基于本基金自身历史的统计。")
        return

    fund_ann = report.tenure.get("annualized")
    fund_mdd = (report.tenure.get("max_drawdown") or {}).get("value")
    fund_type = report.basic.get("fund_type", "同类")

    st.caption(
        f"随机抽取了 {report.peer_count} 只同类型（{fund_type}）基金，在相同时间段内做同样的历史计算；"
        f"对比区间 {fmt_date(report.tenure['start_date'])} 至 {fmt_date(report.tenure['end_date'])}。"
    )
    metric_row(
        [
            ("本基金任期年化", fmt_pct(fund_ann, 2, signed=True), ""),
            ("同类中位数", fmt_pct(report.peer_ann_median, 2, signed=True), "样本的中间水平"),
            ("样本内位置", f"第 {report.peer_rank} / {report.peer_count} 位" if report.peer_rank else "—", "从高到低排"),
            ("本基金最大回撤", fmt_pct(fund_mdd, 2), f"同类中位数 {fmt_pct(report.peer_mdd_median, 2)}"),
        ]
    )

    notes = []
    if fund_ann is not None and report.peer_ann_median is not None:
        if fund_ann > report.peer_ann_median:
            notes.append("这段历史里，它的收益比样本中位数高一些，但这只是小样本里的一次比较，不能说明经理以后也能保持。")
        elif fund_ann < report.peer_ann_median:
            notes.append("这段历史里，它的收益比样本中位数低一些，短期跑输不代表以后一定差，短期跑赢也不代表以后一定好。")
        else:
            notes.append("这段历史里，它的收益和样本中位数差不多。")
    if fund_mdd is not None and report.peer_mdd_median is not None:
        notes.append(f"样本里有 {report.peer_deeper_count} 只基金任期回撤比它更深。")

    notes.append("注意：这只是一个很小的抽样，不是全市场排名，也不能说明这位经理比谁更厉害；样本换一批、时间段换一段，结果都可能不一样。")

    invest_type = str(report.basic.get("invest_type", ""))
    fund_name = str(report.basic.get("name", ""))
    if ("指数" in invest_type) or ("ETF" in fund_name.upper()) or ("指数" in fund_name):
        notes.append(
            "补充：这是一只指数/ETF 类产品，而 tushare 的基金分类比较粗，同类样本里可能混有主动管理基金，"
            "投资策略并不完全相同，对比只能粗略看。"
        )

    st.markdown('<div class="fr-tip">' + "<br><br>".join(notes) + "</div>", unsafe_allow_html=True)


def render_manager_tab(peers: int) -> None:
    st.text_input(
        "基金代码（可以和另一个页面填不同的基金）",
        key="code_input_manager",
        max_chars=12,
        placeholder="例如 000001 / 510300",
        help="与“大白话解读”页共用同一只基金也可以，直接在左边输入同一个代码",
    )
    if st.button("分析这只基金的经理", type="primary", key="btn_manager"):
        code = check_code(st.session_state.get("code_input_manager", ""))
        if code:
            st.session_state["manager_query"] = code

    code = st.session_state.get("manager_query", "")
    if not code:
        st.info("在上面输入 6 位基金代码，然后点「分析这只基金的经理」。")
        return

    with st.spinner(f"正在统计 {code} 的经理任职表现并采集同类样本…（首次通常要 20~60 秒）"):
        try:
            data = load_manager(code, peers)
        except Exception as exc:
            show_error(exc)
            return

    report = data["report"]
    chart = data["chart"]

    render_manager_header(report)
    st.divider()
    render_manager_tenure(report)
    st.divider()

    st.subheader("任期净值对比")
    if chart:
        try:
            st.image(chart, caption="上任前后净值对比（任职起始日归一为 1，虚线为同类样本平均）")
        except Exception as exc:
            st.info(f"图表读取失败（{exc}），文字结论不受影响。")
    else:
        st.info("这次没有生成图表（可能净值数据太少），文字结论仍然有效。")
    if chart:
        st.caption(f"图表已同时保存到本地：{chart}")

    st.divider()
    render_current_managers(report)
    render_manager_history(report)
    st.divider()
    render_peer(report)

    with st.expander("这几个数字到底怎么理解"):
        st.markdown(
            "- **任期年化**：把他上任以来的总涨跌，平均摊到每一年。比如两年涨 21%，大约折合 10%/年。"
            "但它不是“每年稳稳 10%”，中间可能一年涨 40%、一年跌 20%。\n"
            "- **任期最大回撤**：从任期内最高点跌到最低点的幅度，告诉你“最难受的时候有多难受”。"
            "回撤不等于最终亏损：之后涨回来只是浮亏，低点卖出才会变成真亏。\n"
            "- **任职时长**：时间越短，年化越容易被一两波行情放大，参考价值越低。\n"
            "- **同类对比**：只是拿一小批同类型基金做参照，不是评奖，也不是排名榜。"
        )

    st.markdown(
        '<div class="fr-warn">'
        "<b>基金经理过往历史业绩不代表未来表现，</b><br>"
        "<b>不能用来预判以后收益，不构成任何投资建议。</b><br><br>"
        "同一位经理换了基金、换了市场环境，结果可能完全不同；历史高收益也可能伴随更大的回撤。"
        "请结合自己的风险承受能力，独立做决定。"
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 侧边栏 + 主页面
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ 查询设置")
    peers = st.slider(
        "同类对比样本数",
        min_value=3,
        max_value=12,
        value=8,
        help="数值越大多抓几只同类基金，结果更稳，但等待时间更长、更容易触发接口限流。",
    )
    if st.button("清空缓存 / 重新拉取", help="tushare 数据更新后，或想换一批同类样本时点这里。"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.markdown("**示例代码**")
    st.markdown(
        "- `000001` 华夏成长（主动混合）\n"
        "- `510300` 沪深300ETF（场内）\n"
        "- `002599` C 类份额（看销售服务费）"
    )
    st.divider()
    st.caption(
        "数据来源：tushare。所有指标都是对历史数据的统计，"
        "不预测未来，不构成投资建议。"
    )

st.markdown(
    """
    <div class="fr-hero">
      <h1>📈 FundReader · 基金大白话解读</h1>
      <p>把费率、回撤、持仓、基金经理历史业绩翻译成人话<br>
      只做历史统计 · 不预测未来 · 不构成任何投资建议</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_fund, tab_manager = st.tabs(["🧾 基金大白话解读", "👤 基金经理业绩分析"])

with tab_fund:
    render_fund_tab()

with tab_manager:
    render_manager_tab(peers)

st.divider()
st.caption(
    "FundReader ｜ 数据来自 tushare 公开接口 ｜ 历史业绩不代表未来表现，本工具不提供任何投资建议。"
)
