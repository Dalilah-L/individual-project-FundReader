# -*- coding: utf-8 -*-
"""
fund_fetch.py
=============
数据获取模块：所有和 tushare 打交道的代码都放在这里。

职责：
1. 读取 TUSHARE_TOKEN（优先环境变量，其次项目目录下的 .env 文件）；
2. 把用户输入的基金代码整理成 tushare 认识的格式（如 000001.OF / 510300.SH）；
3. 拉取基金基础信息、历史净值、持仓、基金经理任职记录；
4. 对网络异常、权限不足、代码不存在等情况给出中文友好提示。

注意：本模块只负责“取数据”，不做任何投资判断。
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import tushare as ts
except ImportError:  # 依赖没装时给出明确提示，而不是直接崩掉
    ts = None


# ---------------------------------------------------------------------------
# 路径与异常定义
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "output"


class FundReaderError(Exception):
    """项目自定义异常基类，主程序统一捕获后输出中文提示。"""


class TokenError(FundReaderError):
    """没有配置 tushare token，或 token 不可用。"""


class FundNotFoundError(FundReaderError):
    """基金代码不存在或格式不对。"""


class DataFetchError(FundReaderError):
    """接口调用失败：网络、权限、频率限制等。"""


class NoNavDataError(DataFetchError):
    """基金存在，但没有取到可用的历史净值。"""


# ---------------------------------------------------------------------------
# token 读取：绝不把密钥写进源码
# ---------------------------------------------------------------------------

def _read_token_from_env_file(path: Path) -> str:
    """从 .env 文件里读取 TUSHARE_TOKEN（简单实现，不依赖第三方库）。"""
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == "TUSHARE_TOKEN":
                    return value.strip().strip('"').strip("'")
    except OSError:
        return ""
    return ""


def load_token() -> str:
    """
    读取 token 的顺序：
    1. 系统环境变量 TUSHARE_TOKEN（推荐）；
    2. 项目根目录下的 .env 文件（方便本地调试，已被 .gitignore 忽略）。

    两种方式都没有时，抛出带操作步骤的 TokenError。
    """
    token = (os.environ.get("TUSHARE_TOKEN") or "").strip()
    if not token:
        token = _read_token_from_env_file(BASE_DIR / ".env")

    if not token:
        raise TokenError(
            "没有找到 TUSHARE_TOKEN。\n"
            "\n"
            "请先到 https://tushare.pro 注册，复制个人主页里的 token，然后任选一种方式配置：\n"
            "  方式一（推荐，当前终端窗口有效）\n"
            '    PowerShell:  $env:TUSHARE_TOKEN="你的token"\n'
            "    CMD:         set TUSHARE_TOKEN=你的token\n"
            "\n"
            "  方式二（写入用户环境变量，之后新开的终端都有效）\n"
            '    PowerShell:  setx TUSHARE_TOKEN "你的token"\n'
            "\n"
            "  方式三（本地调试）\n"
            "    在项目目录新建 .env 文件，写一行：TUSHARE_TOKEN=你的token\n"
            "    .env 已加入 .gitignore，不会被提交到 git。\n"
            "\n"
            "设置完成后重新打开终端，再运行 python main.py。"
        )

    # 常见占位符，避免用户忘了替换就运行
    if token.lower() in {"your_token", "your_token_here", "xxx", "token"}:
        raise TokenError("TUSHARE_TOKEN 看起来还是占位符，请换成你从 tushare 复制的真实 token。")

    return token


# ---------------------------------------------------------------------------
# 基金代码处理
# ---------------------------------------------------------------------------

def split_fund_code(raw: str) -> Tuple[str, str]:
    """
    把用户输入拆成 (6位数字, 市场后缀)。
    支持：000001、000001.OF、510300.SH、159915.SZ。
    格式不对时抛出 ValueError。
    """
    text = str(raw or "").strip().upper().replace(" ", "")
    match = re.fullmatch(r"(\d{6})(?:\.(OF|SH|SZ))?", text)
    if not match:
        raise ValueError(
            "基金代码格式不正确。请输入 6 位数字，例如 000001，"
            "或带后缀的 000001.OF / 510300.SH。"
        )
    return match.group(1), (match.group(2) or "")


def _candidate_codes(digits: str) -> List[str]:
    """
    用户只输入 6 位数字时，按常见程度生成候选代码。
    场外基金多为 .OF；5 开头的 ETF 在上交所；15/16/18 开头多在深交所。
    """
    if digits.startswith("5"):
        order = ["SH", "OF", "SZ"]
    elif digits.startswith(("15", "16", "18")):
        order = ["SZ", "OF", "SH"]
    else:
        order = ["OF", "SH", "SZ"]
    return [f"{digits}.{suffix}" for suffix in order]


# ---------------------------------------------------------------------------
# 异常判断小工具
# ---------------------------------------------------------------------------

def _is_permission_error(message: str) -> bool:
    return ("40203" in message) or ("没有接口" in message) or ("权限" in message)


def _is_token_error(message: str) -> bool:
    lowered = message.lower()
    if "token" not in lowered:
        return False
    keywords = ("无效", "错误", "不存在", "不对", "失效", "过期", "40101")
    return any(word in message for word in keywords)


def _is_rate_limit_error(message: str) -> bool:
    lowered = message.lower()
    return ("每分钟" in message) or ("频率" in message) or ("访问次数" in message) or ("limit" in lowered and "exceed" in lowered)


def _is_network_error(message: str) -> bool:
    lowered = message.lower()
    keywords = ("timed out", "timeout", "connection", "network", "ssl", "proxy", "远程", "网络")
    return any(word in lowered for word in keywords)


def _clean_text(value) -> str:
    """把 None / NaN / 空格统一成空字符串。"""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def _to_float(value) -> Optional[float]:
    """安全地把 tushare 返回的字符串转成 float，失败返回 None。"""
    text = _clean_text(value)
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def parse_date(value):
    """把 20241226 这样的字符串转成 pandas 时间；失败返回 NaT。"""
    text = _clean_text(value)
    if not text:
        return pd.NaT
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


# ---------------------------------------------------------------------------
# 核心：FundFetcher
# ---------------------------------------------------------------------------
class FundFetcher:
    """
    tushare 数据获取器。

    使用方法：
        fetcher = FundFetcher()
        ts_code, basic = fetcher.resolve_fund("000001")
        nav = fetcher.get_nav(ts_code)
    """

    def __init__(self, token: Optional[str] = None):
        if ts is None:
            raise FundReaderError(
                "缺少 tushare 库。请先在终端执行：pip install -r requirements.txt"
            )

        self.token = (token or load_token()).strip()
        try:
            self.pro = ts.pro_api(self.token)
        except Exception as exc:  # tushare 内部也可能因为 token 格式抛错
            raise TokenError(f"初始化 tushare 失败，请检查 token 是否正确。原始错误：{exc}") from exc

        # 简单的内存缓存，避免同一个基金反复请求接口
        self._basic_cache: Dict[str, dict] = {}
        self._nav_cache: Dict[str, pd.DataFrame] = {}
        self._manager_cache: Dict[str, pd.DataFrame] = {}
        self._manager_history_cache: Dict[str, pd.DataFrame] = {}
        self._stock_names: Dict[str, str] = {}
        self._basic_bulk: Optional[pd.DataFrame] = None
        self._basic_bulk_failed = False

        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        except OSError:
            # 极端情况下（只读目录）不阻塞主流程
            pass

    # ------------------------------------------------------------------
    # 底层请求：统一做重试和中文错误翻译
    # ------------------------------------------------------------------
    def _request(self, api_name: str, **kwargs) -> pd.DataFrame:
        last_error = None
        for attempt in range(3):
            try:
                result = getattr(self.pro, api_name)(**kwargs)
                if result is None:
                    return pd.DataFrame()
                return result
            except Exception as exc:
                message = str(exc)
                last_error = exc

                if _is_permission_error(message):
                    raise DataFetchError(
                        f"当前 tushare 账号没有 {api_name} 接口的访问权限。\n"
                        "可以去 https://tushare.pro/user/token 查看账号积分，"
                        "或在 tushare 文档里确认该接口需要的积分门槛。\n"
                        f"接口原始提示：{message}"
                    ) from exc

                if _is_token_error(message):
                    raise TokenError(
                        "tushare token 无效或已失效，请重新复制 token 并更新环境变量 TUSHARE_TOKEN。\n"
                        f"接口原始提示：{message}"
                    ) from exc

                if _is_rate_limit_error(message):
                    if attempt < 2:
                        time.sleep(12)
                        continue
                    raise DataFetchError(
                        "tushare 接口调用太频繁，被限流了。请等 1 分钟再试，"
                        "或减少同类基金对比的数量。\n"
                        f"接口原始提示：{message}"
                    ) from exc

                if _is_network_error(message) and attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue

                raise DataFetchError(
                    f"调用 tushare 接口 {api_name} 失败，可能是网络不稳定或接口暂时不可用。\n"
                    f"接口原始提示：{message}"
                ) from exc

        raise DataFetchError(
            f"调用 tushare 接口 {api_name} 连续失败。\n最后一次错误：{last_error}"
        )

    # ------------------------------------------------------------------
    # 基金基础信息
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_basic(row: pd.Series) -> dict:
        """把 fund_basic 的一行整理成统一格式的字典。"""
        m_fee = _to_float(row.get("m_fee"))
        c_fee = _to_float(row.get("c_fee"))
        return {
            "ts_code": _clean_text(row.get("ts_code")),
            "name": _clean_text(row.get("name")) or "未命名基金",
            "fund_type": _clean_text(row.get("fund_type")),
            "management": _clean_text(row.get("management")),
            "custodian": _clean_text(row.get("custodian")),
            "found_date": _clean_text(row.get("found_date")),
            "benchmark": _clean_text(row.get("benchmark")),
            "invest_type": _clean_text(row.get("invest_type")),
            "status": _clean_text(row.get("status")),
            # tushare 里费率单位是“百分比”，这里保留原始数值，展示时再除 100
            "m_fee": m_fee,
            "c_fee": c_fee,
        }

    def get_basic(self, ts_code: str) -> dict:
        """获取单只基金的基础信息（带内存缓存）。"""
        if ts_code in self._basic_cache:
            return dict(self._basic_cache[ts_code])

        df = self._request("fund_basic", ts_code=ts_code)
        if df is None or df.empty:
            raise FundNotFoundError(
                f"没有找到代码为 {ts_code} 的基金。请检查代码是否正确，"
                "场外基金一般以 .OF 结尾（如 000001.OF），场内 ETF 用 .SH/.SZ（如 510300.SH）。"
            )

        basic = self._normalise_basic(df.iloc[0])
        self._basic_cache[ts_code] = basic
        return dict(basic)

    def resolve_fund(self, raw_code: str) -> Tuple[str, dict]:
        """
        把用户输入解析成真实存在的基金代码。
        返回 (ts_code, basic)。
        """
        digits, suffix = split_fund_code(raw_code)
        candidates = [f"{digits}.{suffix}"] if suffix else _candidate_codes(digits)

        for code in candidates:
            try:
                basic = self.get_basic(code)
                return code, basic
            except FundNotFoundError:
                continue  # 换下一个后缀再试
            except DataFetchError:
                raise  # 权限/网络问题不必换后缀重试

        raise FundNotFoundError(
            f"没有找到代码为 {digits} 的基金。\n"
            "可能的原因：\n"
            "  1. 代码输错了一位数；\n"
            "  2. 这是一只已清盘/尚未成立的基金；\n"
            "  3. 代码后缀不对。场外基金用 .OF（如 000001.OF），"
            "上交所 ETF 用 .SH（如 510300.SH），深交所 ETF 用 .SZ（如 159915.SZ）。"
        )

    # ------------------------------------------------------------------
    # 全市场基金列表（用于找同类基金做对比，带本地缓存）
    # ------------------------------------------------------------------
    def _read_bulk_cache(self) -> Optional[pd.DataFrame]:
        path = CACHE_DIR / "fund_basic_all.csv"
        try:
            if path.exists() and (time.time() - path.stat().st_mtime) < 24 * 3600:
                df = pd.read_csv(path, dtype=str)
                if not df.empty and {"ts_code", "fund_type"}.issubset(df.columns):
                    return df
        except Exception:
            return None
        return None

    def get_basic_bulk(self) -> pd.DataFrame:
        """
        分页拉取全市场基金基础信息，并缓存 24 小时。
        只有“同类基金对比”会用到；即使失败也不影响单只基金解读。
        """
        if self._basic_bulk is not None:
            return self._basic_bulk

        cached = self._read_bulk_cache()
        if cached is not None:
            self._basic_bulk = cached
            return cached

        if self._basic_bulk_failed:
            raise DataFetchError("之前获取全市场基金列表失败，本次跳过同类对比。")

        frames: List[pd.DataFrame] = []
        offset = 0
        try:
            while offset <= 45000:
                page = self._request(
                    "fund_basic", status="L", offset=offset, limit=15000
                )
                if page is None or page.empty:
                    break
                frames.append(page)
                if len(page) < 15000:
                    break
                offset += 15000

            if not frames:
                raise DataFetchError("tushare 没有返回基金列表。")

            df = pd.concat(frames, ignore_index=True)
            df = df.drop_duplicates(subset=["ts_code"], keep="first")
            keep_cols = [
                col for col in
                ["ts_code", "name", "fund_type", "found_date", "m_fee", "c_fee", "status"]
                if col in df.columns
            ]
            df = df[keep_cols].copy()
            for col in keep_cols:
                df[col] = df[col].astype("string").fillna("").str.strip()

            try:
                path = CACHE_DIR / "fund_basic_all.csv"
                df.to_csv(path, index=False, encoding="utf-8-sig")
            except OSError:
                pass

            self._basic_bulk = df
            return df
        except Exception:
            self._basic_bulk_failed = True
            raise

    # ------------------------------------------------------------------
    # 历史净值
    # ------------------------------------------------------------------
    def get_nav(self, ts_code: str) -> pd.DataFrame:
        """
        获取一只基金的全部历史净值，返回按日期升序的 DataFrame。

        主要列：
          date       : 净值日期（datetime）
          unit_nav   : 单位净值
          accum_nav  : 累计净值
          adj_nav    : 复权净值（把分红再投资算进去，计算收益更准确）
          value      : 本程序统一使用的净值列（优先复权净值）
          value_label: value 的含义，用于图表和文字说明
        """
        if ts_code in self._nav_cache:
            return self._nav_cache[ts_code].copy()

        df = self._request("fund_nav", ts_code=ts_code)
        if df is None or df.empty:
            raise NoNavDataError(
                f"没有取到 {ts_code} 的历史净值。可能这只基金刚成立、已清盘，或数据尚未更新。"
            )

        df = df.copy()
        for col in ("nav_date", "unit_nav", "accum_nav", "adj_nav"):
            if col not in df.columns:
                df[col] = None

        df["nav_date"] = df["nav_date"].astype("string").fillna("").str.strip()
        df = df[df["nav_date"].str.fullmatch(r"\d{8}")]

        for col in ("unit_nav", "accum_nav", "adj_nav"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["date"] = pd.to_datetime(df["nav_date"], format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

        # 优先使用复权净值；没有复权净值时退回累计净值，再退回单位净值
        value = df["adj_nav"].where(df["adj_nav"] > 0)
        label = pd.Series("复权净值", index=df.index, dtype="object")

        mask = value.isna()
        value.loc[mask] = df.loc[mask, "accum_nav"].where(df.loc[mask, "accum_nav"] > 0)
        label.loc[mask] = "累计净值"

        mask = value.isna()
        value.loc[mask] = df.loc[mask, "unit_nav"].where(df.loc[mask, "unit_nav"] > 0)
        label.loc[mask] = "单位净值"

        df["value"] = value
        df["value_label"] = label
        df = df.dropna(subset=["value"]).reset_index(drop=True)

        if df.empty:
            raise NoNavDataError(
                f"{ts_code} 的净值数据里没有有效数值，暂时无法解读。"
            )

        self._nav_cache[ts_code] = df
        return df.copy()

    # ------------------------------------------------------------------
    # 基金经理
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_manager_df(df: pd.DataFrame) -> pd.DataFrame:
        """统一基金经理表的字段格式，尤其是把“在任”的空 end_date 处理干净。"""
        if df is None or df.empty:
            return pd.DataFrame(
                columns=[
                    "ts_code", "ann_date", "name", "begin_date", "end_date",
                    "begin_dt", "end_dt", "is_current", "resume",
                ]
            )

        df = df.copy()
        for col in ("ts_code", "ann_date", "name", "begin_date", "end_date", "resume"):
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].astype("string").fillna("").str.strip()

        df["begin_dt"] = pd.to_datetime(df["begin_date"], format="%Y%m%d", errors="coerce")
        df["end_dt"] = pd.to_datetime(df["end_date"], format="%Y%m%d", errors="coerce")
        # tushare 里“仍在任”的 end_date 常常是一个空格，这里统一成 True
        df["is_current"] = df["end_date"].eq("") | df["end_dt"].isna()

        df = df.sort_values(
            by=["is_current", "begin_date"], ascending=[False, False]
        ).reset_index(drop=True)
        return df

    def get_managers(self, ts_code: str) -> pd.DataFrame:
        """获取某只基金的全部基金经理任职记录（含已离任的）。"""
        if ts_code in self._manager_cache:
            return self._manager_cache[ts_code].copy()

        df = self._request("fund_manager", ts_code=ts_code)
        normalised = self._normalise_manager_df(df)
        self._manager_cache[ts_code] = normalised
        return normalised.copy()

    def get_manager_history(self, manager_name: str) -> pd.DataFrame:
        """
        按姓名获取一位基金经理管理过的所有基金记录。
        说明：不同人可能重名，tushare 只能按姓名匹配，本程序默认同名即同一人。
        """
        name = _clean_text(manager_name)
        if not name:
            return pd.DataFrame()

        if name in self._manager_history_cache:
            return self._manager_history_cache[name].copy()

        df = self._request("fund_manager", name=name)
        normalised = self._normalise_manager_df(df)
        if not normalised.empty:
            normalised = normalised.drop_duplicates(
                subset=["ts_code", "begin_date"], keep="first"
            ).reset_index(drop=True)

        self._manager_history_cache[name] = normalised
        return normalised.copy()

    # ------------------------------------------------------------------
    # 基金名称 / 持仓
    # ------------------------------------------------------------------
    def get_fund_names(self, codes: Sequence[str]) -> Dict[str, str]:
        """
        批量获取基金简称。优先用全市场列表缓存，失败时逐个查询。
        查不到就用代码本身占位，不影响主流程。
        """
        codes = list(dict.fromkeys(codes))
        result: Dict[str, str] = {}

        try:
            bulk = self.get_basic_bulk()
            sub = bulk[bulk["ts_code"].isin(codes)]
            result.update(
                {
                    str(row["ts_code"]): _clean_text(row["name"])
                    for _, row in sub.iterrows()
                }
            )
        except Exception:
            pass

        for code in codes:
            if result.get(code):
                continue
            try:
                result[code] = self.get_basic(code)["name"]
            except Exception:
                result[code] = code
        return result

    def get_peer_candidates(
        self,
        fund_type: str,
        exclude_code: str,
        before_date: str,
        pool_size: int = 24,
    ) -> List[str]:
        """
        找出同类型、成立时间早于指定日期的基金，均匀抽样作为对比样本。

        说明：这是“小样本参考”，不是全市场排名；抽样是为了控制接口调用次数。
        """
        if not fund_type:
            return []

        try:
            bulk = self.get_basic_bulk()
        except Exception:
            return []

        df = bulk.copy()
        df = df[df["fund_type"].astype(str).str.strip() == fund_type]
        df = df[df["ts_code"].astype(str) != exclude_code]
        df = df[df["found_date"].astype(str).str.fullmatch(r"\d{8}")]
        if before_date:
            df = df[df["found_date"].astype(str) < before_date]
        df = df.drop_duplicates(subset=["ts_code"])

        if df.empty:
            return []

        df = df.sort_values("found_date").reset_index(drop=True)
        if len(df) <= pool_size:
            return df["ts_code"].astype(str).tolist()

        # 用等距抽样覆盖不同成立年份的基金，避免只抽到某一年的产品
        indexes = []
        step = (len(df) - 1) / max(pool_size - 1, 1)
        for i in range(pool_size):
            idx = int(round(i * step))
            if idx not in indexes:
                indexes.append(idx)
        return df.iloc[indexes]["ts_code"].astype(str).tolist()

    def get_portfolio(self, ts_code: str, top: int = 10) -> Optional[pd.DataFrame]:
        """
        获取最新一期前十大持仓。取不到时返回 None（属于可选信息，不影响主流程）。
        """
        try:
            df = self._request("fund_portfolio", ts_code=ts_code)
        except Exception:
            return None

        if df is None or df.empty or "end_date" not in df.columns:
            return None

        df = df.copy()
        df["end_date"] = df["end_date"].astype("string").fillna("").str.strip()
        df = df[df["end_date"].str.fullmatch(r"\d{8}")]
        if df.empty:
            return None

        latest = df["end_date"].max()
        df = df[df["end_date"] == latest].copy()
        df["stk_mkv_ratio"] = pd.to_numeric(df.get("stk_mkv_ratio"), errors="coerce")
        df = df.sort_values("stk_mkv_ratio", ascending=False)
        df = df.drop_duplicates(subset=["symbol"], keep="first").head(top)
        return df

    def get_stock_names(self, symbols: Sequence[str]) -> Dict[str, str]:
        """把股票代码翻译成公司简称（用于持仓解读）。失败时返回空字典。"""
        symbols = [s for s in dict.fromkeys(symbols) if s]
        if not symbols:
            return {}

        if not self._stock_names:
            path = CACHE_DIR / "stock_basic_names.csv"
            loaded = False
            try:
                if path.exists() and (time.time() - path.stat().st_mtime) < 7 * 24 * 3600:
                    cached = pd.read_csv(path, dtype=str)
                    if {"ts_code", "name"}.issubset(cached.columns):
                        self._stock_names = dict(zip(cached["ts_code"], cached["name"]))
                        loaded = True
            except Exception:
                loaded = False

            if not loaded:
                try:
                    df = self._request("stock_basic", fields="ts_code,name")
                    if df is not None and not df.empty:
                        df = df.dropna(subset=["ts_code", "name"]).drop_duplicates("ts_code")
                        self._stock_names = {
                            str(row["ts_code"]): _clean_text(row["name"])
                            for _, row in df.iterrows()
                        }
                        try:
                            df.to_csv(path, index=False, encoding="utf-8-sig")
                        except OSError:
                            pass
                except Exception:
                    self._stock_names = {}

        return {s: self._stock_names.get(s, "") for s in symbols}
