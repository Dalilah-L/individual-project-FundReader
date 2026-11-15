# -*- coding: utf-8 -*-
"""
main.py
=======
FundReader · 大学生基金大白话解读器 —— 程序入口。

功能菜单：
  1. 基金大白话解读（费率、回撤、持仓、风险）
  2. 基金经理业绩分析（历史任职统计 + 同类小样本对比 + 任职净值对比图）
  3. 退出

设计原则：
- 只做历史数据解读，不预测未来收益；
- 不输出“建议买 / 值得买”等投资建议；
- 所有异常都转成中文友好提示，程序不直接崩溃。
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Optional

# 让 Windows 下重定向输出时遇到生僻字符也不会崩溃
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

from fund_fetch import (
    DataFetchError,
    FundFetcher,
    FundNotFoundError,
    FundReaderError,
    NoNavDataError,
    OUTPUT_DIR,
    TokenError,
    split_fund_code,
)
from fund_interpret import (
    analyze_fund,
    enable_ansi,
    render_fund_report,
)
from fund_manager_analysis import (
    ManagerDataMissingError,
    analyze_manager,
    render_manager_report,
)
from fund_plot import plot_manager_tenure, plot_nav_history


MENU = """
==============================================================
  FundReader · 大学生基金大白话解读器
  把费率、回撤、持仓、基金经理业绩翻译成人话
  只做历史统计 · 不预测未来 · 不构成投资建议
==============================================================
  1. 基金大白话解读
     （费率 / 最大回撤 / 持仓 / 风险提醒）

  2. 基金经理业绩分析
     （任期年化 / 任期回撤 / 历任基金 / 同类对比）

  3. 退出程序
--------------------------------------------------------------
"""


def print_banner() -> None:
    print(MENU)


def ask_fund_code() -> Optional[str]:
    """询问基金代码；输入 q 返回主菜单。"""
    print()
    print("代码提示：场外基金直接输入 6 位数字（如 000001），")
    print("          场内 ETF 也可以输入 6 位数字（如 510300），程序会自动补后缀。")
    while True:
        try:
            raw = input("请输入基金代码（输入 q 返回主菜单）：").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if raw.lower() in {"q", "quit", "exit"}:
            return None
        if not raw:
            print("代码不能为空，请重新输入。")
            continue
        try:
            split_fund_code(raw)  # 只做格式校验，真正查询在分析函数里
            return raw
        except ValueError as exc:
            print(exc)


def run_fund_interpretation(fetcher: FundFetcher, code: str) -> None:
    print("\n正在获取基金数据，请稍候...")
    result = analyze_fund(fetcher, code)
    render_fund_report(result)

    print()
    try:
        image_path = plot_nav_history(result)
        print(f"净值走势图已保存：{image_path}")
    except Exception as exc:
        print(f"[图表没有生成] {exc}")
        print("文字解读已经完成，不影响你查看上面的内容。")


def run_manager_analysis(fetcher: FundFetcher, code: str) -> None:
    print("\n正在获取基金经理数据并做历史统计...")
    print("（同类基金对比需要多取几只基金的历史净值，第一次运行可能要几十秒）")
    report = analyze_manager(fetcher, code)
    render_manager_report(report)

    print()
    try:
        image_path = plot_manager_tenure(report)
        print(f"基金经理任职对比图已保存：{image_path}")
    except Exception as exc:
        print(f"[图表没有生成] {exc}")
        print("文字分析已经完成，不影响你查看上面的内容。")


def main() -> int:
    enable_ansi()
    print_banner()

    # 启动时先检查 token，避免用户输完代码才发现没配置
    try:
        fetcher = FundFetcher()
    except TokenError as exc:
        print("[无法启动] " + str(exc))
        return 1
    except FundReaderError as exc:
        print(f"[无法启动] {exc}")
        return 1
    except Exception as exc:
        print(f"[无法启动] 初始化数据接口时出错：{exc}")
        return 1

    print("TUSHARE_TOKEN 已配置，数据接口初始化成功。")
    print(f"图片输出目录：{OUTPUT_DIR}")
    print("提示：第一次运行会从 tushare 拉取数据，可能需要等待几秒到几十秒。")

    while True:
        try:
            choice = input("\n请选择功能 [1/2/3]：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n已退出 FundReader。")
            return 0

        if choice in {"3", "q", "quit", "exit"}:
            print("\n已退出 FundReader。记住：基金不保本，历史业绩不代表未来。")
            return 0

        if choice not in {"1", "2"}:
            print("没有这个选项，请输入 1、2 或 3。")
            continue

        code = ask_fund_code()
        if code is None:
            continue

        try:
            if choice == "1":
                run_fund_interpretation(fetcher, code)
            else:
                run_manager_analysis(fetcher, code)
        except FundNotFoundError as exc:
            print(f"\n[基金代码有问题] {exc}")
        except ManagerDataMissingError as exc:
            print(f"\n[基金经理数据不完整] {exc}")
        except NoNavDataError as exc:
            print(f"\n[没有净值数据] {exc}")
        except DataFetchError as exc:
            print(f"\n[数据接口出错] {exc}")
        except TokenError as exc:
            print(f"\n[token 有问题] {exc}")
        except FundReaderError as exc:
            print(f"\n[没有完成] {exc}")
        except ValueError as exc:
            print(f"\n[输入有误] {exc}")
        except KeyboardInterrupt:
            print("\n\n已取消本次查询。")
        except Exception as exc:  # 最后一道保险：任何意外都不让程序崩掉
            print(f"\n[程序遇到意外问题] {exc}")
            if os.environ.get("FUNDREADER_DEBUG") == "1":
                traceback.print_exc()
            else:
                print("可以设置环境变量 FUNDREADER_DEBUG=1 查看详细错误堆栈。")

        try:
            again = input("\n按回车返回主菜单，输入 q 退出程序：").strip().lower()
            if again in {"q", "quit", "exit"}:
                print("\n已退出 FundReader。")
                return 0
        except (EOFError, KeyboardInterrupt):
            print("\n\n已退出 FundReader。")
            return 0


if __name__ == "__main__":
    sys.exit(main())
