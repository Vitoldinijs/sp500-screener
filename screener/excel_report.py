"""Excel report generation.

The workbook is the actual user interface for this project, so it gets real
care: live formulas (edit a price, the P&L recalculates), charts, per-position
sparklines, conditional formatting, and frozen headers with autofilters.

Sheets
  Сводка    dashboard: headline stats, equity chart, sector allocation
  Позиции   open positions with live P&L
  Сделки    closed-trade log
  Капитал   daily equity curve vs benchmark
  Скрин     today's ranking with factor z-scores
  Факторы   current factor weights + learning history
  Инфо      run metadata, data sources, honest caveats
  _данные   hidden helper sheet backing the sparklines
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xlsxwriter

FONT = "Arial"
INK = "#1F2933"
MUTED = "#7B8794"
ACCENT = "#0B5FFF"
GOOD = "#0F9960"
BAD = "#D64545"
BAND = "#F5F7FA"
RULE = "#D9E2EC"

SH_DASH = "Сводка"
SH_POS = "Позиции"
SH_TRADES = "Сделки"
SH_EQ = "Капитал"
SH_SCREEN = "Скрин"
SH_FACT = "Факторы"
SH_INFO = "Инфо"
SH_DATA = "_данные"


# ---------------------------------------------------------------------------
class Styles:
    """Named formats, created once per workbook."""

    def __init__(self, wb: xlsxwriter.Workbook):
        base = {"font_name": FONT, "font_size": 10, "font_color": INK}
        self.wb = wb
        self.title = wb.add_format({**base, "font_size": 20, "bold": True})
        self.subtitle = wb.add_format({**base, "font_size": 10, "font_color": MUTED})
        self.section = wb.add_format({
            **base, "font_size": 12, "bold": True, "bottom": 1,
            "bottom_color": RULE,
        })
        self.header = wb.add_format({
            **base, "bold": True, "font_color": "#FFFFFF", "bg_color": "#334E68",
            "align": "center", "valign": "vcenter", "text_wrap": True, "border": 1,
            "border_color": "#334E68",
        })
        self.cell = wb.add_format({**base, "border": 1, "border_color": RULE})
        self.cell_alt = wb.add_format({**base, "border": 1, "border_color": RULE,
                                       "bg_color": BAND})
        self.text = wb.add_format({**base, "border": 1, "border_color": RULE,
                                   "align": "left"})
        self.bold = wb.add_format({**base, "bold": True})
        self.money = wb.add_format({**base, "num_format": "$#,##0.00", "border": 1,
                                    "border_color": RULE})
        self.money0 = wb.add_format({**base, "num_format": "$#,##0", "border": 1,
                                     "border_color": RULE})
        self.pct = wb.add_format({**base, "num_format": "0.0%", "border": 1,
                                  "border_color": RULE})
        self.pct2 = wb.add_format({**base, "num_format": "0.00%", "border": 1,
                                   "border_color": RULE})
        self.num = wb.add_format({**base, "num_format": "0.00", "border": 1,
                                  "border_color": RULE})
        self.num3 = wb.add_format({**base, "num_format": "0.000", "border": 1,
                                   "border_color": RULE})
        self.int_ = wb.add_format({**base, "num_format": "0", "border": 1,
                                   "border_color": RULE})
        self.date = wb.add_format({**base, "num_format": "yyyy-mm-dd", "border": 1,
                                   "border_color": RULE, "align": "center"})
        self.kpi_label = wb.add_format({**base, "font_color": MUTED, "font_size": 9})
        self.kpi_value = wb.add_format({**base, "font_size": 18, "bold": True})
        self.kpi_money = wb.add_format({**base, "font_size": 18, "bold": True,
                                        "num_format": "$#,##0.00"})
        self.kpi_pct = wb.add_format({**base, "font_size": 18, "bold": True,
                                      "num_format": "0.0%"})
        self.kpi_good = wb.add_format({**base, "font_size": 18, "bold": True,
                                       "num_format": "0.0%", "font_color": GOOD})
        self.kpi_bad = wb.add_format({**base, "font_size": 18, "bold": True,
                                      "num_format": "0.0%", "font_color": BAD})
        self.note = wb.add_format({**base, "font_size": 9, "font_color": MUTED,
                                   "text_wrap": True, "valign": "top"})
        self.warn = wb.add_format({**base, "font_size": 9, "font_color": "#8B6D00",
                                   "bg_color": "#FFF9DB", "text_wrap": True,
                                   "valign": "top", "border": 1,
                                   "border_color": "#F0D48A"})


def _f(v, default=0.0):
    """Coerce to a finite float for writing into a cell."""
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _table_header(ws, styles: Styles, row: int, headers: list[str], widths: list[int]):
    ws.set_row(row, 30)
    for c, (h, w) in enumerate(zip(headers, widths)):
        ws.write(row, c, h, styles.header)
        ws.set_column(c, c, w)


# ---------------------------------------------------------------------------
def build_report(
    path: Path,
    state,
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    screen: pd.DataFrame,
    stats: dict,
    meta: dict,
    cfg,
    price_history: pd.DataFrame | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = xlsxwriter.Workbook(str(path), {"nan_inf_to_errors": True})
    wb.set_properties({
        "title": "S&P 500 Screener — бумажный портфель",
        "author": "sp500-screener (GitHub Actions)",
        "comments": "Автоматический отчёт. Все сделки виртуальные.",
    })
    st = Styles(wb)

    positions = sorted(state.positions, key=lambda p: p.get("entry_date", ""))
    n_pos = len(positions)

    ws_data = _sheet_data(wb, st, positions, price_history)
    _sheet_positions(wb, st, positions, cfg, ws_data is not None,
                     price_history=price_history)
    _sheet_trades(wb, st, trades)
    _sheet_equity(wb, st, equity)
    _sheet_screen(wb, st, screen, cfg)
    _sheet_factors(wb, st, cfg, meta)
    _sheet_dashboard(wb, st, state, stats, meta, cfg, n_pos, equity, positions)
    _sheet_info(wb, st, meta, cfg)

    wb.worksheets_objs.sort(key=lambda s: [
        SH_DASH, SH_POS, SH_TRADES, SH_EQ, SH_SCREEN, SH_FACT, SH_INFO, SH_DATA
    ].index(s.get_name()))
    wb.close()
    return path


# ---------------------------------------------------------------------------
# hidden data sheet (sparkline source)
# ---------------------------------------------------------------------------
def _sheet_data(wb, st, positions, price_history):
    if price_history is None or price_history.empty or not positions:
        return None
    ws = wb.add_worksheet(SH_DATA)
    ws.hide()
    tail = price_history.tail(60)
    for i, p in enumerate(positions):
        t = p["ticker"]
        ws.write(i, 0, t)
        if t in tail.columns:
            vals = [_f(v, None) for v in tail[t].tolist()]
            vals = [v for v in vals if v is not None]
            for j, v in enumerate(vals[-60:]):
                ws.write_number(i, 1 + j, v)
    return ws


# ---------------------------------------------------------------------------
# Позиции
# ---------------------------------------------------------------------------
def _sheet_positions(wb, st, positions, cfg, has_data_sheet, price_history=None):
    ws = wb.add_worksheet(SH_POS)
    ws.hide_gridlines(2)
    hold = int(cfg.strategy["hold_days"])

    ws.write(0, 0, "Открытые позиции", st.title)
    ws.write(1, 0, f"Лестница: одна покупка в день, холд {hold} торговых дней. "
                   "Жёлтая колонка «Тек. цена» — единственное, что нужно править вручную; "
                   "остальное пересчитается формулами.", st.subtitle)

    headers = ["Тикер", "Компания", "Сектор", "Дата входа", "Цена входа", "Акций",
               "Вложено", "Тек. цена", "Стоимость", "P&L $", "P&L %",
               "Дней держим", "До выхода", "Скор входа", "Проверка AV", "Динамика"]
    widths = [9, 26, 22, 12, 11, 10, 11, 11, 12, 11, 9, 12, 11, 11, 14, 14]
    hrow = 3
    _table_header(ws, st, hrow, headers, widths)
    ws.freeze_panes(hrow + 1, 1)

    input_fmt = wb.add_format({
        "font_name": FONT, "font_size": 10, "font_color": "#0000FF",
        "num_format": "$#,##0.00", "border": 1, "border_color": RULE,
        "bg_color": "#FFFDE7",
    })

    r = hrow + 1
    for i, p in enumerate(positions):
        entry = _f(p.get("entry_price"))
        shares = _f(p.get("shares"))
        cost = _f(p.get("cost_basis"))
        last = _f(p.get("last_price"), entry)
        held = int(p.get("bars_held", 0) or 0)

        ws.write_string(r, 0, str(p.get("ticker", "")), st.text)
        ws.write_string(r, 1, str(p.get("name", "") or ""), st.text)
        ws.write_string(r, 2, str(p.get("sector", "") or ""), st.text)
        ws.write_string(r, 3, str(p.get("entry_date", "")), st.date)
        ws.write_number(r, 4, entry, st.money)
        ws.write_number(r, 5, shares, st.num3)
        ws.write_number(r, 6, cost, st.money)
        ws.write_number(r, 7, last, input_fmt)
        ws.write_formula(r, 8, f"=F{r+1}*H{r+1}", st.money, shares * last)
        ws.write_formula(r, 9, f"=I{r+1}-G{r+1}", st.money, shares * last - cost)
        ws.write_formula(r, 10, f"=IFERROR(J{r+1}/G{r+1},0)", st.pct2,
                         (shares * last - cost) / cost if cost else 0.0)
        ws.write_number(r, 11, held, st.int_)
        ws.write_formula(r, 12, f"=MAX(0,{hold}-L{r+1})", st.int_, max(0, hold - held))
        ws.write_number(r, 13, _f(p.get("entry_score")), st.num)
        ws.write_string(r, 14, str(p.get("av_verdict", "") or ""), st.text)

        if has_data_sheet:
            ws.add_sparkline(r, 15, {
                "range": f"'{SH_DATA}'!B{i+1}:BI{i+1}",
                "type": "line", "series_color": ACCENT, "high_point": True,
                "low_point": True,
            })
        r += 1

    last_row = max(r, hrow + 2)
    # totals
    ws.write_string(last_row, 5, "Итого", st.bold)
    ws.write_formula(last_row, 6, f"=SUM(G{hrow+2}:G{last_row})", st.money,
                     sum(_f(p.get("cost_basis")) for p in positions))
    ws.write_formula(last_row, 8, f"=SUM(I{hrow+2}:I{last_row})", st.money,
                     sum(_f(p.get("last_value")) for p in positions))
    ws.write_formula(last_row, 9, f"=SUM(J{hrow+2}:J{last_row})", st.money,
                     sum(_f(p.get("last_value")) - _f(p.get("cost_basis"))
                         for p in positions))
    ws.write_formula(last_row, 10, f"=IFERROR(J{last_row+1}/G{last_row+1},0)", st.pct2, 0)

    if positions:
        ws.conditional_format(hrow + 1, 10, r - 1, 10, {
            "type": "3_color_scale", "min_color": "#F8B4B4",
            "mid_color": "#FFFFFF", "max_color": "#A7E8C0",
        })
        ws.conditional_format(hrow + 1, 12, r - 1, 12, {
            "type": "data_bar", "bar_color": "#9FB3C8", "bar_solid": True,
        })
        ws.autofilter(hrow, 0, r - 1, len(headers) - 1)
    else:
        ws.write(hrow + 1, 0, "Пока нет открытых позиций — лестница только "
                              "начинает набираться.", st.note)
    return ws


# ---------------------------------------------------------------------------
# Сделки
# ---------------------------------------------------------------------------
def _sheet_trades(wb, st, trades: pd.DataFrame):
    ws = wb.add_worksheet(SH_TRADES)
    ws.hide_gridlines(2)
    ws.write(0, 0, "Закрытые сделки", st.title)
    ws.write(1, 0, "Полный журнал. Каждая позиция закрывается по истечении "
                   "срока удержания, по цене открытия следующей сессии.", st.subtitle)

    headers = ["#", "Тикер", "Сектор", "Вход", "Цена входа", "Выход",
               "Цена выхода", "Акций", "Вложено", "Получено", "P&L $", "P&L %",
               "Дней", "Скор входа", "Причина выхода"]
    widths = [6, 9, 22, 12, 11, 12, 11, 10, 11, 11, 11, 9, 8, 11, 26]
    hrow = 3
    _table_header(ws, st, hrow, headers, widths)
    ws.freeze_panes(hrow + 1, 0)

    if trades is None or trades.empty:
        ws.write(hrow + 1, 0, "Сделок ещё нет — первая закроется примерно через "
                              "месяц после старта.", st.note)
        return ws

    df = trades.copy()
    df = df.sort_values("exit_date", ascending=False)
    r = hrow + 1
    for _, t in df.iterrows():
        band = st.cell_alt if (r - hrow) % 2 == 0 else st.cell
        text_fmt = wb.add_format({"font_name": FONT, "font_size": 10,
                                  "border": 1, "border_color": RULE,
                                  "bg_color": BAND if (r - hrow) % 2 == 0 else "#FFFFFF"})
        ws.write_number(r, 0, int(_f(t.get("trade_id"))), band)
        ws.write_string(r, 1, str(t.get("ticker", "")), text_fmt)
        ws.write_string(r, 2, str(t.get("sector", "")), text_fmt)
        ws.write_string(r, 3, str(t.get("entry_date", "")), band)
        ws.write_number(r, 4, _f(t.get("entry_price")), st.money)
        ws.write_string(r, 5, str(t.get("exit_date", "")), band)
        ws.write_number(r, 6, _f(t.get("exit_price")), st.money)
        ws.write_number(r, 7, _f(t.get("shares")), st.num3)
        ws.write_number(r, 8, _f(t.get("cost_basis")), st.money)
        ws.write_number(r, 9, _f(t.get("proceeds")), st.money)
        ws.write_formula(r, 10, f"=J{r+1}-I{r+1}", st.money,
                         _f(t.get("proceeds")) - _f(t.get("cost_basis")))
        ws.write_formula(r, 11, f"=IFERROR(K{r+1}/I{r+1},0)", st.pct2,
                         _f(t.get("return_pct")))
        ws.write_number(r, 12, int(_f(t.get("bars_held"))), st.int_)
        ws.write_number(r, 13, _f(t.get("entry_score")), st.num)
        ws.write_string(r, 14, str(t.get("exit_reason", "")), text_fmt)
        r += 1

    ws.conditional_format(hrow + 1, 11, r - 1, 11, {
        "type": "3_color_scale", "min_color": "#F8B4B4",
        "mid_color": "#FFFFFF", "max_color": "#A7E8C0",
    })
    ws.autofilter(hrow, 0, r - 1, len(headers) - 1)
    return ws


# ---------------------------------------------------------------------------
# Капитал
# ---------------------------------------------------------------------------
def _sheet_equity(wb, st, equity: pd.DataFrame):
    ws = wb.add_worksheet(SH_EQ)
    ws.hide_gridlines(2)
    ws.write(0, 0, "Кривая капитала", st.title)
    ws.write(1, 0, "Портфель против бенчмарка (SPY), обе линии от одного "
                   "стартового капитала.", st.subtitle)

    headers = ["Дата", "Денежные средства", "Стоимость позиций", "Капитал",
               "Позиций", "Бенчмарк"]
    widths = [12, 18, 18, 14, 10, 14]
    hrow = 3
    _table_header(ws, st, hrow, headers, widths)
    ws.freeze_panes(hrow + 1, 1)

    if equity is None or equity.empty:
        ws.write(hrow + 1, 0, "Истории пока нет.", st.note)
        return ws

    df = equity.copy().sort_values("date")
    r = hrow + 1
    for _, e in df.iterrows():
        ws.write_string(r, 0, str(e.get("date", "")), st.date)
        ws.write_number(r, 1, _f(e.get("cash")), st.money)
        ws.write_number(r, 2, _f(e.get("positions_value")), st.money)
        ws.write_formula(r, 3, f"=B{r+1}+C{r+1}", st.money,
                         _f(e.get("cash")) + _f(e.get("positions_value")))
        ws.write_number(r, 4, int(_f(e.get("n_positions"))), st.int_)
        bench = e.get("benchmark_equity")
        if bench is None or not np.isfinite(_f(bench, np.nan)):
            ws.write_blank(r, 5, None, st.money)
        else:
            ws.write_number(r, 5, _f(bench), st.money)
        r += 1

    n = r - (hrow + 1)
    if n >= 2:
        chart = wb.add_chart({"type": "line"})
        chart.add_series({
            "name": "Портфель",
            "categories": [SH_EQ, hrow + 1, 0, r - 1, 0],
            "values": [SH_EQ, hrow + 1, 3, r - 1, 3],
            "line": {"color": ACCENT, "width": 2.0},
        })
        chart.add_series({
            "name": "S&P 500 (SPY)",
            "categories": [SH_EQ, hrow + 1, 0, r - 1, 0],
            "values": [SH_EQ, hrow + 1, 5, r - 1, 5],
            "line": {"color": MUTED, "width": 1.25, "dash_type": "dash"},
        })
        chart.set_title({"name": "Капитал во времени",
                         "name_font": {"name": FONT, "size": 12}})
        chart.set_legend({"position": "bottom", "font": {"name": FONT, "size": 9}})
        chart.set_x_axis({"num_font": {"name": FONT, "size": 8},
                          "major_gridlines": {"visible": False}})
        chart.set_y_axis({"num_font": {"name": FONT, "size": 9},
                          "num_format": "$#,##0",
                          "major_gridlines": {"visible": True,
                                              "line": {"color": RULE}}})
        chart.set_size({"width": 760, "height": 320})
        chart.set_chartarea({"border": {"none": True}})
        ws.insert_chart(hrow + 1, 7, chart)
    return ws


# ---------------------------------------------------------------------------
# Скрин
# ---------------------------------------------------------------------------
def _sheet_screen(wb, st, screen: pd.DataFrame, cfg):
    ws = wb.add_worksheet(SH_SCREEN)
    ws.hide_gridlines(2)
    ws.write(0, 0, "Скрин дня", st.title)
    ws.write(1, 0, "Кросс-секционные оценки: каждая метрика приведена к z-оценке "
                   "внутри индекса, знак развёрнут так, что больше — всегда лучше.",
             st.subtitle)

    blocks = list(cfg["metric_blocks"].keys())
    headers = (["#", "Тикер", "Компания", "Сектор", "Цена", "Композит"]
               + [f"Блок: {b}" for b in blocks]
               + ["P/E", "P/B", "ROE", "Рост выручки", "Момент. 6м", "RSI",
                  "Статус"])
    widths = ([5, 9, 24, 22, 10, 11] + [12] * len(blocks)
              + [9, 9, 9, 13, 12, 8, 24])
    hrow = 3
    _table_header(ws, st, hrow, headers, widths)
    ws.freeze_panes(hrow + 1, 2)

    if screen is None or screen.empty:
        ws.write(hrow + 1, 0, "Скрин ещё не выполнялся.", st.note)
        return ws

    df = screen.head(40).copy()
    r = hrow + 1
    for i, (ticker, row) in enumerate(df.iterrows(), start=1):
        ws.write_number(r, 0, i, st.int_)
        ws.write_string(r, 1, str(ticker), st.text)
        ws.write_string(r, 2, str(row.get("name", "") or ""), st.text)
        ws.write_string(r, 3, str(row.get("sector", "") or ""), st.text)
        ws.write_number(r, 4, _f(row.get("last_close")), st.money)
        ws.write_number(r, 5, _f(row.get("composite")), st.num3)
        c = 6
        for b in blocks:
            ws.write_number(r, c, _f(row.get(f"block_{b}")), st.num)
            c += 1
        ws.write_number(r, c, _f(row.get("pe_ratio")), st.num); c += 1
        ws.write_number(r, c, _f(row.get("pb_ratio")), st.num); c += 1
        ws.write_number(r, c, _f(row.get("roe")), st.pct); c += 1
        ws.write_number(r, c, _f(row.get("revenue_growth")), st.pct); c += 1
        ws.write_number(r, c, _f(row.get("mom_6m")), st.pct); c += 1
        ws.write_number(r, c, _f(row.get("rsi14")), st.int_); c += 1
        status = "проходит" if bool(row.get("eligible")) else str(row.get("exclude_reason", "отфильтрован"))
        ws.write_string(r, c, status, st.text)
        r += 1

    ws.conditional_format(hrow + 1, 5, r - 1, 5, {
        "type": "data_bar", "bar_color": "#7BA7F0", "bar_solid": True,
    })
    ws.conditional_format(hrow + 1, 6, r - 1, 6 + len(blocks) - 1, {
        "type": "3_color_scale", "min_color": "#F8B4B4",
        "mid_color": "#FFFFFF", "max_color": "#A7E8C0",
    })
    ws.autofilter(hrow, 0, r - 1, len(headers) - 1)
    return ws


# ---------------------------------------------------------------------------
# Факторы
# ---------------------------------------------------------------------------
def _sheet_factors(wb, st, cfg, meta):
    ws = wb.add_worksheet(SH_FACT)
    ws.hide_gridlines(2)
    ws.write(0, 0, "Факторы и самообучение", st.title)
    ws.write(1, 0, "Веса блоков подстраиваются раз в месяц по прогнозной силе "
                   "на истории, в узких границах и с откатом при ухудшении.",
             st.subtitle)

    ws.write(3, 0, "Текущие веса блоков", st.section)
    _table_header(ws, st, 4, ["Блок", "Вес", "Метрики внутри блока"], [16, 10, 74])
    weights = cfg.weights
    r = 5
    for b, metrics in cfg["metric_blocks"].items():
        ws.write_string(r, 0, b, st.text)
        ws.write_number(r, 1, _f(weights.get(b)), st.pct)
        ws.write_string(r, 2, ", ".join(metrics), st.text)
        r += 1
    ws.write_string(r, 0, "Сумма", st.bold)
    ws.write_formula(r, 1, f"=SUM(B6:B{r})", st.pct, sum(_f(v) for v in weights.values()))
    ws.conditional_format(5, 1, r - 1, 1, {
        "type": "data_bar", "bar_color": "#7BA7F0", "bar_solid": True,
    })

    r += 2
    ws.write(r, 0, "Журнал изменений весов", st.section)
    r += 1
    log_path = cfg.path("weights_history")
    if log_path.exists():
        try:
            log = pd.read_csv(log_path).tail(24)
        except Exception:  # noqa: BLE001
            log = pd.DataFrame()
    else:
        log = pd.DataFrame()

    if log.empty:
        ws.write(r, 0, "Пока пусто. Подстройка включится после "
                       f"{cfg.learning['min_trades_for_update']} закрытых сделок "
                       "и накопления истории.", st.note)
    else:
        cols = [c for c in ["run_date", "decision", "reason", "ic_value",
                            "ic_quality", "ic_growth", "ic_momentum",
                            "weights_before", "weights_after"] if c in log.columns]
        _table_header(ws, st, r, cols, [12, 14, 34, 10, 10, 10, 12, 26, 26])
        r += 1
        for _, row in log.iterrows():
            for c, col in enumerate(cols):
                val = row.get(col)
                if isinstance(val, (int, float)) and not isinstance(val, bool) and np.isfinite(_f(val, np.nan)):
                    ws.write_number(r, c, _f(val), st.num3)
                else:
                    ws.write_string(r, c, "" if pd.isna(val) else str(val), st.text)
            r += 1

    r += 2
    ws.write(r, 0, "Что означает IC", st.section)
    ws.merge_range(r + 1, 0, r + 3, 6,
                   "IC (information coefficient) — ранговая корреляция между оценкой блока "
                   "и фактической доходностью следующих 21 торговых дней. 0 означает "
                   "отсутствие прогнозной силы, 0.02–0.05 для факторных моделей на акциях "
                   "считается рабочим уровнем. Отрицательный IC на длинном окне — сигнал, "
                   "что блок работает против нас, и его вес будет снижен до нижней границы.",
                   st.note)
    return ws


# ---------------------------------------------------------------------------
# Сводка
# ---------------------------------------------------------------------------
def _sheet_dashboard(wb, st, state, stats, meta, cfg, n_pos, equity, positions):
    ws = wb.add_worksheet(SH_DASH)
    ws.hide_gridlines(2)
    ws.set_column(0, 0, 2)
    ws.set_column(1, 8, 15)

    ws.write(1, 1, "Бумажный портфель S&P 500", st.title)
    ws.write(2, 1, f"Обновлено {meta.get('run_time', '')} · данные на "
                   f"{meta.get('data_date', '—')} · стратегия "
                   f"«{cfg.strategy['name']}» · все сделки виртуальные",
             st.subtitle)

    equity_now = state.equity()
    start = _f(state.start_capital, 1.0)
    total_ret = equity_now / start - 1.0 if start else 0.0

    # KPI strip -------------------------------------------------------------
    kpis = [
        ("Капитал", equity_now, "money"),
        ("Доходность", total_ret, "pct_signed"),
        ("Бенчмарк SPY", stats.get("benchmark_return"), "pct_signed"),
        ("Макс. просадка", stats.get("max_drawdown"), "pct_bad"),
        ("Sharpe", stats.get("sharpe"), "num"),
        ("Сделок закрыто", stats.get("n_trades"), "int"),
        ("Доля прибыльных", stats.get("win_rate"), "pct"),
        ("Открыто позиций", n_pos, "int"),
    ]
    row = 4
    for i, (label, value, kind) in enumerate(kpis):
        col = 1 + (i % 4) * 2
        rr = row + (i // 4) * 3
        ws.write(rr, col, label, st.kpi_label)
        v = _f(value, np.nan)
        if not np.isfinite(v):
            ws.write_string(rr + 1, col, "—", st.kpi_value)
            continue
        if kind == "money":
            ws.write_number(rr + 1, col, v, st.kpi_money)
        elif kind == "int":
            ws.write_number(rr + 1, col, v, st.kpi_value)
        elif kind == "num":
            ws.write_number(rr + 1, col, v, st.kpi_value)
        elif kind == "pct_bad":
            ws.write_number(rr + 1, col, v, st.kpi_bad if v < 0 else st.kpi_pct)
        elif kind == "pct_signed":
            ws.write_number(rr + 1, col, v,
                            st.kpi_good if v >= 0 else st.kpi_bad)
        else:
            ws.write_number(rr + 1, col, v, st.kpi_pct)

    # Detail table ----------------------------------------------------------
    r = row + 7
    ws.write(r, 1, "Состояние портфеля", st.section)
    r += 1
    detail = [
        ("Денежные средства", state.cash, st.money),
        ("Стоимость позиций", state.positions_value(), st.money),
        ("Капитал", equity_now, st.money),
        ("Стартовый капитал", state.start_capital, st.money),
        ("Слотов всего", cfg.strategy["n_slots"], st.int_),
        ("Слотов свободно", state.open_slots(int(cfg.strategy["n_slots"])), st.int_),
        ("Ордеров в очереди", len(state.pending_orders), st.int_),
        ("Срок удержания, дней", cfg.strategy["hold_days"], st.int_),
        ("Средний срок сделки, дней", stats.get("avg_hold"), st.num),
        ("Средняя прибыль сделки", stats.get("avg_win"), st.pct2),
        ("Средний убыток сделки", stats.get("avg_loss"), st.pct2),
        ("Profit factor", stats.get("profit_factor"), st.num),
        ("Волатильность, годовая", stats.get("volatility"), st.pct),
        ("Дней в работе", stats.get("days_live"), st.int_),
    ]
    for label, value, fmt in detail:
        ws.write_string(r, 1, label, st.text)
        v = _f(value, np.nan)
        if np.isfinite(v):
            ws.write_number(r, 2, v, fmt)
        else:
            ws.write_string(r, 2, "—", st.text)
        r += 1

    # live cross-sheet formula so the dashboard follows manual price edits
    ws.write_string(r + 1, 1, "Капитал по листу «Позиции»", st.bold)
    ws.write_formula(
        r + 1, 2,
        f"=C{row+9}+IFERROR(SUM('{SH_POS}'!I5:I200),0)",
        st.money, state.cash + state.positions_value(),
    )
    ws.write(r + 2, 1, "Эта строка считается формулой: поправь «Тек. цена» на "
                       "листе «Позиции» — значение пересчитается.", st.note)

    # Sector allocation chart ---------------------------------------------
    if positions:
        sector_val: dict[str, float] = {}
        for p in positions:
            s = str(p.get("sector", "Unknown"))
            sector_val[s] = sector_val.get(s, 0.0) + _f(p.get("last_value"))
        anchor = r + 5
        ws.write(anchor - 1, 1, "Распределение по секторам", st.section)
        _table_header(ws, st, anchor, ["Сектор", "Стоимость"], [26, 14])
        rr = anchor + 1
        for s, v in sorted(sector_val.items(), key=lambda kv: -kv[1]):
            ws.write_string(rr, 1, s, st.text)
            ws.write_number(rr, 2, v, st.money)
            rr += 1
        pie = wb.add_chart({"type": "doughnut"})
        pie.add_series({
            "name": "Секторы",
            "categories": [SH_DASH, anchor + 1, 1, rr - 1, 1],
            "values": [SH_DASH, anchor + 1, 2, rr - 1, 2],
            "data_labels": {"percentage": True, "font": {"name": FONT, "size": 8}},
        })
        pie.set_title({"name": "Секторы портфеля",
                       "name_font": {"name": FONT, "size": 11}})
        pie.set_legend({"position": "right", "font": {"name": FONT, "size": 9}})
        pie.set_size({"width": 400, "height": 280})
        pie.set_chartarea({"border": {"none": True}})
        ws.insert_chart(anchor, 4, pie)

    # Equity chart mirrored on the dashboard ------------------------------
    if equity is not None and len(equity) >= 2:
        n = len(equity)
        hrow = 3
        chart = wb.add_chart({"type": "line"})
        chart.add_series({
            "name": "Портфель",
            "categories": [SH_EQ, hrow + 1, 0, hrow + n, 0],
            "values": [SH_EQ, hrow + 1, 3, hrow + n, 3],
            "line": {"color": ACCENT, "width": 2.0},
        })
        chart.add_series({
            "name": "S&P 500 (SPY)",
            "categories": [SH_EQ, hrow + 1, 0, hrow + n, 0],
            "values": [SH_EQ, hrow + 1, 5, hrow + n, 5],
            "line": {"color": MUTED, "width": 1.25, "dash_type": "dash"},
        })
        chart.set_title({"name": "Капитал портфеля",
                         "name_font": {"name": FONT, "size": 11}})
        chart.set_legend({"position": "bottom", "font": {"name": FONT, "size": 9}})
        chart.set_y_axis({"num_format": "$#,##0",
                          "num_font": {"name": FONT, "size": 9}})
        chart.set_x_axis({"num_font": {"name": FONT, "size": 8}})
        chart.set_size({"width": 620, "height": 260})
        chart.set_chartarea({"border": {"none": True}})
        ws.insert_chart(row + 7, 4, chart)
    return ws


# ---------------------------------------------------------------------------
# Инфо
# ---------------------------------------------------------------------------
def _sheet_info(wb, st, meta, cfg):
    ws = wb.add_worksheet(SH_INFO)
    ws.hide_gridlines(2)
    ws.set_column(0, 0, 34)
    ws.set_column(1, 1, 86)

    ws.write(0, 0, "Как это работает и чему не стоит верить", st.title)

    r = 2
    ws.write(r, 0, "Параметры запуска", st.section)
    r += 1
    rows = [
        ("Время запуска (UTC)", meta.get("run_time", "")),
        ("Дата данных", str(meta.get("data_date", ""))),
        ("Тикеров во вселенной", meta.get("universe_size", "")),
        ("Прошли фильтры", meta.get("eligible_count", "")),
        ("Кандидат дня", meta.get("candidate", "—")),
        ("Проверка Alpha Vantage", meta.get("av_verdict", "—")),
        ("Комментарий проверки", meta.get("av_reason", "")),
        ("Запросов AV использовано", meta.get("av_calls", 0)),
        ("Источник цен", meta.get("price_provider", "")),
        ("Фундаментал на дату", meta.get("fundamentals_asof", "—")),
        ("Версия состояния", meta.get("state_version", "")),
    ]
    for k, v in rows:
        ws.write_string(r, 0, k, st.text)
        ws.write_string(r, 1, "" if v is None else str(v), st.text)
        r += 1

    r += 1
    ws.write(r, 0, "Регламент данных", st.section)
    r += 1
    for k, v in [
        ("Ежедневно", "Один пакетный запрос OHLCV по всей вселенной. "
                      "Отсюда RSI, момент и переоценка позиций."),
        ("Раз в неделю", "Обновление фундаментальных коэффициентов. Они меняются "
                         "только с квартальными отчётами, ежедневный опрос 500 "
                         "тикеров — лишний риск блокировки."),
        ("Alpha Vantage", "1–2 запроса в день: сверка цены и коэффициентов только "
                          "для финального кандидата. Бесплатный лимит ~25/день."),
        ("Исполнение", "Сигнал считается по закрытию дня T, сделка исполняется "
                       "по цене открытия дня T+1. Это исключает подглядывание "
                       "в будущее."),
    ]:
        ws.write_string(r, 0, k, st.bold)
        ws.write_string(r, 1, v, st.note)
        ws.set_row(r, 30)
        r += 1

    r += 1
    ws.write(r, 0, "Честные оговорки", st.section)
    r += 1
    caveats = [
        "Состав индекса берётся текущий. Форвардная торговля от этого не страдает, "
        "но исторические расчёты внутри модуля самообучения содержат смещение "
        "выжившего: компании, вылетевшие из индекса, в выборке не участвуют.",
        "История фундаментальных показателей начинает накапливаться только с "
        "момента запуска бота. Пока её мало, самообучение почти целиком опирается "
        "на ценовые факторы, а веса фундаментальных блоков двигаются медленно.",
        "Проскальзывание смоделировано фиксированной величиной в базисных пунктах. "
        "Реальное исполнение на открытии бывает хуже, особенно в дни отчётностей.",
        "Портфель из ~21 бумаги концентрирован. Отдельная позиция может дать "
        "просадку, которую индекс даже не заметит.",
        "Это учебный проект и бумажная торговля. Ничто здесь не является "
        "инвестиционной рекомендацией.",
    ]
    for c in caveats:
        ws.merge_range(r, 0, r, 1, c, st.warn)
        ws.set_row(r, 32)
        r += 1
    return ws
