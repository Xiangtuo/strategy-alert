#!/usr/bin/env python3
# monitor_full.py
import os
import json
import yaml
import time
import base64
import hashlib
import logging
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tvDatafeed import TvDatafeed, Interval

# ----------------- 读取配置 -----------------
CFG_FILE = "config.yaml"
with open(CFG_FILE, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

BINANCE_BASE = "https://api.binance.com/api/v3"
SYMBOLS = cfg["binance"]["symbols"]
PRICE_DIFF_THRESHOLD = cfg["binance"]["price_diff_threshold"]
QTY_DROP_RATIO = cfg["binance"]["qty_drop_ratio"]
ORDER_SHIFT_THR = cfg["binance"]["order_shift_threshold"]

GOLD_THR = cfg["spread_check"]["gold_threshold"]
EUR_THR = cfg["spread_check"]["eur_threshold"]
TV_GC = cfg["spread_check"].get("tv_gc_symbol", "GC1!")
TV_EUR = cfg["spread_check"].get("tv_eur_symbol", "EURUSD")

DATA_DIR = cfg["general"]["data_dir"]
LOGS_DIR = cfg["general"]["logs_dir"]
REPORTS_DIR = cfg["general"]["reports_dir"]
DAILY_REPORT = cfg["general"].get("daily_report", True)
DAILY_TIME = cfg["general"].get("daily_report_time_utc8", "08:00")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ----------------- 日志 -----------------
LOG_FILE = os.path.join(LOGS_DIR, "monitor.log")
logging.basicConfig(filename=LOG_FILE,
                    level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ----------------- tvDatafeed -----------------
tv = TvDatafeed()

# ----------------- 钉钉 Webhook 读取（优先从环境变量） -----------------
# 推荐在 CI/Actions 中用 secret: DING_WEBHOOK
DING_WEBHOOK = os.getenv("DING_WEBHOOK") or cfg.get("ding_webhook")

# ----------------- 工具函数 -----------------
def now_str():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S (UTC)")

def get_binance_depth(symbol):
    """获取 ticker price 与 depth top1（bid1/ask1）"""
    try:
        price_r = requests.get(f"{BINANCE_BASE}/ticker/price", params={"symbol": symbol}, timeout=10)
        price_r.raise_for_status()
        price = float(price_r.json()["price"])

        depth_r = requests.get(f"{BINANCE_BASE}/depth", params={"symbol": symbol, "limit": 5}, timeout=10)
        depth_r.raise_for_status()
        depth = depth_r.json()
        bid1_price = float(depth["bids"][0][0])
        bid1_qty = float(depth["bids"][0][1])
        ask1_price = float(depth["asks"][0][0])
        ask1_qty = float(depth["asks"][0][1])

        mid = (bid1_price + ask1_price) / 2.0
        return {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "price": price,
            "bid1_price": bid1_price,
            "bid1_qty": bid1_qty,
            "ask1_price": ask1_price,
            "ask1_qty": ask1_qty,
            "mid": mid
        }
    except Exception as e:
        logger.exception(f"{symbol} 获取深度失败: {e}")
        return None

def save_symbol_row(symbol, row):
    path = os.path.join(DATA_DIR, f"{symbol}.csv")
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df_prev = pd.read_csv(path)
        df_out = pd.concat([df_prev, df], ignore_index=True)
    else:
        df_out = df
    df_out.to_csv(path, index=False)

def compare_and_alert_symbol(symbol, new_row):
    """
    比较新旧数据并判断是否触发告警：
    - 价格偏离（mid 与上次 mid 差 > PRICE_DIFF_THRESHOLD）
    - 深度骤减（bid1/ask1 同价位下数量减少 > QTY_DROP_RATIO）
    - 盘口移位（bid1 下跌 >= ORDER_SHIFT_THR 或 ask1 上涨 >= ORDER_SHIFT_THR）
    """
    path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        save_symbol_row(symbol, new_row)  # 初次写入，不报警
        return None

    df = pd.read_csv(path)
    last = df.iloc[-1].to_dict()

    # price (mid) deviation
    mid_last = float(last["mid"])
    mid_new = float(new_row["mid"])
    mid_diff = abs(mid_new - mid_last)
    price_trigger = mid_diff > PRICE_DIFF_THRESHOLD

    # quantity reduction only if bid1_price / ask1_price unchanged
    qty_trigger = False
    qty_details = []
    if float(new_row["bid1_price"]) == float(last["bid1_price"]):
        if float(last["bid1_qty"]) != 0:
            bid_qty_change = (float(new_row["bid1_qty"]) - float(last["bid1_qty"])) / float(last["bid1_qty"])
            if bid_qty_change < -QTY_DROP_RATIO:
                qty_trigger = True
                qty_details.append(f"bid1 数量减少 {abs(bid_qty_change):.2%}")
    else:
        # bid1价变动 — 也可能是买单撤离或新档位出现，记录为盘口移位情况 below
        logger.info(f"{symbol} bid1 价格变化: {last['bid1_price']} -> {new_row['bid1_price']}，跳过 bid1 数量直接比较")

    if float(new_row["ask1_price"]) == float(last["ask1_price"]):
        if float(last["ask1_qty"]) != 0:
            ask_qty_change = (float(new_row["ask1_qty"]) - float(last["ask1_qty"])) / float(last["ask1_qty"])
            if ask_qty_change < -QTY_DROP_RATIO:
                qty_trigger = True
                qty_details.append(f"ask1 数量减少 {abs(ask_qty_change):.2%}")
    else:
        logger.info(f"{symbol} ask1 价格变化: {last['ask1_price']} -> {new_row['ask1_price']}，跳过 ask1 数量直接比较")

    # order book shift detection (explicit threshold)
    order_shift = False
    shift_reasons = []
    # bid1 price drop (last - new >= threshold)
    if float(last["bid1_price"]) - float(new_row["bid1_price"]) >= ORDER_SHIFT_THR:
        order_shift = True
        shift_reasons.append(f"bid1 下跌 {float(last['bid1_price']) - float(new_row['bid1_price']):.6f}")
    # ask1 price rise (new - last >= threshold)
    if float(new_row["ask1_price"]) - float(last["ask1_price"]) >= ORDER_SHIFT_THR:
        order_shift = True
        shift_reasons.append(f"ask1 上涨 {float(new_row['ask1_price']) - float(last['ask1_price']):.6f}")

    # 保存新行（无论是否报警）
    save_symbol_row(symbol, new_row)

    # 汇总触发
    if price_trigger or qty_trigger or order_shift:
        lines = []
        lines.append(f"【Binance 盘口告警】 {symbol}")
        lines.append(f"时间(UTC): {new_row['timestamp']}")
        lines.append(f"当前 mid: {mid_new:.6f}  上次 mid: {mid_last:.6f}  差: {mid_diff:.6f}")
        lines.append(f"当前 price (ticker): {new_row['price']}")

        lines.append(f"bid1: price={new_row['bid1_price']} qty={new_row['bid1_qty']}  |  上次 price={last['bid1_price']} qty={last['bid1_qty']}")
        lines.append(f"ask1: price={new_row['ask1_price']} qty={new_row['ask1_qty']}  |  上次 price={last['ask1_price']} qty={last['ask1_qty']}")

        reasons = []
        if price_trigger:
            reasons.append(f"价格偏离超过阈值 ({PRICE_DIFF_THRESHOLD})")
        if qty_trigger:
            reasons += qty_details
        if order_shift:
            reasons += shift_reasons

        lines.append("触发原因: " + "；".join(reasons))
        return "\n".join(lines)
    return None

def get_gc_and_eur_prices():
    """获取 GC!（COMEX）和 EUR (TV) 以及 Binance 的 PAXG / EURUSDT"""
    try:
        # TV
        df_gc = tv.get_hist(symbol=TV_GC, exchange='COMEX', interval=Interval.in_1_minute, n_bars=1)
        gc_price = float(df_gc.close.iloc[-1])
        df_eur = tv.get_hist(symbol=TV_EUR, exchange='FX_IDC', interval=Interval.in_1_minute, n_bars=1)
        eur_tv = float(df_eur.close.iloc[-1])

        # Binance
        paxg = float(requests.get(f"{BINANCE_BASE}/ticker/price", params={"symbol": "PAXGUSDT"}, timeout=10).json()["price"])
        eur_usdt = float(requests.get(f"{BINANCE_BASE}/ticker/price", params={"symbol": "EURUSDT"}, timeout=10).json()["price"])

        return {"gc": gc_price, "eur_tv": eur_tv, "paxg": paxg, "eur_usdt": eur_usdt}
    except Exception as e:
        logger.exception("获取 GC/EUR/PAXG/EURUSDT 失败: %s", e)
        return None

def save_spread_row(row):
    path = os.path.join(DATA_DIR, "spreads.csv")
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df_prev = pd.read_csv(path)
        df_out = pd.concat([df_prev, df], ignore_index=True)
    else:
        df_out = df
    df_out.to_csv(path, index=False)

def check_spreads_and_alert():
    """比较 paxg vs gc, eur_usdt vs eur_tv；记录并根据阈值触发告警"""
    p = get_gc_and_eur_prices()
    if not p:
        return None
    gc = p["gc"]; paxg = p["paxg"]
    eur_tv = p["eur_tv"]; eur_usdt = p["eur_usdt"]

    gc_spread = abs(paxg - gc) / gc if gc != 0 else 0.0
    eur_spread = abs(eur_usdt - eur_tv) / eur_tv if eur_tv != 0 else 0.0

    row = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "gc": gc, "paxg": paxg, "gc_spread": gc_spread,
        "eur_tv": eur_tv, "eur_usdt": eur_usdt, "eur_spread": eur_spread
    }
    save_spread_row(row)

    reasons = []
    if gc_spread > GOLD_THR:
        reasons.append(f"PAXG vs GC 价差 {gc_spread:.2%} > {GOLD_THR:.2%} (GC={gc}, PAXG={paxg})")
    if eur_spread > EUR_THR:
        reasons.append(f"EURUSDT vs EUR(TV) 价差 {eur_spread:.2%} > {EUR_THR:.2%} (EUR(TV)={eur_tv}, EURUSDT={eur_usdt})")

    if reasons:
        lines = ["【跨市场价差告警】", f"时间(UTC): {row['timestamp']}"]
        lines += reasons
        return "\n".join(lines)
    return None

# ----------------- 钉钉告警（文本 + 图片） -----------------
def send_ding_text(msg):
    if not DING_WEBHOOK:
        logger.warning("未配置 DING_WEBHOOK，无法发送告警")
        return False
    headers = {"Content-Type": "application/json"}
    payload = {"msgtype": "text", "text": {"content": msg}}
    try:
        r = requests.post(DING_WEBHOOK, data=json.dumps(payload), headers=headers, timeout=10)
        r.raise_for_status()
        logger.info("钉钉文本告警发送成功")
        return True
    except Exception as e:
        logger.exception("钉钉文本告警发送失败: %s", e)
        return False

def send_ding_image(img_path, text=None):
    """
    钉钉机器人 image 消息需要 base64 + md5(image_binary)
    payload: {"msgtype":"image","image":{"base64":"...","md5":"..."}}
    """
    if not DING_WEBHOOK:
        logger.warning("未配置 DING_WEBHOOK，无法发送图片告警")
        return False
    try:
        with open(img_path, "rb") as f:
            img_bin = f.read()
        b64 = base64.b64encode(img_bin).decode()
        md5_hex = hashlib.md5(img_bin).hexdigest()
        payload = {"msgtype": "image", "image": {"base64": b64, "md5": md5_hex}}
        headers = {"Content-Type": "application/json"}
        r = requests.post(DING_WEBHOOK, data=json.dumps(payload), headers=headers, timeout=15)
        r.raise_for_status()
        logger.info("钉钉图片告警发送成功")
        if text:
            send_ding_text(text)
        return True
    except Exception as e:
        logger.exception("钉钉图片告警发送失败: %s", e)
        return False

# ----------------- 报表生成 -----------------
def generate_daily_report():
    """生成双子图：上 - 各symbol价格；下 - 两个价差曲线"""
    try:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        # 上：symbols price
        for s in SYMBOLS:
            path = os.path.join(DATA_DIR, f"{s}.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                axes[0].plot(df["timestamp"], df["mid"], label=f"{s} mid")
        axes[0].set_title("Binance mid price (bid1+ask1)/2")
        axes[0].legend()
        axes[0].grid(True)

        # 下：spreads
        spath = os.path.join(DATA_DIR, "spreads.csv")
        if os.path.exists(spath):
            df_sp = pd.read_csv(spath)
            df_sp["timestamp"] = pd.to_datetime(df_sp["timestamp"])
            axes[1].plot(df_sp["timestamp"], df_sp["gc_spread"], label="PAXG vs GC!", linestyle='-')
            axes[1].plot(df_sp["timestamp"], df_sp["eur_spread"], label="EURUSDT vs EUR(TV)", linestyle='-')
            axes[1].axhline(GOLD_THR, linestyle="--", label="spread threshold", alpha=0.5)
            axes[1].legend()
            axes[1].grid(True)

        plt.tight_layout()
        fname = os.path.join(REPORTS_DIR, f"daily_report_{datetime.utcnow().strftime('%Y-%m-%d')}.png")
        fig.savefig(fname)
        plt.close(fig)

        # 计算关键指标以文本形式返回
        key_metrics = []
        if os.path.exists(spath):
            df_sp = pd.read_csv(spath)
            if not df_sp.empty:
                key_metrics.append(f"GC spread 最大值: {df_sp['gc_spread'].max():.2%}")
                key_metrics.append(f"EUR spread 最大值: {df_sp['eur_spread'].max():.2%}")
                key_metrics.append(f"GC spread 平均值: {df_sp['gc_spread'].mean():.2%}")
                key_metrics.append(f"EUR spread 平均值: {df_sp['eur_spread'].mean():.2%}")
        text = "今日关键指标：\n" + ("\n".join(key_metrics) if key_metrics else "无历史价差数据")
        return fname, text
    except Exception as e:
        logger.exception("生成日报失败: %s", e)
        return None, None

# ----------------- 主流程 -----------------
def main():
    alerts = []

    # 1) Binance 每个 symbol 检查
    for s in SYMBOLS:
        new = get_binance_depth(s)
        if not new:
            continue
        alert = compare_and_alert_symbol(s, new)
        if alert:
            alerts.append(alert)

    # 2) spreads 检查
    sp_alert = check_spreads_and_alert()
    if sp_alert:
        alerts.append(sp_alert)

    # 3) 汇总并发送文本告警（如果有）
    if alerts:
        msg = "⚠️ 异常告警 - 汇总\n时间(UTC): " + datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + "\n\n" + "\n\n".join(alerts)
        logger.warning(msg)
        send_ding_text(msg)
    else:
        logger.info(f"{now_str()} - 无异常")

    # 4) 若为每天指定时间（北京时间），生成日报并发送（可在 CI 中每次都执行，会在时间窗口触发）
    if DAILY_REPORT:
        # 计算当前北京时间（UTC+8）
        now_utc8 = datetime.utcnow() + timedelta(hours=8)
        target = datetime.strptime(DAILY_TIME, "%H:%M").time()
        # 在 07:45 - 08:15 (UTC+8) 时间窗口内生成
        lower = (datetime.combine(now_utc8.date(), target) - timedelta(minutes=15)).time()
        upper = (datetime.combine(now_utc8.date(), target) + timedelta(minutes=15)).time()
        if lower <= now_utc8.time() <= upper:
            report_file, metrics_text = generate_daily_report()
            if report_file:
                send_ding_image(report_file, text=metrics_text)

if __name__ == "__main__":
    main()
