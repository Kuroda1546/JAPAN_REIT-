from __future__ import annotations
from datetime import date, timedelta
from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="JAPAN-REIT アナリスト", page_icon="🏢", layout="wide")
st.title("🏢 JAPAN-REIT アナリスト")
st.caption("東証上場J-REITの投資口価格を横断分析するツール")

@st.cache_data(ttl=21600, show_spinner=False)
def load_master():
    df = pd.read_csv(Path(__file__).with_name("reit_master.csv"), dtype={"code": str})
    df["code"] = df["code"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["ticker"] = df["code"] + ".T"
    return df

@st.cache_data(ttl=21600, show_spinner=False)
def get_prices(tickers, start, end):
    result, errors = {}, {}
    for ticker in tickers:
        try:
            kwargs = {"auto_adjust": False, "progress": False, "threads": False}
            if start == "MAX":
                kwargs["period"] = "max"
            else:
                kwargs.update(start=start, end=end)
            df = yf.download(ticker, **kwargs)
            if df is None or df.empty:
                errors[ticker] = "価格データなし"
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            needed = [c for c in ["High", "Low", "Close"] if c in df.columns]
            if len(needed) != 3:
                errors[ticker] = "高値・安値・終値の列がありません"
                continue
            part = df[needed].copy()
            part.columns = ["High", "Low", "Close"]
            part = part.dropna()
            if not part.empty:
                result[ticker] = part
            else:
                errors[ticker] = "有効な価格データなし"
        except Exception as exc:
            errors[ticker] = f"{type(exc).__name__}: {exc}"
    return result, errors

def analyze(prices, master):
    names = master.set_index("ticker")["name"].to_dict()
    codes = master.set_index("ticker")["code"].to_dict()
    rows = []
    for ticker, df in prices.items():
        if df.empty: continue
        high, low = float(df.High.max()), float(df.Low.min())
        avg, current = float(df.Close.mean()), float(df.Close.iloc[-1])
        first = float(df.Close.iloc[0])
        vol = df.Close.pct_change().dropna().std()
        rows.append({
            "コード": codes.get(ticker, ticker), "銘柄名": names.get(ticker, ticker),
            "ticker": ticker, "現在値": current, "最高値": high,
            "最高値日": df.High.idxmax().strftime("%Y-%m-%d"),
            "平均値": avg, "最安値": low,
            "最安値日": df.Low.idxmin().strftime("%Y-%m-%d"),
            "平均値・最安値の中間値": (avg + low) / 2,
            "最安値からの差額(円)": current - low,
            "最安値からの乖離(%)": (current / low - 1) * 100 if low else np.nan,
            "期間騰落率(%)": (current / first - 1) * 100 if first else np.nan,
            "最高値からの下落率(%)": (current / high - 1) * 100 if high else np.nan,
            "現在値と平均値の乖離(%)": (current / avg - 1) * 100 if avg else np.nan,
            "年率換算ボラティリティ(%)": float(vol * np.sqrt(252) * 100) if pd.notna(vol) else np.nan,
            "データ日数": len(df), "最終データ日": df.index[-1].strftime("%Y-%m-%d")
        })
    return pd.DataFrame(rows)

master = load_master()
with st.sidebar:
    st.header("分析条件")
    period = st.selectbox("分析期間", ["上場来", "過去10年", "過去5年", "過去3年", "過去1年", "年初来", "期間指定"])
    today = date.today()
    if period == "上場来":
        start, end = "MAX", "MAX"
    else:
        days = {"過去10年":3650, "過去5年":1825, "過去3年":1095, "過去1年":365, "年初来":(today-date(today.year,1,1)).days}
        if period == "期間指定":
            chosen = st.date_input("開始日・終了日", value=(today-timedelta(days=1825), today), max_value=today)
            if isinstance(chosen, tuple) and len(chosen) == 2:
                start_date, end_date = chosen
            else:
                start_date, end_date = today-timedelta(days=1825), today
        else:
            start_date, end_date = today-timedelta(days=days[period]), today
        start, end = start_date.isoformat(), (end_date+timedelta(days=1)).isoformat()
    all_tickers = master.ticker.tolist()
    selected = st.multiselect("対象銘柄", all_tickers, default=all_tickers,
        format_func=lambda x: f"{master.loc[master.ticker==x,'code'].iloc[0]}｜{master.loc[master.ticker==x,'name'].iloc[0]}")
    refresh = st.button("📊 価格データを取得・更新", type="primary", use_container_width=True)

if refresh:
    if not selected:
        st.warning("銘柄を1つ以上選択してください。")
    else:
        get_prices.clear()
        with st.spinner(f"{len(selected)}銘柄の価格データを取得中…"):
            prices, errors = get_prices(tuple(selected), start, end)
        st.session_state["prices"] = prices
        st.session_state["errors"] = errors
        st.session_state["selection"] = tuple(selected)
        st.session_state["period"] = (start, end)

prices = st.session_state.get("prices", {})
errors = st.session_state.get("errors", {})
if not prices:
    st.info("左側で銘柄と期間を選択し、「価格データを取得・更新」を押してください。")
    st.markdown("### このアプリで確認できる指標")
    st.markdown("- 最高値：期間中の日次高値の最大値\n- 平均値：期間中の日次終値の算術平均\n- 最安値：期間中の日次安値の最小値\n- 中間値：(平均値＋最安値)÷2")
    st.stop()

# 選択条件を変えた場合は、現在保持しているデータの範囲内で表示する
prices = {k:v for k,v in prices.items() if k in selected}
summary = analyze(prices, master)
if summary.empty:
    st.error("分析可能なデータがありません。期間や銘柄を変更してください。")
    st.stop()

a,b,c,d = st.columns(4)
a.metric("分析可能銘柄", f"{len(summary)} / {len(selected)}")
b.metric("最新データ日", summary["最終データ日"].max())
c.metric("期間騰落率中央値", f"{summary['期間騰落率(%)'].median():+.0f}%")
d.metric("最高値からの下落率中央値", f"{summary['最高値からの下落率(%)'].median():+.0f}%")
with st.expander("価格統計の計算ルール"):
    st.write("最高値＝期間中の日次高値の最大値。平均値＝日次終値の算術平均。最安値＝日次安値の最小値。中間値＝(平均値＋最安値)÷2。分配金を含まない投資口価格ベースです。")

tab1,tab2,tab3 = st.tabs(["市場全体","銘柄ランキング","個別銘柄分析"])
with tab1:
    st.subheader("銘柄別・期間騰落率")
    chart_data = summary.sort_values("期間騰落率(%)", ascending=False).set_index("銘柄名")[["期間騰落率(%)"]]
    st.bar_chart(chart_data.round(0), use_container_width=True)
    st.dataframe(summary.drop(columns=["ticker"]).describe(include="all").T.style.format(lambda x: f"{x:,.0f}" if isinstance(x, (int, float, np.integer, np.floating)) and pd.notna(x) else x), use_container_width=True)

    st.divider()
    st.subheader("平均値・最安値の中間値を下回る銘柄")
    below_mid = summary[summary["現在値"] < summary["平均値・最安値の中間値"]].copy()
    below_mid = below_mid.sort_values("最安値からの乖離(%)", ascending=True)
    st.caption("判定条件：現在値（直近終値）＜（期間平均値＋期間最安値）÷2。該当銘柄を中間値からの価格差が小さい順に表示します。")
    mid_cols = ["コード", "銘柄名", "現在値", "平均値", "最安値", "平均値・最安値の中間値", "最安値からの差額(円)", "最安値からの乖離(%)"]
    if below_mid.empty:
        st.info("現在、条件に該当する銘柄はありません。")
    else:
        st.dataframe(
            below_mid[mid_cols].style.format({
                c: "{:,.0f}" for c in ["現在値", "平均値", "最安値", "平均値・最安値の中間値", "最安値からの差額(円)"]
            }).format({"最安値からの乖離(%)": "{:.0f}%"}),
            use_container_width=True, hide_index=True
        )
        st.download_button(
            "中間値を下回る銘柄CSVを保存",
            below_mid[mid_cols].to_csv(index=False).encode("utf-8-sig"),
            "japan_reit_below_midpoint.csv", "text/csv"
        )

    st.divider()
    st.subheader("最安値に近い銘柄 TOP 5")
    nearest_low = summary.sort_values("最安値からの乖離(%)", ascending=True).head(5)
    st.caption("判定条件：（現在値－期間最安値）÷期間最安値。最安値からの乖離率が小さい順に5銘柄を表示します。")
    low_cols = ["コード", "銘柄名", "現在値", "最安値", "最安値からの差額(円)", "最安値からの乖離(%)", "最安値日"]
    st.dataframe(
        nearest_low[low_cols].style.format({
            c: "{:,.0f}" for c in ["現在値", "最安値", "最安値からの差額(円)"]
        }).format({"最安値からの乖離(%)": "{:.0f}%"}),
        use_container_width=True, hide_index=True
    )
with tab2:
    sort_col = st.selectbox("ランキング基準", ["平均値・最安値の中間値","現在値","最高値","平均値","最安値","期間騰落率(%)","最高値からの下落率(%)","年率換算ボラティリティ(%)"])
    ascending = st.checkbox("昇順（小さい順）", value=False)
    ranked = summary.sort_values(sort_col, ascending=ascending)
    columns = ["コード","銘柄名","現在値","最高値","平均値","最安値","平均値・最安値の中間値","期間騰落率(%)","最高値からの下落率(%)","最安値からの差額(円)","最安値からの乖離(%)","データ日数","最終データ日"]
    st.dataframe(ranked[columns].style.format({c:"{:,.0f}" for c in ["現在値","最高値","平均値","最安値","平均値・最安値の中間値","最安値からの差額(円)","データ日数"]}).format({c:"{:.0f}%" for c in ["期間騰落率(%)","最高値からの下落率(%)","最安値からの乖離(%)"]}), use_container_width=True, hide_index=True)
    st.download_button("ランキングCSVを保存", ranked[columns].to_csv(index=False).encode("utf-8-sig"), "japan_reit_ranking.csv", "text/csv")
with tab3:
    ticker = st.selectbox("銘柄", summary.ticker.tolist(), format_func=lambda x: f"{summary.loc[summary.ticker==x,'コード'].iloc[0]}｜{summary.loc[summary.ticker==x,'銘柄名'].iloc[0]}")
    df = prices[ticker]
    row = summary.loc[summary.ticker==ticker].iloc[0]
    x1,x2,x3,x4 = st.columns(4)
    x1.metric("最高値", f"{row['最高値']:,.0f}円")
    x2.metric("平均値", f"{row['平均値']:,.0f}円")
    x3.metric("最安値", f"{row['最安値']:,.0f}円")
    x4.metric("平均値・最安値の中間値", f"{row['平均値・最安値の中間値']:,.0f}円")
    st.caption(f"最高値日：{row['最高値日']}｜最安値日：{row['最安値日']}")
    st.line_chart(df[["High","Low","Close"]].round(0))
    st.dataframe(row.drop(labels=["ticker"]).to_frame("値").style.format(lambda x: f"{x:,.0f}" if isinstance(x, (int, float, np.integer, np.floating)) and pd.notna(x) else x), use_container_width=True)

with st.expander("取得できなかった銘柄・データの注意", expanded=bool(errors)):
    if errors:
        st.dataframe(pd.DataFrame([{"コード":k.replace(".T",""),"理由":v} for k,v in errors.items()]), use_container_width=True, hide_index=True)
    st.markdown("Yahoo Financeの配信状況・銘柄コード・取得制限等により一部データが取得できない場合があります。価格データは非公式のyfinance経由で取得し、分配金込みのトータルリターンではありません。")
st.caption("データ取得：Yahoo Finance（yfinance経由・非公式）｜初期銘柄一覧：J-REIT 58銘柄｜投資判断はご自身で行ってください。")
