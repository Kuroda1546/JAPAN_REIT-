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

def whole_number_prices(frame, columns):
    """表示用データの株価列を100円単位に四捨五入する（分析元データは変更しない）。"""
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = (pd.to_numeric(result[column], errors="coerce") / 100).round(0).mul(100).astype("Int64")
    return result

PRICE_COLUMNS = ["現在値", "最高値", "平均値", "最安値", "平均値・最安値の中間値", "最安値からの差額(円)", "High", "Low", "Close"]

def safe_csv_upload(uploaded):
    if uploaded is None:
        return None
    raw = uploaded.getvalue()
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(io.BytesIO(raw), dtype={"code": str, "コード": str}, encoding=enc)
        except Exception:
            pass
    return None

def add_market_metrics(summary_frame, price_map):
    metrics = []
    for ticker, frame in price_map.items():
        close = frame["Close"].astype(float).dropna()
        daily = close.pct_change().dropna()
        current = float(close.iloc[-1])
        def period_return(n):
            return (current / float(close.iloc[-n-1]) - 1) * 100 if len(close) > n else np.nan
        dd = close / close.cummax() - 1
        downside = daily[daily < 0]
        low = float(frame["Low"].min())
        metrics.append({
            "ticker": ticker,
            "1ヶ月騰落率(%)": period_return(21),
            "3ヶ月騰落率(%)": period_return(63),
            "6ヶ月騰落率(%)": period_return(126),
            "12ヶ月騰落率(%)": period_return(252),
            "最大ドローダウン(%)": float(dd.min() * 100) if len(dd) else np.nan,
            "最安値からの回復率(%)": (current / low - 1) * 100 if low else np.nan,
            "下方ボラティリティ(%)": float(downside.std() * np.sqrt(252) * 100) if len(downside) > 1 else np.nan,
            "日次VaR 95%(年率換算・%)": float(-daily.quantile(0.05) * np.sqrt(252) * 100) if len(daily) else np.nan
        })
    return summary_frame.merge(pd.DataFrame(metrics), on="ticker", how="left")

def percentile_score(series, higher_is_better=True):
    values = pd.to_numeric(series, errors="coerce")
    return values.rank(pct=True, ascending=higher_is_better) * 100



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
            "平均値・最安値の中間値": round(((avg + low) / 2) / 100) * 100,
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
summary = add_market_metrics(summary, prices)
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

tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8 = st.tabs(["市場全体","銘柄ランキング","個別銘柄分析","分配金・NAV","下落・リスク","仮想ポートフォリオ","バックテスト","金利・市況・割安発掘"])
with tab1:
    st.subheader("銘柄別・期間騰落率")
    chart_data = summary.sort_values("期間騰落率(%)", ascending=False).set_index("銘柄名")[["期間騰落率(%)"]]
    st.bar_chart(chart_data.round(0), use_container_width=True)
    describe_table = summary.drop(columns=["ticker"]).describe(include="all").T
    for price_label in PRICE_COLUMNS:
        if price_label in describe_table.index:
            numeric_values = pd.to_numeric(describe_table.loc[price_label], errors="coerce")
            describe_table.loc[price_label] = (numeric_values / 100).round(0) * 100
    st.dataframe(describe_table.style.format(precision=0, thousands=",", na_rep="—"), use_container_width=True)

    st.divider()
    st.subheader("直近1ヶ月間に中間値を下回った銘柄")
    recent_rows = []
    for ticker, price_df in prices.items():
        if ticker not in set(summary["ticker"]):
            continue
        summary_row = summary.loc[summary["ticker"] == ticker].iloc[0]
        latest_date = price_df.index.max()
        one_month_ago = latest_date - pd.DateOffset(months=1)
        recent_df = price_df.loc[price_df.index > one_month_ago]
        if recent_df.empty:
            continue
        midpoint = float(summary_row["平均値・最安値の中間値"])
        below_days = recent_df[recent_df["Close"] < midpoint]
        if not below_days.empty:
            min_close_date = below_days["Close"].idxmin()
            recent_rows.append({
                "コード": summary_row["コード"],
                "銘柄名": summary_row["銘柄名"],
                "現在値": float(price_df["Close"].iloc[-1]),
                "平均値・最安値の中間値": midpoint,
                "直近1ヶ月の最安終値": float(recent_df["Close"].min()),
                "中間値を下回った日数": len(below_days),
                "中間値を下回った最終日": below_days.index.max().strftime("%Y-%m-%d"),
                "直近1ヶ月の最安終値日": min_close_date.strftime("%Y-%m-%d"),
                "ticker": ticker
            })
    recent_cols = ["コード", "銘柄名", "現在値", "平均値・最安値の中間値", "直近1ヶ月の最安終値", "中間値を下回った日数", "中間値を下回った最終日", "直近1ヶ月の最安終値日"]
    if recent_rows:
        recent_screen = pd.DataFrame(recent_rows).sort_values(["中間値を下回った日数", "直近1ヶ月の最安終値"], ascending=[False, True])
        st.caption("判定条件：選択中の分析期間で算出した中間値（期間平均終値＋期間最安値）÷2を、直近1ヶ月のいずれかの営業日の終値が下回った銘柄。最新取得日を基準に過去1ヶ月を判定します。")
        recent_display = whole_number_prices(recent_screen[recent_cols], PRICE_COLUMNS)
        st.dataframe(recent_display.style.format(precision=0, thousands=",", na_rep="—"), use_container_width=True, hide_index=True)
        st.download_button("直近1ヶ月・中間値割れ銘柄CSVを保存", recent_screen[recent_cols].to_csv(index=False).encode("utf-8-sig"), "japan_reit_recent_month_below_midpoint.csv", "text/csv")
    else:
        st.info("直近1ヶ月間に中間値を下回った銘柄はありません（または該当期間のデータがありません）。")

    st.divider()
    st.subheader("平均値・最安値の中間値を下回る銘柄")
    below_mid = summary[summary["現在値"] < summary["平均値・最安値の中間値"]].copy()
    below_mid = below_mid.sort_values("最安値からの乖離(%)", ascending=True)
    st.caption("判定条件：現在値（直近終値）＜100円単位に四捨五入した中間値。中間値＝（期間平均値＋期間最安値）÷2。")
    mid_cols = ["コード", "銘柄名", "現在値", "平均値", "最安値", "平均値・最安値の中間値", "最安値からの差額(円)", "最安値からの乖離(%)"]
    if below_mid.empty:
        st.info("現在、条件に該当する銘柄はありません。")
    else:
        st.dataframe(
            whole_number_prices(below_mid[mid_cols], PRICE_COLUMNS).style.format(precision=0, thousands=",", na_rep="—").format({"最安値からの乖離(%)": "{:.0f}%"}),
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
        whole_number_prices(nearest_low[low_cols], PRICE_COLUMNS).style.format(precision=0, thousands=",", na_rep="—").format({"最安値からの乖離(%)": "{:.0f}%"}),
        use_container_width=True, hide_index=True
    )
with tab2:
    sort_col = st.selectbox("ランキング基準", ["平均値・最安値の中間値","現在値","最高値","平均値","最安値","期間騰落率(%)","最高値からの下落率(%)","年率換算ボラティリティ(%)"])
    ascending = st.checkbox("昇順（小さい順）", value=False)
    ranked = summary.sort_values(sort_col, ascending=ascending)
    columns = ["コード","銘柄名","現在値","最高値","平均値","最安値","平均値・最安値の中間値","期間騰落率(%)","最高値からの下落率(%)","最安値からの差額(円)","最安値からの乖離(%)","データ日数","最終データ日"]
    ranked_display = whole_number_prices(ranked[columns], PRICE_COLUMNS)
    st.dataframe(ranked_display.style.format(precision=0, thousands=",", na_rep="—").format({c:"{:.0f}%" for c in ["期間騰落率(%)","最高値からの下落率(%)","最安値からの乖離(%)"]}), use_container_width=True, hide_index=True)
    st.download_button("ランキングCSVを保存", ranked[columns].to_csv(index=False).encode("utf-8-sig"), "japan_reit_ranking.csv", "text/csv")
with tab3:
    ticker = st.selectbox("銘柄", summary.ticker.tolist(), format_func=lambda x: f"{summary.loc[summary.ticker==x,'コード'].iloc[0]}｜{summary.loc[summary.ticker==x,'銘柄名'].iloc[0]}")
    df = prices[ticker]
    row = summary.loc[summary.ticker==ticker].iloc[0]
    x1,x2,x3,x4 = st.columns(4)
    x1.metric("最高値", f"{round(row['最高値'] / 100) * 100:,.0f}円")
    x2.metric("平均値", f"{round(row['平均値'] / 100) * 100:,.0f}円")
    x3.metric("最安値", f"{round(row['最安値'] / 100) * 100:,.0f}円")
    x4.metric("平均値・最安値の中間値", f"{round(row['平均値・最安値の中間値'] / 100) * 100:,.0f}円")
    st.caption(f"最高値日：{row['最高値日']}｜最安値日：{row['最安値日']}")
    chart_prices = whole_number_prices(df[["High","Low","Close"]], PRICE_COLUMNS)
    st.line_chart(chart_prices)
    detail = row.drop(labels=["ticker"]).to_frame("値")
    for label in PRICE_COLUMNS:
        if label in detail.index and pd.notna(detail.loc[label, "値"]):
            detail.loc[label, "値"] = f"{round(float(detail.loc[label, '値']) / 100) * 100:,.0f}"
    st.dataframe(detail.style.format(precision=0, thousands=",", na_rep="—"), use_container_width=True)


with tab4:
    st.subheader("分配金利回り・NAV倍率ランキング")
    st.info("分配金・NAV・LTVは株価データからは算出できません。決算資料等で確認した値をCSVで読み込んでください。未入力値は空欄のまま扱います。")
    template = pd.DataFrame(columns=["code","annual_distribution_yen","nav_per_unit_yen","ltv_pct","property_type"])
    st.download_button("ファンダメンタル入力CSVテンプレート", template.to_csv(index=False).encode("utf-8-sig"), "reit_fundamentals_template.csv", "text/csv")
    fund_file = st.file_uploader("J-REITファンダメンタルCSVを読み込む", type=["csv"], key="fundamentals_csv")
    fundamentals = safe_csv_upload(fund_file)
    fundamental_view = None
    if fundamentals is not None and not fundamentals.empty:
        code_col = "code" if "code" in fundamentals.columns else ("コード" if "コード" in fundamentals.columns else None)
        if code_col is None:
            st.error("CSVに code または コード 列が必要です。")
        else:
            fundamentals[code_col] = fundamentals[code_col].astype(str).str.replace(r"\.0$", "", regex=True)
            fundamentals = fundamentals.rename(columns={code_col:"コード"})
            for col in ["annual_distribution_yen","nav_per_unit_yen","ltv_pct","property_type"]:
                if col not in fundamentals.columns:
                    fundamentals[col] = np.nan
            fundamentals["annual_distribution_yen"] = pd.to_numeric(fundamentals["annual_distribution_yen"], errors="coerce")
            fundamentals["nav_per_unit_yen"] = pd.to_numeric(fundamentals["nav_per_unit_yen"], errors="coerce")
            fundamentals["ltv_pct"] = pd.to_numeric(fundamentals["ltv_pct"], errors="coerce")
            fundamental_view = summary.merge(fundamentals, on="コード", how="left")
            fundamental_view["予想分配金利回り(%)"] = fundamental_view["annual_distribution_yen"] / fundamental_view["現在値"] * 100
            fundamental_view["NAV倍率"] = fundamental_view["現在値"] / fundamental_view["nav_per_unit_yen"]
            y1,y2 = st.columns(2)
            with y1:
                st.markdown("**予想分配金利回りランキング**")
                div_rank = fundamental_view.dropna(subset=["予想分配金利回り(%)"]).sort_values("予想分配金利回り(%)", ascending=False)
                div_cols = ["コード","銘柄名","現在値","annual_distribution_yen","予想分配金利回り(%)","ltv_pct"]
                st.dataframe(whole_number_prices(div_rank[div_cols], PRICE_COLUMNS).rename(columns={"annual_distribution_yen":"年間予想分配金(円)","ltv_pct":"LTV(%)"}).style.format(precision=2, thousands=",", na_rep="—"), use_container_width=True, hide_index=True)
                if div_rank.empty: st.warning("分配金データが入力されていません。")
            with y2:
                st.markdown("**NAV倍率ランキング（低い順）**")
                nav_rank = fundamental_view.dropna(subset=["NAV倍率"]).sort_values("NAV倍率")
                nav_cols = ["コード","銘柄名","現在値","nav_per_unit_yen","NAV倍率","ltv_pct"]
                nav_display = whole_number_prices(nav_rank[nav_cols], PRICE_COLUMNS + ["nav_per_unit_yen"]).rename(columns={"nav_per_unit_yen":"1口当たりNAV(円)","ltv_pct":"LTV(%)"})
                st.dataframe(nav_display.style.format({"NAV倍率":"{:.2f}倍","LTV(%)":"{:.1f}%"}, na_rep="—"), use_container_width=True, hide_index=True)
                if nav_rank.empty: st.warning("NAVデータが入力されていません。")
            st.caption("入力値の基準日・予想/実績の別をCSV作成時に必ず確認してください。")
    else:
        st.warning("CSV未読み込みです。テンプレートに最新の年間予想分配金、1口当たりNAV、LTVを入力して読み込んでください。")

with tab5:
    st.subheader("下落率・回復余地ランキング")
    draw_cols = ["コード","銘柄名","現在値","1ヶ月騰落率(%)","3ヶ月騰落率(%)","6ヶ月騰落率(%)","12ヶ月騰落率(%)","最高値からの下落率(%)","最大ドローダウン(%)","最安値からの回復率(%)"]
    draw_rank = summary.sort_values("3ヶ月騰落率(%)", ascending=True)
    st.dataframe(whole_number_prices(draw_rank[draw_cols], PRICE_COLUMNS).style.format({c:"{:.1f}%" for c in draw_cols if "(%)" in c}, na_rep="—"), use_container_width=True, hide_index=True)
    st.download_button("下落・回復ランキングCSV", draw_rank[draw_cols].to_csv(index=False).encode("utf-8-sig"), "reit_drawdown_recovery.csv", "text/csv")
    st.divider()
    st.subheader("リスク分析ランキング")
    risk_cols = ["コード","銘柄名","年率換算ボラティリティ(%)","下方ボラティリティ(%)","最大ドローダウン(%)","日次VaR 95%(年率換算・%)"]
    risk_rank = summary.sort_values("年率換算ボラティリティ(%)", ascending=True)
    st.dataframe(risk_rank[risk_cols].style.format({c:"{:.2f}%" for c in risk_cols if "%" in c}, na_rep="—"), use_container_width=True, hide_index=True)
    st.caption("VaRは日次リターンの5%分位点を√252倍した参考値です。将来の最大損失を保証するものではありません。")

with tab6:
    st.subheader("仮想ポートフォリオ")
    st.caption("仮想保有口数・平均取得価格を入力して試算します。データはブラウザのセッション中のみ保持されます。")
    portfolio_seed = pd.DataFrame([{"コード":str(r["コード"]),"口数":0,"平均取得価格(円)":0} for _,r in summary.head(5).iterrows()])
    if "reit_portfolio" not in st.session_state:
        st.session_state["reit_portfolio"] = portfolio_seed
    portfolio = st.data_editor(st.session_state["reit_portfolio"], num_rows="dynamic", use_container_width=True, key="reit_portfolio_editor")
    st.session_state["reit_portfolio"] = portfolio
    portfolio["コード"] = portfolio["コード"].astype(str)
    portfolio["口数"] = pd.to_numeric(portfolio["口数"], errors="coerce").fillna(0)
    portfolio["平均取得価格(円)"] = pd.to_numeric(portfolio["平均取得価格(円)"], errors="coerce").fillna(0)
    portfolio = portfolio.merge(summary[["コード","銘柄名","現在値"]], on="コード", how="left")
    portfolio["投資元本(円)"] = portfolio["口数"] * portfolio["平均取得価格(円)"]
    portfolio["評価額(円)"] = portfolio["口数"] * portfolio["現在値"]
    portfolio["含み損益(円)"] = portfolio["評価額(円)"] - portfolio["投資元本(円)"]
    portfolio["損益率(%)"] = np.where(portfolio["投資元本(円)"]>0, portfolio["含み損益(円)"] / portfolio["投資元本(円)"] * 100, np.nan)
    pcols=["コード","銘柄名","口数","平均取得価格(円)","現在値","投資元本(円)","評価額(円)","含み損益(円)","損益率(%)"]
    st.dataframe(whole_number_prices(portfolio[pcols], PRICE_COLUMNS + ["平均取得価格(円)"]).style.format({"損益率(%)":"{:.2f}%"}, na_rep="—"), use_container_width=True, hide_index=True)
    p1,p2,p3=st.columns(3)
    p1.metric("投資元本", f"{portfolio['投資元本(円)'].sum():,.0f}円")
    p2.metric("評価額", f"{portfolio['評価額(円)'].sum():,.0f}円")
    p3.metric("含み損益", f"{portfolio['含み損益(円)'].sum():+,.0f}円")

with tab7:
    st.subheader("過去データによるバックテスト")
    st.caption("過去252営業日の終値平均と安値から中間値を毎日算出し、終値が中間値を下回った日に仮想購入。未来情報を使わないローリング方式です。")
    horizon = st.selectbox("検証期間", ["1ヶ月後","3ヶ月後","6ヶ月後"], key="backtest_horizon")
    forward_days = {"1ヶ月後":21,"3ヶ月後":63,"6ヶ月後":126}[horizon]
    signals=[]
    for ticker, frame in prices.items():
        close=frame["Close"].astype(float)
        rolling_avg=close.rolling(252, min_periods=60).mean()
        rolling_low=frame["Low"].astype(float).rolling(252, min_periods=60).min()
        rolling_mid=((rolling_avg+rolling_low)/2/100).round()*100
        future=close.shift(-forward_days)
        signal=(close<rolling_mid).shift(1).fillna(False) & future.notna()
        for dt in close.index[signal.fillna(False)]:
            base=float(close.loc[dt]); later=float(future.loc[dt])
            signals.append({"コード":str(ticker.replace(".T","")),"銘柄名":master.loc[master["ticker"]==ticker,"name"].iloc[0],"購入日":dt.strftime("%Y-%m-%d"),"購入価格":base,"検証後価格":later,"騰落率(%)":(later/base-1)*100})
    bt=pd.DataFrame(signals)
    if not bt.empty:
        win=(bt["騰落率(%)"]>0).mean()*100
        b1,b2,b3=st.columns(3)
        b1.metric("検証シグナル数",f"{len(bt):,}")
        b2.metric("プラスリターン率",f"{win:.1f}%")
        b3.metric("平均騰落率",f"{bt['騰落率(%)'].mean():+.2f}%")
        bt_display = whole_number_prices(bt, ["購入価格","検証後価格"])
        # Streamlit CloudでStylerとnullable Int64の組合せによる描画例外を避ける
        for col in ["購入価格", "検証後価格"]:
            bt_display[col] = pd.to_numeric(bt_display[col], errors="coerce").fillna(0).astype("int64")
        bt_display["騰落率(%)"] = pd.to_numeric(bt_display["騰落率(%)"], errors="coerce").round(2)
        st.dataframe(
            bt_display.sort_values("購入日", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "購入価格": st.column_config.NumberColumn("購入価格", format="%d円"),
                "検証後価格": st.column_config.NumberColumn("検証後価格", format="%d円"),
                "騰落率(%)": st.column_config.NumberColumn("騰落率(%)", format="%.2f%%"),
            },
        )
        st.download_button("バックテストCSV",bt.to_csv(index=False).encode("utf-8-sig"),"reit_backtest.csv","text/csv")
    else:
        st.warning("バックテスト可能なシグナルがありません。分析期間を長くして価格データを取得してください。")
    st.warning("単純検証です。売買手数料、分配金、税金、流動性、重複購入、銘柄入替を完全には反映していません。")

with tab8:
    st.subheader("金利・不動産市況分析")
    st.info("金利・不動産市況データはCSV読み込み方式です。")
    macro_template=pd.DataFrame(columns=["date","japan_10y_yield_pct","reit_index","property_price_index"])
    st.download_button("金利・市況CSVテンプレート",macro_template.to_csv(index=False).encode("utf-8-sig"),"reit_macro_template.csv","text/csv")
    macro_file=st.file_uploader("金利・不動産市況CSVを読み込む",type=["csv"],key="macro_csv")
    macro=safe_csv_upload(macro_file)
    if macro is not None:
        if "date" not in macro.columns:
            st.error("CSVに date 列が必要です。")
        else:
            macro["date"]=pd.to_datetime(macro["date"],errors="coerce")
            for col in ["japan_10y_yield_pct","reit_index","property_price_index"]:
                if col in macro.columns: macro[col]=pd.to_numeric(macro[col],errors="coerce")
            macro=macro.dropna(subset=["date"]).sort_values("date").set_index("date")
            chart_cols=[c for c in ["japan_10y_yield_pct","reit_index","property_price_index"] if c in macro.columns]
            if chart_cols: st.line_chart(macro[chart_cols],use_container_width=True)
            st.dataframe(macro,use_container_width=True)
    st.divider()
    st.subheader("J-REIT割安銘柄自動発掘システム")
    st.caption("株価ベースの割安度・反発・リスクをスコア化。分配金利回りとNAV倍率はファンダメンタルCSV読込時のみ評価します。")
    score=summary[["コード","銘柄名","ticker","現在値","平均値・最安値の中間値","1ヶ月騰落率(%)","3ヶ月騰落率(%)","最安値からの回復率(%)","年率換算ボラティリティ(%)","最大ドローダウン(%)"]].copy()
    score["割安度スコア"]=percentile_score(score["平均値・最安値の中間値"] / score["現在値"],True)
    score["反発モメンタムスコア"]=percentile_score(score["1ヶ月騰落率(%)"],True)
    score["回復余地スコア"]=percentile_score(score["最安値からの回復率(%)"],False)
    score["低リスクスコア"]=percentile_score(score["年率換算ボラティリティ(%)"],False)
    score["総合スコア"]=score[["割安度スコア","反発モメンタムスコア","回復余地スコア","低リスクスコア"]].mean(axis=1)
    score["評価"]=pd.cut(score["総合スコア"],bins=[-1,20,40,60,80,101],labels=["要注意","低評価","中立","有望候補","高評価"])
    if fundamental_view is not None:
        score=score.merge(fundamental_view[["コード","予想分配金利回り(%)","NAV倍率"]],on="コード",how="left")
        score["分配金スコア"]=percentile_score(score["予想分配金利回り(%)"],True)
        score["NAV割安スコア"]=percentile_score(1/score["NAV倍率"],True)
        score["総合スコア"]=score[["割安度スコア","反発モメンタムスコア","回復余地スコア","低リスクスコア","分配金スコア","NAV割安スコア"]].mean(axis=1)
        score["評価"]=pd.cut(score["総合スコア"],bins=[-1,20,40,60,80,101],labels=["要注意","低評価","中立","有望候補","高評価"])
    score=score.sort_values("総合スコア",ascending=False)
    score_cols=[c for c in ["コード","銘柄名","現在値","総合スコア","評価","割安度スコア","反発モメンタムスコア","回復余地スコア","低リスクスコア","予想分配金利回り(%)","NAV倍率"] if c in score.columns]
    score_fmt={c:"{:.1f}" for c in score_cols if "スコア" in c or c=="予想分配金利回り(%)"}
    if "NAV倍率" in score_cols: score_fmt["NAV倍率"]="{:.2f}倍"
    st.dataframe(whole_number_prices(score[score_cols],PRICE_COLUMNS).style.format(score_fmt,na_rep="—"),use_container_width=True,hide_index=True)
    st.download_button("割安銘柄スコアCSV",score[score_cols].to_csv(index=False).encode("utf-8-sig"),"reit_undervalued_score.csv","text/csv")
    st.caption("総合スコアは選択銘柄内の相対評価であり、投資推奨・将来収益の予測ではありません。")

with st.expander("取得できなかった銘柄・データの注意", expanded=bool(errors)):
    if errors:
        st.dataframe(pd.DataFrame([{"コード":k.replace(".T",""),"理由":v} for k,v in errors.items()]), use_container_width=True, hide_index=True)
    st.markdown("Yahoo Financeの配信状況・銘柄コード・取得制限等により一部データが取得できない場合があります。価格データは非公式のyfinance経由で取得し、分配金込みのトータルリターンではありません。")
st.caption("データ取得：Yahoo Finance（yfinance経由・非公式）｜初期銘柄一覧：J-REIT 58銘柄｜投資判断はご自身で行ってください。")
