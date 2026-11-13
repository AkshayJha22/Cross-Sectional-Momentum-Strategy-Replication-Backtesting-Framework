"""
Price Momentum and Trading Volume, 25 years on.
A re-test of Lee & Swaminathan (2000) on modern US equity data.

    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from momvol import data, paper, portfolios as P, stats as S  # noqa: E402

st.set_page_config(page_title="Momentum & Volume, 25 years on", page_icon="📈", layout="wide")

# ---------------------------------------------------------------- style
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
DIVERGING = [[0.0, "#b8302f"], [0.25, "#f0a3a2"], [0.5, "#f0efec"], [0.75, "#86b6ef"], [1.0, "#1c5cab"]]  # red < 0 < blue
FF5 = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def style(fig: go.Figure, height: int = 380, y_title: str | None = None, pct: bool = False) -> go.Figure:
    has_title = bool(fig.layout.title.text)
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=10, t=92 if has_title else 40, b=10),
        title=dict(x=0, xanchor="left", y=0.985, yanchor="top", yref="container",
                   font=dict(size=15, color=INK)),
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color=INK2),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        hoverlabel=dict(bgcolor="white", font_size=12), plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#c9c8c3", zerolinewidth=1, title=y_title,
                     ticksuffix="%" if pct else None)
    return fig


def fmt(x: float, d: int = 1) -> str:
    """Format a number without printing negative zero."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    out = f"{x:.{d}f}"
    return out[1:] if out.startswith("-") and float(out) == 0 else out


# ---------------------------------------------------------------- data (cached)
@st.cache_data(show_spinner=False)
def french():
    ew, vw, f = data.load_french_deciles("equal"), data.load_french_deciles("value"), data.load_french_factors()
    wml = pd.DataFrame({"EW": ew["D10"] - ew["D1"], "VW": vw["D10"] - vw["D1"]})
    return ew, vw, f, wml, data.french_vintage()


@st.cache_resource(show_spinner=False)
def wide():
    return P.to_wide(data.load_panel())


@st.cache_data(show_spinner="Forming 30 momentum x volume portfolios ...")
def jt(J: int, K: int, skip: int, min_price: float) -> pd.DataFrame:
    return P.add_spreads(P.jt_returns(wide(), J, K, skip, min_price))


@st.cache_data(show_spinner=False)
def profile(J: int, skip: int, min_price: float) -> pd.DataFrame:
    return P.formation_profile(wide(), J, skip, min_price)


@st.cache_data(show_spinner=False)
def results(name: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / "results" / f"{name}.parquet")


ew, vw, ff, wml, vintage = french()
ERAS = {
    "1927-1964": ("1927", "1964"),
    "1965-1995 (paper sample)": ("1965", "1995"),
    "1996-2008": ("1996", "2008"),
    "2009-2026": ("2009", "2026"),
}


def era_stats(series: pd.Series) -> pd.DataFrame:
    rows = []
    for name, (a, b) in ERAS.items():
        s = S.summary(series.loc[a:b])
        rows.append({"era": name, **s})
    return pd.DataFrame(rows)


pre = S.summary(wml["EW"].loc["1965":"1995"])
post = S.summary(wml["EW"].loc["1996":])
pre_vw = S.summary(wml["VW"].loc["1965":"1995"])
post_vw = S.summary(wml["VW"].loc["1996":])

# market-state split (Daniel & Moskowitz style): past 24-month market return
mkt = ff["Mkt-RF"] + ff["RF"]
past24 = np.expm1(np.log1p(mkt).rolling(24).sum()).shift(1)


def state_split(a: str, b: str | None) -> dict:
    x = pd.concat([wml["EW"], past24], axis=1, keys=["wml", "p24"]).loc[a:b].dropna()
    up, dn = x.loc[x.p24 > 0, "wml"], x.loc[x.p24 <= 0, "wml"]
    return {"up": 100 * up.mean(), "up_n": len(up), "up_t": S.tstat(up),
            "down": 100 * dn.mean(), "down_n": len(dn), "down_t": S.tstat(dn)}


st_pre, st_post = state_split("1965", "1995"), state_split("1996", None)
worst8 = pd.DataFrame({"wml": wml["EW"], "p24": past24, "mkt": mkt}).loc["1996":].nsmallest(8, "wml")
n_after_down = int((worst8["p24"] <= 0).sum())
n_mkt_up = int((worst8["mkt"] > 0).sum())
post_ew = wml["EW"].loc["1996":]
med_pre, med_post = 100 * wml["EW"].loc["1965":"1995"].median(), 100 * post_ew.median()
post_ex10 = 100 * post_ew.drop(post_ew.nsmallest(10).index).mean()
mid = S.summary(wml["EW"].loc["1996":"2008"])
vw_recent = S.summary(wml["VW"].loc["2010":])

# stock-level defaults (J=6, K=6 is the paper's headline cell)
base = jt(6, 6, 0, 1.0)
alph = results("alphas_j6k6").set_index("series")
ev = results("event_j6")
grid0 = results("grid_skip0")
subp = results("subperiods_j6k6")
val = results("validation")


def m(name: str, df: pd.DataFrame = base) -> float:
    return 100 * df[name].mean()


def t(name: str, df: pd.DataFrame = base) -> float:
    return S.tstat(df[name])


ev1 = ev[ev.year == 1].set_index("series")
ev2 = ev[ev.year == 2].set_index("series")
prof6 = profile(6, 0, 1.0)

# ---------------------------------------------------------------- header
st.title("Price momentum and trading volume, 25 years on")
st.markdown(
    "A re-test of **Lee & Swaminathan (2000), *Price Momentum and Trading Volume*, Journal of Finance** "
    f"on data the authors never saw. Their sample ends in 1995. This page asks which of their findings "
    "still hold in modern US markets, by how much, and why."
)

tab_verdict, tab_mom, tab_vol, tab_why, tab_data = st.tabs(
    ["Verdict", "Momentum, 1927-2026", "Volume replication, 2009-2017", "Why it changed", "Data & method"])

# ================================================================= VERDICT
with tab_verdict:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Winners minus losers, 1965-1995", f"{pre['mean_pct']:.2f}% / month", f"t = {pre['t']:.1f}",
              delta_color="off")
    c2.metric("Same strategy, 1996-2026", f"{post['mean_pct']:.2f}% / month",
              f"{100 * (post['mean_pct'] / pre['mean_pct'] - 1):.0f}% vs paper era, t = {post['t']:.1f}",
              delta_color="normal")
    c3.metric("Early-stage strategy alpha (2009-2017)", f"{alph.loc['Early: R10V1-R1V3', 'ff5mom_alpha_pct']:.2f}% / month",
              f"t = {alph.loc['Early: R10V1-R1V3', 'ff5mom_t']:.1f} after FF5 + momentum", delta_color="off")
    c4.metric("Low-minus-high turnover alpha (2009-2017)", f"{alph.loc['V1-V3', 'ff5_alpha_pct']:.2f}% / month",
              f"t = {alph.loc['V1-V3', 'ff5_t']:.1f} after FF5", delta_color="off")

    st.subheader("Scorecard: the paper's claims against modern data")
    p6 = paper.table2(6, 6).set_index("Portfolio")
    score = pd.DataFrame([
        {"Claim in the paper": "Past winners beat past losers over the next 3-12 months (price momentum)",
         "Paper (1965-95)": f"{paper.TABLE1_R10_R1.loc[6, 6]:.2f}%/mo (J=6, K=6)",
         "Modern evidence": f"CRSP 1996-2026: {post['mean_pct']:.2f}%/mo (t={post['t']:.1f}); "
                            f"stock panel 2009-17: {m('R10-R1'):.2f}%/mo (t={t('R10-R1'):.1f})",
         "Verdict": f"Weaker on average, but mostly because of a few crash months: the median month is "
                    f"{med_post:.2f}% vs {med_pre:.2f}%"},
        {"Claim in the paper": "Momentum is much stronger among high-turnover stocks (R10-R1 in V3 > V1)",
         "Paper (1965-95)": f"V3 {p6.loc['R10-R1', 'V3']:.2f} vs V1 {p6.loc['R10-R1', 'V1']:.2f}%/mo",
         "Modern evidence": f"V3 {m('R10-R1|V3'):.2f} vs V1 {m('R10-R1|V1'):.2f}%/mo "
                            f"(difference t={t('(R10-R1)V3 - (R10-R1)V1'):.1f})",
         "Verdict": "Does not hold: no longer stronger in high-volume stocks, slightly the reverse"},
        {"Claim in the paper": "Low-turnover stocks beat high-turnover stocks, holding momentum fixed",
         "Paper (1965-95)": f"losers: V3-V1 = {p6.loc['R1', 'V3-V1']:.2f}%/mo",
         "Modern evidence": f"losers {m('V3-V1|R1'):.2f}, winners {m('V3-V1|R10'):.2f}%/mo raw; "
                            f"FF5 alpha of V1-V3 {alph.loc['V1-V3', 'ff5_alpha_pct']:.2f}%/mo (t={alph.loc['V1-V3', 'ff5_t']:.1f})",
         "Verdict": "Holds after risk adjustment, but now shows up among winners more than losers"},
        {"Claim in the paper": "Early-stage momentum (low-volume winners, high-volume losers) beats late-stage",
         "Paper (1965-95)": f"Year 1: early {paper.TABLE7.iloc[1]['Year 1']:.1f}% vs late {paper.TABLE7.iloc[2]['Year 1']:.1f}%",
         "Modern evidence": f"Year 1: early {fmt(ev1.loc['Early: R10V1-R1V3', 'mean_pct'])}% vs late "
                            f"{fmt(ev1.loc['Late: R10V3-R1V1', 'mean_pct'])}% (NW t {fmt(ev1.loc['Early: R10V1-R1V3', 'nw_t'])} vs "
                            f"{fmt(ev1.loc['Late: R10V3-R1V1', 'nw_t'])})",
         "Verdict": "Holds: the strongest surviving result"},
        {"Claim in the paper": "Momentum profits reverse at long horizons, faster for late-stage portfolios",
         "Paper (1965-95)": f"late-stage Year 2: {paper.TABLE7.iloc[2]['Year 2']:.1f}%",
         "Modern evidence": f"late-stage Year 2: {ev2.loc['Late: R10V3-R1V1', 'mean_pct']:.1f}% "
                            f"(NW t {ev2.loc['Late: R10V3-R1V1', 'nw_t']:.1f})",
         "Verdict": "Same direction, too little data to confirm (only Years 1-2 observable)"},
    ])
    st.table(score.set_index("Claim in the paper"))

    st.markdown(f"""
**In one paragraph.** Plain momentum still exists. Its average return since 1996 is about half the paper-era
figure, but the typical month is almost unchanged: take out the 10 worst of {len(post_ew)} months and the
post-1996 average is {post_ex10:.2f}% a month, as high as in the paper. What changed is the left tail. Since 1996 the
winner-minus-loser portfolio has lost {abs(st_post['down']):.1f}% a month on average in months that follow a falling
market, against {abs(st_pre['down']):.1f}% in 1965-1995. The paper's signature result, that momentum is concentrated in
heavily traded stocks, does not survive in modern data. What does survive is the paper's deeper idea:
volume tells you *where a stock is in its momentum life cycle*. Buying neglected (low-turnover) winners and
selling crowded (high-turnover) losers still earns about {alph.loc['Early: R10V1-R1V3', 'ff5mom_alpha_pct']:.1f}% a month
after controlling for the Fama-French five factors and momentum itself, while the opposite "late-stage"
combination earns nothing.
""")
    st.caption("All figures are equal-weighted, before trading costs. Use the tabs for the detail behind each line.")

# ================================================================= MOMENTUM 1927-2026
with tab_mom:
    st.markdown(
        f"Before testing the volume refinements, check the foundation. Kenneth French's library sorts **every** "
        f"NYSE, AMEX and NASDAQ stock in CRSP into deciles on its return from month t-12 to t-2 (vintage: {vintage}). "
        "That is survivorship-free and runs to mid-2026, so it shows what happened to momentum after the paper was written.")

    cum = np.log((1 + wml.fillna(0)).cumprod())
    fig = go.Figure()
    for col, color, name in [("EW", BLUE, "Equal-weighted D10-D1"), ("VW", ORANGE, "Value-weighted D10-D1")]:
        fig.add_trace(go.Scatter(x=cum.index, y=np.exp(cum[col]), name=name, line=dict(color=color, width=2),
                                 hovertemplate="%{x|%b %Y}: growth of $1 = %{y:,.1f}<extra>" + name + "</extra>"))
    fig.add_vrect(x0="1965-01-01", x1="1995-12-31", fillcolor=GRAY, opacity=0.12, line_width=0,
                  annotation_text="paper's sample", annotation_position="top left")
    fig.add_vline(x="2000-10-01", line_color=INK2, line_width=1)
    fig.add_annotation(x="2000-10-01", y=1, yref="paper", text="published Oct 2000", showarrow=False,
                       xanchor="left", font=dict(size=11, color=INK2))
    fig.update_yaxes(type="log")
    st.plotly_chart(style(fig, 420, "Growth of $1 in the long-short portfolio (log scale)"), width="stretch")
    st.caption("Long the top decile of past 12-2 month returns, short the bottom decile, rebalanced monthly, no costs.")

    a, b = st.columns(2)
    with a:
        es = era_stats(wml["EW"]).assign(series="Equal-weighted")
        vs = era_stats(wml["VW"]).assign(series="Value-weighted")
        fig = go.Figure()
        for df_, color in [(es, BLUE), (vs, ORANGE)]:
            fig.add_trace(go.Bar(x=df_["era"], y=df_["mean_pct"], name=df_["series"].iloc[0], marker_color=color,
                                 customdata=np.c_[df_["t"], df_["sharpe"]],
                                 hovertemplate="%{x}<br>%{y:.2f}% / month<br>t = %{customdata[0]:.2f}"
                                               "<br>Sharpe %{customdata[1]:.2f}<extra></extra>"))
        fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08, title="Average monthly return by era")
        st.plotly_chart(style(fig, 360, "mean monthly return", pct=True), width="stretch")
    with b:
        roll = 100 * wml["EW"].rolling(120).mean()
        fig = go.Figure(go.Scatter(x=roll.index, y=roll, name="10-year rolling mean", line=dict(color=BLUE, width=2),
                                   hovertemplate="%{x|%b %Y}: %{y:.2f}% / month<extra></extra>"))
        fig.add_hline(y=paper.TABLE1_R10_R1.loc[6, 6], line_color=ORANGE, line_width=1.5,
                      annotation_text=f"paper, J=6 K=6: {paper.TABLE1_R10_R1.loc[6, 6]:.2f}%",
                      annotation_position="top left", annotation_font=dict(size=11, color=INK2))
        fig.update_layout(title="Equal-weighted D10-D1, trailing 10-year average")
        st.plotly_chart(style(fig, 360, "mean monthly return", pct=True), width="stretch")

    st.subheader("Where the lost profit went: crashes after bear markets")
    a, b = st.columns([3, 2])
    with a:
        fig = go.Figure()
        for lab, d_, color in [("1965-1995", st_pre, GRAY), ("1996-2026", st_post, BLUE)]:
            fig.add_trace(go.Bar(x=["Market up over prior 24 months", "Market down over prior 24 months"],
                                 y=[d_["up"], d_["down"]], name=lab, marker_color=color,
                                 customdata=[[d_["up_n"], d_["up_t"]], [d_["down_n"], d_["down_t"]]],
                                 hovertemplate="%{x}<br>%{y:.2f}% / month<br>%{customdata[0]} months, "
                                               "t = %{customdata[1]:.2f}<extra>" + lab + "</extra>"))
        fig.update_layout(barmode="group", bargap=0.4, title="Equal-weighted D10-D1 by market state")
        st.plotly_chart(style(fig, 340, "mean monthly return", pct=True), width="stretch")
    with b:
        worst = (100 * wml["EW"].loc["1996":]).nsmallest(8).rename("D10-D1 %").to_frame()
        worst["market %"] = 100 * mkt.reindex(worst.index)
        worst.index = worst.index.strftime("%b %Y")
        st.markdown("**Worst months since 1996**")
        st.dataframe(worst.round(1), width="stretch")
    st.markdown(f"""
In ordinary times momentum still works: after a rising market the spread earned **{st_post['up']:.2f}%** a month
after 1996, close to the {st_pre['up']:.2f}% of the paper's era. The damage is concentrated in the
{st_post['down_n']} months that followed a falling market, where it lost **{abs(st_post['down']):.2f}%** a month.
The paper's sample contained only {st_pre['down_n']} such months; the modern era has had the 2000-02 bust,
2008-09 and the 2022 bear market, each followed by a sharp rebound led by the beaten-down stocks that momentum is short.
All eight of the worst months below came in months when the market rose.
""")

    table = pd.concat([era_stats(wml["EW"]).assign(weighting="Equal"), era_stats(wml["VW"]).assign(weighting="Value")])
    al = []
    for w_ in ["EW", "VW"]:
        for name, (a_, b_) in ERAS.items():
            r = S.factor_alpha(wml[w_].loc[a_:b_], ff, FF5)
            al.append(r.get("alpha_pct", np.nan))
    table["FF5 alpha %"] = al
    show = table[["weighting", "era", "months", "mean_pct", "t", "sharpe", "skew", "worst_month_pct",
                  "max_drawdown_pct", "FF5 alpha %"]].rename(columns={
        "mean_pct": "mean %/mo", "worst_month_pct": "worst month %", "max_drawdown_pct": "max drawdown %"})
    with st.expander("Full statistics by era"):
        st.dataframe(show.round(2), hide_index=True, width="stretch")
        st.caption("FF5 factors start in July 1963, so the 1927-1964 alpha is blank.")

# ================================================================= VOLUME REPLICATION
with tab_vol:
    st.markdown(
        "The volume tests need stock-level turnover, which Ken French does not publish. This part rebuilds the "
        "paper's 10 x 3 momentum-volume sort from scratch on a panel of about 3,500 US stocks a month, "
        "January 2009 to March 2017 (see *Data & method* for how it was built and validated).")

    with st.container(border=True):
        c = st.columns(4)
        J = c[0].selectbox("Formation period J (months)", [3, 6, 9, 12], index=1)
        K = c[1].selectbox("Holding period K (months)", [3, 6, 9, 12], index=1)
        skip = c[2].selectbox("Gap between formation and holding", [0, 1], index=0,
                              format_func=lambda x: "none (closest to the paper's 1 week)" if x == 0 else "1 month")
        min_price = c[3].selectbox("Minimum price at formation", [1.0, 5.0], index=0,
                                   format_func=lambda x: f"${x:.0f}" + (" (paper)" if x == 1 else ""))
    cur = jt(J, K, skip, min_price)

    st.subheader(f"Table II replica: monthly returns by momentum and volume, J={J}, K={K}")
    mine = P.table2(cur)
    a, b = st.columns(2)
    with a:
        st.markdown(f"**Modern: {cur.index.min():%b %Y} to {cur.index.max():%b %Y}, {len(cur)} months**")
        tb = mine.copy()
        tb.loc[tb.stat == "t", "Portfolio"] = "   (t-stat)"
        st.dataframe(tb.drop(columns="stat").round(2), hide_index=True, width="stretch")
        st.caption("Rows alternate: mean % per month, then t-statistic.")
    with b:
        st.markdown(f"**Paper: {paper.SAMPLE}**")
        pt = paper.table2(J, K)
        st.dataframe(pt, hide_index=True, width="stretch")
        st.caption("Mean % per month, transcribed from the paper's Table II.")

    a, b = st.columns(2)
    with a:
        pt = pt.set_index("Portfolio")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["V1 low", "V2", "V3 high"], y=pt.loc["R10-R1", ["V1", "V2", "V3"]].to_numpy(),
                             name="Paper, 1965-95", marker_color=GRAY,
                             hovertemplate="%{x}: %{y:.2f}% / month<extra>paper</extra>"))
        vals = [m(f"R10-R1|V{v}", cur) for v in (1, 2, 3)]
        tv = [t(f"R10-R1|V{v}", cur) for v in (1, 2, 3)]
        fig.add_trace(go.Bar(x=["V1 low", "V2", "V3 high"], y=vals, name="Modern, 2009-17", marker_color=BLUE,
                             customdata=tv, hovertemplate="%{x}: %{y:.2f}% / month, t = %{customdata:.2f}<extra>modern</extra>"))
        fig.update_layout(barmode="group", bargap=0.35, title="Momentum (R10-R1) within each turnover tercile")
        st.plotly_chart(style(fig, 360, "mean monthly return", pct=True), width="stretch")
    with b:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Losers R1", "Middle R5", "Winners R10"],
                             y=pt.loc[["R1", "R5", "R10"], "V3-V1"].to_numpy(), name="Paper, 1965-95", marker_color=GRAY,
                             hovertemplate="%{x}: %{y:.2f}% / month<extra>paper</extra>"))
        vals = [m(f"V3-V1|R{r}", cur) for r in (1, 5, 10)]
        tv = [t(f"V3-V1|R{r}", cur) for r in (1, 5, 10)]
        fig.add_trace(go.Bar(x=["Losers R1", "Middle R5", "Winners R10"], y=vals, name="Modern, 2009-17",
                             marker_color=BLUE, customdata=tv,
                             hovertemplate="%{x}: %{y:.2f}% / month, t = %{customdata:.2f}<extra>modern</extra>"))
        fig.update_layout(barmode="group", bargap=0.35, title="High minus low turnover (V3-V1), holding momentum fixed")
        st.plotly_chart(style(fig, 360, "mean monthly return", pct=True), width="stretch")

    st.subheader("Every formation and holding period (Table I and II summary)")
    g = grid0.pivot_table(index="J", columns=["series", "K"], values="mean_pct")
    pv = paper.r10_r1_by_volume().set_index(["J", "K"])
    zmax = 2.0
    cols_ = st.columns(3)
    panels = [("R10-R1", "All stocks", paper.TABLE1_R10_R1),
              ("R10-R1|V1", "Low turnover (V1)", pv["R10-R1|V1"].unstack("K")),
              ("R10-R1|V3", "High turnover (V3)", pv["R10-R1|V3"].unstack("K"))]
    for col, (key, label, pref) in zip(cols_, panels):
        modern = g[key]
        z = np.vstack([modern.to_numpy(), pref.to_numpy()])
        text = [[f"{v:.2f}" for v in row] for row in z]
        ylab = [f"J={j} modern" for j in modern.index] + [f"J={j} paper" for j in pref.index]
        fig = go.Figure(go.Heatmap(z=z, x=[f"K={k}" for k in modern.columns], y=ylab, text=text,
                                   texttemplate="%{text}", colorscale=DIVERGING, zmid=0, zmin=-zmax, zmax=zmax,
                                   showscale=False, xgap=2, ygap=2,
                                   hovertemplate="%{y}, %{x}: %{z:.2f}% / month<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(title=f"R10-R1, {label}")
        col.plotly_chart(style(fig, 420), width="stretch")
    st.caption("Mean monthly return in %. Top four rows: modern panel (no gap, $1 minimum price). "
               "Bottom four rows: the paper. Blue is positive, red negative, gray zero.")

    st.subheader("Turnover is a different animal now")
    prof = profile(J, skip, min_price)
    comp = prof.merge(paper.TABLE1_PROFILE_J6, on="Portfolio", suffixes=(" (modern)", " (paper, J=6)"))
    order = ["Portfolio"] + [f"{c} ({w_})" for c in ["avg daily turnover %", "formation return %/month", "median price $"]
                             for w_ in ["modern", "paper, J=6"]] + ["stocks per portfolio"]
    st.dataframe(comp[order].round(3), hide_index=True, width="stretch")
    ratio = (prof6["avg daily turnover %"] / paper.TABLE1_PROFILE_J6["avg daily turnover %"]).mean()
    st.markdown(f"The average stock now turns over about **{ratio:.0f}x** more of its shares each day than in the "
                "paper's sample. The ordering is the same (extreme winners and losers trade most), but the level has "
                "changed so much that 'high volume' means something different. See *Why it changed*.")

    st.subheader("Momentum life cycle: early-stage vs late-stage strategies")
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        order = ["R10-R1", "Early: R10V1-R1V3", "Late: R10V3-R1V1"]
        names = ["Simple R10-R1", "Early stage", "Late stage"]
        for yr, color_p, color_m in [("Year 1", GRAY, BLUE)]:
            fig.add_trace(go.Bar(x=names, y=paper.TABLE7.set_index("series").loc[order, yr], name="Paper, Year 1",
                                 marker_color=color_p, hovertemplate="%{x}: %{y:.1f}% in year 1<extra>paper</extra>"))
            fig.add_trace(go.Bar(x=names, y=ev1.loc[order, "mean_pct"], name="Modern, Year 1", marker_color=color_m,
                                 customdata=ev1.loc[order, "nw_t"],
                                 hovertemplate="%{x}: %{y:.1f}% in year 1, NW t = %{customdata:.2f}<extra>modern</extra>"))
        fig.add_trace(go.Bar(x=names, y=paper.TABLE7.set_index("series").loc[order, "Year 2"], name="Paper, Year 2",
                             marker_color="#c9c8c3", hovertemplate="%{x}: %{y:.1f}% in year 2<extra>paper</extra>"))
        fig.add_trace(go.Bar(x=names, y=ev2.loc[order, "mean_pct"], name="Modern, Year 2", marker_color=AQUA,
                             customdata=ev2.loc[order, "nw_t"],
                             hovertemplate="%{x}: %{y:.1f}% in year 2, NW t = %{customdata:.2f}<extra>modern</extra>"))
        fig.update_layout(barmode="group", bargap=0.3, title="Buy-and-hold return after formation (J=6)")
        st.plotly_chart(style(fig, 380, "return over the year", pct=True), width="stretch")
    with b:
        c_ = np.log((1 + cur[["R10-R1", "Early: R10V1-R1V3", "Late: R10V3-R1V1"]].fillna(0)).cumprod())
        fig = go.Figure()
        for col, color, name in [("Early: R10V1-R1V3", BLUE, "Early stage: long low-vol winners, short high-vol losers"),
                                 ("R10-R1", GRAY, "Simple momentum R10-R1"),
                                 ("Late: R10V3-R1V1", ORANGE, "Late stage: long high-vol winners, short low-vol losers")]:
            fig.add_trace(go.Scatter(x=c_.index, y=np.exp(c_[col]), name=name, line=dict(color=color, width=2),
                                     hovertemplate="%{x|%b %Y}: $%{y:.2f}<extra></extra>"))
        fig.update_layout(title=f"Growth of $1, J={J}, K={K}", legend=dict(orientation="v", y=1.0, x=0.01, yanchor="top"))
        st.plotly_chart(style(fig, 380, "value of $1"), width="stretch")

    st.subheader("Is it just risk? Factor-adjusted returns (J=6, K=6)")
    al = alph.reset_index()[["series", "raw_pct", "raw_t", "ff5_alpha_pct", "ff5_t", "ff5mom_alpha_pct", "ff5mom_t",
                             "b_mkt", "b_smb", "b_hml", "b_rmw", "b_mom"]]
    st.dataframe(al.rename(columns={"raw_pct": "raw %/mo", "raw_t": "t", "ff5_alpha_pct": "FF5 alpha %",
                                    "ff5_t": "t ", "ff5mom_alpha_pct": "FF5+Mom alpha %", "ff5mom_t": "t  ",
                                    "b_mkt": "beta Mkt", "b_smb": "beta SMB", "b_hml": "beta HML", "b_rmw": "beta RMW",
                                    "b_mom": "beta Mom"}).round(2), hide_index=True, width="stretch")
    st.markdown(f"""
High-turnover stocks carry a market beta well above low-turnover stocks and load on small, unprofitable
growth firms. Once those exposures are removed, low turnover beats high turnover by
**{alph.loc['V1-V3', 'ff5_alpha_pct']:.2f}% a month** (t = {alph.loc['V1-V3', 'ff5_t']:.1f}), much stronger than the raw
{alph.loc['V1-V3', 'raw_pct']:.2f}%. That is the paper's "low volume behaves like value" finding, alive and well.
The early-stage strategy keeps a {alph.loc['Early: R10V1-R1V3', 'ff5mom_alpha_pct']:.2f}% monthly alpha even after
controlling for momentum itself, so volume is adding information that the momentum factor does not contain.
Newey-West (6 lags) t-statistics.
""")

    st.subheader("Robustness")
    a, b = st.columns([3, 2])
    with a:
        sp = subp.pivot_table(index="series", columns="period", values="mean_pct")
        sp = sp[["Full sample", "2009-2012", "2013-2017", "Excluding 2009", "Live-captured only (Apr 2015+)"]]
        st.dataframe(sp.round(2), width="stretch")
        st.caption("Mean % per month by subperiod, J=6 K=6. 'Live-captured' months come from snapshots taken each day, "
                   "so they include stocks that later delisted and are free of survivorship bias.")
    with b:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=val.index, y=100 * val["crsp_D10-D1"], name="CRSP (Ken French), all stocks",
                                 line=dict(color=GRAY, width=2), hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra>CRSP</extra>"))
        fig.add_trace(go.Scatter(x=val.index, y=100 * val["panel_R10-R1"], name="This panel",
                                 line=dict(color=BLUE, width=2), hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra>panel</extra>"))
        corr = val["panel_R10-R1"].corr(val["crsp_D10-D1"])
        fig.update_layout(title=f"Validation: 12-2 momentum, correlation {corr:.2f}")
        st.plotly_chart(style(fig, 320, "monthly return", pct=True), width="stretch")

# ================================================================= WHY
with tab_why:
    st.markdown(f"""
### 1. Momentum did not fade so much as crash
The headline numbers say momentum halved after the paper: {pre['mean_pct']:.2f}% a month in 1965-1995 against
{post['mean_pct']:.2f}% since 1996 (equal-weighted). But it stayed strong for more than a decade after publication
({mid['mean_pct']:.2f}% a month in 1996-2008), the median month barely moved ({med_pre:.2f}% vs {med_post:.2f}%), and
removing just the 10 worst months out of {len(post_ew)} lifts the post-1996 average to {post_ex10:.2f}%. The loss
is concentrated in a handful of months.

Those months share a pattern. After a market decline the loser portfolio is full of high-beta stocks, so a long-short
momentum book is effectively short the market just as it rebounds. Daniel and Moskowitz show that momentum crashes
happen in these "panic" states and that past-loser betas can rise above 3 after major declines
([Momentum Crashes, 2016](https://doi.org/10.1016/j.jfineco.2015.12.002)). Here, in months following a down market,
the spread lost {abs(st_post['down']):.2f}% a month after 1996. Of the eight worst months since 1996, {n_after_down} came
after a 24-month market decline and in {n_mkt_up} of them the market itself was *up* (April 2009: market
+{100 * worst8.loc['2009-04', 'mkt'].iloc[0]:.0f}%, momentum {100 * worst8.loc['2009-04', 'wml'].iloc[0]:.0f}%).
The paper's 1965-1995 sample had only {st_pre['down_n']} months after a falling market and no rebound as violent as
2001, 2009 or January 2023.

### 2. Some decay, mostly in large stocks
Publication does matter. A study of 97 published return predictors finds their returns about 58% lower after
publication, and attributes roughly a third of the original return to investors trading on the published result
([McLean & Pontiff, 2016](https://doi.org/10.1111/jofi.12365)). For momentum the sign of that is in the value-weighted
(large-stock) version: since 2010 it has earned {vw_recent['mean_pct']:.2f}% a month (t = {vw_recent['t']:.1f}),
against {pre_vw['mean_pct']:.2f}% in the paper's era. Large, liquid stocks are where systematic funds can trade
momentum cheaply, so that is where you would expect it to be competed away first.

### 3. Turnover no longer measures the same thing
The paper reads high turnover as a sign of investor enthusiasm and glamour, and low turnover as neglect.
In 1965-1995 the average stock turned over around 0.12-0.24% of its shares per day; in this panel it is
{prof6['avg daily turnover %'].min():.2f}-{prof6['avg daily turnover %'].max():.2f}%, roughly {ratio:.0f}x higher.
Much of that increase is machine-driven: Rosenblatt Securities estimated that high-frequency traders accounted
for about two-thirds of US equity volume in 2009 and roughly half by 2013
([MarketsWiki summary](https://www.marketswiki.com/wiki/High-frequency_trading)), and index funds and ETFs now
trade whole baskets at once. When a large share of volume is market making and basket
rebalancing, turnover is a noisier signal of investor attention. That would weaken the paper's sharpest
volume result, that momentum is concentrated in high-volume stocks, which is exactly what we observe.

### 4. But volume still separates crowded trades from neglected ones
Two volume results survive, and both are about *crowding*:
- **Low-turnover stocks beat high-turnover stocks** after adjusting for risk, by
  {alph.loc['V1-V3', 'ff5_alpha_pct']:.2f}% a month. In modern data the effect is strongest among *winners*
  ({m('V3-V1|R10'):.2f}% a month for high-minus-low turnover winners), while in the paper it was strongest among losers.
  Heavily traded winners look like the crowded "late stage" names the paper warns about.
- **Early-stage beats late-stage momentum.** Winners that nobody is trading yet keep going; losers that everybody
  is dumping keep falling; and the reverse combination earns nothing in year 1 and gives back
  {abs(ev2.loc['Late: R10V3-R1V1', 'mean_pct']):.1f}% in year 2. This is the momentum life cycle the paper proposed,
  and it is the most robust thing in this study.

### 5. Caveats on the modern evidence
- **Short sample.** The stock panel covers about eight years ({len(base)} test months for J=6, K=6);
  t-statistics around 2 deserve caution,
  and Years 3-5 of the paper's long-horizon tests cannot be measured at all.
- **Survivorship bias before April 2015.** The 2009-2015 part of the panel was downloaded in March 2015, so stocks that
  delisted earlier are missing. That biases loser returns up. The live-captured months after April 2015 point the same
  way on the early/late and turnover results, but with 24 months they are noisy and the volume-momentum interaction flips
  sign there.
- **No trading costs, equal weights.** The extreme portfolios are full of small stocks that are expensive to trade
  and short; implementable returns would be lower, particularly for the loser legs.
""")

# ================================================================= DATA & METHOD
with tab_data:
    st.markdown(f"""
### Data
| Source | Used for | Coverage |
|---|---|---|
| Kenneth French Data Library: *10 Portfolios Formed on Momentum*, *Fama/French 5 Factors*, *Momentum Factor* | long-run momentum test, factor adjustment, validation | {ew.index.min():%b %Y} to {ew.index.max():%b %Y}, CRSP vintage {vintage} |
| [pystock-data](https://github.com/eliangcs/pystock-data) (Yahoo Finance prices, SEC EDGAR filings, CC BY-SA 4.0) | stock-level momentum x turnover sorts | Jan 2009 to Mar 2017, {wide().ret.shape[1]:,} tickers |

The paper used CRSP daily turnover for NYSE/AMEX stocks. CRSP is licensed, so this project rebuilds a comparable
panel from free data (`scripts/build_panel.py`):

1. **Prices.** Daily closes are chained into returns. Before March 2015 they come from Yahoo's split- and
   dividend-adjusted series; after that, from daily snapshots in which splits are detected from the price move
   *and* the change in share volume (a 2-for-1 split doubles volume; a 50% crash spikes it far more), which
   separates real splits from crashes.
2. **Shares outstanding** are backed out of each 10-K/10-Q as net income / EPS, made available 60 days after
   the period end, and put on the same split basis as volume. Turnover = shares traded / shares outstanding,
   averaged over each month.
3. **Cleaning.** Plain common-stock tickers with SEC filings only (drops ETFs, funds, preferreds, units).
   Bad prints (moves that revert within two weeks, recycled tickers, placeholder prices) are removed.
4. **Validation.** A 12-2 momentum sort on this panel has a correlation of
   {val['panel_R10-R1'].corr(val['crsp_D10-D1']):.2f} with CRSP's own decile spread over the same months
   ({val['panel_R10'].corr(val['crsp_D10']):.2f} for winners, {val['panel_R1'].corr(val['crsp_D1']):.2f} for losers).

### Method (following the paper)
- At each month-end, eligible stocks (price at least $1, J months of returns, turnover available) are ranked
  **independently** into 10 momentum portfolios on their past J-month return and 3 volume portfolios on their
  average daily turnover over the same J months.
- **Holding periods** overlap as in Jegadeesh & Titman (1993): the month-m return is the equal-weighted average
  of the K portfolios formed over the previous K months, so monthly observations do not overlap and simple
  t-statistics apply.
- **Event-time returns** (Years 1 and 2) are buy-and-hold per stock, averaged across stocks, with Newey-West
  (11 lag) t-statistics for the overlapping formation dates.
- **Differences from the paper:** monthly rather than daily data, so the one-week gap becomes either no gap
  or one month (selectable); NASDAQ stocks are included because the 1990s double-counting of dealer volume
  that made the authors exclude them ended with modern market structure; decile breakpoints use all stocks
  rather than NYSE/AMEX only.

### What changed from the original version of this repo
The earlier code ranked 20 present-day mega-caps into "deciles" (two stocks each), computed turnover but never
used it, and appended an intraday ridge-regression backtest unrelated to the paper. It was replaced by this
study. See the README for details.

### Reference
Lee, C. M. C., & Swaminathan, B. (2000). Price Momentum and Trading Volume. *Journal of Finance*, 55(5), 2017-2069.
[{paper.DOI}]({paper.DOI})
""")
