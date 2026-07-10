"""
Credit Enhancement Dashboard - ABS Strategy SIP (prime + subprime auto)

Four analytical lenses on credit enhancement (CE) -- a price, a signal, a
risk-transfer mechanism, and how they converge -- plus the contractual triggers
that make CE dynamic, and a methodology tab. Run with:

    streamlit run app.py

Reads CSVs from data/. Run `python make_sample_data.py` first for sample data,
or drop your real absee.py output + curated structure files into data/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from engine import (data, formulas, montecarlo, scoring, capital_stack,
                    dynamics, triggers)

# --------------------------------------------------------------------------- #
# Palette + shared theme                                                       #
# --------------------------------------------------------------------------- #
PALETTE = ["#405871", "#7f9bb9", "#58706e", "#c9b47e", "#706240",
           "#565871", "#afb0c7", "#b2c1c0"]
INK = "#2b3a4a"
GRID = "#e6e9ee"
AXIS = "#c5ccd6"

C_LOSS = "#706240"        # realized / actual loss (warm)
C_EXPECTED = "#565871"    # expected loss
C_CE = "#58706e"          # credit enhancement / protection
C_DIST = "#7f9bb9"        # simulated distribution
C_LIMIT = "#405871"       # trigger limit
C_BREACH = "#8c4a3f"      # breach / warning

RATING_COLORS = {"AAA": "#405871", "AA": "#7f9bb9", "A": "#58706e",
                 "BBB": "#c9b47e", "BB": "#706240", "B": "#565871"}
GRADE_COLORS = {"prime": "#405871", "subprime": "#c9b47e"}
LOSS_SCALE = [[0.0, "#b2c1c0"], [0.5, "#c9b47e"], [1.0, "#706240"]]

pio.templates["ce"] = go.layout.Template(layout=dict(
    colorway=PALETTE,
    font=dict(family="Inter, Helvetica, Arial, sans-serif", color=INK, size=12),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    title=dict(font=dict(size=15, color="#405871")),
    xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=AXIS,
               ticks="outside", tickcolor=AXIS, title=dict(font=dict(size=12))),
    yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=AXIS,
               ticks="outside", tickcolor=AXIS, title=dict(font=dict(size=12))),
    legend=dict(bgcolor="rgba(255,255,255,0.65)", bordercolor=GRID, borderwidth=1),
))
pio.templates.default = "ce"
px.defaults.template = "ce"

st.set_page_config(page_title="Credit Enhancement Dashboard",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  /* ---- layout & rhythm ---- */
  .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1160px; }
  section.main > div { gap: 0.4rem; }
  hr { border-color: #edf0f3; margin: 1.2rem 0; }

  /* ---- typography ---- */
  html, body, [class*="css"] { font-family: 'Inter','Helvetica Neue',Arial,sans-serif; }
  h1 { font-size: 1.65rem; font-weight: 700; color:#334a5f; letter-spacing:-.02em; margin-bottom:.15rem; }
  h2 { font-size: 1.12rem; font-weight: 650; color:#405871; margin-top:1.5rem; letter-spacing:-.01em; }
  h3, h4 { font-size: .98rem; font-weight: 600; color:#58706e; }
  [data-testid="stCaptionContainer"] { color:#8a97a4; font-size:.82rem; }

  /* ---- metric cards: airy, soft shadow, no hard border ---- */
  [data-testid="stMetric"] {
      background:#ffffff; border:1px solid #eef1f4; border-radius:14px;
      padding:16px 18px; box-shadow:0 1px 3px rgba(40,58,74,.06); }
  [data-testid="stMetricLabel"] p { color:#8a97a4; font-size:.8rem; font-weight:500; }
  [data-testid="stMetricValue"] { color:#334a5f; font-weight:700; font-size:1.5rem; }
  [data-testid="stMetricDelta"] { font-size:.78rem; }

  /* ---- tabs: clean underline style ---- */
  .stTabs [data-baseweb="tab-list"] { gap:2px; border-bottom:1px solid #eef1f4; }
  .stTabs [data-baseweb="tab"] { height:44px; padding:0 18px; background:transparent;
      color:#93a1af; font-weight:600; font-size:.9rem; }
  .stTabs [aria-selected="true"] { color:#405871; border-bottom:2px solid #405871; }

  /* ---- sidebar ---- */
  section[data-testid="stSidebar"] { background:#f7f9fb; border-right:1px solid #eef1f4; }
  section[data-testid="stSidebar"] .block-container { padding-top:1.4rem; }

  /* ---- tables & inputs ---- */
  [data-testid="stDataFrame"] { border:1px solid #eef1f4; border-radius:10px; }
  .stSlider, .stSelectbox { margin-bottom:.3rem; }

  /* ---- hide chrome ---- */
  #MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; }
  [data-testid="stSidebar"],
  [data-testid="stSidebarCollapsedControl"],
  [data-testid="collapsedControl"] { visibility: visible !important; }

  header { background:transparent; }
</style>
""", unsafe_allow_html=True)


TAB_BLURBS = {
    1: "**Lens 1 - CE as a price, over time.** What yield does an investor give up "
       "for protection, and how does that protection *build* as the pool amortizes "
       "against the losses that erode it?",
    2: "**Lens 2 - CE as a signal.** What do an originator's structural choices "
       "reveal about their confidence in the collateral?",
    3: "**Lens 3 - CE as risk transfer.** Apply one shock and watch how each "
       "stakeholder in the capital stack experiences it differently.",
    4: "**Triggers - the contractual tripwires.** Cumulative-net-loss triggers "
       "trap cash and step up OC when losses run hot. They are what makes CE "
       "dynamic - the mechanism behind Lens 1's rising CE path.",
    5: "**Lens 4 - Convergence.** One structural choice, read three ways.",
    6: "**Methodology & data.** How the numbers are sourced, modeled, and what to "
       "trust them for.",
}


@st.cache_data
def get_deals() -> pd.DataFrame:
    return data.load_deals()


@st.cache_data
def get_tranches() -> pd.DataFrame:
    return data.load_tranches()


@st.cache_data
def get_realized() -> pd.DataFrame:
    return data.load_realized()


@st.cache_data
def run_mc(pd_: float, lgd: float, ce: float, rho: float, n: int,
           shock: float) -> dict:
    res = montecarlo.simulate(pd_, lgd, ce, correlation=rho, n_sims=n,
                              shock_multiplier=shock, seed=7)
    return {"losses": res.losses, **res.as_dict()}


def pct(x, d: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.{d}f}%"

def style(fig, height: int = 380, x: str | None = None, y: str | None = None,
          title: str | None = None, legend_top: bool = True):
    # Three stacked bands with real gaps: title (very top) -> legend (its own band)
    # -> plot. Top placement avoids the left y-labels, the right secondary axis,
    # and the x-axis title.
    top = 100 if (title and legend_top) else (72 if legend_top else (50 if title else 16))
    fig.update_layout(
        height=height,
        title=(dict(text=title, x=0.0, xanchor="left", y=0.99, yanchor="top",
                    font=dict(size=14.5, color="#405871")) if title else None),
        margin=dict(t=top, l=12, r=16, b=12),
        legend=(dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0,
                     title_text="", font=dict(size=11),
                     bgcolor="rgba(255,255,255,0)") if legend_top else dict()),
    )
    if x:
        fig.update_xaxes(title_text=x)
    if y:
        fig.update_yaxes(title_text=y)
    return fig


def vmark(fig, x, color: str, text: str | None = None, dash: str = "dash",
          width: float = 1.5) -> None:
    """Vertical marker that is safe on datetime axes.

    plotly's add_vline computes a mean of the x positions to place the annotation,
    which raises on a Timestamp x. Adding the line and the annotation separately
    avoids that path entirely.
    """
    fig.add_vline(x=x, line=dict(color=color, dash=dash, width=width))
    if text:
        fig.add_annotation(x=x, y=1.0, yref="paper", yanchor="bottom",
                           xanchor="left", text=text, showarrow=False,
                           font=dict(color=color, size=10))


# --------------------------------------------------------------------------- #
# Sidebar                                                                       #
# --------------------------------------------------------------------------- #
try:
    deals = get_deals()
    tranches = get_tranches()
    realized = get_realized()
except FileNotFoundError:
    # First launch with an empty data/ -- generate the sample set instead of making
    # the user run make_sample_data.py by hand, then reload.
    import make_sample_data
    with st.spinner("No data found - generating the sample dataset..."):
        make_sample_data.build(data.DATA_DIR)
    get_deals.clear()
    get_tranches.clear()
    get_realized.clear()
    deals = get_deals()
    tranches = get_tranches()
    realized = get_realized()

st.sidebar.title("Controls")
deal_name = st.sidebar.selectbox("Deal", sorted(deals["deal_name"]))
deal = deals[deals["deal_name"] == deal_name].iloc[0]
deal_tr = tranches[tranches["deal_name"] == deal_name].copy()
deal_perf = realized[realized["deal_name"] == deal_name].copy()
# Rankings compare the full universe (prime + subprime); the credibility score is
# risk-normalized, so cross-grade comparison is intentional.
deals_s = deals

st.sidebar.markdown("---")
st.sidebar.subheader("Monte Carlo")
regime = st.sidebar.selectbox("Stress regime", list(montecarlo.SHOCK_REGIMES))
shock = montecarlo.SHOCK_REGIMES[regime]
basel_rho = montecarlo.basel_retail_correlation(deal["assumed_pd"])
uplift = montecarlo.REGIME_RHO_UPLIFT[regime]
rho_adj = st.sidebar.slider("Asset-correlation adjustment", -0.10, 0.20, 0.0, 0.01,
                            help="Base correlation is the Basel 'other retail' value "
                                 "at this deal's PD; the regime adds a downturn "
                                 "uplift. This nudges the result up or down.")
rho = montecarlo.rho_for_regime(deal["assumed_pd"], regime, rho_adj)
st.sidebar.caption(f"Asset correlation (rho) = **{rho:.2f}**  ·  Basel retail base "
                   f"{basel_rho:.2f} (PD {deal['assumed_pd']:.0%}) + {regime} uplift "
                   f"{uplift:.2f} {'+' if rho_adj >= 0 else '-'} {abs(rho_adj):.2f}")
n_sims = st.sidebar.select_slider("Simulations", [10_000, 25_000, 50_000, 100_000],
                                  value=50_000)
st.sidebar.caption(f"Grade: **{deal['grade']}**  ·  Originator: **{deal['originator']}**")

st.title("Credit Enhancement Dashboard")
st.caption(f"{deal_name}  ·  {deal['grade']} auto loan ABS")

tab_intro, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "ABS & CE Primer", "1 · Risk-Adjusted Return", "2 · Originator Confidence",
    "3 · Capital Stack", "4 · Triggers", "5 · Convergence",
    "6 · Methodology & Data"])

with tab_intro:
    st.title("Asset-Backed Securities & Credit Enhancement")
    st.caption("A one-page primer for the rest of the dashboard.")
    st.markdown(
        "An **asset-backed security (ABS)** turns a pool of many small loans into "
        "rated bonds. This dashboard studies **subprime auto-loan ABS** — thousands of "
        "car loans to weaker-credit borrowers, pooled and sold as tranched notes.")

    st.divider()
    st.subheader("How an ABS works")
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown(
            "An ABS does one core thing: it **pools many risky loans and re-slices "
            "them into bonds of different safety** — so most of the pool can be sold as "
            "highly-rated debt even though the individual borrowers are not.")
        st.markdown(
            "**Pool → trust → tranches.** Thousands of car-loan payments are sold into a "
            "bankruptcy-remote **trust**, which issues **tranches**: Class A (senior) "
            "down through the subordinate classes, plus a residual/equity piece at the "
            "bottom.")
        st.markdown(
            "**Two waterfalls, opposite directions:**  \n"
            "• **Cash flows *down*** — senior notes are paid first, the residual last.  \n"
            "• **Losses flow *up*** — the residual and junior notes are wiped out first, "
            "so the senior notes are hit only once losses get severe.")
        st.markdown(
            "That asymmetry is the whole trick: **concentrating risk at the bottom is "
            "what makes the top safe.** The loss level where a tranche first takes a hit "
            "is its **attachment point** — and everything beneath it *is* its credit "
            "enhancement. For the senior notes:")
        st.markdown("> **Senior CE = subordination + overcollateralization + reserve**")

        with right:
            dot = ('digraph { rankdir=TB; bgcolor="transparent"; ranksep=0.22; '
               'node[shape=box,style="rounded,filled",fontname=Helvetica,fontcolor=white,'
               'width=2.7,height=0.42,fixedsize=true,fontsize=11]; '
               'edge[dir=none,color="#c5ccd6"]; '
               'P[label="Subprime auto-loan pool",fillcolor="#7f9bb9",fontcolor="#2b3a4a"]; '
               'A[label="Senior  ·  Class A",fillcolor="#405871"]; '
               'B[label="Mezzanine  ·  B / C",fillcolor="#58706e"]; '
               'D[label="Subordinate  ·  D / E",fillcolor="#706240"]; '
               'R[label="Residual + OC + reserve",fillcolor="#c9b47e",fontcolor="#2b3a4a"]; '
               'P->A->B->D->R; }')
            st.graphviz_chart(dot, use_container_width=True)
            st.caption("cash flows down  ·  losses flow up")


    st.divider()
    st.subheader("Credit enhancement — the buffer")
    st.markdown(
        "**Credit enhancement (CE)** absorbs collateral losses *before* the senior "
        "notes are touched — it's what lets a risky pool support AAA bonds.")
    f = st.columns(4, gap="medium")
    f[0].markdown("**Subordination**  \nJunior notes take losses first.")
    f[1].markdown("**Overcollateralization**  \nMore collateral than notes issued.")
    f[2].markdown("**Reserve account**  \nA cash cushion held in trust.")
    f[3].markdown("**Excess spread**  \nExtra interest — a *soft*, performance-dependent buffer.")
    st.markdown(
        "*How much CE, of what kind, and how it holds up* is the whole subject of this "
        "dashboard.")

    st.divider()
    st.subheader("What the tabs show")
    g = st.columns(5, gap="small")
    g[0].markdown("**Price**  \nYield given up for protection.")
    g[1].markdown("**Signal**  \nHow conservatively it's structured.")
    g[2].markdown("**Risk transfer**  \nWho absorbs a loss.")
    g[3].markdown("**Triggers**  \nTripwires that make CE dynamic.")
    g[4].markdown("**Convergence**  \nHow it all connects.")


# --------------------------------------------------------------------------- #
# Tab 1: CE as Risk-Adjusted Return (now dynamic over time)                    #
# --------------------------------------------------------------------------- #
with tab1:
    st.markdown(TAB_BLURBS[1])
    senior = deal_tr.sort_values("attachment_pct").iloc[-1]
    senior_ce0 = float(senior["attachment_pct"])
    exp_loss = formulas.expected_loss(deal["assumed_pd"], deal["assumed_lgd"])

    # Trigger breach (if any) drives the OC step-up in the CE path.
    ev = triggers.evaluate(deal, deal_perf)
    path = dynamics.ce_path(deal, deal_perf, senior_ce0,
                            breach_month=ev["breach_month"])

    if len(path) == 0:
        st.info("No realized performance for this deal yet.")
    else:
        max_m = int(path["month"].max())
        cc = st.columns([3, 1])
        month = cc[0].slider("Months since closing", 1, max_m, max_m,
                             help="Drag through the deal's life: CE builds as the "
                                  "pool amortizes; realized losses erode the cushion.")
        manual = cc[1].checkbox("Manual CE override")
        snap = dynamics.snapshot(path, month)
        realized_cnl_m = float(snap["realized_cnl"])

        if manual:
            ce_for_mc = st.slider("Override available CE (fraction of pool)",
                                  0.0, 0.99, round(senior_ce0, 3), 0.005)
        else:
            ce_for_mc = max(0.0, float(snap["available_ce_pct"]))

        mc = run_mc(deal["assumed_pd"], deal["assumed_lgd"], ce_for_mc, rho,
                    n_sims, shock)

        c = st.columns(4)
        c[0].metric("Senior CE (current pool)", pct(snap["structural_ce_pct"]),
                    help="Subordination as a share of the shrinking pool - grows "
                         "as the senior amortizes.")
        c[1].metric("Realized cum. loss", pct(realized_cnl_m))
        c[2].metric("Available cushion", pct(snap["available_ce_pct"]),
                    delta="held" if snap["available_ce_pct"] > 0 else "breached")
        c[3].metric("Expected loss (issuance)", pct(exp_loss))
        c = st.columns(3)
        c[0].metric(f"P(CE exhaustion) · {regime}", pct(mc["p_ce_exhaustion"]))
        c[1].metric("MC expected loss", pct(mc["expected_loss"]))
        c[2].metric("99% tail loss", pct(mc["tail_loss_99"]))

        left, right = st.columns(2)
        with left:
            fig = go.Figure()
            fig.add_scatter(x=path["period_end"], y=path["structural_ce_pct"] * 100,
                            name="Senior CE (% current pool)", mode="lines",
                            line=dict(color=C_CE, width=2.6), yaxis="y2")
            fig.add_scatter(x=path["period_end"], y=path["realized_cnl"] * 100,
                            name="Realized cum. loss (% orig.)", mode="lines",
                            line=dict(color=C_LOSS, width=2.6))
            fig.add_scatter(x=path["period_end"], y=path["available_ce_pct"] * 100,
                            name="Available cushion (% orig.)", mode="lines",
                            line=dict(color=C_EXPECTED, width=1.8, dash="dot"))
            vmark(fig, snap["period_end"], AXIS, dash="solid", width=1.5)
            if ev["breach_month"]:
                bm = path[path["month"] == ev["breach_month"]]
                if len(bm):
                    vmark(fig, bm.iloc[0]["period_end"], C_BREACH,
                          text="trigger breach")
            fig.update_layout(
                yaxis2=dict(title="Senior CE (% current pool)", overlaying="y",
                            side="right", showgrid=False, range=[0, 100]))
            style(fig, x="Reporting period", y="% of original pool",
                  title="CE builds as the pool amortizes")
            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig = go.Figure()
            fig.add_histogram(x=mc["losses"] * 100, histnorm="probability", nbinsx=70,
                              marker_color=C_DIST, name="Simulated outcomes")
            fig.add_vline(x=ce_for_mc * 100, line=dict(color=C_CE, width=2),
                          annotation_text="Available CE", annotation_font_color=C_CE)
            fig.add_vline(x=mc["tail_loss_99"] * 100, line=dict(color=C_LOSS, dash="dash"),
                          annotation_text="99% tail", annotation_position="top right",
                          annotation_font_color=C_LOSS)
            style(fig, x="Simulated collateral loss (% of pool)", y="Probability",
                  title=f"Monte Carlo loss distribution · {regime} (rho {rho:.2f})",
                  legend_top=False)
            st.plotly_chart(fig, use_container_width=True)
        st.caption("CE is dynamic: subordination as a share of the amortizing pool "
                   "rises (right axis) while realized losses chip at the cushion "
                   "(left axis). The Monte Carlo uses the *available* cushion at the "
                   "selected month, so P(exhaustion) climbs as losses accrue.")

    tr_sorted = deal_tr.sort_values("attachment_pct").copy()
    tr_sorted["ce_pct"] = tr_sorted["initial_ce_pct"] * 100
    tr_sorted["cpn_pct"] = tr_sorted["coupon_pct"] * 100
    fig = px.scatter(tr_sorted, x="ce_pct", y="cpn_pct", text="tranche",
                     color="rating", size="original_balance",
                     color_discrete_map=RATING_COLORS,
                     labels={"ce_pct": "Credit enhancement (% of pool)",
                             "cpn_pct": "Coupon (%)", "rating": "Rating"})
    fig.update_traces(textposition="top center", textfont_size=10)
    style(fig, height=380, title="Yield given up for protection (up the capital stack)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Up the stack: more CE, lower coupon. The slope is the price of "
               "protection.")


# --------------------------------------------------------------------------- #
# Tab 2: CE as Originator Confidence Signal                                   #
# --------------------------------------------------------------------------- #
with tab2:
    st.markdown(TAB_BLURBS[2])
    cs = scoring.score_deal(deal)

    # The composite is weight-invariant in ORDER (robustness: Spearman >= 0.99),
    # so we report the RANK, not the number -- the ordering is the information.
    ranking = scoring.score_all(deals).reset_index(drop=True)
    ranking.insert(0, "rank", ranking.index + 1)
    n = len(ranking)
    my_rank = int(ranking.loc[ranking["deal_name"] == deal_name, "rank"].iloc[0])

    c = st.columns(3)
    c[0].metric("Conservatism rank", f"#{my_rank} of {n}",
                help="Rank on voluntary structural conservatism. Only the ordering is "
                     "meaningful; the underlying composite is weight-invariant.")
    c[1].metric("Deferred-loss risk", f"{cs.deferred_loss_score:.1f}")
    c[2].metric("Red flags", sum(cs.flags.values()))

    left, right = st.columns(2)
    with left:
        comp = pd.DataFrame({
            "component": [k.replace("_", " ") for k in cs.components],
            "value": list(cs.components.values())})
        fig = px.bar(comp, x="value", y="component", orientation="h")
        fig.update_traces(marker_color="#405871")
        style(fig, height=340, x="Signed contribution", y="",
              title="What pushes this deal up or down the ranking", legend_top=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("**Structural red flags**")
        for k, v in cs.flags.items():
            st.write(f"{'🔴' if v else '🟢'} {k.replace('_', ' ')}")

    st.markdown("**Deals ranked by structural conservatism**")
    show = ranking.drop(columns=["credibility_score"])
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.caption("Only the **ranking** is reported. A sensitivity test (equal weights, "
               "±50% on each, 500 random weight sets) leaves the order essentially "
               "unchanged (Spearman ≥ 0.99) — so the composite number carries no "
               "information beyond the rank, and we don't show it.")

    last_loss = realized.groupby("deal_name")["cum_net_loss_rate"].last().reset_index()
    merged = ranking.merge(last_loss, on="deal_name", how="left")
    if merged["cum_net_loss_rate"].notna().any():
        merged["loss_pct"] = merged["cum_net_loss_rate"] * 100
        merged["short"] = merged["deal_name"].str.replace(r" \(.*\)", "", regex=True)
        fig = px.scatter(merged.dropna(subset=["loss_pct"]), x="rank", y="loss_pct",
                         text="short",
                         labels={"rank": "Conservatism rank (1 = most conservative)",
                                 "loss_pct": "Realized cumulative loss (%)"})
        fig.update_traces(marker_color="#405871", textposition="top center",
                          textfont_size=9,
                          marker=dict(size=12, line=dict(width=1, color="white")))
        style(fig, height=420,
              title="Conservatism rank vs. realized loss (descriptive, not predictive)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Credibility ranks *structural conservatism* (voluntary CE above the "
                   "rating requirement). It is not a loss forecast — realized loss is "
                   "driven by collateral, a separate axis.")
    else:
        st.info("Realized loss not available to overlay yet.")


# --------------------------------------------------------------------------- #
# Tab 3: CE Across the Capital Stack                                          #
# --------------------------------------------------------------------------- #
with tab3:
    st.markdown(TAB_BLURBS[3])
    default_shock = float(min(50.0, deal["assumed_pd"] * deal["assumed_lgd"] * 100 * shock))
    shock_loss = st.slider("Collateral loss shock (% of pool)", 0.0, 60.0,
                           default_shock, 0.5) / 100

    alloc = capital_stack.allocate(deal_tr, shock_loss)
    view = capital_stack.stakeholder_view(deal, deal_tr, shock_loss)
    alloc["bal_m"] = alloc["original_balance"] / 1e6
    alloc["wiped_pct"] = alloc["loss_fraction"] * 100
    alloc["loss_m"] = alloc["loss_amount"] / 1e6
    order = alloc.sort_values("attachment_pct")["tranche"].tolist()

    left, right = st.columns([1.1, 1])
    with left:
        fig = px.bar(alloc, x="bal_m", y="tranche", orientation="h",
                     color="wiped_pct", color_continuous_scale=LOSS_SCALE,
                     range_color=[0, 100],
                     labels={"bal_m": "Tranche balance ($M)", "tranche": "",
                             "wiped_pct": "% wiped"})
        fig.update_layout(yaxis=dict(categoryorder="array", categoryarray=order),
                          coloraxis_colorbar=dict(title="% wiped", thickness=12))
        style(fig, height=380, title=f"Loss allocation at {shock_loss*100:.1f}% collateral loss",
              legend_top=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("**Same shock, four stakeholders**")
        st.metric("Senior loss rate", pct(view["senior"]["loss_rate"]))
        st.metric("Residual value erosion", pct(view["residual"]["erosion"]))
        st.metric("Rating-agency cushion", pct(view["rating_agency"]["cushion"]),
                  delta="adequate" if view["rating_agency"]["cushion"] > 0 else "deficient")
        st.metric("Originator retained-risk cost",
                  pct(view["originator"]["retained_risk_cost"]))

    fig = px.bar(alloc.sort_values("attachment_pct"), x="tranche", y="loss_m",
                 color="rating", color_discrete_map=RATING_COLORS,
                 labels={"tranche": "Tranche", "loss_m": "Loss absorbed ($M)",
                         "rating": "Rating"})
    style(fig, height=320, title="Who bears the loss?")
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# Tab 4: Triggers (dynamic, tied to the CE path in Tab 1)                      #
# --------------------------------------------------------------------------- #
with tab4:
    st.markdown(TAB_BLURBS[4])
    st.caption("A **mechanism demonstration**, not a historical backtest. The selected "
               "deal's structure (priced loss, trigger tightness, OC target) anchors "
               "the shapes; you drive the loss scenario. Watch the trigger fire, trap "
               "cash, and rebuild CE.")

    horizon = 60
    months = np.arange(1, horizon + 1)

    cc = st.columns(3)
    peak_loss = cc[0].slider(
        "Scenario peak cumulative loss (% of pool)", 0.0, 40.0,
        float(round(deal["assumed_pd"] * deal["assumed_lgd"] * 100 * 1.2, 1)), 0.5,
        help="How high the hypothetical cumulative net loss ramps by the end.") / 100
    ramp = cc[1].select_slider("Loss timing", ["slow", "base", "fast"], value="base")
    tight = cc[2].slider(
        "Trigger tightness", 0.0, 1.0,
        float(deal.get("trigger_strength", 0.5)), 0.05,
        help="Higher = tighter trigger, sits closer to priced loss, breaches sooner.")

    # Trigger schedule: deal-anchored, tightness overridable.
    d2 = deal.copy()
    d2["trigger_strength"] = tight
    limit = triggers.cnl_trigger_schedule(d2, horizon)

    # Synthetic loss path: S-curve to the chosen peak; timing shifts the midpoint.
    mid = {"slow": 0.65, "base": 0.45, "fast": 0.30}[ramp] * horizon
    s = 1.0 / (1.0 + np.exp(-0.18 * (months - mid)))
    s = (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s
    loss = peak_loss * s

    seasoned = months >= 6
    breach_mask = (loss > limit) & seasoned
    breach_month = int(months[breach_mask][0]) if breach_mask.any() else None

    # OC response: ramp initial -> target over 12m; step target up 50% after a breach.
    init_oc, tgt = float(deal["initial_oc_pct"]), float(deal["target_oc_pct"])
    def oc_path(with_step: bool):
        vals = []
        for m in months:
            t_eff = tgt * (1.5 if (with_step and breach_month and m >= breach_month) else 1.0)
            vals.append(min(t_eff, init_oc + (t_eff - init_oc) * min(1.0, m / 12)))
        return np.array(vals)
    oc_base, oc_trig = oc_path(False), oc_path(True)

    c = st.columns(2)
    c[0].metric("Outcome", "BREACH" if breach_month else "No breach",
                delta="cash trapped → OC steps up" if breach_month else "excess spread released",
                delta_color="inverse" if breach_month else "normal")
    c[1].metric("Terminal trigger limit", pct(float(limit[-1])))


    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_scatter(x=months, y=limit * 100, name="CNL trigger limit",
                        line=dict(color=C_LIMIT, dash="dash", width=2.2))
        fig.add_scatter(x=months, y=loss * 100, name="Scenario cumulative loss",
                        line=dict(color=C_LOSS, width=2.6))
        if breach_month:
            vmark(fig, breach_month, C_BREACH, text=f"breach · m{breach_month}", dash="dot")
        style(fig, x="Months since closing", y="Cumulative net loss (% of pool)",
              title="Does the scenario breach the trigger?")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = go.Figure()
        fig.add_scatter(x=months, y=oc_base * 100, name="OC target (no breach)",
                        line=dict(color=AXIS, dash="dot", width=2))
        fig.add_scatter(x=months, y=oc_trig * 100, name="OC target (breach step-up)",
                        line=dict(color=C_CE, width=2.6))
        style(fig, x="Months since closing", y="OC target (% of pool)",
              title="A breach traps cash → OC steps up → CE rebuilds")
        st.plotly_chart(fig, use_container_width=True)

    st.caption("While the loss line stays under the trigger, excess spread is released "
               "to the residual. The month it crosses (after the 6-month seasoning "
               "window), the deal fails its trigger: cash is trapped and the OC target "
               "steps up, rebuilding protection — the rising leg you see in Tab 1. Push "
               "the peak-loss and timing sliders to make it breach earlier, later, or "
               "never; tighten the trigger to see it bite sooner.")


# --------------------------------------------------------------------------- #
# Tab 5: Convergence Mindmap (causal argument)                                 #
# --------------------------------------------------------------------------- #
with tab5:
    st.markdown(TAB_BLURBS[5])
    senior = deal_tr.sort_values("attachment_pct").iloc[-1]
    senior_ce = float(senior["attachment_pct"])
    mc = run_mc(deal["assumed_pd"], deal["assumed_lgd"], senior_ce, rho, n_sims, shock)
    cs = scoring.score_deal(deal)
    realized_cum = deal_perf["cum_net_loss_rate"].iloc[-1] if len(deal_perf) else None
    exp_loss = formulas.expected_loss(deal["assumed_pd"], deal["assumed_lgd"])
    coverage = senior_ce / exp_loss if exp_loss else 0.0
    first_loss_ce = deal["initial_oc_pct"] + deal["reserve_fund_pct"]
    held = realized_cum is not None and senior_ce > realized_cum
    realized_txt = pct(realized_cum) if realized_cum is not None else "n/a"
    surplus_txt = pct(senior_ce - realized_cum) if realized_cum is not None else "n/a"
    verdict = "held" if held else ("breached" if realized_cum is not None else "n/a")
    adq_color = "#58706e" if held else ("#706240" if realized_cum is not None else "#b2c1c0")
    ev = triggers.evaluate(deal, deal_perf)

    p_exh = mc["p_ce_exhaustion"]
    avail_ce = deal["initial_oc_pct"] + deal["total_subordination_pct"]
    vol_ce = avail_ce - deal["required_ce_senior_pct"]
    reserve_val = deal["reserve_fund_pct"]
    try:
        xsr = scoring._excess_spread_reliance(deal)
    except Exception:
        xsr = deal["excess_spread_pct"]

    st.caption("An interconnected map. Solid spokes are the framework (in theory); "
               "dashed links show how the subtopics converge for THIS deal (in the "
               "case). Drag nodes to explore.")

    def _render_graph() -> None:
        import json
        import streamlit.components.v1 as components
        from pyvis.network import Network

        H = {"CE": "#405871", "PRICE": "#7f9bb9", "SIGNAL": "#c9b47e",
             "RISK": "#58706e", "SHOCK": "#706240", "TRIG": "#8c4a3f"}
        net = Network(height="640px", width="100%", bgcolor="#ffffff",
                      font_color="#2b3a4a", directed=False, cdn_resources="in_line")
        net.set_options(json.dumps({
            "physics": {"barnesHut": {"gravitationalConstant": -16000,
                                      "centralGravity": 0.2, "springLength": 180,
                                      "springConstant": 0.035, "damping": 0.4,
                                      "avoidOverlap": 0.85},
                        "minVelocity": 0.75, "stabilization": {"iterations": 250}},
            "nodes": {"shape": "dot",
                      "font": {"size": 14, "face": "Helvetica", "color": "#2b3a4a"}},
            "edges": {"font": {"size": 10, "color": "#58706e", "strokeWidth": 4,
                               "strokeColor": "#ffffff", "align": "middle"},
                      "smooth": {"type": "continuous"}},
            "interaction": {"hover": True, "dragNodes": True},
        }))

        def N(i, label, color, size):
            net.add_node(i, label=label, color=color, size=size)

        def E(a, b, label="", dashed=False):
            net.add_edge(a, b, label=label, dashes=dashed,
                         color="#565871" if dashed else "#c7d0da")

        N("CE", "Credit Enhancement", H["CE"], 30)
        for h, lab in [("PRICE", "Price"), ("SIGNAL", "Signal"),
                       ("RISK", "Risk transfer"), ("TRIG", "Triggers"),
                       ("SHOCK", "Shock regimes")]:
            N(h, lab, H[h], 22)
            E("CE", h)
        N("cov", f"CE coverage\n{coverage:.1f}x", H["PRICE"], 14); E("PRICE", "cov")
        N("exh", f"CE exhaustion\n{pct(p_exh)}", H["PRICE"], 14); E("PRICE", "exh")
        N("sur", f"CE surplus\n{surplus_txt}", H["PRICE"], 14); E("PRICE", "sur")
        N("cred", f"Credibility\n{cs.score:.2f}", H["SIGNAL"], 14); E("SIGNAL", "cred")
        N("vol", f"Voluntary CE\n{pct(vol_ce)}", H["SIGNAL"], 14); E("SIGNAL", "vol")
        N("res", f"Reserve\n{pct(reserve_val)}", H["SIGNAL"], 14); E("SIGNAL", "res")
        N("xs", f"Excess-spread\nreliance {pct(xsr)}", H["SIGNAL"], 14); E("SIGNAL", "xs")
        N("sr", f"Senior CE\n{pct(senior_ce)}", H["RISK"], 14); E("RISK", "sr")
        N("rsd", f"Residual\nfirst-loss {pct(first_loss_ce)}", H["RISK"], 14); E("RISK", "rsd")
        N("rtg", f"Rating cushion\n{pct(vol_ce)}", H["RISK"], 14); E("RISK", "rtg")
        trg_lbl = (f"CNL breach\nm{ev['breach_month']}" if ev["breached"]
                   else "CNL trigger\nclear")
        N("cnl", trg_lbl, H["TRIG"], 14); E("TRIG", "cnl")
        N("oc", f"OC step-up\n{'yes' if ev['breached'] else 'no'}", H["TRIG"], 14)
        E("TRIG", "oc")
        for rg in montecarlo.SHOCK_REGIMES:
            rid = "rg_" + rg.split()[0]
            active = rg == regime
            N(rid, (rg + f"\nP(exh) {pct(p_exh)}") if active else rg.split()[0],
              "#706240" if active else "#afb0c7", 16 if active else 10)
            E("SHOCK", rid)
        E("vol", "cov", "sizes", True)
        E("cred", "sur", "tracks", True)
        E("exh", "sr", "senior risk", True)
        E("sur", "rsd", "first loss", True)
        E("vol", "rtg", "same cushion", True)
        E("xs", "sur", "soft CE", True)
        E("rtg", "cov", "rating needs CE", True)
        E("cnl", "oc", "traps cash", True)
        E("oc", "sr", "rebuilds CE", True)
        arid = "rg_" + regime.split()[0]
        E(arid, "exh", "drives", True)
        E(arid, "sr", "drives", True)

        try:
            html = net.generate_html(notebook=False)
        except TypeError:
            html = net.generate_html()
        components.html(html, height=660, scrolling=False)

    try:
        _render_graph()
    except Exception:
        st.info("Interactive map needs pyvis "
                "(`pip install pyvis`). Showing the static chain instead.")
        dot = f"""
        digraph CE {{
          rankdir=LR; bgcolor="transparent";
          node [shape=box, style="rounded,filled", fontname=Helvetica, fontsize=11,
                fontcolor=white];
          edge [fontname=Helvetica, fontsize=9, color="#7f9bb9", fontcolor="#58706e"];
          choice [label="Structural choice\\nCE {coverage:.1f}x EL", fillcolor="#405871"];
          signal [label="Signal\\ncredibility {cs.score:.2f}", fillcolor="#c9b47e"];
          outcome [label="Outcome\\nrealized {realized_txt}", fillcolor="#565871"];
          adequacy [label="CE adequacy\\n{verdict}", fillcolor="{adq_color}"];
          risk [label="Risk allocation\\nsenior protected", fillcolor="#58706e"];
          stress [label="{regime}\\nP(exh) {pct(mc['p_ce_exhaustion'])}", fillcolor="#706240"];
          choice -> signal [label="reveals"]; signal -> outcome [label="predicts"];
          outcome -> adequacy [label="tests"]; adequacy -> risk [label="determines"];
          choice -> stress [label="stressed"];
        }}
        """
        st.graphviz_chart(dot, use_container_width=True)

    st.subheader("The argument in words")
    trig_sentence = (
        f"the CNL trigger **breached at month {ev['breach_month']}**, trapping cash "
        f"and stepping up OC"
        if ev["breached"] else "the CNL trigger **stayed clear** all life")
    st.markdown(
        f"1. **Structural choice -> signal:** this deal sets senior CE at "
        f"{pct(senior_ce)} ({coverage:.1f}x expected loss), scoring {cs.score:.2f} "
        f"on credibility with {sum(cs.flags.values())} red flag(s).\n"
        f"2. **Signal -> outcome:** realized loss came in at {realized_txt} "
        f"(expected {pct(exp_loss)}).\n"
        f"3. **Outcome -> adequacy:** CE therefore **{verdict}**, with surplus "
        f"{surplus_txt} over realized loss; {trig_sentence}.\n"
        f"4. **Adequacy -> risk allocation:** losses are absorbed first by the "
        f"residual / OC ({pct(first_loss_ce)}), leaving the senior protected -- but "
        f"under *{regime}* (rho {rho:.2f}) the modeled P(CE exhaustion) rises to "
        f"{pct(mc['p_ce_exhaustion'])}.\n\n"
        f"*One structural choice, read as price, signal, and risk transfer.*"
    )


# --------------------------------------------------------------------------- #
# Tab 6: Methodology & Data                                                     #
# --------------------------------------------------------------------------- #
with tab6:
    st.markdown(TAB_BLURBS[6])

    st.subheader("Data collection")
    st.markdown(
        "**Scope: prime and subprime auto-loan ABS.** The dashboard spans both "
        "collateral grades; the credibility score is risk-normalized (CE relative to "
        "expected loss, not absolute), so prime and subprime are comparable on the "
        "same axis.\n\n"
        "- **Realized performance** comes from SEC EDGAR **ABS-EE** loan-level "
        "filings (Reg AB II, Schedule AL), pulled by `absee.py`. For each monthly "
        "filing it sums loan-level charge-offs and recoveries to a pool-level "
        "**cumulative net loss curve** (`cum_net_loss_rate`).\n"
        "- **Deal universe** is configured in `deals.json` (trust name + CIK). "
        "Scaling the shelf is just adding entries -- no code changes.\n"
        "- **Structure & pricing** (`deals.csv`, `tranches.csv`) are hand-curated "
        "from prospectuses: OC, reserve, subordination, coupons, attachment points, "
        "rating-agency required CE.\n"
        "- The dataset shown here is **synthetic** (`make_sample_data.py`) but uses "
        "the exact schema `absee.py` emits, so real filings drop straight in.")

    n_deals = len(deals)
    n_breach = sum(
        triggers.evaluate(d, realized[realized["deal_name"] == d["deal_name"]])["breached"]
        for _, d in deals.iterrows())
    c = st.columns(4)
    c[0].metric("Deals", n_deals)
    c[1].metric("Tranches", len(tranches))
    c[2].metric("Perf. rows", len(realized))
    c[3].metric("Deals breaching trigger", int(n_breach))

    st.subheader("Models & formulas")
    st.markdown(
        "**CE metrics (`formulas.py`, Eqs. 1-23).** Expected loss = PD x LGD; CE "
        "coverage = available CE / expected loss; surplus/shortfall = CE - realized "
        "loss; rating cushion = available CE - required CE; plus OC cushion, reserve "
        "ratio, and voluntary-CE gap.\n\n"
        "**Dynamic CE (`dynamics.py`).** CE is not static. As the pool amortizes, "
        "senior notes pay first while subordinate notes and OC are locked out, so "
        "senior CE as a share of the *current* pool grows like "
        "`senior_CE0 / pool_factor`. Trapped excess spread builds OC from its initial "
        "level toward target over ~12 months; a trigger breach steps the target up "
        "50%. The *available cushion* is senior CE at issuance minus realized loss "
        "(original-pool basis) -- the honest adequacy test in Tab 1.\n\n"
        "**Triggers (`triggers.py`).** Cumulative-net-loss trigger schedules are "
        "modeled parametrically: the terminal limit is the priced loss (PD x LGD) "
        "times a headroom that shrinks with `trigger_strength` (tighter = more "
        "protective), ramped over the deal's life with a front-loaded shape so young "
        "deals get more headroom. A breach is the first seasoned month realized CNL "
        "exceeds the limit -- which feeds the OC step-up in `dynamics.py`.\n\n"
        "**Monte Carlo (`montecarlo.py`).** Single-factor **Vasicek** "
        "large-homogeneous-pool model (the basis for Basel IRB and most ABS loss "
        "modeling): simulate a systematic factor, derive a conditional default rate, "
        "multiply by LGD for a portfolio loss rate. Stress scales PD by a regime "
        "multiplier. **Asset correlation is grounded in the Basel IRB 'other retail' "
        "formula** -- the supervisory standard for granular retail pools (auto loans "
        "qualify) -- which makes rho a *decreasing* function of PD, bounded 3%-16%. "
        "Because Basel's rho is point-in-time, each stress regime adds a (stylized) "
        "downturn uplift on top to capture correlation widening in a crisis; the "
        "sidebar then adjusts around that.\n\n"
        "**Credibility (`scoring.py`, Eqs. 12, 27).** Transparent fixed-weight score "
        "over *risk-normalized* features (CE relative to expected loss, not absolute), "
        "plus a deferred-loss flag count.")

    st.markdown(
        f"**Stress regimes and the resulting correlation** "
        f"(Basel retail base for {deal_name} = "
        f"{montecarlo.basel_retail_correlation(deal['assumed_pd']):.2f} at "
        f"PD {deal['assumed_pd']:.0%})")
    reg_tbl = pd.DataFrame({
        "Regime": list(montecarlo.SHOCK_REGIMES),
        "PD multiplier": [f"{m:.1f}x" for m in montecarlo.SHOCK_REGIMES.values()],
        "Downturn rho uplift": [f"+{montecarlo.REGIME_RHO_UPLIFT[r]:.2f}"
                                for r in montecarlo.SHOCK_REGIMES],
        "Effective rho (this deal)": [f"{montecarlo.rho_for_regime(deal['assumed_pd'], r):.2f}"
                                      for r in montecarlo.SHOCK_REGIMES],
    })
    st.dataframe(reg_tbl, use_container_width=True, hide_index=True)

    st.subheader("What to trust this for")
    st.markdown(
        "- **Relative** comparisons across deals (credibility ranking, "
        "trigger headroom, CE adequacy) are the point.\n"
        "- The Vasicek MC and parametric triggers are **stylized** -- calibrate rho, "
        "PD/LGD, and swap in real prospectus trigger tables before any decision use.\n"
        "- Realized losses are only as good as the ABS-EE field mapping; confirm "
        "charge-off / recovery tags with `absee.py inspect` per shelf.")
