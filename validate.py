"""Offline validation of the engine against sample data (no Streamlit needed)."""
import numpy as np
from engine import data, formulas, montecarlo, scoring, capital_stack, triggers, dynamics

deals = data.load_deals()
tranches = data.load_tranches()
realized = data.load_realized()

print("== Credibility vs realized loss (should be NEGATIVE) ==")
rank = scoring.score_all(deals)
last = realized.groupby("deal_name")["cum_net_loss_rate"].last()
rank = rank.assign(realized=rank["deal_name"].map(last))
for _, r in rank.iterrows():
    print(f"  {r['credibility_score']:6.2f}  loss={r['realized']*100:6.2f}%  "
          f"flags={r['n_red_flags']}  {r['deal_name'][:45]}")
corr = np.corrcoef(rank["credibility_score"], rank["realized"])[0, 1]
print(f"  corr(credibility, realized loss) = {corr:+.3f}  "
      f"{'OK (negative)' if corr < 0 else 'WRONG (should be negative)'}")

print("\n== Monte Carlo: strong vs weak shelf, base vs GFC (rho follows regime) ==")
for name in ["AmeriCredit Auto Receivables 2021-3 (subprime)",
             "American Credit Acceptance 2022-3 (subprime)"]:
    d = deals[deals["deal_name"] == name].iloc[0]
    senior = tranches[tranches["deal_name"] == name].sort_values("attachment_pct").iloc[-1]
    for regime in ("Base case", "Global Financial Crisis"):
        rho = montecarlo.rho_for_regime(d["assumed_pd"], regime)
        mc = montecarlo.simulate(d["assumed_pd"], d["assumed_lgd"],
                                 senior["attachment_pct"], correlation=rho,
                                 n_sims=50_000,
                                 shock_multiplier=montecarlo.SHOCK_REGIMES[regime],
                                 seed=7)
        print(f"  {name[:34]:34} {regime:24} rho={rho:.2f}  "
              f"EL={mc.expected_loss*100:5.2f}%  P(exhaust)={mc.p_ce_exhaustion*100:5.2f}%")

print("\n== Triggers: who breaches their CNL schedule? ==")
n_breach = 0
for _, d in deals.iterrows():
    ev = triggers.evaluate(d, realized[realized["deal_name"] == d["deal_name"]])
    n_breach += 1 if ev["breached"] else 0
    bm = ev["breach_month"] if ev["breach_month"] else "-"
    print(f"  {d['deal_name'][:42]:42} breach@={str(bm):>4}  "
          f"trigT={ev['terminal_limit']*100:5.1f}%")
print(f"  {n_breach}/{len(deals)} deals breach")

print("\n== Dynamic CE: builds as pool amortizes (Exeter 2022-2) ==")
name = "Exeter Automobile Receivables 2022-2 (subprime)"
d = deals[deals["deal_name"] == name].iloc[0]
perf = realized[realized["deal_name"] == name]
sce = tranches[tranches["deal_name"] == name]["attachment_pct"].max()
ev = triggers.evaluate(d, perf)
path = dynamics.ce_path(d, perf, sce, breach_month=ev["breach_month"])
for m in (1, 12, 24, len(path)):
    s = dynamics.snapshot(path, m)
    print(f"  month {m:>3}: pool_factor={s['pool_factor']*100:5.1f}%  "
          f"senior CE(cur)={s['structural_ce_pct']*100:5.1f}%  "
          f"avail cushion={s['available_ce_pct']*100:6.1f}%")
assert path["structural_ce_pct"].iloc[-1] >= path["structural_ce_pct"].iloc[0], \
    "CE should grow as the pool amortizes"

print("\n== Capital stack waterfall (Exeter, 15% collateral loss) ==")
tr = tranches[tranches["deal_name"] == name]
alloc = capital_stack.allocate(tr, 0.15)
for _, t in alloc.iterrows():
    print(f"  {t['tranche']:>2} ({t['rating']:>3})  attach={t['attachment_pct']*100:5.1f}%  "
          f"wiped={t['loss_fraction']*100:5.1f}%")

print("\n== Formula spot checks ==")
assert formulas.expected_loss(0.2, 0.6) == 0.12
assert formulas.ce_surplus_shortfall(0.30, 0.155) == 0.145
assert abs(formulas.ce_coverage_ratio(0.30, 0.12) - 2.5) < 1e-9
assert montecarlo.basel_retail_correlation(0.05) > montecarlo.basel_retail_correlation(0.30), \
    "Basel retail rho should fall as PD rises"
assert (montecarlo.rho_for_regime(0.20, "Global Financial Crisis")
        > montecarlo.rho_for_regime(0.20, "Base case")), "stress should widen rho"
print("  Eqs 1-3 + Basel rho(PD) + regime uplift ordering OK")
print("\nVALIDATION COMPLETE")
