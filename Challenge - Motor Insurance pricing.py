# Challenge: motor insurance pricing with GLMs

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

rng = np.random.default_rng(12)
n = 40_000
df = pd.DataFrame({
    "age_band": rng.choice(["18-24", "25-59", "60+"], n, p=[0.12, 0.70, 0.18]),
    "vehicle": rng.choice(["Small", "Mid", "Performance"], n, p=[0.45, 0.40, 0.15]),
    "area": rng.choice(["Urban", "Rural"], n, p=[0.65, 0.35]),
    "exposure": rng.uniform(0.2, 1.0, n),       # policy-years in force
})
true_freq = {"age_band": {"18-24": 1.9, "25-59": 1.0, "60+": 1.1},
             "vehicle": {"Small": 0.9, "Mid": 1.0, "Performance": 1.5},
             "area": {"Urban": 1.3, "Rural": 1.0}}
lam = 0.08 * df.exposure
for col, rel in true_freq.items():
    lam = lam * df[col].map(rel)
df["claims"] = rng.poisson(lam)
sev_mean = (18_000 * df.vehicle.map({"Small": 0.8, "Mid": 1.0, "Performance": 1.6})
            * df.area.map({"Urban": 1.15, "Rural": 1.0}))

shape = 2.0
k = shape * df.claims.clip(lower=1)
df["avg_severity"] = np.where(df.claims > 0, rng.gamma(k, sev_mean / k), np.nan)
print(f"Policies: {n:,}  exposure: {df.exposure.sum():,.0f} years  claims: {df.claims.sum():,}  "
      f"frequency: {df.claims.sum() / df.exposure.sum():.3f} per year")

formula = " ~ C(age_band, Treatment('25-59')) + C(vehicle, Treatment('Mid')) + C(area, Treatment('Rural'))"
# Frequency: Poisson GLM with log(exposure) as offset
freq = smf.glm("claims" + formula, df, family=sm.families.Poisson(),
               offset=np.log(df.exposure)).fit()
# Severity: Gamma GLM (log link) on policies with claims, weighted by claim count
has = df[df.claims > 0]
sev = smf.glm("avg_severity" + formula, has,
              family=sm.families.Gamma(sm.families.links.Log()),
              var_weights=has.claims).fit()

rel = pd.DataFrame({"freq_relativity": np.exp(freq.params),
                    "sev_relativity": np.exp(sev.params)}).drop("Intercept")
rel.index = [i.split("[T.")[1].rstrip("]") for i in rel.index]
print("\nRelativities vs base (25-59, Mid, Rural):")
print(rel.round(2).to_string())

def pure_premium(age_band, vehicle, area):
    row = pd.DataFrame({"age_band": [age_band], "vehicle": [vehicle],
                        "area": [area], "exposure": [1.0]})
    return float(freq.predict(row, offset=np.zeros(1)).iloc[0] * sev.predict(row).iloc[0])

print("\nPure premium per policy-year (expected claims cost):")
for profile in [("25-59", "Mid", "Rural"), ("18-24", "Performance", "Urban"), ("60+", "Small", "Rural")]:
    print(f"{profile[0]:>6} {profile[1]:<11} {profile[2]:6} R{pure_premium(*profile):>9,.0f}")