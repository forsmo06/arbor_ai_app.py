# arbor_ai_app.py
# -*- coding: utf-8 -*-
"""
Arbor AI – samlet Streamlit-app (én fil) med:
- Modusdeteksjon (Hombak/Maier)
- Autokalibrering av fuktsensor (RLS m/glemselsfaktor)
- Dødtidsestimat + forslag til neste prøvetidspunkt
- Guardrails (hard/soft) per resept + rate‑limit
- «MPC‑lite» (ett-trinns fremoversyn / ARX) m/ forklaring
- DoE‑steg (kontrollerte små steg) i rolige perioder
- KPI‑dashboard + A/B‑evaluering pr. skift
- Hendelseslogg og eksport

NB: Dette er en funksjonell «referanse‑app». Koble til dine faktiske tags/sensordata
ved å mate inn CSV eller strømme fra deres kilde. Variabelnavn matcher fagprat hos Arbor.
"""

import io
import time
from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ============ Utils ============ #

@st.cache_data
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file, sep=None, engine="python")
    # Normaliser forventede kolonner hvis mulig
    # Forventet: timestamp, utlopstemp, innlopstemp, trykk_nedre_ovn, friskluftspjeld, 
    # hombak_pc, maier_pc, fukt_sensor, fukt_manuell (kan mangle), bunkerniva_pc
    # Prøv å senke kolonnenavn
    df.columns = [c.strip().lower() for c in df.columns]
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])  
    else:
        # hvis ikke finnes, lag en kunstig tidsakse
        df["timestamp"] = pd.date_range(end=datetime.now(), periods=len(df), freq="5min")
    return df.sort_values("timestamp").reset_index(drop=True)


def ewma(series: pd.Series, alpha: float = 0.3) -> pd.Series:
    out = []
    s = None
    for x in series:
        s = x if s is None else alpha * x + (1 - alpha) * s
        out.append(s)
    return pd.Series(out, index=series.index)


class RLSCalibrator:
    """Enkel RLS (Recursive Least Squares) for fuktsensor-korreksjon.
    Modell: fukt_manuell ≈ a + b * fukt_sensor
    """
    def __init__(self, lam: float = 0.99, delta: float = 1000.0):
        self.lam = lam
        self.theta = np.array([0.0, 1.0])  # [a, b]
        self.P = np.eye(2) * delta

    def update(self, sensor: float, manual: float) -> Tuple[float, float]:
        x = np.array([1.0, sensor])  # design‑vektor
        y = manual
        # RLS oppdatering
        lam_inv = 1.0 / self.lam
        Px = self.P @ x
        k = Px / (self.lam + x.T @ Px)
        e = y - self.theta.T @ x
        self.theta = self.theta + k * e
        self.P = lam_inv * (self.P - np.outer(k, x) @ self.P)
        return float(self.theta[0]), float(self.theta[1])

    def correct(self, sensor: float) -> float:
        a, b = self.theta
        return a + b * sensor


def detect_mode(hombak_pc: float, maier_pc: float) -> float:
    total = max(hombak_pc + maier_pc, 1e-6)
    return np.clip(maier_pc / total, 0.0, 1.0)  # 0=Hombak‑tung, 1=Maier‑tung


def estimate_dead_time_minutes(df: pd.DataFrame, col_input: str, col_output: str, search_max_min: int = 60) -> int:
    """Grovt dødtidsestimat via krysskorrelasjon på avledede endringer."""
    x = df[col_input].diff().fillna(0).values
    y = df[col_output].diff().fillna(0).values
    n = min(len(x), len(y))
    x = x[-n:]
    y = y[-n:]
    # korrelasjon for lags 0..search_max_min (antar 1 rad ≈ 5 min hvis logg er 5min)
    best_lag = 0
    best_corr = -9e9
    for lag in range(0, search_max_min + 1):
        if lag == 0:
            corr = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else 0
        else:
            corr = np.corrcoef(x[:-lag], y[lag:])[0, 1] if np.std(x[:-lag]) > 0 and np.std(y[lag:]) > 0 else 0
        if np.isnan(corr):
            corr = 0
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    return int(best_lag * 5)  # antatt 5 min pr. rad; juster etter dine data


def mpc_lite_suggest(
    current_setpoint: float,
    features: Dict[str, float],
    arx_coef: Dict[str, float],
    constraints: Dict[str, float],
    rate_limit: float = 1.5,
) -> Tuple[float, Dict[str, float]]:
    """Ett-trinns fremoversyn: prognose av fukt 30 min frem og valg av nytt setpunkt.
    arx_coef forventer nøkler: bias, k_setpoint, k_innlop, k_mode, k_friskluft, k_last.
    constraints: dict med hard_min, hard_max, soft_step_max.
    Returnerer (foreslaatt_setpunkt, forklaring_dict)
    """
    # lineær prognose (velg selv bedre modell senere)
    y_hat = (
        arx_coef["bias"]
        + arx_coef["k_setpoint"] * current_setpoint
        + arx_coef["k_innlop"] * features.get("innlopstemp", 0.0)
        + arx_coef["k_mode"] * features.get("mode", 0.0)
        + arx_coef["k_friskluft"] * features.get("friskluftspjeld", 0.0)
        + arx_coef["k_last"] * features.get("bunkerniva_pc", 50.0)
    )

    target = features.get("target_fukt", 1.20)
    # sensitivitet: antatt hvor mye fukt endres pr. 1 °C setpunktendring
    # (kan læres fra DoE; startverdi)
    sens = arx_coef.get("k_dset", -0.10)  # %-poeng fukt per +1 °C utløp, negativt tegn vanligvis

    delta_needed = (target - y_hat) / sens if abs(sens) > 1e-6 else 0.0

    # rate-limit og soft-grenser
    delta_clamped = np.clip(delta_needed, -rate_limit, rate_limit)
    proposed = current_setpoint + float(delta_clamped)

    # hard-grenser
    proposed = float(np.clip(proposed, constraints["hard_min"], constraints["hard_max"]))

    explanation = {
        "prognose_fukt": y_hat,
        "mål_fukt": target,
        "estimert_sensitivitet(%%/°C)": sens,
        "rå_delta": delta_needed,
        "etter_rate_limit": delta_clamped,
        "foreslått_setpunkt": proposed,
        "bidrag_innlop": arx_coef["k_innlop"] * features.get("innlopstemp", 0.0),
        "bidrag_mode": arx_coef["k_mode"] * features.get("mode", 0.0),
        "bidrag_friskluft": arx_coef["k_friskluft"] * features.get("friskluftspjeld", 0.0),
        "bidrag_last": arx_coef["k_last"] * features.get("bunkerniva_pc", 50.0),
    }
    return proposed, explanation


def compute_kpis(df: pd.DataFrame, target: float, window: str = "7D") -> Dict[str, float]:
    recent = df.set_index("timestamp").last(window)
    if recent.empty:
        return {"n": 0}
    fukt = recent["fukt_corr"].dropna() if "fukt_corr" in recent.columns else recent["fukt_manuell"].dropna()
    inside = (fukt.between(target - 0.1, target + 0.1)).mean() * 100 if len(fukt) else np.nan
    std = fukt.std() if len(fukt) else np.nan
    return {
        "antall_punkt": int(len(fukt)),
        "std_fukt": float(std) if pd.notnull(std) else np.nan,
        "andel_innenfor_±0.1pp(%)": float(inside) if pd.notnull(inside) else np.nan,
    }


def add_event(log, level: str, msg: str):
    log.append({"tid": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "nivå": level, "hendelse": msg})


# ============ Streamlit UI ============ #
st.set_page_config(page_title="Arbor AI", layout="wide")
st.title("🌲 Arbor AI – tørkestyring (beta)")

with st.sidebar:
    st.header("Resept & mål")
    resept = st.selectbox("Resept", ["22mm STD", "22mm NG STD", "22mm FUKT"], index=0)
    target_fukt = st.number_input("Mål-fukt (%)", value=1.20, step=0.05, format="%.2f")

    st.markdown("---")
    st.header("Guardrails")
    if resept.startswith("22mm") and "FUKT" not in resept:
        hard_min, hard_max = 133.0, 137.0
    else:
        # eksempel andre grenser
        hard_min, hard_max = 132.0, 139.0
    hard_min = st.number_input("Hard min utløpstemp (°C)", value=float(hard_min))
    hard_max = st.number_input("Hard max utløpstemp (°C)", value=float(hard_max))
    rate_limit = st.number_input("Maks endring pr. justering (°C)", value=1.5, step=0.1)
    undertrykk_min = st.number_input("Min. undertrykk nedre ovn (Pa)", value=270)

    st.markdown("---")
    st.header("Kalibrering & modell")
    lam = st.slider("Glemselsfaktor (RLS)", 0.95, 0.999, 0.99)
    sens0 = st.number_input("Start-sensitivitet (Δfukt per +1°C)", value=-0.10, step=0.01, format="%.2f")

    st.markdown("---")
    st.header("A/B‑oppsett")
    mode_ab = st.radio("Kjøring", ["Baseline (manuell)", "AI (MPC‑lite)"])

    st.markdown("---")
    csv = st.file_uploader("Last opp CSV‑logg (valgfritt)")

# Session state
if "cal" not in st.session_state:
    st.session_state.cal = RLSCalibrator(lam=0.99)
if "events" not in st.session_state:
    st.session_state.events = []

# Data
if csv is not None:
    df = load_csv(csv)
else:
    # Generer en enkel demo-datastrøm
    n = 200
    ts = pd.date_range(end=datetime.now(), periods=n, freq="5min")
    rng = np.random.default_rng(42)
    utlop = 135 + np.cumsum(rng.normal(0, 0.05, n))
    innlop = 180 + rng.normal(0, 5, n)
    frisk = np.clip(30 + rng.normal(0, 3, n), 10, 50)
    hombak = np.clip(70 + rng.normal(0, 5, n), 0, 100)
    maier = 100 - hombak
    bunk = np.clip(50 + rng.normal(0, 10, n), 10, 90)
    # sann fukt (syntetisk):
    f_true = 1.2 + (-0.10)*(utlop - 135) + 0.002*(innlop - 180) + 0.15*(maier/100) + rng.normal(0, 0.05, n)
    f_sens = f_true + rng.normal(0, 0.15, n) + 0.3  # sensor med bias
    # manuell prøve hvert ~6. punkt
    f_man = pd.Series(f_true).where(np.arange(n) % 6 == 0, np.nan)
    df = pd.DataFrame({
        "timestamp": ts,
        "utlopstemp": utlop,
        "innlopstemp": innlop,
        "friskluftspjeld": frisk,
        "hombak_pc": hombak,
        "maier_pc": maier,
        "bunkerniva_pc": bunk,
        "fukt_sensor": f_sens,
        "fukt_manuell": f_man,
        "trykk_nedre_ovn": 275 + rng.normal(0, 5, n),
    })

# Beregn mode, autokalibrering, korrigert fukt
modes = []
f_corr = []
for i, row in df.iterrows():
    mode = detect_mode(row.get("hombak_pc", 50.0), row.get("maier_pc", 50.0))
    modes.append(mode)
    if not np.isnan(row.get("fukt_manuell", np.nan)):
        st.session_state.cal.lam = lam
        a, b = st.session_state.cal.update(row["fukt_sensor"], row["fukt_manuell"])
    f_corr.append(st.session_state.cal.correct(row["fukt_sensor"]))

df["mode"] = modes
df["fukt_corr"] = f_corr

# Dødtidsestimat (grovt)
dead_min = estimate_dead_time_minutes(df, "utlopstemp", "fukt_corr", search_max_min=60)
next_probe_eta = df["timestamp"].iloc[-1] + timedelta(minutes=dead_min)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Dødtid (est.)", f"~{dead_min} min")
col2.metric("Neste prøve (ETA)", next_probe_eta.strftime("%H:%M"))
_kpis = compute_kpis(df, target_fukt, window="7D")
col3.metric("Std. fukt (7d)", f"{_kpis.get('std_fukt', float('nan')):.3f}")
col4.metric("Innenfor ±0.1pp", f"{_kpis.get('andel_innenfor_±0.1pp(%)', float('nan')):.1f}%")

st.markdown("---")
left, right = st.columns([2, 1])

with left:
    st.subheader("Trender")
    st.line_chart(df.set_index("timestamp")[
        ["fukt_sensor", "fukt_corr", "fukt_manuell"]].rename(columns={
            "fukt_sensor": "Fukt sensor",
            "fukt_corr": "Fukt korrigert",
            "fukt_manuell": "Fukt manuell"
        })
    )
    st.line_chart(df.set_index("timestamp")[["utlopstemp", "innlopstemp", "friskluftspjeld"]])

with right:
    st.subheader("A/B og forslag")
    current = df.iloc[-1]

    constraints = {"hard_min": hard_min, "hard_max": hard_max}
    arx_coef = {
        "bias": 0.8,
        "k_setpoint": -0.05,
        "k_innlop": 0.002,
        "k_mode": 0.15,
        "k_friskluft": 0.001,
        "k_last": -0.002,
        "k_dset": sens0,
    }

    features = {
        "innlopstemp": float(current.get("innlopstemp", 180.0)),
        "mode": float(current.get("mode", 0.5)),
        "friskluftspjeld": float(current.get("friskluftspjeld", 30.0)),
        "bunkerniva_pc": float(current.get("bunkerniva_pc", 50.0)),
        "target_fukt": float(target_fukt),
    }

    current_setpoint = float(current.get("utlopstemp", 135.0))
    undertrykk_ok = float(current.get("trykk_nedre_ovn", 270.0)) >= undertrykk_min

    if not undertrykk_ok:
        add_event(st.session_state.events, "ADVARSEL", f"Undertrykk {current.get('trykk_nedre_ovn', np.nan):.0f} Pa < {undertrykk_min} Pa – hold setpunkt!")
        st.warning("Undertrykk under grense – AI fryses midlertidig.")
        proposed = current_setpoint
        explanation = {"grunn": "undertrykk"}
    else:
        proposed, explanation = mpc_lite_suggest(
            current_setpoint=current_setpoint,
            features=features,
            arx_coef=arx_coef,
            constraints=constraints,
            rate_limit=rate_limit,
        )

    st.write(f"**Nåværende utløpstemp:** {current_setpoint:.2f} °C")
    st.write(f"**Foreslått utløpstemp:** {proposed:.2f} °C")
    st.caption("Usikkerhetsbånd foreslås ±0.3 °C i operatørvisning.")

    if mode_ab == "AI (MPC‑lite)" and proposed != current_setpoint and undertrykk_ok:
        add_event(
            st.session_state.events,
            "INFO",
            f"AI foreslår endring {proposed - current_setpoint:+.2f} °C → {proposed:.2f} °C. Årsaker: innløp {features['innlopstemp']} °C, mode {features['mode']:.2f}, frisk {features['friskluftspjeld']}%, last {features['bunkerniva_pc']}%.",
        )

    with st.expander("Forklaring (bidrag)"):
        st.json(explanation)

st.markdown("---")

st.subheader("Design of Experiments (DoE)")
with st.form("doe_form"):
    st.write("Kjør små kontrollerte steg i rolige perioder for å lære sensitivitet pr. resept.")
    doe_step = st.number_input("Steg (°C)", value=0.5, step=0.1)
    doe_hold_min = st.number_input("Holdetid (min)", value=30, step=5)
    doe_run = st.form_submit_button("Planlegg DoE‑sekvens")
    if doe_run:
        add_event(st.session_state.events, "PLAN", f"DoE: steg {doe_step:+.2f} °C, hold {doe_hold_min} min. Logg fukt før/etter for å oppdatere k_dset.")
        st.success("DoE‑sekvens planlagt (operatørmelding sendt i hendelseslogg).")

st.subheader("Hendelser & eksport")
evt_df = pd.DataFrame(st.session_state.events)
st.dataframe(evt_df, use_container_width=True, height=220)

# Eksport
buf = io.StringIO()
evt_df.to_csv(buf, index=False)
st.download_button(
    label="Last ned hendelseslogg (CSV)",
    data=buf.getvalue().encode("utf-8"),
    file_name=f"arbor_ai_hendelser_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv",
)

st.caption(
    """
    Tips:
    - Koble appen til sanntid ved å erstatte demo‑datasettet med stream fra PLC/DCS.
    - Lær ekte sensitivitet (k_dset) fra DoE: 
      Δfukt / Δutløp → oppdater arx_coef["k_dset"].
    - Sett ulike guardrails pr. resept og driftsmodus (mode‑avhengig rate limit).
    - Bruk CUSUM/Drift‑deteksjon på residualer for å trigge re‑kalibrering.
    """
)
