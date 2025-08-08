# arbor_ai_app.py
# -*- coding: utf-8 -*-
"""
Arbor AI – samlet Streamlit-app (én fil) med:
- Modusdeteksjon (Hombak/Maier)
- Autokalibrering av fuktsensor (RLS m/glemselsfaktor)
- Dødtidsestimat + forslag til neste prøvetidspunkt
- Guardrails (hard/soft) per resept + rate-limit
- «MPC-lite» (ett-trinns fremoversyn / ARX) m/ forklaring
- DoE-steg (kontrollerte små steg) i rolige perioder
- KPI-dashboard + A/B-evaluering pr. skift
- Hendelseslogg og eksport
- OCR for håndskrevne/avfotograferte loggark (TrOCR → fallback pytesseract)
"""

import io
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# --- Opsjonelle imports for OCR ---
try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None  # type: ignore

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # type: ignore
    import torch  # type: ignore
except Exception:
    TrOCRProcessor = None  # type: ignore
    VisionEncoderDecoderModel = None  # type: ignore
    torch = None  # type: ignore


# ============ Utils ============ #

@st.cache_data
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        df["timestamp"] = pd.date_range(end=datetime.now(), periods=len(df), freq="5min")
    return df.sort_values("timestamp").reset_index(drop=True)


def ewma(series: pd.Series, alpha: float = 0.3) -> pd.Series:
    out, s = [], None
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
        x = np.array([1.0, sensor])
        y = manual
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
    return float(np.clip(maier_pc / total, 0.0, 1.0))  # 0=Hombak-tung, 1=Maier-tung


def estimate_dead_time_minutes(df: pd.DataFrame, col_input: str, col_output: str, search_max_min: int = 60) -> int:
    """Grovt dødtidsestimat via krysskorrelasjon på avledede endringer."""
    x = df[col_input].diff().fillna(0).values
    y = df[col_output].diff().fillna(0).values
    n = min(len(x), len(y))
    x = x[-n:]
    y = y[-n:]
    best_lag, best_corr = 0, -9e9
    for lag in range(0, search_max_min + 1):
        if lag == 0:
            corr = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else 0
        else:
            corr = np.corrcoef(x[:-lag], y[lag:])[0, 1] if np.std(x[:-lag]) > 0 and np.std(y[lag:]) > 0 else 0
        if np.isnan(corr):
            corr = 0
        if corr > best_corr:
            best_corr, best_lag = corr, lag
    # antatt 5 min pr. rad; juster ved annen frekvens
    return int(best_lag * 5)


def mpc_lite_suggest(
    current_setpoint: float,
    features: Dict[str, float],
    arx_coef: Dict[str, float],
    constraints: Dict[str, float],
    rate_limit: float = 1.5,
) -> Tuple[float, Dict[str, float]]:
    """Ett-trinns fremoversyn: prognose av fukt og valg av nytt setpunkt."""
    y_hat = (
        arx_coef["bias"]
        + arx_coef["k_setpoint"] * current_setpoint
        + arx_coef["k_innlop"] * features.get("innlopstemp", 0.0)
        + arx_coef["k_mode"] * features.get("mode", 0.0)
        + arx_coef["k_friskluft"] * features.get("friskluftspjeld", 0.0)
        + arx_coef["k_last"] * features.get("bunkerniva_pc", 50.0)
    )
    target = features.get("target_fukt", 1.20)
    sens = arx_coef.get("k_dset", -0.10)  # %-poeng fukt per +1 °C utløp (negativ typisk)
    delta_needed = (target - y_hat) / sens if abs(sens) > 1e-6 else 0.0
    delta_clamped = float(np.clip(delta_needed, -rate_limit, rate_limit))
    proposed = float(np.clip(current_setpoint + delta_clamped, constraints["hard_min"], constraints["hard_max"]))
    explanation = {
        "prognose_fukt": float(y_hat),
        "mål_fukt": target,
        "estimert_sensitivitet(%) per °C": sens,
        "rå_delta": float(delta_needed),
        "etter_rate_limit": delta_clamped,
        "foreslått_setpunkt": proposed,
        "bidrag_innlop": float(arx_coef["k_innlop"] * features.get("innlopstemp", 0.0)),
        "bidrag_mode": float(arx_coef["k_mode"] * features.get("mode", 0.0)),
        "bidrag_friskluft": float(arx_coef["k_friskluft"] * features.get("friskluftspjeld", 0.0)),
        "bidrag_last": float(arx_coef["k_last"] * features.get("bunkerniva_pc", 50.0)),
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


# ============ OCR helpers ============ #
@st.cache_resource(show_spinner=False)
def load_trocr_small():
    """Laster en liten TrOCR-modell hvis tilgjengelig. Returnerer (processor, model) eller (None, None)."""
    if TrOCRProcessor is None or VisionEncoderDecoderModel is None:
        return None, None
    try:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
        model.eval()
        return processor, model
    except Exception:
        return None, None


def ocr_with_trocr(img: "Image.Image") -> Optional[str]:
    if TrOCRProcessor is None or VisionEncoderDecoderModel is None or torch is None:
        return None
    processor, model = load_trocr_small()
    if processor is None:
        return None
    try:
        inputs = processor(img, return_tensors="pt")
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text
    except Exception:
        return None


def ocr_with_tesseract(img: "Image.Image") -> Optional[str]:
    if pytesseract is None:
        return None
    try:
        return pytesseract.image_to_string(img, lang="eng+nor")
    except Exception:
        return None


def ocr_read_image(image_bytes: bytes, backend: str = "auto") -> str:
    assert Image is not None, "Pillow (PIL) mangler. Installer pillow."
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_small = img.copy()
    w, h = img_small.size
    if max(w, h) > 1800:
        scale = 1800 / max(w, h)
        img_small = img_small.resize((int(w*scale), int(h*scale)))

    text = None
    if backend in ("auto", "trocr"):
        text = ocr_with_trocr(img_small)
    if (text is None or len(text.strip()) == 0) and backend in ("auto", "tesseract"):
        text = ocr_with_tesseract(img_small)
    return text or ""


def _num(x: str) -> Optional[float]:
    if x is None:
        return None
    x = x.replace(" ", "").replace(",", ".")
    try:
        return float(x)
    except Exception:
        return None


def parse_log_text(text: str) -> Dict[str, Optional[float]]:
    import re
    s = text.lower()
    s = s.replace("ø", "o").replace("å", "a").replace("æ", "ae")
    def rex(p):
        m = re.search(p, s, re.DOTALL)
        return _num(m.group(1)) if m else None

    fields = {
        "utlopstemp": rex(r"utlopstemp[^0-9]*([0-9]{2,3}(?:[.,][0-9])?)"),
        "innlopstemp": rex(r"innlopstemp[^0-9]*([0-9]{2,3}(?:[.,][0-9])?)"),
        "friskluftspjeld": rex(r"frisk[^0-9]*([0-9]{1,3}(?:[.,][0-9])?)"),
        "hombak_pc": rex(r"hombak[^0-9]*([0-9]{1,3})\s?%"),
        "maier_pc": rex(r"maier[^0-9]*([0-9]{1,3})\s?%"),
        "bunkerniva_pc": rex(r"bun(?:ker|kerniva)[^0-9]*([0-9]{1,3})"),
        "fukt_manuell": rex(r"fukt(?:ighet)?[^a-z0-9]{0,10}(?:manuell|prove)?[^0-9]*([0-9](?:[.,][0-9]{1,2})?)"),
        "fukt_sensor": rex(r"fukt[^\n]*sensor[^0-9]*([0-9](?:[.,][0-9]{1,2})?)"),
        "trykk_nedre_ovn": rex(r"(?:trykk|undertrykk)[^\n]*nedre[^0-9]*([0-9]{2,4})"),
    }
    return fields


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
    st.header("A/B-oppsett")
    mode_ab = st.radio("Kjøring", ["Baseline (manuell)", "AI (MPC-lite)"])

    st.markdown("---")
    csv = st.file_uploader("Last opp CSV-logg (valgfritt)")

# Session state
if "cal" not in st.session_state:
    st.session_state.cal = RLSCalibrator(lam=0.99)
if "events" not in st.session_state:
    st.session_state.events = []

# Data init (demo hvis ingen CSV)
if csv is not None:
    df = load_csv(csv)
else:
    n = 200
    ts = pd.date_range(end=datetime.now(), periods=n, freq="5min")
    rng = np.random.default_rng(42)
    utlop = 135 + np.cumsum(rng.normal(0, 0.05, n))
    innlop = 180 + rng.normal(0, 5, n)
    frisk = np.clip(30 + rng.normal(0, 3, n), 10, 50)
    hombak = np.clip(70 + rng.normal(0, 5, n), 0, 100)
    maier = 100 - hombak
    bunk = np.clip(50 + rng.normal(0, 10, n), 10, 90)
    f_true = 1.2 + (-0.10)*(utlop - 135) + 0.002*(innlop - 180) + 0.15*(maier/100) + rng.normal(0, 0.05, n)
    f_sens = f_true + rng.normal(0, 0.15, n) + 0.3
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

# Beregn mode + autokalibrering
modes, f_corr = [], []
for _, row in df.iterrows():
    mode = detect_mode(row.get("hombak_pc", 50.0), row.get("maier_pc", 50.0))
    modes.append(mode)
    if not pd.isna(row.get("fukt_manuell", np.nan)):
        st.session_state.cal.lam = lam
        st.session_state.cal.update(row["fukt_sensor"], row["fukt_manuell"])
    f_corr.append(st.session_state.cal.correct(row["fukt_sensor"]))
df["mode"], df["fukt_corr"] = modes, f_corr

# Dødtid + KPI
dead_min = estimate_dead_time_minutes(df, "utlopstemp", "fukt_corr", search_max_min=60)
next_probe_eta = df["timestamp"].iloc[-1] + timedelta(minutes=dead_min)
_kpis = compute_kpis(df, target_fukt, window="7D")

# --------- Faner ---------
tab_dash, tab_kontroll, tab_doe, tab_kpi, tab_logg, tab_ocr, tab_innst = st.tabs(
    ["📊 Dashboard", "🎛️ Kontroll", "🧪 DoE", "🎯 KPI", "📝 Logg", "📸 OCR", "⚙️ Innstillinger"]
)

with tab_dash:
    st.subheader("Trender")
    st.line_chart(
        df.set_index("timestamp")[["fukt_sensor", "fukt_corr", "fukt_manuell"]].rename(
            columns={"fukt_sensor": "Fukt sensor", "fukt_corr": "Fukt korrigert", "fukt_manuell": "Fukt manuell"}
        )
    )
    st.line_chart(df.set_index("timestamp")[["utlopstemp", "innlopstemp", "friskluftspjeld"]])

with tab_kontroll:
    st.subheader("Forslag og guardrails")
    current = df.iloc[-1]
    constraints = {"hard_min": float(hard_min), "hard_max": float(hard_max)}
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
        add_event(st.session_state.events, "ADVARSEL",
                  f"Undertrykk {current.get('trykk_nedre_ovn', np.nan):.0f} Pa < {undertrykk_min} Pa – hold setpunkt!")
        st.warning("Undertrykk under grense – AI fryses midlertidig.")
        proposed, explanation = current_setpoint, {"grunn": "undertrykk"}
    else:
        proposed, explanation = mpc_lite_suggest(
            current_setpoint=current_setpoint,
            features=features,
            arx_coef=arx_coef,
            constraints=constraints,
            rate_limit=rate_limit,
        )
        if mode_ab == "AI (MPC-lite)" and proposed != current_setpoint:
            add_event(
                st.session_state.events,
                "INFO",
                f"AI foreslår endring {proposed - current_setpoint:+.2f} °C → {proposed:.2f} °C. "
                f"Årsaker: innløp {features['innlopstemp']} °C, mode {features['mode']:.2f}, "
                f"frisk {features['friskluftspjeld']}%, last {features['bunkerniva_pc']}%.",
            )

    c1, c2, c3 = st.columns(3)
    c1.metric("Nåværende utløp", f"{current_setpoint:.2f} °C")
    c2.metric("Foreslått utløp", f"{proposed:.2f} °C")
    c3.metric("Dødtid (est.)", f"~{dead_min} min")

    with st.expander("Forklaring (bidrag)"):
        st.json(explanation)

with tab_doe:
    st.subheader("Design of Experiments (DoE)")
    with st.form("doe_form"):
        st.write("Kjør små kontrollerte steg i rolige perioder for å lære sensitivitet pr. resept.")
        doe_step = st.number_input("Steg (°C)", value=0.5, step=0.1)
        doe_hold_min = st.number_input("Holdetid (min)", value=30, step=5)
        if st.form_submit_button("Planlegg DoE-sekvens"):
            add_event(st.session_state.events, "PLAN",
                      f"DoE: steg {doe_step:+.2f} °C, hold {doe_hold_min} min. "
                      f"Logg fukt før/etter for å oppdatere k_dset.")
            st.success("DoE-sekvens planlagt (operatørmelding i hendelseslogg).")

with tab_kpi:
    st.subheader("KPI siste 7 dager")
    c1, c2, c3 = st.columns(3)
    c1.metric("Std. fukt", f"{_kpis.get('std_fukt', float('nan')):.3f}")
    c2.metric("Innenfor ±0.1pp", f"{_kpis.get('andel_innenfor_±0.1pp(%)', float('nan')):.1f}%")
    c3.metric("Punkter", f"{_kpis.get('antall_punkt', 0)}")

with tab_logg:
    st.subheader("Hendelser & eksport")
    evt_df = pd.DataFrame(st.session_state.events)
    st.dataframe(evt_df, use_container_width=True, height=280)
    buf = io.StringIO(); evt_df.to_csv(buf, index=False)
    st.download_button(
        label="Last ned hendelseslogg (CSV)",
        data=buf.getvalue().encode("utf-8"),
        file_name=f"arbor_ai_hendelser_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

with tab_ocr:
    st.subheader("OCR – les håndskrevet logg fra bilde")
    st.caption("Backend: prøver TrOCR (håndskrift) hvis tilgjengelig, ellers pytesseract. Du kan bytte under.")
    backend = st.radio("OCR-motor", ["auto", "trocr", "tesseract"], horizontal=True)
    img_file = st.file_uploader("Last opp bilde av logg (jpg/png)", type=["jpg", "jpeg", "png"])

    if img_file is not None:
        img_bytes = img_file.read()
        with st.spinner("Leser tekst fra bildet…"):
            text = ocr_read_image(img_bytes, backend=backend)
        if len(text.strip()) == 0:
            st.error("Fant ikke tekst. Prøv et skarpere bilde eller en annen backend.")
        else:
            st.success("Tekst funnet!")
            st.text_area("Råtekst fra OCR", value=text, height=180)
            parsed = parse_log_text(text)
            st.write("**Tolkede felt (kan redigeres før lagring):**")
            cols = st.columns(3)
            with cols[0]:
                utlop = st.number_input("Utløpstemp (°C)", value=float(parsed.get("utlopstemp") or 135.0))
                innlop = st.number_input("Innløpstemp (°C)", value=float(parsed.get("innlopstemp") or 180.0))
                frisk = st.number_input("Friskluft (%)", value=float(parsed.get("friskluftspjeld") or 30.0))
            with cols[1]:
                homb = st.number_input("Hombak (%)", value=float(parsed.get("hombak_pc") or 60.0))
                maie = st.number_input("Maier (%)", value=float(parsed.get("maier_pc") or 40.0))
                bunk = st.number_input("Bunkernivå (%)", value=float(parsed.get("bunkerniva_pc") or 50.0))
            with cols[2]:
                fman = st.number_input("Fukt manuell (%)", value=float(parsed.get("fukt_manuell") or 1.20))
                fsens = st.number_input("Fukt sensor (%)", value=float(parsed.get("fukt_sensor") or 1.50))
                trykk = st.number_input("Trykk nedre ovn (Pa)", value=float(parsed.get("trykk_nedre_ovn") or 275.0))

            new_row = pd.DataFrame([{
                "timestamp": datetime.now(),
                "utlopstemp": utlop,
                "innlopstemp": innlop,
                "friskluftspjeld": frisk,
                "hombak_pc": homb,
                "maier_pc": maie,
                "bunkerniva_pc": bunk,
                "fukt_sensor": fsens,
                "fukt_manuell": fman,
                "trykk_nedre_ovn": trykk,
            }])
            st.dataframe(new_row, use_container_width=True)

            if st.button("Legg inn raden i datasettet"):
                if not pd.isna(fman):
                    st.session_state.cal.lam = lam
                    st.session_state.cal.update(fsens, fman)
                df = pd.concat([df, new_row], ignore_index=True)
                df.loc[df.index[-1], "mode"] = detect_mode(homb, maie)
                df.loc[df.index[-1], "fukt_corr"] = st.session_state.cal.correct(fsens)
                add_event(st.session_state.events, "OCR", "La til én rad fra håndskrevet logg (via OCR)")
                st.success("Raden er lagt inn. Gå til Dashboard/Kontroll for å se oppdatert graf.")

with tab_innst:
    st.subheader("Innstillinger")
    st.write("Flytt gjerne flere innstillinger hit etter hvert (guardrails per resept, A/B-bryter, mm).")

st.caption(
    """
    Tips:
    - Koble appen til sanntid ved å erstatte demo-datasettet med stream fra PLC/DCS.
    - Lær ekte sensitivitet (k_dset) fra DoE: Δfukt / Δutløp → oppdater arx_coef["k_dset"].
    - Sett ulike guardrails pr. resept og driftsmodus (mode-avhengig rate limit).
    - Bruk drift-deteksjon (CUSUM) på residualer for å trigge re-kalibrering.
    - OCR: For best håndskrift, bruk TrOCR. Fallback: pytesseract.
    """
)
