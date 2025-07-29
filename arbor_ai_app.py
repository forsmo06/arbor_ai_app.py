"""
Arbor AI – process monitoring and adjustment tool
=================================================

Dette Streamlit-programmet er en prototyp som viser hvordan operatører kan
loggføre prosessdata, visualisere trender, få AI-baserte justeringsforslag
og gjøre enkel OCR av opplastede bilder. Kjør koden med:
    streamlit run arbor_ai_app.py

Du må ha installert avhengighetene:
    pip install streamlit pandas numpy scikit-learn plotly easyocr

Opprettet: juli 2025
"""

import datetime
import io
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px

try:
    import streamlit as st  # type: ignore
except ImportError:
    raise ImportError(
        "Streamlit is required to run this application. Install it via `pip install streamlit`."
    )

# OCR-biblioteker (valgfritt)
try:
    import easyocr  # type: ignore
except ImportError:
    easyocr = None  # type: ignore

try:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
except ImportError:
    pytesseract = None  # type: ignore

from sklearn.linear_model import LinearRegression  # type: ignore


def initialise_data() -> pd.DataFrame:
    """Initialiser DataFrame med riktige kolonner."""
    return pd.DataFrame(
        columns=[
            "timestamp",
            "temperature",
            "airflow",
            "bunker_level",
            "humidity",
            "manual_feed",
            "ai_feed",
        ]
    )


def simulate_sensor_reading() -> Tuple[float, float, float, float, float]:
    """Generer tilfeldige sensoravlesninger til demo."""
    temperature = np.random.uniform(60, 90)      # °C
    airflow = np.random.uniform(1000, 2000)      # m³/h
    bunker = np.random.uniform(0, 100)           # %
    humidity = np.random.uniform(5, 15)          # %
    manual_feed = np.random.uniform(150, 250)    # kg/h
    return temperature, airflow, bunker, humidity, manual_feed


def update_ai_model(data: pd.DataFrame) -> LinearRegression:
    """Tren en enkel lineær regresjonsmodell på historiske data."""
    model = LinearRegression()
    if len(data) < 5:
        # tren en dummy-modell hvis vi ikke har nok data
        X_dummy = np.random.rand(5, 4)
        y_dummy = np.random.rand(5)
        model.fit(X_dummy, y_dummy)
        return model
    X = data[["temperature", "airflow", "bunker_level", "humidity"]].values
    y = data["manual_feed"].values
    model.fit(X, y)
    return model


def suggest_ai_feed(model: LinearRegression, sample: np.ndarray) -> float:
    """Gi en feed-rate basert på den lærte modellen."""
    return float(model.predict(sample.reshape(1, -1))[0])


def perform_easyocr(img_bytes: bytes) -> str:
    """Utfør OCR med EasyOCR."""
    if easyocr is None:
        return "EasyOCR er ikke installert. Installer via pip install easyocr."
    image = Image.open(io.BytesIO(img_bytes))  # type: ignore[name-defined]
    reader = easyocr.Reader(["en"])
    result = reader.readtext(np.array(image), detail=0)
    return "\n".join(result)


def perform_pytesseract(img_bytes: bytes) -> str:
    """Utfør OCR med pytesseract."""
    if pytesseract is None:
        return "pytesseract er ikke installert. Installer via pip install pytesseract."
    image = Image.open(io.BytesIO(img_bytes))
    try:
        text = pytesseract.image_to_string(image)
    except Exception as e:
        text = f"Feil ved Tesseract: {e}"
    return text


def main() -> None:
    """Hovedfunksjon for Streamlit-appen."""
    st.set_page_config(page_title="Arbor AI App", layout="wide")
    st.title("Arbor AI: Prosessovervåkning og justering")

    st.markdown(
        """
        Denne prototypen viser hvordan en AI‑drevet applikasjon kan støtte operatører hos Arbor i å loggføre prosessdata,
        visualisere trender, motta justeringsforslag og redusere energiforbruk og svinn.  Ved å analysere historiske data
        kan modellen foreslå optimale innstillinger og gi varsler om avvik før de påvirker produksjonen.  Forskning viser at
        prediktivt vedlikehold kan redusere uplanlagt nedetid med opptil **50 %**:contentReference[oaicite:0]{index=0}.
        Lignende AI‑tilnærminger til energistyring kan redusere energikostnader med **omtrent 20 %**:contentReference[oaicite:1]{index=1}.
        """
    )

    # Initier data hvis det ikke finnes fra før
    if "data" not in st.session_state:
        st.session_state["data"] = initialise_data()

    # Sidebar for logging
    st.sidebar.header("Loggføring")
    if st.sidebar.button("Simuler sensormåling"):
        temp, air, bunk, hum, m_feed = simulate_sensor_reading()
        new_record = {
            "timestamp": datetime.datetime.now(),
            "temperature": temp,
            "airflow": air,
            "bunker_level": bunk,
            "humidity": hum,
            "manual_feed": m_feed,
            "ai_feed": np.nan,
        }
        st.session_state["data"] = pd.concat(
            [
                st.session_state["data"],
                pd.DataFrame([new_record]),
            ],
            ignore_index=True,
        )

    st.sidebar.subheader("Manuell registrering")
    manual_temp = st.sidebar.number_input("Utløpstemperatur (°C)", 0.0, 120.0, 75.0)
    manual_air = st.sidebar.number_input("Friskluftmengde (m³/h)", 0.0, 5000.0, 1500.0)
    manual_bunker = st.sidebar.number_input("Bunkernivå (%)", 0.0, 100.0, 50.0)
    manual_humidity = st.sidebar.number_input("Fuktighet (%)", 0.0, 100.0, 10.0)
    manual_feed = st.sidebar.number_input("Matingshastighet (kg/h)", 0.0, 1000.0, 200.0)

    if st.sidebar.button("Legg til manuell logg"):
        new_manual = {
            "timestamp": datetime.datetime.now(),
            "temperature": manual_temp,
            "airflow": manual_air,
            "bunker_level": manual_bunker,
            "humidity": manual_humidity,
            "manual_feed": manual_feed,
            "ai_feed": np.nan,
        }
        st.session_state["data"] = pd.concat(
            [
                st.session_state["data"],
                pd.DataFrame([new_manual]),
            ],
            ignore_index=True,
        )

    # Vis data og tren modell
    data = st.session_state["data"]
    if not data.empty:
        st.subheader("Historiske data")
        st.dataframe(data)

        model = update_ai_model(data)
        rows_to_update = data["ai_feed"].isna()
        for idx in data[rows_to_update].index:
            sample = data.loc[idx, ["temperature", "airflow", "bunker_level", "humidity"]].values
            ai_prediction = suggest_ai_feed(model, np.array(sample, dtype=float))
            st.session_state["data"].at[idx, "ai_feed"] = ai_prediction

        # Plot prosessdata
        st.subheader("Prosessvariabler over tid")
        fig = px.line(
            st.session_state["data"],
            x="timestamp",
            y=["temperature", "airflow", "bunker_level", "humidity"],
            labels={"value": "Måleverdi", "timestamp": "Tid", "variable": "Parameter"},
            title="Historiske prosessvariabler",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Plot manual vs AI
        st.subheader("Sammenligning av manuell vs AI‑styrt mating")
        comp_fig = px.line(
            st.session_state["data"],
            x="timestamp",
            y=["manual_feed", "ai_feed"],
            labels={"value": "Mating (kg/h)", "timestamp": "Tid", "variable": "Type"},
            title="Manuell mating vs AI‑forslag",
        )
        st.plotly_chart(comp_fig, use_container_width=True)

        deviations = np.abs(data["manual_feed"] - data["ai_feed"]) / data["manual_feed"]
        high_dev = deviations > 0.15
        if any(high_dev):
            st.warning(
                f"Det er {high_dev.sum()} logg(r) med mer enn 15 % avvik mellom manuell og AI‑mating."
            )

    # OCR
    st.subheader("Bildeopplasting og avlesning")
    st.write(
        "Last opp et bilde av en manuell måling (f.eks. foto av et fuktighetsinstrument) for å lese av verdien automatisk."
    )
    uploaded_image = st.file_uploader("Last opp bilde", type=["png", "jpg", "jpeg"])
    ocr_engine = st.selectbox("Velg OCR‑motor", ["Ingen", "EasyOCR", "Tesseract"])
    if uploaded_image is not None and ocr_engine != "Ingen":
        img_bytes = uploaded_image.read()
        with st.spinner("Kjører OCR..."):
            if ocr_engine == "EasyOCR":
                text_out = perform_easyocr(img_bytes)
            else:
                text_out = perform_pytesseract(img_bytes)
        st.text_area("Ekstrahert tekst", value=text_out, height=200)

    # Prediktivt vedlikehold
    st.subheader("Prediktivt vedlikehold (eksempel)")
    st.write(
        "Denne seksjonen illustrerer en enkel varslingsmekanisme. Vi beregner et glidende "
        "gjennomsnitt av bunkernivået og markerer hvis trenden faller under 20 %."
    )
    if not data.empty:
        df = st.session_state["data"].copy()
        df["bunker_sma"] = df["bunker_level"].rolling(window=5, min_periods=1).mean()
        fig_bunker = px.line(
            df,
            x="timestamp",
            y=["bunker_level", "bunker_sma"],
            labels={"value": "Bunkernivå (%)", "timestamp": "Tid", "variable": "Serie"},
            title="Bunkernivå og glidende gjennomsnitt",
        )
        st.plotly_chart(fig_bunker, use_container_width=True)
        if any(df["bunker_sma"] < 20):
            st.error(
                "Glidende gjennomsnitt av bunkernivå er under 20 %. Planlegg vedlikehold snart!"
            )


if __name__ == "__main__":
    main()
