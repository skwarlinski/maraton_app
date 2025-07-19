import os
import uuid
import json
import re
import pandas as pd
import streamlit as st
from pycaret.regression import load_model, predict_model
from langfuse import Langfuse
from langfuse.openai import OpenAI
from dotenv import load_dotenv

st.set_page_config(page_title="AI Półmaraton", layout="centered", page_icon="🧠")

# ------------------------------
# Konfiguracja środowiska i modeli
# ------------------------------
load_dotenv()
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST")

model = load_model("model_pycaret")

# ------------------------------
# Ustawienie session state
# ------------------------------
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# ------------------------------
# Funkcje pomocnicze
# ------------------------------
def save_api_key():
    st.session_state["api_key"] = st.session_state.input_api_key
    os.environ["OPENAI_API_KEY"] = st.session_state.api_key

def clear_input():
    st.session_state["user_input"] = ""

def extract_json(text):
    cleaned = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                return None
        return None

def convert_time_to_seconds(time_str):
    try:
        h, m, s = 0, 0, 0
        parts = time_str.strip().split(':')
        if len(parts) == 2:
            m, s = map(int, parts)
        elif len(parts) == 3:
            h, m, s = map(int, parts)
        else:
            return None
        return h * 3600 + m * 60 + s
    except:
        return None

# ------------------------------
# Logika główna
# ------------------------------
def calculate():
    user_input = st.session_state.user_input

    with st.spinner("⏳ Analizuję opis..."):
        if not user_input.strip():
            st.warning("Wprowadź dane w polu tekstowym")
            return

        trace = langfuse.trace(
            name="extract_user_data",
            user_id="user_" + str(uuid.uuid4())
        )
        span = trace.span(
            name="llm_parse",
            input=user_input
        )

        prompt_template = (
            "Na podstawie poniższego tekstu wyodrębnij dane użytkownika: wiek (int), płeć (str: 'mężczyzna' lub 'kobieta'), "
            "czas_5km (format mm:ss lub m:ss lub hh:mm:ss). Zwróć dane w formacie JSON.\n\n"
            "Przykład: {\"wiek\": 29, \"płeć\": \"mężczyzna\", \"czas_5km\": \"25:30\"}\n\n"
            f"Tekst:\n{user_input}"
        )

        try:
            response = openai.chat.completions.create(
                messages=[{"role": "user", "content": prompt_template}],
                model="gpt-4o"
            )
            
            result = response.choices[0].message.content

            generation = span.generation(
                name="extract_runner_json",
                input=user_input,
                output=result,
                metadata={"prompt_template": prompt_template},
                model="gpt-4o"
            )
            span.end()

            data = extract_json(result)

            if not data:
                st.error(f"❌ {result}")
                return

            brak_danych = []
            if not isinstance(data.get("wiek"), int):
                brak_danych.append("wiek")
            if data.get("płeć") not in ["mężczyzna", "kobieta"]:
                brak_danych.append("płeć")
            if not data.get("czas_5km"):
                brak_danych.append("czas_5km")

            if brak_danych:
                st.error(f"Brakuje danych: {', '.join(brak_danych)}")
                return
            
            else:
                st.toast("✅ Wykryto dane: **wiek + płeć + czas 5 km**")

            czas_5km_total_sec = convert_time_to_seconds(data["czas_5km"])
            if czas_5km_total_sec is None:
                st.error("Nieprawidłowy format czasu.")
                return

            tempo_sec = czas_5km_total_sec / 5
            df = pd.DataFrame([{
                "wiek": data["wiek"],
                "płeć_encoded": 1 if data["płeć"] == "mężczyzna" else 0,
                "tempo_sec": tempo_sec
            }])

            prediction = predict_model(model, data=df)
            czas = round(prediction["prediction_label"].values[0], 2)
            hours = int(czas // 3600)
            minutes = int((czas % 3600) // 60)
            seconds = int(czas % 60)
            formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"
            st.success(f"⏱️ Przewidywany czas: **{formatted_time}**")

        except Exception as e:
            st.error("Wystąpił błąd predykcji.")

# ------------------------------
# UI i interakcje
# ------------------------------
if not st.session_state.api_key:
    st.text_input("🔑 Klucz OpenAI API", type="password", key="input_api_key")
    st.button("Zatwierdź", on_click=save_api_key)
    st.info("Wpisz swój klucz OpenAI API, aby korzystać z aplikacji.")
else:
    langfuse = Langfuse()
    openai = OpenAI()

    with st.container():
        st.markdown("""
            <h1 style='text-align: center; font-size: 42px; color: #f9fafb;'>A I 🏃‍♂️ P Ó Ł M A R A T O N </h1>
            <h1 style='text-align: center; font-size: 36px; color: #f9fafb;'>Zaplanuj swój półmaraton</h1>
            <hr style='border: 1px solid gray;'/>
        """, unsafe_allow_html=True
        )

    st.markdown("### 💬 Powiedz mi coś o sobie i swoich wynikach")

    st.text_area(
        "Wpisz wiek, płeć, jeśli masz jakieś rekordy podziel się z nami (np. 5 km w 36 minut)",
        key="user_input"
    )

    st.markdown("<hr style='border: 1px solid #444;'>", unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.button("Oblicz", type="primary", on_click=calculate)

    with col6:
        st.button("Wyczyść dane", type="tertiary", on_click=clear_input)