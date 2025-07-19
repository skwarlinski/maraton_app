
# 🧠 AI Półmaraton
Aplikacja Streamlit do przewidywania czasu ukończenia półmaratonu na podstawie wieku, płci i czasu na 5 km. Wykorzystuje modele ML oraz integrację z OpenAI i Langfuse.

---

## 📦 Wymagania

- Python 3.8+
- Plik `requirements.txt` z zależnościami
- Klucze API do OpenAI i Langfuse

---

## ⚙️ Konfiguracja środowiska

1. **Klonuj repozytorium i przejdź do katalogu projektu:**
   ```bash
   git clone <adres-respo>
   cd maraton_app
   ```

2. **Utwórz i aktywuj wirtualne środowisko (opcjonalnie):**
   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/Mac
   venv\Scripts\activate       # Windows
   ```

3. **Zainstaluj zależności:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Przygotuj plik `.env` (lokalnie):**
   ```
   LANGFUSE_PUBLIC_KEY=pk-...
   LANGFUSE_SECRET_KEY=sk-...
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```
   > **Uwaga:** Na Digital Ocean App Platform dodaj te zmienne przez panel (Settings → Environment Variables).

---

## 🚀 Deployment na Digital Ocean App Platform

1. Zaloguj się do Digital Ocean i utwórz nową aplikację (App Platform).  
2. Wskaż repozytorium z projektem.  
3. W sekcji **Build & Run** ustaw:
   - **Build Command:** *(opcjonalnie puste)*
   - **Run Command:**
     ```bash
     streamlit run app.py --server.port $PORT --server.address 0.0.0.0
     ```
4. Dodaj zmienne środowiskowe w panelu:
   - `LANGFUSE_PUBLIC_KEY`
   - `LANGFUSE_SECRET_KEY`
   - `LANGFUSE_HOST`
5. Upewnij się, że plik `model_pycaret.pkl` jest w repozytorium.  
6. Zatwierdź i uruchom aplikację.

---

## 🛠 Troubleshooting

- **Błąd: Brak kluczy API** – Upewnij się, że zmienne środowiskowe są poprawnie ustawione.  
- **Błąd: Model not found** – Sprawdź, czy plik `.pkl` jest w ścieżkach w `app.py`.  
- **Błąd: Port already in use** – Zmień port lub zatrzymaj inny proces.  
- **Błąd: ImportError** – Sprawdź, czy wszystkie zależności znajdują się w `requirements.txt` i zostały zainstalowane.

---

## 📬 Kontakt

Masz pytania lub sugestie? Odezwij się!

- 💼 [LinkedIn](https://www.linkedin.com/in/hubert-skwarlinski-895437368/)
- 💻 [GitHub](https://github.com/skwarlinski)
- ✉️ Email: [skwarlinskihubert@gmail.com](mailto:skwarlinskihubert@gmail.com)
