"""
app.py - GUI Streamlit per visualizzare l'output di Lesk (classico / avanzato).
Avvio:  streamlit run app.py
"""
import re

import pandas as pd
import streamlit as st

import lesk  # <-- qui puoi sostituire con i tuoi moduli

st.set_page_config(page_title="WSD con Lesk", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Carico le risorse NLTK...")
def init() -> bool:
    lesk.ensure_nltk_data()
    return True


init()


# ------------------------------------------------------------------ helper UI
def evidenzia(testo: str, overlap: set[str]) -> str:
    """Evidenzia in arancione le parole che (anche lemmatizzate) sono in overlap."""
    def sostituisci(m: re.Match) -> str:
        parola = m.group(0)
        if lesk.forme_possibili(parola) & overlap:
            return f":orange[**{parola}**]"
        return parola

    return re.sub(r"[A-Za-z]+", sostituisci, testo)


def mostra_risultato(res: lesk.LeskResult, frase: str) -> None:
    best = res.best
    if best is None:
        st.warning(f"Nessun synset WordNet trovato per «{res.word}».")
        return

    s = best.synset
    st.success(f"**{s.name()}**  ·  score {best.score}")
    st.write(s.definition())
    if s.examples():
        st.caption("Esempi: " + "  ·  ".join(f"_{e}_" for e in s.examples()))
    if res.pos:
        st.caption(f"Filtro POS applicato: `{res.pos}`")

    overlap = set(best.overlap)
    st.markdown("**Overlap evidenziato**")
    st.markdown("Frase: " + evidenzia(frase, overlap))
    glossa = s.definition()
    if s.examples():
        glossa += " | " + " | ".join(s.examples())
    st.markdown("Glossa: " + evidenzia(glossa, overlap))
    st.caption("Parole in comune: " + (", ".join(best.overlap) or "nessuna"))

    df = pd.DataFrame(
        [
            {
                "Synset": r.synset.name(),
                "POS": r.synset.pos(),
                "Score": r.score,
                "Overlap": ", ".join(r.overlap),
                "Definizione": r.synset.definition(),
            }
            for r in res.ranking
        ]
    )
    st.markdown("**Tutti i synset candidati**")
    st.dataframe(df, hide_index=True)
    st.bar_chart(df.set_index("Synset")["Score"])


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Input")
    frase = st.text_input("Frase", "I went to the bank to deposit my money")
    parola = st.text_input("Parola target", "bank")
    modalita = st.radio(
        "Algoritmo",
        ["Lesk classico", "Lesk avanzato", "Confronto affiancato"],
        index=2,
    )
    mostra_pos = st.checkbox("Mostra POS tagging della frase", value=True)
    st.caption("Premi Invio dopo aver modificato un campo per aggiornare.")

# ----------------------------------------------------------------- pagina
st.title("🔎 Word Sense Disambiguation con Lesk")

frase, parola = frase.strip(), parola.strip()
if not frase or not parola:
    st.info("Inserisci una frase e una parola target nella barra laterale.")
    st.stop()

if parola.lower() not in frase.lower():
    st.warning("La parola target non compare nella frase: il contesto potrebbe non avere senso.")

if mostra_pos:
    with st.expander("POS tagging della frase", expanded=False):
        analisi = lesk.analizza_frase(frase)
        st.dataframe(
            pd.DataFrame(analisi, columns=["Token", "Tag Penn", "POS WordNet"]),
            hide_index=True,
        )

if modalita == "Lesk classico":
    mostra_risultato(lesk.lesk_classico(frase, parola), frase)

elif modalita == "Lesk avanzato":
    mostra_risultato(lesk.lesk_avanzato(frase, parola), frase)

else:
    r1 = lesk.lesk_classico(frase, parola)
    r2 = lesk.lesk_avanzato(frase, parola)

    if r1.best and r2.best:
        if r1.best.synset == r2.best.synset:
            st.success(f"Entrambi gli algoritmi scelgono **{r1.best.synset.name()}**.")
        else:
            st.error(
                f"Disaccordo: classico → **{r1.best.synset.name()}**, "
                f"avanzato → **{r2.best.synset.name()}**."
            )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Lesk classico")
        mostra_risultato(r1, frase)
    with col2:
        st.subheader("Lesk avanzato")
        mostra_risultato(r2, frase)
