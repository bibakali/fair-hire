"""
streamlit_app.py
Interface utilisateur Fair Hire
"""

import streamlit as st
import os
import tempfile
from src.agent import run_pipeline, tool_detect_bias
from src.ingestion import load_and_split

# ---------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------

st.set_page_config(
    page_title="Fair Hire — Assistant RH Inclusif",
    page_icon="⚖️",
    layout="wide"
)

# ---------------------------------------------------------------
# CSS personnalisé
# ---------------------------------------------------------------

st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4e79;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .score-good { color: #27ae60; font-weight: bold; }
    .score-warning { color: #f39c12; font-weight: bold; }
    .score-bad { color: #e74c3c; font-weight: bold; }
    .section-header {
        background-color: #f0f4f8;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------

st.markdown('<p class="main-title">⚖️ Fair Hire</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Assistant RH basé sur l\'IA — '
    'Détection de biais & Matching CV/Offre d\'emploi</p>',
    unsafe_allow_html=True
)
st.divider()

# ---------------------------------------------------------------
# Sidebar — Mode d'utilisation
# ---------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")
    mode = st.radio(
        "Mode d'analyse",
        ["🔍 Analyse de biais", "🎯 Matching CV/Offre", "📊 Pipeline complet"],
        index=0
    )
    st.divider()
    st.markdown("### 📖 Guide rapide")
    st.markdown("""
    **Analyse de biais** : Upload une offre d'emploi pour détecter les formulations discriminantes.

    **Matching CV/Offre** : Compare un CV avec une offre et obtiens un rapport détaillé.

    **Pipeline complet** : Les deux analyses en une seule fois.
    """)
    st.divider()
    st.markdown("Built with LangChain · ChromaDB · Mistral")

# ---------------------------------------------------------------
# Mode 1 : Analyse de biais uniquement
# ---------------------------------------------------------------

if "Analyse de biais" in mode:
    st.header("🔍 Analyse de biais dans une offre d'emploi")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📄 Upload de l'offre")
        job_file = st.file_uploader(
            "Dépose ton offre d'emploi (PDF)",
            type=["pdf"],
            key="bias_job"
        )

        if job_file:
            st.success(f"✅ Fichier chargé : {job_file.name}")

            if st.button("🚀 Analyser les biais", type="primary"):
                with st.spinner("Analyse en cours..."):
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp:
                        tmp.write(job_file.read())
                        tmp_path = tmp.name

                    try:
                        chunks = load_and_split(tmp_path)
                        job_text = " ".join(chunks)
                        bias_report, bias_score = tool_detect_bias(job_text)
                        st.session_state["bias_report"] = bias_report
                        st.session_state["bias_score"] = bias_score
                    except Exception as e:
                        st.error(f"Erreur : {e}")
                    finally:
                        os.unlink(tmp_path)

    with col2:
        st.subheader("📊 Résultats")

        if "bias_score" in st.session_state:
            score = st.session_state["bias_score"]

            # Jauge visuelle du score
            if score == 0:
                st.markdown(
                    '<p class="score-good">✅ Aucun biais détecté</p>',
                    unsafe_allow_html=True
                )
            elif score < 0.05:
                st.markdown(
                    f'<p class="score-warning">⚠️ Biais faibles — Score : {score}</p>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<p class="score-bad">🚨 Biais significatifs — Score : {score}</p>',
                    unsafe_allow_html=True
                )

            st.progress(min(score * 10, 1.0))
            st.code(st.session_state["bias_report"])

# ---------------------------------------------------------------
# Mode 2 : Matching CV / Offre
# ---------------------------------------------------------------

elif "Matching" in mode:
    st.header("🎯 Matching CV / Offre d'emploi")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📄 CV du candidat")
        cv_file = st.file_uploader("Upload le CV (PDF)", type=["pdf"], key="match_cv")
        if cv_file:
            st.success(f"✅ {cv_file.name}")

    with col2:
        st.subheader("📋 Offre d'emploi")
        job_file = st.file_uploader(
            "Upload l'offre (PDF)", type=["pdf"], key="match_job"
        )
        if job_file:
            st.success(f"✅ {job_file.name}")

    if cv_file and job_file:
        if st.button("🚀 Lancer le matching", type="primary"):
            with st.spinner("Analyse en cours... (peut prendre 1-2 min)"):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_cv:
                    tmp_cv.write(cv_file.read())
                    cv_path = tmp_cv.name

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_job:
                    tmp_job.write(job_file.read())
                    job_path = tmp_job.name

                try:
                    result = run_pipeline(cv_path, job_path)

                    if result.status == "success":
                        st.success("✅ Analyse terminée !")

                        tab1, tab2, tab3 = st.tabs(
                            ["🎯 Matching", "📝 Résumés", "⚖️ Biais"]
                        )

                        with tab1:
                            st.markdown("### Rapport de matching")
                            st.markdown(result.matching_report)

                        with tab2:
                            st.markdown("### Résumé du CV")
                            st.markdown(result.cv_summary)
                            st.divider()
                            st.markdown("### Résumé de l'offre")
                            st.markdown(result.job_summary)

                        with tab3:
                            st.markdown("### Analyse des biais")
                            st.code(result.bias_report)

                    else:
                        st.error(f"Erreur : {result.error}")

                except Exception as e:
                    st.error(f"Erreur : {e}")
                finally:
                    os.unlink(cv_path)
                    os.unlink(job_path)

# ---------------------------------------------------------------
# Mode 3 : Pipeline complet
# ---------------------------------------------------------------

elif "Pipeline complet" in mode:
    st.header("📊 Pipeline complet")
    st.info(
        "Ce mode combine l'analyse de biais ET le matching en une seule analyse."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        cv_file = st.file_uploader("📄 CV (PDF)", type=["pdf"], key="full_cv")
        if cv_file:
            st.success(f"✅ {cv_file.name}")

    with col2:
        job_file = st.file_uploader("📋 Offre d'emploi (PDF)", type=["pdf"], key="full_job")
        if job_file:
            st.success(f"✅ {job_file.name}")

    if cv_file and job_file:
        if st.button("🚀 Lancer l'analyse complète", type="primary"):
            with st.spinner("Pipeline en cours... (2-3 min)"):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_cv:
                    tmp_cv.write(cv_file.read())
                    cv_path = tmp_cv.name

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_job:
                    tmp_job.write(job_file.read())
                    job_path = tmp_job.name

                try:
                    result = run_pipeline(cv_path, job_path)

                    if result.status == "success":
                        st.success("✅ Pipeline terminé !")

                        # Métriques en haut
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Fichier CV", result.cv_filename)
                        m2.metric("Offre analysée", result.job_filename)
                        m3.metric(
                            "Score de biais",
                            f"{result.bias_score:.4f}",
                            delta="Neutre" if result.bias_score == 0 else "Biais détectés"
                        )

                        st.divider()

                        tab1, tab2, tab3 = st.tabs(
                            ["🎯 Matching", "⚖️ Biais", "📝 Résumés"]
                        )

                        with tab1:
                            st.markdown(result.matching_report)
                        with tab2:
                            st.code(result.bias_report)
                        with tab3:
                            st.markdown("**CV**")
                            st.markdown(result.cv_summary)
                            st.divider()
                            st.markdown("**Offre**")
                            st.markdown(result.job_summary)
                    else:
                        st.error(f"Erreur : {result.error}")

                except Exception as e:
                    st.error(f"Erreur : {e}")
                finally:
                    os.unlink(cv_path)
                    os.unlink(job_path)