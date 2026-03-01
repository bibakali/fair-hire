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
    .main-title { font-size: 2.5rem; font-weight: 700; color: #1f4e79; }
    .subtitle { font-size: 1.1rem; color: #555; margin-bottom: 2rem; }
    .score-good { color: #27ae60; font-weight: bold; }
    .score-warning { color: #f39c12; font-weight: bold; }
    .score-bad { color: #e74c3c; font-weight: bold; }
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
# Sidebar
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
    **Analyse de biais** : Détecte les formulations discriminantes.
    
    **Matching CV/Offre** : Compare un CV avec une offre.
    
    **Pipeline complet** : Les deux analyses en une fois.
    """)
    st.divider()
    st.markdown("Built with LangChain · ChromaDB · Mistral")

# ---------------------------------------------------------------
# Mode 1 : Analyse de biais
# ---------------------------------------------------------------

if "Analyse de biais" in mode:
    st.header("🔍 Analyse de biais dans une offre d'emploi")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📄 Source de l'offre")
        input_mode = st.radio(
            "Mode d'entrée",
            ["📎 Upload PDF", "📋 Coller le texte"],
            horizontal=True,
            key="bias_input_mode"
        )

        job_file = None
        job_text_input = None

        if input_mode == "📎 Upload PDF":
            job_file = st.file_uploader(
                "Dépose ton offre d'emploi (PDF)",
                type=["pdf"],
                key="bias_job"
            )
            if job_file:
                st.success(f"✅ Fichier chargé : {job_file.name}")
        else:
            job_text_input = st.text_area(
                "Colle le texte de l'offre ici",
                height=300,
                placeholder="Colle ici le texte copié depuis LinkedIn, Indeed..."
            )

        can_analyze = job_file is not None or (
            job_text_input is not None and len(job_text_input.strip()) > 0
        )

        if can_analyze:
            if st.button("🚀 Analyser les biais", type="primary"):
                with st.spinner("Analyse en cours..."):
                    try:
                        if job_file:
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=".pdf"
                            ) as tmp:
                                tmp.write(job_file.read())
                                tmp_path = tmp.name
                            chunks = load_and_split(tmp_path)
                            job_text = " ".join(chunks)
                            os.unlink(tmp_path)
                        else:
                            job_text = job_text_input

                        bias_report, bias_score = tool_detect_bias(job_text)
                        st.session_state["bias_report"] = bias_report
                        st.session_state["bias_score"] = bias_score

                    except Exception as e:
                        st.error(f"Erreur : {e}")

    with col2:
        st.subheader("📊 Résultats")
        if "bias_score" in st.session_state:
            score = st.session_state["bias_score"]
            if score == 0:
                st.markdown('<p class="score-good">✅ Aucun biais détecté</p>', unsafe_allow_html=True)
            elif score < 0.05:
                st.markdown(f'<p class="score-warning">⚠️ Biais faibles — Score : {score}</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="score-bad">🚨 Biais significatifs — Score : {score}</p>', unsafe_allow_html=True)
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
        job_input_mode = st.radio(
            "Mode d'entrée offre",
            ["📎 Upload PDF", "📋 Coller le texte"],
            horizontal=True,
            key="match_job_mode"
        )

        job_file = None
        job_text_direct = None

        if job_input_mode == "📎 Upload PDF":
            job_file = st.file_uploader(
                "Upload l'offre (PDF)", type=["pdf"], key="match_job"
            )
            if job_file:
                st.success(f"✅ {job_file.name}")
        else:
            job_text_direct = st.text_area(
                "Colle le texte de l'offre",
                height=250,
                placeholder="Texte copié depuis LinkedIn, Indeed...",
                key="job_text_direct"
            )

    has_cv = cv_file is not None
    has_job = job_file is not None or (
        job_text_direct is not None and len(job_text_direct.strip()) > 0
    )

    if has_cv and has_job:
        if st.button("🚀 Lancer le matching", type="primary"):
            with st.spinner("Analyse en cours... (peut prendre 1-2 min)"):
                try:
                    # CV — toujours PDF
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp_cv:
                        tmp_cv.write(cv_file.read())
                        cv_path = tmp_cv.name

                    # Offre — PDF ou texte
                    if job_file:
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".pdf"
                        ) as tmp_job:
                            tmp_job.write(job_file.read())
                            job_path = tmp_job.name
                    else:
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".txt", mode="w", encoding="utf-8"
                        ) as tmp_job:
                            tmp_job.write(job_text_direct)
                            job_path = tmp_job.name

                    result = run_pipeline(cv_path, job_path)

                    if result.status == "success":
                        st.success("✅ Analyse terminée !")
                        tab1, tab2, tab3 = st.tabs(["🎯 Matching", "📝 Résumés", "⚖️ Biais"])
                        with tab1:
                            st.markdown(result.matching_report)
                        with tab2:
                            st.markdown("**CV**")
                            st.markdown(result.cv_summary)
                            st.divider()
                            st.markdown("**Offre**")
                            st.markdown(result.job_summary)
                        with tab3:
                            st.code(result.bias_report)
                    else:
                        st.error(f"Erreur : {result.error}")

                except Exception as e:
                    st.error(f"Erreur : {e}")
                finally:
                    if os.path.exists(cv_path):
                        os.unlink(cv_path)
                    if os.path.exists(job_path):
                        os.unlink(job_path)

# ---------------------------------------------------------------
# Mode 3 : Pipeline complet
# ---------------------------------------------------------------

elif "Pipeline complet" in mode:
    st.header("📊 Pipeline complet")
    st.info("Ce mode combine l'analyse de biais ET le matching en une seule analyse.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📄 CV du candidat")
        cv_file = st.file_uploader("Upload le CV (PDF)", type=["pdf"], key="full_cv")
        if cv_file:
            st.success(f"✅ {cv_file.name}")

    with col2:
        st.subheader("📋 Offre d'emploi")
        full_job_mode = st.radio(
            "Mode d'entrée offre",
            ["📎 Upload PDF", "📋 Coller le texte"],
            horizontal=True,
            key="full_job_mode"
        )

        full_job_file = None
        full_job_text = None

        if full_job_mode == "📎 Upload PDF":
            full_job_file = st.file_uploader(
                "Upload l'offre (PDF)", type=["pdf"], key="full_job"
            )
            if full_job_file:
                st.success(f"✅ {full_job_file.name}")
        else:
            full_job_text = st.text_area(
                "Colle le texte de l'offre",
                height=250,
                placeholder="Texte copié depuis LinkedIn, Indeed...",
                key="full_job_text"
            )

    has_cv = cv_file is not None
    has_job = full_job_file is not None or (
        full_job_text is not None and len(full_job_text.strip()) > 0
    )

    if has_cv and has_job:
        if st.button("🚀 Lancer l'analyse complète", type="primary"):
            with st.spinner("Pipeline en cours... (2-3 min)"):
                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp_cv:
                        tmp_cv.write(cv_file.read())
                        cv_path = tmp_cv.name

                    if full_job_file:
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".pdf"
                        ) as tmp_job:
                            tmp_job.write(full_job_file.read())
                            job_path = tmp_job.name
                    else:
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".txt", mode="w", encoding="utf-8"
                        ) as tmp_job:
                            tmp_job.write(full_job_text)
                            job_path = tmp_job.name

                    result = run_pipeline(cv_path, job_path)

                    if result.status == "success":
                        st.success("✅ Pipeline terminé !")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("CV", result.cv_filename)
                        m2.metric("Offre", result.job_filename)
                        m3.metric(
                            "Score de biais",
                            f"{result.bias_score:.4f}",
                            delta="Neutre" if result.bias_score == 0 else "Biais détectés"
                        )
                        st.divider()
                        tab1, tab2, tab3 = st.tabs(["🎯 Matching", "⚖️ Biais", "📝 Résumés"])
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
                    if os.path.exists(cv_path):
                        os.unlink(cv_path)
                    if os.path.exists(job_path):
                        os.unlink(job_path)