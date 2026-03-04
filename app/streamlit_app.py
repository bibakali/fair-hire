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
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0d0d1a 50%, #0a0f1a 100%);
        min-height: 100vh;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1a 0%, #111128 100%) !important;
        border-right: 1px solid rgba(139, 92, 246, 0.2);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] .stRadio label {
        background: rgba(139, 92, 246, 0.08);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 10px;
        padding: 10px 14px;
        margin: 4px 0;
        transition: all 0.2s ease;
        cursor: pointer;
        display: block;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(139, 92, 246, 0.2);
        border-color: rgba(139, 92, 246, 0.5);
        transform: translateX(4px);
    }

    h1, h2, h3 { font-family: 'Syne', sans-serif !important; color: #e2e8f0 !important; }
    p, li, span, div { color: #cbd5e1; }

    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(124, 58, 237, 0.6) !important;
    }

    [data-testid="stFileUploader"] {
        background: rgba(139, 92, 246, 0.05) !important;
        border: 2px dashed rgba(139, 92, 246, 0.3) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(139, 92, 246, 0.6) !important;
        background: rgba(139, 92, 246, 0.08) !important;
    }

    .stTextArea textarea {
        background: rgba(15, 15, 30, 0.8) !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .stTextArea textarea:focus {
        border-color: rgba(139, 92, 246, 0.7) !important;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.15) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: rgba(139, 92, 246, 0.05) !important;
        border-radius: 12px !important;
        padding: 4px !important;
        gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #94a3b8 !important;
        border-radius: 10px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
        color: white !important;
    }

    [data-testid="stMetric"] {
        background: rgba(139, 92, 246, 0.08) !important;
        border: 1px solid rgba(139, 92, 246, 0.2) !important;
        border-radius: 16px !important;
        padding: 20px !important;
    }
    [data-testid="stMetricValue"] {
        color: #a78bfa !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
    }

    .stAlert {
        background: rgba(56, 189, 248, 0.08) !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
    }

    hr { border-color: rgba(139, 92, 246, 0.15) !important; }

    .stProgress > div > div {
        background: linear-gradient(90deg, #7c3aed, #38bdf8) !important;
        border-radius: 10px !important;
    }
    .stProgress > div {
        background: rgba(139, 92, 246, 0.1) !important;
        border-radius: 10px !important;
    }

    .score-good { color: #34d399 !important; font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1.2rem; }
    .score-warning { color: #fbbf24 !important; font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1.2rem; }
    .score-bad { color: #f87171 !important; font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1.2rem; }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0a0f; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(#7c3aed, #2563eb); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------

st.markdown("""
<div style="padding: 2rem 0 1rem 0;">
    <div style="
        font-family: 'Syne', sans-serif;
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa 0%, #38bdf8 60%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -3px;
        line-height: 1;
        margin-bottom: 0.5rem;
    ">⚖️ Fair Hire</div>
    <div style="
        font-family: 'DM Sans', sans-serif;
        font-size: 1.1rem;
        color: #94a3b8;
        font-weight: 300;
        letter-spacing: 1px;
        border-left: 3px solid #7c3aed;
        padding-left: 12px;
        margin-top: 8px;
    ">Assistant RH basé sur l'IA — Détection de biais & Matching CV/Offre d'emploi</div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")
    mode = st.radio(
    "Mode d'analyse",
    ["🔍 Analyse de biais", "🎯 Matching CV/Offre", "📊 Pipeline complet", "🤖 Optimiseur ATS"],
    index=0)

    st.divider()
    st.markdown("### 📖 Guide rapide")
    st.markdown("""
    **🔍 Analyse de biais**
    Détecte les formulations discriminantes dans une offre.
    *Résultat en 2 secondes.*

    **🎯 Matching CV/Offre**
    Rapport de matching rapide entre un CV et une offre.
    *Résultat en 5 secondes.*

    **📊 Pipeline complet**
    Matching + biais + résumés détaillés.
    *Analyse approfondie en 30 secondes.*
                
    **🤖 Optimiseur ATS**
    Détecte les mots-clés manquants et réécrit ton CV pour passer les filtres automatiques.*

    """)
    st.divider()
    st.markdown("Built with LangChain · ChromaDB · Mistral")

# ---------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------

def save_temp_file(uploaded_file, suffix=".pdf"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name


def save_temp_text(text, suffix=".txt"):
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, mode="w", encoding="utf-8"
    )
    tmp.write(text)
    tmp.flush()
    tmp.close()
    return tmp.name


def cleanup(*paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.unlink(p)
        except Exception:
            pass

# ---------------------------------------------------------------
# Mode 1 : Analyse de biais
# ---------------------------------------------------------------

if "Analyse de biais" in mode:
    st.header("🔍 Analyse de biais dans une offre d'emploi")
    st.info("Détecte automatiquement les formulations genrées, discriminatoires ou excluantes. Résultat instantané.")

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

        can_analyze = (job_file is not None) or (
            job_text_input is not None and len(job_text_input.strip()) > 0
        )

        if can_analyze:
            if st.button("🚀 Analyser les biais", type="primary"):
                with st.spinner("Analyse en cours..."):
                    tmp_path = None
                    try:
                        if job_file:
                            tmp_path = save_temp_file(job_file)
                            chunks = load_and_split(tmp_path)
                            job_text = " ".join(chunks)
                        else:
                            job_text = job_text_input

                        bias_report, bias_score = tool_detect_bias(job_text)
                        st.session_state["bias_report"] = bias_report
                        st.session_state["bias_score"] = bias_score

                    except Exception as e:
                        st.error(f"Erreur : {e}")
                    finally:
                        cleanup(tmp_path)

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
# Mode 2 : Matching rapide CV / Offre
# ---------------------------------------------------------------

elif "Matching" in mode:
    st.header("🎯 Matching CV / Offre d'emploi")
    st.info("Mode rapide — rapport de matching uniquement, sans analyse de biais. Résultat en 5 secondes.")

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
    has_job = (job_file is not None) or (
        job_text_direct is not None and len(job_text_direct.strip()) > 0
    )

    if has_cv and has_job:
        if st.button("🚀 Lancer le matching", type="primary"):
            with st.spinner("Analyse en cours..."):
                cv_path = None
                job_path = None
                try:
                    cv_path = save_temp_file(cv_file)

                    if job_file:
                        job_path = save_temp_file(job_file)
                    else:
                        job_path = save_temp_text(job_text_direct)

                    result = run_pipeline(cv_path, job_path)

                    if result.status == "success":
                        st.success("✅ Matching terminé !")
                        st.divider()
                        st.markdown(result.matching_report)
                    else:
                        st.error(f"Erreur : {result.error}")

                except Exception as e:
                    st.error(f"Erreur : {e}")
                finally:
                    cleanup(cv_path, job_path)

# ---------------------------------------------------------------
# Mode 3 : Pipeline complet
# ---------------------------------------------------------------

elif "Pipeline complet" in mode:
    st.header("📊 Pipeline complet")
    st.info("Analyse approfondie — matching + détection de biais + résumés détaillés. ~30 secondes.")

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
    has_job = (full_job_file is not None) or (
        full_job_text is not None and len(full_job_text.strip()) > 0
    )

    if has_cv and has_job:
        if st.button("🚀 Lancer l'analyse complète", type="primary"):
            with st.spinner("Pipeline en cours... (~30 secondes)"):
                cv_path = None
                job_path = None
                try:
                    cv_path = save_temp_file(cv_file)

                    if full_job_file:
                        job_path = save_temp_file(full_job_file)
                    else:
                        job_path = save_temp_text(full_job_text)

                    result = run_pipeline(cv_path, job_path)

                    if result.status == "success":
                        st.success("✅ Pipeline terminé !")

                        m1, m2, m3 = st.columns(3)
                        m1.metric("📄 CV analysé", "✅ Chargé")
                        m2.metric("📋 Offre analysée", "✅ Chargée")

                        if result.bias_score == 0:
                            bias_label = "✅ Neutre"
                        elif result.bias_score < 0.05:
                            bias_label = "⚠️ Biais faibles"
                        else:
                            bias_label = "🚨 Biais détectés"

                        m3.metric("⚖️ Score de biais", f"{result.bias_score:.4f}", delta=bias_label)

                        st.divider()

                        tab1, tab2, tab3 = st.tabs(["🎯 Matching", "⚖️ Biais", "📝 Résumés"])

                        with tab1:
                            st.markdown(result.matching_report)

                        with tab2:
                            st.code(result.bias_report)

                        with tab3:
                            st.markdown("**📄 Contexte CV**")
                            st.markdown(result.cv_summary)
                            st.divider()
                            st.markdown("**📋 Contexte Offre**")
                            st.markdown(result.job_summary)

                    else:
                        st.error(f"Erreur : {result.error}")

                except Exception as e:
                    st.error(f"Erreur : {e}")
                finally:
                    cleanup(cv_path, job_path)

# ---------------------------------------------------------------
# Mode 4 : Optimiseur ATS
# ---------------------------------------------------------------

elif "Optimiseur ATS" in mode:
    st.header("🤖 Optimiseur ATS")
    st.info("Détecte les mots-clés manquants dans ton CV et réécrit tes expériences pour passer les filtres ATS.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📄 CV du candidat")
        cv_file = st.file_uploader("Upload le CV (PDF)", type=["pdf"], key="ats_cv")
        if cv_file:
            st.success(f"✅ {cv_file.name}")

    with col2:
        st.subheader("📋 Offre d'emploi")
        ats_job_mode = st.radio(
            "Mode d'entrée",
            ["📎 Upload PDF", "📋 Coller le texte"],
            horizontal=True,
            key="ats_job_mode"
        )
        ats_job_file = None
        ats_job_text = None

        if ats_job_mode == "📎 Upload PDF":
            ats_job_file = st.file_uploader(
                "Upload l'offre (PDF)", type=["pdf"], key="ats_job"
            )
            if ats_job_file:
                st.success(f"✅ {ats_job_file.name}")
        else:
            ats_job_text = st.text_area(
                "Colle le texte de l'offre",
                height=200,
                placeholder="Texte copié depuis LinkedIn, Indeed...",
                key="ats_job_text"
            )

    has_cv = cv_file is not None
    has_job = (ats_job_file is not None) or (
        ats_job_text is not None and len(ats_job_text.strip()) > 0
    )

    if has_cv and has_job:
        if st.button("🚀 Analyser et optimiser", type="primary"):
            with st.spinner("Analyse ATS en cours..."):
                cv_path = None
                job_path = None
                try:
                    from src.ats_optimizer import analyze_ats, rewrite_cv_for_ats

                    cv_path = save_temp_file(cv_file)
                    cv_chunks = load_and_split(cv_path)
                    cv_text = " ".join(cv_chunks)

                    if ats_job_file:
                        job_path = save_temp_file(ats_job_file)
                        job_chunks = load_and_split(job_path)
                        job_text = " ".join(job_chunks)
                    else:
                        job_text = ats_job_text

                    report = analyze_ats(cv_text, job_text)

                    st.success("✅ Analyse terminée !")

                    m1, m2, m3 = st.columns(3)
                    m1.metric(
                        "🎯 Score ATS",
                        f"{int(report.ats_score * 100)}%",
                        delta="Bon" if report.ats_score >= 0.7 else "À améliorer"
                    )
                    m2.metric("✅ Mots-clés présents", len(report.keywords_in_cv))
                    m3.metric("❌ Mots-clés manquants", len(report.missing_keywords))

                    st.divider()

                    tab1, tab2, tab3 = st.tabs([
                        "❌ Mots-clés manquants",
                        "✅ Mots-clés présents",
                        "✏️ CV réécrit"
                    ])

                    with tab1:
                        st.markdown(f"### {report.summary}")
                        if report.missing_keywords:
                            cols = st.columns(3)
                            for i, kw in enumerate(report.missing_keywords):
                                cols[i % 3].markdown(
                                    f'<span style="background:rgba(239,68,68,0.2);'
                                    f'border:1px solid rgba(239,68,68,0.4);'
                                    f'border-radius:20px;padding:4px 12px;'
                                    f'color:#fca5a5;font-size:0.85rem;">'
                                    f'❌ {kw}</span>',
                                    unsafe_allow_html=True
                                )
                        else:
                            st.success("Ton CV contient tous les mots-clés !")

                    with tab2:
                        if report.keywords_in_cv:
                            cols = st.columns(3)
                            for i, kw in enumerate(report.keywords_in_cv):
                                cols[i % 3].markdown(
                                    f'<span style="background:rgba(52,211,153,0.2);'
                                    f'border:1px solid rgba(52,211,153,0.4);'
                                    f'border-radius:20px;padding:4px 12px;'
                                    f'color:#6ee7b7;font-size:0.85rem;">'
                                    f'✅ {kw}</span>',
                                    unsafe_allow_html=True
                                )

                    with tab3:
                        with st.spinner("Réécriture du CV en cours..."):
                            rewritten = rewrite_cv_for_ats(
                                cv_text,
                                report.missing_keywords,
                                job_text
                            )
                            st.markdown(rewritten)

                except Exception as e:
                    st.error(f"Erreur : {e}")
                finally:
                    cleanup(cv_path, job_path)