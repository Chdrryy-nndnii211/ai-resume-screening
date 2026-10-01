import streamlit as st

from screening import (ALL_SKILLS, DEFAULT_WEIGHTS, EDU_REQ, extract_skills,
                       get_model, read_file, score_candidates)

st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading language model (first run only)...")
def load_model():
    return get_model()


load_model()

st.title("📄 AI Resume Screening & Candidate Shortlisting")
st.caption("Decision-support tool: a human recruiter should always make the final call.")

# ---------------------------- sidebar settings ----------------------------
with st.sidebar:
    st.header("Settings")
    min_years = st.slider("Minimum experience (years)", 0, 15, 0)
    min_edu = st.selectbox("Minimum education", list(EDU_REQ), index=0)
    blind = st.checkbox("Blind screening", help="Masks emails, phones, links and pronouns; hides file names.")
    st.subheader("Score weights")
    weights = {k: st.slider(k.capitalize(), 0, 100, int(v * 100)) for k, v in DEFAULT_WEIGHTS.items()}
    top_n = st.number_input("Show top N candidates", 1, 500, 10)

# ------------------------------ main inputs -------------------------------
left, right = st.columns(2)
with left:
    st.subheader("1. Job description")
    jd = st.text_area("Paste the job description", height=250)
    detected = sorted(extract_skills(jd)) if jd else []
    must = st.multiselect("Must-have skills (auto-detected, editable)", ALL_SKILLS, default=detected)
    nice = st.multiselect("Nice-to-have skills", ALL_SKILLS)
with right:
    st.subheader("2. Resumes")
    files = st.file_uploader("Upload PDF / DOCX / TXT files", type=["pdf", "docx", "txt"],
                             accept_multiple_files=True)

if st.button("Rank candidates", type="primary", disabled=not (jd.strip() and files)):
    names, texts = [], []
    for f in files:
        try:
            text = read_file(f.name, f.getvalue())
        except Exception as exc:  # corrupt or unsupported file
            st.warning(f"Could not read {f.name}: {exc}")
            continue
        if text.strip():
            names.append(f.name)
            texts.append(text)
        else:
            st.warning(f"No text found in {f.name} (scanned PDF? OCR is not enabled).")
    if texts:
        with st.spinner("Scoring candidates..."):
            st.session_state["results"] = score_candidates(
                texts, jd, names=names, must=must, nice=nice, min_years=min_years,
                min_education=min_edu, weights=weights, blind=blind)

# ------------------------------- results ----------------------------------
df = st.session_state.get("results")
if df is not None:
    st.divider()
    st.subheader("3. Ranked shortlist")
    shown = df.head(int(top_n))

    c1, c2, c3 = st.columns(3)
    c1.metric("Candidates screened", len(df))
    c2.metric("Strong matches", int((df["match"] == "Strong Match").sum()))
    c3.metric("Average score", f"{df['final_score'].mean():.1f}")

    st.dataframe(
        shown[["candidate", "final_score", "match", "years_exp", "matched_skills", "missing_skills"]],
        column_config={"final_score": st.column_config.ProgressColumn(
            "Score", min_value=0, max_value=100, format="%.1f")},
        hide_index=True)
    st.bar_chart(shown.set_index("candidate")["final_score"])

    with st.expander("Why these scores? (component breakdown, each 0 to 1)"):
        st.dataframe(shown[["candidate", "semantic", "skills", "experience", "education", "extras"]],
                     hide_index=True)

    st.download_button("Download shortlist (CSV)", shown.to_csv(index=False).encode("utf-8"),
                       file_name="shortlist.csv", mime="text/csv")
