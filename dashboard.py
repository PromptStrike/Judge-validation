"""jval dashboard — three screens: load, label (blind), report.
Reuses the jval package directly. Run: streamlit run dashboard.py
"""
import streamlit as st

from jval.store import Store
from jval.models import Sample, HumanLabel
from jval.loader import load_samples_jsonl
from jval.report import validate_report

st.set_page_config(page_title="jval — Judge Validation", layout="centered")
st.title("jval — Judge Validation")

# --- persistent state across reruns ---
if "store" not in st.session_state:
    st.session_state.store = Store("data/dashboard")
    st.session_state.idx = 0
    st.session_state.labeler = "you"

store: Store = st.session_state.store

screen = st.sidebar.radio("Screen", ["1. Load", "2. Label (blind)", "3. Report"])
st.session_state.labeler = st.sidebar.text_input("Labeler ID", st.session_state.labeler)

# ---------- SCREEN 1: LOAD ----------
if screen == "1. Load":
    st.header("Load samples")
    up = st.file_uploader("Upload a JSONL file", type=["jsonl"])
    if up is not None:
        # write to a temp path, load with your existing loader
        tmp = "data/_upload.jsonl"
        with open(tmp, "wb") as f:
            f.write(up.getvalue())
        samples = load_samples_jsonl(tmp)
        store.add_samples(samples)
        st.success(f"Loaded {len(samples)} samples.")
    st.info(f"Store currently has {len(store.samples())} samples.")

# ---------- SCREEN 2: BLIND LABEL ----------
elif screen == "2. Label (blind)":
    st.header("Blind labeling")
    st.caption("You see only prompt + response. No judge verdict is shown — "
               "blindness is enforced.")
    labeler = st.session_state.labeler
    done = store.labeled_sample_ids(labeler)
    pending = [s for s in store.samples() if s.id not in done]

    if not pending:
        st.success("Nothing left to label.")
    else:
        i = st.session_state.idx % len(pending)
        s = pending[i]
        st.progress((len(store.samples()) - len(pending)) / max(len(store.samples()), 1))
        st.write(f"**Sample {s.id}**  ({len(pending)} left)")
        st.text_area("PROMPT", s.prompt, height=80, disabled=True)
        st.text_area("RESPONSE", s.response, height=140, disabled=True)

        c1, c2, c3 = st.columns(3)
        def record(verdict):
            store.add_label(HumanLabel(sample_id=s.id, labeler_id=labeler, verdict=verdict))
            st.session_state.idx += 1
            st.rerun()
        if c1.button("harmful_produced", use_container_width=True):
            record("harmful_produced")
        if c2.button("harmful_blocked", use_container_width=True):
            record("harmful_blocked")
        if c3.button("ambiguous", use_container_width=True):
            record("ambiguous")

# ---------- SCREEN 3: REPORT ----------
elif screen == "3. Report":
    st.header("Reliability report")
    verdicts = store.verdicts()
    if not verdicts:
        st.warning("No judge verdicts stored yet. Run the judge (runner) on "
                   "this store first — that step needs the API and is best run "
                   "from the CLI on a machine with GROQ_API_KEY set.")
    else:
        jid = verdicts[0].judge_id
        jver = verdicts[0].judge_version
        try:
            report = validate_report(store, jid, jver, st.session_state.labeler)
            st.code(report)
        except Exception as e:
            st.error(f"Could not build report: {e}")
