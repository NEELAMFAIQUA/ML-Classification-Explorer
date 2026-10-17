import streamlit as st

from explorer import run_experiment

st.set_page_config(page_title="Classification Explorer", page_icon="📊")
st.title("Classification Explorer")
st.caption("Compare two classifiers on a built-in, labeled teaching dataset.")
model = st.selectbox(
    "Classifier", ["Logistic regression", "K-nearest neighbors"]
)
test_fraction = st.slider("Test-set fraction", 0.1, 0.5, 0.25, 0.05)
seed = st.number_input("Random seed", min_value=0, max_value=10000, value=7)
result = run_experiment(model, test_fraction, int(seed))

st.metric("Hold-out accuracy", f"{result['accuracy']:.1%}")
st.caption(f"Evaluated {result['test_count']} held-out examples.")
st.subheader("Confusion matrix (actual rows × predicted columns)")
st.dataframe(result["confusion_matrix"], use_container_width=True)
st.caption("Class order: " + ", ".join(result["labels"]))
with st.expander("Dataset features"):
    st.write(", ".join(result["features"]))
st.info(
    "The Iris dataset is a small teaching example. Results vary with the split "
    "and do not guarantee performance on other data."
)
