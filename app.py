import streamlit as st
import pandas as pd

from explorer import (
    ad_click_data,
    ad_click_experiment,
    find_similar_passengers,
    synthetic_passengers,
)

st.set_page_config(page_title="Classification Explorer", page_icon="📊")
st.title("Classification Explorer")
st.caption("Two use-case-driven machine-learning exercises.")

ad_tab, passenger_tab = st.tabs(
    ["Ad click prediction · Logistic Regression", "Passenger similarity · KNN"]
)

with ad_tab:
    st.subheader("Will a sample visitor click an ad?")
    fraction = st.slider(
        "Ad model test-set fraction", 0.1, 0.5, 0.25, 0.05, key="ad_fraction"
    )
    result = ad_click_experiment(fraction)
    st.dataframe(result["data"].head(10), use_container_width=True)
    left, right = st.columns(2)
    left.metric("Hold-out accuracy", f"{result['accuracy']:.1%}")
    right.metric("Test examples", result["test_count"])
    st.caption("Features: age and estimated salary · model: standardized logistic regression")
    st.subheader("Confusion matrix (actual rows × predicted columns)")
    st.dataframe(result["confusion_matrix"], use_container_width=True)
    age = st.slider("Example visitor age", 18, 65, 32)
    salary = st.number_input("Example estimated salary ($)", 20000, 150000, 70000, 5000)
    if st.button("Estimate click probability"):
        # Train using the same reproducible experiment data.
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        data = ad_click_data()
        model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=500, random_state=7)
        ).fit(data[["Age", "EstimatedSalary"]], data["Clicked"])
        probability = model.predict_proba(
            pd.DataFrame([{"Age": age, "EstimatedSalary": salary}])
        )[0, 1]
        st.metric("Synthetic click probability", f"{probability:.1%}")
    st.info("Fictional generated data; estimates are only a model-learning exercise.")

with passenger_tab:
    st.subheader("Find similar passenger profiles with KNN")
    st.caption(
        "The bundled records are fictional Titanic-shaped examples, not historical "
        "passenger data. Similarity is based on class, sex, age, fare, and family size."
    )
    passengers = synthetic_passengers()
    st.dataframe(passengers.head(10), use_container_width=True)
    selected = st.selectbox(
        "Passenger profile",
        passengers.index,
        format_func=lambda index: (
            f"Profile {index + 1}: {passengers.loc[index, 'Sex']}, "
            f"class {passengers.loc[index, 'PassengerClass']}, "
            f"age {passengers.loc[index, 'Age']}"
        ),
    )
    count = st.slider("Similar profiles", 1, 10, 5)
    neighbors = find_similar_passengers(passengers, int(selected), count)
    st.dataframe(neighbors, use_container_width=True)
    st.info("KNN distances express feature similarity, not survival likelihood or causation.")
