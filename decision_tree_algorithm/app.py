import streamlit as st

from decision_tree_algorithm.directories import household_data, transformed_data
from decision_tree_algorithm.get_data import generate_data
from decision_tree_algorithm.predictions import make_prediction
from decision_tree_algorithm.preprocessing import preprocess_data

st.set_page_config(page_title="C4.5 Decision Tree Algorithm", layout="wide")


def app():
    st.title("C4.5 Decision Tree Algorithm")
    st.subheader(
        "Implementation study: Using decision tree induction to discover profitable locations to sell pet insurance for a startup company"
    )

    if st.button("Get data"):
        with st.spinner("Get data..."):
            st.markdown("### Household Data data")
            st.write(generate_data(2000, household_data))

    if st.button("Preprocess data"):
        with st.spinner("Preprocessing data..."):
            st.markdown("### Preprocessed data")
            st.write(preprocess_data(household_data, transformed_data))

    if st.button("Make prediction"):
        with st.spinner("Making prediction..."):
            original_data, results = make_prediction(transformed_data)

            st.markdown("### Training data")
            st.write(original_data)

            st.markdown("### Prediction results")
            st.write(results)


if __name__ == "__main__":
    app()
