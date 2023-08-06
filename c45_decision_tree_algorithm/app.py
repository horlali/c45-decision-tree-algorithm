import pandas as pd
import streamlit as st

from c45_decision_tree_algorithm.directories import household_data, transformed_data
from c45_decision_tree_algorithm.generate_data import generate_data
from c45_decision_tree_algorithm.predictions import make_prediction
from c45_decision_tree_algorithm.preprocessing import preprocess_data

st.set_page_config(page_title="C4.5 Decision Tree Algorithm", layout="wide")


def app():
    st.title("C4.5 Decision Tree Algorithm")
    st.subheader(
        "Implementation study: Using decision tree induction to discover profitable locations to sell pet insurance for a startup company"
    )
    col1, col2, col3 = st.columns(3)

    tree = None

    with col1:
        if st.button("Get data"):
            with st.spinner("Get data..."):
                st.markdown("### Household Data data")
                st.write(generate_data(2000, household_data))

    with col2:
        if st.button("Preprocess data"):
            with st.spinner("Preprocessing data..."):
                st.markdown("### Preprocessed data")
                st.write(preprocess_data(household_data, transformed_data))

    with col3:
        if st.button("Train and Test Prediction"):
            with st.spinner("Making prediction..."):
                original_data, results, accuracy = make_prediction(transformed_data)

                st.markdown("### Training data")
                st.write(original_data)

                st.markdown("### Prediction results")
                st.write(results)

                st.markdown("### Accuracy")
                st.write(accuracy)

    household_size = st.text_input("household_size", "1")
    income = st.text_input("income", "10000")
    zipcode = st.number_input("zipcode", min_value=100000, step=1)
    age = st.number_input("age", min_value=0, step=1)
    sex = st.selectbox("sex", ["male", "female"], index=0)
    race = st.selectbox(
        "race",
        ["caucasian", "african american", "hispanic", "asian", "other"],
        index=0,
    )

    if st.button("Submit"):
        with st.spinner("Making prediction..."):
            _, decision, _, tree = make_prediction(transformed_data)
            tree.predict(
                pd.DataFrame(
                    {
                        "household_size": [household_size],
                        "income": [income],
                        "zipcode": [zipcode],
                        "age": [age],
                        "sex": [sex],
                        "race": [race],
                    }
                )
            )
            st.write(tree)
            st.success("Submitted")


if __name__ == "__main__":
    app()
