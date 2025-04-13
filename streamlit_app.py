import streamlit as st
import predic as pred


# Streamlit UI Components
def main():
    st.set_page_config(page_title="FilmFortune", page_icon="🎬")
    st.title("🎥 FilmFortune - Movie Revenue Prediction")

    with st.form("movie_inputs"):
        col1, col2, col3 = st.columns(3)

        with col1:
            inputs = {
                "name": st.text_input("Movie Title"),
                "director": st.text_input("Director"),
                "star": st.text_input("Lead Actor/Actress"),
                "country": st.text_input("Production Country"),
                "company": st.text_input("Production Company"),
            }

        with col2:
            inputs.update({
                "genre": st.text_input("Genre"),
                "writer": st.text_input("Writer"),
                "runtime": st.number_input("Runtime (minutes)", min_value=0),
                "budget": st.number_input("Budget ($)", min_value=0),
                "year": st.number_input("Release Year", min_value=1900, max_value=2100),
            })

        with col3:
            inputs.update({
                "released": st.text_input("Release Date"),
                "rating": st.selectbox("MPAA Rating", ["G", "PG", "PG-13", "R", "NC-17"]),
                "score": st.number_input("IMDB Score", min_value=0.0, max_value=10.0),
                "votes": st.number_input("Initial Votes", min_value=0)
            })

        if st.form_submit_button("Predict Revenue"):
            try:
                prediction = pred.predictor(inputs)

                st.subheader("Prediction Results")
                st.metric(label="Estimated Revenue",
                          value=f"${prediction:,.2f}")

            except Exception as e:
                st.error("An error occurred during prediction. Please try again.")


if __name__ == "__main__":
    main()

