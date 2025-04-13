import streamlit as st
import predictor as pred


def main():
    st.set_page_config(page_title="𝚏𝚒𝚕𝚖𝚏𝚘𝚛𝚝𝚞𝚗𝚎", page_icon="🎬")
    st.title("🎥 𝚏𝚒𝚕𝚖𝚏𝚘𝚛𝚝𝚞𝚗𝚎 - 𝚖𝚘𝚟𝚒𝚎 𝚛𝚎𝚟𝚎𝚗𝚞𝚎 𝚙𝚛𝚎𝚍𝚒𝚌𝚝𝚒𝚘𝚗")

    # Create tabs for top navigation
    tab1, tab2, tab3 = st.tabs(["Predict Revenue", "Insights", "About"])

    # Add content to each tab
    with tab1:
        prediction_page()

    with tab3:
        about_page()

    with tab2:
        insights_page()


def prediction_page():
    # st.title("🎥 FilmFortune - Movie Revenue Prediction")

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


def about_page():
    st.title("About FilmFortune")
    st.write("""
    FilmFortune uses machine learning to predict movie revenue based on key factors like budget,
    cast, genre, and release timing.

    Our model analyzes historical data from successful films to provide accurate revenue estimates
    for your movie project.
    """)

    st.subheader("How It Works")
    st.write("""
    1. Enter your movie details in the form
    2. Our XGBoost model processes the information
    3. Get an estimated box office revenue prediction
    """)


def insights_page():
    st.title("Movie Industry Insights")
    st.write("Explore data trends and patterns from our movie dataset.")

    # Placeholder for future visualizations
    st.info("Data visualizations coming soon!")


if __name__ == "__main__":
    main()