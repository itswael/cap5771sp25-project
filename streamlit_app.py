import streamlit as st
import predictor as pred
import pandas as pd

predictions_with_params = {
    1: {"revenue": 1000000, "params": {"name": "Movie 1", "budget": 5000000, "director": "Director A"}},
    2: {"revenue": 2000000, "params": {"name": "Movie 2", "budget": 10000000, "director": "Director B"}},
    3: {"revenue": 3000000, "params": {"name": "Movie 3", "budget": 15000000, "director": "Director C"}},
    4: {"revenue": 4000000, "params": {"name": "Movie 4", "budget": 20000000, "director": "Director D"}},
    5: {"revenue": 5000000, "params": {"name": "Movie 5", "budget": 25000000, "director": "Director E"}},
    6: {"revenue": 6000000, "params": {"name": "Movie 6", "budget": 30000000, "director": "Director F"}},
    7: {"revenue": 7000000, "params": {"name": "Movie 7", "budget": 35000000, "director": "Director G"}},
    8: {"revenue": 8000000, "params": {"name": "Movie 8", "budget": 40000000, "director": "Director H"}},
    9: {"revenue": 9000000, "params": {"name": "Movie 9", "budget": 45000000, "director": "Director I"}},
    10: {"revenue": 10000000, "params": {"name": "Movie 10", "budget": 50000000, "director": "Director J"}},
}

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
    st.title("Today's Predictions")
    #st.write("Shows plot of your today's prediction insights.")

    # Create three columns with specified width ratios
    col1, col2, col3 = st.columns([2, 5, 3])  # 20%, 50%, 30%

    # Initialize session state for selected point if not exists
    if 'selected_point' not in st.session_state:
        st.session_state.selected_point = 1

    # First column: Sequence number and predicted revenue + selection widget
    with col1:
        st.subheader("Predictions")

        selected_seq = st.selectbox(
            "Select prediction to view details:",
            options=list(predictions_with_params.keys()),
            index=st.session_state.selected_point - 1
        )

        pred_df = pd.DataFrame({
            "Sequence": [k for k in predictions_with_params.keys()],
            "Revenue ($)": [f"${d['revenue']:,.2f}" for d in predictions_with_params.values()],
        }, index=None)
        st.dataframe(pred_df, use_container_width=True, hide_index=True)

        # Add selection widget
        st.session_state.selected_point = selected_seq

    with col2:
        st.subheader("Prediction Trend")

        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'Sequence': list(predictions_with_params.keys()),
            'Revenue': [item['revenue'] for item in predictions_with_params.values()],
        })

        # Create colors list to highlight selected bar
        colors = ['skyblue'] * len(plot_df)
        colors[st.session_state.selected_point - 1] = 'orange'

        # Create interactive plot
        fig = px.bar(
            plot_df,
            x='Sequence',
            y='Revenue',
            title='Revenue Predictions',
            labels={'Revenue': 'Predicted Revenue ($)'}
        )

        # Update marker colors to highlight selected point
        fig.update_traces(marker_color=colors)

        fig.update_layout(
            yaxis_tickprefix='$',
            yaxis_tickformat=',',
            hovermode='closest'
        )

        # Display the plot (without trying to capture clicks)
        st.plotly_chart(fig, use_container_width=True)

        # Third column: Parameter details for the selected prediction
        with col3:
            selected_seq = st.session_state.selected_point
            selected_prediction = predictions_with_params[selected_seq]

            st.subheader(f"Parameters (Seq #{selected_seq})")

            # Display selected prediction's input parameters
            st.write("**Input Parameters:**")
            for key, value in selected_prediction['params'].items():
                st.write(f"- {key}: {value}")

            # Display prediction stats
            st.write("**Prediction Results:**")
            st.write(f"- Revenue: ${selected_prediction['revenue']:,.2f}")
            if 'timestamp' in selected_prediction:
                st.write(f"- Time: {selected_prediction['timestamp']}")

if __name__ == "__main__":
    main()