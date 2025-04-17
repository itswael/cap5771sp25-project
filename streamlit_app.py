import streamlit as st
import predictor as pred
import pandas as pd
import plotly.express as px
from datetime import datetime
from collections import OrderedDict

initial_predictions = OrderedDict({
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
})

def main():
    st.set_page_config(page_title="𝚏𝚒𝚕𝚖𝚏𝚘𝚛𝚝𝚞𝚗𝚎", page_icon="🎬")

    # Initialize predictions in session state if they don't exist
    if 'predictions' not in st.session_state:
        st.session_state.predictions = initial_predictions.copy()

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

                # Store the prediction with parameters
                update_predictions_with_params(prediction, inputs)

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

    # Create three columns with specified width ratios
    col1, col2, col3 = st.columns([2, 5, 3])

    # Initialize selected point if not exists
    if 'selected_point' not in st.session_state:
        st.session_state.selected_point = 1

    # First column: Sequence number and predicted revenue + selection widget
    with col1:
        st.subheader("Predictions")

        # Use the predictions from session state
        keys = list(st.session_state.predictions.keys())
        if keys:
            selected_seq = st.selectbox(
                "Select prediction to view details:",
                options=keys,
                index=min(st.session_state.selected_point - 1, len(keys) - 1) if st.session_state.selected_point <= len(keys) else 0
            )
        else:
            selected_seq = None
            st.write("No predictions available.")

        pred_df = pd.DataFrame({
            "Sequence": list(st.session_state.predictions.keys()),
            "Revenue ($)": [f"${d['revenue']:,.2f}" for d in st.session_state.predictions.values()],
        })
        st.dataframe(pred_df, use_container_width=True, hide_index=True)

        # Add selection widget
        st.session_state.selected_point = selected_seq

    with col2:
        st.subheader("Prediction Trend")

        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'Sequence': list(st.session_state.predictions.keys()),
            'Revenue': [item['revenue'] for item in st.session_state.predictions.values()],
        })

        # Create colors list to highlight selected bar
        colors = ['skyblue'] * len(plot_df)
        if st.session_state.selected_point in st.session_state.predictions:
            selected_index = list(st.session_state.predictions.keys()).index(st.session_state.selected_point)
            colors[selected_index] = 'orange'

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
        if selected_seq:
            selected_prediction = st.session_state.predictions[selected_seq]

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
        else:
            st.write("No prediction selected.")


def update_predictions_with_params(prediction, inputs):
    # Make a deep copy of inputs to ensure we're not storing references
    input_copy = {k: v for k, v in inputs.items()}

    # Create new predictions dictionary
    new_predictions = OrderedDict({
        1: {
            "revenue": float(prediction),  # Ensure it's a float
            "params": input_copy,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    })

    # Add existing predictions (shifted by 1)
    current_keys = list(st.session_state.predictions.keys())
    for i, key in enumerate(current_keys):
        if i >= 9:  # Keep only 9 existing predictions
            break
        new_predictions[i + 2] = st.session_state.predictions[key]

    # Update session state - assign the entire dictionary at once
    st.session_state.predictions = new_predictions

    # Force a rerun to update the UI
    # st.rerun()


if __name__ == "__main__":
    main()