import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title
st.title("📊 Static Streamlit Demo")

# Header & Markdown
st.header("Sales Report (2023)")
st.markdown("This is a **static** dashboard showing dummy sales data.")

# Static DataFrame
data = {
    "Month": ["Jan", "Feb", "Mar", "Apr"],
    "Revenue ($)": [5000, 6200, 7100, 4300],
    "Profit ($)": [2000, 2500, 3000, 1800]
}
df = pd.DataFrame(data)
st.dataframe(df)

# Matplotlib Plot
fig, ax = plt.subplots()
ax.plot(df["Month"], df["Revenue ($)"], marker="o", label="Revenue")
ax.plot(df["Month"], df["Profit ($)"], marker="s", label="Profit")
ax.set_title("Revenue vs Profit (2023)")
ax.legend()
st.pyplot(fig)

# Image
st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)