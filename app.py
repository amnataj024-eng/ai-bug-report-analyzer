import os
import streamlit as st
from bug_report_analyzer import build_graph

st.set_page_config(page_title="AI Bug Report Analyzer", page_icon="🐞", layout="centered")

st.title("🐞 AI Bug Report Analyzer")
st.write("Enter a description of the issue below to generate an automated bug report.")

# Input area
bug_input = st.text_area(
    "Bug Description",
    placeholder="e.g., App crashes when I tap Pay on the checkout page",
    height=150
)

if st.button("Analyze Bug", type="primary"):
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY is missing in your .env file!")
    elif len(bug_input.strip()) < 10:
        st.warning("Please enter at least 10 characters for a proper description.")
    else:
        with st.spinner("Analyzing bug report with AI..."):
            try:
                graph = build_graph()
                result = graph.invoke({"bug_description": bug_input.strip()})
                st.success("Analysis Complete!")
                st.markdown(result["final_report"])
            except Exception as e:
                st.error(f"Error: {e}")