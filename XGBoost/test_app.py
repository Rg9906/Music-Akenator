import streamlit as st

st.set_page_config(
    page_title="Test App",
    layout="wide"
)

st.header("🎵 Test Music Akenator")
st.write("This is a simple test to verify Streamlit works.")

if st.button("Test Button"):
    st.success("Button clicked!")
    st.write("✅ Streamlit is working correctly!")

st.markdown("---")
st.write("If this works, the main app should work too. The magic.py error is a Streamlit internal issue.")
