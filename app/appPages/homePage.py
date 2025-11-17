"""
Home Page - Embeds Gradio live demo
"""
import streamlit as st
import streamlit.components.v1 as components
import socket

def home_page():
    """Display the home page with embedded Gradio demo."""

    # Header
    st.header("*lisa*")
    st.write(
        ":small[An interactive UI for the `lisa` speaker identification model. \
        Test the model below or explore pipeline steps.]"
    )

    st.divider()
    
    # Check if Gradio is running
    if not is_gradio_running(7860):
        st.warning("**Demo is not running!** Open a new terminal and run the following command, then refresh:")
        
        st.markdown("""
        ```bash
        python3 app/demo.py
        ```
        """)
        
        return
    
    # Embed Gradio interface
    components.iframe(
        "http://localhost:7860",
        width=800,
        height=700
    )

def is_gradio_running(port=7860):
    """Check if Gradio app is running on specified port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result == 0


if __name__ == "__main__":
    st.set_page_config(page_title="Speech Demo", layout="wide")
    home_page()