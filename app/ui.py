import streamlit as st
import streamlit_antd_components as sac
from appPages.homePage import home_page
from appPages.page0 import page0
from appPages.page1 import page1
from appPages.page2 import page2
from appPages.page3 import page3
from appPages.page4 import page4
from appPages.page5 import page5

# Streamlit page configurations
st.set_page_config(
    page_title="speaker-recognition",
    layout="wide",
    initial_sidebar_state="expanded"
)

def sidebar_menu():
    """Build sidebar menu items."""
    return [
        sac.MenuItem("Home", icon="house-fill"),
        sac.MenuItem(
            "ML Steps",
            icon="box-fill",
            children=[
                sac.MenuItem(
                    "0-get-data.py", 
                    # description="Collected raw audio data",
                    tag=[sac.Tag('collection', color='orange')]
                ),
                sac.MenuItem(
                    "1-clean-audio.py", 
                    # description="Cleaned and preprocessed audio data",
                    tag=[sac.Tag('cleaning', color='blue')]
                ),
                sac.MenuItem(
                    "2-split-clips.py", 
                    # description="Audio split into clips",
                    tag=[sac.Tag('cleaning', color='blue')]
                ),
                sac.MenuItem(
                    "3-filter-and-balance.py", 
                    # description="Removed silent clips & balanced dataset",
                    tag=[sac.Tag('cleaning', color='blue')]
                ),
                sac.MenuItem(
                    "4-extract-features.py", 
                    # description="Extracted features from audio clips",
                    tag=[sac.Tag("features", color='green')]
                ),
                sac.MenuItem(
                    "5-train-model.py", 
                    # description="Trained speaker recognition model",
                    tag=[sac.Tag('training', color='purple')]
                ),
            ]
        ),
    ]


def display_page(page_name: str):
    """Display the selected page."""
    
    if page_name == "Home":
        home_page()
    elif page_name == "0-get-data.py":
        page0() 
    elif page_name == "1-clean-audio.py":
        page1()
    elif page_name == "2-split-clips.py":
        page2()
    elif page_name == "3-filter-and-balance.py":
        page3()
    elif page_name == "4-extract-features.py":
        page4()
    elif page_name == "5-train-model.py":
        page5()


def main():
    
    # st.logo("app/images/ascii-art.png", size='large')
    
    # Sidebar Menu
    with st.sidebar:
        active_page = sac.menu(
            items=sidebar_menu(),
            open_all=False,
            indent=4,
            size='md',
            key="active_page",
            index=0, # Default to 'home' page (at index 1)
            variant='subtle'
        )

        st.divider()
        
        st.write("Updated: :grey-badge[Sept 28, 2025]")
        # TODO: add your name and a link below (inside the string after the comma) like this...
        # :grey-badge[[Your Name](your-link)]
        st.write("Developers: :grey-badge[[Marco Lanfranchi](https://github.com/marcolanfranchi)], ")
        sac.menu(
            items=[sac.MenuItem(None, icon="github", href="https://github.com/marcolanfranchi/speaker-recognition")],
            key="link_menu",
            index=None,
            indent=4,
            size='xl',
            variant='subtle')
    
    # Display active page content
    display_page(active_page)


if __name__ == "__main__":
    main()