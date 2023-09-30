import streamlit as st
from st_pages import show_pages_from_config, add_page_title


# @st.cache_data
def _load_markup():
    with open("artifacts/about_markup_main.txt", "r") as f_main:
        main_section = f_main.read()
    with open("artifacts/about_markup_tech.txt", "r") as f_tech:
        tech_section = f_tech.read()
    return main_section, tech_section


def init(title, icon):
    """Setup page title and icon and initialize session_state if needed"""
    st.set_page_config(page_title=title, page_icon=icon)
    col1, empty, col2 = st.columns([0.6, 0.2, 0.2])
    col1.markdown(
        f"<h1 style='text-align: center; color: #03989e;'>{icon} {title}</h1> ",
        unsafe_allow_html=True,
    )
    col1.markdown(
        "<h5 style='text-align: center;'>- Get movie recommendation by AI buddy</h5>",
        unsafe_allow_html=True,
    )
    col2.image("images/logo.png", width=100)
    show_pages_from_config()
    st.divider()


def about_the_app():
    main_section, tech_section = _load_markup()
    st.markdown(main_section)
    st.markdown("# Technical Details")
    with st.expander("Click to expand/collapse this section...", expanded=False):
        st.markdown("## Overview")
        st.image("images/movie-recommender.png")
        st.markdown(tech_section)


if __name__ == "__main__":
    ### setup page title
    init(title="Movie Buddy", icon="🎥")
    about_the_app()
