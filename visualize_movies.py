import streamlit as st
import pandas as pd
import plotly.express as px


@st.cache_data
def _load_data(file_name):
    """Load the parquet file with plots, TF-IDF & BERT vector"""
    df = pd.read_parquet(file_name)
    return df


@st.cache_data
def _plot_movies(df):
    """plot a scatter plot of movies"""
    ### plot the UMAP produced data in 3-D
    fig_3d = px.scatter_3d(
        data_frame=df,
        x="comp_1",
        y="comp_2",
        z="comp_3",
        hover_name="title",
        hover_data=["url"],
        height=1000,
        opacity=0.7,
    ).update_layout(margin=dict(l=-0, r=-0, b=0, t=0))

    ### plot the UMAP produced data in 2-D
    fig_2d = px.scatter(
        data_frame=df,
        x="comp_1",
        y="comp_2",
        hover_name="title",
        hover_data=["url"],
        height=800,
        opacity=0.7,
    ).update_layout(margin=dict(l=-0, r=-0, b=0, t=0))
    return fig_3d, fig_2d


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
    st.divider()

    ### load data & initialize session state
    df = _load_data("artifacts/umap_reduced_data.parquet")
    if "umap_data" not in st.session_state:
        st.session_state["umap_data"] = df


def render_page():
    choice = st.radio("Graph Type:", ["2-D", "3-D"])
    fig_3d, fig_2d = _plot_movies(st.session_state["umap_data"])
    if choice == "2-D":
        fig = fig_2d
    else:
        fig = fig_3d
    st.plotly_chart(fig, use_container_width=True)
    st.info("Use the toolbar at top right to zoom/pan into the graph.")


####
#### MAIN APP FLOW
####

### initialize page title, icon, app data and session_state
init(title="Movie Buddy", icon="🎥")

### build page UI and show content
render_page()
