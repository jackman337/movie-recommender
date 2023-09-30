import re
import numpy as np
import streamlit as st
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics.pairwise import cosine_similarity

### For cloud deployment make it "false" due to its large-size, "True" for local run
USE_TFIDF = False

####
#### FUNCTIONS
####


# @st.cache_data(persist="disk")
def _load_data():
    """Load the parquet file with plots, TF-IDF & BERT vector"""
    ### read the movie plot into a dataframe
    df = pd.read_parquet("artifacts/movie_plots.parquet")
    ### load the sbert embedding data and combine it with movie dataframe
    sbert_embeddings = (
        pq.read_table("artifacts/sbert_embeddings.parquet").to_pandas().to_numpy()
    )
    df["sbert"] = list(sbert_embeddings)
    ### load TF-IDF only if flag is True (i.e. running app locally)
    if USE_TFIDF:
        tfidf_embeddings = (
            pq.read_table("artifacts/tfidf_embeddings.parquet").to_pandas().to_numpy()
        )
        ## combine it with movie dataframe
        df["tfidf"] = list(tfidf_embeddings)

    return df


def _reset():
    """
    reset this page completely.
    i.e. remove all forms and filters load full data and start from page 1.
    this callback is used by sidebar "Reset" button and main page "Clear" button
    """
    st.session_state["data"] = st.session_state["orig_data"]
    st.session_state["curr_page"] = 0
    st.session_state["last_page"] = len(st.session_state["data"]) - 1
    st.session_state["filter"] = ""
    st.session_state["cosine_similarity"] = pd.Series(dtype="float64")
    st.session_state["recommended"] = False
    st.experimental_rerun()


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
    df = _load_data()
    if "data" not in st.session_state:
        st.session_state["data"] = df
        # backup the original data-frame, required if we need to reset filter
        st.session_state["orig_data"] = df.copy()
        st.session_state["curr_page"] = 0
        st.session_state["last_page"] = len(st.session_state["data"]) - 1
        st.session_state["filter"] = ""
        st.session_state["cosine_similarity"] = pd.Series(dtype="float64")
        st.session_state["recommended"] = False


def render_page():
    """
    main page UI rendering and content display functionality
    """
    ### render prev, next buttons and page no
    df = st.session_state["data"]
    (
        prev,
        next,
        page_no,
    ) = st.columns([6, 18, 6])

    curr_page = st.session_state["curr_page"]
    last_page = st.session_state["last_page"]

    def _prev_page():
        if st.session_state["curr_page"] == 0:
            st.session_state["curr_page"] = st.session_state["last_page"]
        else:
            st.session_state["curr_page"] -= 1

    def _next_page():
        if st.session_state["curr_page"] == st.session_state["last_page"]:
            st.session_state["curr_page"] = 0
        else:
            st.session_state["curr_page"] += 1

    prev.button("<< Prev Page", on_click=_prev_page)
    next.button("Next Page >>", on_click=_next_page)
    page_no.write(f"Page {curr_page+1} of {last_page + 1}")

    ### render movie content: title, url, plot
    title = df.loc[curr_page, "title"]
    st.subheader(title)
    ## if we are showing the recommended movies then also show the similarity score
    if not st.session_state["cosine_similarity"].empty:
        score = st.session_state["cosine_similarity"][curr_page]
        score = f"ℹ️ Movie similarity Score: {str(round(score, 12))} [min: 0, max: 1.0]"
        st.info(score)
    st.write(df.loc[curr_page, "url"])

    ### render the movie recommendation panel as an expander
    ### initially remains collapsed
    def _render_recommend_panel():
        def __recommend_movies(curr_movie_index, df, k, embed):
            """
            get the recommended movies refreshes the page to show them
            """
            df_matches = __get_similar_movies(
                movie_index=curr_movie_index, df=df, k=k, use_embed=embed
            )
            ### Reset the index so that pages can be displayed correctly
            df_matches = df_matches.reset_index()
            # set recommended movie df as the active dataframe and re-fresh the page
            st.session_state["data"] = df_matches[["title", "url", "plot"]]
            st.session_state["curr_page"] = 0
            st.session_state["last_page"] = len(st.session_state["data"]) - 1
            st.session_state["cosine_similarity"] = df_matches["score"]
            st.session_state["recommended"] = True
            st.experimental_rerun()

        def __get_similar_movies(movie_index, df, k, use_embed="sbert"):
            """
            helper function
            get the associated movie plot embed vectors for the given movie index
            and whole corpus, calculate similarity score and return a dataframe
            containing only similar movies as recommendation
            """
            search_vector = df.loc[movie_index, use_embed]
            corpus_vectors = df[use_embed].values

            ### Calculate the similarity score for given movie against whole corpus of movies
            scores = __get_similarity_scores(search_vector, corpus_vectors)

            ### Sort the scores in descending order and grab the sorted indices
            ### and return a dataframe containing `k` matching movies in order of similarity,
            ### first one will always be the movie we are searching for
            sorted_idx = np.flip(scores.argsort())[:k]
            df_matches = df.iloc[sorted_idx].copy().drop("index", axis=1)
            df_matches["score"] = scores[sorted_idx]
            return df_matches

        def __get_similarity_scores(search_vector, corpus_vectors):
            """
            helper function
            compute the cosine similarity between a given vector vs all
            corpus vectors
            """
            scores = []
            ### function expect 2D array (n_samples, n_features)
            search_vector = search_vector.reshape(1, -1)
            for corpus_vector in corpus_vectors:
                corpus_vector = corpus_vector.reshape(1, -1)
                score = cosine_similarity(search_vector, corpus_vector)
                ### just extract and store the scalar score
                scores.append(score[0][0])
            return np.array(scores)

        with st.expander("Get Similar Movie Recommendations...", expanded=False):
            with st.form(key="recommend"):
                col1, col2 = st.columns(2)
                ### TF-IDF is True only for local run
                if USE_TFIDF:
                    options = ["SBERT", "TFIDF"]
                    help_text = """TF-IDF: A simple algorithm to convert text 
                            into vectors, quick but accuracy is low.
                            SBERT: Sentence Transformer, gives better accuracy"""
                else:
                    options = ["SBERT"]
                    help_text = "SBERT: Sentence Transformer"
                embed_type = col1.radio(
                    label="Embedding Type:",
                    options=options,
                    help=help_text,
                )
                k = col2.slider(
                    label="Number of Recommendations Requested:",
                    min_value=1,
                    max_value=5,
                    step=1,
                )
                if col1.form_submit_button(label="Recommend Movies"):
                    __recommend_movies(
                        curr_movie_index=st.session_state["curr_page"],
                        df=st.session_state["data"],
                        k=k + 1,
                        embed=embed_type.lower(),
                    )

            if st.button("Clear Recommendations"):
                if st.session_state["recommended"] == True:
                    _reset()

    ## invoke the func
    _render_recommend_panel()

    ## show movie plot text
    st.write(df.loc[curr_page, "plot"].replace("$", "\$"))
    st.divider()


def render_sidebar():
    """
    sidebar UI rendering and event handling/callback functionality
    """

    ### sidebar event handling functions
    def _jump_to_page(input_page_no):
        """
        jump to a given page
        """
        ## ensure page number is provided and is within valid range
        if len(input_page_no) > 0:
            input_page_no = int(input_page_no)
            if input_page_no <= 0 or input_page_no > st.session_state["last_page"] + 1:
                err_msg = f'⛔ Page no should be between 1 and {st.session_state["last_page"]+1}'
                st.sidebar.error(err_msg)
            else:
                st.session_state["curr_page"] = input_page_no - 1
                st.experimental_rerun()

    def _search_by_title(search_string):
        """
        filter the data to show only pages that match with search title
        """
        ## if filter is attempted on already filtered data then throw error
        ## and set search_string to "" so that no further filtering happens
        if len(st.session_state["filter"]) > 0 and len(search_string) > 0:
            err_msg = f"⛔ A filter is already applied, 'Reset' to clear it first"
            st.sidebar.error(err_msg)
            search_string = ""

        ## ensure there is a search string entered, then filter the df by matches
        if len(search_string) > 0:
            df = st.session_state["data"]
            df_filtered = df[
                df["title"].str.contains(search_string, flags=re.IGNORECASE, regex=True)
            ].copy()
            ## if matches found
            if len(df_filtered) > 0:
                st.session_state["filter"] = search_string
                # set filtered df as active dataframe and re-fresh the page
                st.session_state["data"] = df_filtered.reset_index()
                st.session_state["curr_page"] = 0
                st.session_state["last_page"] = len(st.session_state["data"]) - 1
                st.experimental_rerun()
            ## if no matches found
            else:
                st.session_state["filter"] = ""
                err_msg = f"⛔ No matches found, please try again."
                st.sidebar.error(err_msg)

    ### render jump to page sidebar form
    with st.sidebar.form(key="jump_to_page", clear_on_submit=True):
        input_page_no = st.text_input(
            label="Jump to page:", placeholder="Enter a page number", key="page_no"
        )
        if st.form_submit_button(
            label="Jump",
        ):
            _jump_to_page(input_page_no)

    ### render search by title sidebar form
    with st.sidebar.form(key="filter_pages", clear_on_submit=True):
        search_string = st.text_input(
            label="Search by movie title:",
            placeholder="Enter search text",
            key="search_string",
        ).strip()

        if st.form_submit_button(
            label="Search",
        ):
            _search_by_title(search_string)

        ## if there is already a filter applied then show this filter as info
        if len(st.session_state["filter"]) > 0:
            info_msg = (
                f"ℹ️ Current filter: [{st.session_state['filter']}], 'Reset' to clear"
            )
            st.sidebar.info(info_msg)

    ### the app page reset button
    if st.sidebar.button(label="Reset"):
        _reset()


####
#### MAIN APP FLOW
####

### initialize page title, icon, app data and session_state
init(title="Movie Buddy", icon="🎥")

### build page UI and show content
render_page()

### build page sidebar UI and show content
render_sidebar()
