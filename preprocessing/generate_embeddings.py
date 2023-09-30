"""
Generate TF-IDF & SBERT embeddings for a given text corpus and save data to disk in 
parquet format
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


def __generate_tfidf_embeddings(df):
    """function to generate TF-IDF embeddings"""
    ### get a TF-IDF representation of text, returned values are sparse vectors
    print(f"Generating TF-IDF embeds for movie plot text...")
    vec = TfidfVectorizer()
    X = vec.fit_transform(df["plot"]).toarray()

    ### convert sparse vectors into a dataframe
    df_embeddings = pd.DataFrame(X)
    return df_embeddings


def __generate_sbert_embeddings(df):
    """function to generate SBERT embeddings"""
    ### get a SBERT representation of a plot, returned values are 384 dim vector
    print(f"Generating SBERT embeds for movie plot text...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_embeddings = model.encode(df["plot"])

    ### convert sparse vectors into a dataframe
    df_embeddings = pd.DataFrame(sentence_embeddings)
    return df_embeddings


def __read_plot_corpus(input_file):
    """function to load plot text data"""
    ### load the dataframe containing the movie plot corpus
    print(f"\nLoading movie plot's text corpus from [{input_file}]...\n")
    df = pd.read_parquet(input_file)
    return df


def __save_embeddings(df, parquet_output_file):
    """function to write processed data to disk"""
    ### store the embeddings as arrow table in a parquet file
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_output_file)
    print(f"Saved embeddings to [{parquet_output_file}]\n")


if __name__ == "__main__":
    target_dir = "../artifacts/"
    input_file = target_dir + "movie_plots.parquet"
    tfidf_output_file = target_dir + "tfidf_embeddings.parquet"
    sbert_output_file = target_dir + "sbert_embeddings.parquet"

    ### load the dataframe containing the movie plot corpus
    df_plots = __read_plot_corpus(input_file)

    ### generate and save TFIDF embeddings
    df_embeddings = __generate_tfidf_embeddings(df_plots)
    __save_embeddings(df_embeddings, tfidf_output_file)
    ### generate and save SBERT embeddings
    df_embeddings = __generate_sbert_embeddings(df_plots)
    __save_embeddings(df_embeddings, sbert_output_file)
