import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler
import umap
import plotly.express as px

if __name__ == "__main__":
    ### load the movie data
    input_file = "../artifacts/movie_plots.parquet"
    print(f"\nLoading movie data from [{input_file}]...")
    df = pd.read_parquet("../artifacts/movie_plots.parquet")
    ### load sbert embeds as pandas dataframe of 384 columns
    input_file = "../artifacts/sbert_embeddings.parquet"
    print(f"Loading embeddings from [{input_file}]...")
    sbert_df = pq.read_table(input_file).to_pandas()

    ## standardize the data, this helps UMAP to converge quickly and produce better o/p
    scaler = StandardScaler()
    sbert_df = pd.DataFrame(scaler.fit_transform(sbert_df))

    ### use UMAP to reduce sbert dimention to 3
    print(f"Running UMAP...")
    reducer = umap.UMAP(n_components=3, random_state=42)
    reduced = reducer.fit_transform(sbert_df)

    ### concat the reduced dim with movie url and title
    sbert_df = pd.DataFrame(data=reduced, columns=["comp_1", "comp_2", "comp_3"])
    sbert_df = pd.concat([df[["url", "title"]], sbert_df], axis=1)

    ### save the reduced data as parquet file
    output_file = "../artifacts/umap_reduced_data.parquet"
    sbert_df.to_parquet(output_file)
    print(f"Saved UMAP reduced output to [{output_file}].")
