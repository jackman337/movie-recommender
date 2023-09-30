---
title: Movie Recommender
emoji: 👀
colorFrom: red
colorTo: pink
sdk: streamlit
sdk_version: 1.27.1
app_file: app.py
pinned: false
license: mit
---

<a href="https://huggingface.co/spaces/sssingh/movie-recommender" target="_blank"><img src="https://img.shields.io/badge/click_here_to_open_streamlit_app-f63366?style=for-the-badge&logo=streamlit&logoColor=black" /></a>


<img src="https://github.com/sssingh/movie-recommender/blob/main/images/title.png?raw=true" width="1000" height="350"/><br><br> 

# Move Recommender

***Movie Buddy -  A Natural Language Processing (NLP) based a movie recommender app.***

>An easy-to-use app that allows users to view the movie plots and synopses of more than 3,500 English-language films released between 2005 and 2022. The app can suggest up to five films that, in terms of story, emotions, setup, etc., are akin to the film in question based on the text of the movie's plot. The app enables users to view and visualize all 3,500+ movies using a scatter plot (2- or 30-dimensional) that displays very similar movies as data points close to one another. This app contains three pages. "About", "Movie Explorer", and "Movie Visualizer"

Although this app is designed to propose similar movies, the same concepts may be applied in a variety of real-world situations...
* **`eCommerce`**: Given a product description (the product that customer is interested in buying) a similar or related product can be suggested to customer to increase sales.
* **`Advertising`**:  Makes use of recommendation systems to deliver customized recommendations for adverts and offers that are likely to be of interest to users.
* **`Entertainment`**: Recommendation systems are used by music and video streaming services, to propose material that is likely to be of interest to viewers.
* **`Education`**: Based on a student's browsing and reading habits, the library can recommend analogous more relevant books (which he/she may be unaware of otherwise) to pique his/her interest in the subject matter.

# App Details

## About Page
Summary of the application and high-level information

### Quick Demo

<img src="https://github.com/sssingh/movie-recommender/blob/main/images/about.gif?raw=true" width="1000" height="450" />

## Movie Explorer Page 
A page where users can browse movie plots and find recommendations for similar films.  

### Quick Demo
<img src="https://github.com/sssingh/movie-recommender/blob/main/images/explorer-1.gif?raw=true" width="1000" height="450" /> <br><br>
<img src="https://github.com/sssingh/movie-recommender/blob/main/images/explorer-2.gif?raw=true" width="1000" height="450" /> <br><br>

### Sidebar Section (Left)
* **Jump To Page Form**: There are 3,500+ movie plots available to explore, this using this option one can directly go a page he/she wishes to view by entering the `page number` and clicking the `Jump` button.
* **Search by movie title Form**: This option allows user to search movie by their title. Movies can searched by entering a `search text`and clicking the `Search` button, this will filter and show only those movies with title containing search term. For example if "batman" is searched then all movies with batman in their title is shown.
* **Reset Button**: Resets the app, removing any search restrictions and any recommendations (explained in the content section below), and displays a list of all movies in the dataset.

### Content Section (Right)
* This page displays one movie plot at a time, along with the movie's title and Wikipedia URL. 
* The top right corner displays the current page number and the total number of pages (i.e., movies) accessible. 
* Using the "Prev Page" and "Next Page" buttons, viewers can move between different movies.
* "Get Similar Movie Recommendations" is a collapsible panel that expands when clicked to disclose the options that can be chosen to obtain movie recommendations:  
    * **Embedding Type Radio Buttons**: Type of encoding utilized to translate words from movie plots into numbers that underpin underlying math. Compared to "TFIDF," the "SBERT" (default) option yields superior results.
    * **Requested Number of Recommendations Slider**: Requested number of similar films to be viewed, up to a maximum of five films.
    * **Recommend Movies Button**: Based on the options you've chosen, this feature determines and displays a list of suggested movie plots, organized by the related "similarity-score" (which runs from 0.0 to 1.0, the higher the score, the more similar the movie is). The movie we're looking for is always the first one displayed (with a similarity score close to 1.0).
    * The "Clear Recommendations" button lists all of the movies in the dataset after clearing the recommendations.
    
## Movie Visualizer Page

### Quick Demo
<img src="https://github.com/sssingh/movie-recommender/blob/main/images/visualizer.gif?raw=true" width="1000" height="450" /> <br><br>

* Displays films as data points in an interactive chart (2-D by default). Movies are plotted as dots/bubbles in 2-D/3-D scatter plots according to how similar they are. In the chart, similar movies are displayed closer together. 
* Hover your mouse over the top right corner of the chart to display toolbar options to zoom, pan, download as image, etc. 
* When using a 2-D chart, you can zoom in on a specific area by clicking and dragging to make a box selection. 
* When using a 3-D chart, you can zoom into the graph by choosing the zoom option and using the mouse to click and drag (or use mouse wheel). Select the rotation option from the toolbar, then drag the mouse to rotate the graph around any of the three axes. 
* To view the movie's specifics, hover over the dots or bubbles.

# Project Source
[👉 Visit GitHub Repo](https://github.com/sssingh/movie-recommender)

# Contact Me
[![email](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:sunil@sunilssingh.me)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/@thesssingh)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sssingh/)
[![website](https://img.shields.io/badge/web_site-8B5BE8?style=for-the-badge&logo=ko-fi&logoColor=white)](https://sunilssingh.me)


# Appendix

## Technical Overview

<img src="https://github.com/sssingh/movie-recommender/blob/main/images/movie-recommender.png?raw=true" width="1000" height="450" /> <br><br>

* The dataset, which includes English movies released between 2005 and 2022, is a corpus of approximately 3,500 movies "plot" texts collected from Wikipedia. 
* The text corpus has been encoded using:
  * [TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
  * [SentenceTransformers (SBERT)](https://www.sbert.net/) embeddings.
* The application uses the [Cosine Similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) metric to determine which movies in the corpus have plots that are similar to a given one.
* The [Uniform Manifold Approximation and Projection for Dimension Reduction (UMAP)](https://umap-learn.readthedocs.io/en/latest/) algorithm is used in the application to reduce SBERT embeddings from 384 dimensions to just 3. The 'plotly' package was used to build the interactive plot.

## Install Locally and Run
### Create Environment
Create a conda (or virtual environment), activate the environment. Conda example command is given below: 
```
conda create --name <my-env-name>
conda activate <my-env-name>
```

### Clone the repo
```
git clone https://github.com/sssingh/movie-recommender
```
**NOTE:** by default TF-IDF embeddings are kept off due to its large size not supported
by free tier of cloud deployment. However, while running locally it can be switched on
by making `USE_TFIDF` to `True` in `explore_movies.py`.

### Install the dependencies  
NOTE: In order to run the data-preprocessing locally (described under Supplementary Details) uncomment the packages listed in requirements.txt at the bottom. These are not required for web deployment of the app hence commented in this repo.
```
cd movie-recommender
pip -r requirements.txt
```

### Run the app
from within `movie-recommender` folder run the application server:
```
streamlit run app.py
```
open the app in the browser by visiting `http://localhost:8501/`

### Supplementary Details
#### Data Collection
Wikipedia pages were scraped to gather the raw data. The repository's 'preprocessing/collect_data.py' python script file contains the source code for the data collection script; to run it, to kick-off data-collection again run the command below from the 'preprocessing' folder.:
```
python collect_data.py
```

**NOTE** Depending on your hardware and internet speed, the aforementioned script may take 30 to 60 minutes to execute. The repository contains the curated and prepared raw movie plots dataset `artifacts/movie_plots.parquet`. 

#### Data Preprocessing
* 'TF-IDF' and 'SBERT' embeddings are produced from the above mentioned raw movie plots dataset and the generated embeddings are stored in `artifacts/tfidf_embeddings.parquet` & `artifacts/sbert_embeddings.parquet` files in this repository. 
* The repository's python script file 'preprocessing/generate_embeddings.py' contains the embedding generation source code. 
* To generate embeddings again run the command below from inside the folder 'preprocessing' folder.:
```
python generate_embeddings.py
```


