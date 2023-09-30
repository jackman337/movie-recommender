""" Extract and collect American movie plots from wikipedia """

from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from urllib import request


def __get_plot(movie_page):
    """Open the given movie webpage, extract and return the plot text for this movie"""
    plot = ""
    try:
        html = request.urlopen(movie_page)
        soup = BeautifulSoup(html, "lxml")
        h = soup.find(id="Plot").find_parent()
        elem = h.find_next_sibling()
        while elem.name == "p":
            plot = plot + elem.text
            elem = elem.find_next_sibling()
    except:
        pass
    return plot


def __extract_movie_plot(movie_url_list):
    """Run through the movie URL list and extract movie plot for each movie"""
    plots = []
    for url in tqdm(movie_url_list):
        plot = __get_plot(url)
        plots.append(plot)
    return plots


def __process_one_year(year_page, base_url, movie_list):
    """Process the page containing movie list for one year and collect the details"""
    # create soup object from html obtained for parsing
    soup = BeautifulSoup(year_page, "lxml")
    all_tables = soup.find_all("table", class_="wikitable")
    for table in all_tables:
        # locate the tables containing list of movies for this year
        ths = table.find_all("th")
        if ths:
            th_text = [th.text.strip() for th in ths]
            # if this is our table with list of movies then extract the movie tile and URL
            if "Production company" in th_text and "Title" in th_text:
                trs = table.tbody.find_all("tr")
                for tr in trs:
                    movie = tr.find_all("i")
                    # if tag found i.e. its non-empty
                    if movie:
                        a = movie[0].find("a")
                        if a:
                            movie_list["url"].append(base_url + a["href"])
                            movie_list["title"].append(movie[0].a["title"])
    return movie_list


def __process_yearly_list(yearly_list_url, base_url):
    """Run through the list of years and process each year one by one and grab the movie URLS"""
    movie_list = {"url": [], "title": []}
    for url in yearly_list_url:
        try:
            html = request.urlopen(url)
            movie_list = __process_one_year(html, base_url, movie_list)
        except:
            pass
    return movie_list


def __get_url_list_by_year(root_url, start_year, end_year):
    """Get the list of URL containing movie list by year"""
    yearly_list_url = [root_url + str(year) for year in range(start_year, end_year + 1)]
    return yearly_list_url


def preprocess_data(start_year, end_year):
    """main function to orchestrate data processing"""
    # Get the list of URL containing movie list by year
    base_url = "https://en.wikipedia.org"
    root_url = base_url + "/wiki/List_of_American_films_of_"
    print(f"\nBuilding yearly url list from year {start_year} to {end_year}...")
    yearly_list_url = __get_url_list_by_year(root_url, start_year, end_year)
    print(f"DONE. Collected {len(yearly_list_url)} year urls\n")

    # Run through the list of years and process each year one by one and grab the
    # movie URLS
    print(f"Building movie url list from yearly url list...")
    movie_list = __process_yearly_list(yearly_list_url, base_url)
    print(f"DONE. Collected {len(movie_list['url'])} movie urls\n")

    # Extract the plot from each movie URL
    print(f"Extracting movie plot from movie urls...")
    movie_plots = __extract_movie_plot(movie_list["url"])

    # Save the collected movie data as compressed parquet file and failed URLS as CSV
    movie_list["plot"] = movie_plots
    df = pd.DataFrame(movie_list)
    df_failed = df[df["plot"] == ""].copy()
    failed_index = df_failed.index
    df = df.drop(failed_index, axis=0)
    df = df.reset_index()

    print(f"Collected {len(df)} movie plots successfully.")
    print(f"Failed to collect {len(df_failed)} movie plots.")
    return df, df_failed


def save_data(df, df_failed, parquet_data_file_, csv_fail_file):
    """function to write processed data to disk"""
    df.to_parquet(parquet_data_file_)
    print(f"Saved movie plots data to [{parquet_data_file_}]")
    df_failed.to_csv(csv_fail_file)
    print(f"Saved failed movie urls to [{csv_fail_file}]\n")


if __name__ == "__main__":
    start_year = 2005
    end_year = 2022
    rel_dir_name = "../artifacts/"
    parquet_data_file_ = rel_dir_name + "movie_plots.parquet"
    csv_fail_file = rel_dir_name + "failed_plots.csv"

    ### collect, pre-process and save the data
    df, df_failed = preprocess_data(start_year, end_year)
    save_data(df, df_failed, parquet_data_file_, csv_fail_file)
