import json
import ast
import logging
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_classic.text_splitter import CharacterTextSplitter
from langchain_classic.schema import Document

logger = logging.getLogger(__name__)


def robust_parse(s):
    """Parse stringified JSON/Python literals from TMDB columns."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(s)
        except Exception:
            return []


def get_director(crew_str: str) -> str:
    crew = robust_parse(crew_str)
    for member in crew:
        if member.get("job") == "Director":
            return member.get("name", "")
    return ""


def get_cast(cast_str: str, limit: int = 5) -> str:
    cast_list = robust_parse(cast_str)
    return ", ".join([actor["name"] for actor in cast_list[:limit]])


def load_and_clean(movies_path: str, credits_path: str) -> pd.DataFrame:
    """Load raw CSVs, merge, and extract structured fields."""
    logger.info("Loading raw TMDB data")
    df_movies = pd.read_csv(movies_path)
    df_credits = pd.read_csv(credits_path)
    df = df_movies.merge(df_credits, on="title")

    logger.info(f"Merged dataset: {len(df)} movies")

    df["Genres"] = df["genres"].apply(
        lambda x: ", ".join([i["name"] for i in robust_parse(x)])
    )
    df["Director"] = df["crew"].apply(get_director)
    df["Cast"] = df["cast"].apply(get_cast)

    df["combined_info"] = df.apply(
        lambda row: (
            f"Type: Movie, Title: {row['title']}, "
            f"Director: {row['Director']}, Cast: {row['Cast']}, "
            f"Released: {row['release_date']}, Genres: {row['Genres']}, "
            f"Vote_Average: {row['vote_average']}, "
            f"Description: {row['overview']}"
        ),
        axis=1,
    )

    logger.info(f"Cleaned dataset: {len(df)} movies")
    return df


def save_combined_csv(df: pd.DataFrame, output_path: str):
    """Save the combined_info column for downstream loading."""
    df[["combined_info"]].to_csv(output_path, index=False)
    logger.info(f"Saved combined CSV to {output_path}")


def load_and_split_documents(
    csv_path: str, chunk_size: int = 1000, chunk_overlap: int = 30
) -> list[Document]:
    """Load the combined CSV and split into LangChain documents."""
    loader = CSVLoader(file_path=csv_path, encoding="utf-8")
    data = loader.load()

    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n",
    )
    docs = splitter.split_documents(documents=data)
    logger.info(f"Split into {len(docs)} document chunks")
    return docs


def run_preprocessing(config: dict) -> list[Document]:
    """Full preprocessing pipeline: load → clean → save → split."""
    df = load_and_clean(
        movies_path=config["data"]["movies_path"],
        credits_path=config["data"]["credits_path"],
    )
    save_combined_csv(df, config["data"]["combined_csv_path"])
    docs = load_and_split_documents(
        csv_path=config["data"]["combined_csv_path"],
        chunk_size=config["retrieval"].get("chunk_size", 1000),
        chunk_overlap=config["retrieval"].get("chunk_overlap", 30),
    )
    return docs