import uuid
import pandas as pd
import numpy as np
from .prompts import extractConcepts, graphPrompt, extractConceptsOAI, graphPromptOAI
from openai import OpenAI



def documents2Dataframe(documents) -> pd.DataFrame:
    rows = []
    for chunk in documents:
        row = {
            "text": chunk.page_content,
            **chunk.metadata,
            "chunk_id": uuid.uuid4().hex,
        }
        rows = rows + [row]

    df = pd.DataFrame(rows)
    return df


def df2ConceptsList(dataframe: pd.DataFrame, OAI=False) -> list:
    # dataframe.reset_index(inplace=True)
    if OAI:
        model = "gpt-3.5-turbo-1106"
        print("Parsing ", dataframe.shape[0], " rows to OpenAI API")
        results = dataframe.apply(
            lambda row: extractConceptsOAI(
                row.text, {"chunk_id": row.chunk_id, "type": "concept"}
            ),
            axis=1,
        )
    else:
        results = dataframe.apply(
            lambda row: extractConcepts(
                row.text, {"chunk_id": row.chunk_id, "type": "concept"}
            ),
            axis=1,
        )
    # invalid json results in NaN
    results = results.dropna()
    results = results.reset_index(drop=True)

    ## Flatten the list of lists to one single list of entities.
    concept_list = np.concatenate(results).ravel().tolist()
    return concept_list


def concepts2Df(concepts_list) -> pd.DataFrame:
    ## Remove all NaN entities
    concepts_dataframe = pd.DataFrame(concepts_list).replace(" ", np.nan)
    concepts_dataframe = concepts_dataframe.dropna(subset=["entity"])
    concepts_dataframe["entity"] = concepts_dataframe["entity"].apply(
        lambda x: x.lower()
    )

    return concepts_dataframe


def df2Graph(dataframe: pd.DataFrame, model="mistral-openorca:latest", OAI=False) -> list:
    # dataframe.reset_index(inplace=True)
    client = OpenAI()
    if OAI:
        model = "gpt-3.5-turbo-1106"
        print("Parsing", dataframe.shape[0], "rows to OpenAI API")
        dataframe = dataframe.head(5)
        results = dataframe.apply(
            lambda row: graphPromptOAI(row.text, {"chunk_id": row.chunk_id}, model, client), axis=1
        )
    else:
        results = dataframe.apply(
            lambda row: graphPrompt(row.text, {"chunk_id": row.chunk_id}, model), axis=1
        )
    # invalid json results in NaN
    results = results.dropna()
    results = results.reset_index(drop=True)

    ## Flatten the list of lists to one single list of entities.
    concept_list = np.concatenate(results).ravel().tolist()
    return concept_list


def graph2Df(nodes_list) -> pd.DataFrame:
    ## Remove all NaN entities
    graph_dataframe = pd.DataFrame(nodes_list).replace(" ", np.nan)
    graph_dataframe = graph_dataframe.dropna(subset=["node_1", "node_2"])
    graph_dataframe["node_1"] = graph_dataframe["node_1"].apply(lambda x: x.lower())
    graph_dataframe["node_2"] = graph_dataframe["node_2"].apply(lambda x: x.lower())

    return graph_dataframe