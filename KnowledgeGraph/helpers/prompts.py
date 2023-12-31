import sys
from yachalk import chalk
sys.path.append("..")

import json
import ollama.client as client
from openai import OpenAI

def clean_json_data(raw_data):
    # Find the index of the first occurrence of '['
    start_index = raw_data.find('[')

    # Find the index of the last occurrence of ']'
    end_index = raw_data.rfind(']')

    # Extract the JSON string
    json_data = raw_data[start_index:end_index+1]

    return json_data



def extractConcepts(prompt: str, metadata={}, model="mistral-openorca:latest"):
    SYS_PROMPT = (
        "Your task is extract the key concepts (and non personal entities) mentioned in the given context. "
        "Extract only the most important and atomistic concepts, if  needed break the concepts down to the simpler concepts."
        "Categorize the concepts in one of the following categories: "
        "[event, concept, place, object, document, organisation, condition, misc]\n"
        "Format your output as a list of json with the following format:\n"
        "[\n"
        "   {\n"
        '       "entity": The Concept,\n'
        '       "importance": The concontextual importance of the concept on a scale of 1 to 5 (5 being the highest),\n'
        '       "category": The Type of Concept,\n'
        "   }, \n"
        "{ }, \n"
        "]\n"
    )
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=prompt)
    try:
        result = json.loads(response)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result


def graphPrompt(input: str, metadata={}, model="mistral-openorca:latest"):
    if model == None:
        model = "mistral-openorca:latest"

    # model_info = client.show(model_name=model)
    # print( chalk.blue(model_info))

    SYS_PROMPT = (
        "You are a network graph maker who extracts terms and their relations from a given context. "
        "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
        "of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n"
        "Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.\n"
            "\tTerms may include object, entity, location, organization, person, \n"
            "\tcondition, acronym, documents, service, concept, etc.\n"
            "\tTerms should be as atomistic as possible\n\n"
        "Thought 2: Think about how these terms can have one on one relation with other terms.\n"
            "\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n"
            "\tTerms can be related to many other terms\n\n"
        "Thought 3: Find out the relation between each such related pair of terms. \n\n"
        "Format your output as a list of json. Each element of the list contains a pair of terms"
        "and the relation between them, like the follwing: \n"
        "[\n"
        "   {\n"
        '       "node_1": "A concept from extracted ontology",\n'
        '       "node_2": "A related concept from extracted ontology",\n'
        '       "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n'
        "   }, {...}\n"
        "]"
    )

    USER_PROMPT = f"context: ```{input}``` \n\n output: "
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT)
    try:
        result = json.loads(response)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result


def extractConceptsOAI(prompt: str, metadata={}, model="gpt-3.5-turbo-1106", client=OpenAI()):
    # Construct the prompt for OpenAI
    sys_prompt = (
        "Your task is extract the key concepts (and non personal entities) mentioned in the given context. "
        "Extract only the most important and atomistic concepts, if needed break the concepts down to the simpler concepts."
        "Categorize the concepts in one of the following categories: "
        "[event, concept, place, object, document, organisation, condition, misc]\n"
        "Format your output as a list of json with the following format:\n"
        "[\n"
        "   {\n"
        '       "entity": The Concept,\n'
        '       "importance": The contextual importance of the concept on a scale of 1 to 5 (5 being the highest),\n'
        '       "category": The Type of Concept,\n'
        "   }, \n"
        "{ }, \n"
        "]\n\n"
    )
    
    # Append the user's prompt to the system prompt
    full_prompt = sys_prompt + "Context: " + prompt

    # Call the OpenAI API
    
    try:
        response = client.chat.completions.create(
            model=model,
            #response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
                ]
        )
        #print(response.choices[0].message.content)
        result = clean_json_data(response.choices[0].message.content)
        result = json.loads(result)
        #print(result)
        result = [dict(item, **metadata) for item in result]
        print("Called OpenAI API")

    except Exception as e:
        print("\n\nERROR ### Here is the buggy response: ", str(e), response.choices[0].message.content, "\n\n")
        result = None
    
    return result


def graphPromptOAI(input: str, metadata={}, model="gpt-3.5-turbo-1106", client=OpenAI()):
    sys_prompt = (
        "You are a network graph maker who extracts terms and their relations from a given context. "
        "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
        "of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n"
        "Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.\n"
        "\tTerms may include object, entity, location, organization, person, \n"
        "\tcondition, acronym, documents, service, concept, etc.\n"
        "\tTerms should be as atomistic as possible\n\n"
        "Thought 2: Think about how these terms can have one on one relation with other terms.\n"
        "\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n"
        "\tTerms can be related to many other terms\n\n"
        "Thought 3: Find out the relation between each such related pair of terms. \n\n"
        "Format your output as a list of json. Each element of the list contains a pair of terms"
        "and the relation between them, like the following: \n"
        "[\n"
        "   {\n"
        '       "node_1": "A concept from extracted ontology",\n'
        '       "node_2": "A related concept from extracted ontology",\n'
        '       "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n'
        "   }, {...}\n"
        "]"
        "Respond only with list of JSON!! i.e. [ { }, { }, ... ], no other syntax or words!"
    )

    user_prompt = f"context: ```{input}``` \n\n output: "
    full_prompt = sys_prompt + user_prompt

    # Call the OpenAI API
    try:
        response = client.chat.completions.create(
            model=model,
            #response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
                ]
        )
        #print(response.choices[0].message.content)
        result = clean_json_data(response.choices[0].message.content)
        result = json.loads(result)
        #print(result)
        result = [dict(item, **metadata) for item in result]
        print("Called OpenAI API")

    except Exception as e:
        print("\n\nERROR ### Here is the buggy response: ", str(e), response.choices[0].message.content, "\n\n")
        result = None
    
    return result
