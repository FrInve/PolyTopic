import utils
import pandas as pd
import numpy as np
from openai import OpenAI
import pickle




def main():


    # Define data path and load data
    df_name = 'dbpedia'
    df_path = f"data/processed/parquet/{df_name}_text.parquet"
    df = pd.read_parquet(df_path)
    docs = df['text'].tolist()

    general_path = f"saved/{df_name}_ctm"
    dict_path = f"{general_path}/{df_name}_topics_dict.pkl"
    topics_dict = utils.load_dict(dict_path)

    final_topics_path = f"{general_path}/{df_name}_final_topics.pkl"
    final_topics = utils.load_dict(final_topics_path)

    word_list_path = f"{general_path}/{df_name}_topics_word_list.json"
    topics_word_list = utils.load_json(word_list_path)


    random_ids = np.load(f"saved/{df_name}_random_samples.npy") 

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="----"
    )

    output_list = []
    counter = 0

    # Build content prompt for each document and evaluate
    for chosen_id in random_ids:
        document, topic_text, num_topics = extract_doc_and_topics(chosen_id, docs, topics_word_list, final_topics)
        content_prompt = create_content_prompt(document, topic_text, num_topics)
        # print(content_prompt)

        result_text = evaluate_with_openai(client=client, content_prompt=content_prompt)
        output_list.append(result_text)

        counter += 1
        if counter%50 == 0:
            print(f"Iteration: {counter}")
            print(result_text)

        # print(result_text)
    
    with open(f"{general_path}/{df_name}_output_list_new", "wb") as f:
        pickle.dump(output_list, f)
    
    




def extract_doc_and_topics(chosen_id, docs, topics_word_list, final_topics):
    document = docs[chosen_id]
    topics_id = [t for t, prob in final_topics[chosen_id]]
    topics = []

    for id in topics_id:
        topics.append(topics_word_list[id])
    
    topic_text = ""

    for i in range(len(topics)):
        topic_text += f"["
        topic_text += f"{topics[i]}"
        topic_text += "]"

        if i < len(topics) - 1:
            topic_text += ", "
        else:
            topic_text += "."
    
    num_topics = len(topics)

    return document, topic_text, num_topics



def create_content_prompt(document, topic_text, num_topics):
    content_prompt = f"""
### Role
You are a data science expert. You possess a deep understanding of topic modeling on textual documents and of evaluation criteria.

### Instructions
Your task is to assess the quality of the topics assigned to the given document. Specifically, you must evaluate how accurately and comprehensively the assigned topics represent the core themes of the document. Each document is already pre-processed. 
Each topic consists of a set of words, with the format [word1 word2 ...]. Each document could be assigned multiple topics, separated by comma. The number of assigned topics is given in input.
Here are the questions you should answer:
Q1. Is the first assigned topic enough to adequately describe the document?
Q2. Does the overall list of assigned topics adequately describe the document?
Q3. Is the document better represented by the entire list than by the first topic only, if there are multiple topics? (Answer accordingly to Q1 if there is only one topic)
Q4. Provide a score from 1 to 5 about the overall quality of the list of the assigned topic, where 1 stands for poor (the topics are not sufficient to represent the document), and 5 stands for excellent (the topics describe all the content of the document).

Answer EXACTLY with the following JSON schema, with JSON booleans (true or false) for the first three questions (no extra text):
{{
    "Q1": true/false,
    "Q2": true/false,
    "Q3": true/false,
    "Q4": int
}}

### Example
Document text:
Yields on certificates of deposit issued by the United Arab Emirates Central Bank were unchanged at 6-1/8 pct, the bank said. The yield applies to maturities of one, two, three and six months.

Number of assigned topics: 3

Assigned topics:
[pct rate bank rates money said interest banks market cut], [bank foreign exchange reserves banks central currency capital rate government], [gulf iran said iranian oil attack states military war united].

Output:
{{
    "Q1": false,
    "Q2": true,
    "Q3": true,
    "Q4": 4
}}


### Input
Document text:
{document}

Number of assigned topics: {num_topics}

Assigned topics:
{topic_text}
"""
    return content_prompt


def evaluate_with_openai(client, content_prompt):
  # First API call with reasoning
    response = client.chat.completions.create(
    model="google/gemma-4-31b-it",
    messages=[
            {
              "role": "user",
              "content": content_prompt
            }
          ],
    extra_body={"reasoning": {"enabled": False}}
  )

    # Extract the assistant message with reasoning_details
    response = response.choices[0].message
    # print(response.content)

    return response.content



if __name__ == "__main__":
    main()
