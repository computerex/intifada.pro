from openai import OpenAI
import os
import sys
import re
import numpy as np
import logging
import json
# cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(enc.encode(text))

def count_tokens_messages(messages):
    return sum([count_tokens(m['content']) for m in messages])

deepinfra_key  = json.loads(open('.env.json').read())['DEEP_INFRA_API_KEY']
openai_key = json.loads(open('.env.json').read())['OPEN_AI_API_KEY']

app = Flask(__name__)

answers_embeddings_global = {}
CHAT_SYSTEM_MESSAGE = open('prompts/system.txt', 'r', encoding='utf-8').read()

def get_llm_client(model):
    if model.startswith('gpt'):
        return OpenAI(
            api_key=openai_key,
        )
    
    client = OpenAI(
        api_key=deepinfra_key,
        base_url="https://api.deepinfra.com/v1/openai",
    )
    return client

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    client = OpenAI(api_key=openai_key)
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_embedding_bulk(docs, model="text-embedding-3-small"):
    client = OpenAI(api_key=openai_key)
    response = client.embeddings.create(input = docs, model=model).data
    return [r.embedding for r in response]

def complete_chat(**kwargs):
    client = get_llm_client(kwargs['model'])
    response = client.chat.completions.create(
        **kwargs
    )
    return response.choices[0].message.content

def cosine_similarity_scoring(input_query, answers_embeddings):
    query_embedding = get_embedding(input_query)
    scores = {k: cosine_similarity([query_embedding], [v])[0][0] for k, v in answers_embeddings.items()}
    # sort by descending score
    scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    return scores

def get_filename_from_answer(answer):
    return re.sub(r' ', '-', answer) + '.txt'

def is_question(text):
    SYSTEM = f"You are identifying whether the following chat message is a question or not. This is chat so punctuation may be absent, err on the side of caution and default to yes if not certain: \n\n{text}\n\nIs this a question? Answer 'yes' or 'no'."
    USER = f"Is this a question? Answer 'yes' or 'no'. \n\n{text} \n\n Answer:"
    response = complete_chat(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
        max_tokens=50,
    )
    return 'yes' in response.lower()

def load_handala_answers():
    answers = {}
    for answer in os.listdir('answers'):
        with open(f'answers/{answer}', 'r', encoding='utf-8') as f:
            answers[answer] = f.read()

    return answers

# serve static/index.html
@app.route('/', methods=['GET'])
def index():
    return app.send_static_file('index.html')

@app.route('/media-bias', methods=['GET'])
def media_bias_index():
    return app.send_static_file('media-bias/index.html')

@app.route('/chat', methods=['POST'])
def chat_message():
    req_data = request.get_json()
    messages = req_data

    messages_cleaned = []
    for message in messages:
        if message['role'] != 'system':
            messages_cleaned.append(message)

    last_message = messages[-1]
    if is_question(last_message['content']):
        query = last_message['content']
        scores = cosine_similarity_scoring(query, answers_embeddings_global)
        top_answers = list(scores.keys())[:1]
        logging.info({k: scores[k] for k in top_answers})

        response = f"""I am a pro-palistinian debating a zionist. Create a comprehensive, detailed response to this question using the information below. 
        Use all the specific key points and information to create the response. But do not refer the information itself! Maintain the 3rd wall. Make the answer self-contained and ready to post.
        Use lists and bullet points whenever possible as they are easier to read.

        Question: '{query}':\n\n"""
        for i, answer in enumerate(top_answers):
            answer_text = open(f'answers/{answer}', 'r', encoding='utf-8').read()
            response += f"{answer}\n\n{answer_text}\n\n"

        response += "Response:"

        logging.info(response)
        # remove user's last message
        messages_cleaned = messages_cleaned[:-1]
        messages_cleaned.append({"role": "user", "content": response})
    
    # add system message as the first message in messages_cleaned
    messages_cleaned.insert(0, {"role": "system", "content": CHAT_SYSTEM_MESSAGE})

    response = complete_chat(
        model="gpt-3.5-turbo",
        messages=messages_cleaned,
        max_tokens=min(16000 - count_tokens_messages(messages_cleaned), 4096),
    )
    return jsonify({"message": response})

@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({"message": "hello world"})

def main():
    global answers_embeddings_global

    # Configure basic logging
    logging.basicConfig(level=logging.INFO)

    answers = load_handala_answers()
    keys = list(answers.keys())

    if not os.path.exists('embeddings.npy'):
        embeddings = get_embedding_bulk(keys)
        answer_embeddings = {k: v for k, v in zip(keys, embeddings)}
        np.save('embeddings.npy', answer_embeddings)
    else:
        logging.info('loading embeddings')
        answer_embeddings = np.load('embeddings.npy', allow_pickle=True).item()
    
    answers_embeddings_global = answer_embeddings
    from waitress import serve
    serve(app, host="0.0.0.0", port=8001)

if __name__ == "__main__":
    main()

