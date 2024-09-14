import tqdm
import csv
import os
from joblib import Memory
import requests
import dotenv

dotenv.load_dotenv()

API_BASE_URL = os.environ['API_BASE_URL']
headers = {'Authorization': 'Bearer '+os.environ['API_TOKEN']}


def run(model, input):
    response = requests.post(f'{API_BASE_URL}{model}', headers=headers, json=input)
    return response.json()


memory = Memory('./cache', verbose=0)


@memory.cache
def get_score(review):
    output = run('@cf/huggingface/distilbert-sst-2-int8', {'text': review})
    for x in output['result']:
        if x['label'] == 'POSITIVE':
            return x['score']
    raise


def process_reviews(input_csv, output_csv):
    results = []

    with open(input_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm.tqdm(list(reader)):
            name = row['name']
            reviews = row['latest_reviews'].removeprefix("['").removesuffix("']").split("', '")

            scores = [get_score(review) for review in reviews]

            results.append([name, scores])

    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['name', 'scores'])

        for result in results:
            writer.writerow(result)


process_reviews('p.csv', 'review_scores.csv')
