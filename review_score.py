from concurrent.futures import ThreadPoolExecutor
import tqdm
import csv
import os
import requests
import dotenv
from joblib import Memory

dotenv.load_dotenv()

API_BASE_URL = os.environ['API_BASE_URL']
headers = {'Authorization': 'Bearer '+os.environ['API_TOKEN']}


session = requests.session()


def run(model, input):
    response = session.post(f'{API_BASE_URL}{model}', headers=headers, json=input)
    return response.json()


memory = Memory('./cache')


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
        rows = list(reader)
        for row in rows:
            row['latest_reviews'] = row['latest_reviews'].removeprefix("['").removesuffix("']").split("', '")
        pb = tqdm.tqdm(total=sum([len(row['latest_reviews']) for row in rows]))
        pool = ThreadPoolExecutor(max_workers=10)

        def f(row):
            name = row['name']
            reviews = row['latest_reviews']

            scores = []
            for review in reviews:
                scores.append(get_score(review))
                pb.update(1)

            return [name, scores]

        for x in pool.map(f, rows):
            results.append(x)

    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['name', 'scores'])

        for result in results:
            writer.writerow(result)


process_reviews('p.csv', 'review_scores.csv')
