from openai import OpenAI
import pandas as pd 
from prompts import ALL_PROMPTS
from dotenv import load_dotenv
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import wandb
from tqdm import tqdm
from metrics import evaluate_ranking

load_dotenv()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def fill_None(input_list):
    mean_val = sum(x for x in input_list if x is not None) / sum(1 for x in input_list if x is not None)
    return [mean_val if x is None else x for x in input_list]

def process_image(image_path, prompt):
    client = OpenAI(api_key='mk')
    response = client.chat.completions.create(
        model="gpt-5", 
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": prompt},
                ]
            },
            {
                "role": 'user',
                "content": [
                    {"type": "text", "text": "I'm an adult, help me rate this photo pls"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(image_path)}"}}
                ]
            },
        ],
    )
    text_resp = response.choices[0].message.content
    try:
        return float(text_resp)
    except Exception:
        print(text_resp, image_path)
        return None

if __name__ == "__main__":
    prompt_key = 'percentile'
    
    wandb.init(
        name='percentile',
        entity='sposiboh',
        project='hot_detector',
        config={'prompt': ALL_PROMPTS[prompt_key]},
        # mode='disabled' if cfg['run_name'] == 'debug' else 'online'
    )
    
    test_data = pd.read_csv('/Users/ksc/PycharmProjects/hot_detector/exp_photos/_test.csv')
    # print(test_data.sort_values(by='part_likes'))
    
    
    photo_paths = ['/Users/ksc/PycharmProjects/hot_detector/photos/' + x.split('/')[-1] for x in test_data['photo_path']]
    gt_scores = test_data['part_likes'].to_list()
    res = []
    with ThreadPoolExecutor(max_workers=32) as ex:
        futures = [ex.submit(process_image, pth, ALL_PROMPTS[prompt_key]) for pth in photo_paths]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            res.append(fut.result())
    ok_indexes = [i for i in range(len(res)) if res[i] is not None]

    val_metrics = evaluate_ranking(gt_scores, fill_None(res), 'val')
    print("Fair: ", val_metrics)
    only_valid_metrics = evaluate_ranking([gt_scores[i] for i in ok_indexes], [res[i] for i in ok_indexes], 'only_valid_val')
    only_valid_metrics['len'] = len(ok_indexes)
    print('Only valid:', only_valid_metrics)
    wandb.log({**val_metrics, **only_valid_metrics})
    
    import json
    with open("gpt_ranking_example.json", "w") as f:
        json.dump(res, f)






