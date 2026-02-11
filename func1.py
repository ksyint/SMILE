import os
from PIL import Image
from clip_retrieval.clip_client import ClipClient
from googletrans import Translator
translator = Translator()

CLIP_PORT=13131
INDICE_PATH="index_h14"

def translate_kor_to_eng(text_kor) :
    text_en = translator.translate(text_kor, src='ko', dest='en').text
    return text_en

def search_by_text(input_text_kor, input_count) :
    text_en = translate_kor_to_eng(input_text_kor)
    client = ClipClient(url=f"http://localhost:{CLIP_PORT}/knn-service", indice_name=INDICE_PATH,
                        num_images=input_count, deduplicate=False,
                        use_safety_model=False, use_violence_detector=False)

    outputs = client.query(text=text_en)
    results = []
    for output in outputs :
        info = {}

        fn1, fn2 = output['image_path'].split('/')[-2:]
        num, _ = os.path.splitext(fn2)

        info['image_path'] = os.path.join(fn1, fn2)
        info['episode'] = int(fn1)
        info['num'] = int(num)
        info['similarity'] = output['similarity']

        results.append(info)
    
    return text_en, results

# input_text_kor = "한유현이 울고 있는 장면"
# character_name = "한유현"
# count1, count2, count3 = 1000, 1000, 10
def search_by_text_and_image(input_text_kor, character_name, count1, count2, count3) :
    text_en = translate_kor_to_eng(input_text_kor)
    client = ClipClient(url=f"http://localhost:{CLIP_PORT}/knn-service", indice_name=INDICE_PATH,
                        num_images=count1, deduplicate=False,
                        use_safety_model=False, use_violence_detector=False)
    outputs1 = client.query(text=text_en)

    img_path = f"./character_images/{character_name}.jpg"
    client = ClipClient(url=f"http://localhost:{CLIP_PORT}/knn-service", indice_name=INDICE_PATH,
                        num_images=count2, deduplicate=False,
                        use_safety_model=False, use_violence_detector=False)
    outputs2 = client.query(image=img_path)

    combine = {}
    for output in outputs1 :
        combine[output['image_path']] = [output['similarity']]
    
    for output in outputs2 :
        if combine.get(output['image_path']) == None :
            combine[output['image_path']] = [output['similarity']]
        else :
            combine[output['image_path']].append(output['similarity'])

    combine_two = {}
    for path, value_list in combine.items() :
        if len(value_list) == 2 :
            combine_two[path] = value_list[0] * value_list[1]

    combine_list = sorted(combine_two.items(), key=lambda x:x[1], reverse=True)
    results = []
    for comb_path, comb_sim in combine_list[:count3] :
        info = {}

        fn1, fn2 = comb_path.split('/')[-2:]
        num, _ = os.path.splitext(fn2)

        info['image_path'] = os.path.join(fn1, fn2)
        info['episode'] = int(fn1)
        info['num'] = int(num)
        info['similarity'] = comb_sim

        results.append(info)

    return text_en, Image.open(img_path).convert('RGB'), results