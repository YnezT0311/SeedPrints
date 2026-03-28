import collections
from tqdm import tqdm
import concurrent.futures
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, LlamaTokenizer
import os

def process_chunk(tokenizer,chunk):
    word_freq = collections.Counter()
    for item in chunk:
        # Count word frequency
        tokens = tokenizer.tokenize(item)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)        
        # Update word frequency count
        word_freq.update(token_ids)
    return word_freq

def process(data, num_threads,tokenizer):
    result = collections.Counter()
    chunk_size = len(data) // num_threads
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_chunk,tokenizer,chunk) for chunk in chunks]
        for future in tqdm(concurrent.futures.as_completed(futures)):
            result += future.result()

    return result

def filter_nonenglish_text(example):
    return all((ord(char) < 592 or (ord(char) in range(1024,1279)))  for char in example)#remain Latin and Cyrillic

def load_tokenizer(path):
    for tokenizer_class in [LlamaTokenizer, AutoTokenizer]:
        try:
            return tokenizer_class.from_pretrained(
                path,
                trust_remote_code=True,
                unk_token="<unk>",
                bos_token="<s>",
                eos_token="</s>"
            )
        except:
            try:
                return tokenizer_class.from_pretrained(path, trust_remote_code=True)
            except:
                continue
    raise ValueError(f"Failed to load tokenizer from {path}")

def sort_tokens_frequency(path,savepath,datanum=400000,num_process=40):
    tokenizer = AutoTokenizer.from_pretrained(path)
    # dataset = load_dataset("Skylion007/openwebtext", cache_dir="/disk2/haonan/tongyao/datasets", trust_remote_code=True)
    dataset = load_from_disk("/disk2/haonan/tongyao/datasets/openwebtext-2048-2B")
    word_freq = collections.Counter()
    for i in range(len(dataset["train"]) // datanum):
        data = dataset["train"][i * datanum: (i + 1) * datanum]["input_ids"]
        # turn input ids to text
        data = [tokenizer.decode(item, skip_special_tokens=True) for item in data]
        data = list(filter(filter_nonenglish_text, data))
        word_freq += process(data, num_process, tokenizer)
    remainder = len(dataset["train"]) % datanum
    data = dataset["train"][-remainder:]["input_ids"]
    data = [tokenizer.decode(item, skip_special_tokens=True) for item in data]
    data = list(filter(filter_nonenglish_text, data))
    word_freq += process(data, num_process, tokenizer)
    least_first = sorted(word_freq.items(), key=lambda item: (item[1], item[0]))

    model_tag = path.split("/")[-1]
    skip_tokens = ['<unk>', '<s>', '</s>']
    picked = OrderedDict()
    count = 0
    for tok_str, freq in least_first:
        if tok_str in skip_tokens:
            continue
        picked[tok_str] = int(freq)
        count += 1
        if count in [1000,2048,4096]:
            K = count
            os.makedirs(savepath, exist_ok=True)
            out_path = os.path.join(savepath, f"{model_tag}_least_{K}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(picked, f, ensure_ascii=False, indent=2)
        
        if count > 4096:
            break

model_path="/disk2/haonan/tongyao/model/init-llama-seed-1000"  # Example model path, adjust as needed
output_path="/disk2/haonan/tongyao/proj_2025_fingerprint/baselines/HuRef-main/sorted_tokens"
sort_tokens_frequency(model_path,output_path)