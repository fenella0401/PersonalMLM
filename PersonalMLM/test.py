from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
import torch
from peft import PeftModel
import csv
import json
import time
import random
import os
import re
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from conversations import conv_templates, get_default_conv_template, SeparatorStyle
import pickle as pkl
import jieba
from statistics import mean
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge_chinese import Rouge
import numpy as np
from chat import firefly

modelname = 'yi'
file_path = 'yi_1_5_5_8_16'
if_complex = False
print('\n---------------------------------')
print('model name:', modelname)

if modelname == 'ChatGLM3-6B':
    model_name_or_path = '../chatglm3-6b'
    #model = AutoModel.from_pretrained("/data/fanzhaoxin/fei/llm/chatglm3-6b", trust_remote_code=True, device='cuda')
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = model.eval()
elif modelname == 'loss_test_cl':
    model_name_or_path = '../chatglm3-6b'
    adapter_name_or_path = '../Firefly/output/loss_test_cl'
    model = AutoModelForCausalLM.from_pretrained(adapter_name_or_path, trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map='auto')
    # 加载adapter
    if adapter_name_or_path is not None:
        model = PeftModel.from_pretrained(model, adapter_name_or_path)
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = model.eval()
elif modelname == 'phi3' or modelname == 'chatglm3' or modelname == 'yi':
    model_name_or_path = '../Yi-6B-Chat'
    adapter_name_or_path = '../Firefly/output/yi-6b-sft-qlora-1w_5k_5k_8_16'
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map='auto')
    # 加载adapter
    if adapter_name_or_path is not None:
        model = PeftModel.from_pretrained(model, adapter_name_or_path)
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_name_or_path, trust_remote_code=True)
    model = model.eval()
elif modelname == 'MING':
    model_name = "../MING-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True)
    context_len = 2048
    model.config.use_cache = True
    model.eval()
elif modelname == 'PULSE':
    model_name = "../PULSE-7bv5"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True).eval()
elif modelname == 'DISC-MedLLM':
    model_name = "../DISC-MedLLM-model"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True)
    model = model.quantize(8).cuda()
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model = model.eval()
elif modelname == 'BianQue':
    model_name = "../BianQue/pretrained_models/bianque1"
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)
else:
    import openai
    openai.api_key = ''

@torch.inference_mode()
def generate_stream(model, tokenizer, params, beam_size,
                    context_len=4096, stream_interval=2):
    prompt = params["prompt"]
    temperature = float(params.get("temperature", 1.2))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    if stop_str == tokenizer.eos_token:
        stop_str = None

    input_ids = tokenizer(prompt).input_ids

    max_src_len = context_len - max_new_tokens - 8
    input_ids = torch.tensor(input_ids[-max_src_len:]).unsqueeze(0).cuda()

    outputs = model.generate(
        inputs=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        num_beams=beam_size,
        temperature=temperature,
    )
    outputs = outputs[0][len(input_ids[0]):]
    output = tokenizer.decode(outputs, skip_special_tokens=True)

    return output

def ming(history):
    '''
    history: [{'role':'user', 'content':'xxx'}, {'role':'assistant', 'content':'xxx'}]
    '''
    conv = conv_templates["bloom"].copy()
    
    for each in history:
        if each['role'] == 'user':
            conv.append_message("USER", each['content'])
        else:
            conv.append_message("ASSISTANT", each['content'])
    
    conv.append_message(conv.roles[1], None)

    generate_stream_func = generate_stream
    prompt = conv.get_prompt()

    params = {
        "prompt": prompt,
        "temperature": 1.2,
        "max_new_tokens": 512,
        "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
    }

    context_len = len(prompt)  + params['max_new_tokens'] + 8
    output_stream = generate_stream_func(model, tokenizer, params, 3, context_len=context_len)
    #print(output_stream)

    return output_stream.strip()

def pulse(history):
    '''
    history: [{'role':'user', 'content':'xxx'}, {'role':'assistant', 'content':'xxx'}]
    '''

    first_instruction = "Instructions: You are Helper, a large language model full of intelligence. Respond conversationally."

    model_type_prompt_map = {
        '医学知识QA': "若我是位患者，你是位资深医生，能够协助解答我的问题和疑虑，请为我提供回复。\n",
        '在线问诊': "假设你是一位经验丰富并且非常谨慎的的医生，会通过和患者进行多次的问答来明确自己的猜测，并且每次只能提一个问题，最终只会推荐相应的检验、检查、就诊科室以及疑似的诊断，请回复患者的提问：\n",
        'Base': "", 
    }

    input_max_len = 1536
    model_type = '在线问诊'
    gen_max_length = 512
    top_k = 6
    top_p = 0.1
    temperature = 0.7

    history[0]['content'] = model_type_prompt_map[model_type] + history[0]['content']
    
    input_ids = tokenizer(
        first_instruction,
        add_special_tokens=False
    ).input_ids + [tokenizer.convert_tokens_to_ids("</s>")]

    for each in history:
        if each['role'] == 'user':
            input_ids += tokenizer("User: " + each['content']).input_ids
        else:
            input_ids += tokenizer("Helper: " + each['content']).input_ids
        input_ids += [tokenizer.convert_tokens_to_ids("</s>")]

    model_inputs = tokenizer.pad(
        {"input_ids": [input_ids + tokenizer("Helper: ").input_ids[:1]]}, 
        return_tensors="pt",
    )

    inputs = model_inputs.input_ids[:,-input_max_len:]
    attention_mask = model_inputs.attention_mask[:,-input_max_len:]

    max_length = inputs.shape[1] + gen_max_length
    min_length = inputs.shape[1] + 1 # add eos

    outputs = model.generate(
        inputs=inputs.cuda(),
        attention_mask=attention_mask.cuda(),
        max_length=max_length,
        min_length=min_length,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        num_return_sequences=1,
        eos_token_id=tokenizer.convert_tokens_to_ids("</s>"),
    )

    outputs_token = outputs[0].tolist()

    new_start_pos = inputs.shape[1]
    new_end_pos = new_start_pos

    while new_end_pos < len(outputs_token) and outputs_token[new_end_pos] != tokenizer.convert_tokens_to_ids("</s>"):
        new_end_pos += 1

    outputs_token = list(tokenizer("Helper: ").input_ids[:1]) + outputs_token[new_start_pos:new_end_pos]

    input_ids += outputs_token
    input_ids += [tokenizer.convert_tokens_to_ids("</s>")] 
    
    otext = tokenizer.decode(
        outputs_token, 
        skip_special_tokens=False
    )

    otext = otext.strip()
    if otext[:3] == "<s>":
        otext = otext[3:]
    otext = otext.strip()
    otext = otext.replace("Helper: ", "")

    return otext

def preprocess(text):
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text

def postprocess(text):
    return text.replace("\\n", "\n").replace("\\t", "\t")

def bianque(messages, sample=True, top_p=1, temperature=0.7):
    '''sample：是否抽样。生成任务，可以设置为True;
    top_p：0-1之间，生成的内容越多样
    max_new_tokens=512 lost...'''

    bot_history = [each['content'] for each in messages[:-1] if each['role'] == 'assistant']
    user_history = [each['content'] for each in messages if each['role'] == 'user']
    
    if len(bot_history)>0:
        context = "\n".join([f"病人：{user_history[i]}\n医生：{bot_history[i]}" for i in range(len(bot_history))])
        input_text = context + "\n病人：" + user_history[-1] + "\n医生："
    else:
        input_text = "病人：" + user_history[0] + "\n医生："
    
    input_text = preprocess(input_text)
    encoding = tokenizer(text=input_text, truncation=True, padding=True, max_length=768, return_tensors="pt").to(device) 
    if not sample:
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=512, num_beams=1, length_penalty=0.6)
    else:
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=512, do_sample=True, top_p=top_p, temperature=temperature, no_repeat_ngram_size=3)
    out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
    return postprocess(out_text[0])

def chat(modelname, messages, tempreture=0.7):
    if modelname == 'GLM-4':
        response = client.chat.completions.create(model="glm-4",messages=messages,temperature=tempreture)
        response = response.choices[0].message.content
    
    if modelname == 'ChatGLM3-6B' or modelname == 'loss_test_tl':
        data_user = messages[-1]['content']
        data_history = messages[:-1]
        response, _ = model.chat(tokenizer, data_user, data_history, temperature=tempreture, max_length=512)
    
    if modelname == 'phi3'  or modelname == 'chatglm3' or modelname == 'yi':
        messages_new = []
        for each in messages:
            messages_new.append({'role': each['role'], 'message': each['content']})
        response = firefly(model, tokenizer, modelname, messages_new[:-1], messages_new[-1]['message'])

    if modelname == 'MING':
        conv = conv_templates["bloom"].copy()
    
        for each in messages:
            if each['role'] == 'user':
                conv.append_message("USER", each['content'])
            else:
                conv.append_message("ASSISTANT", each['content'])
        
        conv.append_message(conv.roles[1], None)

        generate_stream_func = generate_stream
        prompt = conv.get_prompt()

        params = {
            "prompt": prompt,
            "temperature": 1.2,
            "max_new_tokens": 512,
            "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
        }

        context_len = len(prompt)  + params['max_new_tokens'] + 8
        output_stream = generate_stream_func(model, tokenizer, params, 3, context_len=context_len)
        response = output_stream.strip()

    if modelname == 'PULSE':
        response = pulse(messages)
    
    if modelname == 'DISC-MedLLM':
        response = model.chat(tokenizer, messages)

    if modelname == 'BianQue':
        response = bianque(messages)
    
    if modelname == 'ChatGPT':
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=messages)
        response = response.choices[0].message['content']
    
    if modelname == 'GPT-4':
        response = client.chat.completions.create(model="gpt-4",messages=messages,temperature=tempreture)
        response = response.choices[0].message.content

    return response

def get_bleu_score(result, name, refs, hyps, refs_question, hyps_question):
    if if_complex == False:
        cc = SmoothingFunction()
        bleu1_1 = []
        bleu2_1 = []
        bleu3_1 = []
        bleu4_1 = []
        bleu1_10 = []
        bleu2_10 = []
        bleu3_10 = []
        bleu4_10 = []

        for i in range(len(refs)):
            hyp = ' '.join(jieba.cut(hyps[i]))
            hyp = hyp.split()

            ref1 = [' '.join(jieba.cut(each)) for each in refs[i][:1]]
            ref1 = [each.split() for each in ref1]
            ref10 = [' '.join(jieba.cut(each)) for each in refs[i]]
            ref10 = [each.split() for each in ref10]
            
            # ref可以是多个candidate
            bleu1_1.append(sentence_bleu(ref1, hyp, weights=(1, 0, 0, 0), smoothing_function=cc.method1))
            bleu2_1.append(sentence_bleu(ref1, hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=cc.method1))
            bleu3_1.append(sentence_bleu(ref1, hyp, weights=(0.33, 0.33, 0.33, 0), smoothing_function=cc.method1))
            bleu4_1.append(sentence_bleu(ref1, hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=cc.method1))

            bleu1_10.append(sentence_bleu(ref10, hyp, weights=(1, 0, 0, 0), smoothing_function=cc.method1))
            bleu2_10.append(sentence_bleu(ref10, hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=cc.method1))
            bleu3_10.append(sentence_bleu(ref10, hyp, weights=(0.33, 0.33, 0.33, 0), smoothing_function=cc.method1))
            bleu4_10.append(sentence_bleu(ref10, hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=cc.method1))
            
        # 求list的平均
        bleu1_avg_1 = mean(bleu1_1)
        bleu2_avg_1 = mean(bleu2_1)
        bleu3_avg_1 = mean(bleu3_1)
        bleu4_avg_1 = mean(bleu4_1)
        bleu1_avg_10 = mean(bleu1_10)
        bleu2_avg_10 = mean(bleu2_10)
        bleu3_avg_10 = mean(bleu3_10)
        bleu4_avg_10 = mean(bleu4_10)

        result[name]['all'].extend([bleu1_avg_1, bleu2_avg_1, bleu3_avg_1, bleu4_avg_1, bleu1_avg_10, bleu2_avg_10, bleu3_avg_10, bleu4_avg_10])

        return result

def get_rouge_score(result, name, refs, hyps, refs_question, hyps_question):
    if if_complex == False:
        rouge = Rouge()
        
        rouge1_1 = []
        rouge2_1 = []
        rougel_1 = []
        rouge1_10 = []
        rouge2_10 = []
        rougel_10 = []

        for i in range(len(refs)):
            ref = ' '.join(jieba.cut(refs[i][0]))
            hyp = ' '.join(jieba.cut(hyps[i]))
            if ref.replace(' ', '') == '':
                ref = '.'
            if hyp.replace(' ', '') == '':
                hyp = '。'
            scores = rouge.get_scores(hyps=hyp, refs=ref)
            rouge1_1.append(scores[0]['rouge-1']['f'])
            rouge2_1.append(scores[0]['rouge-2']['f'])
            rougel_1.append(scores[0]['rouge-l']['f'])

        rouge1_avg_1 = mean(rouge1_1)
        rouge2_avg_1 = mean(rouge2_1)
        rougel_avg_1 = mean(rougel_1)

        rouge1_10 = []
        rouge2_10 = []
        rougel_10 = []
        for i in range(len(refs)):
            for j, tmp in enumerate(refs[i]):
                if tmp.replace(' ', '') == '':
                    refs[i][j] = '.'
            if hyps[i].replace(' ', '') == '':
                hyps[i] = '。'
            ref = [' '.join(jieba.cut(each)) for each in refs[i]]
            hyp = [' '.join(jieba.cut(hyps[i]))] * len(ref)
            scores = rouge.get_scores(hyps=hyp, refs=ref)
            rouge1_tmp = []
            rouge2_tmp = []
            rougel_tmp = []
            for each in scores:
                rouge1_tmp.append(each['rouge-1']['f'])
                rouge2_tmp.append(each['rouge-2']['f'])
                rougel_tmp.append(each['rouge-l']['f'])
            rouge1_10.append(max(rouge1_tmp))
            rouge2_10.append(max(rouge2_tmp))
            rougel_10.append(max(rougel_tmp))

        rouge1_avg_10 = mean(rouge1_10)
        rouge2_avg_10 = mean(rouge2_10)
        rougel_avg_10 = mean(rougel_10)

        result[name]['all'].extend([rouge1_avg_1, rouge2_avg_1, rougel_avg_1, rouge1_avg_10, rouge2_avg_10, rougel_avg_10])

        return result

def check_is_question(text):
    '''
    检查文本是否为问句
    '''
    question_list = ["？", "?", "吗", "呢", "么", "什么", "有没有", "多少", "几次", "怎么样", "几岁", "多久", "多长", "请问"]
    for token in question_list:
        if token in text:
            return True
    return False

def pqa(result, data, refs, hyps):
    if if_complex == False:
        qtp = 0
        qtpb = 0
        qtbpb = 0
        for i in range(len(refs)):
            if refs[i] == 1 and hyps[i] == 1:
                qtp += 1
            if refs[i] == 1 and hyps[i] != 1:
                qtpb += 1
            if refs[i] != 1 and hyps[i] != 1:
                qtbpb += 1
        pq = qtp / (qtp + qtpb)
        rq = qtp / (qtp + qtbpb)
        if pq + rq == 0:
            pqa_v = 0
        else:
            pqa_v = 2*pq*rq/(pq+rq)
        result[data]['all'].append(pqa_v)
        return result

def evaluate(refs_chunyu, refs_icms, hyps_chunyu, hyps_icms, chunyu_avg_time):
    with open('./test_dataset/refs_chunyu_question.pkl', 'rb') as f:
        refs_chunyu_question = pkl.load(f)
    with open('./test_dataset/refs_icms_question.pkl', 'rb') as f:
        refs_icms_question = pkl.load(f)

    refs_chunyu_new = []
    hyps_chunyu_new = []
    qp_index_chunyu = [0,0,0,0]
    refs_chunyu_question_new = []
    hyps_chunyu_question = []
    for i, each in enumerate(hyps_chunyu):
        each = json.loads(each)
        index = each['index']
        content = each['response']
        refs_chunyu_new.append(refs_chunyu[int(index)])
        refs_chunyu_question_new.append(refs_chunyu_question[int(index)])
        if (int(index)) < 200:
            qp_index_chunyu[0] = i
        elif (int(index)) < (200*2):
            qp_index_chunyu[1] = i
        elif (int(index)) < (200*3):
            qp_index_chunyu[2] = i
        elif (int(index)) < (200*4):
            qp_index_chunyu[3] = i
        else:
            pass
        hyps_chunyu_new.append(content)
        # 问句=1，非问句=0
        if check_is_question(content) == True:
            hyps_chunyu_question.append(1)
        else:
            hyps_chunyu_question.append(0)
    print('qp_index_chunyu', qp_index_chunyu)
    
    refs_icms_new = []
    hyps_icms_new = []
    qp_index_icms = [0,0,0,0]
    refs_icms_question_new = []
    hyps_icms_question = []
    for i, each in enumerate(hyps_icms):
        each = json.loads(each)
        index = each['index']
        content = each['response']
        refs_icms_new.append(refs_icms[int(index)])
        refs_icms_question_new.append(refs_icms_question[int(index)])
        if (int(index)) < 162:
            qp_index_icms[0] = i
        elif (int(index)) < (162*2):
            qp_index_icms[1] = i
        elif (int(index)) < (162*3):
            qp_index_icms[2] = i
        elif (int(index)) < (162*4):
            qp_index_icms[3] = i
        else:
            pass
        hyps_icms_new.append(content)
        if check_is_question(content) == True:
            hyps_icms_question.append(1)
        else:
            hyps_icms_question.append(0)
    print('qp_index_icms', qp_index_icms)

    question_num_chunyu = np.sum(np.array(refs_chunyu_question_new) == 1)
    question_num_icms = np.sum(np.array(refs_icms_question_new) == 1)

    result = {'model': file_path, 'avg_time': chunyu_avg_time, 'chunyu': {'num': len(refs_chunyu_new), 'question_num': int(question_num_chunyu), 'qp_index': qp_index_chunyu, 'all': [], '1': [], '25': [], '50': [], '75': [], '100': [], 'question': [], 'answer': []}, 
              'icms': {'num': len(refs_icms_new), 'question_num': int(question_num_icms), 'qp_index': qp_index_icms, 'all': [], '1': [], '25': [], '50': [], '75': [], '100': [], 'question': [], 'answer': []}}

    result = get_bleu_score(result, 'chunyu', refs_chunyu_new, hyps_chunyu_new, refs_chunyu_question_new, refs_icms_question_new)
    result = get_bleu_score(result, 'icms', refs_icms_new, hyps_icms_new, refs_chunyu_question_new, refs_icms_question_new)
    result = get_rouge_score(result, 'chunyu', refs_chunyu_new, hyps_chunyu_new, refs_chunyu_question_new, refs_icms_question_new)
    result = get_rouge_score(result, 'icms', refs_icms_new, hyps_icms_new, refs_chunyu_question_new, refs_icms_question_new)
    result = pqa(result, 'chunyu', refs_chunyu_question_new, hyps_chunyu_question)
    result = pqa(result, 'icms', refs_icms_question_new, hyps_icms_question)

    with open('./test_dataset/output/result.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    begin_time = time.time()

    with open('./test_dataset/input_chunyu.pkl', 'rb') as f:
        data = pkl.load(f)
    #data = data[:10]
    print('---------------------------------')
    print('chunyu Data loaded.', len(data))
    error_chunyu = []
    chunyu_begin_time = time.time()
    for i, each in enumerate(data):
        if i % 200 == 0:
            print('data num processed:', i)
        try:
            response = chat(modelname, each)
            if type(response) == dict:
                tmp = ''
                for each in response:
                    tmp += response[each]
                response = tmp
            response = response.replace('\n\n', '。').replace('\n', '。').replace('？。', '？').replace('！。', '！').replace('。。', '。').replace('。。。', '。').replace('，。', '，')
            with open('./test_dataset/output/output_chunyu_' + file_path + '.jsonl', 'a', encoding='utf-8') as f:
                #f.write(str(i) + ';;;;' + response + '\n')
                f.write(json.dumps({'index': i, 'response': response}, ensure_ascii=False) + '\n')
        except Exception as e:
            print(e)
            error_chunyu.append(i)
            with open('./test_dataset/output/output_chunyu_' + file_path + '_error.txt', 'a') as f:
                f.write(str(i) + '\n')
            continue
    print('chunyu Done.')
    chunyu_end_time = time.time()
    chunyu_avg_time = (chunyu_end_time - chunyu_begin_time) / len(data)
    print('avg_time:', chunyu_avg_time)
    print('error:', len(error_chunyu))
    print('---------------------------------')

    with open('./test_dataset/input_icms.pkl', 'rb') as f:
        data = pkl.load(f)
    #data = data[:10]
    print('icms Data loaded.', len(data))
    error_icms = []
    for i, each in enumerate(data):
        if i % 200 == 0:
            print('data num processed:', i)
        try:
            response = chat(modelname, each)
            if type(response) == dict:
                tmp = ''
                for each in response:
                    tmp += response[each]
                response = tmp
            response = response.replace('\n\n', '。').replace('\n', '。').replace('？。', '？').replace('！。', '！').replace('。。', '。').replace('。。。', '。').replace('，。', '，').strip()
            with open('./test_dataset/output/output_icms_' + file_path + '.jsonl', 'a', encoding='utf-8') as f:
                #f.write(str(i) + ';;;;' + response + '\n')
                f.write(json.dumps({'index': i, 'response': response}, ensure_ascii=False) + '\n')
        except Exception as e:
            print(e)
            error_icms.append(i)
            with open('./test_dataset/output/output_icms_' + file_path + '_error.txt', 'a') as f:
                f.write(str(i) + '\n')
            continue
    print('icms Done.')
    print('error:', len(error_icms))
    print('---------------------------------')

    with open('./test_dataset/refs_chunyu.pkl', 'rb') as f:
        refs_chunyu = pkl.load(f)
    with open('./test_dataset/refs_icms.pkl', 'rb') as f:
        refs_icms = pkl.load(f)
    with open('./test_dataset/output/output_chunyu_' + file_path + '.jsonl', 'r', encoding='utf-8') as f:
        hyps_chunyu = f.readlines()
    with open('./test_dataset/output/output_icms_' + file_path + '.jsonl', 'r', encoding='utf-8') as f:
        hyps_icms = f.readlines()

    evaluate(refs_chunyu, refs_icms, hyps_chunyu, hyps_icms, chunyu_avg_time)

    end_time = time.time()
    process_time = end_time - begin_time
    print('total time:', process_time, process_time/3600)
    print('Done.')
