from openai import OpenAI
client = OpenAI()

def process_prompt(input, path, task, data, prompt_type):
    prompt_path = f"{path}/prompts/{task}_{data}_{prompt_type}.txt"
    with open(prompt_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    messages = []
    system_content = ""

    for line in lines:
        if line.startswith("Text:"):
            if system_content:  # system_content가 비어있지 않으면 추가
                messages.append({"role": "system", "content": system_content.strip()})
                system_content = ""  # system 메시지를 추가한 후 초기화
            messages.append({"role": "user", "content": line.strip().replace('Text: ', '')})
        elif line.startswith("Sentiment Elements:"):
            messages.append({"role": "assistant", "content": line.strip().replace('Sentiment Elements: ', '')})
        else:
            system_content += line

    if system_content:
        messages.append({"role": "system", "content": system_content.strip()})

    messages.append({"role": "user", "content": input})
    return messages

def llm_chat(prompt, version, temperature, max_seq_length):
    completion = client.chat.completions.create(
        model=version,
        messages = prompt,
        temperature=temperature,
        max_tokens=max_seq_length,
    )
    result = ''
    for choice in completion.choices:
        result += choice.message.content
    return result

def merge_save(inputs, targets, file_name, num=1):
    targets_expanded = [tar for tar in targets for _ in range(num)]
    merged_data = [inp + "####" + tar for inp, tar in zip(inputs, targets_expanded)]
    with open(file_name, 'w', encoding='UTF-8') as file:
        for line in merged_data:
            file.write(line + '\n')

def fix_string(input_str):
    parts = input_str.split(", ")
    adjusted_parts = []
    
    for part in parts:
        quote_count = part.count("'")
        if quote_count >= 3:  
            first_quote_index = part.find("'")  
            last_quote_index = part.rfind("'")  
           
            part = part[:first_quote_index] + '"' + part[first_quote_index+1:last_quote_index] + '"' + part[last_quote_index+1:]
        
        adjusted_parts.append(part)
    
    adjusted_str = ", ".join(adjusted_parts)
    return adjusted_str
