import openai
import time

def call_chatgpt_api_with_retry(llm_model_name,max_retries,message):
    retries = 0

    while retries < max_retries:
        try:
            # Call ChatGPT API
            chat_completion = openai.ChatCompletion.create(
                model=llm_model_name,
                messages=[{"role": "user", "content": message}]
            )
            response = chat_completion.choices[0].message.content
            retries += 1
            return response
        
        except Exception as e:
            print(f"Something wrong: {e}. Retrying in 2 minutes...")
            time.sleep(120)  # Wait 2 minutes
            retries += 1

    print("Max retries reached. Unable to get a response.")
    return None

def call_llm(llm_model_name,message):
    if 'gpt' in llm_model_name:
        response = call_chatgpt_api_with_retry(llm_model_name, 10, message)
    else:
        raise ValueError(f"Unsupported model: {llm_model_name}")
    return response
