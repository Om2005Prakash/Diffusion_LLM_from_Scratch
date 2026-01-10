from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

def get_tokenizer(model_name="answerdotai/ModernBERT-base",
                  bos_token="<BOS>",
                  eos_token="<EOS>",
                  start_token="<START_ID>",
                  end_token="<END_ID>",
                  eot_token="<EOT_ID>"):
    """
    Load tokenizer, add special tokens, and set up chat template.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Define our special tokens, missing in ModernBERT
    special_tokens = {
        "bos_token": bos_token,
        "eos_token": eos_token,
        "additional_special_tokens": [start_token, end_token, eot_token],
    }

    # Add them to tokenizer
    tokenizer.add_special_tokens(special_tokens)

    # Set EOS token as PAD token and CLS to BOS Token
    tokenizer.pad_token = eos_token
    tokenizer.cls_token = bos_token

    # Templete processing for tokenizing pretrain data
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos_token} $A {eos_token}",
        special_tokens=[
            (bos_token, tokenizer.bos_token_id),
            (eos_token, tokenizer.eos_token_id)
        ]
    )

    # Chat templete for SFT
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ bos_token if loop.first else '' }}"
        f"{{{{ '{start_token}' + message['role'] + '{end_token}' }}}}\n"
        "{{ message['content'] }}"
        f"{{{{ '{eot_token}' if message['role'] == 'user' else eos_token }}}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        f"{{{{ '{start_token}' + 'assistant' + '{end_token}' }}}}"
        "{% endif %}"
    )

    return tokenizer

if __name__ == "__main__":
    tok = get_tokenizer()

    print("--------------Pre Train Test-----------------")
    text = "Hello World"
    ids = tok(text, padding=True, return_tensors="pt")["input_ids"][0]
    decoded = tok.decode(ids, skip_special=False)
    print("Text:", text)
    print("ids:", ids)
    print("decoded:", decoded)

    print("\n--------------SFT Test-----------------")
    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi! How can I help you?"},
    ]
    encoded = tok.apply_chat_template(messages, tokenize=True, add_special_tokens=True)
    decoded = tok.decode(encoded, skip_special=False)
    print("Messages:", messages)
    print("Encoded ids:", encoded)
    print("Decoded:", decoded)

    print("\n--------------Generation Prompt Test-----------------")
    messages = [
        {"role": "user", "content": "Hello!"},
    ]

    encoded = tok.apply_chat_template(messages, 
                                      tokenize=True,
                                      add_special_tokens=True, 
                                      add_generation_prompt=True
                                      )
    
    decoded = tok.decode(encoded, skip_special=False)
    print("Messages:", messages)
    print("Encoded ids:", encoded)
    print("Decoded:", decoded)