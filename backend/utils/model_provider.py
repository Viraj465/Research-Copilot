# MODELS = {
#     "Free Providers": [
#             {
#             "value": "llama-3.1-8b-instant",
#             "label": "Llama 3.1 8B (Instant)",
#             "name": "llama-3.1-8b-instant",
#             "provider": ["groq"],
#             "companyName": "Groq",
#         },
#         {
#             "value": "llama-3.3-70b-versatile",
#             "label": "Llama 3.3 70B (Versatile)",
#             "name": "llama-3.3-70b-versatile",
#             "provider": ["groq"],
#             "companyName": "Groq",
#         },
#         {
#             "value": "openai/gpt-oss-120b",
#             "label": "GPT-OSS 120B",
#             "name": "gpt-oss-120b",
#             "provider": ["groq"],
#             "companyName": "Groq",
#         },
#         {
#             "value": "openai/gpt-oss-20b",
#             "label": "GPT-OSS 20B",
#             "name": "gpt-oss-20b",
#             "provider": ["groq"],
#             "companyName": "Groq",
#         },
#         {
#             "value": "qwen/qwen3-32b",
#             "label": "Qwen3 32B",
#             "name": "qwen3-32b",
#             "provider": ["groq"],
#             "companyName": "Groq",
#         }
#     ],

#     "Paid Providers": [
#         {
#         "value": "openai/gpt-4.1",
#         "label": "ChatGPT 4.1",
#         "name": "gpt-4.1",
#         "providers": ["openai"],
#         "companyName": "OpenAI",
#         },
#         {
#             "value": "openai/gpt-4.1-mini",
#             "label": "ChatGPT 4.1 Mini",
#             "name": "gpt-4.1-mini",
#             "providers": ["openai"],
#             "companyName": "OpenAI",
#         },
#         {
#             "value": "openai/gpt-4.1-nano",
#             "label": "ChatGPT 4.1 Nano",
#             "name": "gpt-4.1-nano",
#             "providers": ["openai"],
#             "companyName": "OpenAI",
#         },
#         {
#             "value": "openai/gpt-5",
#             "label": "ChatGPT 5",
#             "name": "gpt-5",
#             "providers": ["openai"],
#             "companyName": "OpenAI",
#         },    
#     ],
# }

# def get_model_by_value(value: str):
#     for group in MODELS.values():
#         for model in group:
#             if model["value"] == value or model["label"] == value:
#                 return model
#     return None
