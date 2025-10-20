# import torch
# # import llama_index.llms
# print("CUDA available:", torch.cuda.is_available())
# print(torch.__version__)
#
# print("cuDNN version:", torch.backends.cudnn.version())
#
# cudnn_available = torch.backends.cudnn.is_available()
# print("cuDNN available:", cudnn_available)
#
#
# print(torch.version.cuda)  # Should print the correct CUDA version
# print(torch.cuda.is_available())  # Should return True if CUDA is available
#
#
#
# # print(dir(llama_index.llms))
#
# # results:
# # CUDA available: True
# # 2.4.1+cu118
# # cuDNN version: 90100
# # cuDNN available: True
#
# from googleapiclient.discovery import build
#
#
# def google_search(query, api_key, cse_id, num_results=3):
#     try:
#         # Initialize the Google Custom Search API client
#         service = build("customsearch", "v1", developerKey=api_key)
#         url = f"https://customsearch.googleapis.com/customsearch/v1?q={query}&cx={cse_id}&key={api_key}"
#         print(url)
#
#         # Perform the search
#         res = service.cse().list(q=query, cx=cse_id, num=num_results).execute()
#
#         # Return the results
#         return res.get("items", [])
#     except Exception as e:
#         print(f"Error: {e}")
#         return None
#
#
# # Example usage
# api_key = "AIzaSyCWBlzpNEEgOkb3GsYdB3SDIOvUmr_h1ig"  # Replace with your actual API key
# cse_id = "5086429ea12f641aa"  # Replace with your actual CSE ID
# search_results = google_search("Python programming", api_key, cse_id, num_results=3)
# print(search_results)

#
# import pyaudio
#
# audio = pyaudio.PyAudio()
# for i in range(audio.get_device_count()):
#     info = audio.get_device_info_by_index(i)
#     print(f"{i}: {info['name']}")

from sentence_transformers import SentenceTransformer

# Run this once while online
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
