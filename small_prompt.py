import os, time, faulthandler
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
faulthandler.dump_traceback_later(60, exit=True)   # after 60 s: print where every thread is stuck, then exit

MODE = "default"        # run once as "default", once as "opts"
MODEL = "gemini-3.5-flash"

# if MODE == "default":
#     print('default')
#     client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# else:
print('NOT default')
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(
        timeout=60_000,
        retry_options=types.HttpRetryOptions(attempts=1),
    ),
)

t = time.monotonic()
i = client.interactions.create(model=MODEL, input="Say hi.", store=False)
print(f"{time.monotonic() - t:.1f}s ->", i.output_text)