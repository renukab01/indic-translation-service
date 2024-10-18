from fastapi import FastAPI, UploadFile, File, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from typing import List
import pandas as pd
from translation import translation_service
import time
import psutil
import threading
from dotenv import load_dotenv

load_dotenv()

class ContinuousMonitor:
    def __init__(self, interval=1):
        self.interval = interval
        self.cpu_usage = []
        self.memory_usage = []
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _monitor(self):
        while self.running:
            self.cpu_usage.append(psutil.cpu_percent(interval=self.interval))
            self.memory_usage.append(psutil.virtual_memory().percent)

    def get_metrics(self):
        return {
            "avg_cpu_usage": sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0,
            "max_cpu_usage": max(self.cpu_usage) if self.cpu_usage else 0,
            "avg_memory_usage": sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
            "max_memory_usage": max(self.memory_usage) if self.memory_usage else 0,
        }

class MetricsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, monitor):
        super().__init__(app)
        self.monitor = monitor

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        end_time = time.time()

        metrics = {
            "path": request.url.path,
            "method": request.method,
            "time_consumed": end_time - start_time,
            **self.monitor.get_metrics()
        }

        print(f"Request metrics: {metrics}")
        return response

monitor = ContinuousMonitor()
monitor.start()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(MetricsMiddleware, monitor=monitor)

@app.on_event("shutdown")
def shutdown_event():
    monitor.stop()

models = {"indic-indic": "ai4bharat/indictrans2-indic-indic-dist-320M",
          "en-indic": "ai4bharat/indictrans2-en-indic-dist-200M",
          "indic-en": "ai4bharat/indictrans2-indic-en-dist-200M"}

indic_languages = ["asm_Beng", "ben_Beng", "brx_Deva", "doi_Deva",
                   "gom_Deva", "guj_Gujr", "hin_Deva", "kan_Knda", "kas_Arab",
                   "kas_Deva", "mai_Deva", "mal_Mlym", "mar_Deva", "mni_Beng",
                   "mni_Mtei", "npi_Deva", "ory_Orya", "pan_Guru", "san_Deva",
                   "sat_Olck", "snd_Arab", "snd_Deva", "tam_Taml", "tel_Telu",
                   "urd_Arab"]

#     Assamese (asm_Beng)
#     Kashmiri (Arabic) (kas_Arab)
#     Punjabi (pan_Guru)
#     Bengali (ben_Beng)
#     Kashmiri (Devanagari) (kas_Deva)
#     Sanskrit (san_Deva)
#     Bodo (brx_Deva)
#     Maithili (mai_Deva)
#     Santali (sat_Olck)
#     Dogri (doi_Deva)
#     Malayalam (mal_Mlym)
#     Sindhi (Arabic) (snd_Arab)
#     English (eng_Latn)
#     Marathi (mar_Deva)
#     Sindhi (Devanagari) (snd_Deva)
#     Konkani (gom_Deva)
#     Manipuri (Bengali) (mni_Beng)
#     Tamil (tam_Taml)
#     Gujarati (guj_Gujr)
#     Manipuri (Meitei) (mni_Mtei)
#     Telugu (tel_Telu)
#     Hindi (hin_Deva)
#     Nepali (npi_Deva)
#     Urdu (urd_Arab)
#     Kannada (kan_Knda)
#     Odia (ory_Orya)

@app.post("/translate_csv/")
def translate(files: List[UploadFile] = File(...)):
    global uploaded_files
    results = []
    for file in files:
        df = pd.read_csv(file.file)
        src_lang = df.columns[0]
        input_sentences = df[src_lang].to_list()

        if src_lang == "eng_Latn":
            model_name = models["en-indic"]
            for tgt_lang in indic_languages:
                output = translation_service(model_name, input_sentences, src_lang, tgt_lang)
                df[tgt_lang]=output['translations']
        else:
            model_name = models["indic-indic"]
            for tgt_lang in indic_languages:
                if tgt_lang != src_lang:
                    output = translation_service(model_name, input_sentences, src_lang, tgt_lang)
                    df[tgt_lang]=output['translations']
            model_name = models["indic-en"]
            output = translation_service(model_name, input_sentences, src_lang, tgt_lang)
            df["eng_Latn"] = output['translations']

        
        output_filename = f"translated_{file.filename}"
        df.to_csv(output_filename, index=False)
        results.append(output_filename)

    return {"output_files": results}

@app.post("/translate_sent/")
def translate_sentences(input_sentences:List[str], src_lang:str, tgt_lang:str):
    if src_lang == "eng_Latn":
        model_name = models["en-indic"]
        output = translation_service(model_name, input_sentences, src_lang, tgt_lang)
    elif tgt_lang == "eng_Latn":
        model_name = models["indic-en"]
        output = translation_service(model_name, input_sentences, src_lang, tgt_lang)
    else:
        model_name = models["indic-indic"]
        output = translation_service(model_name, input_sentences, src_lang, tgt_lang)
    return output