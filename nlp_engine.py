import os
import re
import json
import joblib
import pandas as pd
import torch
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

try:
    from transformers import pipeline as hf_pipeline
except Exception:
    hf_pipeline = None

try:
    from IndicTransToolkit.processor import IndicProcessor
except Exception:
    IndicProcessor = None

DEFAULT_EXPLICIT_WORDS={"fuck","fucking","fucked","shit","bitch","bastard","asshole","dick","piss","motherfucker","damn","ਚੂਤ","ਚੁਤ","ਲੌੜਾ","ਲੰਡ","ਭੈਣਚੋਦ","ਮਾਦਰਚੋਦ","ਹਰਾਮੀ","ਕੁੱਤੀ","ਕੁੱਤਾ","ਗਾਲੀ"}
DEFAULT_REPLACEMENTS={"fuck":"f***","fucking":"f******","fucked":"f*****","shit":"s***","bitch":"b****","bastard":"b******","asshole":"a******","dick":"d***","piss":"p***","motherfucker":"m***********","damn":"d***","ਚੂਤ":"ਚ***","ਚੁਤ":"ਚ***","ਲੌੜਾ":"ਲ***","ਲੰਡ":"ਲ***","ਭੈਣਚੋਦ":"ਭੈ***","ਮਾਦਰਚੋਦ":"ਮਾ***","ਹਰਾਮੀ":"ਹ***","ਕੁੱਤੀ":"ਕ***","ਕੁੱਤਾ":"ਕ***","ਗਾਲੀ":"ਗ***"}

class NLPManager:
    def __init__(self):
        self.base_dir=os.path.dirname(os.path.abspath(__file__))
        self.models_dir=os.path.join(self.base_dir,"models"); os.makedirs(self.models_dir,exist_ok=True)
        self.translation_model_name="ai4bharat/indictrans2-indic-en-dist-200M"
        self.translation_source_language="pan_Guru"; self.translation_target_language="eng_Latn"
        self.translation_tokenizer=None; self.translation_model=None; self.indic_processor=None
        self.sentiment=None; self.emotion=None
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ready=False; self.translation_available=False; self.sentiment_available=False; self.emotion_available=False
        self.translation_cache={}; self.cache_limit=500
        self.custom_sentiment_path=os.path.join(self.models_dir,"custom_sentiment.joblib")
        self.custom_emotion_path=os.path.join(self.models_dir,"custom_emotion.joblib")
        self.custom_sentiment=self._load_joblib(self.custom_sentiment_path); self.custom_emotion=self._load_joblib(self.custom_emotion_path)
        self.explicit_words_path=os.path.join(self.base_dir,"profanity_words.txt"); self.explicit_words=self._load_explicit_words()
        self.replacements=dict(DEFAULT_REPLACEMENTS)
        self.fallback_sentiment_words={"love":0.8,"happy":0.7,"joy":0.7,"good":0.5,"beautiful":0.6,"sad":-0.7,"hate":-0.8,"angry":-0.8,"pain":-0.7,"lonely":-0.7,"cry":-0.7}
        self.fallback_emotions={"love":[("love",.9)],"happy":[("joy",.8)],"joy":[("joy",.8)],"sad":[("sadness",.85)],"cry":[("sadness",.8)],"angry":[("anger",.85)],"fear":[("fear",.85)],"beautiful":[("joy",.5)],"heart":[("love",.5)]}

    def _load_joblib(self,path):
        if not os.path.exists(path): return None
        try:return joblib.load(path)
        except Exception:return None

    def _load_explicit_words(self):
        words=set(DEFAULT_EXPLICIT_WORDS)
        if os.path.exists(self.explicit_words_path):
            try:
                with open(self.explicit_words_path,"r",encoding="utf-8") as f:
                    for line in f:
                        line=line.strip()
                        if line and not line.startswith("#"):words.add(line.lower())
            except Exception:pass
        return words

    def load(self,progress=None):
        def p(msg,i,total=4):
            if progress: progress(msg,i,total)
        errors=[]
        p("Loading Punjabi translation model...",1)
        try:
            if IndicProcessor is None: raise RuntimeError("IndicTransToolkit is not installed. Run: py -m pip install IndicTransToolkit")
            self.indic_processor=IndicProcessor(inference=True)
            self.translation_tokenizer=AutoTokenizer.from_pretrained(self.translation_model_name,trust_remote_code=True)
            self.translation_model=AutoModelForSeq2SeqLM.from_pretrained(self.translation_model_name,trust_remote_code=True).to(self.device)
            self.translation_model.eval(); self.translation_available=True
        except Exception as e:
            self.translation_available=False; errors.append(f"Translation: {e}")
        p("Checking translation quality...",2)
        if self.translation_available:
            try:
                test=self._translate_batch(["ਮੈਂ ਤੈਨੂੰ ਪਿਆਰ ਕਰਦਾ ਹਾਂ"],run_quality=False)[0]
                low=test.lower()
                if not test or not re.search(r"[a-zA-Z]",test) or not any(w in low for w in ("love","you")):
                    raise RuntimeError(f"Local translation self-test failed: {test!r}")
            except Exception as e:
                self.translation_available=False; errors.append(str(e))
        p("Loading sentiment model...",3)
        if self.custom_sentiment is None and hf_pipeline:
            try:
                self.sentiment=hf_pipeline("sentiment-analysis",model="cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",device=0 if self.device.type=="cuda" else -1)
                self.sentiment_available=True
            except Exception as e: errors.append(f"Sentiment model unavailable: {e}")
        else:self.sentiment_available=self.custom_sentiment is not None
        p("Loading emotion model...",4)
        if self.custom_emotion is None and hf_pipeline:
            try:
                self.emotion=hf_pipeline("text-classification",model="j-hartmann/emotion-english-distilroberta-base",device=0 if self.device.type=="cuda" else -1,top_k=None)
                self.emotion_available=True
            except Exception as e: errors.append(f"Emotion model unavailable: {e}")
        else:self.emotion_available=self.custom_emotion is not None
        self.ready=self.translation_available
        if not self.ready and errors: raise RuntimeError("\n\n".join(errors))
        return True

    def is_ready(self):return bool(self.ready and self.translation_available)

    @staticmethod
    def _contains_gurmukhi(text):
        return bool(re.search(r"[\u0A00-\u0A7F]",text))

    def _split_lines(self,text):
        lines=[]
        for raw in re.split(r"\r?\n",text):
            raw=raw.strip()
            if raw: lines.append(raw)
        return lines or [text.strip()]

    def _protect_tokens(self,text):
        protected={}
        pattern=re.compile(r"(?:https?://\S+|www\.\S+|@[\w_]+|#[\w_]+|\b[A-Z][A-Za-z0-9_-]{2,}\b)")
        def repl(m):
            key=f"__KEEP_{len(protected)}__"; protected[key]=m.group(0); return key
        return pattern.sub(repl,text),protected

    def _restore_tokens(self,text,protected):
        for k,v in protected.items(): text=text.replace(k,v)
        return text

    def _translate_batch(self,lines,run_quality=True):
        if not self.translation_available: raise RuntimeError("Punjabi translation model is unavailable")
        keys=[x.strip() for x in lines]
        out=[None]*len(keys); pending=[]; pending_idx=[]
        for i,line in enumerate(keys):
            if not line: out[i]=""; continue
            if not self._contains_gurmukhi(line):out[i]=line; continue
            if line in self.translation_cache:out[i]=self.translation_cache[line]; continue
            pending.append(line); pending_idx.append(i)
        if pending:
            protected=[]; maps=[]
            for line in pending:
                x,m=self._protect_tokens(line); protected.append(x); maps.append(m)
            batch=self.indic_processor.preprocess_batch(protected,src_lang=self.translation_source_language,tgt_lang=self.translation_target_language)
            inputs=self.translation_tokenizer(batch,truncation=True,padding="longest",return_tensors="pt",return_attention_mask=True).to(self.device)
            with torch.no_grad():
                generated=self.translation_model.generate(**inputs,use_cache=True,min_length=0,max_length=256,num_beams=5,num_return_sequences=1,early_stopping=True)
            decoded=self.translation_tokenizer.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=True)
            translations=self.indic_processor.postprocess_batch(decoded,lang=self.translation_target_language)
            for idx,raw,m,t in zip(pending_idx,pending,maps,translations):
                t=self._clean_translation(t); t=self._restore_tokens(t,m); out[idx]=t; self.translation_cache[raw]=t
                if len(self.translation_cache)>self.cache_limit:self.translation_cache.pop(next(iter(self.translation_cache)))
        return [x or "" for x in out]

    def _clean_translation(self,text):
        text=re.sub(r"\s+"," ",str(text)).strip(); text=re.sub(r"\s+([,.!?;:])",r"\1",text)
        return text

    def _quality(self,source,target):
        if not target:return False,"empty output"
        if not re.search(r"[A-Za-z]",target):return False,"output is not English"
        low=target.lower()
        bad_phrases=("jehovah","watchtower","kingdom hall","bible study","witnesses of jehovah")
        if any(x in low for x in bad_phrases):return False,"unrelated hallucinated content detected"
        words=re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?",target)
        if not words:return False,"no English words"
        if len(words)>max(80,len(source.split())*8+20):return False,"output is implausibly long"
        if len(set(w.lower() for w in words))/max(1,len(words))<0.30:return False,"repetitive output"
        if target.count(".")>max(5,len(source.split())*2):return False,"excessive sentence repetition"
        return True,"ok"

    def translate_lines(self,lines):
        lines=[str(x).strip() for x in lines if str(x).strip()]
        if not lines:return [],{"success":False,"complete":False,"translated_lines":0,"failed_lines":0,"total_lines":0,"message":"No lines supplied."}
        if not self.translation_available:return ["" for _ in lines],{"success":False,"complete":False,"translated_lines":0,"failed_lines":len(lines),"total_lines":len(lines),"message":"Local Punjabi translation model is unavailable."}
        translations=self._translate_batch(lines)
        good=[]; failed=[]
        for i,(src,tgt) in enumerate(zip(lines,translations),1):
            if self._contains_gurmukhi(src):
                ok,reason=self._quality(src,tgt)
            else:ok,reason=True,"not Punjabi"
            if ok:good.append(tgt)
            else:good.append(""); failed.append((i,reason))
        success=bool(good) and len(good)>len(failed)
        complete=not failed
        msg="All lines passed translation checks." if complete else "; ".join(f"line {i}: {r}" for i,r in failed[:5])
        return good,{"success":success,"complete":complete,"translated_lines":len(lines)-len(failed),"failed_lines":len(failed),"total_lines":len(lines),"message":msg,"failed_details":[{"line":i,"reason":r} for i,r in failed]}

    def _sentiment(self,text):
        if self.custom_sentiment:
            try:
                pred=self.custom_sentiment.predict([text])[0]; return {"label":str(pred),"score":1.0}
            except Exception:pass
        if self.sentiment:
            try:
                r=self.sentiment(text,truncation=True,max_length=512)[0]; label=str(r["label"]).lower(); mapping={"label_0":"negative","label_1":"neutral","label_2":"positive","negative":"negative","neutral":"neutral","positive":"positive"}; return {"label":mapping.get(label,label),"score":float(r["score"])}
            except Exception:pass
        vals=[]
        for w,v in self.fallback_sentiment_words.items():
            if re.search(rf"\b{re.escape(w)}\b",text.lower()):vals.append(v)
        score=sum(vals)/len(vals) if vals else 0.0
        return {"label":"positive" if score>.15 else "negative" if score<-.15 else "neutral","score":min(1,abs(score))}

    def _emotion(self,text):
        if self.custom_emotion:
            try:pred=self.custom_emotion.predict([text])[0]; return {"label":str(pred),"score":1.0}
            except Exception:pass
        if self.emotion:
            try:
                results=self.emotion(text,truncation=True,max_length=512)
                if isinstance(results,list) and results and isinstance(results[0],list):results=results[0]
                best=max(results,key=lambda x:x.get("score",0)); return {"label":str(best["label"]).lower(),"score":float(best["score"])}
            except Exception:pass
        scores=Counter()
        for w,pairs in self.fallback_emotions.items():
            if re.search(rf"\b{re.escape(w)}\b",text.lower()):
                for lab,s in pairs:scores[lab]+=s
        if not scores:return {"label":"neutral","score":0.5}
        lab,score=scores.most_common(1)[0]; return {"label":lab,"score":min(1,score)}

    def _keywords(self,text):
        words=re.findall(r"[A-Za-z]{3,}",text.lower()); stop={"the","and","you","your","for","that","this","with","from","are","was","have","has","not","but","they","them","our","their","will","just","into","love","like"}; return [w for w,_ in Counter(w for w in words if w not in stop).most_common(12)]

    def _explicit(self,text):
        found=[]; filtered=text
        for word in sorted(self.explicit_words,key=len,reverse=True):
            if not word:continue
            flags=re.IGNORECASE if re.search(r"[A-Za-z]",word) else 0
            pattern=rf"(?<!\w){re.escape(word)}(?!\w)"
            if re.search(pattern,text,flags):
                found.append(word)
                repl=self.replacements.get(word,"*"*max(3,len(word)))
                filtered=re.sub(pattern,repl,filtered,flags=flags)
        return {"count":len(found),"words":sorted(set(found)),"filtered":filtered}

    def analyze(self,text):
        text=str(text).replace("\r\n","\n").strip()
        if not text:raise ValueError("No text supplied.")
        lines=self._split_lines(text); translations,status=self.translate_lines(lines)
        line_analysis=[]; sentiment_scores=[]; emotion_scores=[]
        for i,(src,tr) in enumerate(zip(lines,translations),1):
            usable=bool(tr)
            if usable:
                s=self._sentiment(tr); e=self._emotion(tr); sentiment_scores.append(s); emotion_scores.append(e)
            else:s={"label":"unavailable","score":0}; e={"label":"unavailable","score":0}
            signed={"positive":1,"negative":-1,"neutral":0}.get(s.get("label"),0)*float(s.get("score",0))
            line_analysis.append({"line_no":i,"original":src,"translation":tr,"translation_success":usable,"sentiment":s,"emotion":e,"signed_sentiment":signed})
        valid_s=[x for x in sentiment_scores if x.get("label")!="unavailable"]; valid_e=[x for x in emotion_scores if x.get("label")!="unavailable"]
        if valid_s:
            s_label=Counter(x["label"] for x in valid_s).most_common(1)[0][0]; s_candidates=[x for x in valid_s if x["label"]==s_label]; overall_s={"label":s_label,"score":sum(x["score"] for x in s_candidates)/len(s_candidates)}
        else:overall_s={"label":"unavailable","score":0}
        if valid_e:
            e_label=Counter(x["label"] for x in valid_e).most_common(1)[0][0]; e_candidates=[x for x in valid_e if x["label"]==e_label]; overall_e={"label":e_label,"score":sum(x["score"] for x in e_candidates)/len(e_candidates)}
        else:overall_e={"label":"unavailable","score":0}
        ex=self._explicit(text); success_ratio=status["translated_lines"]/max(1,status["total_lines"]); analysis_conf=(sum(x["score"] for x in valid_e)/len(valid_e) if valid_e else 0.0); overall_conf=0.65*success_ratio+0.35*analysis_conf if valid_e else 0.65*success_ratio
        translation="\n".join(x for x in translations if x) if status["translated_lines"] else ""
        language="Punjabi (Gurmukhi)" if any(self._contains_gurmukhi(x) for x in lines) else "English / Latin"
        return {"original_text":text,"filtered_input":ex["filtered"],"translation":translation,"translated_text":translation,"translation_status":status,"language":language,"words":len(re.findall(r"\S+",text)),"characters":len(text),"lines":len(lines),"sentiment":overall_s,"emotion":overall_e,"overall_confidence":round(float(overall_conf),4),"explicit":ex,"explicit_filter":ex,"keywords":self._keywords(translation),"line_analysis":line_analysis,"model":self.translation_model_name,"translation_style":"Natural English, offline local NMT","warnings":status.get("failed_details",[])}

    def train_custom(self,csv_path,model_type="sentiment"):
        df=pd.read_csv(csv_path)
        cols={c.lower():c for c in df.columns}
        if "text" not in cols or "label" not in cols:raise ValueError("CSV must contain columns named text and label.")
        df=df[[cols["text"],cols["label"]]].dropna(); df.columns=["text","label"]
        df["text"]=df["text"].astype(str); df["label"]=df["label"].astype(str)
        if len(df)<8:raise ValueError("Please provide at least 8 labelled rows.")
        if df["label"].nunique()<2:raise ValueError("Training requires at least 2 classes.")
        pipe=Pipeline([("tfidf",TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,max_features=30000)),("clf",LogisticRegression(max_iter=1200))])
        Xtr,Xte,ytr,yte=train_test_split(df.text,df.label,test_size=.2,random_state=42,stratify=df.label if df.label.value_counts().min()>=2 else None)
        pipe.fit(Xtr,ytr); acc=accuracy_score(yte,pipe.predict(Xte)) if len(yte) else 1.0
        path=self.custom_sentiment_path if model_type=="sentiment" else self.custom_emotion_path; joblib.dump(pipe,path)
        if model_type=="sentiment":self.custom_sentiment=pipe
        else:self.custom_emotion=pipe
        return {"rows":len(df),"classes":sorted(df.label.unique().tolist()),"accuracy":float(acc),"path":path}
