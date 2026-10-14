import os
import re
import json
import csv
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from nlp_engine import NLPManager
from history_db import HistoryDB

APP_TITLE = "Punjabi Lyric AI — NLP Studio"
WINDOW_WIDTH = 1480
WINDOW_HEIGHT = 930

SAMPLES = {
    "Romantic": """ਮੈਂ ਤੈਨੂੰ ਪਿਆਰ ਕਰਦਾ ਹਾਂ
ਤੇਰੇ ਬਿਨਾ ਦਿਲ ਨਹੀਂ ਲੱਗਦਾ
ਤੂੰ ਮੇਰੀ ਜ਼ਿੰਦਗੀ ਦੀ ਖੁਸ਼ੀ ਹੈ
ਤੇਰੇ ਨਾਲ ਹਰ ਪਲ ਸੋਹਣਾ ਲੱਗਦਾ ਹੈ""",
    "Happy": """ਅੱਜ ਦਿਲ ਬਹੁਤ ਖੁਸ਼ ਹੈ
ਸਾਰੇ ਯਾਰ ਮੇਰੇ ਨਾਲ ਨੇ
ਹੱਸਦੇ ਖੇਡਦੇ ਗਾਉਂਦੇ ਰਹੀਏ
ਜ਼ਿੰਦਗੀ ਦੇ ਰੰਗ ਨਿਆਰੇ ਨੇ""",
    "Mixed": """ਕਦੇ ਹੱਸਦਾ ਹਾਂ ਤੇ ਕਦੇ ਰੋ ਪੈਂਦਾ ਹਾਂ
ਤੇਰੀ ਯਾਦ ਦਿਲ ਨੂੰ ਸਤਾਉਂਦੀ ਹੈ
ਫਿਰ ਵੀ ਤੇਰੇ ਨਾਲ ਪਿਆਰ ਕਰਦਾ ਹਾਂ
ਕਿਉਂਕਿ ਤੂੰ ਮੇਰੇ ਦਿਲ ਦੇ ਨੇੜੇ ਹੈ""",
    "Short": """ਤੂੰ ਮੇਰੀ ਜਾਨ ਹੈ
ਮੈਂ ਤੈਨੂੰ ਪਿਆਰ ਕਰਦਾ ਹਾਂ""",
}

class PunjabiLyricApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.minsize(1180, 760)
        self.protocol("WM_DELETE_WINDOW", self.close_app)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.current_theme = "dark"

        self.engine = NLPManager()
        self.db = HistoryDB()
        self.last_result = None
        self.last_text = ""
        self.busy = False
        self.model_ready = False
        self.current_analysis_id = None

        self.sentiment_var = tk.StringVar(value="—")
        self.emotion_var = tk.StringVar(value="—")
        self.confidence_var = tk.StringVar(value="—")
        self.lines_var = tk.StringVar(value="0")
        self.explicit_var = tk.StringVar(value="0")
        self.language_var = tk.StringVar(value="—")
        self.words_var = tk.StringVar(value="0")
        self.characters_var = tk.StringVar(value="0")
        self.keywords_var = tk.StringVar(value="—")
        self.dominant_emotion_var = tk.StringVar(value="—")
        self.explicit_terms_var = tk.StringVar(value="None")
        self.translation_status_var = tk.StringVar(value="Translation status: waiting")
        self.model_status_var = tk.StringVar(value="Loading local models...")

        self._build_ui()
        self._bind_shortcuts()
        self.after(300, self._load_models_async)
        self.refresh_history()
        self.update_input_meta()

    def _bind_shortcuts(self):
        self.bind("<Control-Return>", lambda e: self.start_analysis())
        self.bind("<Control-l>", lambda e: self.clear_all())
        self.bind("<Control-o>", lambda e: self.open_txt())

    def _build_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        header = ctk.CTkFrame(self, corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(header, text="🎵 Punjabi Lyric AI", font=ctk.CTkFont(size=25, weight="bold")).grid(row=0, column=0, padx=(24,8), pady=16)
        ctk.CTkLabel(header, text="Offline Punjabi→English • Sentiment • Emotion • Insights", text_color=("gray40","gray70")).grid(row=0,column=1,sticky="w",padx=8)
        self.model_badge = ctk.CTkLabel(header,text="● Loading",text_color="#f59e0b",font=ctk.CTkFont(size=13,weight="bold"))
        self.model_badge.grid(row=0,column=2,padx=12)
        self.theme_switch = ctk.CTkSwitch(header,text="Light",command=self.toggle_theme)
        self.theme_switch.grid(row=0,column=3,padx=(8,24))

        self.tabs = ctk.CTkTabview(self, corner_radius=12)
        self.tabs.grid(row=1,column=0,sticky="nsew",padx=14,pady=(10,14))
        for name in ("Analyze","Insights","Chat","Emotion Timeline","Charts","History","Train Models"):
            self.tabs.add(name)
        self._build_analyze_tab(); self._build_insights_tab(); self._build_chat_tab()
        self._build_timeline_tab(); self._build_charts_tab(); self._build_history_tab(); self._build_training_tab()

    def _build_analyze_tab(self):
        tab=self.tabs.tab("Analyze"); tab.grid_columnconfigure(0,weight=1); tab.grid_columnconfigure(1,weight=1); tab.grid_rowconfigure(1,weight=1)
        toolbar=ctk.CTkFrame(tab,fg_color="transparent"); toolbar.grid(row=0,column=0,columnspan=2,sticky="ew",pady=(0,10)); toolbar.grid_columnconfigure(7,weight=1)
        self.analyze_button=ctk.CTkButton(toolbar,text="🔍 Analyze",command=self.start_analysis,height=38,font=ctk.CTkFont(weight="bold")); self.analyze_button.grid(row=0,column=0,padx=(0,8))
        ctk.CTkButton(toolbar,text="📂 Open TXT",command=self.open_txt,height=38).grid(row=0,column=1,padx=8)
        ctk.CTkButton(toolbar,text="✨ Sample",command=self.show_samples,height=38).grid(row=0,column=2,padx=8)
        ctk.CTkButton(toolbar,text="🧹 Clear",command=self.clear_all,height=38).grid(row=0,column=3,padx=8)
        ctk.CTkButton(toolbar,text="💾 Save Report",command=self.save_report,height=38).grid(row=0,column=4,padx=8)
        self.safe_display_var=tk.BooleanVar(value=False)
        ctk.CTkCheckBox(toolbar,text="Safe display",variable=self.safe_display_var,command=self.refresh_filtered).grid(row=0,column=5,padx=8)
        ctk.CTkButton(toolbar,text="📋 Copy Translation",command=self.copy_translation,height=38).grid(row=0,column=6,padx=8)
        self.progress=ctk.CTkProgressBar(toolbar,height=8); self.progress.grid(row=0,column=7,sticky="ew",padx=(20,0)); self.progress.set(0)

        inp=ctk.CTkFrame(tab); inp.grid(row=1,column=0,sticky="nsew",padx=(0,7)); inp.grid_rowconfigure(1,weight=1); inp.grid_columnconfigure(0,weight=1)
        ctk.CTkLabel(inp,text="Punjabi Lyrics / Text",font=ctk.CTkFont(size=16,weight="bold")).grid(row=0,column=0,sticky="w",padx=15,pady=(15,8))
        self.input_text=ctk.CTkTextbox(inp,wrap="word",font=ctk.CTkFont(size=15)); self.input_text.grid(row=1,column=0,sticky="nsew",padx=15,pady=(0,8)); self.input_text.insert("1.0",SAMPLES["Romantic"]); self.input_text.bind("<KeyRelease>",self.update_input_meta)
        self.input_meta=ctk.CTkLabel(inp,text="0 words • 0 characters",text_color=("gray40","gray70")); self.input_meta.grid(row=2,column=0,sticky="w",padx=15,pady=(0,15))

        out=ctk.CTkFrame(tab); out.grid(row=1,column=1,sticky="nsew",padx=(7,0)); out.grid_rowconfigure(4,weight=1); out.grid_columnconfigure(0,weight=1)
        ctk.CTkLabel(out,text="English Translation — Natural English",font=ctk.CTkFont(size=16,weight="bold")).grid(row=0,column=0,sticky="w",padx=15,pady=(15,5))
        self.translation_status_label=ctk.CTkLabel(out,textvariable=self.translation_status_var,anchor="w",text_color=("gray40","gray70")); self.translation_status_label.grid(row=1,column=0,sticky="ew",padx=15,pady=(0,8))
        self.translation_text=ctk.CTkTextbox(out,wrap="word",font=ctk.CTkFont(size=14)); self.translation_text.grid(row=2,column=0,sticky="nsew",padx=15,pady=(0,10))
        ctk.CTkLabel(out,text="Filtered / Safe Display",font=ctk.CTkFont(size=16,weight="bold")).grid(row=3,column=0,sticky="nw",padx=15,pady=(5,5))
        self.filtered_text=ctk.CTkTextbox(out,wrap="word",font=ctk.CTkFont(size=14)); self.filtered_text.grid(row=4,column=0,sticky="nsew",padx=15,pady=(0,10))
        stats=ctk.CTkFrame(out); stats.grid(row=5,column=0,sticky="ew",padx=15,pady=(5,15))
        for i,(title,var) in enumerate((("Sentiment",self.sentiment_var),("Emotion",self.emotion_var),("Confidence",self.confidence_var),("Lines",self.lines_var),("Explicit",self.explicit_var))): self._stat_card(stats,i,title,var)

    def _stat_card(self,parent,column,title,variable):
        f=ctk.CTkFrame(parent,corner_radius=10); f.grid(row=0,column=column,sticky="ew",padx=4)
        ctk.CTkLabel(f,text=title,text_color=("gray40","gray70"),font=ctk.CTkFont(size=11)).pack(pady=(10,2)); ctk.CTkLabel(f,textvariable=variable,font=ctk.CTkFont(size=15,weight="bold")).pack(pady=(0,10))

    def _build_insights_tab(self):
        tab=self.tabs.tab("Insights"); tab.grid_columnconfigure(0,weight=1); tab.grid_rowconfigure(1,weight=1)
        ctk.CTkLabel(tab,text="📊 Lyric Insights",font=ctk.CTkFont(size=24,weight="bold")).grid(row=0,column=0,sticky="w",padx=20,pady=20)
        self.insights_frame=ctk.CTkFrame(tab); self.insights_frame.grid(row=1,column=0,sticky="nsew",padx=20,pady=(0,20))
        for i in range(3): self.insights_frame.grid_columnconfigure(i,weight=1)
        for i in range(2): self.insights_frame.grid_rowconfigure(i,weight=1)
        cards=((0,0,"🌐 Language",self.language_var),(0,1,"📝 Words",self.words_var),(0,2,"🔤 Characters",self.characters_var),(1,0,"🔑 Keywords",self.keywords_var),(1,1,"🎭 Dominant Emotion",self.dominant_emotion_var),(1,2,"⚠ Explicit Terms",self.explicit_terms_var))
        for r,c,t,v in cards: self._insight_card(r,c,t,v)

    def _insight_card(self,row,column,title,variable):
        card=ctk.CTkFrame(self.insights_frame,corner_radius=14); card.grid(row=row,column=column,sticky="nsew",padx=8,pady=8)
        ctk.CTkLabel(card,text=title,font=ctk.CTkFont(size=15,weight="bold")).pack(anchor="w",padx=18,pady=(18,8)); ctk.CTkLabel(card,textvariable=variable,wraplength=350,justify="left",anchor="nw",font=ctk.CTkFont(size=14)).pack(fill="both",expand=True,padx=18,pady=(0,18))

    def _build_chat_tab(self):
        tab=self.tabs.tab("Chat"); tab.grid_columnconfigure(0,weight=1); tab.grid_rowconfigure(1,weight=1)
        ctk.CTkLabel(tab,text="💬 Punjabi Lyric AI Chat",font=ctk.CTkFont(size=24,weight="bold")).grid(row=0,column=0,sticky="w",padx=20,pady=(20,10))
        self.chat_box=ctk.CTkTextbox(tab,wrap="word",font=ctk.CTkFont(size=14)); self.chat_box.grid(row=1,column=0,sticky="nsew",padx=20,pady=(0,10)); self.chat_box.configure(state="disabled")
        bottom=ctk.CTkFrame(tab); bottom.grid(row=2,column=0,sticky="ew",padx=20,pady=(0,20)); bottom.grid_columnconfigure(0,weight=1)
        self.chat_input=ctk.CTkEntry(bottom,placeholder_text="Paste Punjabi lyrics or type a question...",height=42); self.chat_input.grid(row=0,column=0,sticky="ew",padx=(10,8),pady=10); self.chat_input.bind("<Return>",lambda e:self.send_chat())
        ctk.CTkButton(bottom,text="Analyze",width=100,height=42,command=self.send_chat).grid(row=0,column=1,padx=5); ctk.CTkButton(bottom,text="Analyze TXT",width=110,height=42,command=self.chat_open_txt).grid(row=0,column=2,padx=5); ctk.CTkButton(bottom,text="Clear",width=80,height=42,command=self.clear_chat).grid(row=0,column=3,padx=(5,10))

    def _build_timeline_tab(self):
        tab=self.tabs.tab("Emotion Timeline"); tab.grid_columnconfigure(0,weight=1); tab.grid_rowconfigure(1,weight=1)
        self.timeline_title=ctk.CTkLabel(tab,text="🎭 Emotion Journey",font=ctk.CTkFont(size=23,weight="bold")); self.timeline_title.grid(row=0,column=0,sticky="w",padx=20,pady=20)
        self.timeline_frame=ctk.CTkFrame(tab); self.timeline_frame.grid(row=1,column=0,sticky="nsew",padx=20,pady=(0,20)); self._show_empty_visual_message(self.timeline_frame,"Analyze lyrics to see the emotion journey.")

    def _build_charts_tab(self):
        tab=self.tabs.tab("Charts"); tab.grid_columnconfigure(0,weight=1); tab.grid_rowconfigure(1,weight=1)
        self.charts_title=ctk.CTkLabel(tab,text="📈 Analysis Charts",font=ctk.CTkFont(size=23,weight="bold")); self.charts_title.grid(row=0,column=0,sticky="w",padx=20,pady=20)
        self.charts_frame=ctk.CTkFrame(tab); self.charts_frame.grid(row=1,column=0,sticky="nsew",padx=20,pady=(0,20)); self._show_empty_visual_message(self.charts_frame,"Analyze lyrics to see charts.")

    def _build_history_tab(self):
        tab=self.tabs.tab("History"); tab.grid_columnconfigure(0,weight=1); tab.grid_rowconfigure(1,weight=1)
        toolbar=ctk.CTkFrame(tab,fg_color="transparent"); toolbar.grid(row=0,column=0,sticky="ew",padx=10,pady=10); toolbar.grid_columnconfigure(4,weight=1)
        for i,(txt,cmd) in enumerate((("↻ Refresh",self.refresh_history),("📂 Open",self.open_history),("🗑 Delete",self.delete_history),("Clear All",self.clear_history))): ctk.CTkButton(toolbar,text=txt,command=cmd).grid(row=0,column=i,padx=5)
        self.history_search=ctk.CTkEntry(toolbar,placeholder_text="Search history..."); self.history_search.grid(row=0,column=4,sticky="ew",padx=10); self.history_search.bind("<KeyRelease>",lambda e:self.refresh_history())
        frame=ctk.CTkFrame(tab); frame.grid(row=1,column=0,sticky="nsew",padx=10,pady=(0,10)); frame.grid_rowconfigure(0,weight=1); frame.grid_columnconfigure(0,weight=1)
        cols=("id","date","source","title","sentiment","emotion","confidence","explicit")
        self.history_tree=ttk.Treeview(frame,columns=cols,show="headings")
        headings={"id":"ID","date":"Date","source":"Source","title":"Title","sentiment":"Sentiment","emotion":"Emotion","confidence":"Confidence","explicit":"Explicit"}; widths={"id":60,"date":150,"source":90,"title":180,"sentiment":100,"emotion":110,"confidence":100,"explicit":80}
        for c in cols: self.history_tree.heading(c,text=headings[c]); self.history_tree.column(c,width=widths[c],anchor="center")
        sb=ttk.Scrollbar(frame,orient="vertical",command=self.history_tree.yview); self.history_tree.configure(yscrollcommand=sb.set); self.history_tree.grid(row=0,column=0,sticky="nsew"); sb.grid(row=0,column=1,sticky="ns"); self.history_tree.bind("<Double-1>",lambda e:self.open_history())

    def _build_training_tab(self):
        tab=self.tabs.tab("Train Models"); tab.grid_columnconfigure(0,weight=1); tab.grid_rowconfigure(3,weight=1)
        ctk.CTkLabel(tab,text="🧠 Custom Model Training",font=ctk.CTkFont(size=24,weight="bold")).grid(row=0,column=0,sticky="w",padx=20,pady=(20,8))
        ctk.CTkLabel(tab,text="CSV columns: text,label. Trained models stay local.",text_color=("gray40","gray70")).grid(row=1,column=0,sticky="w",padx=20,pady=(0,20))
        options=ctk.CTkFrame(tab); options.grid(row=2,column=0,sticky="ew",padx=20,pady=10); self.training_type=tk.StringVar(value="sentiment")
        ctk.CTkRadioButton(options,text="Sentiment",variable=self.training_type,value="sentiment").pack(side="left",padx=15,pady=15); ctk.CTkRadioButton(options,text="Emotion",variable=self.training_type,value="emotion").pack(side="left",padx=15,pady=15); ctk.CTkButton(options,text="📂 Choose CSV & Train",command=self.choose_training_file).pack(side="right",padx=15,pady=10)
        self.training_log=ctk.CTkTextbox(tab,wrap="word",font=ctk.CTkFont(size=13)); self.training_log.grid(row=3,column=0,sticky="nsew",padx=20,pady=(10,20)); self.training_log.configure(state="disabled")

    def _load_models_async(self):
        self.busy=True; self.analyze_button.configure(state="disabled"); self.progress.set(0); threading.Thread(target=self._load_models_worker,daemon=True).start()

    def _load_models_worker(self):
        try:
            def cb(message,current,total): self.after(0,lambda m=message,c=current,t=total:self._model_progress(m,c,t))
            self.engine.load(cb); self.after(0,lambda r=self.engine.is_ready():self._models_loaded(r))
        except Exception as e:
            msg=str(e); self.after(0,lambda m=msg:self._models_failed(m))

    def _model_progress(self,message,current,total):
        self.model_status_var.set(message)
        if total: self.progress.set(min(1,current/total))

    def _models_loaded(self,ready):
        self.busy=False; self.model_ready=bool(ready); self.progress.set(1); self.analyze_button.configure(state="normal")
        if ready:
            self.model_badge.configure(text="● Models Ready",text_color="#22c55e"); self.model_status_var.set("Local NLP models ready.")
        else:
            self.model_badge.configure(text="● Limited",text_color="#f59e0b"); self.model_status_var.set("Some optional NLP models are unavailable.")

    def _models_failed(self,message):
        self.busy=False; self.model_ready=False; self.analyze_button.configure(state="normal"); self.model_badge.configure(text="● Error",text_color="#ef4444"); self.model_status_var.set("Model loading failed.")
        messagebox.showerror("Model Loading Error",message)

    def toggle_theme(self):
        self.current_theme="light" if self.theme_switch.get() else "dark"; ctk.set_appearance_mode(self.current_theme)
        if self.last_result: self.draw_timeline(self.last_result); self.draw_charts(self.last_result)

    def update_input_meta(self,event=None):
        text=self.input_text.get("1.0","end-1c").strip(); self.input_meta.configure(text=f"{len(re.findall(r'\\S+',text))} words • {len(text)} characters")

    def show_samples(self):
        d=ctk.CTkToplevel(self); d.title("Choose Sample"); d.geometry("420x360"); d.transient(self)
        ctk.CTkLabel(d,text="Choose a sample lyric",font=ctk.CTkFont(size=20,weight="bold")).pack(pady=20)
        for name in SAMPLES: ctk.CTkButton(d,text=name,command=lambda n=name,dlg=d:self.load_sample(n,dlg)).pack(fill="x",padx=40,pady=6)

    def load_sample(self,name,dialog):
        self.input_text.delete("1.0","end"); self.input_text.insert("1.0",SAMPLES[name]); self.update_input_meta(); dialog.destroy()

    def _read_text_file(self,path):
        for enc in ("utf-8","utf-8-sig","utf-16","cp1252"):
            try:
                with open(path,"r",encoding=enc) as f: return f.read()
            except UnicodeDecodeError: continue
        raise UnicodeDecodeError("unknown","",0,1,"Could not decode text file")

    def open_txt(self):
        path=filedialog.askopenfilename(title="Open Punjabi Lyrics",filetypes=[("Text files","*.txt"),("All files","*.*")])
        if not path:return
        try:
            text=self._read_text_file(path); self.input_text.delete("1.0","end"); self.input_text.insert("1.0",text); self.update_input_meta()
        except Exception as e: messagebox.showerror("Open Error",str(e))

    def start_analysis(self):
        if self.busy:return
        text=self.input_text.get("1.0","end-1c").strip()
        if not text: messagebox.showwarning("No Text","Paste Punjabi lyrics or text first."); return
        if not self.model_ready: messagebox.showwarning("Models Not Ready","Wait for the local models to finish loading."); return
        self.busy=True; self.analyze_button.configure(state="disabled"); self.progress.configure(mode="indeterminate"); self.progress.start(); threading.Thread(target=self._analysis_worker,args=(text,),daemon=True).start()

    def _analysis_worker(self,text):
        try:
            result=self.engine.analyze(text)
            aid=None
            try: aid=self.db.save_analysis(result,original_text=text,source="manual",title="Manual Analysis")
            except Exception: pass
            self.after(0,lambda r=result,a=aid:self.show_result(r,a,True))
        except Exception as e:
            msg=str(e); self.after(0,lambda m=msg:self.analysis_failed(m))

    def analysis_failed(self,message):
        self.busy=False; self.progress.stop(); self.progress.configure(mode="determinate"); self.progress.set(0); self.analyze_button.configure(state="normal"); messagebox.showerror("Analysis Error",message)

    def show_result(self,result,analysis_id=None,switch_tab=True):
        self.busy=False; self.progress.stop(); self.progress.configure(mode="determinate"); self.progress.set(1); self.analyze_button.configure(state="normal")
        self.last_result=result; self.current_analysis_id=analysis_id; self.last_text=str(result.get("original_text",""))
        self.translation_text.delete("1.0","end"); self.translation_text.insert("1.0",str(result.get("translation","")))
        status=result.get("translation_status",{}); success=bool(status.get("success")); complete=bool(status.get("complete")); tc=status.get("translated_lines",0); fc=status.get("failed_lines",0); total=status.get("total_lines",result.get("lines",0))
        if complete and success: txt=f"✓ Translation complete • {tc}/{total} lines"; color="#22c55e"
        elif success: txt=f"⚠ Partial translation • {tc}/{total} lines • {status.get('message','')}"; color="#f59e0b"
        else: txt=f"⚠ Reliable translation unavailable • {fc}/{total} lines • {status.get('message','')}"; color="#ef4444"
        self.translation_status_var.set(txt); self.translation_status_label.configure(text_color=color)
        self.refresh_filtered()
        sentiment=result.get("sentiment",{}); emotion=result.get("emotion",{}); conf=float(result.get("overall_confidence",0) or 0); explicit=result.get("explicit",{})
        self.sentiment_var.set(str(sentiment.get("label","neutral")).title()); self.emotion_var.set(str(emotion.get("label","neutral")).title()); self.confidence_var.set(f"{conf*100:.1f}%"); self.lines_var.set(str(result.get("lines",0))); self.explicit_var.set(str(explicit.get("count",0)))
        self.update_insights(result); self.draw_timeline(result); self.draw_charts(result); self.refresh_history()
        if switch_tab:self.tabs.set("Analyze")

    def refresh_filtered(self):
        if not self.last_result:return
        text=self.last_result.get("filtered_input",self.last_text) if self.safe_display_var.get() else self.last_text
        self.filtered_text.delete("1.0","end"); self.filtered_text.insert("1.0",text)

    def update_insights(self,result):
        self.language_var.set(result.get("language","Unknown")); self.words_var.set(str(result.get("words",0))); self.characters_var.set(str(result.get("characters",0)))
        self.keywords_var.set(", ".join(map(str,result.get("keywords",[]))) or "None detected")
        self.dominant_emotion_var.set(str(result.get("emotion",{}).get("label","neutral")).title())
        terms=result.get("explicit",{}).get("words",[]); self.explicit_terms_var.set(", ".join(map(str,terms)) if terms else "None detected")

    def clear_all(self):
        self.input_text.delete("1.0","end"); self.translation_text.delete("1.0","end"); self.filtered_text.delete("1.0","end")
        for var,val in ((self.sentiment_var,"—"),(self.emotion_var,"—"),(self.confidence_var,"—"),(self.lines_var,"0"),(self.explicit_var,"0"),(self.language_var,"—"),(self.words_var,"0"),(self.characters_var,"0"),(self.keywords_var,"—"),(self.dominant_emotion_var,"—"),(self.explicit_terms_var,"None"),(self.translation_status_var,"Translation status: waiting")): var.set(val)
        self.translation_status_label.configure(text_color=("gray40","gray70")); self.last_result=None; self.last_text=""; self.current_analysis_id=None; self.update_input_meta(); self._clear_frame(self.timeline_frame); self._clear_frame(self.charts_frame); self._show_empty_visual_message(self.timeline_frame,"Analyze lyrics to see the emotion journey."); self._show_empty_visual_message(self.charts_frame,"Analyze lyrics to see charts.")

    def copy_translation(self):
        text=self.translation_text.get("1.0","end-1c").strip()
        if not text:return
        self.clipboard_clear(); self.clipboard_append(text); self.update();

    def chat_append(self,speaker,message):
        self.chat_box.configure(state="normal"); self.chat_box.insert("end",f"{speaker}\n{message}\n\n"); self.chat_box.see("end"); self.chat_box.configure(state="disabled")

    def send_chat(self):
        if self.busy:return
        text=self.chat_input.get().strip()
        if not text:return
        if not self.model_ready:self.chat_append("SYSTEM","Local NLP models are still loading."); return
        self.chat_input.delete(0,"end"); self.chat_append("YOU",text); self.chat_append("PUNJABI LYRIC AI","Analyzing..."); self.busy=True; threading.Thread(target=self._chat_worker,args=(text,),daemon=True).start()

    def _chat_worker(self,text):
        try:
            result=self.engine.analyze(text); aid=None
            try: aid=self.db.save_analysis(result,original_text=text,source="chat",title="Chat Message")
            except Exception: pass
            self.after(0,lambda r=result,a=aid:self._chat_result(r,a))
        except Exception as e:
            msg=str(e); self.after(0,lambda m=msg:self._chat_failed(m))

    def _chat_result(self,result,analysis_id=None):
        s=result.get("sentiment",{}); e=result.get("emotion",{}); status=result.get("translation_status",{}); ex=result.get("explicit",{}); conf=float(result.get("overall_confidence",0) or 0)
        translation=result.get("translation","") if status.get("success") else "Reliable English translation was not available."
        response=f"Translation: {translation}\n\nSentiment: {str(s.get('label','neutral')).title()}\nEmotion: {str(e.get('label','neutral')).title()}\nConfidence: {conf*100:.1f}%\nExplicit terms: {ex.get('count',0)}"
        if ex.get("count",0): response+=f"\n\nSafe display:\n{result.get('filtered_input','')}"
        self.chat_append("PUNJABI LYRIC AI",response); self.show_result(result,analysis_id,False); self.tabs.set("Chat")

    def _chat_failed(self,message): self.busy=False; self.chat_append("PUNJABI LYRIC AI",f"Analysis failed:\n{message}")

    def chat_open_txt(self):
        if self.busy:return
        path=filedialog.askopenfilename(title="Open TXT for Chat Analysis",filetypes=[("Text files","*.txt"),("All files","*.*")])
        if not path:return
        try:text=self._read_text_file(path).strip()
        except Exception as e:messagebox.showerror("File Error",str(e)); return
        if not text:messagebox.showwarning("Empty File","The selected file contains no text."); return
        if not self.model_ready:self.chat_append("SYSTEM","Local NLP models are still loading."); return
        self.chat_append("YOU",f"[File: {os.path.basename(path)}]\n{text}"); self.chat_append("PUNJABI LYRIC AI","Analyzing file..."); self.busy=True; threading.Thread(target=self._chat_worker,args=(text,),daemon=True).start()

    def clear_chat(self):
        self.chat_box.configure(state="normal"); self.chat_box.delete("1.0","end"); self.chat_box.configure(state="disabled"); self.chat_input.delete(0,"end")

    def _clear_frame(self,frame):
        for w in frame.winfo_children(): w.destroy()

    def _show_empty_visual_message(self,frame,message): ctk.CTkLabel(frame,text=message,text_color=("gray40","gray70"),font=ctk.CTkFont(size=15)).pack(expand=True)

    def _get_line_data(self,result):
        out=[]
        for i,item in enumerate(result.get("line_analysis",[]) or [],1):
            if not isinstance(item,dict):continue
            em=item.get("emotion",{}) if isinstance(item.get("emotion",{}),dict) else {}; se=item.get("sentiment",{}) if isinstance(item.get("sentiment",{}),dict) else {}
            out.append({"line_no":item.get("line_no",i),"emotion":str(em.get("label","neutral")).lower(),"emotion_score":float(em.get("score",0) or 0),"sentiment":str(se.get("label","neutral")).lower(),"signed_sentiment":float(item.get("signed_sentiment",0) or 0),"translation_success":bool(item.get("translation_success",False))})
        return out

    def draw_timeline(self,result):
        self._clear_frame(self.timeline_frame); lines=self._get_line_data(result)
        if not lines:self._show_empty_visual_message(self.timeline_frame,"No line-level emotion data available."); return
        dominant=str(result.get("emotion",{}).get("label","neutral")).title(); self.timeline_title.configure(text=f"🎭 Emotion Journey • {len(lines)} lines • Dominant: {dominant}")
        bg="#ffffff" if self.current_theme=="light" else "#212121"; fg="#111111" if self.current_theme=="light" else "#eeeeee"
        fig=Figure(figsize=(10,5),dpi=100,facecolor=bg); ax=fig.add_subplot(111); ax.set_facecolor(bg)
        xs=[x["line_no"] for x in lines]; ys=[x["emotion_score"] for x in lines]; ax.plot(xs,ys,marker="o",linewidth=2)
        for x in lines: ax.annotate(x["emotion"].title() if x["translation_success"] else "N/A",(x["line_no"],x["emotion_score"]),xytext=(0,8),textcoords="offset points",ha="center",fontsize=8,color=fg)
        ax.set_ylim(0,1.08); ax.set_xlabel("Lyric Line",color=fg); ax.set_ylabel("Emotion Confidence",color=fg); ax.set_title("Emotion Journey Across Lyrics",color=fg); ax.tick_params(colors=fg)
        for sp in ax.spines.values():sp.set_color(fg)
        fig.tight_layout(); canvas=FigureCanvasTkAgg(fig,master=self.timeline_frame); canvas.draw(); canvas.get_tk_widget().pack(fill="both",expand=True)

    def draw_charts(self,result):
        self._clear_frame(self.charts_frame); lines=self._get_line_data(result)
        valid=[x for x in lines if x["translation_success"]]; counts={}
        for x in valid:counts[x["emotion"]]=counts.get(x["emotion"],0)+1
        if not lines:self._show_empty_visual_message(self.charts_frame,"No chart data available."); return
        total=sum(counts.values()) or 1; bg="#ffffff" if self.current_theme=="light" else "#212121"; fg="#111111" if self.current_theme=="light" else "#eeeeee"
        fig=Figure(figsize=(12,5),dpi=100,facecolor=bg); a1=fig.add_subplot(121); a2=fig.add_subplot(122); a1.set_facecolor(bg); a2.set_facecolor(bg)
        if counts:
            names=list(counts); vals=[counts[n]/total for n in names]; bars=a1.bar(names,vals)
            for b,v in zip(bars,vals):a1.text(b.get_x()+b.get_width()/2,b.get_height()+.01,f"{v*100:.0f}%",ha="center",fontsize=8,color=fg)
            a1.set_ylim(0,1.15); a1.tick_params(axis="x",rotation=35,colors=fg); a1.tick_params(axis="y",colors=fg)
        a1.set_title("Emotion Distribution",color=fg); a1.set_ylabel("Proportion",color=fg)
        xs=[x["line_no"] for x in lines]; ys=[x["signed_sentiment"] for x in lines]; a2.plot(xs,ys,marker="o",linewidth=2); a2.axhline(0,linewidth=1); a2.set_ylim(-1,1); a2.set_title("Sentiment Trend",color=fg); a2.set_xlabel("Lyric Line",color=fg); a2.set_ylabel("Sentiment Score",color=fg); a2.tick_params(colors=fg)
        for ax in (a1,a2):
            for sp in ax.spines.values():sp.set_color(fg)
        fig.tight_layout(); canvas=FigureCanvasTkAgg(fig,master=self.charts_frame); canvas.draw(); canvas.get_tk_widget().pack(fill="both",expand=True)

    def save_report(self):
        if not self.last_result:messagebox.showwarning("No Analysis","Analyze some lyrics before saving a report."); return
        path=filedialog.asksaveasfilename(title="Save Analysis Report",defaultextension=".txt",filetypes=[("Text Report","*.txt"),("JSON","*.json"),("CSV","*.csv")])
        if not path:return
        try:
            r=self.last_result; ext=os.path.splitext(path)[1].lower()
            if ext==".json":
                with open(path,"w",encoding="utf-8") as f:json.dump(r,f,ensure_ascii=False,indent=2)
            elif ext==".csv":
                with open(path,"w",encoding="utf-8-sig",newline="") as f:
                    w=csv.writer(f); w.writerow(["Line","Original","Translation","Translation Success","Sentiment","Sentiment Score","Emotion","Emotion Score"])
                    for x in r.get("line_analysis",[]):
                        s=x.get("sentiment",{}); e=x.get("emotion",{}); w.writerow([x.get("line_no",""),x.get("original",""),x.get("translation",""),x.get("translation_success",False),s.get("label","neutral"),s.get("score",0),e.get("label","neutral"),e.get("score",0)])
            else:
                with open(path,"w",encoding="utf-8") as f:
                    f.write("PUNJABI LYRIC AI — NLP REPORT\n"+"="*60+"\n\nORIGINAL TEXT\n"+r.get("original_text","")+"\n\nENGLISH TRANSLATION\n"+r.get("translation","")+"\n\nTRANSLATION STATUS\n"+json.dumps(r.get("translation_status",{}),ensure_ascii=False,indent=2)+"\n\nANALYSIS\n"+json.dumps({k:r.get(k) for k in ("language","sentiment","emotion","overall_confidence","keywords","explicit")},ensure_ascii=False,indent=2)+"\n\nLINE-BY-LINE\n"+json.dumps(r.get("line_analysis",[]),ensure_ascii=False,indent=2))
            messagebox.showinfo("Report Saved",f"Report saved successfully:\n{path}")
        except Exception as e:messagebox.showerror("Save Error",str(e))

    def refresh_history(self):
        if not hasattr(self,"history_tree"):return
        for item in self.history_tree.get_children():self.history_tree.delete(item)
        search=self.history_search.get().strip().lower() if hasattr(self,"history_search") else ""
        try:rows=self.db.recent(200)
        except Exception:rows=[]
        for row in rows:
            searchable=" ".join(map(str,row)).lower()
            if search and search not in searchable:continue
            vals=list(row); vals[6]=f"{float(vals[6])*100:.1f}%" if vals[6] is not None else "0%"; self.history_tree.insert("","end",values=vals)

    def _selected_history_id(self):
        s=self.history_tree.selection()
        if not s:return None
        vals=self.history_tree.item(s[0],"values"); return vals[0] if vals else None

    def open_history(self):
        rid=self._selected_history_id()
        if rid is None:messagebox.showwarning("Select History","Select an analysis first."); return
        try:r=self.db.get(rid)
        except Exception as e:messagebox.showerror("History Error",str(e)); return
        if r:self.show_result(r,rid,True)

    def delete_history(self):
        rid=self._selected_history_id()
        if rid is None:messagebox.showwarning("Select History","Select an analysis to delete."); return
        if messagebox.askyesno("Delete Analysis","Delete the selected analysis?"):
            try:self.db.delete(rid); self.refresh_history()
            except Exception as e:messagebox.showerror("Delete Error",str(e))

    def clear_history(self):
        if messagebox.askyesno("Clear History","Delete all saved analyses?"):
            try:self.db.clear(); self.refresh_history()
            except Exception as e:messagebox.showerror("Clear Error",str(e))

    def training_log_append(self,text):
        self.training_log.configure(state="normal"); self.training_log.insert("end",text+"\n"); self.training_log.see("end"); self.training_log.configure(state="disabled")

    def choose_training_file(self):
        if self.busy:messagebox.showwarning("Busy","Please wait for the current operation to finish."); return
        path=filedialog.askopenfilename(title="Choose Training CSV",filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if not path:return
        typ=self.training_type.get(); self.training_log_append(f"Training {typ} from {path}..."); self.busy=True; threading.Thread(target=self._train_worker,args=(path,typ),daemon=True).start()

    def _train_worker(self,path,typ):
        try:r=self.engine.train_custom(path,model_type=typ); self.after(0,lambda x=r:self._training_complete(x))
        except Exception as e:msg=str(e); self.after(0,lambda m=msg:self._training_failed(m))

    def _training_complete(self,r):
        self.busy=False; self.training_log_append("✓ Training completed."); self.training_log_append(f"Rows: {r.get('rows',0)}"); self.training_log_append(f"Classes: {', '.join(map(str,r.get('classes',[])))}"); self.training_log_append(f"Accuracy: {float(r.get('accuracy',0))*100:.2f}%"); self.training_log_append(f"Saved: {r.get('path','')}"); messagebox.showinfo("Training Complete","Custom model trained successfully.")

    def _training_failed(self,message):self.busy=False; self.training_log_append("✗ Training failed."); self.training_log_append(message); messagebox.showerror("Training Error",message)

    def close_app(self):
        try:self.db.close()
        except Exception:pass
        self.destroy()

if __name__=="__main__": PunjabiLyricApp().mainloop()
