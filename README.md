# Punjabi Lyric AI — Revamped Offline Edition

## What changed
- Replaced the previous NLLB translation path with **AI4Bharat IndicTrans2 Indic→English Distilled 200M**.
- Direct model loading works with Transformers 5.x; the project does not use the removed Transformers v5 `pipeline("translation")` path.
- Punjabi is explicitly configured as `pan_Guru` and English as `eng_Latn`.
- Added translation self-test and hallucination checks. If a line looks unrelated or unsafe to trust, it is rejected instead of being displayed as a confident translation.
- Line-by-line translation preserves lyric line structure.
- Added a stronger GUI, working Chat tab, history, charts, timeline, reports, custom classifiers, safe display, and copy-translation button.
- Removed the About tab.

## First setup
Use Python 3.12 on Windows.

```powershell
py -m pip install -r requirements.txt
py app.py
```

The first run downloads the local Hugging Face models. Internet is required only for the initial model download. After the models are cached, translation/inference is local.

### IndicTrans2 access
The AI4Bharat IndicTrans2 model page may require accepting its Hugging Face access conditions before the model files can be downloaded. If the first launch reports an access/gating error, open the model page, sign in to Hugging Face, accept the model conditions, then run the app again.

Model: `ai4bharat/indictrans2-indic-en-dist-200M`

## Use
1. Paste Punjabi lyrics into **Analyze**.
2. Press **Analyze** or Ctrl+Enter.
3. Read the English translation.
4. If the local model cannot produce a trustworthy translation, the app says so instead of inventing unrelated text.
5. Use Insights, Chat, Emotion Timeline, Charts and History as needed.

## Important limitation
This is a local machine-translation system, not a human Punjabi lyricist. It is designed to avoid the previous catastrophic hallucinations, but slang, wordplay, dialect, cultural references and highly poetic lyrics can still be imperfect. The app intentionally flags/rejects suspicious output rather than pretending uncertain output is correct.
