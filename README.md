# 🧭 DecisionOS — AI Decision & Negotiation Engine

Streamlit + CrewAI + Groq (`openai/gpt-oss-120b`). Five agents (Finance, Risk, Operations,
Negotiation, Decision) analyze a business decision; **Python calculates the scores**, not the LLM.

## Structure
```
app.py               Streamlit UI (setup, results, What-If, executive summary)
crew.py              Sequential CrewAI crew + scenario summary
config.py            Groq/CrewAI LLM setup (change the model here)
decision_engine.py   Weighted scoring, validation, What-If sensitivity (no LLM)
prompts.py           All prompt text + guardrails
agents/              One file per agent (finance, risk, operations, negotiation, decision)
```

## Run locally
```
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...                          # Windows: set GROQ_API_KEY=gsk_...
streamlit run app.py
```

## Deploy on Streamlit Cloud
1. Push this folder to GitHub (do NOT commit secrets).
2. share.streamlit.io → New app → pick repo, branch, main file `app.py`.
3. **Advanced settings → Python version: 3.12**; Secrets: `GROQ_API_KEY = "gsk_..."`.
4. Deploy.

## Notes
- CrewAI 1.x has no built-in Groq provider; `config.py` uses Groq's OpenAI-compatible endpoint.
- Decision support only: quality, risk and strategic values are manually entered business inputs.
