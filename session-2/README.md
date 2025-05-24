# Session 2: LLMs, Prompt Engineering

Links:
- [Example 1: Research paper summarizer](https://github.com/ShawhinT/AI-Builders-Bootcamp-4/blob/main/session-2/example_1-paper_summarizer.ipynb)
- [Example 2: Text classifier](https://github.com/ShawhinT/AI-Builders-Bootcamp-4/blob/main/session-2/example_2-text-classifier.ipynb)
- [Example 3: Local visual QA with LLaMA 3.2 Vision](https://github.com/ShawhinT/AI-Builders-Bootcamp-4/blob/main/session-2/example_3-local_visual_QA.ipynb)

## How to run the examples

1. Clone this repo
2. Navigate to downloaded folder and create new venv
```
python -m venv s2-env
```
3. Activate venv
```
# mac/linux
source s2-env/bin/activate

# windows
.\s2-env\Scripts\activate.bat
```
4. Install dependencies
```
pip install -r requirements.txt
```
5. Launch Jupyter Lab
```
jupyter lab
```

Note: Examples 1 and 2 import an openai API key from a `.env` file. You can alternatively set the API key as an environment variable in your terminal.
```
export OPENAI_API_KEY=<your-openai-api-key>
```
