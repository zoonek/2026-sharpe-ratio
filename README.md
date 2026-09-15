# 2026-sharpe-ratio

Code to reproduce the numeric examples and plots
from the paper 
*[Signal-to-noise ratio inference under volatility clustering and heavy tails](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6568702)*
(formerly "Sharpe ratio under GARCH returns").

Run the code as follows: 
```python
uv venv
uv pip install -r requirements.txt
uv run python functions.py
mkdir outputs
for notebook in *.ipynb
do
  uv run papermill "$notebook" outputs/"$notebook"
done
```
