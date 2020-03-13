# use example

use comand line
```
python gen_data.py gbpjpy15.csv
```

```python
import sac
#spread : gbpjpy = 10, eurusd = 3
#pip_cost : /jpy = 1000, /usd = 100000
agent = sac.agent(spread=10, pip_cost=1000, leverage=100, min_lots=0.01, assets=100000, available_assets_rate=0.4,
                 restore=True, step_size=96, n=3, lr=1e-5)

agent.run()
```
