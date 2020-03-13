# use example

use comand line
```
python gen_data.py file_name
```

```python
import sac
agent = sac.agent(spread=10, pip_cost=1000, leverage=100, min_lots=0.01, assets=100000, available_assets_rate=0.4,
                 restore=True, step_size=96, n=3, lr=1e-5)
agent.run()
```

# argument
spread : Trade cost.
pip_cost : Magnification when calculating pip.
  example : gbpjpy, 133.745 - 133.744 = 0.001, 0.001 * 1000 = 1pip.
leverage : Available assets = assets * leverage
