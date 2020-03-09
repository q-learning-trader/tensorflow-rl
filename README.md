# use example

```
use comand line

python gen_data.py file_name
```

* types: PG = continuous action space, DQN = discrete action space
* rewards: 2 = continuous action space, 1 = discrete action space
```python
import sac
agent = sac.agent(spread=10, pip_cost=1000, leverage=100, min_lots=0.01, assets=100000, available_assets_rate=0.4,
                 restore=True, step_size=96, n=3, lr=1e-5, rewards=2)
agent.run(types="PG")
```
