# 複数のポジションを保有する場合、最初のポジションから処理をして行く
import numpy as np


class Reward:
    def __init__(self, spread, leverage, pip_cost, min_lots=0.01, assets=1000000, available_assets_rate=0.4):
        self.spread = spread
        self.leverage = leverage
        self.pip_cost = pip_cost
        self.min_lots = min_lots
        self.initial_assets = assets
        self.assets = assets
        self.available_assets_rate = available_assets_rate

        self.max_los_cut = None
        self.total_gain = []
        self.l = []
        # use rewards.max_los_cut = -np.mean(atr) * pip_cost

        self.growth_rate = []
        self.positions = None

    def reset(self):
        self.assets = self.initial_assets

        self.growth_rate = []
        self.positions = None
        self.total_gain = []
        self.l = []

    def reward(self, trend, high, low, action, atr, scale_atr):
        old_action = None
        position = None
        gain = 0
        self.los_cut = self.max_los_cut
        old_assets = self.assets

        self.losses = []
        undetermined_assets = self.assets
        self.positions = None
        self.minimum_required_margin = trend[0] * self.pip_cost / self.leverage

        for action, trend, idx, atr, scale_atr in zip(action, trend, range(len(high)), atr, scale_atr):
            if old_action == None and position == None and self.assets > self.minimum_required_margin * 2 and action != 2:
                self.minimum_required_margin = trend * self.pip_cost / self.leverage
                lot = self.assets * self.available_assets_rate / self.minimum_required_margin * self.min_lots
                self.lot = int(lot * 10 ** 2) / (10 ** 2)

                self.positions = trend + self.spread if action == 0 else trend - self.spread
                self.tp = min(abs(self.max_los_cut), atr * self.pip_cost)
                self.los_cut = max(self.max_los_cut, -atr * self.pip_cost)
                old_action = action

            elif old_action != None:
                if self.positions != None:
                    if old_action == 0:
                        gain = (trend - self.positions) * self.pip_cost * self.lot * 100
                        loss = (low[idx - 1] - self.positions) * self.pip_cost * self.lot * 100
                    else:
                        gain = (self.positions - trend) * self.pip_cost * self.lot * 100
                        loss = (self.positions - high[idx - 1]) * self.pip_cost * self.lot * 100

                    loss = min(gain, loss)
                    tp = self.tp * self.lot * 100
                    loss_cut = self.los_cut * self.lot * 100
                    gain = loss_cut if loss <= loss_cut else gain
                    undetermined_assets = max(self.assets + gain, self.assets + loss_cut)

                    confirm = gain == loss_cut
                    buy = old_action != action and action != 2
                    if confirm or buy:
                        self.assets = undetermined_assets

                        if self.assets > self.minimum_required_margin * 2:
                            if action == 2:
                                self.positions = None
                                old_action = None
                            else:
                                self.minimum_required_margin = trend * self.pip_cost / self.leverage
                                lot = self.assets * self.available_assets_rate / self.minimum_required_margin * self.min_lots
                                self.lot = int(lot * 10 ** 2) / (10 ** 2)
                                self.positions = trend + self.spread if action == 0 else trend - self.spread
                                self.tp = min(abs(self.max_los_cut), atr * self.pip_cost)
                                self.los_cut = max(self.max_los_cut, -atr * self.pip_cost)
                        else:
                            self.positions = None

                    # else:
                    #     print("gain {}".format(gain))
                    #     print("loss {}".format(loss_cut))
                    #     # undetermined_assets = self.assets + gain

                    old_action = action if action != 2 else old_action

                elif self.assets > self.minimum_required_margin * 2:
                    old_action = None

            self.total_gain.append(undetermined_assets)
            try:
                if self.total_gain[-1] == self.total_gain[-2]:
                    self.total_gain[ -1] = undetermined_assets - undetermined_assets * 0.1
            except:
                pass

            self.losses.append(self.los_cut)
            # print(undetermined_assets)

            gain = 0
            self.growth_rate.append(np.log(undetermined_assets / self.initial_assets) * 100)
        self.assets = undetermined_assets

class Reward2(Reward):
    def reward(self, trend, high, low, action, leverage, atr, scale_atr):
        old_action = None
        position = None
        gain = 0
        self.los_cut = self.max_los_cut
        old_assets = self.assets

        self.losses = []
        undetermined_assets = self.assets
        self.positions = None
        self.minimum_required_margin = trend[0] * self.pip_cost / self.leverage


        for action, leverage, trend, idx, atr, scale_atr in zip(action, leverage, trend, range(len(high)), atr, scale_atr):

            if old_action == None and position == None and self.assets > self.minimum_required_margin * 2 and action != 2:
                self.minimum_required_margin = trend * self.pip_cost / (self.leverage * abs(leverage))
                lot = self.assets * self.available_assets_rate / self.minimum_required_margin * self.min_lots
                self.lot = int(lot * 10 ** 2) / (10 ** 2)

                self.positions = trend + self.spread if action == 0 else trend - self.spread
                self.tp = min(abs(self.max_los_cut), atr * self.pip_cost)
                self.los_cut = max(self.max_los_cut, -atr * self.pip_cost)
                old_action = action

            elif old_action != None:
                if self.positions != None:
                    if old_action == 0:
                        gain = (trend - self.positions) * self.pip_cost * self.lot * 100
                        loss = (low[idx - 1] - self.positions) * self.pip_cost * self.lot * 100
                    else:
                        gain = (self.positions - trend) * self.pip_cost * self.lot * 100
                        loss = (self.positions - high[idx - 1]) * self.pip_cost * self.lot * 100

                    loss = min(gain, loss)
                    tp = self.tp * self.lot * 100
                    loss_cut = self.los_cut * self.lot * 100
                    gain = loss_cut if loss <= loss_cut else gain
                    undetermined_assets = max(self.assets + gain, self.assets + loss_cut)

                    confirm = gain == loss_cut
                    buy = old_action != action and action != 2
                    if confirm or buy:
                        self.assets = undetermined_assets

                        if self.assets > self.minimum_required_margin * 2:
                            if action == 2:
                                self.positions = None
                                old_action = None
                            else:
                                self.minimum_required_margin = trend * self.pip_cost / (self.leverage * abs(leverage))
                                lot = self.assets * self.available_assets_rate / self.minimum_required_margin * self.min_lots
                                self.lot = int(lot * 10 ** 2) / (10 ** 2)
                                self.positions = trend + self.spread if action == 0 else trend - self.spread
                                self.tp = min(abs(self.max_los_cut), atr * self.pip_cost)
                                self.los_cut = max(self.max_los_cut, -atr * self.pip_cost)
                        else:
                            self.positions = None

                    # else:
                    #     print("gain {}".format(gain))
                    #     print("loss {}".format(loss_cut))
                    #     # undetermined_assets = self.assets + gain

                    old_action = action if action != 2 else old_action

                elif self.assets > self.minimum_required_margin * 2:
                    old_action = None

            self.total_gain.append(undetermined_assets)
            try:
                if self.total_gain[-1] == self.total_gain[-2]:
                    self.total_gain[ -1] = undetermined_assets - undetermined_assets * 0.1
            except:
                pass

            self.losses.append(self.los_cut)
            # print(undetermined_assets)

            gain = 0
            self.growth_rate.append(np.log(undetermined_assets / self.initial_assets) * 100)
        self.assets = undetermined_assets


class Reward3(Reward):
    def reward(self, trend, high, low, leverage, atr, scale_atr):
        old_action = None
        position = None
        gain = 0
        self.los_cut = self.max_los_cut
        self.lots = []
        old_assets = self.assets

        undetermined_assets = self.assets
        self.positions = None
        self.minimum_required_margin = trend[0] * self.pip_cost / self.leverage

        for leverage, trend, idx, atr, scale_atr in zip(leverage, trend, range(len(high)), atr, scale_atr):

            if self.assets > 0:
                if idx == 0:
                    self.minimum_required_margin = trend * self.pip_cost / (self.leverage * abs(leverage))
                    lot = self.assets * self.available_assets_rate / self.minimum_required_margin * self.min_lots
                    self.lot = int(lot * 10 ** 2) / (10 ** 2)
                    self.positions = trend + self.spread if leverage >= 0 else trend - self.spread
                    self.los_cut = max(self.max_los_cut, -atr * self.pip_cost)

                elif self.positions is not None:
                    if old_action == 0:
                        gain = (trend - self.positions) * self.pip_cost * self.lot * 100
                        loss = (low[idx - 1] - self.positions) * self.pip_cost * self.lot * 100
                    else:
                        gain = (self.positions - trend) * self.pip_cost * self.lot * 100
                        loss = (self.positions - high[idx - 1]) * self.pip_cost * self.lot * 100

                    gain = self.los_cut * self.lot * 100 if loss <= self.los_cut else gain

                    self.assets = max(self.assets + gain, self.assets + self.los_cut * self.lot * 100)

                    self.minimum_required_margin = trend * self.pip_cost / (self.leverage * abs(leverage))
                    lot = self.assets * self.available_assets_rate / self.minimum_required_margin * self.min_lots
                    if lot > 0:
                        self.lot = int(lot * 10 ** 2) / (10 ** 2)
                        self.positions = trend + self.spread if leverage >= 0 else trend - self.spread
                        self.los_cut = max(self.max_los_cut, -atr * self.pip_cost)
                    else:
                        self.positions = None

                old_action = 0 if leverage >= 0 else 1


            self.total_gain.append(self.assets)
            self.lots.append(self.lot)
            gain = 0
            self.growth_rate.append(np.log(self.assets / self.initial_assets) * 100)