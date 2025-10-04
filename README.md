# F3 Crypto AI Trading Bot

The **F3 Crypto AI Trading Bot**, developed by **F3 AI Labs**, is an advanced, AI-powered cryptocurrency trading system focused on achieving optimal performance in the PEP/USDT market. Engineered with deep reinforcement learning (RL) algorithms, it dynamically adapts to real-time market changes and uses strategic risk management techniques to maximize profits. Its modular design and layered architecture provide a highly flexible framework that can be scaled and adapted to additional assets.

---

## Key AI-Driven Features

### Reinforcement Learning (RL) Foundation

This bot is built upon a reinforcement learning framework, allowing it to iteratively improve based on cumulative rewards from trading actions. The bot dynamically learns which strategies yield the highest returns by training in a custom environment tailored to the PEP/USDT trading pair. Each action—buy, sell, or hold—affects its understanding of market conditions and informs future trades, leading to progressively optimized decision-making.

- **Multi-Episode Training**: Each episode represents a simulated trading session where the bot evaluates its performance and adjusts based on reward signals. This episodic learning allows the bot to explore various trading scenarios, honing its responses to a wide range of market conditions.

### Deep Neural Network Model (`SingleAssetModel`)

The core decision-making engine of the bot is a single-head neural network model designed to evaluate and act on real-time trading signals. This model is lightweight yet effective, using a fully connected network with ReLU activation to process critical trading inputs and produce action probabilities.

- **Input Features**: The model takes in current asset price, PEP balance, and USDT balance—data that encapsulates the trading environment. This input structure enables the model to contextualize market movements with its own holdings, creating a feedback loop where actions are informed by both market conditions and account balance.
- **Output**: The model outputs action probabilities, representing the likelihood of the optimal action (buy, sell, or hold) at any given time. These probabilities are converted to actionable decisions, empowering the bot to execute market orders intelligently.

### Custom Trading Environment

Using OpenAI’s Gym as a base, the `MultiAssetTradingEnv` is a bespoke environment that aligns directly with cryptocurrency trading’s nuances. This environment facilitates a controlled and optimized setting for the reinforcement learning model to learn and adapt efficiently.

- **Environment Simulation**: Simulates realistic trading conditions, where every action affects the bot’s balance and market position. By tracking PEP and USDT balances and calculating total balance rewards, this environment mirrors the impacts of real-world crypto trading.
- **Reward System**: Custom rewards are based on cumulative balance growth, encouraging the model to maximize profit while penalizing losses, leading to a self-improving AI that constantly seeks higher rewards.

### Real-Time Decision-Making

The bot processes live market data, evaluates current conditions using pre-trained models, and makes real-time trading decisions. This AI-powered adaptability enables it to respond to market fluctuations as they occur, optimizing trades on the fly. Key algorithms include:

- **EMA Crossover Strategy**: Uses short- and long-term EMAs (Exponential Moving Averages) to detect buy/sell signals based on trend momentum. This is ideal for spotting shifts in market direction.
- **RSI-Based Readiness Detection**: Uses the Relative Strength Index (RSI) to measure market overbought/oversold levels. By dynamically adjusting the readiness threshold, the bot decides when to engage in higher-risk trades.

### Adaptive Risk Management

The bot incorporates pyramid risk management strategies to dynamically adjust risk per trade. This risk management technique scales exposure based on previous trades’ success, allowing it to take calculated risks during favorable trends and limit losses during adverse movements.

- **Incremental Risk Levels**: Uses a multi-tiered approach to gradually increase position sizes based on performance, ensuring the bot capitalizes on positive trends without overexposing its account.
- **Automatic Risk Adjustment**: Resets risk levels when a trade doesn’t perform as expected, providing an additional safeguard against potential losses.

---

## Project Structure and AI Components

1. **`f3_bot.py`** - **Trading Execution & Core Logic**
   - Executes live trades, manages cached historical and real-time data, and calculates trade rewards.
   - Integrates with XEGGEX API for real-time market data and order execution, feeding data to the neural network model.

2. **`multi_asset_trading_env.py`** - **Custom Reinforcement Learning Environment**
   - Provides an OpenAI Gym-compatible environment where the bot learns to buy, sell, or hold based on its cumulative reward function.
   - **Action Space**: 3 discrete actions (Hold, Buy, Sell).
   - **Observation Space**: Captures normalized values for PEP price, PEP balance, and USDT balance.

3. **`multi_head_model.py`** - **Neural Network Model**
   - The `SingleAssetModel` class handles prediction with a feedforward neural network.
   - **Architecture**: Uses a single shared layer and output layer, optimized for high-speed inference, ideal for time-sensitive financial applications.
   - **Reinforcement Signal**: The model continuously adapts based on cumulative reward tracking, improving as it processes more data.

4. **`train_multi_asset_rl_model.py`** - **Model Training and Optimization**
   - Handles the end-to-end training process using the custom environment and `SingleAssetModel`.
   - **Training Cycle**: For each episode, the bot explores different strategies, learning to maximize returns over 80 episodes.
   - **Optimizer**: Uses Adam optimizer to minimize loss functions during training, iteratively adjusting weights based on rewards.

---

## AI Strategies & Reinforcement Techniques

### Trading Strategies

- **RSI Strategy**: Uses the RSI indicator to detect market conditions:
  - **Buy Signal**: RSI < 30 (oversold).
  - **Sell Signal**: RSI > 70 (overbought).
- **EMA Strategy**: Uses EMA crossovers for trend-based trading:
  - **Buy Signal**: Short EMA > Long EMA.
  - **Sell Signal**: Short EMA < Long EMA.

### Reward-Driven Learning

The bot uses a reward system to reinforce profitable actions and discourage losses. The cumulative reward is calculated after each action, shaping the bot’s future responses and helping it distinguish between favorable and unfavorable market conditions.

---

## Performance Logging and Analytics

- **Trading Log**: Tracks every action taken, balance after each trade, and cumulative rewards in `trading.log`.
- **Real-Time Performance Tracking**: Displays cumulative rewards and current account balance after each action, allowing for an at-a-glance understanding of bot profitability.
- **Detailed Backtesting Reports**: Each backtest generates a `backtest_results.csv`, showing individual trade actions, profit per trade, and cumulative rewards.

---

## Future Enhancements

- **Multi-Asset Expansion**: Extend model to support additional assets, enabling diversified trading capabilities.
- **Advanced Indicator Suite**: Incorporate more indicators such as MACD, Stochastic Oscillator, and Bollinger Bands for richer decision-making.
- **Deep Reinforcement Learning (DRL)**: Experiment with DRL techniques like Proximal Policy Optimization (PPO) or Soft Actor-Critic (SAC) for enhanced AI decision-making.
- **Automated Hyperparameter Tuning**: Implement automated optimization for trading parameters, enabling the bot to self-optimize based on recent performance.

---

## Technical Insights

- **Neural Network Tuning**: The AI model is optimized for lightweight performance, designed to run efficiently even with high-frequency trading intervals. Each layer has been sized to balance inference speed and decision-making accuracy.
- **Custom Reinforcement Learning Environment**: This bespoke Gym environment enables rapid training cycles, ideal for reinforcement learning, where the bot can process a high volume of episodes in a short time frame.
- **Risk Mitigation Through Pyramid Strategy**: The adaptive pyramid strategy systematically increases risk during favorable conditions, maximizing gains while employing a fallback mechanism to reset risk after losses.

---

## License

This project is licensed under the MIT License. See `LICENSE` for more details.

---

Developed by **F3 AI Labs**, this README highlights the advanced AI capabilities, design, and planned advancements of the **F3 Crypto AI Trading Bot**, showcasing its sophisticated decision-making and adaptability to handle the dynamic cryptocurrency market.
