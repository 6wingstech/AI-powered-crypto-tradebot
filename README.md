# AI-powered-crypto-tradebot
A trade bot for cryptocurrencies powered by machine learning

DESCRIPTION

This is a trade bot for crypto currencies developed with an LSTM neural network. It is currently used to manage a live portfolio.

The entire trade system consists of 2 parts. The first part is AI assisted, and it is used to make predictions on the future outlook of the market. The second part processes those predictions and decides what actions to take (buy, sell etc.). Currently, it only takes long positions and does not short, but that can be easily added. The short side has been disabled only due to poor backtesting results. 

This repo only contains the AI/prediction part of the bot and the trading part is disabled. Messaging to slack is also disabled.

The system is currently being operated with a USD base, however it can be set to operate with a BTC base (LTC/BTC, ETH/BTC, etc.) or ETH base. Those should run without issue, albeit I have not yet tested results yet. I am simply using a USD base because I prefer having a fiat base currency.

The trade bot works on the 30-min time frame, and positions are held for an average of 1-5 days. Over the span of 50 days, it should do on average 5-15 trades per pair.

Data that is fed into the AI is taken from binance. No API key is needed to get this data so the system should function as is, without an API key. It should be noted that binance USD pairs are based in USDT (or TUSD, PAX, etc. but USDT has most liquidity and pairs). Therefore, if you are trading with fiat USD (and not USDT) understand that there may be a difference between USD and USDT. Although they are designed to have equal value, they don't always trade 1:1. 

The system (predictions only) can be run by running the run.py file after installing dependencies. If you do not have a GPU enabled computer, switch the cuda() commands to cpu(), else get an error on the neural network training part.

HOW THE AI WORKS

The bot currently trains on the past 4 months of data. 4 months has been tested to be optimal for me at this point in time, but a longer time frame is welcome and encouraged.

The AI is re-trained twice a day at 2am and 2pm so the model is constantly changing and current.

The bot will post all relevent activity onto a designated slack channel. The first thing it will post is a chart that looks like the one below. OHLC data is taken from the source (live data) for the pair and calculations are made. The chart posted simply shows how it is labelling the data (where it labels a buy and sell) so it can be visualized and confirmed. The neural network will begin training on that data.

The training takes about 2-4 minutes. The training stats (validation loss, training loss, accuracy) are posted in chart form to the same slack channel.

After the model is saved, the model is used to backtest against the past 50 days of data. The challenge here (if trading enabled) is to adjust how sensitive the trading side should be with regards to the predictions. Should the system trigger a buy after the first 'buy' reading? Should it wait for 10 readings? Typically, less volatile pairs like BTC/USD have shown better results with a less sensitive setting (more confirmations before action) and volatile pairs like XRP/USD work better with higher sensitivity.

The system will backtest using 6 different settings, such as a (1 'buy' reading to trigger a 'buy': 1 'sell' reading to trigger a sell, or 5 readings to trigger action) and will choose the one with the best backtesting performance to use for trading. Again, this setting may change every time the model is re-trained (twice a day) as the backtesting dictates. 

Backtest performance is measured by 3 categories; trade count (over past 50 days), return (% gain/loss), and win rate. 

If either return or win rate is unsatisfactory (default: win rate less than 65% or return less than 5%) then the model will re-train using a different timeframe; preset to 6 months, and then 8 months if still not satisfactory. A model that does not pass the backtesting will still make predictions, but trading will be disabled for the pair until the next time it re-trains to a satisfactory result.

PREDICTION

Predictions are made every 15 minutes at 00, 15, 30, and 45 on the clock. Predictions will be announced on the designated slack channel.

Actions are calculated based on those predictions and the settings for the trade bot described above. Orders will be placed in small increments over the course of the next 15 - 30 min to establish the desired position. This is so that the orders don't move the market and is necessary when taking larger positions, and also to space out the average price. I typically limit market orders to $3k usd value, even on very liquid exchanges. 

Predictions will be one of the following 6: Strong sell, sell, neutral, strong buy, buy, no edge. 'No edge' is what is displayed when the confidence level for the prediction is below the threshold set in the settings, default to 70%. 
