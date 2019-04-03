from _ta_nn import *

import pandas as pd
import numpy as np
import time
import sys
from datetime import datetime
from _calculations import slack_message, record_csv, record_trade, record_profit
import ast
from _graphs import *
from _trading_b import *
import csv
import os
import random

from _build_data import *
from _brain import *

class Bot:
    def __init__(self, base, pairs, max_position, status='on', kraken='on'):
        self._base_currency = base
        self._directory = 'D:/ai_trading/' + str(base.lower()) + '/'
        self._directory_backtest = 'D:/ai_trading/' + str(base.lower()) + '/backtesting/'
        self._directory_predictions = 'D:/ai_trading/' + str(base.lower()) + '/predictions/'
        self._btc_directory = 'D:/ai_trading/btc/'
        self._tokens = pairs
        self._start_time = int(time.time())
        self._interval = 200
        self._time_between = 15
        self._chart_interval = 7000
        self._k = a['bina']['jp']['apiKey']
        self._s = a['bina']['jp']['secret']
        self._max_position = max_position
        self._status = status
        self._actions_msgs = {'buy': 'Bought ', 'sell': 'Sold ', 'short': 'Shorted ', 'cover': 'Covered ', 'none': 'none'}

        self._execution_count_dict_directory = self._directory + 'execution_dict.csv'

        self._retrain_hours = ['02', '14']

        self._acc_target = 0.7
        self._krak_instructions_file = 'D:/ai_trading/k/_instructions.csv'
        self._krak_trading = kraken

        self._backtests = [[1, 3], [3, 1], [1, 1], [3, 3], [5, 5], [8, 8], [random.randint(1, 10), random.randint(1, 10)]]
        self._execution_count_dict = {}
        self._exit_count_dict = {}

        self._max_trade_size = int(max_position/15)
        self._max_sell_size = int(max_position/12)
        self._enabled_activity = ['buy']
        self._closing_activities = ['sell']
        self._trading_pairs = []

        self._5_decs = ['ADA', 'IOTA', 'XRP', 'XLM']

        self.stats_file = self._directory + str(base) + '_AI_BOT_STATS.csv'
        self.trades_file = self._directory + str(base) + '_AI_BOT_TRADES.csv'
        self.profit_file = self._directory + str(base) + '_AI_BOT_PROFIT.csv'

        if base == 'USD':
            self.slack_channel = 'ai-predictions'
        elif base == 'BTC':
            self.slack_channel = 'ai-predictions-btc'
        elif base == 'ETH':
            self.slack_channel = 'ai-predictions-eth'
        self.botname = 'AI BOT'

        self._trade_channel = 'ai-trading-jp'
        self._backtest_channel = 'ai-backtesting'
        self._results_channel = 'ai-algos'

    def _check_server(self):
        _s = 1
        while _s != 0:
            client = load_account(self._k, self._s)
            _s = check_bina_server(client)
            if _s != 0:
                print(str(datetime.now().strftime('%d/%m/%Y %H:%M:%S')))
                print('Binance servers offline..')
                print()
                time.sleep(600)
        print('Servers fine')

    def train_all(self, df, index, pair, model_file, _tf=None):
        if _tf == None:
            data_df = build_data(pair, slack='test')
        else:
            data_df = build_data(pair, _tf=_tf, slack='test')
        try:
            ff = self._directory_backtest + str(pair) + '_latest_data.csv'
            data_df.to_csv(ff, index=False)
            print('Data file saved.')
        except:
            pass

        _bot_name = str(df.at[index, 'Asset']) + '-' + str(self._base_currency) + ' AI'

        train_data, labels = create_train_data(data_df, 'result_24_48hr')
        _dict, model, accuracy, val_loss, train_df = train_nn(pair, train_data, labels)
        chart_name = str(pair) + ' Neural Network Training'
        training_chart(train_df, chart_name, 'Train Loss', 'Validation Loss', 'Accuracy', self._backtest_channel)
        df.at[index, 'Accuracy'] = round(accuracy*100, 2)
        df.at[index, 'Data_Dict'] = str(_dict)
        torch.save(model, model_file)
        train_msg = 'AI re-trained with ' + str(round(accuracy*100, 2)) + '% Accuracy and ' + str(round(val_loss, 5)) + ' Validation loss.'
        print(str(df.at[index, 'Asset']) + ' ' + train_msg)
        slack_message(train_msg, self._backtest_channel, _bot_name)
        print('Backtesting...')
        _trades, _profit, _win_rate = self._backtest(pair, model_file, _dict)

        df.at[index, 'Last_Trained'] = str(datetime.now().strftime('%d/%m/%Y %H:%M'))

        return df, _trades, _profit, _win_rate

    def _record_backtest_predictions(self, file, prediction, accuracy, price):
        try:
            with open(file) as f:
                numline = 10
        except:
            numline = 0
        entry = [prediction, accuracy, price]
        with open(file, 'a', newline='') as fp:
            wr = csv.writer(fp, dialect='excel')
            if numline == 0:
                wr.writerow(['prediction', 'accuracy', 'price'])
            wr.writerow(entry)
            fp.close()

    def _backtest(self, pair, model_file, _dict):
        try:
            backtest_df = build_data(pair, mode='test', _tf='3 months ago')
            model = torch.load(model_file)
            temp_backtest_file = self._directory_backtest + str(pair) + '_backtesting_temp.csv'
            try:
                os.remove(temp_backtest_file)
            except:
                pass
            print('Generating past predictions..')
            _bull = ['Bullish', 'Very Bullish']
            _bear = ['Bearish', 'Very Bearish']
            _doublechecks = []

            for index, row in backtest_df.iterrows():
                _input = create_input_for_backtest(backtest_df, index)
                _in = process_data(_dict, _input)
                prediction, accuracy, tensor = predict(model, _in)
                self._record_backtest_predictions(temp_backtest_file, prediction, accuracy, row['Close'])
                if (prediction in _bull) and (accuracy > self._acc_target):
                    _doublechecks.append([index, row['Close'], 'buy'])
                elif prediction in _bear:
                    _doublechecks.append([index, row['Close'], 'sell'])

            chart_file = str(pair) + ' raw predictions.jpg'
            readings_chart(backtest_df, chart_file, str(pair), 'Close', _doublechecks)
            slack_file(chart_file, self._backtest_channel)
            os.remove(chart_file)

            _results = {'entry': 0, 'exit': 0, 'trades': 0, 'profit': 0.0, 'win_rate': 0.0}
            for i in self._backtests:
                try:
                    _trade_count, _profit_count, _total_win_rate = self._backtest_trading(pair, temp_backtest_file, i[0], i[1], 2400)
                    if _trade_count >= 1:
                        if _results['entry'] == 0:
                            _results['entry'], _results['exit'], _results['trades'], _results['profit'], _results['win_rate'] = i[0], i[1], _trade_count, float(_profit_count), float(_total_win_rate)
                        else:
                            _profit_difference = ((float(_profit_count) - _results['profit'])/_results['profit']) * 100 if _results['profit'] > 0 else 100
                            _win_rate_difference = ((float(_total_win_rate) - _results['win_rate'])/_results['win_rate']) * 100 if _results['win_rate'] > 0 else 100
                            if (_profit_difference > 0) and (_win_rate_difference > 0):
                                _results['entry'], _results['exit'], _results['trades'], _results['profit'], _results['win_rate'] = i[0], i[1], _trade_count, float(_profit_count), float(_total_win_rate)
                            elif ((_profit_difference > 0) and (_profit_difference > abs(_win_rate_difference) and (_total_win_rate > 66))) or ((_win_rate_difference > 0) and (_win_rate_difference > abs(_profit_difference) and (_total_win_rate > 66))):
                                _results['entry'], _results['exit'], _results['trades'], _results['profit'], _results['win_rate'] = i[0], i[1], _trade_count, float(_profit_count), float(_total_win_rate)
                except:
                    pass

            self._execution_count_dict[pair] = _results['entry']
            self._exit_count_dict[pair] = _results['exit']
            algo_msg = str(_results['entry']) + ':' + str(_results['exit']) + ' algo activated. 50 day trade count: ' + str(_results['trades']) + ', return: ' + str(_results['profit']) + '%, win rate: ' + str(_results['win_rate']) + '%'
            print(algo_msg)
            _botname = pair.replace(self._base_currency, '') + ' AI' if self._base_currency != 'USD' else pair.replace('USDT', '') + ' AI'
            slack_message(algo_msg, self._results_channel, _botname)
            _trade_count, _profit_count, _total_win_rate = _results['trades'], _results['profit'], _results['win_rate']

            #os.remove(temp_backtest_file)
        except Exception as e:
            msg = 'Error in backtesting: ' + str(e)
            print(msg)
            slack_message(msg, 'errors', self.botname)
            _trade_count, _profit_count, _total_win_rate = 0, 0, 0

        return _trade_count, _profit_count, _total_win_rate

    def _backtest_trading(self, pair, backtest_file, entry_count, exit_count, dataframe_count):
        try:
            backtest_df = pd.read_csv(backtest_file)
            backtest_df = backtest_df[-dataframe_count:]

            _win_rate = 0
            _trade_count = 0
            _profit_count = 0
            entry_price = 0.0
            position = 'none'

            buys = []
            sells = []
            readings = []

            _count = 0

            _readings = []

            _bull = ['Bullish', 'Very Bullish']
            _bear = ['Bearish', 'Very Bearish']

            for index, row in backtest_df.iterrows():
                if position == 'none':
                    if (backtest_df.at[index, 'prediction'] in _bull) and (backtest_df.at[index, 'accuracy'] > self._acc_target):
                        _count += 1

                        if _count >= entry_count:
                            readings.append([index, float(row['price']), 'buy'])
                            position = 'long'
                            entry_price = float(row['price'])
                            _count = 0

            
                elif position == 'long':
                    if backtest_df.at[index, 'prediction'] in _bear:
                        _count += 1
                        if (_count >= exit_count):
                            _profit = round(((float(row['price']) - entry_price) / entry_price) * 100, 2)
                            if _profit > 0:
                                _win_rate += 1
                            readings.append([index, float(row['price']), 'sell'])
                            _trade_count += 1
                            _profit_count += _profit
                            position = 'none'
                            _count = 0


            chart_file = str(pair) + ' backtesting 50 days.jpg'
            readings_chart(backtest_df, chart_file, str(pair), 'price', readings)

            try:
                _total_win_rate = round((float(_win_rate)/float(_trade_count))*100, 2)
                _msg = str(entry_count) + ':' + str(exit_count) + ' Algo Total trades: ' + str(_trade_count) + ', Total gain: ' + str(round(_profit_count, 2)) + '%, Win rate: ' + str(_total_win_rate) + '%'
            except Exception as e:
                print(e)

            _bot_name = str(pair) + ' Backtesting 50 Days (Long only) Results'
            slack_message(_msg, self._backtest_channel, _bot_name)
            slack_file(chart_file, self._backtest_channel)
            os.remove(chart_file)
        except Exception as e:
            msg = 'Error in backtesting trades: ' + str(e)
            print(msg)
            #slack_message(msg, 'errors', self.botname)

        return _trade_count, round(_profit_count, 2), _total_win_rate


    def _get_trade_qty(self, client, df, index, action, asset, current_price):
        avg_price=current_price
        qty = 0
        if self._status == 'on':
            pair = (str(asset) + 'USDT') if self._base_currency == 'USD' else (str(asset) + str(self._base_currency))
            positions = ast.literal_eval(df.at[index, 'Positions'])

            balance = 0
            if (action in self._enabled_activity) and (pair in self._trading_pairs):
                base = 'USDT' if self._base_currency == 'USD' else self._base_currency
                balance = get_balance(client, base)
                if (balance > (self._max_position/4)) and (round(positions['trade_amount'], 2) == 0):
                    if len(positions['orders_open']) == 0:
                        if balance > (self._max_position/2):
                            qty = float(round(int(self._max_position/2)/float(current_price), 2)) if asset not in self._5_decs else round(int(self._max_position/2)/float(current_price))
                        else:
                            qty = float(round(int(self._max_position/4)/float(current_price), 2)) if asset not in self._5_decs else round(int(self._max_position/4)/float(current_price))
                    elif float(positions['qty'])*float(current_price) < self._max_position * 0.8:
                        qty = float(round(int(self._max_position/4)/float(current_price), 2)) if asset not in self._5_decs else round(int(self._max_position/4)/float(current_price))
                    else:
                        print('Cannot ' + str(action) + ' anymore: max position')
                        qty = 0.0
                    positions['trade_amount'] = qty
                else:
                    print('Insufficient funds to trade.')

            elif action in self._closing_activities:
                if round(positions['trade_amount'], 2) == 0:
                    qty = float(positions['qty'])
                    positions['trade_amount'] = qty

            df.at[index, 'Positions'] = str(positions)

        return df, qty

    def _execute_trade(self, client, df, index, pair, current_price):
        orderId = ''
        positions = ast.literal_eval(df.at[index, 'Positions'])

        if positions['trade_amount'] > 0:
            try:
                max_order_size = float(self._max_trade_size/float(current_price)) if positions['action'] in self._enabled_activity else float(self._max_sell_size/float(current_price))
                _trade_qty = min(float(positions['trade_amount']), max_order_size)
                _trade_qty = round(_trade_qty, 2) if df.at[index, 'Asset'] not in self._5_decs else round(_trade_qty)
                orderId = market_order(client, pair, positions['action'], _trade_qty)
                positions['orders_open'].append(str(orderId)) if positions['action'] in self._enabled_activity else positions['orders_close'].append(str(orderId))

            except Exception as e:
                err_msg = 'Error with order execution: ' + str(orerId) + ' : ' + str(e)
                print(err_msg)
                slack_message(err_msg, 'errors', self.botname)
                orderId = ''
                positions['trade_amount'] = 0
                self._check_server()

        df.at[index, 'Positions'] = str(positions)

        return orderId, df

    def _check_order(self, client, df, index, asset, pair, orderId):
        positions = ast.literal_eval(df.at[index, 'Positions'])
        _total, _qty, avg_price = 0.0, 0.0, 0.0
        try:
            avg_price, _total, _qty = check_order(client, pair, orderId)
            _qty = round(_qty, 2) if asset not in self._5_decs else round(_qty)
            print(str(positions['action']) + ' ' + str(_qty) + ' ' + str(pair) + ' @ ' + str(avg_price) + ' for ' + str(_total))
            positions['trade_amount'] -= float(_qty)

        except Exception as e:
            err_msg = 'Error with order checking: ' + str(orderId) + ' : ' + str(e)
            print(err_msg)
            slack_message(err_msg, 'errors', self.botname)
            self._check_server()

        if positions['action'] in self._enabled_activity:
            positions['qty'] += _qty
        else:
            positions['qty'] -= _qty

        try:
            self._restock_bnb()
        except Exception as e:
            msg = 'Couldnt restock BNB: ' + str(e)
            print(msg)
            slack_message(msg, 'errors', self.botname)
            self._check_server()

        if _qty > 0:
            record_trade(self.trades_file, str(asset), positions['action'], avg_price, _qty, _total, orderId)
            print('Trade recorded')

        df.at[index, 'Positions'] = str(positions)

        return df, avg_price, _total, _qty

    def _execute(self, df, index, asset, pair):
        positions = ast.literal_eval(df.at[index, 'Positions'])
        _total = 0
        _qty = 0
        avg_price = 0
        orderId = 'None'
        current_price = get_price(pair)
        client = load_account(self._k, self._s)
        _vwap = recent_vwap(pair)
        if ((float(current_price) < float(_vwap)) and (positions['action'] == 'buy')) or ((float(current_price) > float(_vwap)) and (positions['action'] == 'sell')):
            orderId, df = self._execute_trade(client, df, index, pair, float(current_price))
            time.sleep(3)
        else:
            print('Trade skipped. Waiting for a better price. vwap: ' + str(_vwap) + ', price: ' + str(current_price))

        if (orderId != '') and (orderId != 'None'):
            df, avg_price, _total, _qty = self._check_order(client, df, index, asset, pair, orderId)
            self._save_state(df)
            positions = ast.literal_eval(df.at[index, 'Positions'])

        if avg_price > 0:
            mm = self._actions_msgs[positions['action']]
            nonusd_string = ' with ' + str(self._base_currency) + ' @ '
            usd_string = ' @ $'
            filler_string = usd_string if self._base_currency == 'USD' else nonusd_string

            round2 = ['BTCUSDT', 'ETHUSDT', 'LTCUSDT']
            avg_price = round(avg_price, 2) if pair in round2 else round(avg_price, 5)
            _total = round(_total, 2) if 'USDT' in pair else round(_total, 5)
            new_msg = mm + str(_qty) + ' ' + str(asset) + filler_string + str(avg_price) + ' (Total: ' + str(_total) + ')'

            p_l_msg = ''
            if (positions['action'] in self._enabled_activity) and (float(positions['qty']) > 0):
                positions['position'] = 'long' if positions['action'] == 'buy' else 'short'

            elif positions['action'] in self._closing_activities:
                if ((positions['qty'] < 0.1) and (asset != 'BTC')) or ((positions['qty'] < 0.01) and (asset == 'BTC')):
                    print('Trade closed')
                    p_l = self._get_pl(pair, positions['orders_open'], positions['orders_close'])
                    positions['orders_open'] = []
                    positions['orders_close'] = []
                    p_l = round(p_l, 2) if 'USDT' in pair else round(p_l, 5)
                    p_l_msg = ' Profit: $' + str(p_l) if 'USDT' in pair else ' Profit: ' + str(p_l)
                    print(p_l_msg)

                    positions['qty'] = 0
                    positions['position'] = 'none'
                    positions['confidence'] = 0.0
                    positions['action'] = 'none'
                elif (positions['qty'] > 0) and (self._base_currency == 'USD'):
                    try:
                        _current_balance = get_balance(client, asset)
                        print('Remaining ' + str(asset) + ': ' + str(_current_balance))
                        if (_current_balance*float(current_price)) < 5:
                            print('Trade closed')
                            p_l = self._get_pl(pair, positions['orders_open'], positions['orders_close'])
                            positions['orders_open'] = []
                            positions['orders_close'] = []
                            p_l = round(p_l, 2) if 'USDT' in pair else round(p_l, 5)
                            p_l_msg = ' Profit: $' + str(p_l) if 'USDT' in pair else ' Profit: ' + str(p_l)
                            print(p_l_msg)

                            positions['qty'] = 0
                            positions['position'] = 'none'
                            positions['confidence'] = 0.0
                            positions['action'] = 'none'
                    except Exception as e:
                        print('Couldnt check balance: ' + str(e))

            df.at[index, 'Positions'] = str(positions)
            self._save_state(df)

            if _qty > 0:
                new_msg = new_msg + p_l_msg
                slack_message(new_msg, self._trade_channel, self.botname)

            if round(float(positions['trade_amount']), 1) == 0:
                chart_data_df = load_dataframe_1(pair, '24 hours ago')
                chart_data_df = chart_data_df[chart_data_df.Timestamp > (int(time.time())*1000)-21600000]
                display_chart(chart_data_df, self._trade_channel, self._base_currency, str(pair)[0:3], 'Close')

        return df

    def _get_pl(self, pair, open_list, close_list):
        client = load_account(self._k, self._s)
        p_l, _return = get_p_l_for_trade(client, pair, open_list, close_list)
        record_profit(self.profit_file, pair, p_l, _return)
        return p_l

    def _restock_bnb(self):
        client = load_account(self._k, self._s)
        bnb_balance = get_balance(client, 'BNB')
        if bnb_balance < 1.5:
            pair = 'BNB' + 'USDT' if str(self._base_currency) == 'USD' else 'BNB' + str(self._base_currency)
            orderId = market_order(client, pair, 'buy', 2)
            time.sleep(3)
            avg_price, cost, qty = check_order(client, pair, orderId)
            record_profit(self.profit_file, 'BNB', cost*-1, 0)
            print('Bought ' + str(qty) + ' BNB for commissions.')

    def _save_state(self, df):
        try:
            df.to_csv(self.stats_file, index=False)
        except:
            pass

    def _most_recent_position(self, pair):
        client = load_account(self._k, self._s)
        _trading_off_on = toggle_trade_based_on_recent_trades(client, pair)
        return _trading_off_on

    def start(self):
        self._check_server()

        df = pd.DataFrame(columns=['Asset', 'Algo_Action', 'Count', 'Positions', 'Updated', 'Last_Trained', 'Accuracy', 'Backtest', 'Chart_Update', 'Data_Dict'])

        token_list = np.asarray(self._tokens)
        df['Asset'] = token_list

        # It will try to load from a file if already exists (i.e. when the bot is restarted)
        try:
            df = pd.read_csv(self.stats_file)

        except:
            for index, row in df.iterrows():
                df.at[index, 'Algo_Action'] = 'None'
                df.at[index, 'Count'] = 0
                df.at[index, 'Count_'] = 0
                df.at[index, 'Positions'] = "{'position': 'none', 'qty': 0.0, 'confidence': 0.0, 'orders_open': [], 'orders_close': [], 'trade_amount': 0.0, 'action': 'none', 'validation': 0}"
                df.at[index, 'Updated'] = str(datetime.now().strftime('%d/%m/%Y %H:%M'))
                df.at[index, 'Last_Trained'] = 'None'
                df.at[index, 'Accuracy'] = 0.0
                df.at[index, 'Backtest'] = "{'trades': 0, 'win_rate': 0.0, 'gain': 0.0, 'retrain_count': 0, 'retrain': 'no'}"
                df.at[index, 'Chart_Update'] = int(time.time())
                df.at[index, 'Data_Dict'] = 'None'

        df = df.astype({"Algo_Action": 'str', "Last_Trained": 'str'})

        try:
            exe_df = pd.read_csv(self._execution_count_dict_directory)
            self._execution_count_dict = ast.literal_eval(exe_df.at[0, 'dict'])
            self._exit_count_dict = ast.literal_eval(exe_df.at[1, 'dict'])
        except:
            pass

        print('Bot started.')
        print('Trading: ' + str(self._status))
        for index, row in df.iterrows():
            if df.at[index, 'Last_Trained'] != 'None':
                pair = str(row['Asset']) + self._base_currency if self._base_currency != 'USD' else str(row['Asset']) + 'USDT'
                self._trading_pairs.append(pair)
        print('Enabled pairs: ' + str(self._trading_pairs))
        print()

        while time.time() < 9e20:
            hour_of_day = datetime.now().strftime('%H')
            current_minute = datetime.now().strftime('%M')
            checkpoints = ['00', '15', '30', '45'] if self._base_currency == 'USD' else ['59', '14', '29', '44']
            retrain_times = ['01', '02', '03', '16', '17', '18', '31', '32', '33', '46', '47', '48']

            _15_min_predictions = []
            _krak_instructions = {}
            _update_krak = ''
            for index, row in df.iterrows():
                position = ast.literal_eval(df.at[index, 'Positions'])
                backtest = ast.literal_eval(df.at[index, 'Backtest'])
                pair = str(row['Asset']) + self._base_currency if self._base_currency != 'USD' else str(row['Asset']) + 'USDT'

                # TRADE EXECUTION
                if ((round(position['trade_amount'], 2) < 0.1) and (row['Asset'] != 'BTC')) or ((round(position['trade_amount'], 2) < 0.01) and (row['Asset'] == 'BTC')):
                    position['trade_amount'] = 0
                    if (position['action'] == 'sell') or (position['qty'] == 0):
                        position['position'] = 'none'
                        position['action'] = 'none'
                    df.at[index, 'Positions'] = str(position)
                    self._save_state(df)
                    position = ast.literal_eval(df.at[index, 'Positions'])

                if round(position['trade_amount'], 2) > 0:
                    if (int(time.time()) < int(position['validation']) + 1500) or (position['action'] in self._closing_activities):
                        df = self._execute(df, index, str(row['Asset']), pair)
                        time.sleep(20)
                    elif int(time.time()) > int(position['validation']) + 1500:
                        position['validation'] = 0
                        position['action'] = 'none'
                        position['trade_amount'] = 0
                        df.at[index, 'Positions'] = str(position)
                        self._save_state(df)
                        position = ast.literal_eval(df.at[index, 'Positions'])
                elif position['action'] != 'none':
                    position['action'] = 'none'

                model_file = self._directory + str(row['Asset']) + ' AI.pt'
                if (len(self._trading_pairs) == 0) or (os.path.isfile(model_file) == False):
                    df.at[index, 'Last_Trained'] = 'None'

                if ((current_minute in checkpoints) and (datetime.timestamp(datetime.strptime(df.at[index, 'Updated'], '%d/%m/%Y %H:%M')) + self._interval < int(time.time()))) or (df.at[index, 'Last_Trained'] == 'None'):
                    if df.at[index, 'Last_Trained'] != 'None':
                        print()
                        print(str(datetime.now().strftime('%d/%m/%Y %H:%M:%S')))
                        print()
                        print('Asset: ' + str(row['Asset']))
                        print('Backtesting 50 day trade count: ' + str(backtest['trades']) + ', win rate: ' + str(backtest['win_rate']) + '%, estimated gain: ' + str(backtest['gain']) + '%')
                        print()
                    _dict3 = ''

                    if backtest['retrain'] == 'no':
                        try:
                            _cur_date = datetime.timestamp(datetime.strptime(df.at[index, 'Last_Trained'], '%d/%m/%Y %H:%M'))
                        except Exception as e:
                            print('Training new model..')
                            _cur_date = 0
                        if ((_cur_date + 21000 < int(time.time())) and (hour_of_day in self._retrain_hours)) or (df.at[index, 'Last_Trained'] == 'None'):
                            df, _trades, _profit, _win_rate = self.train_all(df, index, pair, model_file)
                            backtest['retrain_count'], backtest['trades'], backtest['win_rate'], backtest['gain'] = 0, _trades, _win_rate, _profit
                            df.at[index, 'Backtest'] = str(backtest)
                            if pair not in self._trading_pairs:
                                self._trading_pairs.append(pair)
                                #added_msg = 'Trading turned on for ' + str(pair)
                                #slack_message(added_msg, self.slack_channel, self.botname)
                            df.at[index, 'Updated'] = str(datetime.now().strftime('%d/%m/%Y %H:%M'))
                            model3 = torch.load(model_file)
                            print('AI loaded.')
                            _dict3 = ast.literal_eval(df.at[index, 'Data_Dict'])
                            self._save_state(df)
                        else:
                            model3 = torch.load(model_file)
                            print('AI loaded.')
                            _dict3 = ast.literal_eval(df.at[index, 'Data_Dict'])
                            print('Dictionaries loaded.')

                    elif current_minute in retrain_times:
                        print('Retraining ' + str(pair) + ' AI model')
                        df, _trades, _profit, _win_rate = self.train_all(df, index, pair, model_file, _tf=backtest['retrain'])
                        backtest['trades'], backtest['win_rate'], backtest['gain'], backtest['retrain'] = _trades, _win_rate, _profit, 'no'
                        df.at[index, 'Backtest'] = str(backtest)
                        if pair not in self._trading_pairs:
                            self._trading_pairs.append(pair)
                            added_msg = 'Trading turned on for ' + str(pair)
                            slack_message(added_msg, self._results_channel, self.botname)
                        df.at[index, 'Updated'] = str(datetime.now().strftime('%d/%m/%Y %H:%M'))
                        model3 = torch.load(model_file)
                        print('AI loaded.')
                        _dict3 = ast.literal_eval(df.at[index, 'Data_Dict'])
                        self._save_state(df)        

                    if df.at[index, 'Last_Trained'] != 'None':
                        if (float(backtest['win_rate']) < 65) or (float(backtest['gain']) < 3) or (int(backtest['trades']) == 0):
                            if pair in self._trading_pairs:
                                self._trading_pairs.remove(pair)
                                remove_msg = 'Trading turned off on ' + str(pair) + ' due to poor backtesting results.'
                                print(remove_msg)
                                slack_message(remove_msg, self._results_channel, self.botname)

                            df.at[index, 'Last_Trained'] = 'None'
                            if backtest['retrain_count'] == 0:
                                backtest['retrain'] = '5 months ago'
                            elif backtest['retrain_count'] == 1:
                                backtest['retrain'] = '6 months ago'
                            elif backtest['retrain_count'] > 1:
                                backtest['retrain'] = 'no'

                            backtest['retrain_count'] += 1
                            df.at[index, 'Backtest'] = str(backtest)
                            self._save_state(df)

                    # PREDICT
                    current_price = 0
                    if _dict3 != '':
                        _accuracy3 = 0.0
                        prediction3 = ''
                        try:
                            new_data_df = build_data(pair, mode='predict')
                            prediction3, _accuracy3, tensor3 = get_prediction(_dict3, model3, pair, new_data_df)

                            _bot_name = str(row['Asset']) + str(self._base_currency) + ' PREDICTION'

                            current_price = get_price(pair)

                            strin = ' ($' if self._base_currency == 'USD' else ' ('

                            _pred_msg = prediction3
                            if float(_accuracy3) < 0.85:
                                prediction3 = 'No edge'

                            msg = str(row['Asset']) + ' ' + str(prediction3) + strin + str(current_price) + ')'
                            print_msg = str(row['Asset']) + ' ' + str(_pred_msg) + strin + str(current_price) + ')'
                            print(print_msg + ' Accuracy: ' + str(round(_accuracy3, 4)))
                            _15_min_predictions.append(msg)
                            #slack_message(msg, self.slack_channel, _bot_name)

                            df.at[index, 'Updated'] = str(datetime.now().strftime('%d/%m/%Y %H:%M'))
                            self._save_state(df)

                            filename_csv = self._directory_predictions + str(row['Asset']) + '_predictions.csv'
                            record_csv(filename_csv, str(row['Asset']), tensor3, str(current_price))
                            _current_algo_action = 'none'

                            try:
                                # GET TRADE SPECS
                                _bull = ['Very Bullish', 'Bullish']
                                _bear = ['Very Bearish', 'Bearish']

                                print('Position: ' + position['position'] + ' (' + str(position['qty']) + ')')

                                if position['position'] == 'none':
                                    if (prediction3 in _bull) and (float(_accuracy3) > self._acc_target):
                                        _current_algo_action = 'buy'
                                        df.at[index, 'Count'] = int(df.at[index, 'Count']) + 1

                                elif position['position'] == 'long':
                                    if prediction3 in _bear:
                                        _current_algo_action = 'sell'
                                        df.at[index, 'Count_'] = int(df.at[index, 'Count_']) + 1
                                    elif (prediction3 in _bull) and (float(_accuracy3) > self._acc_target):
                                        _current_algo_action = 'buy'
                                        df.at[index, 'Count'] = int(df.at[index, 'Count']) + 1


                                df.at[index, 'Algo_Action'] = _current_algo_action
                                self._save_state(df)

                            except Exception as e:
                                e_msg = 'Problem with calculating trade specs: ' + str(e)
                                print(str(e_msg))
                                slack_message(e_msg, 'errors', self.botname)
                                self._check_server()

                        except Exception as e:
                            print('Problem with making prediction: ' + str(e))


                        try:
                            # Calculate Action
                            _count_var = self._execution_count_dict[pair] if pair in self._execution_count_dict else 1
                            _exit_count = self._exit_count_dict[pair] if pair in self._exit_count_dict else 1
                            action = 'none'
                            if (df.at[index, 'Count'] >= _count_var) or (df.at[index, 'Count_'] >= _exit_count):
                                client = load_account(self._k, self._s)
                                trade_switch = recent_trade(client, pair, 3600)
                                if trade_switch == 'on':
                                    action = str(df.at[index, 'Algo_Action'])
                                    position['action'] = str(df.at[index, 'Algo_Action'])
                                    position['validation'] = int(time.time())
                                    df.at[index, 'Count'] = 0
                                    df.at[index, 'Count_'] = 0
                                    df.at[index, 'Algo_Action'] = 'none'
                                    df.at[index, 'Positions'] = str(position)
                                    self._save_state(df)
                                else:
                                    print('Trade skipped due to too frequent trading.')
                                    df.at[index, 'Count'] = 0
                                    df.at[index, 'Count_'] = 0
                                    df.at[index, 'Algo_Action'] = 'none'
                                    self._save_state(df)
                            if action != 'none':
                                _krak_instructions[row['Asset']] = {'timestamp': int(time.time()), 'action': str(action)}
                                client = load_account(self._k, self._s)
                                df, _trade_amount = self._get_trade_qty(client, df, index, action, str(row['Asset']), current_price)
                                self._save_state(df)

                                # PAPER TRADES
                                position = ast.literal_eval(df.at[index, 'Positions'])
                                if _trade_amount == 0:
                                    print()
                                    print('Paper trading: ' + str(action))
                                    record_if = ['short', 'buy']
                                    close_if = ['cover', 'sell']
                                    message = ''
                                    if (position['position'] == 'none') and (action in record_if):
                                        position['position'] = 'long' if action == 'buy' else 'short'
                                        filler_ = ' @ $' if self._base_currency == 'USD' else ' @ ' 
                                        message = self._actions_msgs[action] + str(row['Asset']) + filler_ + str(current_price)

                                    if action in close_if:
                                        position['position'] = 'none'
                                        position['confidence'] = 0.0
                                        position['action'] = 'none'
                                        df.at[index, 'Positions'] = str(position)
                                        self._save_state(df)
                                        filler_ = ' @ $' if self._base_currency == 'USD' else ' @ ' 
                                        message = self._actions_msgs[action] + str(row['Asset']) + filler_ + str(current_price)

                                    if message != '':
                                        if self._base_currency != 'USD':
                                            message = message + ' (' + str(self._base_currency) + ')'
                                        #slack_message(message, 'ai-trading', self.botname)
                                        record_trade(self.trades_file, str(row['Asset']), action, current_price, 0, 0, '')
                                        print('Paper trade recorded: ' + str(action) + ' ' + str(row['Asset']) + ' @ ' + str(current_price))

                                    df.at[index, 'Positions'] = str(position)
                                    self._save_state(df)
                                else:
                                    print()
                                    print('Trade action: ' + str(action) + ' ' + str(_trade_amount))

                        except Exception as e:
                            e_msg = 'Problem with calculating action: ' + str(e)
                            print(str(e_msg))
                            slack_message(e_msg, 'errors', self.botname)
                            self._check_server()

                    _update_krak = 'yes' if self._krak_trading == 'on' else 'no'

                if (int(row['Chart_Update']) < int(time.time())) and (current_minute == '00'):
                    chart_data_df = load_dataframe_1(pair, '24 hours ago')
                    chart_data_df = chart_data_df[chart_data_df.Timestamp > (int(time.time())*1000)-21600000]
                    display_chart(chart_data_df, self.slack_channel, self._base_currency,  row['Asset'], 'Close')
                    df.at[index, 'Chart_Update'] = int(time.time()) + self._chart_interval
                    print('Chart updated.')
                    self._save_state(df)

            if _15_min_predictions:
                _botname = str(hour_of_day) + ':' + str(current_minute) + ' PREDICTIONS'
                for i in _15_min_predictions:
                    if 'Very Bullish' in i:
                        i = i.replace('Very Bullish', 'Strong Buy')
                    elif 'Bullish' in i:
                        i = i.replace('Bullish', 'Buy')
                    elif 'Very Bearish' in i:
                        i = i.replace('Very Bearish', 'Strong Sell')
                    elif 'Bearish' in i:
                        i = i.replace('Bearish', 'Sell')
                    slack_message(str(i), self.slack_channel, _botname)

            #send instructions to kraken
            try:
                exe_df = pd.DataFrame([[str(self._execution_count_dict)], [str(self._exit_count_dict)]], columns=['dict'])
                exe_df.to_csv(self._execution_count_dict_directory, index=False)
            except:
                pass

            if _update_krak == 'yes':
                df_krak = pd.DataFrame([str(_krak_instructions)], columns=['action'])
                try:
                    df_krak.to_csv(self._krak_instructions_file, index=False)
                    print('Kraken instructions updated.')
                except Exception as e:
                    print('Couldnt save Kraken instructions: ' + str(e))

            time.sleep(self._time_between)

