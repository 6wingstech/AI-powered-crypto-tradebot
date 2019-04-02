import matplotlib.pyplot as plt
from slackclient import SlackClient
import requests

import os
import time


def display_chart(df, channel, base, coin, a, b=None, c=None):
	plotting = [a]
	if b != None:
		plotting.append(b)
	if c != None:
		plotting.append(c)

	plots = df[plotting].plot(subplots=False, figsize=(10, 6))
	titl = str(coin) + str(base) + ' 6 HR CHART'

	chartname = titl + '.jpg'
	plt.title(titl)
	plt.ylabel('Price')
	ax = plt.gca()
	ax.axes.get_xaxis().set_visible(False)

	plt.savefig(chartname)
	time.sleep(1)
	slack_file(chartname, channel)
	try:
		os.remove(chartname)
	except:
		pass
	plt.close('all')

def training_chart(df, name, a, b, c, channel):
	plotting = [a, b, c]

	plots = df[plotting].plot(subplots=False, figsize=(10, 6))

	chartname = name + '.jpg'
	plt.title(name)
	plt.ylabel('Score')
	ax = plt.gca()
	ax.axes.get_xaxis().set_visible(False)

	plt.savefig(chartname)
	time.sleep(1)
	slack_file(chartname, channel)
	plt.close('all')
	try:
		os.remove(chartname)
	except:
		pass

def backtesting_chart(df, coin, a, buys, sells):
	plotting = [a]

	plots = df[plotting].plot(subplots=False, figsize=(20, 15))
	titl = str(coin) + ' Backtesting'

	chartname = titl + '.jpg'
	plt.title(titl)
	plt.ylabel('Price')
	ax = plt.gca()
	ax.axes.get_xaxis().set_visible(False)

	if buys:
		for i in buys:
			length = i[0]
			price = i[1]
			message = i[2]
			plt.text(length, price, message,
				horizontalalignment='center',
				verticalalignment='center',
				color='green',
				fontsize=10)

	if sells:
		for i in sells:
			length = i[0]
			price = i[1]
			message = i[2]
			plt.text(length, price, message,
				horizontalalignment='center',
				verticalalignment='center',
				color='red',
				fontsize=10)


	plt.savefig(chartname)
	plt.close('all')

def readings_chart(df, file, coin, a, readings):
	plotting = [a]

	plots = df[plotting].plot(subplots=False, figsize=(30, 25))
	titl = str(coin) + ' Backtesting'

	chartname = titl + '.jpg'
	plt.title(titl)
	plt.ylabel('Price')
	ax = plt.gca()
	ax.axes.get_xaxis().set_visible(False)

	if readings:
		for i in readings:
			length = i[0]
			price = i[1]
			message = i[2]
			if message == 'sell':
				clr = 'red'
			elif message == 'buy':
				clr = 'green'
			else:
				clr = 'black'
			plt.text(length, price, message,
				horizontalalignment='center',
				verticalalignment='center',
				color=clr,
				fontsize=12)

	plt.savefig(file)
	plt.close('all')

