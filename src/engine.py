import xlrd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import containers as sim


def sim_engine():
	# This function will collect data and feed it into the simulation engine
	# side functionality is to allow multiprocessing to handle multiple
	# simultaneous simulations at once

	# Manually import three  features for now
	# 1. Federal Reserve Rate (FFR)
	# 2. Gold
	# 3. Inflation

	# Get Federal Reserve Rate data
	ffr_book = xlrd.open_workbook('FFR.xlsx')
	# Get first sheet
	ffr_sheet = ffr_book.sheet_by_index(0)
	# Get date column
	ffr_date = ffr_sheet.col(0)
	# Convert date to datetime obj
	ffr_date_datetime = [xlrd.xldate_as_datetime(date.value, ffr_book.datemode) \
						for date in ffr_date[1:]]
	# Convert date to unix time
	ffr_date_unix = np.array([date.replace(tzinfo=datetime.timezone.utc).timestamp() \
					for date in ffr_date_datetime])
	# Get FFR column
	ffr_rate = ffr_sheet.col(1)
	# Convert rate to int
	ffr_rate =np.array([rate.value for rate in ffr_rate[1:]])

	# Get Gold data
	gld_book = xlrd.open_workbook('GLD.xlsx')
	# Get first sheet
	gld_sheet = gld_book.sheet_by_index(0)
	# Get date column
	gld_date = gld_sheet.col(0)
	# Convert date to datetime obj
	gld_date_datetime = [xlrd.xldate_as_datetime(date.value, gld_book.datemode) \
						for date in gld_date[1:]]
	# Convert date to unix time
	gld_date_unix = np.array([date.replace(tzinfo=datetime.timezone.utc).timestamp() \
					for date in gld_date_datetime])
	# Get Gold RATE column
	gld_rate = gld_sheet.col(1)
	# Convert rate to int
	gld_rate = np.array([rate.value for rate in gld_rate[1:]])


	# Get INF data
	inf_book = xlrd.open_workbook('INF.xlsx')
	# Get first sheet
	inf_sheet = inf_book.sheet_by_index(0)
	# Get date column
	inf_date = inf_sheet.col(0)
	# Convert date to datetime obj
	inf_date_datetime = [xlrd.xldate_as_datetime(date.value, inf_book.datemode) \
						for date in inf_date[1:]]
	# Convert date to unix time
	inf_date_unix = np.array([date.replace(tzinfo=datetime.timezone.utc).timestamp() \
					for date in inf_date_datetime])
	# Get Gold RATE column
	inf_rate = inf_sheet.col(1)
	# Convert rate to int
	inf_rate = np.array([rate.value for rate in inf_rate[1:]])

	# Preserve original gld and ffr rate
	orig_gld_rate = gld_rate
	orig_ffr_rate = ffr_rate

	# Resample data to yearly
	gld_rate = np.interp(inf_date_unix, gld_date_unix, gld_rate)
	ffr_rate = np.interp(inf_date_unix, ffr_date_unix, ffr_rate)

	# Relabel common time for readbility
	global_date_unix = inf_date_unix

	# Set up environment

	# Set max number of generations
	max_gen = 10000
	cur_gen = 0
	best_sim = (0, [])

	# Make initial population
	pop = sim.Population(4, ffr_rate, inf_rate, gld_rate)

	# Main loop
	while cur_gen < max_gen:
		# First calculate fitness of population
		pop.calc_all_fitness()
		# Get best individuals
		pop.get_best_individuals()
		# Record best individual
		best_cur_individual = pop.best_individuals[0]
		# Compare best current gen's individual to entire history
		if best_cur_individual.fitness > best_sim[0]:
			best_sim = (best_cur_individual.fitness, best_cur_individual.genes)
		# Mate best individuals to renew population
		pop.mate_individuals()

		# Increment
		cur_gen += 1
		# Print occasionally
		if cur_gen%10:
			print('Iter: {} Cost: {}'.format(cur_gen, best_sim[0]))

	# Print out some metrics regarding the best individual from the sim
	cost = 1/best_sim[0]
	op = best_sim[1][0]
	A = best_sim[1][1]
	z = best_sim[1][2]
	B = best_sim[1][3]
	j = best_sim[1][4]

	print('Cost: {} Op: {} A: {} z: {} B: {} j: {}'\
			.format(cost, op, A, z, B, j))

	# Make hypothesis function
	hypo = lambda oper, A, v1, z, B, v2, j : oper(A * v1 ** z, B * v2 ** j)
	# Calculate predicted gold rate
	pred_gold_rate = hypo(op, A, ffr_rate, z, B, inf_rate, j)

	plt.figure()
	plt.plot(ffr_date_unix, orig_ffr_rate, label='Fed Rate')
	plt.plot(inf_date_unix, inf_rate, label='Inf Rate')
	plt.xlabel('Time')
	plt.ylabel('Percent Change')
	plt.legend()

	plt.figure()
	plt.plot(gld_date_unix, orig_gld_rate, label='Gold Rate')
	plt.plot(global_date_unix, pred_gold_rate, label='Predicted Gold Rate')
	plt.xlabel('Time')
	plt.ylabel('Price')
	plt.legend()
	plt.show()




if __name__ == "__main__":
	sim_engine()

