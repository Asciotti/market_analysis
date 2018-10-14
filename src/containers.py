import numpy as np
import random
import operator
import math

OPERATORS = [operator.add, operator.sub, operator.truediv, operator.mul]
COEF_LIM = [-10, 10]

class Individual:
	fitness = 0
	# Follows format OPS(Ax^z, By^j)
	# OPS = gene[0]
	# A = gene[1]
	# z = gene[2]
	# B = gene[3]
	# j = gene[4]
	gene_range = [4, -1, -1, -1, -1]

	def __init__(self):
		self.genes = []
		for gene_lim in self.gene_range:
			# gene limit signifies range of options for a specific gene
			# -1 indictates a continuous number on range COEF_LIM
			if gene_lim == -1:
				self.genes.append(self._get_rand_coef())
			else:
				self.genes.append(self._get_rand_ops())

	def calc_fitness(self, input1, input2, truth):
		# Calculates the fitness aka the inverse of the cost function
		# given input values and truth

		# Converts input data to numpy array if needed
		if type(input1) is not np.ndarray:
			input1 = np.array(input1)
		if type(input2) is not np.ndarray:
			input2 = np.array(input2)
		if type(truth) is not np.ndarray:
			truth = np.array(truth)

		# Extract parameters of hypothesis function
		op = self.genes[0]
		A = self.genes[1]
		z = self.genes[2]
		B = self.genes[3]
		j = self.genes[4]

		# Calculate hypothesis
		guess = self._hypothesis(op, A, z, B, j, input1, input2)

		# Calculate scalar cost
		cost = self._cost_func(guess, truth)

		# Calc fitness
		self.fitness = 1/cost

	def mutate(self):
		# Mutate will randomly select one or more bits and modify them
		gene = self.genes
		for idx, gene_lim in enumerate(self.gene_range):
			# First flip a coin to determine if gene gets modified
			draw = random.random()
			# Do not modify
			if draw < 0.5:
				continue
			# Modify
			else:
				# gene limit signifies range of options for a specific gene
				# -1 indictates a continuous number on range COEF_LIM
				if gene_lim == -1:
					gene[idx] = (self._get_rand_coef())
				else:
					gene[idx] = (self._get_rand_ops())
		# Save new genome for individual
		self.genes = gene

	def swap(self, start_idx, genes):
		# Swaps `genes` with current individual's genes beginning at 
		# start_idx
		self.genes[start_idx:len(genes)] = genes


	def _get_rand_coef(self):
		# Coefficients limited to range of COEF_LIM
		return random.uniform(COEF_LIM[0], COEF_LIM[1])

	def _get_rand_ops(self):
		# Operators limited to range of OPERATORS list
		return random.choice(OPERATORS)

	def _hypothesis(self, oper, A, z, B, j, v1, v2):
		# Returns hypothsis following function OPS(Ax^z, By^j)
		# where OPS can be multiply, divide, app, subtract
		return oper(A * v1 ** z, B * v2 ** j)

	def _cost_func(self, hypothesis, truth):
		# Returns cost of given hypothesis compared to truth using RMSE
		# (Root mean square error)

		# Calculate Root mean squared error
		rmse = np.sqrt(np.sum((hypothesis - truth) ** 2) / truth.shape[0])

		return rmse

class Population:
	size = 0
	individuals = []
	best_individuals = []

	def __init__(self, size, input1, input2, truth):
		# Clean size input to be even number...for now
		if size%2:
			print('WARNING: Requested size {} but rounded to even val'.format(size))
			size = size + 1
		self.size = size
		self.input1 = input1
		self.input2 = input2
		self.truth = truth
		self.individuals = []
		for i in range(size):
			self.individuals.append(Individual())

	def mate_individuals(self):
		# Given a population's best individuals, mate them and repopulate
		# Assumes we cross populate with `One point` crosser over at idx 3
		# Create two random offspring which will start off with random genes
		# but we will replace them with their parent's genes
		os_1 = Individual()
		os_2 = Individual()
		# Copy best two individuals (aka parents)
		best_1 = self.best_individuals[0]
		best_2 = self.best_individuals[1]
		# Copy parents to offspring (os)
		os_1.swap(0,best_1.genes)
		os_2.swap(0,best_2.genes)
		# Now swap two offspring's genomic tail
		os_1.genes[3:], os_2.genes[3:] = os_2.genes[3:], os_1.genes[3:]
		# Replenish population using the two base offspring
		new_individuals = []
		# Add original best individuals
		new_individuals.append(best_1)
		new_individuals.append(best_2)
		# Mutate offspring 1 and 2
		os_1.mutate()
		os_2.mutate()
		new_individuals.append(os_1)
		new_individuals.append(os_2)
		# Create rest of the population using os_1 and os_2 as the foundation
		new_num_individuals = math.floor((self.size - 4) / 2)
		for i in range(new_num_individuals):
			for individual in [os_1, os_2]:
				# Make new individual
				new_individual = Individual()
				# Copy original offspring genome to new individual
				new_individual.swap(0, individual.genes)
				# Mutate new individual
				new_individual.mutate()
				# Add new individual to population
				new_individuals.append(new_individual)

		# Save new list of individuals for population
		self.individuals = new_individuals

	def calc_all_fitness(self):
		# Calculates the fitness of every individual in the population
		for individual in self.individuals:
			individual.calc_fitness(self.input1, self.input2, self.truth)

	def get_best_individuals(self):
		# Using fitness of all individuals in population, gets two best
		# individuals for now
		fitness = [(idx, ind.fitness) for idx, ind in enumerate(self.individuals)]
		# Convert to numpy for sorting
		fitness = np.array(fitness, dtype=([('idx', np.int8), ('fitness', np.float32)]))
		# Sort individuals by fitness only
		sorted_individuals = np.sort(fitness, order='fitness')
		# Get best `number` of individuals
		best_individuals = sorted_individuals[-2:]
		# Save best_individuals
		self.best_individuals = [self.individuals[idx] for idx, fitness in best_individuals]