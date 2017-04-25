class Polynomial:
	'''A class to represent a polynomial with finite amount of degrees.'''
	def __init__(self, coef):
		if len(coef) is 0:
			raise Exception('Cannot initialize polynomial no coefficients.')
			
		# Trim trailing 0's
		idx = len(coef) - 1
		while idx > 0 and coef[idx] is 0:
			idx -= 1
		
		if idx < 0:
			self._coefficients = [0]
		self._coefficients = coef[:idx + 1]
		
	@property
	def coefficients(self):
		'''List of coefficients '''
		return self._coefficients
	
	@coefficients.setter
	def coefficients(self, value):
		self._coefficients = value
	
	def nested_coefficients(self):
		'''String representation in nested coefficients form of this polynormial.'''
		origin_coef = self.coefficients
		res = self.coefficients[:]
		
		# If there are less than 3 non_zero coefficients, just return the standard form coefficients.
		num_non_zero = len([1 for ele in origin_coef if ele is not 0])
		if num_non_zero < 3:
			return res
		
		# Find the first non-zero coefficient starting at degree one
		prev_nonzero_idx = 1
		while res[prev_nonzero_idx] is 0:
			prev_nonzero_idx += 1
		
		if prev_nonzero_idx < len(res) - 1:
			cur_nonzero_idx = prev_nonzero_idx + 1
			while cur_nonzero_idx < len(res):
				while res[cur_nonzero_idx] is 0:
					cur_nonzero_idx += 1
				# Found the next non-zero ceofficient
				res[cur_nonzero_idx] = origin_coef[cur_nonzero_idx] / origin_coef[prev_nonzero_idx]
				prev_nonzero_idx = cur_nonzero_idx
				cur_nonzero_idx += 1
				if cur_nonzero_idx >= len(res):
					return res
				
		return res
	
	def evaluate(self, val):
		tmp_x = 1
		res = 0
		for ele in self.coefficients:
			res += ele * tmp_x
			tmp_x *= val
		return res

	def __eq__(self, other):
		'''Override the default Equals behavior'''
		if isinstance(other, self.__class__):
			return self.coefficients == other.coefficients
		return False

	def __repr__(self):
		'''String representation in standard coefficients form of this polynormial.'''
		return self.standard_coef_rep()
	
	def standard_coef_rep(self, expand=False, show_mul_op=False, var_char='x'):
		'''String representation in standard coefficients form of this polynormial.'''
		if len(self.coefficients) is 1:
			return '{0}'.format(self.coefficients[0])
		res = ''
		is_first_non_zero_element = True
		for deg, val in enumerate(self.coefficients):
			if val is 0:
				continue
			if is_first_non_zero_element:
				if val < 0:
					res += '-'
				if deg is 0 or val is not 1:
					res += '{0}'.format(abs(val))
				if deg is not 0 and show_mul_op:
					res += ' * '
				res += self._variable_str(deg, expand, var_char)
			else:
				if val < 0:
					res += ' - '
				elif val > 0:
					res += ' + '
				if val is not 1:
					res += '{0}'.format(abs(val))
				if show_mul_op:
					res += ' * '
				res += self._variable_str(deg, expand, var_char)
			is_first_non_zero_element = False
		return res
	
	def standard_coef_code(self, var_char='x'):
		'''Code in standard coefficients form of this polynormial.'''
		return self.standard_coef_rep(expand=True, show_mul_op=True, var_char=var_char)

	def nested_coef_rep(self, expand=False, show_mul_op=False, var_char='x'):
		'''String representation in nested coefficients form of this polynormial.'''
		origin_coef = self.nested_coefficients()
		num_non_zero = len([1 for ele in origin_coef if ele is not 0])
		if num_non_zero < 3:
			return self.standard_coef_rep(expand, show_mul_op, var_char)
		
		nonzero_idx_1 = 0
		while origin_coef[nonzero_idx_1] is 0:
			nonzero_idx_1 += 1

		nonzero_idx_2 = nonzero_idx_1 + 1
		while origin_coef[nonzero_idx_2] is 0:
			nonzero_idx_2 += 1
		
		outter_coef = origin_coef[:nonzero_idx_2 + 1]
		inner_coef = origin_coef[nonzero_idx_2:]
		inner_coef[0] = 1
		
		outter_poly = Polynomial(outter_coef)
		if outter_poly == Polynomial([0]):
			return '0'
		elif outter_coef == Polynomial([1]):
			return Polynomial(inner_coef).nested_coef_rep(expand, show_mul_op, var_char)
		else:
			res = outter_poly.standard_coef_rep(expand, show_mul_op, var_char)
			if show_mul_op:
				res += ' * '
			inner_nested_rep = Polynomial(inner_coef).nested_coef_rep(expand, show_mul_op, var_char)
			res = res + '(' + inner_nested_rep +')'
			return res
		
	
	def nested_coef_code(self, var_char='x'):
		'''Code in nested coefficients form of this polynormial.'''
		return self.nested_coef_rep(expand=True, show_mul_op=True, var_char=var_char)
		
		
	# Helper Methods
	def _variable_str(self, degree, expand=False, var_char='x'):
		'''
Helper method to generate the variable string for specific degree.
e.g.:
	degree 0 => ''
	degree 1 => 'x'
	degree 2 => 'x^2' or 'x * x' \n'''
		if degree < 0:
			raise Exception('Degree of polynomial cannot be less than 0.')
		if degree is 0:
			return ''
		elif degree is 1:
			return var_char
		else:
			if expand:
			 	return var_char + ' * {0}'.format(var_char) * (degree - 1)
			else:
				return '{0}^{1}'.format(var_char, degree)

if __name__ == '__main__':
	test_coefs = [[0], [0, 0], [1], [0, 2], [0, 0, -9], [-1.5, 2.3, 6.4, 2.3, 5.7], [0, 3.5, -3, 0, 5], [1, 3.5, -3, 0, 5, 0, 0]]
	for val in test_coefs:
		poly = Polynomial(val)
		print(poly.coefficients)
		print(poly.standard_coef_rep())
		print(poly.standard_coef_code('a1'))
		print(poly.nested_coef_rep())
		print(poly.nested_coef_code('a1'))
		print()
		print('--------------')
		print()

