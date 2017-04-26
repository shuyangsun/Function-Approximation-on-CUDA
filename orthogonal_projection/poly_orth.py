import math

# Helper Functions

def __standard_basis(degree):
	res = []
	for i in range(degree + 1):
		num_coefficient = i + 1
		coefficients = [0] * num_coefficient
		coefficients[-1] = 1
		res.append(Polynomial(coefficients))
	return res

def __gram_schmidt(v_j, e_lst, start, end):
	numerator = v_j
	for ele in e_lst:
		projection = ele * inner_product(v_j, ele, start, end)
		numerator -= projection
	denominator = norm(numerator, start, end)
	return numerator * (1 / denominator)
	
# Class Polynomial

class Polynomial:
	'''A class to represent a polynomial with finite amount of degrees.'''
	def __init__(self, coeff):
		if len(coeff) is 0:
			raise Exception('Cannot initialize polynomial no coefficients.')
			
		# Trim trailing 0's
		idx = len(coeff) - 1
		while idx > 0 and coeff[idx] is 0:
			idx -= 1
		
		if idx < 0:
			self._coefficients = [0]
		self._coefficients = tuple(coeff[:idx + 1])
		
	@property
	def coefficients(self):
		'''List of coefficients '''
		return list(self._coefficients)
	
	def nested_coefficients(self):
		'''String representation in nested coefficients form of this polynormial.'''
		origin_coeff = self.coefficients
		res = self.coefficients[:]
		
		# If there are less than 3 non_zero coefficients, just return the standard form coefficients.
		num_non_zero = len([1 for ele in origin_coeff if ele is not 0])
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
				res[cur_nonzero_idx] = origin_coeff[cur_nonzero_idx] / origin_coeff[prev_nonzero_idx]
				prev_nonzero_idx = cur_nonzero_idx
				cur_nonzero_idx += 1
				if cur_nonzero_idx >= len(res):
					return res
				
		return res
	
	def degree(self):
		'''Returns the degree of polynomial. If the polynomial is identical to 0, the degree returned
will be 0, instead of -inf (negative infinity, the correct degree defined in mathematics).'''
		return len(self.coefficients) - 1

	def evaluate(self, val):
		tmp_x = 1
		res = 0
		for ele in self.coefficients:
			res += ele * tmp_x
			tmp_x *= val
		return res

	def __eq__(self, other):
		'''Override the default Equals operator'''
		if isinstance(other, self.__class__):
			return self.coefficients == other.coefficients
		return False
	
	def __mul__(self, other):
		'''Override the default Multiply operator'''
		if isinstance(other, self.__class__):
			self_coeff = self.coefficients
			other_coeff = other.coefficients
			res_deg = self.degree() + other.degree()
			res_coef = [0] * (res_deg + 1)
			for i in range(self.degree() + 1):
				for j in range(other.degree() + 1):
					res_coef[i + j] += self_coeff[i] * other_coeff[j]
			return Polynomial(res_coef)
		else:
			res_coeff = [other * a for a in self.coefficients]
			return Polynomial(res_coeff)
		
	def __add__(self, other):
		'''Override the default Add operator'''
		if isinstance(other, self.__class__):
			coeff1 = self.coefficients
			coeff2 = other.coefficients
			res_coeff = [a1 + a2 for a1, a2 in zip(coeff1, coeff2)]
			len_diff = len(coeff1) - len(coeff2)
			if len_diff < 0:
				res_coeff += coeff2[len(coeff1):]
			elif len_diff > 0:
				res_coeff += coeff1[len(coeff2):]
			return Polynomial(res_coeff)
		else:
			res_coeff = self.coefficients[:]
			res_coeff[0] += other
			return Polynomial(res_coeff)
	
	def __sub__(self, other):
		'''Override the default Subtraction operator'''
		return self + other * (-1)

	def __repr__(self):
		'''String representation in standard coefficients form of this polynormial.'''
		return self.standard_coeff_rep()
	
	def standard_coeff_rep(self, expand=False, show_mul_op=False, var_char='x'):
		'''String representation in standard coefficients form of this polynormial.'''
		if len(self.coefficients) is 1:
			return '{0}'.format(self.coefficients[0])
		res = ''
		is_first_non_zero_element = True
		for deg, val in enumerate(self.coefficients):
			if self._is_almost_zero(val):
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
	
	def standard_coeff_code(self, var_char='x'):
		'''Code in standard coefficients form of this polynormial.'''
		return self.standard_coeff_rep(expand=True, show_mul_op=True, var_char=var_char)

	def nested_coeff_rep(self, expand=False, show_mul_op=False, var_char='x'):
		'''String representation in nested coefficients form of this polynormial.'''
		origin_coeff = self.nested_coefficients()
		num_non_zero = len([1 for ele in origin_coeff if ele is not 0])
		if num_non_zero < 3:
			return self.standard_coeff_rep(expand, show_mul_op, var_char)
		
		nonzero_idx_1 = 0
		while origin_coeff[nonzero_idx_1] is 0:
			nonzero_idx_1 += 1

		nonzero_idx_2 = nonzero_idx_1 + 1
		while origin_coeff[nonzero_idx_2] is 0:
			nonzero_idx_2 += 1
		
		outter_coeff = origin_coeff[:nonzero_idx_2 + 1]
		inner_coeff = origin_coeff[nonzero_idx_2:]
		inner_coeff[0] = 1
		
		outter_poly = Polynomial(outter_coeff)
		if outter_poly == Polynomial([0]):
			return '0'
		elif outter_coeff == Polynomial([1]):
			return Polynomial(inner_coeff).nested_coeff_rep(expand, show_mul_op, var_char)
		else:
			res = outter_poly.standard_coeff_rep(expand, show_mul_op, var_char)
			if show_mul_op:
				res += ' * '
			inner_nested_rep = Polynomial(inner_coeff).nested_coeff_rep(expand, show_mul_op, var_char)
			res = res + '(' + inner_nested_rep +')'
			return res
		
	
	def nested_coeff_code(self, var_char='x'):
		'''Code in nested coefficients form of this polynormial.'''
		return self.nested_coeff_rep(expand=True, show_mul_op=True, var_char=var_char)
		
		
	# Helper Methods
	def _is_almost_zero(self, val):
		'''Deal with numerical error.'''
		return abs(val) < 0.0000001

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

def derivative(polynomial):
	'''Taking the derivative of a polynomial, return the result.'''
	if polynomial.degree() <= 0:
		return Polynomial([0])

	res_coeff = polynomial.coefficients[1:]
	for deg, val in enumerate(res_coeff):
		res_coeff[deg] = val * (deg + 1)
	return Polynomial(res_coeff)

def integrate(polynomial, domain=[]):
	'''Takes a polynomial, and take it's integral.
If the domain is not specified, the return result is another polynomial
that's the integral of the original polynomial, with constant as 0.
If the domain is specified, the return result is the integrated value on
the given domain.
	'''
	if len(domain) is not 2 and len(domain) is not 0:
		raise Exception('Cannot integrate with wrong length of domain')

	if len(domain) is 2:
		res_poly = integrate(polynomial)
		return res_poly.evaluate(domain[1]) - res_poly.evaluate(domain[0])
	
	# Produce the integral function.
	res_coeff = polynomial.coefficients
	for deg, a in enumerate(res_coeff):
		if a is not 0:
			res_coeff[deg] = a/(deg + 1)
	res_coeff = [0] + res_coeff
	return Polynomial(res_coeff)

def inner_product(poly1, poly2, start, end):
	return integrate(poly1 * poly2, [start, end])

def norm(poly, start, end):
	return math.sqrt(inner_product(poly, poly, start, end))

def orthonormal_basis(degree, start, end):
	std_basis = __standard_basis(degree)
	res = []
	for v_j in std_basis:
		e_j = __gram_schmidt(v_j, res, start, end)
		res.append(e_j)
	return res

if __name__ == '__main__':
	orth_basis = orthonormal_basis(8, -2, 2)
	for ele in orth_basis:
		print(ele)

