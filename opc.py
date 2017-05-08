#!/usr/bin/env python
"""
Orthogonal Projection Calculator
    
    
    --version
        Prints the version of Orthogonal Projection Calculator.    
    
    --help
        Prints help information.
    
    --print <string> [-nested, -code, -var <name>]
        Prints a polynomial, interpreted in standard coefficients form from <string>.
        
        | -nested
        |   Print in nested coefficients format (e.g., "2 - 2x^2 +8x^4" => "2 - 2x^2(1 - 4x^2)").
        | -code
        |   Print in computer program code format (e.g., "3x^2" => "3 * x * x").
        | -var <name>
        |   Specify the string for variable name (e.g., "3 + x^2" => "3 + <name>^2").
    
    --derivative <string> [-nested, -code, -var <name>]
        Prints the derivative of given polynomial, interpreted in standard coefficients form from <string>.
        
        | Optional arguments: see "--print".
    
    --integrate <string> [<from> <to>, -nested, -code, -var <name>]
        Prints the derivative of given polynomial, interpreted in standard coefficients form from <string>. If no
        integration domain is specified, result polynomial will be printed. If integration domain <from> and <to> is
        specified, numerical result will be printed.
        
        | Optional arguments: see "--print".

    --std-basis <degree> [-nested, -code, -var <ch>]
        Prints the standard basis of the vector space of polynomials with highest degree as <degree> .
        
        | Optional arguments: see "--print".
    
    --orth-basis <from> <to> <degree> [-nested, -code, -var <ch>]
        Prints the orthonormal basis for the vector space of polynomials with highest degree as <degree>, with
        inner product defined as <f, g> = INTEGRATE f(x) * g(x) dx FROM <from> TO <to>.
        
        | Optional arguments: see "--print".
        
    --approximation <function> <from> <to> <degree> [-nested, -code, -var <ch>]
        Prints the approximation of <function> as a polynomial with highest degree as <degree>, with
        inner product defined as <f, g> = INTEGRATE f(x) * g(x) dx FROM <from> TO <to>.

        | Optional arguments: see "--print".

"""

__author__ = "Shuyang Sun"
__copyright__ = "© 2017 Shuyang Sun. All rights reserved."
__license__ = "MIT"
__version__ = "0.0.2"
__maintainer__ = "Shuyang Sun"
__email__ = "sunbuffett@gmail.com"
__status__ = "Beta"


import math
import sys
import sympy as sp
import time
from sympy.parsing.sympy_parser import parse_expr
from sympy.integrals import Integral
from enum import Enum


# Helper Functions

def _is_almost_zero(val):
    return abs(val) < 0.00000001


def _variable_str(degree, expand=False, double_stars=False, var_char='x'):
    """
    Helper method to generate the variable string for specific degree.
    e.g.:
        degree 0 => ''
        degree 1 => 'x'
        degree 2 => 'x^2' or 'x * x'
    """
    power_op = '^'
    if double_stars:
        power_op = '**'
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
            return '{0}{1}{2}'.format(var_char, power_op, degree)


def _gram_schmidt(v_j, e_lst, start, end):
    numerator = v_j
    for e_j in e_lst:
        projection = e_j * inner_product(v_j, e_j, start, end)
        numerator -= projection
    denominator = norm(numerator, start, end)
    return numerator * (1 / denominator)


def _print_poly(poly, nested, code, ch):
    if nested:
        if code:
            print(poly.nested_coeff_code(ch))
        else:
            print(poly.nested_coeff_rep(expand=False, show_mul_op=False, var_char=ch))
    else:
        if code:
            print(poly.standard_coeff_code(ch))
        else:
            print(poly.standard_coeff_rep(expand=False, show_mul_op=False, var_char=ch))


def _print_std_basis(degree, nested, code, ch):
    print('Standard basis for vector space of polynomials with degree {0}:'.format(degree))
    print()
    res = standard_basis(degree)
    for idx, ele in enumerate(res):
        print('v{0} = '.format(idx + 1), end='')
        _print_poly(ele, nested, code, ch)
    print()


def _print_orth_basis(integrate_from, integrate_to, degree, nested, code, ch):
    print('Orthonormal basis for inner product space of polynomials with degree {0}, with inner product defined as\n'
          '<f, g> = INTEGRATE f(x) * g(x) dx FROM {1} TO {2}:'.format(degree, integrate_from, integrate_to))
    print()
    res = orthonormal_basis(integrate_from, integrate_to, degree)
    for idx, ele in enumerate(res):
        print('e{0} = '.format(idx + 1), end='')
        _print_poly(ele, nested, code, ch)
    print()


def _print_approximation(func, integrate_from, integrate_to, degree, nested, code, ch):
    print()
    res = approximate(func, integrate_from, integrate_to, degree)
    print()
    print('f(x) = ', end='')
    _print_poly(res, nested, code, ch)
    print()


def _print_derivative(poly, nested, code, ch):
    res = derivative(poly)
    print('Derivative of polynomial f(x) = {0}:'.format(poly.standard_coeff_rep()))
    print()
    print('d/dx = ', end='')
    _print_poly(res, nested, code, ch)
    print()


def _print_integration(poly, integrate_from, integrate_to, nested, code, ch):
    print('Derivative of polynomial f(x) = {0}'.format(poly.standard_coeff_rep()), end='')
    domain = list()
    if integrate_from is not None and integrate_to is not None:
        domain = [integrate_from, integrate_to]
        print(' from {0} to {1}'.format(integrate_from, integrate_to), end='')
    print(':')
    print()
    res = integrate(poly, domain)
    _print_poly(res, nested, code, ch)
    print()


def _get_float_value(val):
    val = _remove_white_spaces(val)
    if 'pi' in val:
        if val == 'pi':
            return math.pi
        end = -2
        if '*' in val:
            end = -3
        scalar_str = val[:end]
        if scalar_str is '-':
            scalar = -1
        else:
            scalar = float(scalar_str)
        return scalar * math.pi
    else:
        return float(val)


def _remove_white_spaces(string):
    res = str(string)
    res = res.replace(' ', '')
    res = res.replace('\t', '')
    res = res.replace('\n', '')
    res = res.replace('\r', '')
    return res


def _get_degree_and_coeff(string):
    has_x = 'x' in string
    has_power = '^' in string
    if not has_x and not has_power:
        return 0, _get_float_value(string)
    elif has_x and not has_power:
        if string[0] == 'x':
            return 1, 1
        else:
            return 1, _get_float_value(string[:-1])
    else:
        for idx, ch in enumerate(string):
            if ch == 'x':
                if idx is 0:
                    coeff = 1.0
                else:
                    coeff = _get_float_value(string[:idx])
                degree = int(string[idx + 2:])
                return degree, coeff


def _to_coeff_lst(degree_and_coeff_lst):
    degrees = [deg_coeff[0] for deg_coeff in degree_and_coeff_lst]
    max_deg = max(degrees)
    res = [0] * (max_deg + 1)
    for ele in degree_and_coeff_lst:
        res[ele[0]] = ele[1]
    return res


def _split_to_coeff_sections(string):
    """
    Separate the a string that represents a polynomial in the standard coefficient form into list of strings, each
    element is a combination of coefficient, variable name and degree.
    :param string: String representation of polynomial.
    :return: List of strings, each contains coefficient, variable, and degree.
    """
    res = list()
    idx = 0
    no_space = _remove_white_spaces(string)
    while idx < len(no_space):
        is_plus_or_minus = no_space[idx] == '+' or no_space[idx] == '-'
        has_e = idx > 1 and no_space[idx - 1] is 'e'
        if is_plus_or_minus and idx is not 0 and not has_e:
            res.append(no_space[:idx])
            no_space = no_space[idx:]
            idx = 0
        else:
            idx += 1

    if idx >= len(no_space) and len(no_space) is not 0:
        res.append(no_space)

    return res


def _nested_coeff_rep_with_nested_coeff(nested_coeff, expand=False, show_mul_op=False, var_char='x'):
    num_non_zero = len([1 for ele in nested_coeff if not _is_almost_zero(ele)])
    if num_non_zero < 3:
        return Polynomial(nested_coeff).standard_coeff_rep(expand, show_mul_op, var_char)

    nonzero_idx_1 = 0
    while _is_almost_zero(nested_coeff[nonzero_idx_1]):
        nonzero_idx_1 += 1

    nonzero_idx_2 = nonzero_idx_1 + 1
    while _is_almost_zero(nested_coeff[nonzero_idx_2]):
        nonzero_idx_2 += 1

    outer_coeff = nested_coeff[:nonzero_idx_2 + 1]
    inner_coeff = nested_coeff[nonzero_idx_2:]
    inner_coeff[0] = 1

    outer_poly = Polynomial(outer_coeff)
    if outer_poly == Polynomial([0]):
        return '0'
    elif outer_poly == Polynomial([1]):
        return _nested_coeff_rep_with_nested_coeff(inner_coeff, expand, show_mul_op, var_char)
    else:
        res = outer_poly.standard_coeff_rep(expand, show_mul_op, var_char)
        if show_mul_op:
            res += ' * '
        inner_nested_rep = _nested_coeff_rep_with_nested_coeff(inner_coeff, expand, show_mul_op, var_char)
        res = res + '(' + inner_nested_rep + ')'
        return res


class _Intention(Enum):
    """
    Intention of user via program arguments.
    """
    Version = 0,
    Help = 1,
    GenerateStandardBasis = 2
    GenerateOrthogonalBasis = 3
    PolynomialEvaluation = 4
    Derivative = 5
    Integration = 6
    ApproximateWithPolynomial = 7
    PrintPolynomial = 8


def _arg_parser(argv):
    """
    Parse the argument list passed into this program.
    :param argv: Original arguments of the system.
    :return: None if not understood, or a tuple in the format:
    (intention, nested_form, code_form, var_char, remaining_argv).
    """
    if len(argv) <= 1:
        return None

    argv = argv[1:]
    command_intention_dict = {
        '--version': _Intention.Version,
        '--help': _Intention.Help,
        '--print': _Intention.PrintPolynomial,
        '--orth-basis': _Intention.GenerateOrthogonalBasis,
        '--std-basis': _Intention.GenerateStandardBasis,
        '--eval': _Intention.PolynomialEvaluation,
        '--derivative': _Intention.Derivative,
        '--integrate': _Intention.Integration,
        '--approximate': _Intention.ApproximateWithPolynomial
    }

    keys = command_intention_dict.keys()
    arg0 = argv[0]
    if any(arg0 == key for key in keys):
        res = list()
        res.append(command_intention_dict[arg0])
        argv = argv[1:]

        config = '-nested'
        if any(arg == config for arg in argv):
            res.append(True)
            argv.remove(config)
        else:
            res.append(False)
        config = '-code'
        if any(arg == config for arg in argv):
            res.append(True)
            argv.remove(config)
        else:
            res.append(False)
        if any(arg == '-var' for arg in argv):
            idx = argv.index('-var') + 1
            res.append(argv[idx])
            del argv[idx]
            del argv[idx - 1]
        else:
            res.append('x')
        res.append(argv)
        return tuple(res)
    return None


# Class Polynomial

class Polynomial:
    """
    A finite degree polynomial.
    """

    def __init__(self, value):
        """
        Initialize a Polynomial with either a list of coefficients, or a string.
        :param value: Coefficients or string.
        """
        coeff_lst = value
        if isinstance(value, str):
            self._original_str = value[:]
            str_lst = _split_to_coeff_sections(value)
            degree_coeff_lst = [_get_degree_and_coeff(string) for string in str_lst]
            coeff_lst = _to_coeff_lst(degree_coeff_lst)
        if len(value) is 0:
            raise Exception('Cannot initialize polynomial no coefficients.')

        self._coefficients = tuple(self._trim_zeros(coeff_lst))

    @property
    def initialization_str(self):
        res = self.standard_coeff_rep()
        if hasattr(self, '_original_str'):
            res = self._original_str[:]
        return res

    @property
    def coefficients(self):
        """
        Get a list of standard coefficients of this polynomial.
        :return: Coefficients.
        """
        return list(self._coefficients)

    @property
    def nested_coefficients(self):
        """
        Nested coefficients of this polynomial.
        :return: List of nested coefficients.
        """
        origin_coeff = self.coefficients
        res = self.coefficients[:]

        # If there are less than 3 non_zero coefficients, just return the standard form coefficients.
        num_non_zero = len([1 for coeff in origin_coeff if coeff is not 0])
        if num_non_zero < 3:
            return res

        # Find the first non-zero coefficient starting at degree one
        prev_nonzero_idx = 1
        while _is_almost_zero(res[prev_nonzero_idx]):
            prev_nonzero_idx += 1

        if prev_nonzero_idx < len(res) - 1:
            cur_nonzero_idx = prev_nonzero_idx + 1
            while cur_nonzero_idx < len(res):
                while _is_almost_zero(res[cur_nonzero_idx]):
                    cur_nonzero_idx += 1
                # Found the next non-zero coefficient
                res[cur_nonzero_idx] = origin_coeff[cur_nonzero_idx] / origin_coeff[prev_nonzero_idx]
                prev_nonzero_idx = cur_nonzero_idx
                cur_nonzero_idx += 1
                if cur_nonzero_idx >= len(res):
                    return res

        return res

    def degree(self):
        """
        Returns the degree of polynomial. If the polynomial is identical to 0, the degree returned
        will be 0, instead of -inf (negative infinity, the correct degree defined in mathematics).
        :return: Degree of polynomial.
        """
        """"""
        return len(self.coefficients) - 1

    def evaluate(self, val):
        """
        Calculate the result of polynomial with given value of variable.
        :param val: Value for parameter.
        :return: Calculated result.
        """
        tmp_x = 1
        res = 0
        for ele in self.coefficients:
            res += ele * tmp_x
            tmp_x *= val
        return res

    def __call__(self, *args, **kwargs):
        if len(args) is 0:
            return
        elif len(args) is 1:
            return self.evaluate(args[0])
        res = list()
        for arg in args:
            res.append(self.evaluate(arg))
            return res

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._trim_zeros(self.coefficients) == self._trim_zeros(other.coefficients)
        return False

    def __mul__(self, other):
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

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return (-1) * self

    def __add__(self, other):
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

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return other + -self

    def __repr__(self):
        return self.standard_coeff_rep()

    def standard_coeff_rep(self, expand=False, show_mul_op=False, double_stars=False, var_char='x'):
        """
        String representation in standard coefficients form of this polynomial.
        :param expand: 'x * x' if True, 'x^2' otherwise. 
        :param show_mul_op: '2.5 * x' is True, '2.5x' otherwise.
        :param var_char: String for variable name.
        :return: Standard coefficient representation.
        """
        if len(self.coefficients) is 1:
            return '{0}'.format(self.coefficients[0])
        res = ''
        is_first_non_zero_element = True
        for deg, val in enumerate(self.coefficients):
            if _is_almost_zero(val):
                continue
            if is_first_non_zero_element:
                if val < 0:
                    res += '-'
                if deg is 0 or val is not 1:
                    res += '{0}'.format(abs(val))
                if show_mul_op and deg is not 0:
                    res += ' * '
                res += _variable_str(deg, expand, double_stars, var_char)
            else:
                if val < 0:
                    res += ' - '
                elif val > 0:
                    res += ' + '
                if val is not 1:
                    res += '{0}'.format(abs(val))
                if show_mul_op:
                    res += ' * '
                res += _variable_str(deg, expand, double_stars, var_char)
            is_first_non_zero_element = False
        return res

    def standard_coeff_code(self, var_char='x'):
        """
        Code representation in standard coefficients form of this polynomial.
        :param expand: 'x * x' if True, 'x^2' otherwise. 
        :param show_mul_op: '2.5 * x' is True, '2.5x' otherwise.
        :param var_char: String for variable name.
        :return: Code of standard coefficient representation.
        """
        return self.standard_coeff_rep(expand=True, show_mul_op=True, var_char=var_char)

    def nested_coeff_rep(self, expand=False, show_mul_op=False, var_char='x'):
        """
        String representation in nested coefficients form of this polynomial.
        :param expand: 'x * x' if True, 'x^2' otherwise. 
        :param show_mul_op: '2.5 * x' is True, '2.5x' otherwise.
        :param var_char: String for variable name.
        :return: Nested coefficient representation.
        """
        nested_coeff = self.nested_coefficients
        return _nested_coeff_rep_with_nested_coeff(nested_coeff, expand, show_mul_op, var_char)


    def nested_coeff_code(self, var_char='x'):
        """
        Code representation in nested coefficients form of this polynomial.
        :param expand: 'x * x' if True, 'x^2' otherwise.
        :param show_mul_op: '2.5 * x' is True, '2.5x' otherwise.
        :param var_char: String for variable name.
        :return: Code of nested coefficient representation.
        """
        return self.nested_coeff_rep(expand=True, show_mul_op=True, var_char=var_char)

    # Helper Methods

    def _trim_zeros(self, lst):
        idx = len(lst) - 1
        while idx > 0 and _is_almost_zero(lst[idx]):
            idx -= 1

        if idx < 0:
            return [0]

        return lst[:idx + 1]


# Public Functions

def derivative(polynomial):
    """
    Take the derivative of a polynomial.
    :param polynomial: Polynomial to take derivative.
    :return: Result of derivative.
    """
    if polynomial.degree() <= 0:
        return Polynomial([0])

    res_coeff = polynomial.coefficients[1:]
    for deg, val in enumerate(res_coeff):
        res_coeff[deg] = val * (deg + 1)
    return Polynomial(res_coeff)


def integrate(polynomial, domain=list()):
    """
    Takes a polynomial, and take it's integral.
    If the domain is not specified, the return result is another polynomial that's the integral of the original
    polynomial, with constant as 0.
    If the domain is specified, the return result is the integrated value on the given domain.
    :param polynomial: Polynomial to take integral.
    :param domain: Start and end of the domain for this integration. If this list is empty, this functions returns the
    polynomial instead of numerical result.
    :return: Polynomial or numerical result of the integration.
    """
    if len(domain) is not 2 and len(domain) is not 0:
        raise Exception('Cannot integrate with wrong length of domain')

    if len(domain) is 2:
        res_poly = integrate(polynomial)
        return res_poly(domain[1]) - res_poly(domain[0])

    # Produce the integral function.
    res_coeff = polynomial.coefficients
    for deg, a in enumerate(res_coeff):
        if a is not 0:
            res_coeff[deg] = a / (deg + 1)
    res_coeff = [0] + res_coeff
    return Polynomial(res_coeff)


def inner_product(poly1, poly2, start, end):
    """
    Calculate the inner product result, with inner product defined as <f, g> = INTEGRATE f(x) * g(x) dx FROM a to b.
    There is numerical loss, but this method guarantees <f, g> >= 0 if f == g.
    :param poly1: f
    :param poly2: g
    :param start: a
    :param end: b
    :return: Numerical result of inner product.
    """
    res = integrate(poly1 * poly2, [start, end])
    # Because of numerical loss, <v, v> could be less than 0. Return 0 if it's less than 0.
    if poly1 == poly2 and res <= 0:
        return 0
    return res


def norm(poly, start, end):
    """
    Calculate the norm of a polynomial, with norm defined as ||f|| = SQRT(INTEGRATE f^2(x) dx FROM a to b).
    :param poly: f
    :param start: a
    :param end: b
    :return: Numerical result of the norm.
    """
    inner_product_res = inner_product(poly, poly, start, end)
    return math.sqrt(inner_product_res)


def standard_basis(degree):
    """
    Generates a standard basis of a vector space of polynomials with given highest degree.
    :param degree: Highest degree of polynomial.
    :return: List of polynomials in standard basis of Pm(R), with m = degree.
    """
    res = list()
    for i in range(degree + 1):
        num_coefficient = i + 1
        coefficients = [0] * num_coefficient
        coefficients[-1] = 1
        res.append(Polynomial(coefficients))
    return res


def orthonormal_basis(start, end, degree):
    """
    Generate an orthonormal basis of Pm(R), with inner product defined as <f, g> = INTEGRATE f(x) * g(x) dx FROM a to b.
    :param start: a
    :param end: b
    :param degree: m
    :return: List of orthonormal basis of Pm(R).
    """
    std_basis = standard_basis(degree)
    res = list()
    for v_j in std_basis:
        e_j = _gram_schmidt(v_j, res, start, end)
        res.append(e_j)
    return res


def approximate(func, from_val, to_val, degree):
    """
    Approximate given continuous function over real numbers, using a polynomial of given degree, with inner product
    defined as <f, g> = INTEGRATE f(x) * g(x) dx from a to b.
    :param func: Function to approximate represented in a string, with variable as 'x'.
    :param from_val: a as a string.
    :param to_val: b as a string.
    :param degree: Highest degree of result polynomial, as an integer.
    :return: Approximated polynomial function in string format.
    """
    print('------ Started Calculating Approximation ------')
    print()
    print('f(x) = {0}'.format(func))
    print()
    start_time = time.time()
    orth_basis = orthonormal_basis(_get_float_value(from_val), _get_float_value(to_val), degree)
    print('Orthonormal basis:')
    for idx, ele in enumerate(orth_basis):
        print('e{0} = {1}'.format(idx + 1, ele))
    print()
    x = sp.Symbol('x')
    res = 0
    func_str = '({0})'.format(func)
    for idx, e_j in enumerate(orth_basis):
        start_time_e_j = time.time()
        print('Calculating projection on e{0}... '.format(idx + 1), end='')
        e_j_str = '({0})'.format(e_j.standard_coeff_rep(show_mul_op=True, double_stars=True))
        product_str = '{0} * {1}'.format(func_str, e_j_str)
        func_product = parse_expr(product_str)
        tmp = Integral(func_product, (x, from_val, to_val)).as_sum(100, method="midpoint").n()
        tmp *= parse_expr(e_j_str)
        res += tmp
        end_time_e_j = time.time()
        print('%.2fs' % (end_time_e_j - start_time_e_j))
        # Print the current result
        tmp_res = str(res)
        tmp_res = tmp_res.replace('**', '^')
        tmp_res = tmp_res.replace('*', '')
        tmp_res = Polynomial(tmp_res)
        print()
        print('    f{0}(x) = {1}'.format(idx + 1, tmp_res))
        print()
    end_time = time.time()
    print()
    print("Duration: %.2fs" % (end_time - start_time))
    print('------ Finished Calculating Approximation ------')
    res = str(res)
    res = res.replace('**', '^')
    res = res.replace('*', '')
    res = Polynomial(res)
    return res

if __name__ == '__main__':
    arg_config = _arg_parser(sys.argv)
    if arg_config is None:
        print('Unrecognized program argument.')

    intention = arg_config[0]
    nested = arg_config[1]
    code = arg_config[2]
    ch = arg_config[3]
    argv = arg_config[4]

    if intention is _Intention.Version:
        print(__version__)
    elif intention is _Intention.Help:
        print(__doc__)
    elif intention is _Intention.PrintPolynomial:
        _print_poly(Polynomial(argv[0]), nested, code, ch)
    elif intention is _Intention.Derivative:
        _print_derivative(Polynomial(argv[0]), nested, code, ch)
    elif intention is _Intention.Integration:
        if len(argv) is 1:
            _print_integration(Polynomial(argv[0]), None, None, nested, code, ch)
        else:
            _print_integration(Polynomial(argv[0]),
                               _get_float_value(argv[1]),
                               _get_float_value(argv[2]),
                               nested, code, ch)
    elif intention is _Intention.GenerateStandardBasis:
        _print_std_basis(int(argv[0]), nested, code, ch)
    elif intention is _Intention.GenerateOrthogonalBasis:
        _print_orth_basis(_get_float_value(argv[0]), _get_float_value(argv[1]), int(argv[2]), nested, code, ch)
    elif intention is _Intention.ApproximateWithPolynomial:
        _print_approximation(argv[0], argv[1], argv[2], int(argv[3]), nested, code, ch)
    print('Program finished execution.')
    exit(0)
