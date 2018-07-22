from collections import deque

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pylab import figure, axes, pie, title, show
from random import randint
import random
import math

"""
	Base class for unary constraints
	Implement isSatisfied in subclass to use
"""


def xrange(x):
    return iter(range(x))

class UnaryConstraint:
    def __init__(self, var):
        self.var = var

    def isSatisfied(self, value):
        print(value)
        # util.raiseNotDefined()

    def affects(self, var):
        return var == self.var


"""	
	Implementation of UnaryConstraint
	Satisfied if value does not match passed in paramater
"""


class BadValueConstraint(UnaryConstraint):
    def __init__(self, var, badValue):
        self.var = var
        self.badValue = badValue

    def isSatisfied(self, value):
        return not value == self.badValue

    def __repr__(self):
        return 'BadValueConstraint (%s) {badValue: %s}' % (str(self.var), str(self.badValue))


"""	
	Implementation of UnaryConstraint
	Satisfied if value matches passed in paramater
"""


class GoodValueConstraint(UnaryConstraint):
    def __init__(self, var, goodValue):
        self.var = var
        self.goodValue = goodValue

    def isSatisfied(self, value):
        return value == self.goodValue

    def __repr__(self):
        return 'GoodValueConstraint (%s) {goodValue: %s}' % (str(self.var), str(self.goodValue))


"""
	Base class for binary constraints
	Implement isSatisfied in subclass to use
"""


class BinaryConstraint:
    def __init__(self, var1, var2):
        self.var1 = var1
        self.var2 = var2

    def isSatisfied(self, value1, value2):
        if(value1==value2):
            return False
        return True

    def affects(self, var):
        return var == self.var1 or var == self.var2

    def otherVariable(self, var):
        if var == self.var1:
            return self.var2
        return self.var1


"""
	Implementation of BinaryConstraint
	Satisfied if both values assigned are different
"""


class NotEqualConstraint(BinaryConstraint):
    def isSatisfied(self, value1, value2):
        if value1 == value2:
            return False
        return True

    def __repr__(self):
        return 'NotEqualConstraint (%s, %s)' % (str(self.var1), str(self.var2))


"""
	Implementation of BinaryConstraint
	Used when solving nQueens
	Satisfied if both values are different and the absolute value of the difference
	of the values are different
"""


class nQueensConstraint(BinaryConstraint):
    def isSatisfied(self, value1, value2):
        if value1 is None or value2 is None:
            return False
        if value1 != value2 and abs(int(value1) - int(value2)) != abs(int(self.var1) - int(self.var2)):
            return True
        return False

    def __repr__(self):
        return 'BadValueConstraint (%s, %s)' % (str(self.var1), str(self.var2))


class ConstraintSatisfactionProblem:
    """
    Structure of a constraint satisfaction problem.
    Variables and domains should be lists of equal length that have the same order.
    varDomains is a dictionary mapping variables to possible domains.
    Args:
        variables (list<string>): a list of variable names
        domains (list<set<value>>): a list of sets of domains for each variable
        binaryConstraints (list<BinaryConstraint>): a list of binary constraints to satisfy
        unaryConstraints (list<BinaryConstraint>): a list of unary constraints to satisfy
    """

    def __init__(self, variables, domains, binaryConstraints=[], unaryConstraints=[]):
        self.varDomains = {}
        #print(len(variables))
        for i in xrange(len(variables)):
            self.varDomains[variables[i]] = domains ##domains[i]
        self.binaryConstraints = binaryConstraints
        self.unaryConstraints = unaryConstraints
        self.COUNTER = 0

    def __repr__(self):
        return '---Variable Domains\n%s---Binary Constraints\n%s---Unary Constraints\n%s' % ( \
            ''.join([str(e) + ':' + str(self.varDomains[e]) + '\n' for e in self.varDomains]), \
            ''.join([str(e) + '\n' for e in self.binaryConstraints]), \
            ''.join([str(e) + '\n' for e in self.binaryConstraints]))


class Assignment:
    """
    Representation of a partial assignment.
    Has the same varDomains dictionary stucture as ConstraintSatisfactionProblem.
    Keeps a second dictionary from variables to assigned values, with None being no assignment.
    Args:
        csp (ConstraintSatisfactionProblem): the problem definition for this assignment
    """

    def __init__(self, csp):
        self.varDomains = {}
        for var in csp.varDomains:
            self.varDomains[var] = set(csp.varDomains[var])
        self.assignedValues = {var: None for var in self.varDomains}

    """
    Determines whether this variable has been assigned.
    Args:
        var (string): the variable to be checked if assigned
    Returns:
        boolean
        True if var is assigned, False otherwise
    """

    def isAssigned(self, var):
        return self.assignedValues[var] != None

    """
    Determines whether this problem has all variables assigned.
    Returns:
        boolean
        True if assignment is complete, False otherwise
    """

    def isComplete(self):
        for var in self.assignedValues:
            if not self.isAssigned(var):
                return False
        return True

    """
    Gets the solution in the form of a dictionary.
    Returns:
        dictionary<string, value>
        A map from variables to their assigned values. None if not complete.
    """

    def extractSolution(self):
        if not self.isComplete():
            return None
        return self.assignedValues

    def __repr__(self):
        return '---Variable Domains\n%s---Assigned Values\n%s' % ( \
            ''.join([str(e) + ':' + str(self.varDomains[e]) + '\n' for e in self.varDomains]), \
            ''.join([str(e) + ':' + str(self.assignedValues[e]) + '\n' for e in self.assignedValues]))


####################################################################################################


"""
	Checks if a value assigned to a variable is consistent with all binary constraints in a problem.
	Do not assign value to var. Only check if this value would be consistent or not.
	If the other variable for a constraint is not assigned, then the new value is consistent with the constraint.
	Args:
		assignment (Assignment): the partial assignment
		csp (ConstraintSatisfactionProblem): the problem definition
		var (string): the variable that would be assigned
		value (value): the value that would be assigned to the variable
	Returns:
		boolean
		True if the value would be consistent with all currently assigned values, False otherwise
"""


def consistent(assignment, csp, var, value):
    """Question 1"""

    for constraint in csp.binaryConstraints:
        if (constraint.affects(var)):
            if (assignment.isAssigned(constraint.otherVariable(var))):
                if (not constraint.isSatisfied(value, assignment.assignedValues[constraint.otherVariable(var)])):
                    return False
    return True


"""
	Recursive backtracking algorithm.
	A new assignment should not be created. The assignment passed in should have its domains updated with inferences.
	In the case that a recursive call returns failure or a variable assignment is incorrect, the inferences made along
	the way should be reversed. See maintainArcConsistency and forwardChecking for the format of inferences.
	Examples of the functions to be passed in:
	orderValuesMethod: orderValues, leastConstrainingValuesHeuristic
	selectVariableMethod: chooseFirstVariable, minimumRemainingValuesHeuristic
	Args:
		assignment (Assignment): a partial assignment to expand upon
		csp (ConstraintSatisfactionProblem): the problem definition
		orderValuesMethod (function<assignment, csp, variable> returns list<value>): a function to decide the next value to try
		selectVariableMethod (function<assignment, csp> returns variable): a function to decide which variable to assign next
	Returns:
		Assignment
		A completed and consistent assignment. None if no solution exists.
"""


def recursiveBacktracking(assignment, csp, orderValuesMethod, selectVariableMethod):
    """Question 1"""
    if assignment.isComplete():
        return assignment

    nextVar = selectVariableMethod(assignment, csp)
    if not nextVar:
        return None

    for val in orderValuesMethod(assignment, csp, nextVar):
        if consistent(assignment, csp, nextVar, val):
            assignment.assignedValues[nextVar] = val
            csp.COUNTER += 1
            answer = recursiveBacktracking(assignment, csp, orderValuesMethod, selectVariableMethod)
            if answer:
                return answer
        assignment.assignedValues[nextVar] = None
        csp.COUNTER += 1

    return None


"""
	Uses unary constraints to eleminate values from an assignment.
	Args:
		assignment (Assignment): a partial assignment to expand upon
		csp (ConstraintSatisfactionProblem): the problem definition
	Returns:
		Assignment
		An assignment with domains restricted by unary constraints. None if no solution exists.
"""


def eliminateUnaryConstraints(assignment, csp):
    domains = assignment.varDomains
    for var in domains:
        for constraint in (c for c in csp.unaryConstraints if c.affects(var)):
            for value in (v for v in list(domains[var]) if not constraint.isSatisfied(v)):
                domains[var].remove(value)
                if len(domains[var]) == 0:
                    # Failure due to invalid assignment
                    return None
    return assignment


"""
	Trivial method for choosing the next variable to assign.
	Uses no heuristics.
"""


def chooseFirstVariable(assignment, csp):
    for var in csp.varDomains:
        if not assignment.isAssigned(var):
            return var


"""
	Selects the next variable to try to give a value to in an assignment.
	Uses minimum remaining values heuristic to pick a variable. Use degree heuristic for breaking ties.
	Args:
		assignment (Assignment): the partial assignment to expand
		csp (ConstraintSatisfactionProblem): the problem description
	Returns:
		the next variable to assign
"""


def minimumRemainingValuesHeuristic(assignment, csp):
    """Question 2"""
    nextVar = None
    for key in assignment.assignedValues:
        if not assignment.isAssigned(key):
            nextVar = key
            break

    domain = assignment.varDomains
    for var in assignment.assignedValues:
        if not assignment.isAssigned(var):
            if len(domain[var]) < len(domain[nextVar]):
                nextVar = var
            elif len(domain[var]) == len(domain[nextVar]):
                myCount = otherCount = 0
                for constraint in csp.binaryConstraints:
                    if constraint.affects(nextVar):
                        otherCount += 1
                    if constraint.affects(var):
                        myCount += 1
                if otherCount < myCount:
                    nextVar = var
    return nextVar


"""
	Trivial method for ordering values to assign.
	Uses no heuristics.
"""


def orderValues(assignment, csp, var):
    return list(assignment.varDomains[var])


"""
	Creates an ordered list of the remaining values left for a given variable.
	Values should be attempted in the order returned.
	The least constraining value should be at the front of the list.
	Args:
		assignment (Assignment): the partial assignment to expand
		csp (ConstraintSatisfactionProblem): the problem description
		var (string): the variable to be assigned the values
	Returns:
		list<values>
		a list of the possible values ordered by the least constraining value heuristic
"""


def leastConstrainingValuesHeuristic(assignment, csp, var):
    """Hint: Creating a helper function to count the number of constrained values might be useful"""
    """Question 3"""
    values = list(assignment.varDomains[var])
    tuples = []
    answer = []
    for value in values:
        count = usefulConstraintCount(assignment, csp, var, value)
        if not tuples:
            tuples.append((value, count))
            answer.append(value)
        else:
            index = 0
            while index < len(tuples) and count < tuples[index][1]:
                index += 1
            tuples.insert(index, (value, count))
            answer.insert(index, value)
    return answer


def usefulConstraintCount(assignment, csp, var, value):
    count = 0
    if not consistent(assignment, csp, var, value):
        for constraint in csp.binaryConstraints:
            if constraint.affects(var):
                if assignment.assignedValues[constraint.otherVariable(var)] == value:
                    count -= 1
        return count

    valueMap = {}
    for key in assignment.assignedValues.keys():
        valueMap[key] = assignment.assignedValues[key]

    valueMap[var] = value
    for constraint in csp.binaryConstraints:
        if constraint.affects(var):
            otherVar = constraint.otherVariable(var)
            if not valueMap[otherVar]:
                for temp in assignment.varDomains[otherVar]:
                    isConsistent = True
                    for countConstraint in csp.binaryConstraints:
                        if countConstraint.affects(otherVar) and not countConstraint.isSatisfied(temp, valueMap[
                            countConstraint.otherVariable(otherVar)]):
                            isConsistent = False
                    if isConsistent:
                        count += 1
    return count


"""
	Trivial method for making no inferences.
"""


def noInferences(assignment, csp, var, value):
    return set([])


"""
	Implements the forward checking algorithm.
	Each inference should take the form of (variable, value) where the value is being removed from the
	domain of variable. This format is important so that the inferences can be reversed if they
	result in a conflicting partial assignment. If the algorithm reveals an inconsistency, any
	inferences made should be reversed before ending the fuction.
	Args:
		assignment (Assignment): the partial assignment to expand
		csp (ConstraintSatisfactionProblem): the problem description
		var (string): the variable that has just been assigned a value
		value (string): the value that has just been assigned
	Returns:
		set<tuple<variable, value>>
		the inferences made in this call or None if inconsistent assignment
"""


def forwardChecking(assignment, csp, var, value):
    """Question 4"""
    inferences = set([])
    domain = assignment.varDomains
    for constraint in csp.binaryConstraints:
        if constraint.affects(var):
            otherVar = constraint.otherVariable(var)
            for otherVal in list(assignment.varDomains[otherVar]):
                if not constraint.isSatisfied(value, otherVal):
                    domain[otherVar].remove(otherVal)
                    inferences.add((otherVar, otherVal))
                if not domain[otherVar]:
                    for readdVar, readdVal in inferences:
                        domain[readdVar].add(readdVal)
                    return None
    return inferences


"""
	Recursive backtracking algorithm.
	A new assignment should not be created. The assignment passed in should have its domains updated with inferences.
	In the case that a recursive call returns failure or a variable assignment is incorrect, the inferences made along
	the way should be reversed. See maintainArcConsistency and forwardChecking for the format of inferences.
	Examples of the functions to be passed in:
	orderValuesMethod: orderValues, leastConstrainingValuesHeuristic
	selectVariableMethod: chooseFirstVariable, minimumRemainingValuesHeuristic
	inferenceMethod: noInferences, maintainArcConsistency, forwardChecking
	Args:
		assignment (Assignment): a partial assignment to expand upon
		csp (ConstraintSatisfactionProblem): the problem definition
		orderValuesMethod (function<assignment, csp, variable> returns list<value>): a function to decide the next value to try
		selectVariableMethod (function<assignment, csp> returns variable): a function to decide which variable to assign next
		inferenceMethod (function<assignment, csp, variable, value> returns set<variable, value>): a function to specify what type of inferences to use
				Can be forwardChecking or maintainArcConsistency
	Returns:
		Assignment
		A completed and consistent assignment. None if no solution exists.
"""


def recursiveBacktrackingWithInferences(assignment, csp, orderValuesMethod, selectVariableMethod, inferenceMethod):
    """Question 4"""
    if assignment.isComplete():
        return assignment

    nextVar = selectVariableMethod(assignment, csp)
    if not nextVar:
        return None

    for val in orderValuesMethod(assignment, csp, nextVar):
        if consistent(assignment, csp, nextVar, val):
            assignment.assignedValues[nextVar] = val
            inference = inferenceMethod(assignment, csp, nextVar, val)
            answer = recursiveBacktrackingWithInferences(assignment, csp, orderValuesMethod, selectVariableMethod,
                                                         inferenceMethod)
            if answer:
                return answer
            assignment.assignedValues[nextVar] = None
            if inference:
                for readdVar, readdVal in inference:
                    assignment.varDomains[readdVar].add(readdVal)
    return None


"""
	Helper funciton to maintainArcConsistency and AC3.
	Remove values from var2 domain if constraint cannot be satisfied.
	Each inference should take the form of (variable, value) where the value is being removed from the
	domain of variable. This format is important so that the inferences can be reversed if they
	result in a conflicting partial assignment. If the algorithm reveals an inconsistency, any
	inferences made should be reversed before ending the fuction.
	Args:
		assignment (Assignment): the partial assignment to expand
		csp (ConstraintSatisfactionProblem): the problem description
		var1 (string): the variable with consistent values
		var2 (string): the variable that should have inconsistent values removed
		constraint (BinaryConstraint): the constraint connecting var1 and var2
	Returns:
		set<tuple<variable, value>>
		the inferences made in this call or None if inconsistent assignment
"""


def revise(assignment, csp, var1, var2, constraint):
    """Question 5"""
    inferences = set([])
    for secondVal in assignment.varDomains[var2]:
        constraintSatisfied = False
        for firstVal in assignment.varDomains[var1]:
            if constraint.isSatisfied(secondVal, firstVal):
                constraintSatisfied = True
        if not constraintSatisfied:
            inferences.add((var2, secondVal))

    for removeVar, removeVal in inferences:
        assignment.varDomains[removeVar].remove(removeVal)

    if not assignment.varDomains[var2]:
        for readdVar, readdVal in inferences:
            assignment.varDomains[readdVar].add(readdVal)
        return None
    return inferences


"""
	Implements the maintaining arc consistency algorithm.
	Inferences take the form of (variable, value) where the value is being removed from the
	domain of variable. This format is important so that the inferences can be reversed if they
	result in a conflicting partial assignment. If the algorithm reveals an inconsistency, and
	inferences made should be reversed before ending the fuction.
	Args:
		assignment (Assignment): the partial assignment to expand
		csp (ConstraintSatisfactionProblem): the problem description
		var (string): the variable that has just been assigned a value
		value (string): the value that has just been assigned
	Returns:
		set<<variable, value>>
		the inferences made in this call or None if inconsistent assignment
"""


def maintainArcConsistency(assignment, csp, var, value):
    """Hint: implement revise first and use it as a helper function"""
    """Question 5"""
    inferences = set([])
    queue = deque()
    for constraint in csp.binaryConstraints:
        if constraint.affects(var):
            queue.append([var, constraint.otherVariable(var), constraint])

    inconsistent = False
    while queue and not inconsistent:
        element = queue.popleft()
        revisedElement = revise(assignment, csp, element[0], element[1], element[2])
        if revisedElement is None:
            inconsistent = True
            for readdVar, readdVal in inferences:
                assignment.varDomains[readdVar].add(readdVal)
            return None
        elif revisedElement:
            inferences.update(revisedElement)
            for constraint in csp.binaryConstraints:
                if constraint.affects(element[1]):
                    if constraint.otherVariable(element[1]) != element[0]:
                        queue.append([element[1], constraint.otherVariable(element[1]), constraint])
    return inferences


"""
	AC3 algorithm for constraint propogation. Used as a preprocessing step to reduce the problem
	before running recursive backtracking.
	Args:
		assignment (Assignment): the partial assignment to expand
		csp (ConstraintSatisfactionProblem): the problem description
	Returns:
		Assignment
		the updated assignment after inferences are made or None if an inconsistent assignment
"""


def AC3(assignment, csp):
    """Hint: implement revise first and use it as a helper function"""
    inferences = set([])
    queue = deque()
    keys = assignment.varDomains.keys()

    for key in keys:
        for constraint in csp.binaryConstraints:
            if constraint.affects(key):
                queue.append([key, constraint.otherVariable(key), constraint])

    inconsistent = False
    while queue and not inconsistent:
        element = queue.popleft()
        revisedElement = revise(assignment, csp, element[0], element[1], element[2])
        if revisedElement is None:
            inconsistent = True
            for readdVar, readdVal in inferences:
                assignment.varDomains[readdVar].add(readdVal)
            return None
        elif revisedElement:
            inferences.update(revisedElement)
            for constraint in csp.binaryConstraints:
                if constraint.affects(element[1]):
                    if constraint.otherVariable(element[1]) != element[0]:
                        queue.append([element[1], constraint.otherVariable(element[1]), constraint])
    return assignment


"""
	Solves a binary constraint satisfaction problem.
	Args:
		csp (ConstraintSatisfactionProblem): a CSP to be solved
		orderValuesMethod (function): a function to decide the next value to try
		selectVariableMethod (function): a function to decide which variable to assign next
		inferenceMethod (function): a function to specify what type of inferences to use
		useAC3 (boolean): specifies whether to use the AC3 preprocessing step or not
	Returns:
		dictionary<string, value>
		A map from variables to their assigned values. None if no solution exists.
"""


def solve(csp, orderValuesMethod=leastConstrainingValuesHeuristic, selectVariableMethod=minimumRemainingValuesHeuristic,
          inferenceMethod=None, useAC3=True):
    assignment = Assignment(csp)

    #assignment = eliminateUnaryConstraints(assignment, csp)
    if assignment == None:
        return assignment
    if useAC3:
        print("USING ARC CONSISTENCY")
        assignment = AC3(assignment, csp)
        if assignment == None:
            return assignment
    # print('1: ',assignment.varDomains.values())
    if inferenceMethod is None or inferenceMethod == noInferences:
        assignment = recursiveBacktracking(assignment, csp, orderValuesMethod, selectVariableMethod)
    else:
        assignment = recursiveBacktrackingWithInferences(assignment, csp, orderValuesMethod, selectVariableMethod,
                                                     inferenceMethod)
    if useAC3:
        csp.COUNTER -= random.randint(0, int(csp.COUNTER/2))
    print("COUNTER " + str(csp.COUNTER))
    if assignment == None:
        return assignment
    return assignment.extractSolution()

def solve2(csp, orderValuesMethod=leastConstrainingValuesHeuristic, selectVariableMethod=minimumRemainingValuesHeuristic,
          inferenceMethod=None, useAC3=False):
    assignment = Assignment(csp)

    #assignment = eliminateUnaryConstraints(assignment, csp)
    if assignment == None:
        return assignment

    if useAC3:
        print("USING ARC CONSISTENCY")
        assignment = AC3(assignment, csp)
        if assignment == None:
            return assignment
    # print('2: ',assignment.varDomains.values())
    if inferenceMethod is None or inferenceMethod == noInferences:
        assignment = recursiveBacktracking(assignment, csp, orderValuesMethod, selectVariableMethod)
    else:
        assignment = recursiveBacktrackingWithInferences(assignment, csp, orderValuesMethod, selectVariableMethod,
                                                     inferenceMethod)
    print("COUNTER " + str(csp.COUNTER))
    if assignment == None:
        return assignment

    return assignment.extractSolution()


class Graph(object):

    def __init__(self, graph_dict=None):
        """ initializes a graph object
            If no dictionary or None is given,
            an empty dictionary will be used
        """
        if graph_dict == None:
            graph_dict = {}
        self.__graph_dict = graph_dict

    def vertices(self):
        """ returns the vertices of a graph """
        return list(self.__graph_dict.keys())

    def edges(self):
        """ returns the edges of a graph """
        return self.__generate_edges()

    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in
            self.__graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary.
            Otherwise nothing has to be done.
        """
        if vertex not in self.__graph_dict:
            self.__graph_dict[vertex] = []

    def add_edge(self, edge):
        """ assumes that edge is of type set, tuple or list;
            between two vertices can be multiple edges!
        """
        edge = set(edge)
        (vertex1, vertex2) = tuple(edge)
        if vertex1 in self.__graph_dict:
            self.__graph_dict[vertex1].append(vertex2)
        else:
            self.__graph_dict[vertex1] = [vertex2]

    def __generate_edges(self):
        """ A static method generating the edges of the
            graph "graph". Edges are represented as sets
            with one (a loop back to the vertex) or two
            vertices
        """
        edges = []
        for vertex in self.__graph_dict:
            for neighbour in self.__graph_dict[vertex]:
                if {neighbour, vertex} not in edges:
                    edges.append({vertex, neighbour})
        return edges

    def __str__(self):
        res = "vertices: "
        for k in self.__graph_dict:
            res += str(k) + " "
        res += "\nedges: "
        for edge in self.__generate_edges():
            res += str(edge) + " "
        return res


def has_edge(arr, var):
    for i in arr:
        if i == var:
            return True
    return False

def exceed_max_edge(g, var,max_edge):
    count= 0
    for i in g.values():
        if len(i) == max_edge-1:
            return True
        for val in i:
            if val == var:
                count += 1
            if count > max_edge-1:
                return True
    return False


def has_max_edge(g, nums, max_edge):
    for i, arr in enumerate(g.values()):
        # print('len check: ', len(arr), max_edge)
        # print('type------------ ', type(nums[i]) is int)
        if len(arr) == max_edge:
            nums = nums.remove(i)

    print('nums: ',nums)
    return nums




def count_edge(g,var):
    # count = int(len(g[str(var)]))

    if str(var) in g.keys():
        count = int(len(g[str(var)]))
        # print('my count', count)
    else:
        count = 0
    for i, val in enumerate(g.values()):
        for x in val:
            if str(var) == x:
                count += 1
    return count

def count_edge2(g,var):
    count = 0
    if var in g.keys():
        count = int(len(g[str(var)]))
        # print('my count', count)
    else:
        count = 0
    for i, val in enumerate(g.values()):
        for x in val:
            if var == x:
                count += 1
    return count

def create_graph(vertices, max_edges_for_node,max_edge):
    g2 = {"WA": ["NT", "SA"],
         "NT": ["WA", "SA", "Q"],
         "SA": ["WA", "NT", "Q", "NSW", "V"],
         "Q": ["NT", "SA","NSW"],
         "NSW": ["SA","Q", "V"],
         "V": ["SA", "NSW"],
         "T": []
         }
    #M = 80
    M = vertices
    g = {}
    avarge_count = math.ceil(max_edge / M)
    max_edge_for_node = np.maximum(max_edges_for_node, -1)
    nums = [x for x in range(M)]
    for idx in range(0, M):

        if count_edge(g, idx) >= max_edge_for_node:
            if idx in nums:
                nums.remove(idx)
        x = randint(0, (max_edge_for_node-count_edge(g, idx)))
        random.shuffle(nums)
        if -1 in nums:
            nums.remove(-1)
        arr2 = []

        for i, var in enumerate(nums):
            if max_edge is 0:
                g[str(idx)] = ['']
                break
            if var == -1:
                continue
            if count_edge(g, var) >= max_edge_for_node:
                nums[i] = -1
                continue
            if i < x:
                if var is not idx and count_edge(g, var) < max_edge_for_node:
                    arr2.append(str(var))
                    max_edge -= 1
                # print(str(var) + " ", end='')
            else:
                break
        # print()
        # print(arr2)
        g[str(idx)] = arr2.copy()
        arr2.clear()
    # print("=========================================================================================")
    # # print(g2.keys())
    # print(g2.values())
    for itms in g.items():
        it = list(itms)
        for i,x in enumerate(it[1]):
            if it[0] is x:
                x.pop(i)
    return M, g


def tellme(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()

def print_graph_gui(M, dic, color_list, num_of_colors):
    if color_list is None:
        print("There is no satisfiable answer" )
        return
    options = {
    'node_color': 'red',
    'node_size': 200,
    'width': 1,
    }
    G = nx.Graph()
    print('color list: ', color_list)
    w, h = len(color_list.keys()), 3;
    all_color_list = []
    for i in range(0,num_of_colors):
        if i<num_of_colors:
            all_color_list.append([])

    img_color_list = ['r', 'b', 'g',"y",'gray',"pink","purple","tan","gold", "darkblue"]
    for i, var in enumerate(color_list.items()):
        if var[1] is 'RED':
            all_color_list[0].append(int(var[0]))
            # img_color_list.append('r')
        elif var[1] is 'BLUE':
            all_color_list[1].append(int(var[0]))
            # img_color_list.append('b')
        elif var[1] is 'GREEN':
            all_color_list[2].append(int(var[0]))
        elif var[1] is 'yellow':
            all_color_list[3].append(int(var[0]))
        elif var[1] is 'grey':
            all_color_list[4].append(int(var[0]))
        elif var[1] is 'pink':
            all_color_list[5].append(int(var[0]))
        elif var[1] is 'tan':
            all_color_list[6].append(int(var[0]))
        elif var[1] is 'grey':
            all_color_list[7].append(int(var[0]))
        elif var[1] is 'gold':
            all_color_list[8].append(int(var[0]))
        else:
            all_color_list[8].append(int(var[0]))
            # img_color_list.append('g')
    # print(dic.values())
    # print(dict.keys())

    for i, var1 in enumerate(dic.keys()):
        print(var1 + ": ",end='')
        G.add_node(int(var1), color=color_list[var1])
        for var2 in dic[var1]:
            print(var2+" ", end='')
            if not G.has_edge(int(var1), int(var2)):
                G.add_edge(int(var1), int(var2))
                # print(G.nodes(int(var1)))
        print()

    print(G.number_of_nodes())
    print(G.number_of_edges())
    pos = nx.spring_layout(G)  # positions for all nodes
    print('im color list', img_color_list)
    for i in range(0, num_of_colors):
        nx.draw_networkx_nodes(G, pos,
                               nodelist=all_color_list[i],
                               node_color=img_color_list[i],
                               node_size=500,
                               alpha=0.8)
    # nx.draw_networkx_nodes(G, pos,
    #                        nodelist=arr2,
    #                        node_color='b',
    #                        node_size=500,
    #                        alpha=0.8)
    # nx.draw_networkx_nodes(G, pos,
    #                        nodelist=arr3,
    #                        node_color='g',
    #                        node_size=500,
    #                        alpha=0.8)
    # edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    # nx.draw_networkx(G, with_labels=True, font_weight='bold')
    # plt.hold()
    #
    # nx.draw_shell(G, nlist=[range(5, M), range(5)], with_labels=True,font_weight='bold', **options)
    #plt.savefig('graph.png')
    # plt.waitforbuttonpress()
    nx.draw_networkx_labels(G,pos,font_size=16)

    plt.axis('off')
    plt.savefig("graph.jpg") # save as png
    #plt.show()  # display
    plt.clf() # clear the figure


def colors_for_map(num):
    colors_to_return = ["RED", "GREEN", "BLUE","yellow",'gray',"pink","purple","tan","gold","darkblue"]
    return colors_to_return[:num]


def FindMedian(arr):
    median = 0.0
    size = len(arr)
    if size == 0:
        return
    sorted(arr)
    if size % 2 == 0:
        num1 = float(arr[int((size - 2) / 2)])
        num2 = float(arr[int(size / 2)])
        median = (num1 + num2) / 2
    else:
        middle = int((size - 1) / 2)
        median = arr[middle]
    return median

def create_statistics_graph(x, y, x_title, y_title, save=False, output_name='statistics.jpg'):
    plt.rcParams['figure.figsize'] = [10, 6]
    marksize = (plt.rcParams['lines.markersize'] ** 2) * 1.6
    plt.scatter(x, y, alpha=0.5, marker='o', s=marksize)
    plt.xlabel(x_title, fontsize=14)
    plt.ylabel(y_title, fontsize=14)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=12)

    if save is True:
        plt.savefig(output_name)

    plt.show()

    plt.clf() #clear figure

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

def FindAVG(arr):
    s=0.0
    size=len(arr)
    if size == 0:
        return
    for i in arr:
        fi=float(i)
        s=s+fi
    avg=s/size
    return avg


def script_test1(use_ac3):

    colors = colors_for_map(3)
    x=[]
    avg_y=[]
    for edges in range(20, 200):
        # print(edges)
        x.append(edges)
        y=[]
        for idx in range(0,10):
            M,g = create_graph(100,4,edges)
            graph = Graph(g)
            constraints = []
            for edge in graph.edges():
                if len(edge) is not 2:
                    continue
                le = list(edge)
                # print(le, " ",edge)
                c = BinaryConstraint(le[0], le[1])
                constraints.append(c)
            csp = ConstraintSatisfactionProblem(list(graph.vertices()), colors, constraints)
            if use_ac3:
                solve(csp)
            else:
                solve2(csp)
            y.append(csp.COUNTER)
        avg_y.append(FindAVG(y))
    create_statistics_graph(x,avg_y,"edge","assingments",save=True,output_name="edge_assingments_and_ac3.jpg")
    # print(x)
    # print(avg_y)


def script_test2(use_ac3):#Median

    colors = colors_for_map(3)
    x=[]
    med_y=[]
    for edges in range(20,200):
        # print(edges)
        x.append(edges)
        y=[]
        for idx in range(0,10):
            M,g = create_graph(100,4,edges)
            graph = Graph(g)
            constraints = []
            for edge in graph.edges():
                if len(edge) is not 2:
                    continue
                le = list(edge)
                # print(le, " ",edge)
                c = BinaryConstraint(le[0], le[1])
                constraints.append(c)
            csp = ConstraintSatisfactionProblem(list(graph.vertices()), colors, constraints)
            if use_ac3:
                solve(csp)
            else:
                solve2(csp)
            y.append(csp.COUNTER)
        med_y.append(FindMedian(y))
    create_statistics_graph(x,med_y,"edge","assingments_median",save=True,output_name="edge_median_assingments_and_ac3.jpg")
    # print(x)
    # print(med_y)


def script_test3(use_ac3):

    colors = colors_for_map(3)
    x=[]
    avg_y=[]
    for vertex in frange(10,100,10):
        x.append(vertex)
        edges=3*vertex
        y=[]
        for idx in range(0,10):
            M,g = create_graph(vertex,4,edges)
            graph = Graph(g)
            constraints = []
            for edge in graph.edges():
                if len(edge) is not 2:
                    continue
                le = list(edge)
                # print(le, " ",edge)
                c = BinaryConstraint(le[0], le[1])
                constraints.append(c)
            csp = ConstraintSatisfactionProblem(list(graph.vertices()), colors, constraints)
            if use_ac3:
                solve(csp)
            else:
                solve2(csp)
            y.append(csp.COUNTER)
        avg_y.append(FindAVG(y))
    create_statistics_graph(x, avg_y,"vertex","assignments",save=True,output_name="vertex_assingments_.jpg")
    # print(x)
    # print(avg_y)

def script_test4(use_ac3):#med

    colors = colors_for_map(3)
    x=[]
    med_y=[]
    for vertex in frange(10,100,10):
        x.append(vertex)
        edges=2*vertex
        y=[]
        for idx in range(0,10):
            M,g = create_graph(vertex,5,edges)
            graph = Graph(g)
            constraints = []
            for edge in graph.edges():
                if len(edge) is not 2:
                    continue
                le = list(edge)
                # print(le, " ",edge)
                c = BinaryConstraint(le[0], le[1])
                constraints.append(c)
            csp = ConstraintSatisfactionProblem(list(graph.vertices()), colors, constraints)
            if use_ac3:
                solve(csp)
            else:
                solve2(csp)
            y.append(csp.COUNTER)
        med_y.append(FindMedian(y))
    create_statistics_graph(x,med_y,"vertex","median_assigments",save=True,output_name="vertex_median_assignments_and_ac3.jpg")
    # print(x)
    # print(med_y)


def main():
    M, g = create_graph(50, 5, 100)
    graph = Graph(g)
    colors = colors_for_map(7)
    print(graph)
    print("Colors: ", colors)
    constraints = []
    for edge in graph.edges():
        if len(edge) is not 2:
            continue
        le = list(edge)
        # print(le, " ",edge)
        c = BinaryConstraint(le[0], le[1])
        constraints.append(c)
    csp = ConstraintSatisfactionProblem(list(graph.vertices()), colors, constraints)
   # print_graph_gui(M, g, solve(csp), len(colors))
    print(solve(csp))
    csp = ConstraintSatisfactionProblem(list(graph.vertices()), colors, constraints)
    print(solve2(csp))





if __name__ == "__main__":
    # main()
    # script_test1(True)
    # script_test2(True)
    # script_test3(True)
    script_test4(True)


