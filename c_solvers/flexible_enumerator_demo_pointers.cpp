#include <chrono>
#include <iostream>
#include <vector>

/* Solution using plain old pointers; instead, shared pointers could be used
(replace each A* with std::shared_ptr<A> and each new A() with std::make_shared<A>(),
but that is an order of magnitude slower, at least with the Visual Studio compiler
I use (however, it should not be that much slower)*/

class Expression { // pseudo-interface: an abstract base class
public:
	virtual bool is_true() = 0;
	virtual ~Expression() {} // destructor
};

class Variable : public Expression {
public:
	bool value = false;
	virtual bool is_true() {
		return value;
	}
};

class Not : public Expression {
private:
	Expression* expression;
public:
	Not(Expression* expression) : expression(expression) {} // constructor with member initialization

	virtual bool is_true() {
		return !this->expression->is_true();
	}
};

class And : public Expression {
private:
	std::vector<Expression*> expressions;
public:
	And(std::vector<Expression*> expressions) : expressions(expressions) {} // constructor with member initialization

	virtual bool is_true() {
		for (Expression* element : this->expressions) {
			if (!element->is_true()) {
				return false;
			}
		}
		return true;
	}
};

class Or : public Expression {
private:
	std::vector<Expression*> expressions;
public:
	Or(std::vector<Expression*> expressions) : expressions(expressions) {} // constructor with member initialization

	virtual bool is_true() {
		for (Expression* expression : this->expressions) {
			if (expression->is_true()) {
				return true;
			}
		}
		return false;
	}
};

class Problem {
private:
	std::vector<Variable*> variables;
	std::vector<Expression*> constraints;
public:
	Problem(std::vector<Variable*> variables) : variables(variables) {} // constructor with member initialization

	void add_constraint(Expression* constraint) {
		this->constraints.push_back(constraint);
	}

	void clear_constraints() {
		this->constraints.clear();
	}

	unsigned long count_models_by_enumeration() {
		unsigned long result = 0;
		double iteration_count = std::pow(2, this->variables.size());
		for (unsigned long i = 0; i < iteration_count; i++) {
			// Assign
			unsigned long remaining_value = i;
			for (Variable* variable : this->variables) {
				variable->value = remaining_value % 2 == 1;
				remaining_value /= 2;
			}
			// Check SAT
			bool satisfied = true;
			for (Expression* constraint : this->constraints) {
				if (!constraint->is_true()) {
					satisfied = false;
					break;
				}
			}
			if (satisfied) {
				result += 1;
			}
		}
		return result;
	}
};

int main() {
	int num_variables = 20;
	std::vector<Variable*> variables;
	std::vector<Expression*> variables_expr; // type-safety FTW
	for (int i = 0; i < num_variables; i++) {
		Variable* variable = new Variable();
		variables.push_back(variable);
		variables_expr.push_back(variable);
	}

	Problem problem(variables);
	problem.add_constraint(new Or(variables_expr));
	std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
	long solutionCount = problem.count_models_by_enumeration();
	std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
	std::cout << "OR - Time: " << (endTime - startTime).count() / 1e9
		<< " s, Solutions: " << solutionCount << "\n";

	problem.clear_constraints();
	problem.add_constraint(new And(variables_expr));
	startTime = std::chrono::steady_clock::now();
	solutionCount = problem.count_models_by_enumeration();
	endTime = std::chrono::steady_clock::now();
	std::cout << "AND - Time: " << (endTime - startTime).count() / 1e9
		<< " s, Solutions: " << solutionCount << "\n";
}
