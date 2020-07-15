#include <chrono>
#include <gecode/search.hh>
#include <gecode/int.hh> // also includes Boolean stuff

using namespace Gecode;

class DemoProblem : public Space {
private:
	BoolVarArray x;
public:
	DemoProblem(int num_variables) : x(*this, num_variables, 0, 1) { //constructor declaration combined with initialization of the field x
        // Constraints:
        rel(*this, BOT_OR, x, 1); // OR(x) == 1
        // Search strategy: select first unassigned variable, assign lowest value (aka 0)
        branch(*this, x, BOOL_VAR_NONE(), BOOL_VAL_MIN());
	}

    // Search support: Space must implement copy() functionality
    DemoProblem(DemoProblem& d) : Space(d) { // copy constructor, called below
        x.update(*this, d.x); // update variables, which means copying them
    }

    virtual Space* copy() {
        return new DemoProblem(*this);
    }

    // Allow printing solution from outside the class (optional method)
    // (if no solution searched yet, then just range of variables printed)
    void print() {
        std::cout << x << std::endl;
    }

    unsigned long count_models_natively() {
        unsigned long result = 0;
        DFS<DemoProblem> search(this); // search engines: DFS, LDS, BAB (+ meta-engines RBS, PBS)
        while (DemoProblem* solution = search.next()) {
            result += 1;
            delete solution; // important, else the RAM will be filled quickly
        }
        return result;
    }
};

int main() {
    DemoProblem* model = new DemoProblem(20);
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    long solutionCount = model->count_models_natively();
    std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    std::cout << "OR - Time: " << (endTime - startTime).count() / 1e9
        << " s, Solutions: " << solutionCount << "\n";
}
