#include <chrono>
#include <iostream>

// use of unsigned types improves performance
unsigned long count_or_models_by_enumeration(unsigned short num_variables) {
    unsigned long result = 0;
    double iteration_count = std::pow(2, num_variables);
    for (unsigned long i = 0; i < iteration_count; i++) {
        unsigned long remaining_value = i;
        for (unsigned short var_idx = 0; var_idx < num_variables; var_idx++) {
            if (remaining_value % 2 == 1) {
                result += 1;
                break;
            }
            remaining_value /= 2;
        }
    }
    return result;
}

unsigned long count_and_models_by_enumeration(unsigned short num_variables) {
    unsigned long result = 0;
    double iteration_count = std::pow(2, num_variables);
    for (unsigned long i = 0; i < iteration_count; i++) {
        unsigned long remaining_value = i;
        for (unsigned short var_idx = 0; var_idx < num_variables; var_idx++) {
            if (remaining_value % 2 == 0) {
                goto Assignment_End;
            }
            remaining_value /= 2;
        }
        result += 1;
        Assignment_End:;
    }
    return result;
}

int main() {
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    long solutionCount = count_or_models_by_enumeration(30);
    std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    std::cout << "OR - Time: " << (endTime - startTime).count() / 1e9
        << " s, Solutions: " << solutionCount << "\n";

    startTime = std::chrono::steady_clock::now();
    solutionCount = count_and_models_by_enumeration(30);
    endTime = std::chrono::steady_clock::now();
    std::cout << "AND - Time: " << (endTime - startTime).count() / 1e9
        << " s, Solutions: " << solutionCount << "\n";
}
