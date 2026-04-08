#include <cstdlib>
#include <ctime>
#include "rng.hpp"

void initRandom(){
    std::srand(std::time({}));
}

float randomFloat(){
    return (float(std::rand())/RAND_MAX)*2.0 - 1.0;
}
