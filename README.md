# TFG Victor

## How to use

1. Setup for release <br>
    `cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -S . -B build -DCMAKE_BUILD_TYPE=Release`
    
    Setup for debug <br>
    `cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -S . -B build -DCMAKE_BUILD_TYPE=Debug`
2. Compile <br>
    `cmake --build build --parallel`
3. Run <br>
    `./_bin/tfg`
