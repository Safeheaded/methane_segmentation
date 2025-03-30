#!/bin/bash

# Ustawienie interpretera uv
PYTHON_EXEC=uv

# Definicja listy skryptów i ich argumentów
declare -a scripts=(
    "train.py unet --epochs=1"
    "train.py u2net --epochs=1"
)

# Iteracja przez listę skryptów
i=1
for script in "${scripts[@]}"; do
    echo "Running script $i: $PYTHON_EXEC run $script"
    $PYTHON_EXEC run $script
    if [ $? -ne 0 ]; then
        echo "Script $i ($script) has failed"
        exit 1
    fi
    ((i++))
done

echo "All job done!"
