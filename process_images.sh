#!/bin/bash

# Navigate to the bin directory
cd bin

# Iterate through all .jpg images in ../images
for img in ../images/*.jpg; do
    # Create an output path in the results directory
    output_file="../results4/$(basename "$img" .jpg).txt"

    # Run the program on the current image
    ./demo_phi3v -m ../outputs/phi3v_q4_k.mllm -v ../vocab/phi3v_vocab.mllm -t 6 --limits 3000 -i "$img" -o "$output_file"

    # Print status
    echo "Processed: $img -> $output_file"
done

# Navigate back to the project root
cd ..
