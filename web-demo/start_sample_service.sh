#!/bin/sh

#python3 sample_service.py $model
for model in 'formal' 'impolite' 'informal' 'not_offensive' 'not_receptive' 'offensive' 'polite' 'receptive'; do
    python3 web-demo/sample_service.py --model=$model
done
