#!/usr/bin/zsh
find $1/bicgkernel/ -name "*.stderr" | xargs -I{} zsh -c 'echo {} && grep -B 5 -A 5 -n "Traceback" {}'
