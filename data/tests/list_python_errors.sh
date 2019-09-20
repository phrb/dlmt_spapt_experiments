#!/usr/bin/zsh
find $1/bicgkernel/ -name "*.stderr" | xargs -I{} zsh -c 'echo {} && grep -B 2 -A 2 -n "Traceback" {}'
