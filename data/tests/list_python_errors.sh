#!/usr/bin/zsh
find $1/*/ -name "*.stderr" | xargs -I{} zsh -c 'echo {} && grep -B 5 -A 5 -n "Traceback" {}'
