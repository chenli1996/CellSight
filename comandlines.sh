# close all python processes except this one
# kill -9 $(ps aux | grep '[p]ython' | awk '{print $2}')
kill -9 $(ps | grep '[p]ython' | awk '{print $1}')