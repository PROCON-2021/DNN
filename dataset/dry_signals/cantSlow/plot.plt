set grid
set terminal pdf
set datafile separator ","
set xlabel "秒[sec]"

set key outside

set output "e.pdf"
plot "1.csv" using 1:2 with lines title "1" , "1.csv" using 1:3 with lines title "2" , "1.csv" using 1:4 with lines title "3" , "1.csv" using 1:5 with lines title "4"
plot "2.csv" using 1:2 with lines title "1" , "2.csv" using 1:3 with lines title "2" , "2.csv" using 1:4 with lines title "3" , "2.csv" using 1:5 with lines title "4"
plot "3.csv" using 1:2 with lines title "1" , "3.csv" using 1:3 with lines title "2" , "3.csv" using 1:4 with lines title "3" , "3.csv" using 1:5 with lines title "4"
plot "4.csv" using 1:2 with lines title "1" , "4.csv" using 1:3 with lines title "2" , "4.csv" using 1:4 with lines title "3" , "4.csv" using 1:5 with lines title "4"
plot "5.csv" using 1:2 with lines title "1" , "5.csv" using 1:3 with lines title "2" , "5.csv" using 1:4 with lines title "3" , "5.csv" using 1:5 with lines title "4"
plot "6.csv" using 1:2 with lines title "1" , "6.csv" using 1:3 with lines title "2" , "6.csv" using 1:4 with lines title "3" , "6.csv" using 1:5 with lines title "4"
plot "7.csv" using 1:2 with lines title "1" , "7.csv" using 1:3 with lines title "2" , "7.csv" using 1:4 with lines title "3" , "7.csv" using 1:5 with lines title "4"
plot "8.csv" using 1:2 with lines title "1" , "8.csv" using 1:3 with lines title "2" , "8.csv" using 1:4 with lines title "3" , "8.csv" using 1:5 with lines title "4"
plot "9.csv" using 1:2 with lines title "1" , "9.csv" using 1:3 with lines title "2" , "9.csv" using 1:4 with lines title "3" , "9.csv" using 1:5 with lines title "4"
plot "10.csv" using 1:2 with lines title "1" , "10.csv" using 1:3 with lines title "2" , "10.csv" using 1:4 with lines title "3" , "10.csv" using 1:5 with lines title "4"
plot "11.csv" using 1:2 with lines title "1" , "11.csv" using 1:3 with lines title "2" , "11.csv" using 1:4 with lines title "3" , "11.csv" using 1:5 with lines title "4"
plot "12.csv" using 1:2 with lines title "1" , "12.csv" using 1:3 with lines title "2" , "12.csv" using 1:4 with lines title "3" , "12.csv" using 1:5 with lines title "4"
plot "13.csv" using 1:2 with lines title "1" , "13.csv" using 1:3 with lines title "2" , "13.csv" using 1:4 with lines title "3" , "13.csv" using 1:5 with lines title "4"
plot "14.csv" using 1:2 with lines title "1" , "14.csv" using 1:3 with lines title "2" , "14.csv" using 1:4 with lines title "3" , "14.csv" using 1:5 with lines title "4"
plot "15.csv" using 1:2 with lines title "1" , "15.csv" using 1:3 with lines title "2" , "15.csv" using 1:4 with lines title "3" , "15.csv" using 1:5 with lines title "4"
plot "16.csv" using 1:2 with lines title "1" , "16.csv" using 1:3 with lines title "2" , "16.csv" using 1:4 with lines title "3" , "16.csv" using 1:5 with lines title "4"
plot "17.csv" using 1:2 with lines title "1" , "17.csv" using 1:3 with lines title "2" , "17.csv" using 1:4 with lines title "3" , "17.csv" using 1:5 with lines title "4"
plot "18.csv" using 1:2 with lines title "1" , "18.csv" using 1:3 with lines title "2" , "18.csv" using 1:4 with lines title "3" , "18.csv" using 1:5 with lines title "4"
plot "19.csv" using 1:2 with lines title "1" , "19.csv" using 1:3 with lines title "2" , "19.csv" using 1:4 with lines title "3" , "19.csv" using 1:5 with lines title "4"
plot "20.csv" using 1:2 with lines title "1" , "20.csv" using 1:3 with lines title "2" , "20.csv" using 1:4 with lines title "3" , "20.csv" using 1:5 with lines title "4"
plot "21.csv" using 1:2 with lines title "1" , "21.csv" using 1:3 with lines title "2" , "21.csv" using 1:4 with lines title "3" , "21.csv" using 1:5 with lines title "4"
plot "22.csv" using 1:2 with lines title "1" , "22.csv" using 1:3 with lines title "2" , "22.csv" using 1:4 with lines title "3" , "22.csv" using 1:5 with lines title "4"
plot "23.csv" using 1:2 with lines title "1" , "23.csv" using 1:3 with lines title "2" , "23.csv" using 1:4 with lines title "3" , "23.csv" using 1:5 with lines title "4"
plot "24.csv" using 1:2 with lines title "1" , "24.csv" using 1:3 with lines title "2" , "24.csv" using 1:4 with lines title "3" , "24.csv" using 1:5 with lines title "4"
plot "25.csv" using 1:2 with lines title "1" , "25.csv" using 1:3 with lines title "2" , "25.csv" using 1:4 with lines title "3" , "25.csv" using 1:5 with lines title "4"

unset output
