int main ( ) { int t , num1 , num2 , cost = 0 , i , num , j , temp ; scanf ( "%d" , & t ) ; for ( i = 0 ; i < t ; i ++ ) { cost = 0 ; scanf ( "%d" , & num ) ; if ( num > 0 ) scanf ( "%d" , & num1 ) ; for ( j = 0 ; j < num - 1 ; j ++ ) { scanf ( "%d" , & num2 ) ; temp = ( num1 < num2 ) ? num1 : num2 ; printf ( "temp : %d " , temp ) ; cost += temp ; printf ( "cost : %d " , cost ) ; num1 = num2 ; } printf ( "%d " , cost ) ; } return 0 ; }