
int isValid() {

  double actual = 2.5;
  double s_sum = 0.0;
  double q_sum = 0.0;
  double rand1=0.1, rand2=0.9;
  double expected=0.0;
  int ii,jj;
  double diff=0.0;


  for (ii=0; ii<nx; ii++) {
    s_sum+=s[ii]*rand1*rand2;
    q_sum+=q[ii]*rand1*rand2;
    }

  expected = s_sum/q_sum;

  diff=abs(expected-actual);

  fprintf(stderr, "expected=%f\n",expected);
  fprintf(stderr, "actual=%f\n",actual);
  fprintf(stderr, "diff=%f\n",diff);
  fprintf(stderr, "diff=%d\n",(diff < 0.00000001));

  if (diff < 0.00000001)
    return 1;
  else
    return 0;
}




