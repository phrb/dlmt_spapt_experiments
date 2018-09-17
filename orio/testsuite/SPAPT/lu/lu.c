/*@ begin PerfTuning (
  def build
  {
    arg build_command = 'timeout --kill-after=30s --signal=9 20m gcc -O3 -fopenmp -DDYNAMIC';
    arg libs = '-lm -lrt';
  }

  def performance_counter
  {
    arg repetitions = 10;
  }

  def performance_params
  {
  #  param PERM[] = [
  #   ['i','j'],
  #   ['j','i'],
  #  ];
    param T2_I[] = [1,2,4,8,16,32,64,128,256,512,1024,2048];
    param T2_J[] = [1,2,4,8,16,32,64,128,256,512,1024,2048];
    param T2_Ia[] = [1,2,4,8,16,32,64,128,256,512,1024,2048];
    param T2_Ja[] = [1,2,4,8,16,32,64,128,256,512,1024,2048];

    param U2_I[]  = range(1,31);
    param U2_J[]  = range(1,31);

    param RT2_I[] = [1,2,4,8,16,32];
    param RT2_J[] = [1,2,4,8,16,32];

    param SCR[]  = [False,True];
    param VEC2[] = [False,True];
    param OMP[] = [False,True];

    param U1_K[]  = range(1,31);
    param U1_J[]  = range(1,31);
    param VEC1[] = [False,True];

    constraint tileI2 = ((T2_Ia == 1) or (T2_Ia % T2_I == 0));
    constraint tileJ2 = ((T2_Ja == 1) or (T2_Ja % T2_J == 0));
    constraint reg_capacity = (RT2_I*RT2_J <= 150);
    constraint unroll_limit = ((U2_I == 1) or (U2_J == 1));

  }

  def search
  {
    arg algorithm = 'DLMT';
    arg total_runs = 1;
    arg dlmt_federov_sampling = 100;
    arg dlmt_extra_experiments = 0;
    arg dlmt_design_multiplier = 1.3;
    arg dlmt_steps = 1;
    arg dlmt_aov_threshold = 0.05;
    arg dlmt_linear = '["T2_I", "T2_J", "T2_Ia", "T2_Ja", "U2_I", "U2_J", "RT2_I", "RT2_J", "SCR", "VEC2", "OMP", "U1_K", "U1_J", "VEC1"]';
    arg dlmt_quadratic = '["T2_I", "T2_J", "T2_Ia", "T2_Ja", "U2_I", "U2_J", "RT2_I", "RT2_J","U1_K", "U1_J"]';
    arg dlmt_cubic = '["T2_I", "T2_J", "T2_Ia", "T2_Ja", "U2_I", "U2_J", "RT2_I", "RT2_J", "U1_K", "U1_J"]';
  }

  def input_params
  {
    param N[] = [2000];
  }

  def input_vars
  {
    arg decl_file = 'decl_code.h';
    arg init_file = 'init_code.c';
 }

  def validation {
    arg validation_file = 'validation_3x.c';
  }
) @*/

int i,j, k,t;
int it, jt, kt;
int ii, jj, kk;
int iii, jjj, kkk;

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))



/*@ begin Loop (

   transform Composite(
      unrolljam = (['k','j'],[U1_K,U1_J]),
      scalarreplace = (SCR, 'double'),
      vector = (VEC1, ['ivdep','vector always'])
     )
  for (k=0; k<=N-1; k++) {
    for (j=k+1; j<=N-1; j++)
      A[k][j] = A[k][j]/A[k][k];

   transform Composite(
      tile = [('i',T2_I,'ii'),('j',T2_J,'jj'),
             (('ii','i'),T2_Ia,'iii'),(('jj','j'),T2_Ja,'jjj')],
      unrolljam = (['i','j'],[U2_I,U2_J]),
      scalarreplace = (SCR, 'double'),
      regtile = (['i','j'],[RT2_I,RT2_J]),
      vector = (VEC2, ['ivdep','vector always']),
      openmp = (OMP, 'omp parallel for private(iii,jjj,ii,jj,i,j)')
    )
    for(i=k+1; i<=N-1; i++)
      for (j=k+1; j<=N-1; j++)
        A[i][j] = A[i][j] - A[i][k]*A[k][j];
  }
) @*/
/*@ end @*/

/*@ end @*/
