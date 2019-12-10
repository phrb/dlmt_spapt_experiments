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
    # Cache tiling
    param T1_I[] = [1,2,4,8,16,32,64,128,256,512,1024,2048];
    param T1_J[] = [1,2,4,8,16,32,64,128,256,512,1024,2048];

    param T2_I[] = [1,2,4,8,16,32,64,128,256,512,1024,2048];
    param T2_J[] = [1,2,4,8,16,32,64,128,256,512,1024,2048];

    # Array copy
    #param ACOPY_s[] = [False,True];
    #param ACOPY_q[] = [False,True];

    # Unroll-jam
    param U1_I[] = range(1,31);
    param U_I[]  = range(1,31);
    param U_J[]  = range(1,31);

    # Register tiling
    param RT_I[] = [1,2,4,8,16,32];
    param RT_J[] = [1,2,4,8,16,32];

    # Scalar replacement
    #param SCR[]  = [False,True];

    # Vectorization
    #param VEC1[] = [False,True];
    #param VEC2[] = [False,True];

    # Parallelization
    #param OMP[] = [False,True];

    # Constraints
    constraint tileI = ((T2_I == 1) or (T2_I % T1_I == 0));
    constraint tileJ = ((T2_J == 1) or (T2_J % T1_J == 0));

    constraint reg_capacity = (RT_I*RT_J <= 150);
    constraint unroll_limit = ((U_I == 1) or (U_J == 1));
  }

  def search
  {
    arg algorithm = 'Randomsearch';
    arg total_runs = 400;
  }

  def input_params
  {
  param N[] = [40000];
  }

  def input_vars
  {
  arg decl_file = 'decl.h';
  arg init_file = 'init.c';
  }

  def validation {
    arg validation_file = 'validation_3x.c';
  }
) @*/

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

int i,j, k;
int it, jt, kt;
int ii, jj, kk;
int iii, jjj, kkk;

/*@ begin Loop(

  transform Composite(
    unrolljam = (['i'],[U1_I]),
    vector = (False, ['ivdep','vector always'])
  )
  for (i = 0; i <= ny-1; i++)
    s[i] = 0;

  transform Composite(
    tile = [('i',T1_I,'ii'),('j',T1_J,'jj'),
            (('ii','i'),T2_I,'iii'),(('jj','j'),T2_J,'jjj')],
    unrolljam = (['i','j'],[U_I,U_J]),
    scalarreplace = (False, 'double'),
    regtile = (['i','j'],[RT_I,RT_J]),
    vector = (False, ['ivdep','vector always']),
    openmp = (False, 'omp parallel for private(iii,jjj,ii,jj,i,j)')
  )
  for (i = 0; i <= nx-1; i++) {
    q[i] = 0;
    for (j = 0; j <= ny-1; j++) {
      s[j] = s[j] + r[i]*A[i*ny+j];
      q[i] = q[i] + A[i*ny+j]*p[j];
    }
  }

) @*/
/*@ end @*/

/*@ end @*/
