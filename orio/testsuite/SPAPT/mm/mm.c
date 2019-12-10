
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
    param T1_I[] = [1,2,4,8,16,32,64,128,256,512,1024,2048];
    param T1_J[] = [1,2,4,8,16,32,64,128,256,512,1024,2048];
    param T1_K[] = [1,2,4,8,16,32,64,128,256,512,1024,2048];

    param T1_Ia[] = [1,2,4,8,16,32,64,128,256,512,1024,2048];
    param T1_Ja[] = [1,2,4,8,16,32,64,128,256,512,1024,2048];
    param T1_Ka[] = [1,2,4,8,16,32,64,128,256,512,1024,2048];

    param U_I[] = range(1,31);
    param U_J[] = range(1,31);
    param U_K[] = range(1,31);

    param SCREP[] = [False,True];
    param VEC[] = [False,True];
    param OMP[] = [False,True];

    param PAR[] = [False,True];

    constraint tileI = ((T1_Ia == 1) or (T1_Ia % T1_I == 0));
    constraint tileJ = ((T1_Ja == 1) or (T1_Ja % T1_J == 0));
    constraint tileK = ((T1_Ka == 1) or (T1_Ka % T1_K == 0));
    constraint reg_capacity = (U_I*U_J + U_I*U_K + U_J*U_K <= 150);
    constraint unroll_limit = ((U_I == 1) or (U_J == 1) or (U_K == 1));
  }

  def search
  {
    arg algorithm = 'DLMT';
    arg total_runs = 1;
    arg dlmt_federov_sampling = 30;
    arg dlmt_extra_experiments = 1;
    arg dlmt_design_multiplier = 1.2;
    arg dlmt_steps = 4;
    arg dlmt_aov_threshold = 0.05;

    arg dlmt_linear = '["T1_I", "T1_J", "T1_K", "T1_Ia", "T1_Ja", "T1_Ka", "U_I", "U_J", "U_K", "SCREP", "VEC", "OMP", "PAR"]';
    arg dlmt_quadratic = '["T1_I", "T1_J", "T1_K", "T1_Ia", "T1_Ja", "T1_Ka", "U_I", "U_J", "U_K"]';
    arg dlmt_cubic = '["T1_I", "T1_J", "T1_K", "T1_Ia", "T1_Ja", "T1_Ka", "U_I", "U_J", "U_K"]';
  }

  def input_params
  {
    param CONT = 1000;
    param NCONT = 1000;
    param M = 2000;
    param N = 2000;
    param K = 500;
  }
  def input_vars
  {
    decl static double A[M][K] = random;
    decl static double B[K][N] = random;
    decl static double C[M][N] = 0;
  }
  def validation
  {
  arg validation_file = 'validation_3x.c';
  }

) @*/

int i, j, k;
int ii, jj, kk;
int iii, jjj, kkk;

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

/*@ begin Loop(
  transform Composite(
    tile = [('i',T1_I,'ii'),('j',T1_J,'jj'),('k',T1_K,'kk'),
            (('ii','i'),T1_Ia,'iii'),(('jj','j'),T1_Ja,'jjj'),(('kk','k'),T1_Ka,'kkk')],
    unrolljam = (['i','j','k'],[U_I,U_J,U_K]),
    scalarreplace = (SCREP, 'double', 'scv_'),
    vector = (VEC, ['ivdep','vector always']),
    openmp = (OMP, 'omp parallel for private(iii,jjj,kkk,ii,jj,kk,i,j,k)')
  )
  for(i=0; i<=M-1; i++)
    for(j=0; j<=N-1; j++)
      for(k=0; k<=K-1; k++)
        C[i][j] = C[i][j] + A[i][k] * B[k][j];

) @*/

/*@ end @*/
/*@ end @*/
