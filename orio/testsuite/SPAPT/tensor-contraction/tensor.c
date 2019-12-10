
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
    let VR = 500;
    let OR = 10;
    param VRANGE = VR;
    param ORANGE = OR;

    param T1_I[] = [1,2,4,8,16,32,64,128,256,512,1024,2048];
    param T1_J[] = [1,2,4,8,16,32,64,128,256,512,1024,2048];
    param T1_K[] = [1,2,4,8,16,32,64,128,256,512,1024,2048];
    param T1_L[] = [1,2,4,8,16,32,64,128,256,512,1024,2048];
    param T1_M[] = [1,2,4,8,16,32,64,128,256,512,1024,2048];

    param T1_Ia[] = [1,2,4,8,16,32,64,128,256,512,1024,2048];
    param T1_Ja[] = [1,2,4,8,16,32,64,128,256,512,1024,2048];
    param T1_Ka[] = [1,2,4,8,16,32,64,128,256,512,1024,2048];
    param T1_La[] = [1,2,4,8,16,32,64,128,256,512,1024,2048];
    param T1_Ma[] = [1,2,4,8,16,32,64,128,256,512,1024,2048];

    param U1_I[] = range(1,31);
    param U1_J[] = range(1,31);
    param U1_K[] = range(1,31);
    param U1_L[] = range(1,31);
    param U1_M[] = range(1,31);

    param SCREP[] = [False,True];
    param VEC[] = [False,True];
    param OMP[] = [False,True];

    constraint tileI = ((T1_Ia == 1) or (T1_Ia % T1_I == 0));
    constraint tileJ = ((T1_Ja == 1) or (T1_Ja % T1_J == 0));
    constraint tileK = ((T1_Ka == 1) or (T1_Ka % T1_K == 0));
    constraint tileL = ((T1_La == 1) or (T1_La % T1_L == 0));
    constraint tileM = ((T1_Ma == 1) or (T1_Ma % T1_M == 0));
    constraint reg_capacity = (U1_I*U1_J*U1_K*U1_L +  U1_I*U1_M*U1_K*U1_L + U1_J*U1_M <= 130);
    constraint unroll_limit = ((U1_I == 1) or (U1_J == 1) or (U1_K == 1) or (U1_L == 1) or (U1_M == 1) );
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

    arg dlmt_linear = '["T1_I", "T1_J", "T1_K", "T1_L", "T1_M", "T1_Ia", "T1_Ja", "T1_Ka", "T1_La", "T1_Ma", "U1_I", "U1_J", "U1_K", "U1_L", "U1_M", "SCREP", "VEC", "OMP"]';
    arg dlmt_quadratic = '["T1_I", "T1_J", "T1_K", "T1_L", "T1_M", "T1_Ia", "T1_Ja", "T1_Ka", "T1_La", "T1_Ma", "U1_I", "U1_J", "U1_K", "U1_L", "U1_M"]';
    arg dlmt_cubic = '["T1_I", "T1_J", "T1_K", "T1_L", "T1_M", "T1_Ia", "T1_Ja", "T1_Ka", "T1_La", "T1_Ma", "U1_I", "U1_J", "U1_K", "U1_L", "U1_M"]';
  }

  def input_params
  {
    param VSIZE = 500;
    param OSIZE = 25;
    param V = 500;
    param O = 25;

    }

  def input_vars
  {

    decl dynamic double A2[V][O] = random;
    decl dynamic double T[V][O][O][O] = random;
    decl dynamic double R[V][V][O][O] = 0;
  }

  def validation
  {
  arg validation_file = 'validation_3x.c';
  }
) @*/

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

int i, j, k,l,m;
int ii, jj, kk,ll,mm;
int iii, jjj, kkk,lll,mmm;


/*@ begin Loop(
  transform Composite(
    tile = [('i',T1_I,'ii'),('j',T1_J,'jj'),('k',T1_K,'kk'),('l',T1_L,'ll'),('m',T1_M,'mm'),
            (('ii','i'),T1_Ia,'iii'),(('jj','j'),T1_Ja,'jjj'),(('kk','k'),T1_Ka,'kkk'), (('ll','l'),T1_La,'lll'), (('mm','m'),T1_Ma,'mmm')],
    unrolljam = (['i','j','k','l','m'],[U1_I,U1_J,U1_K,U1_L,U1_M]),
    scalarreplace = (SCREP, 'double', 'scv_'),
    vector = (VEC, ['ivdep','vector always']),
    openmp = (OMP, 'omp parallel for private(iii,jjj,kkk,lll,mmm,ii,jj,kk,ll,mm,i,j,k,l,m)')
  )
  for(i=0; i<=V-1; i++)
    for(j=0; j<=V-1; j++)
      for(k=0; k<=O-1; k++)
        for(l=0; l<=O-1; l++)
      for(m=0; m<=O-1; m++)
        R[i][j][k][l] = R[i][j][k][l] + T[i][m][k][l] * A2[j][m];

) @*/

/*@ end @*/
/*@ end @*/
