#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_integration.h>
#include "wrappers.h"

#define Toy_fR_EPSILON_CONVERGENCE 1e-9
#define Toy_fR_REDSHIFT_START      1.619154804176531e7//1e14
#define Toy_fR_NPOINTS             10000

GSL_Spline Toy_fR_Einstein_GeffSpline;
GSL_Spline Toy_fR_Einstein_HSpline;
GSL_Spline Toy_fR_Einstein_dHdaSpline;
GSL_Spline Toy_fR_Einstein_Mass2Spline;
GSL_Spline Toy_fR_Einstein_M_1Spline;
GSL_Spline Toy_fR_Einstein_M_2Spline;
GSL_Spline Toy_fR_Einstein_uSpline;
GSL_Spline Toy_fR_Einstein_duSpline;
GSL_Spline Toy_fR_Einstein_VSpline;
GSL_Spline Toy_fR_Einstein_rhorSpline;
GSL_Spline Toy_fR_Einstein_rhomSpline;


//=============================================================================
// List of internal functions
//=============================================================================

//double Toy_fR_HubbleFunction(double x, double x1, , double x2, , double x3, double xr, double xm);
double H_E(double x, double u, double du, double rhor, double rhom, double n, double al); // Auxiliar H to simplify notation
double dH_E(double x, double u, double du, double rhor, double rhom, double n, double al);
double HLCDM(double x);
double OmegaR(double x, double u, double du, double rhor, double rhom, double n, double al);
double OmegaM(double x, double u, double du, double rhor, double rhom, double n, double al);
double OmegaDE_E(double x, double u, double du, double rhor, double rhom, double n, double al);
double F_E(double u);
double R_E(double u, double n, double al);
double V_E(double u, double n, double al);
double Vu_E(double u, double n, double al);
double Mass2_E(double u, double n, double al);
double M1_E(double u, double n, double al);
double M2_E(double u, double n, double al);

int    Toy_fR_Einstein_ode(double x, const double y[], double dydx[], void *params);
void   Toy_fR_Einstein_solve_ode(double ui, double dui, double rhori, double rhomi, double *x_arr, double *u_arr, double *du_arr, double *rhor_arr, double *rhom_arr);

//=============================================================================
// Cosmological and model parameters
//=============================================================================
struct Parameters {
  double Omegam0; // kappa^2 rhom0/[3 H^2 phi0]
  double Omegar0; // kappa^2 rhor0/[3 H^2 phi0]
  double OmegaDE0; // kappa^2 Lambda/[3 H^2 phi0]
  double h0_fR;
  double n_fR;         // The n parameter
  double alpha_fR;     // The alpha parameter

  double zini;      // Starting redshift for integration
  double epsilon;   // Convergence criterion
  double u_ini;     //Initial condition for the scalar field
  double du_ini;    //Initial condition for the derivative of the scalar field
  double rho_r_ini; //Initial condition for radiation density  
  double rho_m_ini; //Initial condition for matter density

  int npts;         // Number of points to store
  double *x_arr;    //  x  = log(a) array
  double *u_arr;    //  u array,  u is the scalar field.   u = phi
  double *du_arr;   //  du array, du is the derivative of the scalar field.   du = d\phi/ \ d\tau
//  double *V_arr;   //  x_3 array
  double *rho_r_arr;   //  x_r array  
  double *rho_m_arr;   //  x_m array  
  
  int ThisTask;     // MPI parameter
} ee;

//=============================================================================
// Lookup-functions for use after we have done the calculation
//=============================================================================
double Toy_fR_Einstein_Hubble_of_a(double a){
  return Lookup_GSL_Spline(&Toy_fR_Einstein_HSpline, log(a));
}
double Toy_fR_Einstein_dHubbleda_of_a(double a){
  return Lookup_GSL_Spline(&Toy_fR_Einstein_dHdaSpline, log(a));
}
double Toy_fR_Einstein_Mass2_of_a(double a){
  return Lookup_GSL_Spline(&Toy_fR_Einstein_Mass2Spline, log(a));
}
double Toy_fR_Einstein_M_1_of_a(double a){
  return Lookup_GSL_Spline(&Toy_fR_Einstein_M_1Spline, log(a));
}
double Toy_fR_Einstein_M_2_of_a(double a){
  return Lookup_GSL_Spline(&Toy_fR_Einstein_M_2Spline, log(a));
}
double Toy_fR_Einstein_u_of_a(double a){
  return Lookup_GSL_Spline(&Toy_fR_Einstein_uSpline, log(a));
}
double Toy_fR_Einstein_du_of_a(double a){
  return Lookup_GSL_Spline(&Toy_fR_Einstein_duSpline, log(a));
}
double Toy_fR_Einstein_V_of_a(double a){
  return Lookup_GSL_Spline(&Toy_fR_Einstein_VSpline, log(a));
}
double Toy_fR_Einstein_rho_r_of_a(double a){
  return Lookup_GSL_Spline(&Toy_fR_Einstein_rhorSpline, log(a));
}
double Toy_fR_Einstein_rho_m_of_a(double a){
  return Lookup_GSL_Spline(&Toy_fR_Einstein_rhomSpline, log(a));
}




//=============================================================================
// Scalar potential
// We have H(a=1) = h = little h
//=============================================================================
//  F = \partial f / \partial R
double F_E(double u){
  return exp(sqrt(2)*u);
}

// Ricci scalar 
double R_E(double u, double n, double al){
  return pow((1-F_E(u))/(n*al),1/(n-1.));
}


//Scalar potential
double V_E(double u, double n, double al){
  return al*(1.-n)*pow(R_E(u,n,al),n)/pow(F_E(u),2)/2.;
}


//Derivative of the scalar potential
double Vu_E(double u, double n, double al){
  return sqrt(2.)*(R_E(u,n,al)/F_E(u)/2. - 2.*V_E(u,n,al));
}

// M_1 = V_\phi\phi, second derivative of the scalar potential
double M1_E(double u, double n, double al){
  double fRR = -al*n*(n-1.)*pow(R_E(u,n,al),n-2);
  return 1./fRR - 3.*R_E(u,n,al)/F_E(u) + 8.*V_E(u,n,al);
}

// M_2 = V_\phi\phi\phi, thierd derivative of the sacalr potential
double M2_E(double u, double n, double al){
  double fRR = -al*n*(n-1.)*pow(R_E(u,n,al),n-2.);
  double fRRR = -al*n*(n-1.)*(n-2.)*pow(R_E(u,n,al),n-3.);
  return sqrt(2.)*(-F_E(u)*fRRR/pow(fRR,3) - 3./fRR + 7.*R_E(u,n,al)/F_E(u) - 16.*V_E(u,n,al));
}

//=========================================================//
// The M^{2}(a) function (acctually M^{2}(a)/H0^2)               //
// M^{2} = V_{\phi\phi}/3. We normalized by H0^{2}
// We normalized by H0^{2} at the end of the document in the interpolations funtions.
//=========================================================//

double Mass2_E(double u, double n, double al){
  double V_phi_phi = M1_E(u, n, al);   
  return V_phi_phi/3.;
}



//=============================================================================
// The Hubble function in terms of x = log(a), y = log(phi/phi0) and dy = dy/dx
//
//=============================================================================
double H_E(double x, double u, double du, double rho_r, double rho_m, double n, double al){
  return sqrt( rho_r + rho_m + du*du/exp(x)/exp(x)/2. + V_E(u,n,al)/3.);
}

double dH_E(double x, double u, double du, double rho_r, double rho_m, double n, double al){
  return (-4.*rho_r -3.*rho_m -3.*du*du/exp(x)/exp(x))/H_E(x,u,du,rho_r,rho_m,n,al)/2.;
}



//=============================================================================
// Hubble function for LCDM with the same value of Omega_m0 and Omega_r0
// We have H(a=1) = h = little h
//=============================================================================
double HLCDM(double x){
  double H0 = 1.e7*ee.h0_fR/LIGHT;
  return H0*sqrt(ee.Omegar0 * exp(-4*x) + ee.Omegam0 * exp(-3*x) + ee.OmegaDE0);
}

//=============================================================================
// Density parameters
//=============================================================================
double OmegaM(double x, double u, double du, double rho_r, double rho_m, double n, double al){
  return rho_r/pow(H_E(x, u, du, rho_r, rho_m, n, al), 2);
}
double OmegaR(double x, double u, double du, double rho_r, double rho_m, double n, double al){
  return rho_m/pow(H_E(x, u, du, rho_r, rho_m, n, al), 2);
}
double OmegaDE_E(double x, double u, double du, double rho_r, double rho_m, double n, double al){
  return 1.0 - OmegaM(x, u, du, rho_r, rho_m, n, al) - OmegaR(x, u, du, rho_r, rho_m, n, al);
}


//=============================================================================
// The ODE for the Toy f(R) model in Einstein frame
//=============================================================================
int Toy_fR_Einstein_ode(double x, const double y[], double dydx[], void *params){
  double H = H_E(x, y[0], y[1], y[2], y[3], ee.n_fR, ee.alpha_fR);
  double Vphi = Vu_E(y[0], ee.n_fR, ee.alpha_fR);
  double a = exp(x);

  dydx[0] =  y[1]/H/a;
  dydx[1] = -2.*y[1] - a*Vphi/H/3. + a*y[3]/sqrt(2)/H;
  dydx[2] = -4.*y[2];
  dydx[3] = -3.*y[3] - y[1]*y[3]/a/H/sqrt(2); 

  return GSL_SUCCESS;
}

//=============================================================================
// Solve the ODE for the initial condition (y, dy/dx) = (yi, dyi) 
// Stores the values found in the provided arrays
//=============================================================================
void Toy_fR_Einstein_solve_ode(double ui, double dui, double rhori, double rhomi, double *x_arr, double *u_arr, double *du_arr, double *rho_r_arr, double *rho_m_arr){
  const double xini = log(1.0 / (1.0 + ee.zini));
  const double xend = log(1.0);
  const double deltax = (xend-xini)/(double)(ee.npts-1);

  // Set up ODE system //rk2  rkf45  bsimp msbdf
  gsl_odeiv2_system Toy_fR_Einstein_sys = {Toy_fR_Einstein_ode, NULL, 5, NULL};
  gsl_odeiv2_driver * Toy_fR_Einstein_ode_driver = gsl_odeiv2_driver_alloc_y_new (&Toy_fR_Einstein_sys,  gsl_odeiv2_step_rk2, 1e-8, 1e-8, 0.0);

  // Set IC
  double y[4] = {ui, dui, rhori, rhomi};

  double ode_x = xini;

  // Set IC in array
  x_arr[0]  = xini;
  u_arr[0] = y[0];
  du_arr[0] = y[1];
  rho_r_arr[0] = y[2];
  rho_m_arr[0] = y[3];

  // Solve the ODE
  for(int i = 1; i < ee.npts; i++){
    double xnow = xini + i * deltax;

    int status = gsl_odeiv2_driver_apply(Toy_fR_Einstein_ode_driver, &ode_x, xnow, y);
    if(status != GSL_SUCCESS){
      printf("Error in integrating at x = %f  u = %f, du = %f, rho_r = %f and rho_m = %f  \n", xnow, y[0], y[1], y[2], y[3]);
      exit(1);
    }

    x_arr[i]  = xnow;
    u_arr[i]  = y[0];
    du_arr[i]  = y[1];
    rho_r_arr[i]  = y[2];
    rho_m_arr[i]  = y[3];
  }
}


//=============================================================================
//============================================================================= 
//============================================================================= 
//=============================================================================

void Toy_fR_Einstein_Solve_Background(double *Omegam_0, double *h){
  
  //=========================================
  // Set the parameters needed by the solver
  //=========================================
  ee.n_fR       = 0.8;
  ee.alpha_fR   = 0.035;  

  ee.npts      = Toy_fR_NPOINTS;
  ee.zini      = Toy_fR_REDSHIFT_START;
  ee.epsilon   = Toy_fR_EPSILON_CONVERGENCE;
 
  ee.x_arr      = my_malloc(sizeof(double)*ee.npts);
  ee.u_arr      = my_malloc(sizeof(double)*ee.npts);
  ee.du_arr     = my_malloc(sizeof(double)*ee.npts);
  ee.rho_r_arr  = my_malloc(sizeof(double)*ee.npts);
  ee.rho_m_arr  = my_malloc(sizeof(double)*ee.npts);

  double rho_r_i = 3.198579042726645e17;//2.7796829014640762e44+1.9008961412625685e44;  //rho_g_ini + rho_ur_ini
  double rho_m_i = 4.712890281877432e13;   //rho_b_ini + rho_cdm_ini
  double du_i    = 0.005183709371285337;//0.005183709371285337;
  double u_i     = -0.000017379054922651744;


  Toy_fR_Einstein_solve_ode(u_i, du_i, rho_r_i, rho_m_i, ee.x_arr, ee.u_arr, ee.du_arr, ee.rho_r_arr, ee.rho_m_arr);


  // The Hubble parameter we find
  double H0_fR = H_E(0.0, ee.u_arr[ee.npts-1], ee.du_arr[ee.npts-1], ee.rho_r_arr[ee.npts-1], ee.rho_m_arr[ee.npts-1], ee.n_fR, ee.alpha_fR);
  *h = H0_fR*LIGHT/1.e7;
  *Omegam_0 = ee.Omegam0  =  OmegaM(0.0, ee.u_arr[ee.npts-1], ee.du_arr[ee.npts-1], ee.rho_r_arr[ee.npts-1], ee.rho_m_arr[ee.npts-1], ee.n_fR, ee.alpha_fR);
  ee.OmegaDE0 = OmegaDE_E(0.0, ee.u_arr[ee.npts-1], ee.du_arr[ee.npts-1], ee.rho_r_arr[ee.npts-1], ee.rho_m_arr[ee.npts-1], ee.n_fR, ee.alpha_fR);
 
  printf("Cosmological parameters today \n");  
  printf("-> h  = %f, wished %f \n", H0_fR*LIGHT/1.e7, 0.677681);
  printf("-> Omega_m  = %f, wished %f \n", ee.Omegam0, 4.94697150e-02+2.42893296e-01);
  printf("-> Omega_DE  = %f, wished %f  \n", ee.OmegaDE0, 0.7075459772003001);  
  

  // Make arrays for splining
  double *loga        = my_malloc(sizeof(double)*ee.npts);
  double *H           = my_malloc(sizeof(double)*ee.npts);
  double *dHda        = my_malloc(sizeof(double)*ee.npts);
  double *M_1         = my_malloc(sizeof(double)*ee.npts);  
  double *M_2         = my_malloc(sizeof(double)*ee.npts);
  double *Mass2       = my_malloc(sizeof(double)*ee.npts);  
  double *u_arr       = my_malloc(sizeof(double)*ee.npts);
  double *du_arr      = my_malloc(sizeof(double)*ee.npts);
  double *rho_r_arr   = my_malloc(sizeof(double)*ee.npts); 
  double *rho_m_arr   = my_malloc(sizeof(double)*ee.npts);

  for(int i = 0; i < ee.npts; i++){
    loga[i]  = ee.x_arr[i];
    H[i]     = H_E(ee.x_arr[i], ee.u_arr[i], ee.du_arr[i], ee.rho_r_arr[i], ee.rho_m_arr[i], ee.n_fR, ee.alpha_fR) / (H0_fR);
    dHda[i]  = dH_E(ee.x_arr[i], ee.u_arr[i], ee.du_arr[i], ee.rho_r_arr[i], ee.rho_m_arr[i], ee.n_fR, ee.alpha_fR)/ (H0_fR);
    M_1[i]   = M1_E(ee.u_arr[i], ee.n_fR, ee.alpha_fR)/H0_fR/H0_fR;
    M_2[i]   = M2_E(ee.u_arr[i], ee.n_fR, ee.alpha_fR)/H0_fR/H0_fR;    
    Mass2[i] = Mass2_E(ee.u_arr[i],ee.n_fR, ee.alpha_fR)/H0_fR/H0_fR;    
  }
  
  // Spline up results
  Create_GSL_Spline(&Toy_fR_Einstein_HSpline,     loga, H,      ee.npts);
  Create_GSL_Spline(&Toy_fR_Einstein_dHdaSpline,  loga, dHda,   ee.npts);
  Create_GSL_Spline(&Toy_fR_Einstein_M_1Spline,   loga, M_1,    ee.npts);  
  Create_GSL_Spline(&Toy_fR_Einstein_M_2Spline,   loga, M_2,    ee.npts);    
  Create_GSL_Spline(&Toy_fR_Einstein_Mass2Spline,   loga, Mass2,    ee.npts);  
  Create_GSL_Spline(&Toy_fR_Einstein_uSpline,    loga, u_arr, ee.npts);
  Create_GSL_Spline(&Toy_fR_Einstein_duSpline,    loga, du_arr, ee.npts);
  Create_GSL_Spline(&Toy_fR_Einstein_rhorSpline,    loga, rho_r_arr, ee.npts);
  Create_GSL_Spline(&Toy_fR_Einstein_rhomSpline,    loga, rho_m_arr, ee.npts);


//=========================================
// Export normalized H(a) to file
//=========================================
FILE *fp = fopen("Toy_fr_Einstein_background.dat","w");

if(fp == NULL){
  printf("Error: could not open output file\n");
  exit(1);
}

//int nout = 500;  // resolution of output file
for(int i=0; i<ee.npts; i++){
  fprintf(fp,"%e %e %e %e %e %e %e %e \n", ee.x_arr[i], H[i], ee.u_arr[i], ee.du_arr[i], ee.rho_r_arr[i], ee.rho_m_arr[i], M_1[i], M_2[i]);
}

fclose(fp);
//=========================================
//
//=========================================


  // Free up memory
  my_free(loga);
  my_free(H);
  my_free(dHda);
  my_free(M_1); 
  my_free(M_2); 
  my_free(Mass2);       
  my_free(ee.x_arr);
  my_free(ee.u_arr);
  my_free(ee.du_arr);
  my_free(ee.rho_r_arr); 
  my_free(ee.rho_m_arr);
}

