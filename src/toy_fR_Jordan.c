#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_integration.h>
#include "wrappers.h"

#define Toy_fR_EPSILON_CONVERGENCE 1e-6
#define Toy_fR_REDSHIFT_START      1.e6
#define Toy_fR_NPOINTS             10000

GSL_Spline Toy_fR_GeffSpline;
GSL_Spline Toy_fR_HSpline;
GSL_Spline Toy_fR_dHdaSpline;
GSL_Spline Toy_fR_M2Spline;
GSL_Spline Toy_fR_dM2daSpline;
GSL_Spline Toy_fR_M_1Spline;
GSL_Spline Toy_fR_M_2Spline;
GSL_Spline Toy_fR_x1Spline;
GSL_Spline Toy_fR_x2Spline;
GSL_Spline Toy_fR_x3Spline;
GSL_Spline Toy_fR_xrSpline;
GSL_Spline Toy_fR_xmSpline;


//=============================================================================
// 
// This module takes in physical cosmological parameters and the JBD
// constant and solves the background equation and computes the hubble constant
// today.
//
// Input:    n, OmegaM, OmegaR, OmegaDE, npts
// Returns:  HubbleParameter. 
//
// After we are done solving then H(a), dHda(a) and Geff(a) is availiable
// externally from the functions listed below
//
// void   JBD_Solve_Background(double n, double omegam, double omegaDE, double omegar, double *h);
// double JBD_Hubble_of_a(double a);
// double JBD_dHubbleda_of_a(double a);
// double JBD_GeffG_of_a(double a);
//
//=============================================================================

//=============================================================================
// List of internal functions
//=============================================================================

//double Toy_fR_HubbleFunction(double x, double x1, , double x2, , double x3, double xr, double xm);
double Toy_fR_H(double x, double x1, double x2, double x3, double xr, double xm); // Auxiliar H to simplify notation
double Toy_fR_dHubbleFunctiondx(double x, double x1, double x2, double x3, double xr, double xm);
double Toy_fR_HLCDM(double x);
double Toy_fR_OmegaR(double x, double x1, double x2, double x3, double xr, double xm);
double Toy_fR_OmegaM(double x, double x1, double x2, double x3, double xr, double xm);
double Toy_fR_FR(double x, double x3);
double Toy_fR_M2(double x, double x2, double x3);
double Toy_fR_dM2da(double x, double x1, double x2, double x3, double xr, double xm);
double Toy_fR_M_1(double x, double x3);
double Toy_fR_M_2(double x, double x3);
int    Toy_fR_ode(double x, const double y[], double dydx[], void *params);
void   Toy_fR_solve_ode(double x1i, double x2i, double x3i, double xri, double xmi, double *x_arr, double *x1_arr, double *x2_arr, double *x3_arr, double *xr_arr, double *xm_arr);

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
  double x3i;

  int npts;         // Number of points to store
  double *x_arr;    //  x  = log(a) array
  double *x1_arr;   //  x_1 array
  double *x2_arr;   //  x_2 array
  double *x3_arr;   //  x_3 array
  double *xr_arr;   //  x_r array  
  double *xm_arr;   //  x_m array  
  
  int ThisTask;     // MPI parameter
} ff;

//=============================================================================
// Lookup-functions for use after we have done the calculation
//=============================================================================
double Toy_fR_Hubble_of_a(double a){
  return Lookup_GSL_Spline(&Toy_fR_HSpline, log(a));
}
double Toy_fR_dHubbleda_of_a(double a){
  return Lookup_GSL_Spline(&Toy_fR_dHdaSpline, log(a));
}
double Toy_fR_M2_of_a(double a){
  return Lookup_GSL_Spline(&Toy_fR_M2Spline, log(a));
}
double Toy_fR_dM2da_of_a(double a){
  return Lookup_GSL_Spline(&Toy_fR_dM2daSpline, log(a));
}
double Toy_fR_M_1_of_a(double a){
  return Lookup_GSL_Spline(&Toy_fR_M_1Spline, log(a));
}
double Toy_fR_M_2_of_a(double a){
  return Lookup_GSL_Spline(&Toy_fR_M_2Spline, log(a));
}
double Toy_fR_x1_of_a(double a){
  return Lookup_GSL_Spline(&Toy_fR_x1Spline, log(a));
}
double Toy_fR_x2_of_a(double a){
  return Lookup_GSL_Spline(&Toy_fR_x2Spline, log(a));
}
double Toy_fR_x3_of_a(double a){
  return Lookup_GSL_Spline(&Toy_fR_x3Spline, log(a));
}
double Toy_fR_xr_of_a(double a){
  return Lookup_GSL_Spline(&Toy_fR_xrSpline, log(a));
}
double Toy_fR_xm_of_a(double a){
  return Lookup_GSL_Spline(&Toy_fR_xmSpline, log(a));
}


//=============================================================================
// The Hubble function in terms of x = log(a), y = log(phi/phi0) and dy = dy/dx
// We have H(a=1) = h = little h
//=============================================================================
double Toy_fR_H(double x, double x1, double x2, double x3, double xr, double xm){
  return sqrt( x1 + x2 + x3 + xr + xm);
}

/*double Toy_fR_HubbleFunction(double x, double x1, double x2, double x3, double xr, double xm){
  return H(x, x1, x2, x3, xr, xm) / 100.; // !!!! Look at the normalization H(a=0) = h !!!
}*/
double Toy_fR_dHubbleFunctiondx(double x, double x1, double x2, double x3, double xr, double xm){
  double H   = Toy_fR_H(x, x1, x2, x3, xr, xm);
  return (x3-2.*H*H)/H;
}

//=============================================================================
// The f_{R} function in terms of x = log(a), and x3 = R/6
//=============================================================================

double Toy_fR_FR(double x, double x3){
  return 1-ff.alpha_fR*ff.n_fR*pow(6*x3,ff.n_fR-1);
}

//=========================================================//
// The M^{2}(a) function (acctually M^{2}(a)/H0^2)               //
// M^{2} = (f_{R}/f_{RR} -R)/3. We normalized by H0^{2}
// We normalized by H0^{2} at the end of the document in the interpolations funtions.
//=========================================================//

double Toy_fR_M2(double x, double x2, double x3){
  double m = ff.n_fR*(1. + x3/x2)/(x3/x2);   //m(r) = R*f_{RR}/f_{R}
  return 2.*x3*(1./m - 1.);              
}

double Toy_fR_dM2da(double x, double x1, double x2, double x3, double xr, double xm){
  double H = Toy_fR_H(x, x1, x2, x3, xr, xm);
  double m = ff.n_fR*(1. + x3/x2)/(x3/x2);
  double dx2 = exp(x)*x1*(x2+x3/m)/H/H;
  double dx3 = -exp(x)*x1*x3/m/H/H;
  double drda = (dx3-(x3/x2)*dx2)/x2;
  double dmda = (ff.n_fR-m)*(x2/x3)*drda;

  return 2.*dx3*(1./m - 1.) - 2.*x3*dmda/m/m;
}

//=========================================================//
// The M_{1} function (acctually M_{2}(a)/H0^2)               //
// M_{1} = \partial R / \partial f_{R}
// We normalized by H0^{2} at the end of the document in the interpolations funtions.
//=========================================================//
double Toy_fR_M_1(double x, double x3){
  return -pow(6.*x3,2.-ff.n_fR) / (ff.n_fR - 1.) / ff.alpha_fR / ff.n_fR;
}

//=========================================================//
// The M_{2} function (acctually M_{2}(a)/H0^2)               //
// M_{2} = \partial^{2} R / \partial f_{R}^{2}
// We normalized by H0^{2} at the end of the document in the interpolations funtions.
//=========================================================//
double Toy_fR_M_2(double x, double x3){
  return (2.- ff.n_fR)*pow(6.*x3, 3. - 2.*ff.n_fR)/pow(ff.n_fR - 1.,2)/pow(ff.alpha_fR*ff.n_fR,2);
}

//=============================================================================
// Hubble function for LCDM with the same value of Omega_m0 and Omega_r0
// We have H(a=1) = h = little h
//=============================================================================
double Toy_fR_HLCDM(double x){
  double H0 = 1.e7*ff.h0_fR/LIGHT;
  return H0*sqrt(ff.Omegar0 * exp(-4*x) + ff.Omegam0 * exp(-3*x) + ff.OmegaDE0);
}

//=============================================================================
// Density parameters
//=============================================================================
double Toy_fR_OmegaM(double x, double x1, double x2, double x3, double xr, double xm){
  //  double om  = FJ*exp(Lxm) / (H*H);
  return Toy_fR_FR(x, x3)*xm / pow(Toy_fR_H(x, x1, x2, x3, xr, xm), 2); // be carefull with the normalization of H
}
double Toy_fR_OmegaR(double x, double x1, double x2, double x3, double xr, double xm){
  return Toy_fR_FR(x, x3)*xr / pow(Toy_fR_H(x, x1, x2, x3, xr, xm), 2); // be carefull with the normalization of H
}
double Toy_fR_OmegaDE(double x, double x1, double x2, double x3, double xr, double xm){
  return 1.0 - Toy_fR_OmegaM(x, x1, x2, x3, xr, xm) - Toy_fR_OmegaR(x, x1, x2, x3, xr, xm); // be carefull with the normalization of H
}


//=============================================================================
// The ODE for the JBD scalar field
//=============================================================================
int Toy_fR_ode(double x, const double y[], double dydx[], void *params){
  double H = Toy_fR_H(x, y[0], y[1], y[2], y[3], y[4]);
  double m = ff.n_fR*(1. + y[2]/y[1])/(y[2]/y[1]);

//  dydx[0] = -5.*y[0] - 4.*y[1] - 2.*y[2] - y[3] + y[0]*y[0]/H/H +y[0]*y[2]/H/H;
  dydx[0] = pow(y[0]/H,2) - 4.*y[0] + y[0]*y[2]/H/H - 3.*y[1] - y[2] + y[3]- H*H;
  dydx[1] = y[0]*(y[1] + y[2]/m)/H/H;
  dydx[2] = -y[0]*y[2]/m/H/H; 
  dydx[3] = -4.*y[3] + y[0]*y[3]/H/H;
  dydx[4] = -3.*y[4] + y[0]*y[4]/H/H; 

  return GSL_SUCCESS;
}

//=============================================================================
// Solve the ODE for the initial condition (y, dy/dx) = (yi, dyi) 
// Stores the values found in the provided arrays
//=============================================================================
void Toy_fR_solve_ode(double x1i, double x2i, double x3i, double xri, double xmi, double *x_arr, double *x1_arr, double *x2_arr, double *x3_arr, double *xr_arr, double *xm_arr){
  const double xini = log(1.0 / (1.0 + ff.zini));
  const double xend = log(1.0);
  const double deltax = (xend-xini)/(double)(ff.npts-1);

  // Set up ODE system //rk2  rkf45  bsimp msbdf
  gsl_odeiv2_system Toy_fR_sys = {Toy_fR_ode, NULL, 5, NULL};
  gsl_odeiv2_driver * Toy_fR_ode_driver = gsl_odeiv2_driver_alloc_y_new (&Toy_fR_sys,  gsl_odeiv2_step_rk2, 1e-8, 1e-8, 0.0);

  // Set IC
  double y[5] = {x1i, x2i, x3i, xri, xmi};

  double ode_x = xini;

  // Set IC in array
  x_arr[0]  = xini;
  x1_arr[0] = y[0];
  x2_arr[0] = y[1];
  x3_arr[0] = y[2];
  xr_arr[0] = y[3];
  xm_arr[0] = y[4];

  // Solve the ODE
  for(int i = 1; i < ff.npts; i++){
    double xnow = xini + i * deltax;

    int status = gsl_odeiv2_driver_apply(Toy_fR_ode_driver, &ode_x, xnow, y);
    if(status != GSL_SUCCESS){
      printf("Error in integrating at x = %f  yr = %f and ym = %f \n", xnow, y[3], y[4]);
      exit(1);
    }

    x_arr[i]  = xnow;
    x1_arr[i]  = y[0];
    x2_arr[i]  = y[1];
    x3_arr[i]  = y[2];
    xr_arr[i]  = y[3];
    xm_arr[i]  = y[4];            
  }
}

//=============================================================================
// Find correct initial conditions using bisection
//=============================================================================
void Toy_fR_find_correct_IC_using_bisection(){
  const int npts = ff.npts;
  double *x_arr  = ff.x_arr;
  double *x1_arr = ff.x1_arr;
  double *x2_arr = ff.x2_arr;
  double *x3_arr = ff.x3_arr;
  double *xr_arr = ff.xr_arr;
  double *xm_arr = ff.xm_arr;    

  // Find the correct initial condition
  double x3low  = 0.0;
  double x3high = 0.34;
  double x3now, x1i, x2i, x3i, xri, xmi;
  double X2, X3, HLCDM;
  double H0 = 1.e7*ff.h0_fR/LIGHT;

  double alpha_low  = 0.0;
  double alpha_high = 10.0;
  double alpha_now;


  HLCDM = Toy_fR_HLCDM(log(1.0 / (1.0 + ff.zini)));

  xri = H0*H0*ff.Omegar0*pow(1+ff.zini,4);
  xmi = H0*H0*ff.Omegam0*pow(1+ff.zini,3);
  X2  = -1.e-3; 
  x2i = X2*HLCDM*HLCDM;

  int istep = 0;
  while(istep < 50){
    ++istep;

    // Current value for phii
    x3now = (x3low+x3high)/2.0;

    // Solve ODE
    X3  = (1. + x3now*1.e-2)*1e-3;
    x3i = ff.x3i = X3*HLCDM*HLCDM;
    x1i = (1. - X2 - X3)*HLCDM*HLCDM - xmi - xri;
    Toy_fR_solve_ode(x1i, x2i, x3i, xri, xmi, x_arr, x1_arr, x2_arr, x3_arr, xr_arr, xm_arr);

    // Check for convergence
    double H0_shooting = Toy_fR_H(0, x1_arr[npts-1], x2_arr[npts-1], x3_arr[npts-1], xr_arr[npts-1], xm_arr[npts-1]);
    if( fabs(H0_shooting - H0) < ff.epsilon) {
//#ifdef ToyfRDEBUG
      printf("Convergence of solution found after %i iterations\n", istep);
      printf("The initial conditions are \n");
      printf("-> x_1 = %f  \n", x1i);
      printf("-> x_2 = %f  \n", x2i);
      printf("-> x_3 = %f  \n", x3i);
      printf("-> x_r = %f  \n", xri);
      printf("-> x_m = %f  \n", xmi);
      printf("-> h  = %f, wished %f\n", H0_shooting*LIGHT/1.e7, H0*LIGHT/1.e7);
//#endif
      break;
    }
    // Bisection step
    if(H0_shooting - H0 < 0.0){
      x3low = x3now;
    } else {
      x3high = x3now;
    }    
  }



//=============================================================================
// Find correct alpha
//=============================================================================


  int jstep = 0;
  while(jstep < 50){
    ++jstep;

    // Current value for phii
    alpha_now = ff.alpha_fR = (alpha_low + alpha_high)/2.0;

    // Check for convergence
  double  Omega_m_shooting = Toy_fR_OmegaM(log(1.0 / (1.0 + ff.zini)), x1_arr[npts-1], x2_arr[npts-1], x3_arr[npts-1], xr_arr[npts-1], xm_arr[npts-1]);
    if( fabs(Omega_m_shooting - ff.Omegam0) < ff.epsilon ) {
      printf("Convergence of solution for alpha found after %i iterations\n", jstep);
      printf("-> alpha  = %f \n", ff.alpha_fR);
      printf("-> Omega_m  = %f, wished %f\n", Omega_m_shooting, ff.Omegam0);
      break;
    }

    // Bisection step
    if(Omega_m_shooting - ff.Omegam0 > 0.0){
      alpha_low = alpha_now;
    } else {
      alpha_high = alpha_now;
    }
  }

}

void Toy_fR_Solve_Background(double n_fR, double omegam, double omegaDE, double omegar, double h0_fR, double *h){
  
  //=========================================
  // Set the parameters needed by the solver
  //=========================================
  ff.n_fR      = n_fR;
  ff.Omegam0   = omegam;
  ff.Omegar0   = omegar;
  ff.OmegaDE0  = omegaDE;
  ff.h0_fR     = h0_fR;
  ff.npts      = Toy_fR_NPOINTS;
  ff.zini      = Toy_fR_REDSHIFT_START;
  ff.epsilon   = Toy_fR_EPSILON_CONVERGENCE;
 
  ff.x_arr     = my_malloc(sizeof(double)*ff.npts);
  ff.x1_arr    = my_malloc(sizeof(double)*ff.npts);
  ff.x2_arr    = my_malloc(sizeof(double)*ff.npts);
  ff.x3_arr    = my_malloc(sizeof(double)*ff.npts);
  ff.xr_arr    = my_malloc(sizeof(double)*ff.npts);
  ff.xm_arr    = my_malloc(sizeof(double)*ff.npts);      
  
  // Find the correct IC (after this is done we have the correct solution in ff-arrays)
  Toy_fR_find_correct_IC_using_bisection();

  // The Hubble parameter we find
  double H0_fR = Toy_fR_H(0.0, ff.x1_arr[ff.npts-1], ff.x2_arr[ff.npts-1], ff.x3_arr[ff.npts-1], ff.xr_arr[ff.npts-1], ff.xm_arr[ff.npts-1]);
  *h = H0_fR*LIGHT/1.e7;

  // Make arrays for splining
  double *loga     = my_malloc(sizeof(double)*ff.npts);
  double *H        = my_malloc(sizeof(double)*ff.npts);
  double *dHda     = my_malloc(sizeof(double)*ff.npts);
  double *M2       = my_malloc(sizeof(double)*ff.npts);
  double *dM2da    = my_malloc(sizeof(double)*ff.npts);
  double *M_1      = my_malloc(sizeof(double)*ff.npts);  
  double *M_2      = my_malloc(sizeof(double)*ff.npts);
  double *x1_arr   = my_malloc(sizeof(double)*ff.npts);
  double *x2_arr   = my_malloc(sizeof(double)*ff.npts);
  double *x3_arr   = my_malloc(sizeof(double)*ff.npts); 
  double *xr_arr   = my_malloc(sizeof(double)*ff.npts);
  double *xm_arr   = my_malloc(sizeof(double)*ff.npts);            
  for(int i = 0; i < ff.npts; i++){
    loga[i]  = ff.x_arr[i];
    H[i]     = Toy_fR_H(ff.x_arr[i], ff.x1_arr[i], ff.x2_arr[i], ff.x3_arr[i], ff.xr_arr[i], ff.xm_arr[i]) / (H0_fR);
    dHda[i]  = Toy_fR_dHubbleFunctiondx(ff.x_arr[i], ff.x1_arr[i], ff.x2_arr[i], ff.x3_arr[i], ff.xr_arr[i], ff.xm_arr[i])/ (H0_fR);
    M2[i]    = Toy_fR_M2(ff.x_arr[i], ff.x2_arr[i], ff.x3_arr[i])/H0_fR/H0_fR; 
    dM2da[i] = Toy_fR_dM2da(ff.x_arr[i], ff.x1_arr[i], ff.x2_arr[i], ff.x3_arr[i], ff.xr_arr[i], ff.xm_arr[i])/H0_fR/H0_fR;
    M_1[i]   = Toy_fR_M_1(ff.x_arr[i],   ff.x3_arr[i])/H0_fR/H0_fR;
    M_2[i]   = Toy_fR_M_2(ff.x_arr[i], ff.x3_arr[i])/H0_fR/H0_fR;    
  }
  
  // Spline up results
  Create_GSL_Spline(&Toy_fR_HSpline,     loga, H,      ff.npts);
  Create_GSL_Spline(&Toy_fR_dHdaSpline,  loga, dHda,   ff.npts);
  Create_GSL_Spline(&Toy_fR_M2Spline,    loga, M2,     ff.npts);  
  Create_GSL_Spline(&Toy_fR_dM2daSpline, loga, dM2da,  ff.npts);    
  Create_GSL_Spline(&Toy_fR_M_1Spline,   loga, M_1,    ff.npts);  
  Create_GSL_Spline(&Toy_fR_M_2Spline,   loga, M_2,    ff.npts);    
  Create_GSL_Spline(&Toy_fR_x1Spline,    loga, x1_arr, ff.npts);
  Create_GSL_Spline(&Toy_fR_x2Spline,    loga, x2_arr, ff.npts);
  Create_GSL_Spline(&Toy_fR_x3Spline,    loga, x3_arr, ff.npts);
  Create_GSL_Spline(&Toy_fR_xrSpline,    loga, xr_arr, ff.npts);
  Create_GSL_Spline(&Toy_fR_xmSpline,    loga, xm_arr, ff.npts);  


//=========================================
// Export normalized H(a) to file
//=========================================
FILE *fp = fopen("Toy_fr_background.dat","w");

if(fp == NULL){
  printf("Error: could not open output file\n");
  exit(1);
}

//int nout = 500;  // resolution of output file
for(int i=0; i<ff.npts; i++){
  fprintf(fp,"%e %e %e %e %e %e %e %e %e %e %e\n", ff.x_arr[i], H[i], ff.x1_arr[i], ff.x2_arr[i], ff.x3_arr[i], ff.xr_arr[i], ff.xm_arr[i], M2[i], dM2da[i], M_1[i], M_2[i]);
}

fclose(fp);
//=========================================
//
//=========================================


  // Free up memory
  my_free(loga);
  my_free(H);
  my_free(dHda);
  my_free(M2);
  my_free(dM2da);
  my_free(M_1); 
  my_free(M_2);      
  my_free(ff.x_arr);
  my_free(ff.x1_arr);
  my_free(ff.x2_arr);
  my_free(ff.x3_arr); 
  my_free(ff.xr_arr);
  my_free(ff.xm_arr);     
}

