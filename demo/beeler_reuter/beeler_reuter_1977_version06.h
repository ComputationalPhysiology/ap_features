#include <math.h>
#include <string.h>
// Gotran generated C/C++ code for the "beeler_reuter_1977_version06" model

enum state {
  STATE_m,
  STATE_h,
  STATE_j,
  STATE_Cai,
  STATE_d,
  STATE_f,
  STATE_x1,
  STATE_V,
  NUM_STATES,
};

enum parameter {
  PARAM_E_Na,
  PARAM_g_Na,
  PARAM_g_Nac,
  PARAM_g_s,
  PARAM_IstimAmplitude,
  PARAM_IstimEnd,
  PARAM_IstimPeriod,
  PARAM_IstimPulseDuration,
  PARAM_IstimStart,
  PARAM_C,
  NUM_PARAMS,
};

enum monitored {
  MONITOR_i_Na,
  MONITOR_alpha_m,
  MONITOR_beta_m,
  MONITOR_alpha_h,
  MONITOR_beta_h,
  MONITOR_alpha_j,
  MONITOR_beta_j,
  MONITOR_E_s,
  MONITOR_i_s,
  MONITOR_alpha_d,
  MONITOR_beta_d,
  MONITOR_alpha_f,
  MONITOR_beta_f,
  MONITOR_i_x1,
  MONITOR_alpha_x1,
  MONITOR_beta_x1,
  MONITOR_Istim,
  MONITOR_i_K1,
  MONITOR_dm_dt,
  MONITOR_dh_dt,
  MONITOR_dj_dt,
  MONITOR_dCai_dt,
  MONITOR_dd_dt,
  MONITOR_df_dt,
  MONITOR_dx1_dt,
  MONITOR_dV_dt,
  NUM_MONITORED,
};

// Init state values
void init_state_values(double* states)
{
  states[STATE_m] = 0.011;
  states[STATE_h] = 0.988;
  states[STATE_j] = 0.975;
  states[STATE_Cai] = 0.0001;
  states[STATE_d] = 0.003;
  states[STATE_f] = 0.994;
  states[STATE_x1] = 0.0001;
  states[STATE_V] = -84.624;
}

// Default parameter values
void init_parameters_values(double* parameters)
{
  parameters[PARAM_E_Na] = 50.0;
  parameters[PARAM_g_Na] = 0.04;
  parameters[PARAM_g_Nac] = 3e-05;
  parameters[PARAM_g_s] = 0.0009;
  parameters[PARAM_IstimAmplitude] = 0.5;
  parameters[PARAM_IstimEnd] = 50000.0;
  parameters[PARAM_IstimPeriod] = 1000.0;
  parameters[PARAM_IstimPulseDuration] = 1.0;
  parameters[PARAM_IstimStart] = 10.0;
  parameters[PARAM_C] = 0.01;
}

// State index
int state_index(const char name[])
{
  if (strcmp(name, "m")==0)
  {
    return STATE_m;
  }
  else if (strcmp(name, "h")==0)
  {
    return STATE_h;
  }
  else if (strcmp(name, "j")==0)
  {
    return STATE_j;
  }
  else if (strcmp(name, "Cai")==0)
  {
    return STATE_Cai;
  }
  else if (strcmp(name, "d")==0)
  {
    return STATE_d;
  }
  else if (strcmp(name, "f")==0)
  {
    return STATE_f;
  }
  else if (strcmp(name, "x1")==0)
  {
    return STATE_x1;
  }
  else if (strcmp(name, "V")==0)
  {
    return STATE_V;
  }
  return -1;
}

// Parameter index
int parameter_index(const char name[])
{
  if (strcmp(name, "E_Na")==0)
  {
    return PARAM_E_Na;
  }
  else if (strcmp(name, "g_Na")==0)
  {
    return PARAM_g_Na;
  }
  else if (strcmp(name, "g_Nac")==0)
  {
    return PARAM_g_Nac;
  }
  else if (strcmp(name, "g_s")==0)
  {
    return PARAM_g_s;
  }
  else if (strcmp(name, "IstimAmplitude")==0)
  {
    return PARAM_IstimAmplitude;
  }
  else if (strcmp(name, "IstimEnd")==0)
  {
    return PARAM_IstimEnd;
  }
  else if (strcmp(name, "IstimPeriod")==0)
  {
    return PARAM_IstimPeriod;
  }
  else if (strcmp(name, "IstimPulseDuration")==0)
  {
    return PARAM_IstimPulseDuration;
  }
  else if (strcmp(name, "IstimStart")==0)
  {
    return PARAM_IstimStart;
  }
  else if (strcmp(name, "C")==0)
  {
    return PARAM_C;
  }
  return -1;
}

int monitored_index(const char name[])
{
  if (strcmp(name, "i_Na")==0)
  {
    return MONITOR_i_Na;
  }
  else if (strcmp(name, "alpha_m")==0)
  {
    return MONITOR_alpha_m;
  }
  else if (strcmp(name, "beta_m")==0)
  {
    return MONITOR_beta_m;
  }
  else if (strcmp(name, "alpha_h")==0)
  {
    return MONITOR_alpha_h;
  }
  else if (strcmp(name, "beta_h")==0)
  {
    return MONITOR_beta_h;
  }
  else if (strcmp(name, "alpha_j")==0)
  {
    return MONITOR_alpha_j;
  }
  else if (strcmp(name, "beta_j")==0)
  {
    return MONITOR_beta_j;
  }
  else if (strcmp(name, "E_s")==0)
  {
    return MONITOR_E_s;
  }
  else if (strcmp(name, "i_s")==0)
  {
    return MONITOR_i_s;
  }
  else if (strcmp(name, "alpha_d")==0)
  {
    return MONITOR_alpha_d;
  }
  else if (strcmp(name, "beta_d")==0)
  {
    return MONITOR_beta_d;
  }
  else if (strcmp(name, "alpha_f")==0)
  {
    return MONITOR_alpha_f;
  }
  else if (strcmp(name, "beta_f")==0)
  {
    return MONITOR_beta_f;
  }
  else if (strcmp(name, "i_x1")==0)
  {
    return MONITOR_i_x1;
  }
  else if (strcmp(name, "alpha_x1")==0)
  {
    return MONITOR_alpha_x1;
  }
  else if (strcmp(name, "beta_x1")==0)
  {
    return MONITOR_beta_x1;
  }
  else if (strcmp(name, "Istim")==0)
  {
    return MONITOR_Istim;
  }
  else if (strcmp(name, "i_K1")==0)
  {
    return MONITOR_i_K1;
  }
  else if (strcmp(name, "dm_dt")==0)
  {
    return MONITOR_dm_dt;
  }
  else if (strcmp(name, "dh_dt")==0)
  {
    return MONITOR_dh_dt;
  }
  else if (strcmp(name, "dj_dt")==0)
  {
    return MONITOR_dj_dt;
  }
  else if (strcmp(name, "dCai_dt")==0)
  {
    return MONITOR_dCai_dt;
  }
  else if (strcmp(name, "dd_dt")==0)
  {
    return MONITOR_dd_dt;
  }
  else if (strcmp(name, "df_dt")==0)
  {
    return MONITOR_df_dt;
  }
  else if (strcmp(name, "dx1_dt")==0)
  {
    return MONITOR_dx1_dt;
  }
  else if (strcmp(name, "dV_dt")==0)
  {
    return MONITOR_dV_dt;
  }
  return -1;
}

// Compute the right hand side of the beeler_reuter_1977_version06 ODE
void rhs(const double *__restrict states, const double t, const double
  *__restrict parameters, double* values)
{

  // Assign states
  const double m = states[STATE_m];
  const double h = states[STATE_h];
  const double j = states[STATE_j];
  const double Cai = states[STATE_Cai];
  const double d = states[STATE_d];
  const double f = states[STATE_f];
  const double x1 = states[STATE_x1];
  const double V = states[STATE_V];

  // Assign parameters
  const double E_Na = parameters[PARAM_E_Na];
  const double g_Na = parameters[PARAM_g_Na];
  const double g_Nac = parameters[PARAM_g_Nac];
  const double g_s = parameters[PARAM_g_s];
  const double IstimAmplitude = parameters[PARAM_IstimAmplitude];
  const double IstimEnd = parameters[PARAM_IstimEnd];
  const double IstimPeriod = parameters[PARAM_IstimPeriod];
  const double IstimPulseDuration = parameters[PARAM_IstimPulseDuration];
  const double IstimStart = parameters[PARAM_IstimStart];
  const double C = parameters[PARAM_C];

  // Expressions for the Sodium current component
  const double i_Na = (g_Nac + g_Na*(m*m*m)*h*j)*(-E_Na + V);

  // Expressions for the m gate component
  const double alpha_m = (-47. - V)/(-1. + 0.00909527710169582*exp(-0.1*V));
  const double beta_m = 0.709552672748991*exp(-0.056*V);
  values[STATE_m] = (1. - m)*alpha_m - beta_m*m;

  // Expressions for the h gate component
  const double alpha_h = 5.49796243870906e-10*exp(-0.25*V);
  const double beta_h = 1.7/(1. + 0.158025320889648*exp(-0.082*V));
  values[STATE_h] = (1. - h)*alpha_h - beta_h*h;

  // Expressions for the j gate component
  const double alpha_j = 1.86904730072229e-10*exp(-0.25*V)/(1. +
    1.67882752999566e-7*exp(-0.2*V));
  const double beta_j = 0.3/(1. + 0.0407622039783662*exp(-0.1*V));
  values[STATE_j] = (1. - j)*alpha_j - beta_j*j;

  // Expressions for the Slow inward current component
  const double E_s = -82.3 - 13.0287*log(0.001*Cai);
  const double i_s = g_s*(-E_s + V)*d*f;
  values[STATE_Cai] = 7.0e-6 - 0.07*Cai - 0.01*i_s;

  // Expressions for the d gate component
  const double alpha_d = 0.095*exp(1./20. - V/100.)/(1. +
    1.43328813856966*exp(-0.0719942404607631*V));
  const double beta_d = 0.07*exp(-44./59. - V/59.)/(1. + exp(11./5. + V/20.));
  values[STATE_d] = (1. - d)*alpha_d - beta_d*d;

  // Expressions for the f gate component
  const double alpha_f = 0.012*exp(-28./125. - V/125.)/(1. +
    66.5465065250986*exp(0.149925037481259*V));
  const double beta_f = 0.0065*exp(-3./5. - V/50.)/(1. + exp(-6. - V/5.));
  values[STATE_f] = (1. - f)*alpha_f - beta_f*f;

  // Expressions for the Time dependent outward current component
  const double i_x1 = 0.00197277571153285*(-1. +
    21.7584023961971*exp(0.04*V))*exp(-0.04*V)*x1;

  // Expressions for the X1 gate component
  const double alpha_x1 = 0.0311584109863426*exp(0.0826446280991736*V)/(1. +
    17.4117080633277*exp(0.0571428571428571*V));
  const double beta_x1 = 0.000391646440562322*exp(-0.0599880023995201*V)/(1.
    + exp(-4./5. - V/25.));
  values[STATE_x1] = (1. - x1)*alpha_x1 - beta_x1*x1;

  // Expressions for the Time independent outward current component
  const double i_K1 = 0.0035*(4.6 + 0.2*V)/(1. -
    0.398519041084514*exp(-0.04*V)) + 0.0035*(-4. +
    119.856400189588*exp(0.04*V))/(8.33113748768769*exp(0.04*V) +
    69.4078518387552*exp(0.08*V));

  // Expressions for the Stimulus protocol component
  const double Istim = (t - IstimStart - IstimPeriod*floor((t -
    IstimStart)/IstimPeriod) <= IstimPulseDuration && t <= IstimEnd && t >=
    IstimStart ? IstimAmplitude : 0.);

  // Expressions for the Membrane component
  values[STATE_V] = (-i_K1 - i_Na - i_s - i_x1 + Istim)/C;
}

// Computes monitored expressions of the beeler_reuter_1977_version06 ODE
void monitor(const double *__restrict states, const double t, const double
  *__restrict parameters, double* monitored)
{

  // Assign states
  const double m = states[STATE_m];
  const double h = states[STATE_h];
  const double j = states[STATE_j];
  const double Cai = states[STATE_Cai];
  const double d = states[STATE_d];
  const double f = states[STATE_f];
  const double x1 = states[STATE_x1];
  const double V = states[STATE_V];

  // Assign parameters
  const double E_Na = parameters[PARAM_E_Na];
  const double g_Na = parameters[PARAM_g_Na];
  const double g_Nac = parameters[PARAM_g_Nac];
  const double g_s = parameters[PARAM_g_s];
  const double IstimAmplitude = parameters[PARAM_IstimAmplitude];
  const double IstimEnd = parameters[PARAM_IstimEnd];
  const double IstimPeriod = parameters[PARAM_IstimPeriod];
  const double IstimPulseDuration = parameters[PARAM_IstimPulseDuration];
  const double IstimStart = parameters[PARAM_IstimStart];
  const double C = parameters[PARAM_C];

  // Expressions for the Sodium current component
  monitored[MONITOR_i_Na] = (g_Nac + g_Na*(m*m*m)*h*j)*(-E_Na + V);

  // Expressions for the m gate component
  monitored[MONITOR_alpha_m] = (-47. - V)/(-1. +
    0.00909527710169582*exp(-0.1*V));
  monitored[MONITOR_beta_m] = 0.709552672748991*exp(-0.056*V);
  monitored[MONITOR_dm_dt] = (1. - m)*monitored[1] - m*monitored[2];

  // Expressions for the h gate component
  monitored[MONITOR_alpha_h] = 5.49796243870906e-10*exp(-0.25*V);
  monitored[MONITOR_beta_h] = 1.7/(1. + 0.158025320889648*exp(-0.082*V));
  monitored[MONITOR_dh_dt] = (1. - h)*monitored[3] - h*monitored[4];

  // Expressions for the j gate component
  monitored[MONITOR_alpha_j] = 1.86904730072229e-10*exp(-0.25*V)/(1. +
    1.67882752999566e-7*exp(-0.2*V));
  monitored[MONITOR_beta_j] = 0.3/(1. + 0.0407622039783662*exp(-0.1*V));
  monitored[MONITOR_dj_dt] = (1. - j)*monitored[5] - j*monitored[6];

  // Expressions for the Slow inward current component
  monitored[MONITOR_E_s] = -82.3 - 13.0287*log(0.001*Cai);
  monitored[MONITOR_i_s] = g_s*(-monitored[7] + V)*d*f;
  monitored[MONITOR_dCai_dt] = 7.0e-6 - 0.07*Cai - 0.01*monitored[8];

  // Expressions for the d gate component
  monitored[MONITOR_alpha_d] = 0.095*exp(1./20. - V/100.)/(1. +
    1.43328813856966*exp(-0.0719942404607631*V));
  monitored[MONITOR_beta_d] = 0.07*exp(-44./59. - V/59.)/(1. + exp(11./5. +
    V/20.));
  monitored[MONITOR_dd_dt] = (1. - d)*monitored[9] - d*monitored[10];

  // Expressions for the f gate component
  monitored[MONITOR_alpha_f] = 0.012*exp(-28./125. - V/125.)/(1. +
    66.5465065250986*exp(0.149925037481259*V));
  monitored[MONITOR_beta_f] = 0.0065*exp(-3./5. - V/50.)/(1. + exp(-6. -
    V/5.));
  monitored[MONITOR_df_dt] = (1. - f)*monitored[11] - f*monitored[12];

  // Expressions for the Time dependent outward current component
  monitored[MONITOR_i_x1] = 0.00197277571153285*(-1. +
    21.7584023961971*exp(0.04*V))*exp(-0.04*V)*x1;

  // Expressions for the X1 gate component
  monitored[MONITOR_alpha_x1] =
    0.0311584109863426*exp(0.0826446280991736*V)/(1. +
    17.4117080633277*exp(0.0571428571428571*V));
  monitored[MONITOR_beta_x1] =
    0.000391646440562322*exp(-0.0599880023995201*V)/(1. + exp(-4./5. -
    V/25.));
  monitored[MONITOR_dx1_dt] = (1. - x1)*monitored[14] - monitored[15]*x1;

  // Expressions for the Time independent outward current component
  monitored[MONITOR_i_K1] = 0.0035*(4.6 + 0.2*V)/(1. -
    0.398519041084514*exp(-0.04*V)) + 0.0035*(-4. +
    119.856400189588*exp(0.04*V))/(8.33113748768769*exp(0.04*V) +
    69.4078518387552*exp(0.08*V));

  // Expressions for the Stimulus protocol component
  monitored[MONITOR_Istim] = (t - IstimStart - IstimPeriod*floor((t -
    IstimStart)/IstimPeriod) <= IstimPulseDuration && t <= IstimEnd && t >=
    IstimStart ? IstimAmplitude : 0.);

  // Expressions for the Membrane component
  monitored[MONITOR_dV_dt] = (-monitored[0] - monitored[13] - monitored[17] -
    monitored[8] + monitored[16])/C;
}

// Compute a forward step using the explicit Euler algorithm to the
// beeler_reuter_1977_version06 ODE
void forward_explicit_euler(double *__restrict states, const double t, const
  double dt, const double *__restrict parameters)
{

  // Assign states
  const double m = states[STATE_m];
  const double h = states[STATE_h];
  const double j = states[STATE_j];
  const double Cai = states[STATE_Cai];
  const double d = states[STATE_d];
  const double f = states[STATE_f];
  const double x1 = states[STATE_x1];
  const double V = states[STATE_V];

  // Assign parameters
  const double E_Na = parameters[PARAM_E_Na];
  const double g_Na = parameters[PARAM_g_Na];
  const double g_Nac = parameters[PARAM_g_Nac];
  const double g_s = parameters[PARAM_g_s];
  const double IstimAmplitude = parameters[PARAM_IstimAmplitude];
  const double IstimEnd = parameters[PARAM_IstimEnd];
  const double IstimPeriod = parameters[PARAM_IstimPeriod];
  const double IstimPulseDuration = parameters[PARAM_IstimPulseDuration];
  const double IstimStart = parameters[PARAM_IstimStart];
  const double C = parameters[PARAM_C];

  // Expressions for the Sodium current component
  const double i_Na = (g_Nac + g_Na*(m*m*m)*h*j)*(-E_Na + V);

  // Expressions for the m gate component
  const double alpha_m = (-47. - V)/(-1. + 0.00909527710169582*exp(-0.1*V));
  const double beta_m = 0.709552672748991*exp(-0.056*V);
  const double dm_dt = (1. - m)*alpha_m - beta_m*m;
  states[STATE_m] = dt*dm_dt + m;

  // Expressions for the h gate component
  const double alpha_h = 5.49796243870906e-10*exp(-0.25*V);
  const double beta_h = 1.7/(1. + 0.158025320889648*exp(-0.082*V));
  const double dh_dt = (1. - h)*alpha_h - beta_h*h;
  states[STATE_h] = dt*dh_dt + h;

  // Expressions for the j gate component
  const double alpha_j = 1.86904730072229e-10*exp(-0.25*V)/(1. +
    1.67882752999566e-7*exp(-0.2*V));
  const double beta_j = 0.3/(1. + 0.0407622039783662*exp(-0.1*V));
  const double dj_dt = (1. - j)*alpha_j - beta_j*j;
  states[STATE_j] = dt*dj_dt + j;

  // Expressions for the Slow inward current component
  const double E_s = -82.3 - 13.0287*log(0.001*Cai);
  const double i_s = g_s*(-E_s + V)*d*f;
  const double dCai_dt = 7.0e-6 - 0.07*Cai - 0.01*i_s;
  states[STATE_Cai] = dt*dCai_dt + Cai;

  // Expressions for the d gate component
  const double alpha_d = 0.095*exp(1./20. - V/100.)/(1. +
    1.43328813856966*exp(-0.0719942404607631*V));
  const double beta_d = 0.07*exp(-44./59. - V/59.)/(1. + exp(11./5. + V/20.));
  const double dd_dt = (1. - d)*alpha_d - beta_d*d;
  states[STATE_d] = dt*dd_dt + d;

  // Expressions for the f gate component
  const double alpha_f = 0.012*exp(-28./125. - V/125.)/(1. +
    66.5465065250986*exp(0.149925037481259*V));
  const double beta_f = 0.0065*exp(-3./5. - V/50.)/(1. + exp(-6. - V/5.));
  const double df_dt = (1. - f)*alpha_f - beta_f*f;
  states[STATE_f] = dt*df_dt + f;

  // Expressions for the Time dependent outward current component
  const double i_x1 = 0.00197277571153285*(-1. +
    21.7584023961971*exp(0.04*V))*exp(-0.04*V)*x1;

  // Expressions for the X1 gate component
  const double alpha_x1 = 0.0311584109863426*exp(0.0826446280991736*V)/(1. +
    17.4117080633277*exp(0.0571428571428571*V));
  const double beta_x1 = 0.000391646440562322*exp(-0.0599880023995201*V)/(1.
    + exp(-4./5. - V/25.));
  const double dx1_dt = (1. - x1)*alpha_x1 - beta_x1*x1;
  states[STATE_x1] = dt*dx1_dt + x1;

  // Expressions for the Time independent outward current component
  const double i_K1 = 0.0035*(4.6 + 0.2*V)/(1. -
    0.398519041084514*exp(-0.04*V)) + 0.0035*(-4. +
    119.856400189588*exp(0.04*V))/(8.33113748768769*exp(0.04*V) +
    69.4078518387552*exp(0.08*V));

  // Expressions for the Stimulus protocol component
  const double Istim = (t - IstimStart - IstimPeriod*floor((t -
    IstimStart)/IstimPeriod) <= IstimPulseDuration && t <= IstimEnd && t >=
    IstimStart ? IstimAmplitude : 0.);

  // Expressions for the Membrane component
  const double dV_dt = (-i_K1 - i_Na - i_s - i_x1 + Istim)/C;
  states[STATE_V] = dt*dV_dt + V;
}

// Compute a forward step using the rush larsen algorithm to the
// beeler_reuter_1977_version06 ODE
void forward_rush_larsen(double *__restrict states, const double t, const
  double dt, const double *__restrict parameters)
{

  // Assign states
  const double m = states[STATE_m];
  const double h = states[STATE_h];
  const double j = states[STATE_j];
  const double Cai = states[STATE_Cai];
  const double d = states[STATE_d];
  const double f = states[STATE_f];
  const double x1 = states[STATE_x1];
  const double V = states[STATE_V];

  // Assign parameters
  const double E_Na = parameters[PARAM_E_Na];
  const double g_Na = parameters[PARAM_g_Na];
  const double g_Nac = parameters[PARAM_g_Nac];
  const double g_s = parameters[PARAM_g_s];
  const double IstimAmplitude = parameters[PARAM_IstimAmplitude];
  const double IstimEnd = parameters[PARAM_IstimEnd];
  const double IstimPeriod = parameters[PARAM_IstimPeriod];
  const double IstimPulseDuration = parameters[PARAM_IstimPulseDuration];
  const double IstimStart = parameters[PARAM_IstimStart];
  const double C = parameters[PARAM_C];

  // Expressions for the Sodium current component
  const double i_Na = (g_Nac + g_Na*(m*m*m)*h*j)*(-E_Na + V);

  // Expressions for the m gate component
  const double alpha_m = (-47. - V)/(-1. + 0.00909527710169582*exp(-0.1*V));
  const double beta_m = 0.709552672748991*exp(-0.056*V);
  const double dm_dt = (1. - m)*alpha_m - beta_m*m;
  const double dm_dt_linearized = -alpha_m - beta_m;
  states[STATE_m] = (fabs(dm_dt_linearized) > 1.0e-8 ? (-1.0 +
    exp(dt*dm_dt_linearized))*dm_dt/dm_dt_linearized : dt*dm_dt) + m;

  // Expressions for the h gate component
  const double alpha_h = 5.49796243870906e-10*exp(-0.25*V);
  const double beta_h = 1.7/(1. + 0.158025320889648*exp(-0.082*V));
  const double dh_dt = (1. - h)*alpha_h - beta_h*h;
  const double dh_dt_linearized = -alpha_h - beta_h;
  states[STATE_h] = (fabs(dh_dt_linearized) > 1.0e-8 ? (-1.0 +
    exp(dt*dh_dt_linearized))*dh_dt/dh_dt_linearized : dt*dh_dt) + h;

  // Expressions for the j gate component
  const double alpha_j = 1.86904730072229e-10*exp(-0.25*V)/(1. +
    1.67882752999566e-7*exp(-0.2*V));
  const double beta_j = 0.3/(1. + 0.0407622039783662*exp(-0.1*V));
  const double dj_dt = (1. - j)*alpha_j - beta_j*j;
  const double dj_dt_linearized = -alpha_j - beta_j;
  states[STATE_j] = (fabs(dj_dt_linearized) > 1.0e-8 ? (-1.0 +
    exp(dt*dj_dt_linearized))*dj_dt/dj_dt_linearized : dt*dj_dt) + j;

  // Expressions for the Slow inward current component
  const double E_s = -82.3 - 13.0287*log(0.001*Cai);
  const double i_s = g_s*(-E_s + V)*d*f;
  const double dCai_dt = 7.0e-6 - 0.07*Cai - 0.01*i_s;
  const double dE_s_dCai = -13.0287/Cai;
  const double di_s_dE_s = -g_s*d*f;
  const double dCai_dt_linearized = -0.07 - 0.01*dE_s_dCai*di_s_dE_s;
  states[STATE_Cai] = Cai + (fabs(dCai_dt_linearized) > 1.0e-8 ? (-1.0 +
    exp(dt*dCai_dt_linearized))*dCai_dt/dCai_dt_linearized : dt*dCai_dt);

  // Expressions for the d gate component
  const double alpha_d = 0.095*exp(1./20. - V/100.)/(1. +
    1.43328813856966*exp(-0.0719942404607631*V));
  const double beta_d = 0.07*exp(-44./59. - V/59.)/(1. + exp(11./5. + V/20.));
  const double dd_dt = (1. - d)*alpha_d - beta_d*d;
  const double dd_dt_linearized = -alpha_d - beta_d;
  states[STATE_d] = (fabs(dd_dt_linearized) > 1.0e-8 ? (-1.0 +
    exp(dt*dd_dt_linearized))*dd_dt/dd_dt_linearized : dt*dd_dt) + d;

  // Expressions for the f gate component
  const double alpha_f = 0.012*exp(-28./125. - V/125.)/(1. +
    66.5465065250986*exp(0.149925037481259*V));
  const double beta_f = 0.0065*exp(-3./5. - V/50.)/(1. + exp(-6. - V/5.));
  const double df_dt = (1. - f)*alpha_f - beta_f*f;
  const double df_dt_linearized = -alpha_f - beta_f;
  states[STATE_f] = (fabs(df_dt_linearized) > 1.0e-8 ? (-1.0 +
    exp(dt*df_dt_linearized))*df_dt/df_dt_linearized : dt*df_dt) + f;

  // Expressions for the Time dependent outward current component
  const double i_x1 = 0.00197277571153285*(-1. +
    21.7584023961971*exp(0.04*V))*exp(-0.04*V)*x1;

  // Expressions for the X1 gate component
  const double alpha_x1 = 0.0311584109863426*exp(0.0826446280991736*V)/(1. +
    17.4117080633277*exp(0.0571428571428571*V));
  const double beta_x1 = 0.000391646440562322*exp(-0.0599880023995201*V)/(1.
    + exp(-4./5. - V/25.));
  const double dx1_dt = (1. - x1)*alpha_x1 - beta_x1*x1;
  const double dx1_dt_linearized = -alpha_x1 - beta_x1;
  states[STATE_x1] = (fabs(dx1_dt_linearized) > 1.0e-8 ? (-1.0 +
    exp(dt*dx1_dt_linearized))*dx1_dt/dx1_dt_linearized : dt*dx1_dt) + x1;

  // Expressions for the Time independent outward current component
  const double i_K1 = 0.0035*(4.6 + 0.2*V)/(1. -
    0.398519041084514*exp(-0.04*V)) + 0.0035*(-4. +
    119.856400189588*exp(0.04*V))/(8.33113748768769*exp(0.04*V) +
    69.4078518387552*exp(0.08*V));

  // Expressions for the Stimulus protocol component
  const double Istim = (t - IstimStart - IstimPeriod*floor((t -
    IstimStart)/IstimPeriod) <= IstimPulseDuration && t <= IstimEnd && t >=
    IstimStart ? IstimAmplitude : 0.);

  // Expressions for the Membrane component
  const double dV_dt = (-i_K1 - i_Na - i_s - i_x1 + Istim)/C;
  const double di_K1_dV = 0.0007/(1. - 0.398519041084514*exp(-0.04*V)) +
    0.0167798960265423*exp(0.04*V)/(8.33113748768769*exp(0.04*V) +
    69.4078518387552*exp(0.08*V)) + 0.0035*(-4. +
    119.856400189588*exp(0.04*V))*(-0.333245499507508*exp(0.04*V) -
    5.55262814710042*exp(0.08*V))/((8.33113748768769*exp(0.04*V) +
    69.4078518387552*exp(0.08*V))*(8.33113748768769*exp(0.04*V) +
    69.4078518387552*exp(0.08*V))) - 5.5792665751832e-5*(4.6 +
    0.2*V)*exp(-0.04*V)/((1. - 0.398519041084514*exp(-0.04*V))*(1. -
    0.398519041084514*exp(-0.04*V)));
  const double di_Na_dV = g_Nac + g_Na*(m*m*m)*h*j;
  const double di_s_dV = g_s*d*f;
  const double di_x1_dV = 0.00171697791075903*x1 - 7.89110284613141e-5*(-1. +
    21.7584023961971*exp(0.04*V))*exp(-0.04*V)*x1;
  const double dV_dt_linearized = (-di_K1_dV - di_Na_dV - di_s_dV -
    di_x1_dV)/C;
  states[STATE_V] = (fabs(dV_dt_linearized) > 1.0e-8 ? (-1.0 +
    exp(dt*dV_dt_linearized))*dV_dt/dV_dt_linearized : dt*dV_dt) + V;
}
