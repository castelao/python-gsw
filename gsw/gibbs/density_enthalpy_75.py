# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from .constants import sfac, soffset
from ..utilities import match_args_return

__all__ = ['alpha',
        'alpha_on_beta',
        'beta',
        'dynamic_enthalpy',
        'rho',
        'specvol']


a000 = -1.5649734675e-5
a001 =  1.8505765429e-5
a002 = -1.1736386731e-6
a003 = -3.6527006553e-7
a004 =  3.1454099902e-7
a010 =  5.5524212968e-5
a011 = -2.3433213706e-5
a012 =  4.2610057480e-6
a013 =  5.7391810318e-7
a020 = -4.9563477777e-5
a021 =  2.37838968519e-5
a022 = -1.38397620111e-6
a030 =  2.76445290808e-5
a031 = -1.36408749928e-5
a032 = -2.53411666056e-7
a040 = -4.0269807770e-6
a041 =  2.5368383407e-6
a050 =  1.23258565608e-6
a100 =  3.5009599764e-5
a101 = -9.5677088156e-6
a102 = -5.5699154557e-6
a103 = -2.7295696237e-7
a110 = -7.4871684688e-5
a111 = -4.7356616722e-7
a112 =  7.8274774160e-7
a120 =  7.2424438449e-5
a121 = -1.03676320965e-5
a122 =  2.32856664276e-8
a130 = -3.50383492616e-5
a131 =  5.1826871132e-6
a140 = -1.6526379450e-6
a200 = -4.3592678561e-5
a201 =  1.1100834765e-5
a202 =  5.4620748834e-6
a210 =  7.1815645520e-5
a211 =  5.8566692590e-6
a212 = -1.31462208134e-6
a220 = -4.3060899144e-5
a221 =  9.4965918234e-7
a230 =  1.74814722392e-5
a300 =  3.4532461828e-5
a301 = -9.8447117844e-6
a302 = -1.3544185627e-6
a310 = -3.7397168374e-5
a311 = -9.7652278400e-7
a320 =  6.8589973668e-6
a400 = -1.1959409788e-5
a401 =  2.5909225260e-6
a410 =  7.7190678488e-6
a500 =  1.3864594581e-6

v000 =  1.0769995862e-3
v001 = -6.0799143809e-5
v002 =  9.9856169219e-6
v003 = -1.1309361437e-6
v004 =  1.0531153080e-7
v005 = -1.2647261286e-8
v006 =  1.9613503930e-9
v010 = -1.5649734675e-5
v011 =  1.8505765429e-5
v012 = -1.1736386731e-6
v013 = -3.6527006553e-7
v014 =  3.1454099902e-7
v020 =  2.7762106484e-5
v021 = -1.1716606853e-5
v022 =  2.1305028740e-6
v023 =  2.8695905159e-7
v030 = -1.6521159259e-5
v031 =  7.9279656173e-6
v032 = -4.6132540037e-7
v040 =  6.9111322702e-6
v041 = -3.4102187482e-6
v042 = -6.3352916514e-8
v050 = -8.0539615540e-7
v051 =  5.0736766814e-7
v060 =  2.0543094268e-7
v100 = -3.1038981976e-4
v101 =  2.4262468747e-5
v102 = -5.8484432984e-7
v103 =  3.6310188515e-7
v104 = -1.1147125423e-7
v110 =  3.5009599764e-5
v111 = -9.5677088156e-6
v112 = -5.5699154557e-6
v113 = -2.7295696237e-7
v120 = -3.7435842344e-5
v121 = -2.3678308361e-7
v122 =  3.9137387080e-7
v130 =  2.4141479483e-5
v131 = -3.4558773655e-6
v132 =  7.7618888092e-9
v140 = -8.7595873154e-6
v141 =  1.2956717783e-6
v150 = -3.3052758900e-7
v200 =  6.6928067038e-4
v201 = -3.4792460974e-5
v202 = -4.8122251597e-6
v203 =  1.6746303780e-8
v210 = -4.3592678561e-5
v211 =  1.1100834765e-5
v212 =  5.4620748834e-6
v220 =  3.5907822760e-5
v221 =  2.9283346295e-6
v222 = -6.5731104067e-7
v230 = -1.4353633048e-5
v231 =  3.1655306078e-7
v240 =  4.3703680598e-6
v300 = -8.5047933937e-4
v301 =  3.7470777305e-5
v302 =  4.9263106998e-6
v310 =  3.4532461828e-5
v311 = -9.8447117844e-6
v312 = -1.3544185627e-6
v320 = -1.8698584187e-5
v321 = -4.8826139200e-7
v330 =  2.2863324556e-6
v400 =  5.8086069943e-4
v401 = -1.7322218612e-5
v402 = -1.7811974727e-6
v410 = -1.1959409788e-5
v411 =  2.5909225260e-6
v420 =  3.8595339244e-6
v500 = -2.1092370507e-4
v501 =  3.0927427253e-6
v510 =  1.3864594581e-6
v600 =  3.1932457305e-5


@match_args_return
def alpha(SA, CT, p):
    """
    Thermal expansion coefficient from CT (75-term equation)

    Calculates the thermal expansion coefficient of seawater with respect to
    Conservative Temperature using the computationally-efficient expression
    for specific volume in terms of SA, CT and p (Roquet et al., 2015).

    Parameters
    ----------
    SA : array_like
         Absolute Salinity  [g/kg]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]
    p : array_like
        sea pressure ( i.e. absolute pressure - 10.1325 dbar ) [dbar]

    Returns
    -------
    alpha : array_like
            thermal expansion coefficient [K :math:`-1`]
            with respect to Conservative Temperature

    Notes
    -----
    Note that this 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    infunnel(SA,CT,p) is avaialble to be used if one wants to test if some of
    one's data lies outside this "funnel".

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are
    MxN.

    Version
    -------
    3.05 (27th November, 2015)

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
       of seawater - 2010: Calculation and use of thermodynamic properties.
       Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
       UNESCO (English), 196 pp. Available from http://www.TEOS-10.org
       See Eqn. (2.18.3) of this TEOS-10 manual.

    .. [2] McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
       Accurate and computationally efficient algorithms for potential
       temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
       pp. 730-741.

    .. [3] Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
       polynomial expressions for the density and specifc volume of seawater
       using the TEOS-10 standard. Ocean Modelling.
    """

    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)

    xs = np.sqrt(sfac * SA + soffset)
    ys = CT * 0.025
    z = p * 1e-4

    v_CT_part = ( a000 + xs*(a100 + xs*(a200 + xs*(a300 + xs*(a400
        + a500*xs)))) + ys*(a010 + xs*(a110 + xs*(a210 + xs*(a310
        + a410*xs))) + ys*(a020 + xs*(a120 + xs*(a220 + a320*xs))
        + ys*(a030 + xs*(a130 + a230*xs) + ys*(a040 + a140*xs
        + a050*ys )))) + z*(a001 + xs*(a101 + xs*(a201 + xs*(a301
        + a401*xs))) + ys*(a011 + xs*(a111 + xs*(a211 + a311*xs))
        + ys*(a021 + xs*(a121 + a221*xs) + ys*(a031 + a131*xs
        + a041*ys))) + z*(a002 + xs*(a102 + xs*(a202 + a302*xs))
        + ys*(a012 + xs*(a112 + a212*xs) + ys*(a022 + a122*xs
        + a032*ys)) + z*(a003 + a103*xs + a013*ys + a004*z))) )

    v = ( v000 + xs*(v010 + xs*(v020 + xs*(v030 + xs*(v040 + xs*(v050
        + v060*xs))))) + ys*(v100 + xs*(v110 + xs*(v120 + xs*(v130
        + xs*(v140 + v150*xs)))) + ys*(v200 + xs*(v210 + xs*(v220
        + xs*(v230 + v240*xs))) + ys*(v300 + xs*(v310 + xs*(v320
        + v330*xs)) + ys*(v400 + xs*(v410 + v420*xs) + ys*(v500
        + v510*xs + v600*ys))))) + z*(v001 + xs*(v011 + xs*(v021
        + xs*(v031 + xs*(v041 + v051*xs)))) + ys*(v101 + xs*(v111
        + xs*(v121 + xs*(v131 + v141*xs))) + ys*(v201 + xs*(v211
        + xs*(v221 + v231*xs)) + ys*(v301 + xs*(v311 + v321*xs)
        + ys*(v401 + v411*xs + v501*ys)))) + z*(v002 + xs*(v012
        + xs*(v022 + xs*(v032 + v042*xs))) + ys*(v102 + xs*(v112
        + xs*(v122 + v132*xs)) + ys*(v202 + xs*(v212 + v222*xs)
        + ys*(v302 + v312*xs + v402*ys))) + z*(v003 + xs*(v013
        + v023*xs) + ys*(v103 + v113*xs + v203*ys) + z*(v004
        + v014*xs + v104*ys + z*(v005 + v006*z))))) )

    alpha_return = 0.025*v_CT_part/v

    return(alpha_return)


@match_args_return
def alpha_on_beta(SA, CT, p):

    """
    Thermal expansion divided by saline contraction (75-term equation)

    Calculates alpha divided by beta, where alpha is the thermal
    expansion coefficient and beta is the saline contraction coefficient
    of seawater from Absolute Salinity and Conservative Temperature.
    This function uses the computationally-efficient expression for
    specific volume in terms of SA, CT and p (Roquet et al., 2015).

    Parameters
    ----------
    SA : array_like
         Absolute Salinity  [g/kg]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]
    p : array_like
        sea pressure ( i.e. absolute pressure - 10.1325 dbar ) [dbar]

    Returns
    -------
    alpha_on_beta : array_like
                    Thermal expansion coefficient with respect to
                    Conservative Temperature divided by the saline
                    contraction coefficient at constant Conservative
                    Temperature [ kg g :math:`-1` K :math:`-1` ]

    Notes
    -----
    Note that this 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    infunnel(SA,CT,p) is avaialble to be used if one wants to test if some of
    one's data lies outside this "funnel".

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are
    MxN.

    Version
    -------
    3.05 (27th November, 2015)

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
       of seawater - 2010: Calculation and use of thermodynamic properties.
       Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
       UNESCO (English), 196 pp. Available from http://www.TEOS-10.org
       See Eqn. (2.18.3) of this TEOS-10 manual.

    .. [2] McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
       Accurate and computationally efficient algorithms for potential
       temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
       pp. 730-741.

    .. [3] Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
       polynomial expressions for the density and specifc volume of seawater
       using the TEOS-10 standard. Ocean Modelling.
    """

    SA = np.maximum(SA, 0)

    xs = np.sqrt(sfac * SA + soffset)
    ys = CT * 0.025
    z = p * 1e-4

    v_CT_part = ( a000 + xs*(a100 + xs*(a200 + xs*(a300 + xs*(a400
        + a500*xs)))) + ys*(a010 + xs*(a110 + xs*(a210 + xs*(a310
        + a410*xs))) + ys*(a020 + xs*(a120 + xs*(a220 + a320*xs))
        + ys*(a030 + xs*(a130 + a230*xs) + ys*(a040 + a140*xs
        + a050*ys )))) + z*(a001 + xs*(a101 + xs*(a201 + xs*(a301
        + a401*xs))) + ys*(a011 + xs*(a111 + xs*(a211 + a311*xs))
        + ys*(a021 + xs*(a121 + a221*xs) + ys*(a031 + a131*xs
        + a041*ys))) + z*(a002 + xs*(a102 + xs*(a202 + a302*xs))
        + ys*(a012 + xs*(a112 + a212*xs) + ys*(a022 + a122*xs
        + a032*ys)) + z*(a003 + a103*xs + a013*ys + a004*z))) )

    v_SA_part = ( b000 + xs*(b100 + xs*(b200 + xs*(b300 + xs*(b400
        + b500*xs)))) + ys*(b010 + xs*(b110 + xs*(b210 + xs*(b310
        + b410*xs))) + ys*(b020 + xs*(b120 + xs*(b220 + b320*xs))
        + ys*(b030 + xs*(b130 + b230*xs) + ys*(b040 + b140*xs
        + b050*ys)))) + z*(b001 + xs*(b101 + xs*(b201 + xs*(b301
        + b401*xs))) + ys*(b011 + xs*(b111 + xs*(b211 + b311*xs))
        + ys*(b021 + xs*(b121 + b221*xs) + ys*(b031 + b131*xs
        + b041*ys))) + z*(b002 + xs*(b102 + xs*(b202 + b302*xs))
        + ys*(b012 + xs*(b112 + b212*xs) + ys*(b022 + b122*xs
        + b032*ys)) + z*(b003 +  b103*xs + b013*ys + b004*z))) )

    return -(v_CT_part * xs) / (20. * sfac * v_SA_part)


@match_args_return
def beta(SA,CT,p):

    """
    gsw.beta                   saline contraction coefficient at constant
                              Conservative Temperature (76-term equation)
    ======================================================================

    USAGE:
    beta = gsw.beta(SA,CT,p)

    DESCRIPTION:
    Calculates the saline (i.e. haline) contraction coefficient of
    seawater at constant Conservative Temperature using the
    computationally-efficient 76-term expression for specific volume in
    terms of SA, CT and p (Roquet et al., 2014).

    Note that the 76-term equation has been fitted in a restricted range
    of parameter space, and is most accurate inside the "oceanographic
    funnel" described in McDougall et al. (2010).  The GSW library
    function "gsw.infunnel(SA,CT,p)" is avaialble to be used if one
    wants to test if some of one's data lies outside this "funnel".

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]
    p   =  sea pressure                                         [ dbar ]
        ( i.e. absolute pressure - 10.1325 dbar )

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are
    MxN.

    OUTPUT:
    beta  =  saline contraction coefficient                     [ kg/g ]
          at constant Conservative Temperature

    AUTHOR:
    Paul Barker and Trevor McDougall                [ help@teos-10.org ]

    VERSION NUMBER: 3.05 (27th November, 2015)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No.
    56, UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org
    See Eqn. (2.19.3) of this TEOS-10 manual.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of
    seawater using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)

    #deltaS = 24;
    sfac = 0.0248826675584615             # sfac = 1/(40*(35.16504/35)).
    offset = 5.971840214030754e-1                # offset = deltaS*sfac.

    x2 = sfac*SA
    xs = np.sqrt(x2 + offset)
    ys = CT*0.025
    z = p*1e-4

    v_SA_part = ( b000 + xs*(b100 + xs*(b200 + xs*(b300 + xs*(b400
        + b500*xs)))) + ys*(b010 + xs*(b110 + xs*(b210 + xs*(b310
        + b410*xs))) + ys*(b020 + xs*(b120 + xs*(b220 + b320*xs))
        + ys*(b030 + xs*(b130 + b230*xs) + ys*(b040 + b140*xs
        + b050*ys)))) + z*(b001 + xs*(b101 + xs*(b201 + xs*(b301
        + b401*xs))) + ys*(b011 + xs*(b111 + xs*(b211 + b311*xs))
        + ys*(b021 + xs*(b121 + b221*xs) + ys*(b031 + b131*xs
        + b041*ys))) + z*(b002 + xs*(b102 + xs*(b202 + b302*xs))
        + ys*(b012 + xs*(b112 + b212*xs) + ys*(b022 + b122*xs
        + b032*ys)) + z*(b003 +  b103*xs + b013*ys + b004*z))) )

    v = ( v000 + xs*(v010 + xs*(v020 + xs*(v030 + xs*(v040 + xs*(v050
        + v060*xs))))) + ys*(v100 + xs*(v110 + xs*(v120 + xs*(v130
        + xs*(v140 + v150*xs)))) + ys*(v200 + xs*(v210 + xs*(v220
        + xs*(v230 + v240*xs)))  + ys*(v300 + xs*(v310 + xs*(v320
        + v330*xs)) + ys*(v400 + xs*(v410 + v420*xs) + ys*(v500
        + v510*xs + v600*ys))))) + z*(v001 + xs*(v011 + xs*(v021
        + xs*(v031 + xs*(v041 + v051*xs)))) + ys*(v101 + xs*(v111
        + xs*(v121 + xs*(v131 + v141*xs))) + ys*(v201 + xs*(v211
        + xs*(v221 + v231*xs)) + ys*(v301 + xs*(v311 + v321*xs)
        + ys*(v401 + v411*xs + v501*ys)))) + z*(v002 + xs*(v012
        + xs*(v022 + xs*(v032 + v042*xs))) + ys*(v102 + xs*(v112
        + xs*(v122 + v132*xs)) + ys*(v202 + xs*(v212 + v222*xs)
        + ys*(v302 + v312*xs + v402*ys))) + z*(v003 + xs*(v013
        + v023*xs) + ys*(v103 + v113*xs + v203*ys) + z*(v004
        + v014*xs + v104*ys + z*(v005 + v006*z))))) )

    beta_return = -v_SA_part*0.5*sfac/(v*xs)

    return(beta_return)


@match_args_return
def dynamic_enthalpy(SA,CT,p):

    """
    gsw.dynamic_enthalpy                    dynamic enthalpy of seawater
                                                      (76-term equation)
    ====================================================================

    USAGE:
    dynamic_enthalpy = gsw.dynamic_enthalpy(SA,CT,p)

    DESCRIPTION:
    Calculates dynamic enthalpy of seawater using the computationally-
    efficient expression for specific volume in terms of SA, CT and p
    (Roquet et al., 2014).  Dynamic enthalpy is defined as enthalpy
    minus potential enthalpy (Young, 2010).

    Note that the 76-term equation has been fitted in a restricted range
    of parameter space, and is most accurate inside the "oceanographic
    funnel" described in IOC et al. (2010).  The GSW library function
    "gsw.infunnel(SA,CT,p)" is avaialble to be used if one wants to test
    if some of one's data lies outside this "funnel".

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar )

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are
    MxN.

    OUTPUT:
    dynamic_enthalpy  =  dynamic enthalpy                       [ J/kg ]

    AUTHOR:
    Trevor McDougall and Paul Barker                [ help@teos-10.org ]

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No.
    56, UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org
    See section 3.2 of this TEOS-10 Manual.

    McDougall, T. J., 2003: Potential enthalpy: A conservative oceanic
    variable for evaluating heat content and heat fluxes. Journal of
    Physical Oceanography, 33, 945-963.
    See Eqns. (18) and (22)

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of seawater
    using the TEOS-10 standard. Ocean Modelling.

    Young, W.R., 2010: Dynamic enthalpy, Conservative Temperature, and
    the seawater Boussinesq approximation. Journal of Physical
    Oceanography, 40, 394-400.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)

    # Set lower temperature limit for water that is much colder than the
    # freezing termperature.
    CT_frozen = - 1.79e-02 - 6.7157e-2*SA - 9.2708e-04*p

    Icold = CT < (CT_frozen - 0.2)
    if Icold.any():
        CT[Icold] = np.nan

    db2Pa = 1e4                      # factor to convert from dbar to Pa
    cp0 = 3991.86795711963     # from Eqn. (3.3.3) of IOC et al. (2010).

    sfac = 0.0248826675584615             # sfac = 1/(40*(35.16504/35)).
    offset = 5.971840214030754e-1

    x2 = sfac*SA
    xs = np.sqrt(x2 + offset)
    ys = CT*0.025
    z = p*1e-4

    dynamic_enthalpy_part = ( z*(h001 + xs*(h101 + xs*(h201 + xs*(h301
        + xs*(h401 + xs*(h501 + h601*xs))))) + ys*(h011 + xs*(h111
        + xs*(h211 + xs*(h311 + xs*(h411 + h511*xs)))) + ys*(h021
        + xs*(h121 + xs*(h221 + xs*(h321 + h421*xs))) + ys*(h031
        + xs*(h131 + xs*(h231 + h331*xs)) + ys*(h041 + xs*(h141
        + h241*xs) + ys*(h051 + h151*xs + h061*ys))))) + z*(h002
        + xs*(h102 + xs*(h202 + xs*(h302 + xs*(h402 + h502*xs))))
        + ys*(h012 + xs*(h112 + xs*(h212 + xs*(h312 + h412*xs)))
        + ys*(h022 + xs*(h122 + xs*(h222 + h322*xs)) + ys*(h032
        + xs*(h132 + h232*xs) + ys*(h042 + h142*xs + h052*ys))))
        + z*(h003 + xs*(h103 + xs*(h203 + xs*(h303 + h403*xs)))
        + ys*(h013 + xs*(h113 + xs*(h213 + h313*xs)) + ys*(h023
        + xs*(h123 + h223*xs) + ys*(h033 + h133*xs + h043*ys)))
        + z*(h004 + xs*(h104 + h204*xs) + ys*(h014 + h114*xs + h024*ys)
        + z*(h005 + h105*xs + h015*ys + z*(h006 + h007*z)))))) )

    dynamic_enthalpy_return = dynamic_enthalpy_part*1e8
    # Note. 1e8 = db2Pa*1e4

    return(dynamic_enthalpy_return)


@match_args_return
def rho(SA,CT,p):

    """
    gsw.rho                           in-situ density (76-term equation)
    ====================================================================

    USAGE:
    rho = gsw.rho(SA,CT,p)

    DESCRIPTION:
    Calculates in-situ density from Absolute Salinity and Conservative
    Temperature, using the computationally-efficient expression for
    specific volume in terms of SA, CT and p  (Roquet et al., 2014).

    Note that potential density with respect to reference pressure, pr,
    is obtained by calling this function with the pressure argument
    being pr (i.e. "gsw.rho(SA,CT,pr)").

    Note that the computationally-efficient expression has been fitted
    in a restricted range of parameter space, and is most accurate
    inside the "oceanographic funnel" described in IOC et al. (2010).
    The GSW library function "gsw.infunnel(SA,CT,p)" is avaialble to be
    used if one wants to test if some of one's data lies outside this
    "funnel".

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar )

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are
    MxN.

    OUTPUT:
    rho  =  in-situ density                                     [ kg/m ]

    AUTHOR:
    Paul Barker and Trevor McDougall                [ help@teos-10.org ]

    VERSION NUMBER: 3.05 (27th November, 2015)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No.
    56,UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org
    See appendix A.20 and appendix K of this TEOS-10 Manual.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of
    seawater using the TEOS-10 standard. Ocean Modelling.
    """
    SA = np.maximum(SA, 0)

    xs = np.sqrt(sfac * SA + soffset)
    ys = CT * 0.025
    z = p * 1e-4

    v = ( v000 + xs*(v010 + xs*(v020 + xs*(v030 + xs*(v040
        + xs*(v050 + v060*xs))))) + ys*(v100 + xs*(v110 + xs*(v120
        + xs*(v130 + xs*(v140 + v150*xs)))) + ys*(v200 + xs*(v210
        + xs*(v220 + xs*(v230 + v240*xs)))  + ys*(v300 + xs*(v310
        + xs*(v320 + v330*xs)) + ys*(v400 + xs*(v410 + v420*xs)
        + ys*(v500 + v510*xs + v600*ys))))) + z*(v001 + xs*(v011
        + xs*(v021 + xs*(v031 + xs*(v041 + v051*xs)))) + ys*(v101
        + xs*(v111 + xs*(v121 + xs*(v131 + v141*xs))) + ys*(v201
        + xs*(v211 + xs*(v221 + v231*xs)) + ys*(v301 + xs*(v311
        + v321*xs) + ys*(v401 + v411*xs + v501*ys)))) + z*(v002
        + xs*(v012 + xs*(v022 + xs*(v032 + v042*xs))) + ys*(v102
        + xs*(v112 + xs*(v122 + v132*xs)) + ys*(v202 + xs*(v212
        + v222*xs) + ys*(v302 + v312*xs + v402*ys))) + z*(v003
        + xs*(v013 + v023*xs) + ys*(v103 + v113*xs + v203*ys)
        + z*(v004 + v014*xs + v104*ys + z*(v005 + v006*z))))) )

    return 1. / v


@match_args_return
def specvol(SA,CT,p):

    """
    gsw_specvol                       specific volume (76-term equation)
    ====================================================================

    USAGE:
    specvol = gsw.specvol(SA,CT,p)

    DESCRIPTION:
    Calculates specific volume from Absolute Salinity, Conservative
    Temperature and pressure, using the computationally-efficient
    76-term polynomial expression for specific volume
    (Roquet et al., 2014).

    Note that the 76-term equation has been fitted in a restricted range
    of parameter space, and is most accurate inside the "oceanographic
    funnel" described in McDougall et al. (2011).  The GSW library
    function "gsw.infunnel(SA,CT,p)" is available to be used if one
    wants to test if some of one's data lies outside this "funnel".

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar )

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are
    MxN.

    OUTPUT:
    specvol  =  specific volume                               [ m^3/kg ]

    AUTHOR:
    Fabien Roquet

    VERSION NUMBER: 3.05 (27th November, 2015)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No.
    56, UNESCO (English), 196 pp.  Available from
    http://www.TEOS-10.org.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of
    seawater using the TEOS-10 standard. Ocean Modelling.
    """

    SA = np.maximum(SA, 0)

    #deltaS = 24;
    sfac = 0.0248826675584615           # sfac = 1/(40*(35.16504/35)).
    offset = 5.971840214030754e-1              # offset = deltaS*sfac.

    x2 = sfac*SA
    xs = np.sqrt(x2 + offset)
    ys = CT*0.025
    z = p*1e-4

    specvol_return = ( v000 + xs*(v010 + xs*(v020 + xs*(v030 + xs*(v040
        + xs*(v050 + v060*xs))))) + ys*(v100 + xs*(v110 + xs*(v120
        + xs*(v130 + xs*(v140 + v150*xs)))) + ys*(v200 + xs*(v210
        + xs*(v220 + xs*(v230 + v240*xs))) + ys*(v300 + xs*(v310
        + xs*(v320 + v330*xs)) + ys*(v400 + xs*(v410 + v420*xs)
        + ys*(v500 + v510*xs + v600*ys))))) + z*(v001 + xs*(v011
        + xs*(v021 + xs*(v031 + xs*(v041 + v051*xs)))) + ys*(v101
        + xs*(v111 + xs*(v121 + xs*(v131 + v141*xs))) + ys*(v201
        + xs*(v211 + xs*(v221 + v231*xs)) + ys*(v301 + xs*(v311
        + v321*xs) + ys*(v401 + v411*xs + v501*ys)))) + z*(v002
        + xs*(v012 + xs*(v022 + xs*(v032 + v042*xs))) + ys*(v102
        + xs*(v112 + xs*(v122 + v132*xs)) + ys*(v202 + xs*(v212
        + v222*xs) + ys*(v302 + v312*xs + v402*ys))) + z*(v003
        + xs*(v013 + v023*xs) + ys*(v103 + v113*xs + v203*ys)
        + z*(v004 + v014*xs + v104*ys + z*(v005 + v006*z))))) )

    return(specvol_return)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
