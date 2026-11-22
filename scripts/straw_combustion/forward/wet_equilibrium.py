"""1-bar ln Kp fits on 450–1300 K. Study fits of gas-phase equilibria."""

import warp as wp

from .common import F


@wp.func
def kh(y: F, ly: F):
    """Maximum ln K fit residual 7.2757e-05."""
    return wp.exp(
        F(0.104738718417)
        + F(-16.8339579902) / y
        + F(1.37128542357) * ly
        + F(-0.332953771336) * y
        + F(0.0213438049922) * y * y
    )


@wp.func
def k11(y: F, ly: F):
    """Maximum ln K fit residual 0.00030899."""
    return wp.exp(
        F(-18.5262767281)
        + F(20.6638321039) / y
        + F(0.0380384081331) * ly
        + F(1.21431232966) * y
        + F(-0.193554598454) * y * y
    )


@wp.func
def k12(y: F, ly: F):
    """Maximum ln K fit residual 0.00017657."""
    return wp.exp(
        F(-18.7676810828)
        + F(39.2209165874) / y
        + F(-0.886353477491) * ly
        + F(2.08159353016) * y
        + F(-0.32991053264) * y * y
    )


@wp.func
def k13(y: F, ly: F):
    """Maximum ln K fit residual 8.9862e-05."""
    return wp.exp(
        F(-16.2048206194)
        + F(22.369682883) / y
        + F(1.21794765852) * ly
        + F(0.364619876465) * y
        + F(-0.079748229817) * y * y
    )


@wp.func
def k17b(y: F, ly: F):
    """Maximum ln K fit residual 0.00017462."""
    return wp.exp(
        F(-4.7400420036)
        + F(0.706410011982) / y
        + F(-0.409290006273) * ly
        + F(1.32009265942) * y
        + F(-0.221970356756) * y * y
    )


@wp.func
def k16(y: F, ly: F):
    """Maximum ln K fit residual 0.00026179."""
    return wp.exp(
        F(8.72117479268)
        + F(-9.02332450218) / y
        + F(1.34353952426) * ly
        + F(-2.75007544206) * y
        + F(0.461123939912) * y * y
    )


@wp.func
def k19(y: F, ly: F):
    """Maximum ln K fit residual 0.00021652."""
    return wp.exp(
        F(-2.44935607146)
        + F(3.34144239463) / y
        + F(0.955036407078) * ly
        + F(-0.781645162899) * y
        + F(0.138912981386) * y * y
    )


@wp.func
def k18(y: F, ly: F):
    """Maximum ln K fit residual 0.00014377."""
    return wp.exp(
        F(-2.55409478988)
        + F(20.1754003848) / y
        + F(-0.416249016487) * ly
        + F(-0.448691391562) * y
        + F(0.117569176394) * y * y
    )


@wp.func
def k20(y: F, ly: F):
    """Maximum ln K fit residual 0.00013621."""
    return wp.exp(
        F(-2.45812174499)
        + F(0.0172757141961) / y
        + F(-0.733015712444) * ly
        + F(1.38401988235) * y
        + F(-0.228818497831) * y * y
    )
