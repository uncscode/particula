""" calculate the wall loss coefficient
"""


def wlc(
    approx="none",
):
    """ calculate the dilution loss coefficient
    """
    return 0 if approx == "none" else 0.0

# function [gwall,kwall1985v0,fchar1] = ...
#     calcgdeps(dp,u,rhop,ke,fchar,k,Elec,xdebyeeg, debyeeg1)
# % the wall term
# dp = dp.*1e-9; % diameter in meters
# ggrav = 9.80665; % m / s2
# temp = 298.15; % temperature in K
# bolk = 1.3806e-23; % Boltzmann's constant in m^2 kg s^-2 K-1
# rgassconstant = 8.314462; % gas constant bolk/avogadro's
# massgas = 28.97; % mass of gas in Da (g/mol)
# pressure = 101325; % pressure in Pa
# muair = 18.27e-6; % air viscosity in kg/m-s or P-s
# % mean free path of gass in m
# cbargass = sqrt(8.*rgassconstant.*temp./(pi().*massgas.*1e-3)); % m/s
# mfpair = 2.*muair.*cbargass./(pressure); % in m
# Kn = 2 .* mfpair ./ ( dp ); % Knudsen number
# Cc = 1 + Kn .* ( 1.257 + 0.4 .* exp( - 1.1 ./ Kn )); %  correction
# mobp = Cc ./ ( 3 .* pi .* muair .* dp ); % mobility of particle
# difp = bolk.* temp.* mobp; % diffusion of particle
# rchamber = 1.34; % in m
# vsett = 1000.*rhop.*Cc.*(dp.^2).*ggrav./(18.*muair); % settling velocity
# xdummy = pi.*vsett./(2.*sqrt(ke.*difp)); % see integral
# elch = 1.602e-19; % elementary charge in C
# kwall1985v0 = zeros(21,length(dp)); % skeleton for kwall
# % loop over charging states
# for nuchi = 1:11
#     velec = (nuchi-1).*elch.*Elec.*Cc./(3.*pi.*muair.*dp); % elec velocity
#     ydummy = pi.*velec./(2.*sqrt(ke.*difp)); % see integral
#     % debye-related quantities
#     debyevalueplus2 =  debye1simple(xdebyeeg, debyeeg1, xdummy+ydummy);
#     debyevalueminus2 = debye1simple(xdebyeeg, debyeeg1, xdummy-ydummy);
#     % put all together
#     kwall1985v00 = 3.*sqrt(ke.*difp)./(pi.*rchamber.*xdummy).*...
#         (((xdummy+ydummy).^2)./2 + ...
#         (1).*debyevalueplus2 +...
#         (1).*debyevalueminus2);
#     % +1 and -1 are the same, etc
#     kwall1985v0(nuchi+10,:) = kwall1985v00;
#     if nuchi >= 2
#         kwall1985v0(12-nuchi,:) = kwall1985v00;
#     end
# end
# % calculate the actual gterm

# gwall = - bsxfun(@times, bsxfun(@times,kwall1985v0',u),fchar); % fraction

# gwall = sum(gwall'); % sum
# % recalculate the charge distribution
# fchar0 = bsxfun(@times,bsxfun(@times,u,fchar),(1-kwall1985v0'.*k));
# fchar1 = bsxfun(@rdivide,fchar0,sum(fchar0)); % normalize
# 	return
# end

# % Nested functions
# function yy = debye1simple(xdebyeeg, debyeeg1, xx)
# yy = ...
#     1.6449.*(xx>50) + ...
#     (-(xx.^2)/2).*(xx<-1e4) + ...
#     (xx.*(xx >= -1e4 & xx <= 50).*...
#     interp1(xdebyeeg,debyeeg1,xx.*...
#     (xx >= -1e4 & xx <= 50))).*(xx >= -1e4 & xx <= 50);
# end