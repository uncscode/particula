"""Define additional paired challenges selected during the two-hour campaign."""
import json
from dataclasses import asdict
from pathlib import Path
from campaign import Experiment, groups, ALL, DUST, SAMPLES, CONTROLS, BASE_SEED

def sequence():
    # Separate the Brownian seed ranges of stochastic realizations. The same
    # range is retained within each treatment and its matched control.
    independent=[10000019*i for i in range(1,9)]
    # Pair the same arriving dust population with a weakly and a strongly
    # ice-active resident spectrum in identical air at both loading conventions.
    for abundance in ['equal','air_volume']:
        for resident in ['RuralContinental','Marine']:
            yield from groups(['isdac','reference'],['closed',DUST,'aged_'+resident,'Marine'],independent[:6],
                              resident=resident,abundance=abundance)
    # At a shared-device checkpoint, prioritize low-cost concentration and
    # well-mixture sensitivities before any more high-resolution trajectories.
    yield from groups(['very_cold'],['grain_feldspar','grain_direct_feldspar'],independent,grain_diameter=2.)
    for share in [0.,.25,.5,.75]:
        yield from groups(['reference','vigorous_dry'],[DUST,'Marine','RuralContinental'],independent[:4],share=share)
    for n in [10.,100.]:
        yield from groups(['reference','cold_deep','very_cold'],['grain_inert','grain_feldspar','grain_quartz'],
                          independent[:4],grain_number=n)
    # Check whether the mixing response survives less numerical diffusion.
    for grid,seeds in [(20,independent[:2]),(30,independent[:1])]:
        yield from groups(['reference','mix2','mix8','synthetic_roll400_1800'],
                          [DUST,'aged_'+DUST],seeds,grid=grid)
    # Separate step size from grid refinement in the sign-changing thin layer.
    for factor in [.5,.25]:
        yield from groups(['isdac','mpace'],['closed','aged_'+DUST,DUST,'ContinentalPristine'],
                          independent[:2],dt_factor=factor)
    yield from groups(['isdac','vigorous_dry'],['closed','aged_'+DUST,DUST,'ContinentalPristine'],
                      independent[:2],density=96)
    # Same statistical weights and trajectories, but no mineral freezing.
    for sky in ['reference','mix2','very_cold','cold_deep']:
        yield from groups([sky],['grain_sham_feldspar','grain_feldspar','grain_sham_quartz','grain_quartz'],independent[:4])
    # Check switching the marine/dust ranking and the field contrasts on a much finer mesh.
    for sky in ['reference','isdac','mpace']:
        yield from groups([sky],['closed','aged_'+DUST,DUST,'Marine'],independent[:1],grid=40,abundance='air_volume')
    for sky in ['reference']:
        yield from groups([sky],['closed','aged_'+DUST,DUST,'Marine'],independent[:1],grid=40)
    yield from groups(['reference','vigorous_dry','isdac','mpace','isdac_we4p125','mpace_we8p2'],
                      ALL,independent,abundance='air_volume')
    yield from groups(['very_cold'],['grain_feldspar','grain_direct_feldspar'],independent[:5],grain_diameter=2.)
    for slope in [0.,.25,.75]:
        yield from groups(['reference','warm_thin','cold_deep','isdac'],[*CONTROLS,'ContinentalPolluted',DUST,'Marine'],independent[:3],slope=slope)
    for plate in range(1,6):
        yield from groups(['reference','vigorous_dry'],[*SAMPLES,'aged_'+DUST],independent[:2],plate=plate)
    for reset in [0,1]:
        yield from groups(['reference','isdac','vigorous_dry'],[*CONTROLS,DUST,'Marine','RuralContinental'],independent[:3],law='aging',reset=reset)
    yield from groups(['qbase','qplus025','qplus050','mpace_flux5','mpace_flux10','mpace_flux25','mpace_flux50'],
                      [DUST,'closed','aged_'+DUST,'RuralContinental','Marine'],independent[:4])
    for grid,seeds in [(10,independent),(20,independent[:3])]:
        yield from groups(['reference','vigorous_dry','isdac','mpace','isdac_we4p125','mpace_we8p2'],ALL,seeds,grid=grid,abundance='air_volume')
    yield from groups(['reference','cold_deep','very_cold'],['grain_inert','grain_feldspar','grain_quartz'],independent[:4])
    yield from groups(['isdac','mpace'],['closed','ContinentalPristine','aged_'+DUST],independent[:3],grid=30)
    for grid,seeds in [(10,independent),(20,independent[:3])]:
        yield from groups(['reference','vigorous_dry','cold_deep','isdac','mpace','warm_thin','weak_dry','moist_mixed'],
                          ALL,seeds,grid=grid)
    for sky in ['mix2','mix8','synthetic_roll200_1800','synthetic_roll400_1800',
                'mpace_roll200_1800','isdac_roll200_1800']:
        yield from groups([sky],[DUST,'closed','aged_'+DUST,'RuralContinental','Marine'],independent)
    for sky in ['isdac','mpace','reference','vigorous_dry']:
        for mm in [1.2,4.,7.2,12.,20.]:
            name=sky+'_we'+str(mm).replace('.','p')
            yield from groups([name],[DUST,'closed','aged_'+DUST,'RuralContinental','Marine'],independent[:4])
    for sky,mm in [('isdac',4.125),('mpace',8.2)]:
        label=sky+'_we'+str(mm).replace('.','p')
        for grid,seeds in [(10,independent),(20,independent[:4]),(40,independent[:2])]:
            yield from groups([label],ALL,seeds,grid=grid)
    # Anchor sign changes with paired 40- and 60-cell-horizontal resolutions.
    for grid,seeds in [(40,independent[:2]),(60,independent[:1])]:
        yield from groups(['isdac','mpace','vigorous_dry'],
                          ['closed',DUST,'aged_'+DUST,'RuralContinental','Marine'],seeds,grid=grid)
    yield from groups(['very_cold'],['grain_feldspar','grain_direct_feldspar'],independent,grain_diameter=2.)
    for number in [10.,100.]:
        yield from groups(['reference','cold_deep','very_cold'],['grain_inert','grain_feldspar','grain_quartz'],
                          independent[:4],grain_number=number)
    # First do the literature-profile, thermodynamic controls and mineral pilots.
    yield from groups(['isdac','mpace'], ALL,range(8))
    for sky in ['synthetic_roll0_1800','synthetic_roll200_1800','synthetic_roll400_1800','synthetic_roll200_600',
                'isdac_roll200_1800','mpace_roll200_1800','dry_roll200_1800']:
        yield from groups([sky], [DUST,'aged_'+DUST,'closed','Marine','RuralContinental'],range(6))
    yield from groups(['mpace_flux5','mpace_flux10','mpace_flux25','mpace_flux50','mpace_flux100',
                       'isdac_ccn50','mpace_ccn200'],[DUST,'aged_'+DUST,'closed','RuralContinental','Marine'],range(6))
    for grid in [10,20]:
        yield from groups(['reference','cold_deep','very_cold'],['grain_inert','grain_feldspar','grain_quartz'],range(3),grid=grid)
    yield from groups(['qbase','qplus025','qplus050'],[DUST,'aged_'+DUST,'RuralContinental','Marine'],range(8))
    yield from groups(['isdac_no_mix','isdac_mix8','mpace_no_mix','mpace_mix8'],ALL,range(4))
    yield from groups(['isdac','mpace'], [*CONTROLS,'Marine','RuralContinental',DUST],range(4),grid=20)
    # Source dilution and openness in the vigorous dry regime.
    for share in [0.,.25,.5,.75]:
        yield from groups(['reference','vigorous_dry'],[DUST,'Marine','RuralContinental'],range(8),share=share)
    for sky in ['reference','cold_deep','very_cold']:
        for delta in [-.8,.8]:
            yield from groups([sky],['grain_feldspar','grain_quartz'],range(3),site_shift=delta)
        for diameter in [.5,2.]:
            yield from groups([sky],['grain_feldspar','grain_quartz','grain_inert'],range(4),grain_diameter=diameter)
    yield from groups(['isdac','mpace','vigorous_dry'],[DUST,'aged_'+DUST,'closed','RuralContinental','Marine'],range(3),grid=30)
    for resident in ['Marine','RuralContinental','CombustionDominated','ContinentalPristine','ContinentalPolluted']:
        yield from groups(['reference','vigorous_dry'],[*SAMPLES,'aged_'+resident,'closed'],range(4),resident=resident)

if __name__=='__main__':
    path=Path(__file__).parent/'results'/'priority_cases.json'
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps([asdict(x) for x in sequence()]))
    temporary.replace(path)
