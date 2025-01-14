# Run simulation with settings from config and custom stimulus domain

from pathlib import Path
import dolfin
import matplotlib.pyplot as plt
import simcardems
import numpy as np
import sys
try:
    import ufl
except ImportError:
    import ufl_legacy as ufl

import typing


logger = simcardems.utils.getLogger(__name__)

here = Path(__file__).absolute().parent
outdir = here / "results"
T=74000
save_freq=1
cellmodel="fully_coupled_Tor_Land"
cell_init_file="" #"init_5000beats.json"
disease_state="healthy"
PCL=1000
load_state=True
drug_factors_file=sys.argv[2]
popu_factors_file=sys.argv[1]
beats_saved=1
def stimulus_domain(mesh):
    marker = 1
    #wave:
    subdomain = dolfin.CompiledSubDomain("x[0] < 1.0")
    domain = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim())
    domain.set_all(0)
    subdomain.mark(domain, marker)

    #stimall:
    #domain = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim())
    #domain.set_all(marker)
    return simcardems.geometry.StimulusDomain(domain=domain, marker=marker)

#Create a new class for EMcoupling that overwrites the built-in function
class EMCoupling(simcardems.models.fully_coupled_Tor_Land.EMCoupling):
    def __init__(
        self,
        geometry,
        **state_params,
    ) -> None:
        super().__init__(geometry=geometry, **state_params)
        self.ICaL = dolfin.Function(self.V_ep, name="ICaL")
        self.INaL = dolfin.Function(self.V_ep, name="INaL")
        self.IKr = dolfin.Function(self.V_ep, name="IKr")
        self.Ito = dolfin.Function(self.V_ep, name="Ito")
        self.IK1 = dolfin.Function(self.V_ep, name="IK1")
        self.IKs = dolfin.Function(self.V_ep, name="IKs")        

    def register_datacollector(self, collector) -> None:
        super().register_datacollector(collector=collector)
        collector.register("ep", "INaL", self.INaL)
        collector.register("ep", "ICaL", self.ICaL)
        collector.register("ep", "IKr", self.IKr)
        collector.register("ep", "IKs", self.IKs)
        collector.register("ep", "IK1", self.IK1)
        collector.register("ep", "Ito", self.Ito)

    def ep_to_coupling(self):
        super().ep_to_coupling()
        self.INaL.assign(
            dolfin.project(
                INaL(
                    self.ep_solver.vs,
                    parameters=self.ep_solver.ode_solver._model.parameters(),
                ),
            ),
        )
        self.ICaL.assign(
            dolfin.project(
                ICaL(
                    self.ep_solver.vs,
                    parameters=self.ep_solver.ode_solver._model.parameters(),
                ),
            ),
        )
        self.IKr.assign(
            dolfin.project(
                IKr(
                    self.ep_solver.vs,
                    parameters=self.ep_solver.ode_solver._model.parameters(),
                ),
            ),
        )
        self.Ito.assign(
            dolfin.project(
                Ito(
                    self.ep_solver.vs,
                    parameters=self.ep_solver.ode_solver._model.parameters(),
                ),
            ),
        )
        self.IKs.assign(
            dolfin.project(
                IKs(
                    self.ep_solver.vs,
                    parameters=self.ep_solver.ode_solver._model.parameters(),
                ),
            ),
        )
        self.IK1.assign(
            dolfin.project(
                IK1(
                    self.ep_solver.vs,
                    parameters=self.ep_solver.ode_solver._model.parameters(),
                ),
            ),
        )

def INaL(vs, parameters):
    (
        v,
        CaMKt,
        m,
        h,
        j,
        hp,
        jp,
        mL,
        hL,
        hLp,
        a,
        iF,
        iS,
        ap,
        iFp,
        iSp,
        d,
        ff,
        fs,
        fcaf,
        fcas,
        jca,
        ffp,
        fcafp,
        nca,
        nca_i,
        kr_c0,
        kr_c1,
        kr_c2,
        kr_o,
        kr_i,
        xs1,
        xs2,
        Jrelnp,
        Jrelp,
        ki,
        kss,
        cass,
        cansr,
        cajsr,
        XS,
        XW,
        CaTrpn,
        TmB,
        Cd,
        cai,
        nai,
        nass,
    ) = vs

    # Assign parameters
    scale_INaL = parameters["scale_INaL"]
    nao = parameters["nao"]
    F = parameters["F"]
    R = parameters["R"]
    T = parameters["T"]
    CaMKo = parameters["CaMKo"]
    KmCaM = parameters["KmCaM"]
    KmCaMK = parameters["KmCaMK"]

    # Drug factor
    scale_drug_INaL = parameters["scale_drug_INaL"]

    # Population factors
    scale_popu_GNaL = parameters["scale_popu_GNaL"]

    HF_scaling_CaMKa = parameters["HF_scaling_CaMKa"]
    HF_scaling_GNaL = parameters["HF_scaling_GNaL"]

    # Init return args

    # Expressions for the CaMKt component
    CaMKb = CaMKo * (1.0 - CaMKt) / (1.0 + KmCaM / cass)
    CaMKa = (CaMKb + CaMKt) * HF_scaling_CaMKa

    # Expressions for the reversal potentials component
    ENa = R * T * ufl.ln(nao / nai) / F

    # Expressions for the INaL component
    GNaL = 0.0279 * scale_INaL * scale_drug_INaL * scale_popu_GNaL * HF_scaling_GNaL
    fINaLp = 1.0 / (1.0 + KmCaMK / CaMKa)
    return (-ENa + v) * ((1.0 - fINaLp) * hL + fINaLp * hLp) * GNaL * mL

def ICaL(vs, parameters):
    (
        v,
        CaMKt,
        m,
        h,
        j,
        hp,
        jp,
        mL,
        hL,
        hLp,
        a,
        iF,
        iS,
        ap,
        iFp,
        iSp,
        d,
        ff,
        fs,
        fcaf,
        fcas,
        jca,
        ffp,
        fcafp,
        nca,
        nca_i,
        kr_c0,
        kr_c1,
        kr_c2,
        kr_o,
        kr_i,
        xs1,
        xs2,
        Jrelnp,
        Jrelp,
        ki,
        kss,
        cass,
        cansr,
        cajsr,
        XS,
        XW,
        CaTrpn,
        TmB,
        Cd,
        cai,
        nai,
        nass,    
    ) = vs

    # Assign parameters
    scale_ICaL = parameters["scale_ICaL"]
    cao = parameters["cao"]
    F = parameters["F"]
    R = parameters["R"]
    T = parameters["T"]
    CaMKo = parameters["CaMKo"]
    KmCaM = parameters["KmCaM"]
    KmCaMK = parameters["KmCaMK"]
    ICaL_fractionSS = parameters["ICaL_fractionSS"]
    nao = parameters["nao"]
    ko = parameters["ko"]
    # Expressions for the reversal potentials component
    vffrt=v*F*F/(R*T);
    vfrt=v*F/(R*T);
    Aff = parameters ["Aff"]
    
    
    # Drug factor
    scale_drug_ICaL = parameters["scale_drug_ICaL"]

    # Population factors
    scale_popu_GCaL = parameters["scale_popu_GCaL"]

    # HF factors
    HF_scaling_CaMKa = parameters["HF_scaling_CaMKa"]


    # Expressions for the CaMKt component
    CaMKb = CaMKo * (1.0 - CaMKt) / (1.0 + KmCaM / cass)
    CaMKa = (CaMKb + CaMKt) * HF_scaling_CaMKa

    # Expressions for the ICaL component
    Afs = 1.0 - Aff
    f = Aff * ff + Afs * fs
    Afcaf = 0.3 + 0.6 / (1.0 + 0.36787944117144233 * ufl.exp(0.1 * v))
    Afcas = 1.0 - Afcaf
    fca = Afcaf * fcaf + Afcas * fcas
    fp = Aff * ffp + Afs * fs
    fcap = Afcaf * fcafp + Afcas * fcas

    clo = 150
    cli = 24
    Io = 0.5 * (nao + ko + clo + 4.0 * cao) / 1000.0
    Ii = 0.5 * (nass + kss + cli + 4.0 * cass) / 1000.0
    dielConstant = 74
    constA = 1.82 * 1000000.0 * ufl.elem_pow(dielConstant * T, -1.5)

    gamma_cai = gamma_cai = ufl.exp(
            -constA * 4.0 * (ufl.sqrt(Ii) / (1.0 + ufl.sqrt(Ii)) - 0.3 * Ii),
        )
    gamma_cao = gamma_cao = ufl.exp(
            -constA * 4.0 * (ufl.sqrt(Io) / (1.0 + ufl.sqrt(Io)) - 0.3 * Io),
        )
    gammaCaoMyo = gamma_cao
    gammaCaiMyo = gamma_cai
    
    PCa = 0.000083757 * scale_ICaL * scale_drug_ICaL * scale_popu_GCaL
    PCap = 1.1 * PCa
    
    PhiCaL_ss = (
            4.0
            * vffrt
            * (gamma_cai * cass * ufl.exp(2.0 * vfrt) - gamma_cao * cao)
            / (-1.0 + ufl.exp(2.0 * vfrt))
        )
    PhiCaL_i = (
            4.0
            * vffrt
            * (gamma_cai * cai * ufl.exp(2.0 * vfrt) - gamma_cao * cao)
            / (-1.0 + ufl.exp(2.0 * vfrt))
        )


    fICaLp = 1.0 / (1.0 + KmCaMK / CaMKa)
    
    ICaL_tot_ss = (1.0 - fICaLp) * (
            (1.0 - nca) * f + fca * jca * nca
        ) * PCa * PhiCaL_ss * d + (
            (1.0 - nca) * fp + fcap * jca * nca
        ) * PCap * PhiCaL_ss * d * fICaLp

    ICaL_tot_i = (1.0 - fICaLp) * (
            (1.0 - nca) * f + fca * jca * nca
        ) * PCa * PhiCaL_i * d + (
            (1.0 - nca) * fp + fcap * jca * nca
        ) * PCap * PhiCaL_i * d * fICaLp
    ICaL_ss = ICaL_tot_ss * ICaL_fractionSS
    ICaL_i = ICaL_tot_i * (1.0 - ICaL_fractionSS)
    return ICaL_ss + ICaL_i


def IKr (vs, parameters):
    (
        v,
        CaMKt,
        m,
        h,
        j,
        hp,
        jp,
        mL,
        hL,
        hLp,
        a,
        iF,
        iS,
        ap,
        iFp,
        iSp,
        d,
        ff,
        fs,
        fcaf,
        fcas,
        jca,
        ffp,
        fcafp,
        nca,
        nca_i,
        kr_c0,
        kr_c1,
        kr_c2,
        kr_o,
        kr_i,
        xs1,
        xs2,
        Jrelnp,
        Jrelp,
        ki,
        kss,
        cass,
        cansr,
        cajsr,
        XS,
        XW,
        CaTrpn,
        TmB,
        Cd,
        cai,
        nai,
        nass,
    ) = vs
    # Assign parameters
    scale_IKr = parameters["scale_IKr"]
    F = parameters["F"]
    R = parameters["R"]
    T = parameters["T"]
    ko = parameters["ko"]
    # Expressions for the reversal potentials component
    EK = R * T * ufl.ln(ko / ki) / F
    # Drug factor
    scale_drug_IKr = parameters["scale_drug_IKr"]

    # Population factors
    scale_popu_GKr = parameters["scale_popu_GKr"]

    # HF factors

    # Expressions for the INaL component
    GKr = 0.0321 * scale_IKr * scale_drug_IKr * scale_popu_GKr
    return GKr * ufl.sqrt(ko / 5.0) * kr_o * (-EK + v)

def Ito (vs, parameters):
    (
        v,
        CaMKt,
        m,
        h,
        j,
        hp,
        jp,
        mL,
        hL,
        hLp,
        a,
        iF,
        iS,
        ap,
        iFp,
        iSp,
        d,
        ff,
        fs,
        fcaf,
        fcas,
        jca,
        ffp,
        fcafp,
        nca,
        nca_i,
        kr_c0,
        kr_c1,
        kr_c2,
        kr_o,
        kr_i,
        xs1,
        xs2,
        Jrelnp,
        Jrelp,
        ki,
        kss,
        cass,
        cansr,
        cajsr,
        XS,
        XW,
        CaTrpn,
        TmB,
        Cd,
        cai,
        nai,
        nass,
    ) = vs
    # Assign parameters
    F = parameters["F"]
    R = parameters["R"]
    T = parameters["T"]
    Gto = parameters["Gto"]
    KmCaM = parameters["KmCaM"]
    KmCaMK = parameters["KmCaMK"]
    CaMKo = parameters["CaMKo"]
    ko = parameters["ko"]
    # Expressions for the reversal potentials component
    EK = R * T * ufl.ln(ko / ki) / F 
    # Drug factor
    scale_drug_Ito = parameters["scale_drug_Ito"]

    # Population factors
    scale_popu_Gto = parameters["scale_popu_Gto"]

    # HF factors
    HF_sacling_Gto = parameters["HF_scaling_Gto"]
    HF_scaling_CaMKa = parameters["HF_scaling_CaMKa"]

    # Expressions for the CaMKt component
    CaMKb = CaMKo * (1.0 - CaMKt) / (1.0 + KmCaM / cass)
    CaMKa = (CaMKb + CaMKt) * HF_scaling_CaMKa

    # Expressions for the Ito component
    AiF = 1.0 / (1.0 + 0.24348537187522867 * ufl.exp(0.006613756613756614 * v))
    AiS = 1.0 - AiF
    i = AiF * iF + AiS * iS
    ip = AiF * iFp + AiS * iSp
    fItop = 1.0 / (1.0 + KmCaMK / CaMKa)
    Gto_scale =  Gto * scale_drug_Ito * scale_popu_Gto * HF_sacling_Gto
    return (-EK + v) * Gto_scale * ((1.0 - fItop) * a * i + ap * fItop * ip)

def IKs (vs, parameters):
    (
        v,
        CaMKt,
        m,
        h,
        j,
        hp,
        jp,
        mL,
        hL,
        hLp,
        a,
        iF,
        iS,
        ap,
        iFp,
        iSp,
        d,
        ff,
        fs,
        fcaf,
        fcas,
        jca,
        ffp,
        fcafp,
        nca,
        nca_i,
        kr_c0,
        kr_c1,
        kr_c2,
        kr_o,
        kr_i,
        xs1,
        xs2,
        Jrelnp,
        Jrelp,
        ki,
        kss,
        cass,
        cansr,
        cajsr,
        XS,
        XW,
        CaTrpn,
        TmB,
        Cd,
        cai,
        nai,
        nass,
    ) = vs
    # Assign parameters
    scale_IKs = parameters["scale_IKs"]
    F = parameters["F"]
    R = parameters["R"]
    T = parameters["T"]
    ko = parameters["ko"]
    PKNa = parameters["PKNa"]
    nao = parameters["nao"]
    # Expressions for the reversal potentials component
    EKs = R * T * ufl.ln((ko + PKNa * nao) / (PKNa * nai + ki)) / F
    # Drug factor
    scale_drug_IKs = parameters["scale_drug_IKs"]

    # Population factors
    scale_popu_GKs = parameters["scale_popu_GKs"]

    # HF factors

    # Expressions for the IIKs component
    KsCa = 1.0 + 0.6 / (1.0 + 6.481821026062645e-07 * ufl.elem_pow(1.0 / cai, 1.4))
    GKs = 0.0011 * scale_IKs * scale_drug_IKs * scale_popu_GKs
    return (-EKs + v) * GKs * KsCa * xs1 * xs2

def IK1 (vs, parameters):
    (
        v,
        CaMKt,
        m,
        h,
        j,
        hp,
        jp,
        mL,
        hL,
        hLp,
        a,
        iF,
        iS,
        ap,
        iFp,
        iSp,
        d,
        ff,
        fs,
        fcaf,
        fcas,
        jca,
        ffp,
        fcafp,
        nca,
        nca_i,
        kr_c0,
        kr_c1,
        kr_c2,
        kr_o,
        kr_i,
        xs1,
        xs2,
        Jrelnp,
        Jrelp,
        ki,
        kss,
        cass,
        cansr,
        cajsr,
        XS,
        XW,
        CaTrpn,
        TmB,
        Cd,
        cai,
        nai,
        nass,
    ) = vs
    # Assign parameters
    scale_IK1 = parameters["scale_IK1"]
    F = parameters["F"]
    R = parameters["R"]
    T = parameters["T"]
    ko = parameters["ko"]
    # Expressions for the reversal potentials component
    EK = R * T * ufl.ln(ko / ki) / F 
    # Drug factor
    scale_drug_IK1 = parameters["scale_drug_IK1"]

    # Population factors
    scale_popu_GK1 = parameters["scale_popu_GK1"]

    # HF factors
    HF_scaling_GK1 = parameters["HF_scaling_GK1"]

    # Expressions for the IK1 component
    aK1 = 4.094 / (1.0 + ufl.exp(0.1217 * (v - EK - 49.934)))
    bK1 = (
            15.72 * ufl.exp(0.0674 * (v - EK - 3.257))
            + ufl.exp(0.0618 * (v - EK - 594.31))
        ) / (1.0 + ufl.exp(-0.1629 * (v - EK + 14.207)))
    K1ss = aK1 / (aK1 + bK1)
    GK1 = 0.6992 * scale_IK1 * scale_drug_IK1 * scale_popu_GK1 * HF_scaling_GK1
    return ufl.sqrt(ko/5) * (-EK + v) * GK1 * K1ss
   
#Don't use the built-in function to load the state, but create a new one here that we can change.
#def load_state(
#    cls,
#    path: typing.Union[str, Path],
#    drug_factors_file: typing.Union[str, Path] = "",
#    popu_factors_file: typing.Union[str, Path] = "",
#    disease_state="healthy",
#    PCL: float = 1000,
#):
#    logger.debug(f"Load state from path {path}")
#    path = Path(path)
#    if not path.is_file():
#        raise FileNotFoundError(f"File {path} does not exist")
#
#    logger.debug("Open file with h5py")
#    with simcardems.save_load_functions.h5pyfile(path) as h5file:
#        config = simcardems.Config(**simcardems.save_load_functions.h5_to_dict(h5file["config"]))
#
#    return cls.from_state(
#        path=path,
#        drug_factors_file=drug_factors_file,
#        popu_factors_file=popu_factors_file,
#        disease_state=disease_state,
#        PCL=PCL,
#    )

geo = simcardems.geometry.load_geometry(
    mesh_path="geometries/slab.h5",
    stimulus_domain=stimulus_domain,
)

config = simcardems.Config(
    outdir=outdir,
    coupling_type=cellmodel,
    T=T,
    cell_init_file=cell_init_file,
    show_progress_bar=False,
    save_freq=save_freq,
    disease_state=disease_state,
    dt_mech=0.5,
    mech_threshold=0.05,
    PCL=PCL,
    geometry_path="geometries/slab.h5",
    geometry_schema_path="geometries/slab.json",
    load_state=load_state,
    drug_factors_file=drug_factors_file,
    popu_factors_file=popu_factors_file,
)

if config.load_state:
    logger.info("Load previously saved state")
    #coupling = load_state(
    #    cls=EMCoupling,
    #    path=here / "results/state.h5",
    #    drug_factors_file=config.drug_factors_file,
    #    popu_factors_file=config.popu_factors_file,
    #    disease_state=config.disease_state,
    #    PCL=config.PCL,
    #)
    coupling = EMCoupling.from_state(
        path=here / "results/state.h5",
        drug_factors_file=config.drug_factors_file,
        popu_factors_file=config.popu_factors_file,
        disease_state=config.disease_state,
        PCL=config.PCL,
    )
    
    logger.info("Using as Popu file: " + popu_factors_file)
    logger.info("PCL is: " + str(config.PCL) + " Simulation time: " + str(config.T))
else:
    logger.info("Create new state")
    coupling = simcardems.models.em_model.setup_EM_model_from_config(
        config=config,
        geometry=geo,
    )

def save_condition(i, T, dt):
  return i > 0 and T >= 40000 and i % int(1000 / dt) == 0

#runner = simcardems.Runner.from_models(config=config, coupling=coupling)

runner = simcardems.Runner.from_models(config=config, coupling=coupling)
runner.solve(T=config.T, save_freq=config.save_freq, show_progress_bar=config.show_progress_bar, default_save_condition=save_condition) #config.show_progress_bar)

#Load geometry from file
#Create results-files with only last beat
#simcardems.postprocess.extract_sub_results(
#    results_file=here / "results/results.h5",
#    output_file=here / "results/results_sub.h5",
#    t_start=T-beats_saved*PCL,
#    t_end=T)

#loader = simcardems.DataLoader(outdir / "results_sub.h5")

