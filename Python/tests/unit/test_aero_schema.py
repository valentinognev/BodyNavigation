from cadac.aero_map.schema import required_tables


def test_aim5_required_tables():
    assert required_tables("aim5", "AIM5") == (
        "cl_aim_vs_alpha_mach",
        "cd_aim_on_vs_alpha_mach",
        "cd_aim_off_vs_alpha_mach",
    )


def test_plane_required_tables():
    assert required_tables(None, "PLANE") == (
        "cl_30MAC_vs_mach_alphax",
        "cd_30MAC_vs_mach_alphax",
        "cl_35MAC_vs_mach_alphax",
        "cd_35MAC_vs_mach_alphax",
        "cl_40MAC_vs_mach_alphax",
        "cd_40MAC_vs_mach_alphax",
    )


def test_plane6_required_tables():
    assert required_tables(None, "PLANE6") == (
        "cx_vs_elev_alpha",
        "cxq_vs_alpha",
        "cyr_vs_alpha",
        "cyp_vs_alpha",
        "cz_vs_alpha",
        "czq_vs_alpha",
        "cl_vs_beta_alpha",
        "cldr_vs_beta_alpha",
        "clda_vs_beta_alpha",
        "clr_vs_alpha",
        "clp_vs_alpha",
        "cm_vs_elev_alpha",
        "cmq_vs_alpha",
        "cn_vs_beta_alpha",
        "cnda_vs_beta_alpha",
        "cndr_vs_beta_alpha",
        "cnr_vs_alpha",
        "cnp_vs_alpha",
    )


def test_sraam6_required_tables():
    names = required_tables("sraam6", "MISSILE6")
    assert names == (
        "ca0_vs_mach",
        "caa_vs_mach",
        "cad_vs_mach",
        "caoff_vs_mach",
        "cyp_vs_mach_alpha",
        "cndq_vs_mach_alpha",
        "cn0_vs_mach_alpha",
        "cnp_vs_mach_alpha",
        "cllap_vs_mach_alpha",
        "cllp_vs_mach_alpha",
        "clldp_vs_mach_alpha",
        "clm0_vs_mach_alpha",
        "clmp_vs_mach_alpha",
        "clmq_vs_mach",
        "clmdq_vs_mach_alpha",
        "clnp_vs_mach_alpha",
    )
    assert len(names) == len(set(names))


def test_agm6_required_tables_brief_order():
    names = required_tables("agm6", "MISSILE6")
    assert names == (
        "ca0_vs_mach",
        "caa_vs_mach",
        "cad_vs_mach",
        "cndq_vs_mach",
        "clmdq_vs_mach",
        "clmq_vs_mach",
        "cllap_vs_mach",
        "clldp_vs_mach",
        "cllp_vs_mach",
        "cn0_vs_mach_alpha",
        "cnp_vs_mach_alpha",
        "clm0_vs_mach_alpha",
        "clmp_vs_mach_alpha",
        "cyp_vs_mach_alpha",
        "clnp_vs_mach_alpha",
    )
    assert len(names) == len(set(names))


def test_cruise5_required_tables():
    assert required_tables("cruise5", "CRUISE3") == (
        "cd0_vs_mach",
        "cl0_vs_mach",
        "cla_vs_mach",
        "ckk_vs_mach",
        "cla0_vs_mach",
    )


def test_sam6_required_tables():
    names = required_tables("sam6", "MISSILE6")
    assert names == (
        "ca0_vs_mach,betax,alphax",
        "cad_vs_mach",
        "cab_vs_mach",
        "cy0_vs_mach,betax,alphax",
        "cydr_vs_mach,betax,alphax",
        "cn0_vs_mach,betax,alphax",
        "cndq_vs_mach,betax,alphax",
        "cll0_vs_mach,betax,alphax",
        "cllp_vs_mach",
        "clldp_vs_mach,betax,alphax",
        "clm0_vs_mach,betax,alphax",
        "clmq_vs_mach",
        "clmdq_vs_mach,betax,alphax",
        "cln0_vs_mach,betax,alphax",
        "clnr_vs_mach",
        "clndr_vs_mach,betax,alphax",
    )
    assert len(names) == len(set(names))


def test_rocket6_required_tables_slv1():
    names = required_tables("rocket6", "HYPER6")
    assert names == (
        "ca0slv1_vs_mach",
        "caaslv1_vs_mach",
        "ca0bslv1_vs_mach",
        "cn0slv1_vs_mach_alpha",
        "clm0slv1_vs_mach_alpha",
        "clmqslv1_vs_mach",
    )
    assert len(names) == len(set(names))


def test_unknown_pair_empty_tuple():
    assert required_tables("ghame5", "HYPER5") == ()
    assert required_tables("ghame6", "HYPER6") == ()
    assert required_tables("unknown", "MISSILE6") == ()
    assert required_tables(None, "MISSILE6") == ()
