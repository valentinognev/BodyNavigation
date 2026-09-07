from cadac_cpp.extract_cpp import extract_vehicle_types, PROGRAM_DIRS


HYPER6_SNIPPET = r'''
Cadac *set_obj_type(fstream &input, Module *module_list, int num_modules)
{
	if (!strcmp(temp,"HYPER6"))
		obj=new Hyper(...);
	else if (!strcmp(temp,"SAT3"))
		obj=new Satellite(...);
	else if (!strcmp(temp,"RADAR0"))
		obj=new Radar(...);
	return obj;
}
'''


def test_hyper6_types_include_sat3_and_radar0():
    assert extract_vehicle_types(HYPER6_SNIPPET) == ["HYPER6", "SAT3", "RADAR0"]


def test_ignores_commented_strcmp():
    text = '// if(!strcmp(temp,"GHOST"))\nif(!strcmp(temp,"CRUISE3")) {}'
    assert extract_vehicle_types(text) == ["CRUISE3"]


def test_twelve_program_dirs():
    assert set(PROGRAM_DIRS) == {
        "HYPER3", "FALCON5", "FALCON6", "HYPER5", "HYPER6",
        "AIM5", "CRUISE5", "MAGSIX", "ROCKET6", "SAM6", "SRAAM6", "AGM6",
    }


from cadac_cpp.extract_cpp import extract_def_modules, extract_modes


def test_def_modules():
    text = """
void Cruise::def_aerodynamics() {}
void Cruise::def_propulsion() {}
void Round3::def_newton() {}
// void Cruise::def_ghost() {}
"""
    assert extract_def_modules(text) == ["aerodynamics", "propulsion", "newton"]


def test_modes_from_if_not_comments():
    text = """
	if(mprop==1||mprop==2){
	if(mprop==0){
	// if(mprop==9){
	if(mins==0)
	else if(maut==24)
"""
    assert ("mprop", 1) in extract_modes(text)
    assert ("mprop", 2) in extract_modes(text)
    assert ("mprop", 0) in extract_modes(text)
    assert ("mins", 0) in extract_modes(text)
    assert ("maut", 24) in extract_modes(text)
    assert ("mprop", 9) not in extract_modes(text)


def test_modes_from_switch_cases():
    text = """
	switch(mact){ case 0: case 2: }
	// switch(mprop){ case 9: }
"""
    assert ("mact", 0) in extract_modes(text)
    assert ("mact", 2) in extract_modes(text)
    assert ("mprop", 9) not in extract_modes(text)
