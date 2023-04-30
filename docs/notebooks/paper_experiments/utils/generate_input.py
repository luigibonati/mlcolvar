def gen_plumed(model_name : str, 
                file_path : str,
                potential_formula : str,
                opes_mode : str = 'OPES_METAD'):

    file_path = f'{file_path}/plumed.dat'
    file = open(file_path, 'w')

    input=f'''# vim:ft=plumed
UNITS NATURAL

p: POSITION ATOM=1

# define modified Muller Brown potential
ene: CUSTOM ARG=p.x,p.y PERIODIC=NO ...
FUNC={potential_formula}
...

pot: BIASVALUE ARG=ene

# load deep cv pytorch model
cv: PYTORCH_MODEL FILE=../{model_name} ARG=p.x,p.y 

# apply bias
opes: {opes_mode} ARG=cv.node-0 PACE=500 BARRIER=16 STATE_WSTRIDE=10000 STATE_WFILE=State.data    

PRINT FMT=%g STRIDE=100 FILE=COLVAR ARG=p.x,p.y,cv.*,opes.*

ENDPLUMED

    '''
    print(input, file=file)
    file.close()

def gen_plumed_tica(model_name : str,
                    file_path : str,
                    rfile_path : str,
                    potential_formula : str,
                    static_bias_cv : str = None,
                    static_model_path : str = None,
                    opes_mode : str = 'OPES_METAD',
                    opes_args : str = 'tica.node-0'):

    file_path = f'{file_path}/plumed.dat'
    file = open(file_path, 'w')

    input=f'''# vim:ft=plumed
UNITS NATURAL

p: POSITION ATOM=1

# define modified Muller Brown potential
ene: CUSTOM ARG=p.x,p.y PERIODIC=NO ...
FUNC={potential_formula}
...

pot: BIASVALUE ARG=ene

# load deep cv pytorch model
tica: PYTORCH_MODEL FILE=../{model_name} ARG=p.x,p.y
'''
    print(input, file=file)
    if static_model_path is not None:

        input=f'cv: PYTORCH_MODEL FILE={static_model_path} ARG=p.x,p.y'
        print(input, file=file)

    if static_bias_cv is not None:
        input=f'''# apply static bias from previous sim
static: OPES_METAD ARG={static_bias_cv} ...
    RESTART=YES
    STATE_RFILE={rfile_path}
    BARRIER=16
    PACE=10000000
...
'''
        print(input, file=file)

    input=f'''# apply bias
opes: {opes_mode} ARG={opes_args} PACE=500 BARRIER=16
'''
    print(input, file=file)

    if static_bias_cv is not None:
        input=f'''PRINT FMT=%g STRIDE=100 FILE=COLVAR ARG=p.x,p.y,tica.*,opes.*,{static_bias_cv},static.*

ENDPLUMED'''
    else:
        input=f'''PRINT FMT=%g STRIDE=100 FILE=COLVAR ARG=p.x,p.y,tica.*,opes.*

ENDPLUMED'''

    print(input, file=file)
    file.close()


def gen_input_md(inital_position : str,
                file_path : str,
                nsteps : int ):
    file_path = f'{file_path}/input_md.dat'
    file = open(file_path, 'w')
    input=f'''nstep             {nsteps}
tstep             0.005
temperature       1.0
friction          10.0
random_seed       42
plumed_input      plumed.dat
dimension         2
replicas          1
basis_functions_1 BF_POWERS ORDER=1 MINIMUM=-4.0 MAXIMUM=+3.0
basis_functions_2 BF_POWERS ORDER=1 MINIMUM=-1.0 MAXIMUM=+2.5
input_coeffs       input_md-potential.dat
initial_position   {inital_position}
output_coeffs           /dev/null
output_potential        /dev/null
output_potential_grid   10
output_histogram        /dev/null'''
    print(input, file=file)
    file.close()

def gen_input_md_potential(file_path : str):
    file_path = f'{file_path}/input_md-potential.dat'
    file = open(file_path, 'w')
    input=f'''#! FIELDS idx_dim1 idx_dim2 pot.coeffs index description
#! SET type LinearBasisSet
#! SET ndimensions  2
#! SET ncoeffs_total  1
#! SET shape_dim1  2
#! SET shape_dim2  2
    0       0         1.0000000000000000e+00       0  1*1
#!-------------------'''
    print(input, file=file)
    file.close()