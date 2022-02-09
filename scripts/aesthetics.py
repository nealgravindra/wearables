import seaborn as sns
import matplotlib.pyplot as plt

# %matplotlib inline
# plot settings
plt.rc('font', size = 9)
plt.rc('font', family='sans serif')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['legend.frameon']=False
plt.rcParams['axes.grid']=False
plt.rcParams['legend.markerscale']=1
plt.rcParams['savefig.dpi'] = 600
sns.set_style("ticks")


models = [
    'kNN',
    'RandomForest',
    'Gradient Boosting',
    'TimeSeriesForest',
    'GRU',
    'VGG-1D',
    'InceptionTime (Random)',
    'Ours',
]
model_cmap = {k:plt.get_cmap('cubehelix', 9)(i) for i, k in enumerate(models)}

slp_binary_color = '#4297A0'

ptb_cmap = {False: '#2F5061', True: '#E57F84'}

split_cmap = {'train': '#4297A0', 'test': '#F4EAE6'}

def p_encoder(p):
    if p > 0.05:
        label = '' # n.s.
    elif p <= 0.001:
        label = '***'
    elif p <= 0.05 and p > 0.01:
        label = '*'
    elif p <= 0.01 and p > 0.001:
        label = '**'
    else: 
        label = 'Unclassified'
    return label

traintest_ptbyn_cmap = {'train_ptby': '#F4EAE6',
                        'train_ptbn': '#4297A0',
                        'test_ptby': '#E57F84',
                        'test_ptbn': '#2F5061'}


errgrp_cmap = {'lt10wks': '#4297A0', # green
               'Higher-than-actual': '#E57F84', #red
               'Lower-than-actual': '#2F5061', # blue
               }

sleep_cmap = {0:'#FAD02C', 1:'#333652'}

md_node_annotation = {
    'marital': 'Social determinants of health',
    'gestage_by': 'Pregnancy outcome',
    'ethnicity': 'Social determinants of health',
    'race': 'Social determinants of health',
    'bmi_1vis': 'Activity-related',
    'prior_ptb_all': 'Comorbidities',
    'fullterm_births': 'Comorbidities',
    'surghx_none': 'Comorbidities',
    'alcohol': 'Social determinants of health',
    'smoke': 'Social determinants of health',
    'drugs': 'Social determinants of health',
    'hypertension': 'Comorbidities',
    'pregestational_diabetes': 'Comorbidities',
    'asthma_yes___2': 'Comorbidities',
    'asthma_yes___3': 'Comorbidities',
    'asthma_yes___4': 'Comorbidities',
    'asthma_yes___5': 'Comorbidities',
    'asthma_yes___7': 'Comorbidities',
    'asthma_yes___8': 'Comorbidities',
    'asthma_yes___10': 'Comorbidities',
    'asthma_yes___13': 'Comorbidities',
    'asthma_yes___18': 'Depression-related',
    'asthma_yes___19': 'Depression-related',
    'asthma_yes___20': 'Comorbidities',
    'other_disease': 'Comorbidities',
    'gestational_diabetes': 'Pregnancy outcome',
    'ghtn': 'Pregnancy outcome',
    'preeclampsia': 'Pregnancy outcome',
    'rh': 'Pregnancy outcome',
    'corticosteroids': 'Pregnancy outcome',
    'abuse': 'Social determinants of health',
    'assist_repro': 'Pregnancy outcome',
    'gyn_infection': 'Pregnancy outcome',
    'maternal_del_weight': 'Pregnancy outcome',
    'ptb_37wks': 'Pregnancy outcomes',
    'art_excess': 'Labs',
    'art_lactate': 'Labs',
    'ven_ph': 'Labs',
    'ven_pco2': 'Labs',
    'ven_lactate': 'Labs',
    'anes_type': 'Pregnancy outcome',
    'epidural': 'Pregnancy outcome',
    'deliv_mode': 'Pregnancy outcome',
    'infant_wt': 'Pregnancy outcome',
    'infant_length': 'Pregnancy outcome',
    'head_circ': 'Pregnancy outcome',
    'death_baby': 'Pregnancy outcome',
    'neonatal_complication': 'Pregnancy outcome',
    'ervisit': 'Pregnancy outcome',
    'ppvisit_dx': 'Pregnancy outcome',
    'education': 'Social determinants of health',
    'paidjob1': 'Social determinants of health',
    'work_hrs1': 'Social determinants of health',
    'income_annual1': 'Social determinants of health',
    'income_support1': 'Social determinants of health',
    'bc_past1': 'Comorbidities',
    'months_noprego1': 'Pregnancy outcome',
    'premature_birth1': 'Pregnancy outcome',
    'stress1_1': 'Stress-related',
    'stress2_1': 'Stressrelated',
    'stress3_1': 'Stress-related',
    'stress4_1': 'Stress-related',
    'stress5_1': 'Stress-related',
    'stress6_1': 'Stress-related',
    'stress7_1': 'Stress-related',
    'stress8_1': 'Stress-related',
    'stress9_1': 'Stress-related',
    'stress10_1': 'Stress-related',
    'workreg_1trim': 'Sleep-quality',
    'slpwake_1trim': 'Sleep-quality',
    'slp30_1trim': 'Sleep-quality',
    'sleep_qual1': 'Sleep-quality',
    'slpenergy1': 'Sleep-quality',
    'sitting1': 'Sleep-quality',
    'tv1': 'Sleep-quality',
    'inactive1': 'Sleep-quality',
    'passenger1': 'Sleep-quality',
    'reset1': 'Sleep-quality',
    'talking1': 'Sleep-quality',
    'afterlunch1': 'Sleep-quality',
    'cartraffic1': 'Sleep-quality',
    'edinb2_1trim': 'Depression-related',
    'edinb3_1trim': 'Depression-related',
    'edinb4_1trim': 'Depression-related',
    'edinb5_1trim': 'Depression-related',
    'edinb6_1trim': 'Depression-related',
    'edinb7_1trim': 'Depression-related',
    'edinb8_1trim': 'Depression-related',
    'edinb9_1trim': 'Depression-related',
    'edinb10_1trim': 'Depression-related',
    'IS': 'Activity-related',
    'IV': 'Activity-related',
    'ISm': 'Activity-related',
    'IVm': 'Activity-related',
    'min_rest': 'Activity-related',
    'ave_logpseudocount_sleep': 'Activity-related',
    'ave_logpseudocount_wknd': 'Activity-related',
    'ave_logpseudocount_wkday': 'Activity-related',
    'ave_logpseudocount_day': 'Activity-related',
    'ave_logpseudocount_night': 'Activity-related',
    'KPAS': 'Activity-related'
}