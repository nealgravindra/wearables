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
    'gestage_by': 'Procedural',
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
    'assist_repro': 'Procedural',
    'gyn_infection': 'Pregnancy outcome',
    'maternal_del_weight': 'Pregnancy outcome',
    'ptb_37wks': 'Pregnancy outcome',
    'art_excess': 'Procedural',
    'art_lactate': 'Procedural',
    'ven_ph': 'Procedural',
    'ven_pco2': 'Procedural',
    'ven_lactate': 'Procedural',
    'anes_type': 'Procedural',
    'epidural': 'Procedural',
    'deliv_mode': 'Procedural',
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
    'stress2_1': 'Stress-related',
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
    'edinb1_1trim': 'Depression-related',
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
    'KPAS': 'Activity-related',
    'age_enroll': 'Social determinants of health',
    'insur': 'Social determinants of health',
    'asthma_yes___1': 'Comorbidities',
    'cbc_hct': 'Procedural',
    'cbc_wbc': 'Procedural',
    'cbc_plts': 'Procedural',
    'cbc_mcv': 'Procedural',
    'art_pco2': 'Procedural',
    'art_po2': 'Procedural',
    'ven_po2': 'Procedural',
    'regular_period1': 'Comorbidities',
    'period_window1': 'Comorbidities',
    'menstrual_days1': 'Comorbidities',
    'bc_years1': 'Comorbidities',
    'choosesleep_1trim': 'Sleep-quality',
    'RA': 'Activity-related',
    'ave_logpseudocount_wake': 'Activity-related',
    'PQSI': 'Sleep-quality',
    'EpworthSS': 'Sleep-quality',
    'Edinburgh': 'Depression-related',
}

# md_node_category_cmap = {
#     'Pregnancy outcome': '#171a21',
#     'Comorbidities': '#617073',
#     'Sleep-quality': '#7a93ac',
#     'Activity-related': '#92bceka',
#     'Social determinants of health': '#afb3f7',
#     'Depression-related': '#8db580',
#     'Stress-related': '#efd6ac',
#     'Labs': '#c44900'
# }

# 9 cols: 
arr = ["b0f2b4","baf2e9","bad7f2","f2bac9",
       "f2e2ba","37393a","f991cc","6d696a",
       "2c4251"]

md_node_category_cmap = {
    'Pregnancy outcome': '#b0f2b4',
    'Comorbidities': '#baf2e9',
    'Sleep-quality': '#bad7f2',
    'Social determinants of health': '#f2bac9',
    'Activity-related': '#f2e2ba',
    'Depression-related': '#37393a',
    'Procedural': '#f991cc',
    'Stress-related': '#6d696a'
}

max_categorical_obsVexp_ratio = {
    'marital': 1.0,
    'gestage_by': 0.0,
    'insur': 0.0,
    'ethnicity': 1.0,
    'race': 0.0,
    'prior_ptb_all': 1.0,
    'fullterm_births': 0.0,
    'surghx_none': 0.0,
    'alcohol': 2.0,
    'smoke': 2.0,
    'drugs': 2.0,
    'hypertension': 2.0,
    'pregestational_diabetes': 2.0,
    'asthma_yes___1': 1.0,
    'asthma_yes___2': 1.0,
    'asthma_yes___3': 1.0,
    'asthma_yes___4': 1.0,
    'asthma_yes___5': 1.0,
    'asthma_yes___7': 1.0,
    'asthma_yes___8': 1.0,
    'asthma_yes___10': 1.0,
    'asthma_yes___13': 1.0,
    'asthma_yes___18': 1.0,
    'asthma_yes___19': 1.0,
    'asthma_yes___20': 1.0,
    'other_disease': 2.0,
    'gestational_diabetes': 1.0,
    'ghtn': 1.0,
    'preeclampsia': 1.0,
    'rh': 1.0,
    'corticosteroids': 1.0,
    'abuse': 1.0,
    'assist_repro': 1.0,
    'gyn_infection': 1.0,
    'ptb_37wks': 1.0,
    'anes_type': 6.0,
    'epidural': 0.0,
    'deliv_mode': 3.0,
    'death_baby': 1.0,
    'neonatal_complication': 1.0,
    'ervisit': 1.0,
    'ppvisit_dx': 1.0,
    'education': 0.0,
    'paidjob1': 0.0,
    'work_hrs1': 1.0,
    'income_annual1': 1.0,
    'income_support1': 0.0,
    'regular_period1': 0.0,
    'period_window1': 1.0,
    'menstrual_days1': 0.0,
    'bc_past1': 0.0,
    'months_noprego1': 1.0,
    'premature_birth1': 0.0,
    'stress1_1': 0.0,
    'stress2_1': 0.0,
    'stress3_1': 0.0,
    'stress4_1': 0.0,
    'stress5_1': 0.0,
    'stress6_1': 0.0,
    'stress7_1': 0.0,
    'stress8_1': 0.0,
    'stress9_1': 0.0,
    'stress10_1': 0.0,
    'workreg_1trim': 0.0,
    'choosesleep_1trim': 0.0,
    'slpwake_1trim': 0.0,
    'slp30_1trim': 0.0,
    'sleep_qual1': 0.0,
    'slpenergy1': 0.0,
    'sitting1': 0.0,
    'tv1': 0.0,
    'inactive1': 0.0,
    'passenger1': 0.0,
    'reset1': 5.0,
    'talking1': 4.0,
    'afterlunch1': 0.0,
    'cartraffic1': 0.0,
    'edinb1_1trim': 4.0,
    'edinb2_1trim': 4.0,
    'edinb3_1trim': 0.0,
    'edinb4_1trim': 0.0,
    'edinb5_1trim': 4.0,
    'edinb6_1trim': 4.0,
    'edinb7_1trim': 0.0,
    'edinb8_1trim': 4.0,
    'edinb9_1trim': 4.0,
    'edinb10_1trim': 4.0,
}
    

