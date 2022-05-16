from main import model_launcher

# 'phebus/config_files/assessment_cite.json',
# 'phebus/config_files/assessment_eptz.json',


list_scenarios = ['phebus_31/config_files/assessment_cite.json',
                  'phebus_31/config_files/assessment_eptz.json',
                  'phebus_31/config_files/assessment_reducedtax.json',
                  'phebus_31/config_files/assessment_carbontax.json',
                  'phebus_31/config_files/assessment_cee.json']

for scenario in list_scenarios:
    print(scenario)
    model_launcher(path=scenario)
