from main import model_launcher


list_scenarios = ['phebus/config_files/scenario_eptz.json',
                  'phebus/config_files/scenario_reducedtax.json',
                  'phebus/config_files/scenario_carbontax.json']

for scenario in list_scenarios:
    print(scenario)
    model_launcher(path=scenario)
